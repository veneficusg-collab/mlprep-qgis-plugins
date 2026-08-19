/*
   Step 4 deposition-zone dynamic-alpha MPI port.

   This is a TauDEM-style C++/MPI implementation of the routing core in
   the historical Python reference. It preserves the current Step 4 behavior:
   - TauDEM D-Infinity flow proportions
   - flow-path distance mode
   - source cells from a binary source raster
   - per-source alpha from an alpha raster
   - path-dependent dynamic alpha adjusted by DFI
   - beta acceptance rule: beta >= dynamic alpha
   - deterministic candidate selection: highest beta, then lower alpha

   It intentionally does not overwrite or depend on the historical Python file.
*/

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <queue>
#include <string>
#include <vector>

#include "commonLib.h"
#include "createpart.h"
#include "initneighbor.h"
#include "linearpart.h"
#include "tiffIO.h"

namespace {

constexpr float kFloatNodata = -3.4028235e38f;
constexpr double kDoubleNodata = static_cast<double>(kFloatNodata);
constexpr double kAlphaAngleValidMin = 1.0;
constexpr double kAlphaAngleValidMax = 90.0;
constexpr double kFixedAlphaMin = 6.0;
constexpr double kFixedAlphaMax = 72.0;
constexpr double kDefaultProportionThreshold = 0.2;
constexpr double kDefaultDfiMid = 0.5;
constexpr double kDefaultAlphaGainPerMeter = 0.03333;
constexpr double kFixedDfiDeadband = 0.05;
constexpr double kBetaTieTolerance = 1.0e-6;
constexpr double kMaxPlanimetricFlowPathLengthM = 200.0;

struct Options {
    std::string fel;
    std::string ang;
    std::string source;
    std::string dfi;
    std::string alpha;
    std::string out_dynamic_alpha;
    std::string out_beta;
    std::string out_dfs;
    std::string out_mask;
    std::string out_deposition;
    std::string out_parent_alpha;
    std::string stats_json;
    double proportion_threshold = kDefaultProportionThreshold;
    double dfi_mid = kDefaultDfiMid;
    double alpha_gain_per_meter = kDefaultAlphaGainPerMeter;
};

void usage(const char* exe) {
    std::fprintf(
        stderr,
        "Usage:\n"
        "  mpiexec -n <N> %s --fel <fel.tif> --ang <ang.tif> --source <source.tif> \\\n"
        "      --dfi <dfi.tif> --alpha <alpha.tif> --out-alpha <dynamic_alpha.tif> \\\n"
        "      --out-beta <beta.tif> --out-dfs <dfs.tif> --out-mask <mask.tif> \\\n"
        "      --out-deposition <deposition.tif> [options]\n\n"
        "Options:\n"
        "  --threshold <value>        D-Infinity proportion threshold, default 0.2\n"
        "  --dfi-mid <value>          DFI midpoint, default 0.5\n"
        "  --alpha-gain-per-meter <value>  Alpha change rate per meter, default 0.03333\n"
        "  --out-parent-alpha <tif>   Optional parent dynamic-alpha debug raster\n"
        "  --stats-json <json>        Optional machine-readable run statistics\n",
        exe);
}

bool parse_args(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        bool missing_value = false;
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", name);
                missing_value = true;
                return std::string();
            }
            return argv[++i];
        };

        if (key == "--fel") opt.fel = need_value("--fel");
        else if (key == "--ang") opt.ang = need_value("--ang");
        else if (key == "--source") opt.source = need_value("--source");
        else if (key == "--dfi") opt.dfi = need_value("--dfi");
        else if (key == "--alpha") opt.alpha = need_value("--alpha");
        else if (key == "--out-alpha") opt.out_dynamic_alpha = need_value("--out-alpha");
        else if (key == "--out-beta") opt.out_beta = need_value("--out-beta");
        else if (key == "--out-dfs") opt.out_dfs = need_value("--out-dfs");
        else if (key == "--out-mask") opt.out_mask = need_value("--out-mask");
        else if (key == "--out-deposition") opt.out_deposition = need_value("--out-deposition");
        else if (key == "--out-parent-alpha") opt.out_parent_alpha = need_value("--out-parent-alpha");
        else if (key == "--stats-json") opt.stats_json = need_value("--stats-json");
        else if (key == "--threshold") opt.proportion_threshold = std::atof(need_value("--threshold").c_str());
        else if (key == "--dfi-mid") opt.dfi_mid = std::atof(need_value("--dfi-mid").c_str());
        else if (key == "--alpha-gain-per-meter") opt.alpha_gain_per_meter = std::atof(need_value("--alpha-gain-per-meter").c_str());
        else if (key == "--help" || key == "-h") return false;
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", key.c_str());
            return false;
        }
        if (missing_value) return false;
    }

    return !opt.fel.empty() && !opt.ang.empty() && !opt.source.empty() &&
           !opt.dfi.empty() && !opt.alpha.empty() && !opt.out_dynamic_alpha.empty() &&
           !opt.out_beta.empty() && !opt.out_dfs.empty() && !opt.out_mask.empty() &&
           !opt.out_deposition.empty() && opt.proportion_threshold >= 0.0f &&
           opt.proportion_threshold <= 1.0f && opt.alpha_gain_per_meter >= 0.0f;
}

char* cstr(const std::string& value) {
    return const_cast<char*>(value.c_str());
}

bool finite_non_nodata(float value) {
    return std::isfinite(value) && value != kFloatNodata;
}

bool finite_non_nodata(double value) {
    return std::isfinite(value) && value != kDoubleNodata;
}

bool source_is_binary(float value) {
    return std::fabs(value) <= 1.0e-7f || std::fabs(value - 1.0f) <= 1.0e-7f;
}

double compute_delta_alpha(
    double dfi_value,
    double dfi_mid,
    double alpha_gain_per_meter,
    double step_distance_m) {
    const double delta = dfi_value - dfi_mid;
    if (std::fabs(delta) < kFixedDfiDeadband) return 0.0;
    return alpha_gain_per_meter * delta * step_distance_m;
}

double clamp_dynamic_alpha(double value) {
    if (value < kFixedAlphaMin) return kFixedAlphaMin;
    if (value > kFixedAlphaMax) return kFixedAlphaMax;
    return value;
}

void get_safe_dxdyc(tdpartition* partition, int row, int ny, double& dxc, double& dyc) {
    int safe_row = row;
    if (safe_row < 0) safe_row = 0;
    if (safe_row >= ny) safe_row = ny - 1;
    partition->getdxdyc(safe_row, dxc, dyc);
}

bool candidate_is_better(
    double best_beta,
    double best_dynamic_alpha,
    double beta_candidate,
    double alpha_candidate) {
    if (!std::isfinite(best_beta) || best_beta == kDoubleNodata) return true;
    if (beta_candidate > best_beta + kBetaTieTolerance) return true;
    if (std::fabs(beta_candidate - best_beta) <= kBetaTieTolerance &&
        alpha_candidate < best_dynamic_alpha - kBetaTieTolerance) {
        return true;
    }
    return false;
}

bool compare_or_abort(tiffIO& reference, const tiffIO& other, const char* label, int rank) {
    if (!reference.compareTiff(other)) {
        if (rank == 0) std::fprintf(stderr, "%s raster does not match the D-Infinity grid.\n", label);
        return false;
    }
    return true;
}

void read_partition(tiffIO& tif, tdpartition* partition, int xstart, int ystart) {
    tif.read(xstart, ystart, partition->getny(), partition->getnx(), partition->getGridPointer());
}

linearpart<double>* create_double_partition(long totalX, long totalY, double dxA, double dyA) {
    linearpart<double>* partition = new linearpart<double>();
    partition->init(totalX, totalY, dxA, dyA, MPI_DOUBLE, kDoubleNodata);
    return partition;
}

void write_double_partition_as_float(
    tiffIO& tif,
    linearpart<double>* partition,
    int xstart,
    int ystart,
    int nx,
    int ny) {
    std::vector<float> buffer(static_cast<size_t>(nx) * static_cast<size_t>(ny), kFloatNodata);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            double value = kDoubleNodata;
            partition->getData(x, y, value);
            if (finite_non_nodata(value)) {
                buffer[static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x)] =
                    static_cast<float>(value);
            }
        }
    }
    tif.write(xstart, ystart, ny, nx, buffer.data());
}

int run_dynamic_alpha_mpi(const Options& opt) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);

    const double begin_t = MPI_Wtime();
    if (rank == 0) {
        std::printf("Step4 deposition-zone dynamic-alpha MPI port\n");
        std::printf("Processes: %d\n", size);
    }

    tiffIO ang(cstr(opt.ang), FLOAT_TYPE);
    const long totalX = ang.getTotalX();
    const long totalY = ang.getTotalY();
    const double dxA = ang.getdxA();
    const double dyA = ang.getdyA();

    if (rank == 0 && ang.getproj() == 1) {
        std::fprintf(
            stderr,
            "Warning: Python Step 4 requires projected rasters in meters. "
            "This port follows TauDEM distance handling, but projected metric rasters are recommended.\n");
    }

    tdpartition* flowData = CreateNewPartition(ang.getDatatype(), totalX, totalY, dxA, dyA, ang.getNodata());
    int xstart = 0;
    int ystart = 0;
    flowData->localToGlobal(0, 0, xstart, ystart);
    flowData->savedxdyc(ang);
    read_partition(ang, flowData, xstart, ystart);

    tiffIO fel(cstr(opt.fel), FLOAT_TYPE);
    tiffIO source(cstr(opt.source), FLOAT_TYPE);
    tiffIO dfi(cstr(opt.dfi), FLOAT_TYPE);
    tiffIO alpha(cstr(opt.alpha), FLOAT_TYPE);
    if (!compare_or_abort(ang, fel, "DEM/fel", rank) ||
        !compare_or_abort(ang, source, "source", rank) ||
        !compare_or_abort(ang, dfi, "DFI", rank) ||
        !compare_or_abort(ang, alpha, "alpha", rank)) {
        return 2;
    }

    tdpartition* felData = CreateNewPartition(fel.getDatatype(), totalX, totalY, dxA, dyA, fel.getNodata());
    tdpartition* sourceData = CreateNewPartition(source.getDatatype(), totalX, totalY, dxA, dyA, source.getNodata());
    tdpartition* dfiData = CreateNewPartition(dfi.getDatatype(), totalX, totalY, dxA, dyA, dfi.getNodata());
    tdpartition* alphaData = CreateNewPartition(alpha.getDatatype(), totalX, totalY, dxA, dyA, alpha.getNodata());
    read_partition(fel, felData, xstart, ystart);
    read_partition(source, sourceData, xstart, ystart);
    read_partition(dfi, dfiData, xstart, ystart);
    read_partition(alpha, alphaData, xstart, ystart);

    const int nx = flowData->getnx();
    const int ny = flowData->getny();

    flowData->share();
    felData->share();
    sourceData->share();
    dfiData->share();
    alphaData->share();

    linearpart<double>* beta = create_double_partition(totalX, totalY, dxA, dyA);
    linearpart<double>* dfs = create_double_partition(totalX, totalY, dxA, dyA);
    linearpart<double>* dynamicAlpha = create_double_partition(totalX, totalY, dxA, dyA);
    linearpart<double>* parentDynamicAlpha =
        opt.out_parent_alpha.empty() ? nullptr : create_double_partition(totalX, totalY, dxA, dyA);
    linearpart<double>* srcElev = create_double_partition(totalX, totalY, dxA, dyA);
    tdpartition* sourceMask = CreateNewPartition(SHORT_TYPE, totalX, totalY, dxA, dyA, static_cast<int16_t>(0));
    tdpartition* processedMask = CreateNewPartition(SHORT_TYPE, totalX, totalY, dxA, dyA, static_cast<int16_t>(0));
    tdpartition* neighbor = CreateNewPartition(SHORT_TYPE, totalX, totalY, dxA, dyA, static_cast<int16_t>(-32768));
    tdpartition* maskOut = CreateNewPartition(SHORT_TYPE, totalX, totalY, dxA, dyA, static_cast<int16_t>(-32768));
    tdpartition* depositionOut = CreateNewPartition(SHORT_TYPE, totalX, totalY, dxA, dyA, static_cast<int16_t>(-32768));

    double* dist = new double[static_cast<size_t>(ny) * 9];
    for (int y = 0; y < ny; ++y) {
        double dxc = 0.0;
        double dyc = 0.0;
        flowData->getdxdyc(y, dxc, dyc);
        for (int k = 1; k <= 8; ++k) {
            dist[static_cast<size_t>(y) * 9 + k] =
                std::sqrt(dxc * dxc * d1[k] * d1[k] + dyc * dyc * d2[k] * d2[k]);
        }
    }

    long long local_terrain = 0;
    long long local_alpha_valid = 0;
    long long local_marked_sources = 0;
    long long local_sources = 0;
    long long local_source_without_valid_alpha = 0;
    long long local_invalid_source = 0;
    long long local_invalid_alpha = 0;
    long long local_invalid_flow_angle = 0;
    long long local_source_alpha_bounds = 0;
    long long local_invalid_dfi = 0;
    long long local_valid_dfi = 0;
    double local_source_alpha_min = std::numeric_limits<double>::infinity();
    double local_source_alpha_max = -std::numeric_limits<double>::infinity();
    double local_source_alpha_sum = 0.0;
    double local_dfi_min = std::numeric_limits<double>::infinity();
    double local_dfi_max = -std::numeric_limits<double>::infinity();
    double local_flow_angle_min = std::numeric_limits<double>::infinity();
    double local_flow_angle_max = -std::numeric_limits<double>::infinity();

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            int16_t zero = 0;
            maskOut->setData(x, y, zero);
            depositionOut->setData(x, y, zero);
            sourceMask->setData(x, y, zero);
            processedMask->setData(x, y, zero);

            if (felData->isNodata(x, y) || flowData->isNodata(x, y)) continue;
            ++local_terrain;

            float flow_angle = 0.0f;
            flowData->getData(x, y, flow_angle);
            if (!std::isfinite(flow_angle) || flow_angle < 0.0f || flow_angle > 2.0 * PI + 1.0e-6) {
                ++local_invalid_flow_angle;
            } else {
                local_flow_angle_min = std::min(local_flow_angle_min, static_cast<double>(flow_angle));
                local_flow_angle_max = std::max(local_flow_angle_max, static_cast<double>(flow_angle));
            }

            float src = 0.0f;
            if (!sourceData->isNodata(x, y)) {
                sourceData->getData(x, y, src);
                if (!source_is_binary(src)) ++local_invalid_source;
                if (src > 0.0f) ++local_marked_sources;
            }

            bool alpha_is_valid = false;
            float alpha_value = 0.0f;
            if (!alphaData->isNodata(x, y)) {
                alphaData->getData(x, y, alpha_value);
                if (!std::isfinite(alpha_value) ||
                    alpha_value < kAlphaAngleValidMin ||
                    alpha_value > kAlphaAngleValidMax) {
                    ++local_invalid_alpha;
                } else {
                    alpha_is_valid = true;
                    ++local_alpha_valid;
                }
            }

            if (!dfiData->isNodata(x, y)) {
                float dv = 0.0f;
                dfiData->getData(x, y, dv);
                if (!std::isfinite(dv) || dv < 0.0f || dv > 1.0f) ++local_invalid_dfi;
                else {
                    ++local_valid_dfi;
                    local_dfi_min = std::min(local_dfi_min, static_cast<double>(dv));
                    local_dfi_max = std::max(local_dfi_max, static_cast<double>(dv));
                }
            }

            if (!sourceData->isNodata(x, y) && src > 0.0f) {
                if (!alpha_is_valid) {
                    ++local_source_without_valid_alpha;
                } else {
                    if (alpha_value < kFixedAlphaMin || alpha_value > kFixedAlphaMax) {
                        ++local_source_alpha_bounds;
                    } else {
                        float z = 0.0f;
                        felData->getData(x, y, z);
                        beta->setData(x, y, static_cast<double>(alpha_value));
                        dfs->setData(x, y, 0.0);
                        dynamicAlpha->setData(x, y, static_cast<double>(alpha_value));
                        if (parentDynamicAlpha != nullptr) {
                            parentDynamicAlpha->setData(x, y, static_cast<double>(alpha_value));
                        }
                        srcElev->setData(x, y, static_cast<double>(z));
                        sourceMask->setData(x, y, static_cast<int16_t>(1));
                        local_source_alpha_min =
                            std::min(local_source_alpha_min, static_cast<double>(alpha_value));
                        local_source_alpha_max =
                            std::max(local_source_alpha_max, static_cast<double>(alpha_value));
                        local_source_alpha_sum += static_cast<double>(alpha_value);
                        ++local_sources;
                    }
                }
            }
        }
    }

    long long global_terrain = 0;
    long long global_alpha_valid = 0;
    long long global_marked_sources = 0;
    long long global_sources = 0;
    long long global_source_without_valid_alpha = 0;
    long long global_invalid_source = 0;
    long long global_invalid_alpha = 0;
    long long global_invalid_flow_angle = 0;
    long long global_source_alpha_bounds = 0;
    long long global_invalid_dfi = 0;
    long long global_valid_dfi = 0;
    double global_source_alpha_min = 0.0;
    double global_source_alpha_max = 0.0;
    double global_source_alpha_sum = 0.0;
    double global_dfi_min = 0.0;
    double global_dfi_max = 0.0;
    double global_flow_angle_min = 0.0;
    double global_flow_angle_max = 0.0;
    MPI_Allreduce(&local_terrain, &global_terrain, 1, MPI_LONG_LONG, MPI_SUM, MCW);
    MPI_Allreduce(&local_alpha_valid, &global_alpha_valid, 1, MPI_LONG_LONG, MPI_SUM, MCW);
    MPI_Allreduce(&local_marked_sources, &global_marked_sources, 1, MPI_LONG_LONG, MPI_SUM, MCW);
    MPI_Allreduce(&local_sources, &global_sources, 1, MPI_LONG_LONG, MPI_SUM, MCW);
    MPI_Allreduce(
        &local_source_without_valid_alpha,
        &global_source_without_valid_alpha,
        1,
        MPI_LONG_LONG,
        MPI_SUM,
        MCW);
    MPI_Allreduce(&local_invalid_source, &global_invalid_source, 1, MPI_LONG_LONG, MPI_SUM, MCW);
    MPI_Allreduce(&local_invalid_alpha, &global_invalid_alpha, 1, MPI_LONG_LONG, MPI_SUM, MCW);
    MPI_Allreduce(
        &local_invalid_flow_angle,
        &global_invalid_flow_angle,
        1,
        MPI_LONG_LONG,
        MPI_SUM,
        MCW);
    MPI_Allreduce(&local_source_alpha_bounds, &global_source_alpha_bounds, 1, MPI_LONG_LONG, MPI_SUM, MCW);
    MPI_Allreduce(&local_invalid_dfi, &global_invalid_dfi, 1, MPI_LONG_LONG, MPI_SUM, MCW);
    MPI_Allreduce(&local_valid_dfi, &global_valid_dfi, 1, MPI_LONG_LONG, MPI_SUM, MCW);
    MPI_Allreduce(&local_source_alpha_min, &global_source_alpha_min, 1, MPI_DOUBLE, MPI_MIN, MCW);
    MPI_Allreduce(&local_source_alpha_max, &global_source_alpha_max, 1, MPI_DOUBLE, MPI_MAX, MCW);
    MPI_Allreduce(&local_source_alpha_sum, &global_source_alpha_sum, 1, MPI_DOUBLE, MPI_SUM, MCW);
    MPI_Allreduce(&local_dfi_min, &global_dfi_min, 1, MPI_DOUBLE, MPI_MIN, MCW);
    MPI_Allreduce(&local_dfi_max, &global_dfi_max, 1, MPI_DOUBLE, MPI_MAX, MCW);
    MPI_Allreduce(&local_flow_angle_min, &global_flow_angle_min, 1, MPI_DOUBLE, MPI_MIN, MCW);
    MPI_Allreduce(&local_flow_angle_max, &global_flow_angle_max, 1, MPI_DOUBLE, MPI_MAX, MCW);

    if (rank == 0) {
        std::printf("Terrain-valid cells: %lld\n", global_terrain);
        std::printf("Initiating source cells: %lld\n", global_sources);
    }
    if (global_terrain == 0 || global_sources == 0 || global_invalid_source > 0 ||
        global_invalid_alpha > 0 || global_invalid_flow_angle > 0 ||
        global_source_alpha_bounds > 0 || global_invalid_dfi > 0 ||
        global_valid_dfi == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "Input validation failed.\n");
            std::fprintf(stderr, "  invalid source cells: %lld\n", global_invalid_source);
            std::fprintf(stderr, "  invalid alpha cells: %lld\n", global_invalid_alpha);
            std::fprintf(stderr, "  invalid D-Infinity angle cells: %lld\n", global_invalid_flow_angle);
            std::fprintf(stderr, "  source alpha outside [%.1f, %.1f]: %lld\n",
                         kFixedAlphaMin, kFixedAlphaMax, global_source_alpha_bounds);
            std::fprintf(stderr, "  invalid DFI cells: %lld\n", global_invalid_dfi);
            std::fprintf(stderr, "  valid DFI cells: %lld\n", global_valid_dfi);
        }
        return 3;
    }

    delete sourceData;
    sourceData = nullptr;
    delete alphaData;
    alphaData = nullptr;

    std::queue<node> que;
    neighbor->clearBorders();
    initNeighborDinfup(neighbor, flowData, &que, nx, ny, 0, nullptr, nullptr, 0);

    const double read_t = MPI_Wtime();
    bool finished = false;
    long long local_processed = 0;
    long long local_candidate_evaluations = 0;
    long long local_accepted = 0;
    long long local_skipped_missing_dfi = 0;
    long long local_skipped_runout_distance_cap = 0;

    while (!finished) {
        while (!que.empty()) {
            node current = que.front();
            que.pop();
            const int x = current.x;
            const int y = current.y;
            if (!flowData->isInPartition(x, y) || felData->isNodata(x, y)) continue;
            int16_t already_processed = 0;
            processedMask->getData(x, y, already_processed);
            if (already_processed != 0) continue;
            processedMask->setData(x, y, static_cast<int16_t>(1));
            ++local_processed;

            double best_beta = kDoubleNodata;
            double best_alpha = kDoubleNodata;
            double best_dfs = kDoubleNodata;
            double best_parent_alpha = kDoubleNodata;
            double best_src_elev = kDoubleNodata;
            beta->getData(x, y, best_beta);
            dynamicAlpha->getData(x, y, best_alpha);
            dfs->getData(x, y, best_dfs);
            if (parentDynamicAlpha != nullptr) {
                parentDynamicAlpha->getData(x, y, best_parent_alpha);
            }
            srcElev->getData(x, y, best_src_elev);

            for (short k = 1; k <= 8; ++k) {
                const int ux = x + d1[k];
                const int uy = y + d2[k];
                if (!flowData->hasAccess(ux, uy) || flowData->isNodata(ux, uy)) continue;

                float upstream_angle = 0.0f;
                flowData->getData(ux, uy, upstream_angle);
                double dxc = 0.0;
                double dyc = 0.0;
                get_safe_dxdyc(flowData, uy, ny, dxc, dyc);
                const double p = prop(upstream_angle, (k + 4) % 8, dxc, dyc);
                if (p <= 0.0 || p < opt.proportion_threshold) continue;

                if (dfiData->isNodata(x, y)) {
                    ++local_skipped_missing_dfi;
                    continue;
                }

                double upstream_alpha = kDoubleNodata;
                double upstream_dfs = kDoubleNodata;
                double upstream_src_elev = kDoubleNodata;
                dynamicAlpha->getData(ux, uy, upstream_alpha);
                dfs->getData(ux, uy, upstream_dfs);
                srcElev->getData(ux, uy, upstream_src_elev);
                if (!finite_non_nodata(upstream_alpha) ||
                    !finite_non_nodata(upstream_dfs) ||
                    !finite_non_nodata(upstream_src_elev)) {
                    continue;
                }

                float dfi_value = 0.0f;
                dfiData->getData(x, y, dfi_value);
                if (!std::isfinite(dfi_value) || dfi_value < 0.0f || dfi_value > 1.0f) continue;

                const double step_distance = dist[static_cast<size_t>(y) * 9 + k];
                const double d = upstream_dfs + step_distance;
                if (d <= 0.0) continue;
                if (d > kMaxPlanimetricFlowPathLengthM) {
                    ++local_skipped_runout_distance_cap;
                    continue;
                }

                float cell_elev = 0.0f;
                felData->getData(x, y, cell_elev);
                const double beta_candidate =
                    std::atan((upstream_src_elev - static_cast<double>(cell_elev)) / d) * 180.0 / PI;

                const double delta_alpha = compute_delta_alpha(
                    static_cast<double>(dfi_value),
                    opt.dfi_mid,
                    opt.alpha_gain_per_meter,
                    step_distance);
                const double alpha_candidate = clamp_dynamic_alpha(upstream_alpha + delta_alpha);

                ++local_candidate_evaluations;
                if (beta_candidate < alpha_candidate) continue;
                ++local_accepted;

                if (candidate_is_better(best_beta, best_alpha, beta_candidate, alpha_candidate)) {
                    best_beta = beta_candidate;
                    best_alpha = alpha_candidate;
                    best_dfs = d;
                    best_parent_alpha = upstream_alpha;
                    best_src_elev = upstream_src_elev;
                }
            }

            if (finite_non_nodata(best_beta)) {
                beta->setData(x, y, best_beta);
                dynamicAlpha->setData(x, y, best_alpha);
                dfs->setData(x, y, best_dfs);
                if (parentDynamicAlpha != nullptr) {
                    parentDynamicAlpha->setData(x, y, best_parent_alpha);
                }
                srcElev->setData(x, y, best_src_elev);
            }

            float angle = 0.0f;
            flowData->getData(x, y, angle);
            double dxc = 0.0;
            double dyc = 0.0;
            get_safe_dxdyc(flowData, y, ny, dxc, dyc);
            for (short k = 1; k <= 8; ++k) {
                const double p = prop(angle, k, dxc, dyc);
                if (p <= 0.0) continue;
                const int dx = x + d1[k];
                const int dy = y + d2[k];
                if (!flowData->hasAccess(dx, dy) || flowData->isNodata(dx, dy)) continue;
                neighbor->addToData(dx, dy, static_cast<int16_t>(-1));
                int16_t ncount = -32768;
                if (flowData->isInPartition(dx, dy) && neighbor->getData(dx, dy, ncount) == 0) {
                    node next;
                    next.x = dx;
                    next.y = dy;
                    que.push(next);
                }
            }
        }

        dynamicAlpha->share();
        dfs->share();
        srcElev->share();
        neighbor->addBorders();

        for (int x = 0; x < nx; ++x) {
            int16_t border = 0;
            int16_t inside = 0;
            if (neighbor->getData(x, -1, border) != 0 && neighbor->getData(x, 0, inside) == 0) {
                node n;
                n.x = x;
                n.y = 0;
                que.push(n);
            }
            if (neighbor->getData(x, ny, border) != 0 && neighbor->getData(x, ny - 1, inside) == 0) {
                node n;
                n.x = x;
                n.y = ny - 1;
                que.push(n);
            }
        }
        neighbor->clearBorders();

        finished = que.empty();
        finished = neighbor->ringTerm(finished);
    }

    long long local_runout = 0;
    long long local_deposition = 0;
    long long local_remaining_dependency = 0;
    double local_dynamic_alpha_min = std::numeric_limits<double>::infinity();
    double local_dynamic_alpha_max = -std::numeric_limits<double>::infinity();
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            int16_t mask = 0;
            int16_t dep = 0;
            if (!felData->isNodata(x, y) && !flowData->isNodata(x, y)) {
                double b = kDoubleNodata;
                beta->getData(x, y, b);
                if (finite_non_nodata(b)) {
                    mask = 1;
                    ++local_runout;
                    double final_alpha = kDoubleNodata;
                    dynamicAlpha->getData(x, y, final_alpha);
                    if (finite_non_nodata(final_alpha)) {
                        local_dynamic_alpha_min = std::min(local_dynamic_alpha_min, final_alpha);
                        local_dynamic_alpha_max = std::max(local_dynamic_alpha_max, final_alpha);
                    }
                    int16_t src = 0;
                    sourceMask->getData(x, y, src);
                    if (src == 0) {
                        dep = 1;
                        ++local_deposition;
                    } else {
                        beta->setData(x, y, kDoubleNodata);
                    }
                }

                int16_t processed = 0;
                processedMask->getData(x, y, processed);
                if (processed == 0) ++local_remaining_dependency;
            }
            maskOut->setData(x, y, mask);
            depositionOut->setData(x, y, dep);
        }
    }

    long long global_processed = 0;
    long long global_candidate_evaluations = 0;
    long long global_accepted = 0;
    long long global_skipped_missing_dfi = 0;
    long long global_skipped_runout_distance_cap = 0;
    long long global_runout = 0;
    long long global_deposition = 0;
    long long global_remaining_dependency = 0;
    double global_dynamic_alpha_min = 0.0;
    double global_dynamic_alpha_max = 0.0;
    MPI_Allreduce(&local_processed, &global_processed, 1, MPI_LONG_LONG, MPI_SUM, MCW);
    MPI_Allreduce(&local_candidate_evaluations, &global_candidate_evaluations, 1, MPI_LONG_LONG, MPI_SUM, MCW);
    MPI_Allreduce(&local_accepted, &global_accepted, 1, MPI_LONG_LONG, MPI_SUM, MCW);
    MPI_Allreduce(&local_skipped_missing_dfi, &global_skipped_missing_dfi, 1, MPI_LONG_LONG, MPI_SUM, MCW);
    MPI_Allreduce(&local_skipped_runout_distance_cap, &global_skipped_runout_distance_cap, 1, MPI_LONG_LONG, MPI_SUM, MCW);
    MPI_Allreduce(&local_runout, &global_runout, 1, MPI_LONG_LONG, MPI_SUM, MCW);
    MPI_Allreduce(&local_deposition, &global_deposition, 1, MPI_LONG_LONG, MPI_SUM, MCW);
    MPI_Allreduce(&local_remaining_dependency, &global_remaining_dependency, 1, MPI_LONG_LONG, MPI_SUM, MCW);
    MPI_Allreduce(&local_dynamic_alpha_min, &global_dynamic_alpha_min, 1, MPI_DOUBLE, MPI_MIN, MCW);
    MPI_Allreduce(&local_dynamic_alpha_max, &global_dynamic_alpha_max, 1, MPI_DOUBLE, MPI_MAX, MCW);

    const double compute_t = MPI_Wtime();

    tiffIO outAlpha(cstr(opt.out_dynamic_alpha), FLOAT_TYPE, kFloatNodata, ang);
    write_double_partition_as_float(outAlpha, dynamicAlpha, xstart, ystart, nx, ny);
    tiffIO outBeta(cstr(opt.out_beta), FLOAT_TYPE, kFloatNodata, ang);
    write_double_partition_as_float(outBeta, beta, xstart, ystart, nx, ny);
    tiffIO outDfs(cstr(opt.out_dfs), FLOAT_TYPE, kFloatNodata, ang);
    write_double_partition_as_float(outDfs, dfs, xstart, ystart, nx, ny);
    tiffIO outMask(cstr(opt.out_mask), SHORT_TYPE, static_cast<double>(-32768), ang);
    outMask.write(xstart, ystart, ny, nx, maskOut->getGridPointer());
    tiffIO outDep(cstr(opt.out_deposition), SHORT_TYPE, static_cast<double>(-32768), ang);
    outDep.write(xstart, ystart, ny, nx, depositionOut->getGridPointer());
    if (parentDynamicAlpha != nullptr) {
        tiffIO outParent(cstr(opt.out_parent_alpha), FLOAT_TYPE, kFloatNodata, ang);
        write_double_partition_as_float(outParent, parentDynamicAlpha, xstart, ystart, nx, ny);
    }

    const double write_t = MPI_Wtime();
    double local_read = read_t - begin_t;
    double local_compute = compute_t - read_t;
    double local_write = write_t - compute_t;
    double local_total = write_t - begin_t;
    double read_sum = 0.0;
    double compute_sum = 0.0;
    double write_sum = 0.0;
    double total_sum = 0.0;
    MPI_Allreduce(&local_read, &read_sum, 1, MPI_DOUBLE, MPI_SUM, MCW);
    MPI_Allreduce(&local_compute, &compute_sum, 1, MPI_DOUBLE, MPI_SUM, MCW);
    MPI_Allreduce(&local_write, &write_sum, 1, MPI_DOUBLE, MPI_SUM, MCW);
    MPI_Allreduce(&local_total, &total_sum, 1, MPI_DOUBLE, MPI_SUM, MCW);

    if (rank == 0) {
        std::printf("Processed cells: %lld\n", global_processed);
        std::printf("Candidate evaluations: %lld\n", global_candidate_evaluations);
        std::printf("Accepted propagations: %lld\n", global_accepted);
        std::printf("Skipped missing DFI candidates: %lld\n", global_skipped_missing_dfi);
        std::printf("Skipped runout distance cap candidates: %lld\n", global_skipped_runout_distance_cap);
        std::printf(
            "Maximum planimetric flow-path length: %.3f m\n",
            kMaxPlanimetricFlowPathLengthM);
        std::printf("Runout cells: %lld\n", global_runout);
        std::printf("Pure depositional cells: %lld\n", global_deposition);
        std::printf("Remaining dependency cells: %lld\n", global_remaining_dependency);
        std::printf("Average read time: %.3f s\n", read_sum / size);
        std::printf("Average compute time: %.3f s\n", compute_sum / size);
        std::printf("Average write time: %.3f s\n", write_sum / size);
        std::printf("Average total time: %.3f s\n", total_sum / size);
    }

    int stats_write_failed = 0;
    if (rank == 0 && !opt.stats_json.empty()) {
        std::ofstream stats(opt.stats_json, std::ios::out | std::ios::trunc);
        if (!stats) {
            std::fprintf(stderr, "Could not create engine statistics JSON: %s\n", opt.stats_json.c_str());
            stats_write_failed = 1;
        } else {
            const double source_fraction =
                global_terrain > 0 ? static_cast<double>(global_sources) / global_terrain : 0.0;
            const double marked_source_fraction =
                global_terrain > 0 ? static_cast<double>(global_marked_sources) / global_terrain : 0.0;
            const double dfi_valid_fraction =
                global_terrain > 0 ? static_cast<double>(global_valid_dfi) / global_terrain : 0.0;
            const double source_alpha_mean =
                global_sources > 0 ? global_source_alpha_sum / global_sources : 0.0;
            stats << std::setprecision(17)
                  << "{\n"
                  << "  \"schema_version\": 1,\n"
                  << "  \"processes\": " << size << ",\n"
                  << "  \"counts\": {\n"
                  << "    \"terrain_valid_cells\": " << global_terrain << ",\n"
                  << "    \"alpha_valid_cells\": " << global_alpha_valid << ",\n"
                  << "    \"marked_source_cells\": " << global_marked_sources << ",\n"
                  << "    \"source_cells\": " << global_sources << ",\n"
                  << "    \"source_cells_without_valid_alpha\": "
                  << global_source_without_valid_alpha << ",\n"
                  << "    \"dfi_valid_cells\": " << global_valid_dfi << ",\n"
                  << "    \"terrain_valid_cells_without_valid_dfi\": "
                  << (global_terrain - global_valid_dfi) << ",\n"
                  << "    \"processed_cells\": " << global_processed << ",\n"
                  << "    \"candidate_evaluations\": " << global_candidate_evaluations << ",\n"
                  << "    \"accepted_candidate_propagations\": " << global_accepted << ",\n"
                  << "    \"candidate_cells_skipped_missing_dfi\": "
                  << global_skipped_missing_dfi << ",\n"
                  << "    \"candidates_skipped_runout_length_cap\": "
                  << global_skipped_runout_distance_cap << ",\n"
                  << "    \"runout_cells\": " << global_runout << ",\n"
                  << "    \"pure_depositional_cells\": " << global_deposition << ",\n"
                  << "    \"remaining_dependency_cells\": " << global_remaining_dependency << "\n"
                  << "  },\n"
                  << "  \"fractions\": {\n"
                  << "    \"source_fraction\": " << source_fraction << ",\n"
                  << "    \"marked_source_fraction\": " << marked_source_fraction << ",\n"
                  << "    \"dfi_valid_fraction_over_terrain\": " << dfi_valid_fraction << "\n"
                  << "  },\n"
                  << "  \"ranges\": {\n"
                  << "    \"dinf_flow_radians\": [" << global_flow_angle_min << ", "
                  << global_flow_angle_max << "],\n"
                  << "    \"dfi\": [" << global_dfi_min << ", " << global_dfi_max << "],\n"
                  << "    \"source_alpha_degrees\": {\"min\": " << global_source_alpha_min
                  << ", \"max\": " << global_source_alpha_max
                  << ", \"mean\": " << source_alpha_mean << "},\n"
                  << "    \"dynamic_alpha_degrees\": {\"min\": " << global_dynamic_alpha_min
                  << ", \"max\": " << global_dynamic_alpha_max << "}\n"
                  << "  },\n"
                  << "  \"timing_seconds\": {\n"
                  << "    \"average_read\": " << read_sum / size << ",\n"
                  << "    \"average_compute\": " << compute_sum / size << ",\n"
                  << "    \"average_write\": " << write_sum / size << ",\n"
                  << "    \"average_total\": " << total_sum / size << "\n"
                  << "  },\n"
                  << "  \"parameters\": {\n"
                  << "    \"proportion_threshold\": " << opt.proportion_threshold << ",\n"
                  << "    \"dfi_mid\": " << opt.dfi_mid << ",\n"
                  << "    \"alpha_gain_per_meter\": " << opt.alpha_gain_per_meter << ",\n"
                  << "    \"max_planimetric_flow_path_length_m\": "
                  << kMaxPlanimetricFlowPathLengthM << "\n"
                  << "  }\n"
                  << "}\n";
            if (!stats.good()) {
                std::fprintf(stderr, "Failed writing engine statistics JSON: %s\n", opt.stats_json.c_str());
                stats_write_failed = 1;
            }
        }
    }
    MPI_Bcast(&stats_write_failed, 1, MPI_INT, 0, MCW);

    delete[] dist;
    delete flowData;
    delete felData;
    delete dfiData;
    delete beta;
    delete dfs;
    delete dynamicAlpha;
    if (parentDynamicAlpha != nullptr) delete parentDynamicAlpha;
    delete srcElev;
    delete sourceMask;
    delete processedMask;
    delete neighbor;
    delete maskOut;
    delete depositionOut;

    if (stats_write_failed != 0) return 10;
    return global_remaining_dependency == 0 ? 0 : 9;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MCW, &rank);

    Options opt;
    if (!parse_args(argc, argv, opt)) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    const int rc = run_dynamic_alpha_mpi(opt);
    MPI_Finalize();
    return rc;
}
