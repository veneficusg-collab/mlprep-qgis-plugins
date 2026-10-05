#!/usr/bin/env python3
"""
Step 4 end-to-end local verification -- supersedes test_step4_fast.py.

WHAT WAS WRONG (confirmed from your own diagnostic + source + metadata)
-------------------------------------------------------------------------
1. dinfflowdir writes slp.tif as a TANGENT (rise/run), not degrees.
   Your slp.tif: min=0, max=4.082 (a real ~76 degree slope).
2. step0_baseline_alpha_angle.py takes --slope and assumes it is already
   in degrees -- it does no conversion. Fed the tangent directly, every
   valid cell reads as "slope < 10 degrees" (its class 0), which that
   script deliberately writes as alpha NoData by policy.
   Your alpha_metadata.json: class 0 = 234,312,676 / 234,312,676 cells
   (100%). alpha.tif is entirely NoData.
3. step4_deposition_zone_mpi.cpp only counts a cell as an "initiating
   source cell" if alpha is ALSO valid there. With alpha 100% empty,
   every one of your 43.3M PINN-flagged source cells silently fails that
   check and falls into a counter the C++ computes but never prints.
   That's why "Initiating source cells: 0" never changed across 21 runs
   no matter what the source raster contained.

WHAT THIS SCRIPT DOES
----------------------
Using only files already on disk from your last run (no Go/PINN/TauDEM
rerun needed):
  1. Converts slp.tif (tangent) -> slp_degrees.tif (real degrees).
  2. Re-invokes the REAL step0_baseline_alpha_angle.py on the corrected
     raster to regenerate alpha.tif -- same code path production uses,
     not a reimplementation.
  3. Builds source_final.tif / dfi_final.tif / alpha_final.tif honoring
     every rule in step4_deposition_zone_mpi.cpp (binary source, placed
     only where alpha is valid AND alpha within [6, 72]).
  4. Runs the compiled C++/MPI binary directly and streams its output.

Expect this to take well under a minute for steps 1-3, plus whatever the
C++ engine itself needs for the real routing pass.

USAGE
-----
Edit CONFIG below, then:
  ~/.mlprep_envs/venv/bin/python test_step4_complete.py
"""

import os
import sys
import json
import subprocess
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ============================== CONFIG ================================
OUTPUT_DIR = "/Users/ml-prepproject/Downloads/Plugin_v4.3 Output 14/deposition_zone_output"
PLUGIN_DIR = os.path.expanduser(
    "~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/MLPREP_Plugin_v4_3"
)
PYTHON_EXE = os.path.expanduser("~/.mlprep_envs/venv/bin/python")
MPI_EXE_CANDIDATES = ["/opt/homebrew/bin/mpiexec", "/usr/local/bin/mpiexec", "/opt/homebrew/bin/mpirun"]
CORES = 4

STEP0_SCRIPT = os.path.join(PLUGIN_DIR, "ml_dfi_handoff", "steps",
                             "step0_baseline_alpha_angle", "step0_baseline_alpha_angle.py")
CPP_EXE = os.path.join(PLUGIN_DIR, "ml_dfi_handoff", "steps", "step4_deposition_zone",
                        "cpp_mpi_port", "build_manual", "step4_deposition_zone_mpi")

# Engine constants, copied verbatim from step4_deposition_zone_mpi.cpp
ALPHA_VALID_MIN, ALPHA_VALID_MAX = 1.0, 90.0
ALPHA_FIXED_MIN, ALPHA_FIXED_MAX = 6.0, 72.0
DFI_MID = 0.69
SOURCE_THRESHOLD = 0.01
OUT_NODATA = -9999.0
# ========================================================================


def find_mpi_exe():
    for c in MPI_EXE_CANDIDATES:
        if os.path.exists(c):
            return c
    return "mpiexec"


def step1_convert_slope(output_dir):
    print("\n--- Step 1: converting slp.tif (tangent) to degrees ---")
    slp = os.path.join(output_dir, "slp.tif")
    slp_deg = os.path.join(output_dir, "slp_degrees.tif")
    with rasterio.open(slp) as src:
        tangent = src.read(1)
        nodata = src.nodata
        meta = src.profile.copy()
    valid = np.isfinite(tangent)
    if nodata is not None:
        valid &= (tangent != np.float32(nodata))
    degrees = np.full(tangent.shape, OUT_NODATA, dtype=np.float32)
    degrees[valid] = np.degrees(np.arctan(tangent[valid])).astype(np.float32)
    print(f"  tangent range: [{tangent[valid].min():.4f}, {tangent[valid].max():.4f}]")
    print(f"  degrees range: [{degrees[valid].min():.4f}, {degrees[valid].max():.4f}]")
    meta.update(dtype=rasterio.float32, nodata=OUT_NODATA, compress="LZW")
    with rasterio.open(slp_deg, "w", **meta) as dst:
        dst.write(degrees, 1)
    return slp_deg


def step2_rerun_step0(output_dir, slp_deg):
    print("\n--- Step 2: regenerating alpha.tif via the real Step 0 script ---")
    if not os.path.exists(STEP0_SCRIPT):
        sys.exit(f"[FATAL] Step 0 script not found: {STEP0_SCRIPT}")
    alpha_out = os.path.join(output_dir, "alpha.tif")
    cmd = [PYTHON_EXE, STEP0_SCRIPT,
           "--slope", slp_deg, "--output-alpha", alpha_out,
           "--nodata", str(OUT_NODATA), "--overwrite"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(f"[FATAL] Step 0 failed with exit code {result.returncode}")
    meta_path = os.path.join(output_dir, "alpha_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"  valid_cell_count: {meta['valid_cell_count']}")
    print(f"  alpha_nodata_cell_count: {meta['alpha_nodata_cell_count']}")
    print("  cells per class:", {k: v for k, v in meta["cell_count_per_class"].items() if v > 0})
    return alpha_out


def step3_build_cpp_inputs(output_dir, alpha_path):
    print("\n--- Step 3: building alpha-aware source/dfi/alpha rasters ---")
    ang = os.path.join(output_dir, "ang.tif")
    fel = os.path.join(output_dir, "fel.tif")
    dfi_clean = os.path.join(output_dir, "dfi_clean.tif")
    if not os.path.exists(dfi_clean):
        dfi_clean = os.path.join(output_dir, "ml_dfi_probability.tif")
    pinn_tif = os.path.join(output_dir, "ml_dfi_probability.tif")

    with rasterio.open(ang) as t:
        t_meta = t.profile.copy()
        t_transform, t_crs, t_shape = t.transform, t.crs, t.shape
        ang_nodata = t.nodata
        ang_data = t.read(1).astype(np.float32)
    with rasterio.open(fel) as f:
        fel_data = f.read(1).astype(np.float32)
        fel_nodata = f.nodata

    valid_terrain = np.isfinite(ang_data) & np.isfinite(fel_data)
    if ang_nodata is not None:
        valid_terrain &= (ang_data != np.float32(ang_nodata))
    if fel_nodata is not None:
        valid_terrain &= (fel_data != np.float32(fel_nodata))
    print(f"  terrain cells: {int(valid_terrain.sum())}")
    del fel_data

    with rasterio.open(alpha_path) as a:
        alpha_data = a.read(1).astype(np.float32)
        alpha_nodata = a.nodata
    alpha_present = valid_terrain.copy()
    if alpha_nodata is not None:
        alpha_present &= (alpha_data != np.float32(alpha_nodata))
    alpha_bad = alpha_present & (~np.isfinite(alpha_data) |
                                 (alpha_data < ALPHA_VALID_MIN) | (alpha_data > ALPHA_VALID_MAX))
    alpha_data[alpha_bad] = np.float32(OUT_NODATA)
    alpha_present &= ~alpha_bad
    alpha_usable = alpha_present & (alpha_data >= ALPHA_FIXED_MIN) & (alpha_data <= ALPHA_FIXED_MAX)
    print(f"  alpha valid on terrain: {int(alpha_present.sum())} | "
          f"usable [{ALPHA_FIXED_MIN},{ALPHA_FIXED_MAX}]: {int(alpha_usable.sum())}")

    out_meta = t_meta.copy()
    out_meta.update(dtype=rasterio.float32, nodata=OUT_NODATA, count=1)
    final_alpha = os.path.join(output_dir, "alpha_final.tif")
    alpha_out = np.full(t_shape, OUT_NODATA, dtype=np.float32)
    alpha_out[alpha_present] = alpha_data[alpha_present]
    with rasterio.open(final_alpha, "w", **out_meta) as dst:
        dst.write(alpha_out, 1)
    del alpha_out, alpha_data

    with rasterio.open(dfi_clean) as d:
        dfi_warp = np.full(t_shape, OUT_NODATA, dtype=np.float32)
        reproject(source=rasterio.band(d, 1), destination=dfi_warp,
                  src_transform=d.transform, src_crs=d.crs,
                  dst_transform=t_transform, dst_crs=t_crs,
                  src_nodata=d.nodata, dst_nodata=OUT_NODATA, resampling=Resampling.nearest)
    dfi_ok = valid_terrain & (dfi_warp != np.float32(OUT_NODATA)) & np.isfinite(dfi_warp) & \
             (dfi_warp >= 0.0) & (dfi_warp <= 1.0)
    dfi_out = np.full(t_shape, OUT_NODATA, dtype=np.float32)
    dfi_out[valid_terrain] = np.float32(DFI_MID)
    dfi_out[dfi_ok] = np.clip(dfi_warp[dfi_ok], 0.0, 1.0)
    print(f"  dfi predicted: {int(dfi_ok.sum())} | neutral-filled: {int(valid_terrain.sum() - dfi_ok.sum())}")
    final_dfi = os.path.join(output_dir, "dfi_final.tif")
    with rasterio.open(final_dfi, "w", **out_meta) as dst:
        dst.write(dfi_out, 1)
    del dfi_warp, dfi_out

    with rasterio.open(pinn_tif) as p:
        pinn_data = np.full(t_shape, OUT_NODATA, dtype=np.float32)
        reproject(source=rasterio.band(p, 1), destination=pinn_data,
                  src_transform=p.transform, src_crs=p.crs,
                  dst_transform=t_transform, dst_crs=t_crs,
                  src_nodata=p.nodata, dst_nodata=OUT_NODATA, resampling=Resampling.nearest)
    is_source = valid_terrain & (pinn_data > SOURCE_THRESHOLD) & \
                (pinn_data != np.float32(OUT_NODATA)) & np.isfinite(pinn_data)
    legal = is_source & alpha_usable
    print(f"  PINN candidates: {int(is_source.sum())} | legal (alpha-usable): {int(legal.sum())}")
    if legal.sum() == 0:
        sys.exit("[FATAL] Still zero legal source cells after the units fix -- "
                 "share this script's full output, something else is off.")

    source_out = np.full(t_shape, OUT_NODATA, dtype=np.float32)
    source_out[valid_terrain] = 0.0
    source_out[legal] = 1.0
    final_source = os.path.join(output_dir, "source_final.tif")
    with rasterio.open(final_source, "w", **out_meta) as dst:
        dst.write(source_out, 1)

    return final_source, final_dfi, final_alpha


def step4_run_cpp(output_dir, final_source, final_dfi, final_alpha):
    print("\n--- Step 4: running the C++/MPI engine ---")
    fel = os.path.join(output_dir, "fel.tif")
    ang = os.path.join(output_dir, "ang.tif")
    if not os.path.exists(CPP_EXE):
        sys.exit(f"[FATAL] C++ binary not found: {CPP_EXE}")
    mpi_exe = find_mpi_exe()
    cmd = [
        mpi_exe, "-n", str(CORES), CPP_EXE,
        "--fel", fel, "--ang", ang,
        "--source", final_source, "--dfi", final_dfi, "--alpha", final_alpha,
        "--out-alpha", os.path.join(output_dir, "step4_dynamic_alpha.tif"),
        "--out-beta", os.path.join(output_dir, "step4_beta_angle.tif"),
        "--out-dfs", os.path.join(output_dir, "step4_dfs.tif"),
        "--out-mask", os.path.join(output_dir, "step4_runout_mask.tif"),
        "--out-deposition", os.path.join(output_dir, "step4_pure_depositional_mask.tif"),
        "--threshold", "0.01", "--dfi-mid", str(DFI_MID), "--alpha-gain-per-meter", "0.03333",
    ]
    print(" ".join(cmd), "\n")
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONHOME", None)
    clean_env.pop("PYTHONPATH", None)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=clean_env)
    for line in iter(process.stdout.readline, ""):
        if line.strip():
            print(f" > {line.strip()}")
    process.wait()
    print(f"\n--- EXIT CODE: {process.returncode} ---")


if __name__ == "__main__":
    if not os.path.isdir(OUTPUT_DIR):
        sys.exit(f"[FATAL] OUTPUT_DIR not found: {OUTPUT_DIR}")
    slp_deg = step1_convert_slope(OUTPUT_DIR)
    alpha_path = step2_rerun_step0(OUTPUT_DIR, slp_deg)
    final_source, final_dfi, final_alpha = step3_build_cpp_inputs(OUTPUT_DIR, alpha_path)
    step4_run_cpp(OUTPUT_DIR, final_source, final_dfi, final_alpha)
