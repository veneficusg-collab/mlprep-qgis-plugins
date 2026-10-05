#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 1: Observed alpha-angle (a_obs) workflow."""

from __future__ import annotations

import multiprocessing
import csv
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np
from osgeo import ogr
from shapely import wkt as shapely_wkt
from shapely.geometry import Polygon, box

from dp1_models import Step1Config
from dp1_support import (
    CONTOUR_INTERVAL_M,
    LayerSpec,
    Params,
    complexity_ratio,
    configure_logging,
    ensure_projected_meter_crs,
    ensure_qgis_runtime_only,
    fmt_qc_reason,
    load_step1_core_engine,
    log,
    log_timing,
    normalize_polygon_geometry,
    open_dem_dataset,
    read_landslide_rows,
    resolve_output_gpkg,
    timer_start,
    validate_params,
    write_gpkg_layers,
    write_qc_csv,
)
from dp1_worker import (
    Step1PolygonJob,
    Step1PolygonWorkerResult,
    execute_step1_polygon_job,
)


def _core_config_from_params(p: Params) -> Step1Config:
    return Step1Config(
        contour_interval_m=float(CONTOUR_INTERVAL_M),
        buffer_min_m=float(p.buffer_min_m),
        buffer_pixel_multiple=float(p.buffer_pixel_multiple),
        boundary_sample_step_m=float(p.boundary_sample_step_m),
        crown_percentile_high=float(p.crown_percentile_high),
        qc_percentile_high=float(p.qc_percentile_high),
        complexity_threshold=float(p.complexity_threshold),
        enable_complexity_filter=bool(p.enable_early_polygon_filter),
        endpoint_snap_search_radius_px=int(p.endpoint_snap_search_radius_px),
    )


def _new_polygon_record(polygon_id: str, crown_id: str, toe_id: str) -> Dict[str, Any]:
    return {
        "polygon_id": polygon_id,
        "crown_id": crown_id,
        "toe_id": toe_id,
        "polygon_geometry": None,
        "crown_point": None,
        "toe_point": None,
        "runout_path": None,
        "H_c": None,
        "H_f": None,
        "H": None,
        "L": None,
        "L_2d": None,
        "L_3d": None,
        "a_obs_deg": None,
        "polygon_area_m2": None,
        "complexity_ratio": None,
        "early_excluded": False,
        "path_method": "",
        "path_success": False,
        "fallback_triggered": False,
        "diagnostic_reason": "OK",
        "buffer_dist_m": None,
        "path_steps": None,
        "outside_mask_steps": None,
        "outside_mask_len_2d": None,
        "outside_mask_len_3d": None,
        "outside_mask_fraction_2d": None,
        "outside_mask_fraction_3d": None,
        "max_consecutive_outside_steps": None,
        "tortuosity_2d": None,
        "reasons": set(),
    }


def _qc_pass_a_validate_geometry_and_overlap(
    rec: Dict[str, Any],
    geom: Any,
    dem_extent_poly: Polygon,
) -> Optional[Polygon]:
    reasons: Set[str] = rec["reasons"]
    normalized = normalize_polygon_geometry(geom)
    if normalized is None:
        reasons.add("invalid_geometry")
        return None
    poly_geom, analysis_poly = normalized
    rec["polygon_geometry"] = poly_geom
    if not analysis_poly.intersects(dem_extent_poly):
        reasons.add("no_dem_overlap")
        return None
    return analysis_poly


def _qc_pass_b_apply_complexity_filter(rec: Dict[str, Any], analysis_poly: Polygon, p: Params) -> bool:
    reasons: Set[str] = rec["reasons"]
    ratio = complexity_ratio(analysis_poly)
    area_m2 = float(analysis_poly.area)
    rec["polygon_area_m2"] = area_m2
    rec["complexity_ratio"] = ratio
    is_complex = bool(
        p.enable_early_polygon_filter
        and ratio is not None
        and ratio > float(p.complexity_threshold)
    )
    if is_complex:
        reasons.add("complexity_ratio_exceeds_threshold")
    is_small = area_m2 < float(p.early_exclusion_area_threshold_m2)
    if is_small:
        reasons.add("polygon_area_below_threshold")
    should_exclude = load_step1_core_engine().should_exclude_polygon_early(
        area_m2,
        ratio,
        enabled=bool(p.enable_early_polygon_filter),
        complexity_threshold=float(p.complexity_threshold),
        area_threshold_m2=float(p.early_exclusion_area_threshold_m2),
    )
    if should_exclude:
        reasons.add("early_excluded_complex_or_small_polygon")
        rec["diagnostic_reason"] = "early_excluded_complex_or_small_polygon"
        rec["early_excluded"] = True
        return True
    return False


def _default_max_workers() -> int:
    return int(min(4, max(1, (os.cpu_count() or 2) - 1)))


def _resolve_parallel_settings(p: Params) -> tuple[bool, int]:
    max_workers = int(p.max_workers) if p.max_workers is not None else _default_max_workers()
    if int(p.worker_chunk_size) != 1:
        log(
            f"Step 1 worker_chunk_size={int(p.worker_chunk_size)} is not supported in v1; "
            "forcing worker_chunk_size=1.",
        )
        p.worker_chunk_size = 1
    if not bool(p.enable_parallel):
        return False, int(max_workers)
    if int(max_workers) <= 1:
        log("Step 1 parallel mode requested with max_workers <= 1; using serial execution.")
        return False, int(max_workers)
    return True, int(max_workers)


def _build_polygon_worker_job(
    rec: Dict[str, Any],
    analysis_poly: Polygon,
    dem_path: str,
    pixel_size: float,
    p: Params,
    order_index: int,
) -> Step1PolygonJob:
    return Step1PolygonJob(
        order_index=int(order_index),
        polygon_id=str(rec["polygon_id"]),
        crown_id=str(rec["crown_id"]),
        toe_id=str(rec["toe_id"]),
        analysis_polygon_wkt=str(analysis_poly.wkt),
        dem_path=str(dem_path),
        pixel_size_m=float(pixel_size),
        core_config=_core_config_from_params(p),
    )


def _shape_from_wkt(wkt_value: Optional[str]) -> Optional[Any]:
    if not wkt_value:
        return None
    return shapely_wkt.loads(wkt_value)


def _apply_polygon_worker_result(rec: Dict[str, Any], result: Step1PolygonWorkerResult) -> None:
    reasons: Set[str] = rec["reasons"]
    reasons.update(str(reason) for reason in result.qc_reasons if str(reason).strip())
    rec["crown_point"] = _shape_from_wkt(result.crown_point_wkt)
    rec["toe_point"] = _shape_from_wkt(result.toe_point_wkt)
    rec["runout_path"] = _shape_from_wkt(result.runout_path_wkt)
    rec["H_c"] = result.h_c
    rec["H_f"] = result.h_f
    rec["H"] = result.h_m
    rec["L"] = result.l_m
    rec["L_2d"] = result.l_2d_m
    rec["L_3d"] = result.l_3d_m
    rec["a_obs_deg"] = result.a_obs_deg
    rec["path_method"] = str(result.path_method)
    rec["path_success"] = bool(result.path_success)
    rec["fallback_triggered"] = bool(result.fallback_triggered)
    rec["diagnostic_reason"] = str(result.diagnostic_reason or "OK")
    rec["buffer_dist_m"] = result.buffer_dist_m
    rec["path_steps"] = result.path_steps
    rec["outside_mask_steps"] = result.outside_mask_steps
    rec["outside_mask_len_2d"] = result.outside_mask_len_2d
    rec["outside_mask_len_3d"] = result.outside_mask_len_3d
    rec["outside_mask_fraction_2d"] = result.outside_mask_fraction_2d
    rec["outside_mask_fraction_3d"] = result.outside_mask_fraction_3d
    rec["max_consecutive_outside_steps"] = result.max_consecutive_outside_steps
    rec["tortuosity_2d"] = result.tortuosity_2d


def _log_polygon_timings(result: Step1PolygonWorkerResult) -> None:
    polygon_id = str(result.polygon_id)
    timing_labels = (
        ("workspace_setup_s", "workspace_setup"),
        ("buffer_creation_s", "buffer_creation"),
        ("window_extraction_s", "dem_window_extraction"),
        ("endpoint_derivation_s", "endpoint_derivation"),
        ("contour_generation_s", "contour_generation"),
        ("node_extraction_s", "node_extraction"),
        ("path_generation_s", "path_generation"),
        ("total_s", "polygon_total"),
    )
    for key, label in timing_labels:
        if key not in result.timings:
            continue
        log(f"[timing] {polygon_id}.{label}: {float(result.timings[key]):.3f}s")


def _unexpected_future_failure(job: Step1PolygonJob, debug_traceback: str) -> Step1PolygonWorkerResult:
    return Step1PolygonWorkerResult(
        order_index=int(job.order_index),
        polygon_id=str(job.polygon_id),
        success=False,
        qc_reasons=("worker_exception",),
        diagnostic_reason="worker_exception",
        debug_traceback=str(debug_traceback),
        unexpected_error=True,
    )


def _run_polygon_jobs(
    jobs: Sequence[Step1PolygonJob],
    p: Params,
    worker_fn: Callable[[Step1PolygonJob], Step1PolygonWorkerResult] = execute_step1_polygon_job,
    on_result: Optional[Callable[[Step1PolygonWorkerResult], None]] = None,
) -> list[Step1PolygonWorkerResult]:
    if not jobs:
        return []

    parallel_enabled, max_workers = _resolve_parallel_settings(p)
    if (not parallel_enabled) or len(jobs) <= 1:
        results: list[Step1PolygonWorkerResult] = []
        for idx, job in enumerate(jobs):
            try:
                result = worker_fn(job)
            except Exception:
                result = _unexpected_future_failure(job, traceback.format_exc(limit=8).strip())
            if on_result is not None:
                on_result(result)
            results.append(result)
            if bool(p.fail_fast) and bool(result.unexpected_error):
                for pending_job in jobs[idx + 1 :]:
                    cancelled = Step1PolygonWorkerResult.cancelled(pending_job)
                    if on_result is not None:
                        on_result(cancelled)
                    results.append(cancelled)
                break
        return results

    try:
        mp_context = multiprocessing.get_context(str(p.worker_start_method))
    except ValueError as exc:
        raise ValueError(f"Unsupported Step 1 worker_start_method: {p.worker_start_method}") from exc

    results_by_index: dict[int, Step1PolygonWorkerResult] = {}
    completion_order: list[int] = []
    fail_fast_triggered = False
    future_to_job: dict[Any, Step1PolygonJob] = {}

    with ProcessPoolExecutor(max_workers=int(max_workers), mp_context=mp_context) as executor:
        for job in jobs:
            future_to_job[executor.submit(worker_fn, job)] = job

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            if int(job.order_index) in results_by_index:
                continue
            if future.cancelled():
                result = Step1PolygonWorkerResult.cancelled(job)
            else:
                try:
                    result = future.result()
                except Exception:
                    result = _unexpected_future_failure(job, traceback.format_exc(limit=8).strip())
            results_by_index[int(job.order_index)] = result
            completion_order.append(int(job.order_index))
            if on_result is not None:
                on_result(result)
            if bool(p.fail_fast) and bool(result.unexpected_error) and not fail_fast_triggered:
                fail_fast_triggered = True
                for pending_future, pending_job in future_to_job.items():
                    if pending_future is future or pending_future.done():
                        continue
                    if pending_future.cancel():
                        cancelled = Step1PolygonWorkerResult.cancelled(pending_job)
                        results_by_index[int(pending_job.order_index)] = cancelled
                        completion_order.append(int(pending_job.order_index))
                        if on_result is not None:
                            on_result(cancelled)

    ordered_indices = sorted(results_by_index) if bool(p.preserve_order) else completion_order
    deduped_indices: list[int] = []
    seen: set[int] = set()
    for idx in ordered_indices:
        if int(idx) in seen:
            continue
        seen.add(int(idx))
        deduped_indices.append(int(idx))
    return [results_by_index[idx] for idx in deduped_indices]


def _qc_pass_d_apply_global_outlier_filter(records: Sequence[Dict[str, Any]], qc_percentile_high: float) -> None:
    vals: List[float] = []
    for rec in records:
        if len(rec["reasons"]) != 0:
            continue
        a_val = rec.get("a_obs_deg")
        if a_val is None:
            continue
        try:
            a_float = float(a_val)
        except Exception:
            continue
        if np.isfinite(a_float):
            vals.append(a_float)
    hi = load_step1_core_engine().upper_percentile_threshold(vals, float(qc_percentile_high))
    if hi is None:
        return
    for rec in records:
        if len(rec["reasons"]) != 0:
            continue
        a = rec.get("a_obs_deg")
        if a is None or (not np.isfinite(float(a))):
            rec["reasons"].add("nan_inf")
        elif float(a) > hi:
            rec["reasons"].add("out_of_bounds")


def _finalize_qc_flags(records: Sequence[Dict[str, Any]]) -> None:
    for rec in records:
        reasons: Set[str] = rec["reasons"]
        if len(reasons) == 0:
            rec["qc_flag"] = "OK"
            rec["qc_reason"] = "OK"
        else:
            rec["qc_flag"] = "BAD"
            rec["qc_reason"] = fmt_qc_reason(reasons)


def _compute_h_over_l(h_value: Any, l_value: Any) -> Optional[float]:
    try:
        h_float = float(h_value)
        l_float = float(l_value)
    except Exception:
        return None
    if not np.isfinite(h_float) or not np.isfinite(l_float) or l_float <= 0.0:
        return None
    return float(h_float / l_float)


def _add_runout_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    row["h_over_l"] = _compute_h_over_l(row.get("H"), row.get("L"))
    return row


def _stats_for_values(values: Sequence[Any]) -> Dict[str, Any]:
    arr = _coerce_finite_values(values)
    if int(arr.size) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if int(arr.size) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def _write_runout_class_summary_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    metrics = ("h_over_l", "a_obs_deg", "H", "L", "L_2d", "L_3d")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["runout", "metric", "count", "mean", "median", "std", "min", "max", "q25", "q75"])
        for runout_class in ("long", "medium", "short"):
            class_rows = [row for row in rows if str(row.get("runout", "")).lower() == runout_class]
            for metric in metrics:
                stats = _stats_for_values([row.get(metric) for row in class_rows])
                writer.writerow(
                    [
                        runout_class,
                        metric,
                        stats["count"],
                        stats["mean"],
                        stats["median"],
                        stats["std"],
                        stats["min"],
                        stats["max"],
                        stats["q25"],
                        stats["q75"],
                    ]
                )


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to generate Step 1 skewness visuals. "
            "Install/enable matplotlib in your QGIS Python environment."
        ) from exc
    return plt


def _coerce_finite_values(values: Sequence[Any]) -> np.ndarray:
    out: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            val = float(value)
        except Exception:
            continue
        if np.isfinite(val):
            out.append(val)
    return np.asarray(out, dtype=np.float64)


def _sample_skewness(values: Any) -> Optional[float]:
    arr = _coerce_finite_values(values)
    n = int(arr.size)
    if n < 3:
        return None
    mean = float(np.mean(arr))
    centered = arr - mean
    m2 = float(np.mean(centered**2))
    if m2 <= 1e-12:
        return 0.0
    m3 = float(np.mean(centered**3))
    g1 = float(m3 / (m2 ** 1.5))
    return float((math.sqrt(n * (n - 1)) / (n - 2)) * g1)


def _describe_skewness(skewness: Optional[float]) -> str:
    if skewness is None:
        return "insufficient sample"
    if skewness > 0.5:
        return "right-skewed"
    if skewness < -0.5:
        return "left-skewed"
    return "approximately symmetric"


def _build_skewness_output_paths(out_dir: str, output_prefix: str) -> Dict[str, str]:
    return {
        "aobs_complete_skewness_png": os.path.join(out_dir, f"{output_prefix}_aobs_complete_skewness.png"),
        "aobs_clean_skewness_png": os.path.join(out_dir, f"{output_prefix}_aobs_clean_skewness.png"),
        "h_complete_skewness_png": os.path.join(out_dir, f"{output_prefix}_h_complete_skewness.png"),
        "h_clean_skewness_png": os.path.join(out_dir, f"{output_prefix}_h_clean_skewness.png"),
        "l_complete_skewness_png": os.path.join(out_dir, f"{output_prefix}_l_complete_skewness.png"),
        "l_clean_skewness_png": os.path.join(out_dir, f"{output_prefix}_l_clean_skewness.png"),
    }


def _plot_metric_skewness(
    plt: Any,
    values: Sequence[Any],
    *,
    metric_label: str,
    x_label: str,
    subset_label: str,
    out_png: str,
) -> None:
    arr = _coerce_finite_values(values)
    color = "#2F6B4F" if "clean" in str(subset_label).lower() else "#C16A2A"
    fig, (ax_hist, ax_box) = plt.subplots(
        1,
        2,
        figsize=(11, 4.6),
        gridspec_kw={"width_ratios": [4.0, 1.15]},
    )

    ax_hist.set_title(f"Step 1 {metric_label} Distribution ({subset_label})")
    ax_hist.set_xlabel(x_label)
    ax_hist.set_ylabel("Count")

    if int(arr.size) == 0:
        ax_hist.text(
            0.5,
            0.5,
            f"No valid {metric_label} values",
            ha="center",
            va="center",
            transform=ax_hist.transAxes,
        )
        ax_box.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_box.transAxes)
        ax_box.set_axis_off()
    else:
        bin_count = max(8, min(28, int(math.ceil(math.sqrt(float(arr.size)))) * 2))
        ax_hist.hist(arr, bins=bin_count, color=color, edgecolor="white", linewidth=0.8, alpha=0.9)
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        skewness = _sample_skewness(arr)
        ax_hist.axvline(mean_val, color="#1D3557", linestyle="-", linewidth=1.6, label=f"mean = {mean_val:.2f}")
        ax_hist.axvline(median_val, color="#6A040F", linestyle="--", linewidth=1.6, label=f"median = {median_val:.2f}")
        ax_hist.legend(loc="upper left", fontsize=8, frameon=False)
        ax_hist.text(
            0.98,
            0.95,
            "\n".join(
                [
                    f"n = {int(arr.size)}",
                    f"skewness = {skewness:.3f}" if skewness is not None else "skewness = n/a",
                    f"shape = {_describe_skewness(skewness)}",
                ]
            ),
            transform=ax_hist.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "#F4F1EA", "edgecolor": "#D7C8AE", "boxstyle": "round,pad=0.35"},
        )

        ax_box.boxplot(
            arr,
            vert=False,
            patch_artist=True,
            widths=0.5,
            boxprops={"facecolor": color, "alpha": 0.65, "edgecolor": "#333333"},
            medianprops={"color": "#6A040F", "linewidth": 1.6},
            whiskerprops={"color": "#333333"},
            capprops={"color": "#333333"},
        )
        ax_box.set_title("Spread")
        ax_box.set_xlabel(x_label)
        ax_box.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def run_step1(p: Params) -> Dict[str, str]:
    """Run observed-alpha extraction for mapped landslide polygons."""

    # Step 1 is polygon-by-polygon: validate the input geometry/DEM, derive
    # crown/toe/path metrics for each landslide, then write clean and complete
    # output layers for later analysis.
    ensure_qgis_runtime_only()
    configure_logging(p.log_level)
    validate_params(p)
    out_gpkg = resolve_output_gpkg(p.output_gpkg, p.output_prefix)
    out_dir = os.path.dirname(out_gpkg)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    qc_csv = os.path.join(out_dir, f"{p.output_prefix}_qc_summary.csv")
    runout_class_summary_csv = os.path.join(out_dir, f"{p.output_prefix}_runout_class_summary.csv")
    stage_t0 = timer_start()
    log("Reading inputs...")

    with open_dem_dataset(p.dem) as dem_ds:
        dem_crs = dem_ds.crs
        if dem_crs is None:
            raise RuntimeError("DEM has no CRS.")
        ensure_projected_meter_crs(dem_crs)
        landslide_rows, crs_out = read_landslide_rows(
            source=p.landslide_polygons,
            polygon_id_field=p.polygon_id_field,
            target_crs=dem_crs,
            target_crs_wkt=dem_ds.crs_wkt,
        )
        pixel_size = float(max(abs(float(dem_ds.transform.a)), abs(float(dem_ds.transform.e))))
        dem_extent_poly = box(*dem_ds.bounds)

    records: List[Dict[str, Any]] = []
    jobs: List[Step1PolygonJob] = []
    worker_record_indices: List[int] = []
    n_total = int(len(landslide_rows))
    parallel_enabled, max_workers = _resolve_parallel_settings(p)
    log(
        "Step 1 routing now uses contour-midpoint chaining generated from per-landslide buffered local DEM clips. "
        f"Contour interval is fixed at {float(CONTOUR_INTERVAL_M):.1f} m."
    )
    if parallel_enabled:
        log(
            "Step 1 parallel mode is enabled for per-polygon workers. "
            f"Using start_method={p.worker_start_method!s}, max_workers={int(max_workers)}, worker_chunk_size=1."
        )
    else:
        log("Step 1 parallel mode is disabled; using serial per-polygon execution.")
    log(f"Processing {n_total} landslide polygon(s)...")

    for row in landslide_rows:
        polygon_id = str(row["polygon_id"])
        crown_id = f"{polygon_id}_C1"
        toe_id = f"{polygon_id}_T1"
        rec = _new_polygon_record(polygon_id, crown_id, toe_id)
        analysis_poly = _qc_pass_a_validate_geometry_and_overlap(rec, row.get("geometry"), dem_extent_poly)
        records.append(rec)
        if analysis_poly is None:
            continue
        if _qc_pass_b_apply_complexity_filter(rec, analysis_poly, p):
            continue
        order_index = int(len(records) - 1)
        jobs.append(_build_polygon_worker_job(rec, analysis_poly, p.dem, pixel_size, p, order_index))
        worker_record_indices.append(order_index)

    completed_count = int(len(records) - len(jobs))
    early_excluded_count = sum(bool(rec["early_excluded"]) for rec in records)
    if early_excluded_count:
        log(
            f"Early-excluded {early_excluded_count} complex or small polygon(s) before DEM and path processing "
            f"(complexity ratio > {float(p.complexity_threshold):.3f} or "
            f"area < {float(p.early_exclusion_area_threshold_m2):.1f} m2)."
        )
    worker_results: list[Step1PolygonWorkerResult] = []

    def _handle_worker_result(result: Step1PolygonWorkerResult) -> None:
        nonlocal completed_count
        rec = records[int(result.order_index)]
        _apply_polygon_worker_result(rec, result)
        _log_polygon_timings(result)
        if bool(result.unexpected_error):
            log(
                f"Step 1 worker failure for polygon {result.polygon_id}: {result.diagnostic_reason}",
            )
            if str(result.debug_traceback).strip():
                log(str(result.debug_traceback))
        completed_count += 1
        if (completed_count % p.progress_every == 0) or (completed_count == n_total):
            log(f"Progress: {completed_count}/{n_total} polygon(s) processed")

    worker_results = _run_polygon_jobs(jobs, p, on_result=_handle_worker_result)
    if completed_count < n_total:
        log(f"Progress: {n_total}/{n_total} polygon(s) processed")

    _qc_pass_d_apply_global_outlier_filter(records, p.qc_percentile_high)
    _finalize_qc_flags(records)

    crown_rows: List[Dict[str, Any]] = []
    qc_rows: List[Dict[str, Any]] = []
    polygon_rows: List[Dict[str, Any]] = []
    toe_rows: List[Dict[str, Any]] = []
    runout_rows: List[Dict[str, Any]] = []
    if bool(p.preserve_order):
        output_record_indices = list(range(len(records)))
    else:
        worker_index_set = set(worker_record_indices)
        non_worker_indices = [idx for idx in range(len(records)) if idx not in worker_index_set]
        completed_worker_indices = [int(result.order_index) for result in worker_results]
        output_record_indices = [*non_worker_indices, *completed_worker_indices]
    for rec_idx in output_record_indices:
        rec = records[int(rec_idx)]
        qc_rows.append(
            _add_runout_fields(
                {
                    "polygon_id": rec["polygon_id"],
                    "crown_id": rec["crown_id"],
                    "polygon_area_m2": rec["polygon_area_m2"],
                    "complexity_ratio": rec["complexity_ratio"],
                    "early_excluded": rec["early_excluded"],
                    "H": rec["H"],
                    "L": rec["L"],
                    "L_2d": rec["L_2d"],
                    "L_3d": rec["L_3d"],
                    "a_obs_deg": rec["a_obs_deg"],
                    "path_method": rec["path_method"],
                    "fallback_triggered": rec["fallback_triggered"],
                    "diagnostic_reason": rec["diagnostic_reason"],
                    "outside_mask_steps": rec["outside_mask_steps"],
                    "outside_mask_len_2d": rec["outside_mask_len_2d"],
                    "outside_mask_len_3d": rec["outside_mask_len_3d"],
                    "outside_mask_fraction_2d": rec["outside_mask_fraction_2d"],
                    "outside_mask_fraction_3d": rec["outside_mask_fraction_3d"],
                    "max_consecutive_outside_steps": rec["max_consecutive_outside_steps"],
                    "tortuosity_2d": rec["tortuosity_2d"],
                    "qc_flag": rec["qc_flag"],
                    "qc_reason": rec["qc_reason"],
                }
            )
        )
        poly_geom = rec["polygon_geometry"]
        if poly_geom is None:
            continue
        polygon_rows.append(
            _add_runout_fields(
                {
                    "polygon_id": rec["polygon_id"],
                    "H_c": rec["H_c"],
                    "H_f": rec["H_f"],
                    "H": rec["H"],
                    "L": rec["L"],
                    "L_2d": rec["L_2d"],
                    "L_3d": rec["L_3d"],
                    "polygon_area_m2": rec["polygon_area_m2"],
                    "a_obs_deg": rec["a_obs_deg"],
                    "path_method": rec["path_method"],
                    "path_success": rec["path_success"],
                    "fallback_triggered": rec["fallback_triggered"],
                    "diagnostic_reason": rec["diagnostic_reason"],
                    "outside_mask_steps": rec["outside_mask_steps"],
                    "outside_mask_len_2d": rec["outside_mask_len_2d"],
                    "outside_mask_len_3d": rec["outside_mask_len_3d"],
                    "outside_mask_fraction_2d": rec["outside_mask_fraction_2d"],
                    "outside_mask_fraction_3d": rec["outside_mask_fraction_3d"],
                    "max_consecutive_outside_steps": rec["max_consecutive_outside_steps"],
                    "tortuosity_2d": rec["tortuosity_2d"],
                    "qc_flag": rec["qc_flag"],
                    "qc_reason": rec["qc_reason"],
                    "geometry": poly_geom,
                }
            )
        )
        if rec["crown_point"] is not None:
            crown_rows.append(
                _add_runout_fields(
                    {
                    "polygon_id": rec["polygon_id"],
                    "crown_id": rec["crown_id"],
                    "H_c": rec["H_c"],
                    "H_f": rec["H_f"],
                    "H": rec["H"],
                    "L": rec["L"],
                    "L_2d": rec["L_2d"],
                    "L_3d": rec["L_3d"],
                    "polygon_area_m2": rec["polygon_area_m2"],
                    "a_obs_deg": rec["a_obs_deg"],
                    "path_method": rec["path_method"],
                    "path_success": rec["path_success"],
                    "fallback_triggered": rec["fallback_triggered"],
                    "diagnostic_reason": rec["diagnostic_reason"],
                    "outside_mask_steps": rec["outside_mask_steps"],
                    "outside_mask_len_2d": rec["outside_mask_len_2d"],
                    "outside_mask_len_3d": rec["outside_mask_len_3d"],
                    "outside_mask_fraction_2d": rec["outside_mask_fraction_2d"],
                    "outside_mask_fraction_3d": rec["outside_mask_fraction_3d"],
                    "max_consecutive_outside_steps": rec["max_consecutive_outside_steps"],
                    "tortuosity_2d": rec["tortuosity_2d"],
                    "qc_flag": rec["qc_flag"],
                    "qc_reason": rec["qc_reason"],
                    "geometry": rec["crown_point"],
                    }
                )
            )
        if rec["toe_point"] is not None:
            toe_rows.append(
                {
                    "polygon_id": rec["polygon_id"],
                    "toe_id": rec["toe_id"],
                    "H_f": rec["H_f"],
                    "path_method": rec["path_method"],
                    "diagnostic_reason": rec["diagnostic_reason"],
                    "geometry": rec["toe_point"],
                }
            )
        if rec["runout_path"] is not None:
            runout_rows.append(
                {
                    "polygon_id": rec["polygon_id"],
                    "crown_id": rec["crown_id"],
                    "toe_id": rec["toe_id"],
                    "L_m": rec["L"],
                    "L_2d_m": rec["L_2d"],
                    "L_3d_m": rec["L_3d"],
                    "drop_m": rec["H"],
                    "path_method": rec["path_method"],
                    "fallback_triggered": rec["fallback_triggered"],
                    "diagnostic_reason": rec["diagnostic_reason"],
                    "outside_mask_steps": rec["outside_mask_steps"],
                    "outside_mask_len_2d": rec["outside_mask_len_2d"],
                    "outside_mask_len_3d": rec["outside_mask_len_3d"],
                    "outside_mask_fraction_2d": rec["outside_mask_fraction_2d"],
                    "outside_mask_fraction_3d": rec["outside_mask_fraction_3d"],
                    "max_consecutive_outside_steps": rec["max_consecutive_outside_steps"],
                    "tortuosity_2d": rec["tortuosity_2d"],
                    "notes": rec["diagnostic_reason"],
                    "geometry": rec["runout_path"],
                }
            )

    crown_clean_rows = [r for r in crown_rows if str(r.get("qc_flag", "BAD")).upper() == "OK"]
    polygon_clean_rows = [r for r in polygon_rows if str(r.get("qc_flag", "BAD")).upper() == "OK"]

    layer_specs: List[LayerSpec] = [
        LayerSpec(
            name=f"{p.output_prefix}_crown_points_complete",
            rows=crown_rows,
            columns=[
                "polygon_id", "crown_id", "H_c", "H_f", "H", "L", "L_2d", "L_3d", "h_over_l", "a_obs_deg", "path_method",
                "path_success", "fallback_triggered", "diagnostic_reason", "outside_mask_fraction_2d",
                "outside_mask_steps", "outside_mask_len_2d", "outside_mask_len_3d",
                "outside_mask_fraction_3d", "max_consecutive_outside_steps", "tortuosity_2d",
                "qc_flag", "qc_reason",
            ],
            geometry_type=ogr.wkbPoint,
        ),
        LayerSpec(
            name=f"{p.output_prefix}_crown_points_clean",
            rows=crown_clean_rows,
            columns=[
                "polygon_id", "crown_id", "H_c", "H_f", "H", "L", "L_2d", "L_3d", "h_over_l", "a_obs_deg", "path_method",
                "path_success", "fallback_triggered", "diagnostic_reason", "outside_mask_fraction_2d",
                "outside_mask_steps", "outside_mask_len_2d", "outside_mask_len_3d",
                "outside_mask_fraction_3d", "max_consecutive_outside_steps", "tortuosity_2d",
                "qc_flag", "qc_reason",
            ],
            geometry_type=ogr.wkbPoint,
        ),
        LayerSpec(
            name=f"{p.output_prefix}_landslide_polygons_clean",
            rows=polygon_clean_rows,
            columns=[
                "polygon_id", "H_c", "H_f", "H", "L", "L_2d", "L_3d", "h_over_l", "a_obs_deg",
                "polygon_area_m2", "path_method", "path_success", "fallback_triggered", "diagnostic_reason",
                "outside_mask_steps", "outside_mask_len_2d", "outside_mask_len_3d",
                "outside_mask_fraction_2d", "outside_mask_fraction_3d",
                "max_consecutive_outside_steps", "tortuosity_2d", "qc_flag", "qc_reason",
            ],
            geometry_type=ogr.wkbMultiPolygon,
        ),
    ]
    if p.write_optional_layers:
        layer_specs.extend(
            [
                LayerSpec(
                    name=f"{p.output_prefix}_toe_points_complete",
                    rows=toe_rows,
                    columns=["polygon_id", "toe_id", "H_f", "path_method", "diagnostic_reason"],
                    geometry_type=ogr.wkbPoint,
                ),
                LayerSpec(
                    name=f"{p.output_prefix}_runout_paths_complete",
                    rows=runout_rows,
                    columns=[
                        "polygon_id", "crown_id", "toe_id", "L_m", "L_2d_m", "L_3d_m", "drop_m", "path_method",
                        "fallback_triggered", "diagnostic_reason", "outside_mask_steps",
                        "outside_mask_len_2d", "outside_mask_len_3d", "outside_mask_fraction_2d",
                        "outside_mask_fraction_3d", "max_consecutive_outside_steps", "tortuosity_2d", "notes",
                    ],
                    geometry_type=ogr.wkbMultiLineString,
                ),
            ]
        )

    t0_write = timer_start()
    log("Writing output layers...")
    write_gpkg_layers(out_gpkg, layer_specs, crs_out)
    log_timing("output_writing", t0_write)
    if p.write_csv_summary:
        write_qc_csv(qc_csv, qc_rows)
        _write_runout_class_summary_csv(runout_class_summary_csv, crown_clean_rows)
    plt = _require_matplotlib()
    skewness_outputs = _build_skewness_output_paths(out_dir, p.output_prefix)
    t0_visuals = timer_start()
    _plot_metric_skewness(
        plt,
        [row.get("a_obs_deg") for row in crown_rows],
        metric_label="a_obs",
        x_label="a_obs (degrees)",
        subset_label="Complete",
        out_png=skewness_outputs["aobs_complete_skewness_png"],
    )
    _plot_metric_skewness(
        plt,
        [row.get("a_obs_deg") for row in crown_clean_rows],
        metric_label="a_obs",
        x_label="a_obs (degrees)",
        subset_label="Clean",
        out_png=skewness_outputs["aobs_clean_skewness_png"],
    )
    _plot_metric_skewness(
        plt,
        [row.get("H") for row in crown_rows],
        metric_label="H",
        x_label="H (m)",
        subset_label="Complete",
        out_png=skewness_outputs["h_complete_skewness_png"],
    )
    _plot_metric_skewness(
        plt,
        [row.get("H") for row in crown_clean_rows],
        metric_label="H",
        x_label="H (m)",
        subset_label="Clean",
        out_png=skewness_outputs["h_clean_skewness_png"],
    )
    _plot_metric_skewness(
        plt,
        [row.get("L") for row in crown_rows],
        metric_label="L",
        x_label="L (m, 3D path length)",
        subset_label="Complete",
        out_png=skewness_outputs["l_complete_skewness_png"],
    )
    _plot_metric_skewness(
        plt,
        [row.get("L") for row in crown_clean_rows],
        metric_label="L",
        x_label="L (m, 3D path length)",
        subset_label="Clean",
        out_png=skewness_outputs["l_clean_skewness_png"],
    )
    log_timing("distribution_skewness_visuals", t0_visuals)
    n_ok = sum(1 for rec in records if rec["qc_flag"] == "OK")
    n_bad = len(records) - n_ok
    log(f"Processed polygons: {len(records)}")
    log(f"QC OK: {n_ok}, QC BAD: {n_bad}")
    log_timing("total_runtime", stage_t0)

    outputs = {
        "output_gpkg": out_gpkg,
        "crown_points_complete": f"{out_gpkg}|layername={p.output_prefix}_crown_points_complete",
        "crown_points_clean": f"{out_gpkg}|layername={p.output_prefix}_crown_points_clean",
        "landslide_polygons_clean": f"{out_gpkg}|layername={p.output_prefix}_landslide_polygons_clean",
        **skewness_outputs,
    }
    if p.write_optional_layers:
        outputs.update(
            {
                "toe_points_complete": f"{out_gpkg}|layername={p.output_prefix}_toe_points_complete",
                "runout_paths_complete": f"{out_gpkg}|layername={p.output_prefix}_runout_paths_complete",
            }
        )
    if p.write_csv_summary:
        outputs["qc_summary_csv"] = qc_csv
        outputs["runout_class_summary_csv"] = runout_class_summary_csv
    return outputs


def run_step1_native(
    *,
    landslide_polygons: str,
    polygon_id_field: str,
    dem: str,
    output_gpkg: str,
    output_prefix: str,
    buffer_min_m: float,
    buffer_pixel_multiple: float,
    boundary_sample_step_m: float,
    crown_percentile_high: float,
    endpoint_snap_search_radius_px: int,
    qc_percentile_high: float,
    enable_early_polygon_filter: bool,
    complexity_threshold: float,
    early_exclusion_area_threshold_m2: float = 250.0,
    write_csv_summary: bool,
    write_optional_layers: bool,
    progress_every: int,
    log_level: str,
    enable_parallel: bool = True,
    max_workers: Optional[int] = 4,
    worker_start_method: str = "spawn",
    preserve_order: bool = True,
    fail_fast: bool = False,
    worker_chunk_size: int = 1,
) -> Dict[str, str]:
    params = Params(
        landslide_polygons=landslide_polygons,
        polygon_id_field=polygon_id_field,
        dem=dem,
        output_gpkg=output_gpkg,
        output_prefix=output_prefix,
        buffer_min_m=float(buffer_min_m),
        buffer_pixel_multiple=float(buffer_pixel_multiple),
        boundary_sample_step_m=float(boundary_sample_step_m),
        crown_percentile_high=float(crown_percentile_high),
        endpoint_snap_search_radius_px=int(endpoint_snap_search_radius_px),
        qc_percentile_high=float(qc_percentile_high),
        enable_early_polygon_filter=bool(enable_early_polygon_filter),
        complexity_threshold=float(complexity_threshold),
        early_exclusion_area_threshold_m2=float(early_exclusion_area_threshold_m2),
        write_csv_summary=bool(write_csv_summary),
        write_optional_layers=bool(write_optional_layers),
        progress_every=int(progress_every),
        log_level=str(log_level),
        enable_parallel=bool(enable_parallel),
        max_workers=None if max_workers is None else int(max_workers),
        worker_start_method=str(worker_start_method),
        preserve_order=bool(preserve_order),
        fail_fast=bool(fail_fast),
        worker_chunk_size=int(worker_chunk_size),
    )
    return dict(run_step1(params))
