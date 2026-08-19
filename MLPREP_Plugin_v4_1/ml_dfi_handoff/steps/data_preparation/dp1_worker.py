from __future__ import annotations

import atexit
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional, Sequence, cast

import numpy as np
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, Point, Polygon

from dp1_models import Step1Config, Step1CoreInput

from dp1_support import (
    ContourExtractionResult,
    GdalRasterDataset,
    detect_crown,
    detect_toe,
    densify_boundary_points,
    extract_contour_nodes_from_buffered_window,
    fmt_qc_reason,
    load_step1_core_engine,
    mask_polygon_to_window,
    open_dem_dataset,
    sample_dem_points,
    snap_point_to_mask,
)


def _geometry_to_wkt(geom: object) -> Optional[str]:
    if geom is None:
        return None
    geometry_wkt = getattr(geom, "wkt", None)
    if not geometry_wkt:
        return None
    return str(geometry_wkt)


_DEM_DATASETS: dict[str, GdalRasterDataset] = {}


def _get_worker_dem_dataset(dem_path: str) -> GdalRasterDataset:
    path = str(dem_path)
    dataset = _DEM_DATASETS.get(path)
    if dataset is None or dataset.closed:
        dataset = open_dem_dataset(path)
        _DEM_DATASETS[path] = dataset
    return dataset


def _close_worker_dem_datasets() -> None:
    for dataset in _DEM_DATASETS.values():
        dataset.close()
    _DEM_DATASETS.clear()


atexit.register(_close_worker_dem_datasets)


def _compact_traceback() -> str:
    return traceback.format_exc(limit=8).strip()


def _new_timing_map() -> dict[str, float]:
    return {}


@dataclass(frozen=True)
class Step1PolygonJob:
    order_index: int
    polygon_id: str
    crown_id: str
    toe_id: str
    analysis_polygon_wkt: str
    dem_path: str
    pixel_size_m: float
    core_config: Step1Config


@dataclass
class Step1PolygonWorkerResult:
    order_index: int
    polygon_id: str
    success: bool
    qc_reasons: tuple[str, ...] = ()
    diagnostic_reason: str = "OK"
    debug_traceback: str = ""
    unexpected_error: bool = False
    crown_point_wkt: Optional[str] = None
    toe_point_wkt: Optional[str] = None
    runout_path_wkt: Optional[str] = None
    h_c: Optional[float] = None
    h_f: Optional[float] = None
    h_m: Optional[float] = None
    l_m: Optional[float] = None
    l_2d_m: Optional[float] = None
    l_3d_m: Optional[float] = None
    a_obs_deg: Optional[float] = None
    path_method: str = ""
    path_success: bool = False
    fallback_triggered: bool = False
    buffer_dist_m: Optional[float] = None
    path_steps: Optional[int] = None
    outside_mask_steps: Optional[int] = None
    outside_mask_len_2d: Optional[float] = None
    outside_mask_len_3d: Optional[float] = None
    outside_mask_fraction_2d: Optional[float] = None
    outside_mask_fraction_3d: Optional[float] = None
    max_consecutive_outside_steps: Optional[int] = None
    tortuosity_2d: Optional[float] = None
    contour_node_count: int = 0
    worker_workspace: str = ""
    timings: dict[str, float] = field(default_factory=_new_timing_map)

    @classmethod
    def cancelled(cls, job: Step1PolygonJob) -> "Step1PolygonWorkerResult":
        return cls(
            order_index=int(job.order_index),
            polygon_id=str(job.polygon_id),
            success=False,
            qc_reasons=("worker_cancelled",),
            diagnostic_reason="worker_cancelled",
            unexpected_error=True,
        )


def _finish_result(
    result: Step1PolygonWorkerResult,
    total_t0: float,
) -> Step1PolygonWorkerResult:
    result.timings.setdefault("total_s", float(time.perf_counter() - total_t0))
    return result


def _failure_result(
    job: Step1PolygonJob,
    reasons: Sequence[str],
    *,
    diagnostic_reason: Optional[str] = None,
    debug_traceback: str = "",
    unexpected_error: bool = False,
    crown_point: Optional[Point] = None,
    toe_point: Optional[Point] = None,
    h_c: Optional[float] = None,
    h_f: Optional[float] = None,
    h_m: Optional[float] = None,
    buffer_dist_m: Optional[float] = None,
    worker_workspace: str = "",
    timings: Optional[dict[str, float]] = None,
) -> Step1PolygonWorkerResult:
    ordered = tuple(sorted(str(reason) for reason in reasons if str(reason).strip()))
    timing_values: dict[str, float] = {
        str(key): float(value)
        for key, value in dict(timings or {}).items()
    }
    return Step1PolygonWorkerResult(
        order_index=int(job.order_index),
        polygon_id=str(job.polygon_id),
        success=False,
        qc_reasons=ordered,
        diagnostic_reason=str(diagnostic_reason or fmt_qc_reason(set(ordered))),
        debug_traceback=str(debug_traceback),
        unexpected_error=bool(unexpected_error),
        crown_point_wkt=_geometry_to_wkt(crown_point),
        toe_point_wkt=_geometry_to_wkt(toe_point),
        h_c=h_c,
        h_f=h_f,
        h_m=h_m,
        buffer_dist_m=buffer_dist_m,
        worker_workspace=str(worker_workspace),
        timings=timing_values,
    )


def execute_step1_polygon_job(job: Step1PolygonJob) -> Step1PolygonWorkerResult:
    total_t0 = time.perf_counter()
    timings: dict[str, float] = {}
    worker_workspace = ""
    timings["workspace_setup_s"] = float(time.perf_counter() - total_t0)

    crown_point: Optional[Point] = None
    toe_point: Optional[Point] = None
    h_c: Optional[float] = None
    h_f: Optional[float] = None
    buffer_dist: Optional[float] = None

    try:
        polygon = cast(Polygon, shapely_wkt.loads(job.analysis_polygon_wkt))
        core = load_step1_core_engine()

        with nullcontext(_get_worker_dem_dataset(job.dem_path)) as dem_ds:
            t0_buffer = time.perf_counter()
            buffer_dist = max(
                float(job.core_config.buffer_min_m),
                float(job.core_config.buffer_pixel_multiple) * float(job.pixel_size_m),
            )
            buffer_poly = polygon.buffer(float(buffer_dist))
            timings["buffer_creation_s"] = float(time.perf_counter() - t0_buffer)

            t0_window = time.perf_counter()
            window = dem_ds.read_window(buffer_poly.bounds)
            if window is None:
                return _finish_result(
                    _failure_result(
                        job,
                        ("local_dem_empty",),
                        h_c=h_c,
                        h_f=h_f,
                        buffer_dist_m=buffer_dist,
                        worker_workspace=worker_workspace,
                        timings=timings,
                    ),
                    total_t0,
                )

            buffer_mask = np.asarray(mask_polygon_to_window(buffer_poly, window) & window.valid_mask, dtype=bool)
            landslide_mask = np.asarray(mask_polygon_to_window(polygon, window) & window.valid_mask, dtype=bool)
            timings["window_extraction_s"] = float(time.perf_counter() - t0_window)
            if not np.any(landslide_mask):
                return _finish_result(
                    _failure_result(
                        job,
                        ("no_local_valid_cells",),
                        h_c=h_c,
                        h_f=h_f,
                        buffer_dist_m=buffer_dist,
                        worker_workspace=worker_workspace,
                        timings=timings,
                    ),
                    total_t0,
                )

            t0_endpoints = time.perf_counter()
            boundary_pts = densify_boundary_points(polygon, float(job.core_config.boundary_sample_step_m))
            if not boundary_pts:
                return _finish_result(
                    _failure_result(
                        job,
                        ("boundary_sampling_failed",),
                        h_c=h_c,
                        h_f=h_f,
                        buffer_dist_m=buffer_dist,
                        worker_workspace=worker_workspace,
                        timings=timings,
                    ),
                    total_t0,
                )
            boundary_samples = sample_dem_points(dem_ds, boundary_pts, dem_ds.nodata)
            toe_pt, h_f, toe_status = detect_toe(boundary_samples)
            crown_pt, h_c, crown_status = detect_crown(
                boundary_samples,
                float(job.core_config.crown_percentile_high),
            )
            timings["endpoint_derivation_s"] = float(time.perf_counter() - t0_endpoints)

            reasons: set[str] = set()
            if toe_status != "ok":
                reasons.add(str(toe_status))
            if crown_status != "ok":
                reasons.add(str(crown_status))
            if h_f is None or h_c is None:
                reasons.add("nodata")
            if toe_pt is None:
                reasons.add("toe_not_found")
            if crown_pt is None:
                reasons.add("crown_not_found")

            if crown_pt is not None:
                crown_point = Point(float(crown_pt.x), float(crown_pt.y))
            if toe_pt is not None:
                toe_point = Point(float(toe_pt.x), float(toe_pt.y))

            if crown_point is None or toe_point is None:
                return _finish_result(
                    _failure_result(
                        job,
                        tuple(reasons),
                        h_c=h_c,
                        h_f=h_f,
                        h_m=(float(h_c) - float(h_f)) if h_c is not None and h_f is not None else None,
                        crown_point=crown_point,
                        toe_point=toe_point,
                        buffer_dist_m=buffer_dist,
                        worker_workspace=worker_workspace,
                        timings=timings,
                    ),
                    total_t0,
                )

            crown_cell = snap_point_to_mask(
                crown_point,
                window,
                landslide_mask,
                int(job.core_config.endpoint_snap_search_radius_px),
            )
            toe_cell = snap_point_to_mask(
                toe_point,
                window,
                landslide_mask,
                int(job.core_config.endpoint_snap_search_radius_px),
            )
            if crown_cell is None or toe_cell is None:
                reasons.add("endpoint_snap_failed")
                return _finish_result(
                    _failure_result(
                        job,
                        tuple(reasons),
                        h_c=h_c,
                        h_f=h_f,
                        h_m=(float(h_c) - float(h_f)) if h_c is not None and h_f is not None else None,
                        crown_point=crown_point,
                        toe_point=toe_point,
                        buffer_dist_m=buffer_dist,
                        worker_workspace=worker_workspace,
                        timings=timings,
                    ),
                    total_t0,
                )

            contour_result: ContourExtractionResult = extract_contour_nodes_from_buffered_window(
                window=window,
                buffer_mask=buffer_mask,
                landslide_polygon=polygon,
                polygon_id=str(job.polygon_id),
                contour_interval_m=float(job.core_config.contour_interval_m),
            )
            timings["contour_generation_s"] = float(contour_result.contour_generation_s)
            timings["node_extraction_s"] = float(contour_result.node_extraction_s)
            timings["contour_pipeline_s"] = float(contour_result.total_s)

            t0_path = time.perf_counter()
            crown_cell_xy = (int(crown_cell[0]), int(crown_cell[1]))
            toe_cell_xy = (int(toe_cell[0]), int(toe_cell[1]))
            core_input = Step1CoreInput(
                config=job.core_config,
                contour_nodes=tuple(contour_result.nodes),
                crown_xy=(float(crown_point.x), float(crown_point.y)),
                toe_xy=(float(toe_point.x), float(toe_point.y)),
                crown_elev=float(h_c) if h_c is not None else None,
                toe_elev=float(h_f) if h_f is not None else None,
                pixel_size_m=float(job.pixel_size_m),
                dem_array=np.asarray(window.array, dtype=np.float64),
                valid_mask=np.asarray(window.valid_mask, dtype=bool),
                landslide_mask=np.asarray(landslide_mask, dtype=bool),
                origin_x=float(window.origin_x),
                origin_y=float(window.origin_y),
                pixel_width=float(window.pixel_width),
                pixel_height=float(window.pixel_height),
                crown_cell=crown_cell_xy,
                toe_cell=toe_cell_xy,
                landslide_polygon=polygon,
                metadata={"polygon_id": str(job.polygon_id)},
            )
            core_result = core.run_step1_core(core_input)
            timings["path_generation_s"] = float(time.perf_counter() - t0_path)

            polygon_result = core_result.polygon_results[0] if core_result.polygon_results else None
            path_result = core_result.path_result
            if polygon_result is not None:
                reasons.update(str(reason) for reason in polygon_result.qc_reasons if str(reason).strip())

            if path_result is None:
                reasons.add("no_path_result")
                return _finish_result(
                    _failure_result(
                        job,
                        tuple(reasons),
                        h_c=h_c,
                        h_f=h_f,
                        h_m=(float(h_c) - float(h_f)) if h_c is not None and h_f is not None else None,
                        crown_point=crown_point,
                        toe_point=toe_point,
                        buffer_dist_m=buffer_dist,
                        worker_workspace=worker_workspace,
                        timings=timings,
                    ),
                    total_t0,
                )

            runout_path = None
            if bool(path_result.success) and len(path_result.path_xy) >= 2:
                runout_path = LineString(path_result.path_xy)

            return _finish_result(
                Step1PolygonWorkerResult(
                    order_index=int(job.order_index),
                    polygon_id=str(job.polygon_id),
                    success=bool(path_result.success),
                    qc_reasons=tuple(sorted(reasons)),
                    diagnostic_reason=str(path_result.diagnostic_reason or "OK"),
                    unexpected_error=False,
                    crown_point_wkt=_geometry_to_wkt(crown_point),
                    toe_point_wkt=_geometry_to_wkt(toe_point),
                    runout_path_wkt=_geometry_to_wkt(runout_path),
                    h_c=h_c,
                    h_f=h_f,
                    h_m=polygon_result.h_m if polygon_result is not None else None,
                    l_m=polygon_result.l_m if polygon_result is not None else path_result.l_3d,
                    l_2d_m=path_result.l_2d,
                    l_3d_m=path_result.l_3d,
                    a_obs_deg=polygon_result.a_obs_deg if polygon_result is not None else None,
                    path_method=str(path_result.method_used),
                    path_success=bool(path_result.success),
                    fallback_triggered=bool(path_result.fallback_triggered),
                    buffer_dist_m=buffer_dist,
                    path_steps=path_result.path_steps,
                    outside_mask_steps=path_result.outside_mask_steps,
                    outside_mask_len_2d=path_result.outside_mask_len_2d,
                    outside_mask_len_3d=path_result.outside_mask_len_3d,
                    outside_mask_fraction_2d=path_result.outside_mask_fraction_2d,
                    outside_mask_fraction_3d=path_result.outside_mask_fraction_3d,
                    max_consecutive_outside_steps=path_result.max_consecutive_outside_steps,
                    tortuosity_2d=path_result.tortuosity_2d,
                    contour_node_count=int(len(contour_result.nodes)),
                    worker_workspace=worker_workspace,
                    timings=timings,
                ),
                total_t0,
            )
    except Exception:
        return _finish_result(
            _failure_result(
                job,
                ("worker_exception",),
                diagnostic_reason="worker_exception",
                debug_traceback=_compact_traceback(),
                unexpected_error=True,
                crown_point=crown_point,
                toe_point=toe_point,
                h_c=h_c,
                h_f=h_f,
                h_m=(float(h_c) - float(h_f)) if h_c is not None and h_f is not None else None,
                buffer_dist_m=buffer_dist,
                worker_workspace=worker_workspace,
                timings=timings,
            ),
            total_t0,
        )
