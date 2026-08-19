from __future__ import annotations

import argparse
import importlib
import json
import os
import site
import sys
import warnings
from pathlib import Path
from typing import Any, cast


EXPECTED_RUNTIME_VERSIONS = {
    "python": "3.12.10",
    "qgis": "3.40.7-Bratislava",
    "gdal": "3.10.3",
    "proj": "9.6.0",
    "numpy": "2.5.1",
    "shapely": "2.1.2",
    "rasterio": "1.5.0",
    "pyproj": "3.7.2",
    "matplotlib": "3.10.9",
}


def _resolve_path(root: Path, value: str) -> str:
    raw = str(value).strip()
    if "|" in raw:
        base, suffix = raw.split("|", 1)
        qgis_suffix = f"|{suffix}"
    else:
        base, qgis_suffix = raw, ""
    path = Path(base).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    return f"{resolved}{qgis_suffix}"


def load_run_parameters(config_path: str) -> dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8-sig") as handle:
        raw_payload: Any = json.load(handle)
    if not isinstance(raw_payload, dict):
        raise RuntimeError(f"Config root must be an object: {path}")
    payload = {str(key): value for key, value in cast(dict[object, Any], raw_payload).items()}
    step_cfg = dict(payload.get("dp1", payload.get("step1", {})))
    parallel_cfg = dict(step_cfg.get("parallel", {}))
    max_workers_value = parallel_cfg.get("max_workers", 4)

    output_dir = _resolve_path(path.parent, str(step_cfg["output_dir"]))
    os.makedirs(output_dir, exist_ok=True)
    if "contour_interval" in step_cfg:
        warnings.warn(
            "dp1.contour_interval is deprecated and ignored; DP1 always uses fixed local 2.0 m contours.",
            DeprecationWarning,
            stacklevel=2,
        )
    legacy_keys = (
        "path_algorithm", "dinf_flow_raster", "taudem_pitremove_exe",
        "taudem_dinf_flow_dir_exe", "mpiexec_exe", "mpiexec_processes",
        "toe_reach_tolerance_px", "buffer_escape_mode", "flat_tolerance_m",
        "max_consecutive_flat_steps", "max_divergence_factor",
        "stagnation_window_steps", "max_steps_per_initial_distance_px",
        "max_outside_consecutive_steps", "max_outside_fraction_2d_for_steepest",
        "outside_mask_excess_fraction", "steepest_w_drop", "steepest_w_toe",
        "steepest_w_inside", "steepest_w_turn", "least_cost_outside_penalty",
        "least_cost_uphill_penalty_per_m", "least_cost_away_from_toe_penalty",
        "least_cost_turn_penalty",
    )
    ignored = [key for key in legacy_keys if key in step_cfg]
    if ignored:
        warnings.warn(
            "Deprecated DP1 options are ignored: " + ", ".join(ignored),
            DeprecationWarning,
            stacklevel=2,
        )

    return {
        "landslide_polygons": _resolve_path(path.parent, str(step_cfg["landslide_polygons"])),
        "polygon_id_field": str(step_cfg.get("id_field", "OBJECTID")),
        "dem": _resolve_path(path.parent, str(step_cfg["dem"])),
        "output_gpkg": os.path.join(output_dir, "dp1.gpkg"),
        "output_prefix": str(step_cfg.get("output_prefix", "dp1")),
        "buffer_min_m": float(step_cfg.get("buffer_min_m", 10.0)),
        "buffer_pixel_multiple": float(step_cfg.get("buffer_pixel_multiple", 2.0)),
        "boundary_sample_step_m": float(step_cfg.get("boundary_sample_step_m", 10.0)),
        "crown_percentile_high": float(step_cfg.get("crown_percentile_high", 90.0)),
        "endpoint_snap_search_radius_px": int(step_cfg.get("endpoint_snap_search_radius_px", 4)),
        "qc_percentile_high": float(step_cfg.get("qc_percentile_high", 90.0)),
        "enable_early_polygon_filter": bool(
            step_cfg.get("enable_early_polygon_filter", step_cfg.get("enable_complexity_filter", True))
        ),
        "complexity_threshold": float(step_cfg.get("complexity_threshold", 1.0)),
        "early_exclusion_area_threshold_m2": float(
            step_cfg.get(
                "early_exclusion_area_threshold_m2",
                step_cfg.get("small_complex_polygon_area_threshold_m2", 250.0),
            )
        ),
        "write_csv_summary": bool(step_cfg.get("write_csv_summary", True)),
        "write_optional_layers": bool(step_cfg.get("write_optional_layers", True)),
        "progress_every": int(step_cfg.get("progress_every", 100)),
        "log_level": str(step_cfg.get("log_level", "INFO")),
        "enable_parallel": bool(parallel_cfg.get("enable_parallel", True)),
        "max_workers": 4 if max_workers_value is None else int(max_workers_value),
        "worker_start_method": str(parallel_cfg.get("worker_start_method", "spawn")),
        "preserve_order": bool(parallel_cfg.get("preserve_order", True)),
        "fail_fast": bool(parallel_cfg.get("fail_fast", False)),
        "worker_chunk_size": int(parallel_cfg.get("worker_chunk_size", 1)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DP1 observed-alpha data preparation in a QGIS runtime."
    )
    parser.add_argument("--config", help="Path to the DP1 JSON config.")
    parser.add_argument(
        "--check-environment",
        action="store_true",
        help="Validate and report the frozen QGIS/OSGeo4W runtime without running DP1.",
    )
    return parser


def _disable_user_site_packages() -> None:
    user_site_values = site.getusersitepackages()
    user_sites = [user_site_values] if isinstance(user_site_values, str) else list(user_site_values)
    normalized_user_sites = {
        os.path.normcase(os.path.abspath(value))
        for value in user_sites
        if str(value).strip()
    }

    def is_user_site(entry: str) -> bool:
        normalized_entry = os.path.normcase(os.path.abspath(entry or os.curdir))
        return any(
            normalized_entry == user_site
            or normalized_entry.startswith(user_site + os.sep)
            for user_site in normalized_user_sites
        )

    sys.path[:] = [entry for entry in sys.path if not is_user_site(entry)]
    os.environ["PYTHONNOUSERSITE"] = "1"


def _require_dedicated_interpreter(step_root: Path) -> None:
    expected_prefix = os.path.normcase(
        os.path.abspath(step_root.parents[1] / ".venv-dp1")
    )
    actual_prefix = os.path.normcase(os.path.abspath(sys.prefix))
    if actual_prefix != expected_prefix:
        raise RuntimeError(
            "DP1 must be launched through steps\\data_preparation\\launch_dp1.py. "
            f"Expected environment {expected_prefix}, received {actual_prefix}."
        )


def main() -> int:
    step_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(step_root))
    _require_dedicated_interpreter(step_root)
    _disable_user_site_packages()
    parser = build_parser()
    args = parser.parse_args()
    from dp1_native import run_step1_native
    from dp1_support import ensure_qgis_runtime_only

    if args.check_environment:
        import matplotlib
        import numpy
        import pyproj
        import rasterio
        import shapely
        from shapely import wkt as shapely_wkt
        from shapely.geometry import box
        from osgeo import gdal, osr
        qgis_core = importlib.import_module("qgis.core")

        ensure_qgis_runtime_only()
        smoke_geometry = box(0.0, 0.0, 10.0, 10.0)
        smoke_wkt = smoke_geometry.wkt
        if not shapely_wkt.loads(smoke_wkt).equals(smoke_geometry):
            raise RuntimeError("DP1 Shapely geometry/WKT smoke check failed.")
        versions = {
            "python": sys.version.split()[0],
            "qgis": qgis_core.Qgis.QGIS_VERSION,
            "gdal": gdal.VersionInfo("RELEASE_NAME"),
            "proj": ".".join(
                str(value)
                for value in (
                    osr.GetPROJVersionMajor(),
                    osr.GetPROJVersionMinor(),
                    osr.GetPROJVersionMicro(),
                )
            ),
            "numpy": numpy.__version__,
            "shapely": shapely.__version__,
            "rasterio": rasterio.__version__,
            "pyproj": pyproj.__version__,
            "matplotlib": matplotlib.__version__,
        }
        mismatches = {
            name: {"expected": expected, "actual": versions[name]}
            for name, expected in EXPECTED_RUNTIME_VERSIONS.items()
            if versions[name] != expected
        }
        if mismatches:
            details = "; ".join(
                f"{name}: expected {values['expected']}, found {values['actual']}"
                for name, values in mismatches.items()
            )
            raise RuntimeError(
                "DP1 native environment does not match the frozen contract. "
                + details
            )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "environment": ".venv-dp1",
                    "executable": sys.executable,
                    **versions,
                    "geometry_smoke_check": "passed",
                },
                indent=2,
            )
        )
        return 0

    if not args.config:
        parser.error("--config is required unless --check-environment is used.")

    raw = run_step1_native(**load_run_parameters(args.config))
    result = {"step": "dp1", "outputs": dict(raw), "metrics": {}, "warnings": []}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
