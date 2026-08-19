import argparse
import atexit
import gc
import json
import logging
import os
import shutil
import sys
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def configure_geospatial_environment() -> dict[str, str]:
    """Prefer the active environment's bundled GDAL/PROJ data over external installs."""
    site_roots = [
        Path(sys.prefix) / "Lib" / "site-packages",
        Path(sys.base_prefix) / "Lib" / "site-packages",
    ]
    proj_candidates = [
        "rasterio/proj_data",
        "pyogrio/proj_data",
        "fiona/proj_data",
        "pyproj/proj_dir/share/proj",
    ]
    gdal_candidates = [
        "rasterio/gdal_data",
        "fiona/gdal_data",
    ]

    configured: dict[str, str] = {}

    for root in site_roots:
        for relative in proj_candidates:
            candidate = root / relative
            if candidate.exists():
                candidate_text = str(candidate)
                os.environ["PROJ_LIB"] = candidate_text
                os.environ["PROJ_DATA"] = candidate_text
                configured["PROJ"] = candidate_text
                break
        if "PROJ" in configured:
            break

    for root in site_roots:
        for relative in gdal_candidates:
            candidate = root / relative
            if candidate.exists():
                candidate_text = str(candidate)
                os.environ["GDAL_DATA"] = candidate_text
                configured["GDAL"] = candidate_text
                break
        if "GDAL" in configured:
            break

    return configured


CONFIGURED_GEOSPATIAL_DIRS = configure_geospatial_environment()

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds, transform as window_transform

try:
    from shapely import from_wkb, make_valid
except ImportError:
    make_valid = None
    from shapely.wkb import loads as from_wkb


CONFIG = {
    "landslide_polygon_path": r"",
    "dem_path": r"",
    "output_dir": r"",
    "landslide_id_field": "",
    "depositional_percentile": 15,
    "non_depositional_percentile": 30,
    "nodata_value": -9999,
    "output_pixel_type": "int16",
    "min_valid_dem_cells": 5,
    "all_touched": False,
    "parallel_processing": True,
    "max_workers": 4,
}


LOG_NAME = "step1_prepare_inventory_labels"
MASK_OUTPUTS = {
    "depositional": "depositional_zone_raster.tif",
    "non_depositional": "non_depositional_zone_raster.tif",
    "landslide_id": "landslide_id_raster.tif",
    "label": "ml_dfi_label_raster.tif",
    "summary": "label_summary.csv",
    "run_config": "run_config.json",
    "log": "processing_log.txt",
}
STALE_SIDECAR_SUFFIXES = [".ovr", ".aux.xml"]
PATH_CONFIG_KEYS = ("landslide_polygon_path", "dem_path", "output_dir")
_WORKER_DEM_DATASET: Any = None
_WORKER_DEM_REFERENCE: dict[str, Any] | None = None
_WORKER_CLASSIFICATION_SETTINGS: dict[str, Any] | None = None


def setup_logging(output_dir: Path) -> tuple[logging.Logger, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / MASK_OUTPUTS["log"]
    logger = logging.getLogger(LOG_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger, log_path


def remove_stale_file(path: Path, logger: logging.Logger) -> None:
    if not path.exists():
        return
    if path.is_dir():
        logger.warning("Skipping stale output cleanup for directory: %s", path)
        return
    path.unlink()
    logger.info("Removed stale output from earlier Step 1 workflow: %s", path)


def cleanup_output_sidecars(output_paths: dict[str, Path], logger: logging.Logger) -> None:
    for output_path in output_paths.values():
        if output_path.suffix.lower() == ".tif":
            for sidecar_suffix in STALE_SIDECAR_SUFFIXES:
                remove_stale_file(Path(f"{output_path}{sidecar_suffix}"), logger)


def close_file_log_handlers(logger: logging.Logger) -> None:
    """Close file handlers before moving a staged log on Windows."""
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)


def publish_output_bundle(
    staged_paths: dict[str, Path],
    output_dir: Path,
    logger: logging.Logger,
) -> dict[str, Path]:
    """Publish all validated files and restore the previous bundle if publication fails."""
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(tempfile.mkdtemp(prefix=".step1-backup-", dir=output_dir.parent))
    final_paths = {key: output_dir / path.name for key, path in staged_paths.items()}
    backed_up: dict[str, Path] = {}
    published: set[str] = set()

    try:
        for key, final_path in final_paths.items():
            if final_path.exists():
                backup_path = backup_dir / final_path.name
                os.replace(final_path, backup_path)
                backed_up[key] = backup_path

        for key, staged_path in staged_paths.items():
            if not staged_path.exists():
                raise FileNotFoundError(f"Validated staged output is missing: {staged_path}")
            os.replace(staged_path, final_paths[key])
            published.add(key)
    except Exception as publish_error:
        rollback_errors: list[str] = []
        for key in published:
            final_path = final_paths[key]
            if key not in backed_up and final_path.exists():
                try:
                    final_path.unlink()
                except OSError as exc:
                    rollback_errors.append(f"remove new {final_path}: {exc}")
        for key, backup_path in backed_up.items():
            if backup_path.exists():
                try:
                    os.replace(backup_path, final_paths[key])
                except OSError as exc:
                    rollback_errors.append(f"restore {final_paths[key]}: {exc}")
        if rollback_errors:
            raise RuntimeError(
                "Step 1 publication failed and rollback was incomplete. Previous outputs are "
                f"retained in {backup_dir}. Problems: {'; '.join(rollback_errors)}"
            ) from publish_error
        shutil.rmtree(backup_dir, ignore_errors=True)
        raise
    else:
        shutil.rmtree(backup_dir, ignore_errors=True)

    try:
        cleanup_output_sidecars(final_paths, logger)
    except OSError as exc:
        logger.warning("Could not remove one or more stale raster sidecars: %s", exc)
    return final_paths


def split_vector_layer_spec(path_text: str) -> tuple[Path, str | None]:
    if "|layername=" not in path_text:
        return Path(path_text), None
    file_part, layer_name = path_text.split("|layername=", 1)
    layer_name = layer_name.strip()
    if not layer_name:
        raise ValueError(f"Vector layer suffix is missing a layer name: {path_text}")
    return Path(file_part), layer_name


def resolve_config_path(path_text: str, config_dir: Path, *, vector: bool = False) -> str:
    """Resolve a config path without losing a GeoPackage layer suffix."""
    if vector:
        path, layer_name = split_vector_layer_spec(path_text)
    else:
        path = Path(path_text)
        layer_name = None

    path = path.expanduser()
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    else:
        path = path.resolve()

    resolved = str(path)
    if layer_name is not None:
        resolved = f"{resolved}|layername={layer_name}"
    return resolved


def load_json_config(config_path: Path) -> dict[str, Any]:
    """Load either a plain Step 1 object or a named Step 1 section."""
    with config_path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        raise ValueError("Step 1 JSON config must contain an object.")

    for section_name in ("prepare_inventory_labels", "step1"):
        section = loaded.get(section_name)
        if section is not None:
            if not isinstance(section, dict):
                raise ValueError(f"JSON config section '{section_name}' must contain an object.")
            loaded = section
            break

    unknown = sorted(set(loaded) - set(CONFIG))
    if unknown:
        raise ValueError(f"Unknown Step 1 config keys: {', '.join(unknown)}")

    resolved = dict(loaded)
    config_dir = config_path.parent.resolve()
    for key in PATH_CONFIG_KEYS:
        value = resolved.get(key)
        if value is None or not str(value).strip():
            continue
        resolved[key] = resolve_config_path(
            str(value).strip(),
            config_dir,
            vector=(key == "landslide_polygon_path"),
        )
    return resolved


def output_dtype_name(config: dict[str, Any]) -> str:
    dtype_name = np.dtype(str(config["output_pixel_type"])).name
    if not np.issubdtype(np.dtype(dtype_name), np.integer):
        raise ValueError("output_pixel_type must be an integer dtype such as int16 or int32.")
    nodata_value = int(config["nodata_value"])
    dtype_info = np.iinfo(np.dtype(dtype_name))
    if nodata_value < int(dtype_info.min) or nodata_value > int(dtype_info.max):
        raise ValueError(
            f"nodata_value {nodata_value} cannot be represented by output_pixel_type {dtype_name}."
        )
    return dtype_name


def validate_inputs(config: dict[str, Any]) -> None:
    landslide_path_text = str(config["landslide_polygon_path"]).strip()
    dem_path_text = str(config["dem_path"]).strip()
    output_dir_text = str(config["output_dir"]).strip()

    if not landslide_path_text:
        raise ValueError("CONFIG['landslide_polygon_path'] is required.")
    if not dem_path_text:
        raise ValueError("CONFIG['dem_path'] is required.")
    if not output_dir_text:
        raise ValueError("CONFIG['output_dir'] is required.")

    landslide_path, _ = split_vector_layer_spec(landslide_path_text)
    if not landslide_path.exists():
        raise FileNotFoundError(f"Landslide polygon file does not exist: {landslide_path}")
    if not Path(dem_path_text).exists():
        raise FileNotFoundError(f"DEM raster does not exist: {dem_path_text}")
    output_dir = Path(output_dir_text)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {output_dir}")

    percentile = float(config["depositional_percentile"])
    non_depositional_percentile = float(config["non_depositional_percentile"])
    if percentile <= 0.0 or percentile >= 100.0:
        raise ValueError("depositional_percentile must be > 0 and < 100.")
    if non_depositional_percentile <= 0.0 or non_depositional_percentile >= 100.0:
        raise ValueError("non_depositional_percentile must be > 0 and < 100.")
    if percentile + non_depositional_percentile >= 100.0:
        raise ValueError(
            "depositional_percentile + non_depositional_percentile must be < 100 "
            "so an uncertain middle zone remains excluded from training."
        )

    min_valid_dem_cells = int(config["min_valid_dem_cells"])
    if min_valid_dem_cells < 1:
        raise ValueError("min_valid_dem_cells must be at least 1.")
    if not isinstance(config["parallel_processing"], bool):
        raise ValueError("parallel_processing must be true or false.")
    max_workers = int(config["max_workers"])
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")

    output_dtype_name(config)


def repair_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    repaired = gdf.copy()

    def repair_geometry(geometry: Any) -> Any:
        if geometry is None or geometry.is_empty or geometry.is_valid:
            return geometry
        if make_valid is not None:
            return make_valid(geometry)
        return geometry.buffer(0)

    repaired["geometry"] = repaired.geometry.apply(repair_geometry)
    repaired = repaired[repaired.geometry.notna()].copy()
    repaired = repaired[~repaired.geometry.is_empty].copy()
    if repaired.empty:
        raise ValueError("Vector layer has no valid non-empty geometries after repair.")
    invalid_after = int((~repaired.geometry.is_valid).sum())
    if invalid_after:
        raise ValueError("Vector layer still contains invalid geometries after repair.")
    return repaired


def read_vector_layer(path_text: str) -> gpd.GeoDataFrame:
    vector_path, layer_name = split_vector_layer_spec(path_text)
    read_kwargs = {"layer": layer_name} if layer_name else {}
    gdf = gpd.read_file(vector_path, **read_kwargs)
    if gdf.empty:
        raise ValueError(f"Vector layer is empty: {path_text}")
    if "geometry" not in gdf.columns:
        raise ValueError(f"Vector layer has no geometry column: {path_text}")
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError(f"Vector layer has no non-empty geometries: {path_text}")
    gdf = repair_geometries(gdf)
    geometry_types = set(gdf.geometry.geom_type.unique().tolist())
    unexpected_types = sorted(geometry_types - {"Polygon", "MultiPolygon"})
    if unexpected_types:
        raise ValueError(
            "Landslide inventory must contain only Polygon or MultiPolygon geometry; "
            f"found: {', '.join(unexpected_types)}"
        )
    return gdf


def attach_landslide_ids(
    gdf: gpd.GeoDataFrame,
    landslide_id_field: str,
    logger: logging.Logger,
) -> tuple[gpd.GeoDataFrame, str]:
    gdf = gdf.copy()
    gdf["_feature_index"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
    if landslide_id_field and landslide_id_field in gdf.columns:
        assigned_ids: list[str] = []
        for feature_index, raw_value in zip(gdf["_feature_index"], gdf[landslide_id_field], strict=False):
            text_value = "" if pd.isna(raw_value) else str(raw_value).strip()
            assigned_ids.append(text_value or f"LS_{int(feature_index):06d}")
        duplicate_ids = pd.Series(assigned_ids, dtype="string").duplicated(keep=False)
        if bool(duplicate_ids.any()):
            examples = sorted(set(pd.Series(assigned_ids, dtype="string")[duplicate_ids].tolist()))[:5]
            raise ValueError(
                f"Configured landslide_id_field '{landslide_id_field}' contains duplicate or "
                f"fallback-colliding IDs, including: {', '.join(examples)}"
            )
        gdf["_landslide_id"] = assigned_ids
        id_source = landslide_id_field
    else:
        if landslide_id_field:
            logger.warning(
                "Configured landslide_id_field '%s' was not found. Internal IDs will be created.",
                landslide_id_field,
            )
        gdf["_landslide_id"] = [f"LS_{idx:06d}" for idx in gdf["_feature_index"]]
        id_source = "internal_id"
    return gdf, id_source


def load_and_reproject_vectors(
    landslide_polygon_path: str,
    dem_crs: CRS,
    landslide_id_field: str,
    logger: logging.Logger,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    landslides = read_vector_layer(landslide_polygon_path)
    if landslides.crs is None:
        raise ValueError("Landslide polygon layer has no CRS.")
    landslide_reprojected = bool(landslides.crs != dem_crs)
    if landslide_reprojected:
        logger.info("Reprojecting landslide polygons from %s to %s.", landslides.crs, dem_crs)
        landslides = landslides.to_crs(dem_crs)
    landslides, id_source = attach_landslide_ids(landslides, landslide_id_field, logger)
    multipart_part_counts = landslides.geometry.apply(
        lambda geometry: len(geometry.geoms) if geometry.geom_type == "MultiPolygon" else 1
    )
    multipart_feature_count = int((multipart_part_counts > 1).sum())
    if multipart_feature_count:
        logger.warning(
            "%d landslide features contain multiple polygon parts; all parts of each feature "
            "will share one elevation distribution and one landslide group ID.",
            multipart_feature_count,
        )

    diagnostics = {
        "landslide_feature_count": int(len(landslides)),
        "landslide_id_source": id_source,
        "landslides_reprojected_to_dem_crs": landslide_reprojected,
        "multipart_feature_count": multipart_feature_count,
    }
    return landslides, diagnostics


def clamped_geometry_window(bounds: tuple[float, float, float, float], reference: dict[str, Any]) -> Window | None:
    raw_window = from_bounds(*bounds, transform=reference["transform"])
    col_off = max(0, int(np.floor(raw_window.col_off)))
    row_off = max(0, int(np.floor(raw_window.row_off)))
    col_stop = min(reference["width"], int(np.ceil(raw_window.col_off + raw_window.width)))
    row_stop = min(reference["height"], int(np.ceil(raw_window.row_off + raw_window.height)))
    if col_stop <= col_off or row_stop <= row_off:
        return None
    return Window(col_off, row_off, col_stop - col_off, row_stop - row_off)


def allocate_grid_array(
    shape: tuple[int, int],
    dtype: str | np.dtype[Any],
    fill_value: int | bool,
    name: str,
    working_dir: Path | None,
) -> np.ndarray:
    """Allocate in RAM for tests or as a disk-backed array for production runs."""
    if working_dir is None:
        return np.full(shape, fill_value, dtype=dtype)
    working_dir.mkdir(parents=True, exist_ok=True)
    array = np.memmap(working_dir / f".{name}.memmap", mode="w+", dtype=dtype, shape=shape)
    array[...] = fill_value
    array.flush()
    return array


def close_grid_array(array: np.ndarray | None) -> None:
    if not isinstance(array, np.memmap):
        return
    array.flush()
    mmap_handle = getattr(array, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()


def _dataset_grid_reference(dataset: Any) -> dict[str, Any]:
    return {
        "transform": dataset.transform,
        "width": int(dataset.width),
        "height": int(dataset.height),
    }


def _close_worker_dem_dataset() -> None:
    global _WORKER_DEM_DATASET
    if _WORKER_DEM_DATASET is not None:
        _WORKER_DEM_DATASET.close()
        _WORKER_DEM_DATASET = None


def _initialize_polygon_worker(
    dem_path: str,
    classification_settings: dict[str, Any],
) -> None:
    """Open one reusable read-only DEM handle in each spawned worker."""
    global _WORKER_DEM_DATASET
    global _WORKER_DEM_REFERENCE
    global _WORKER_CLASSIFICATION_SETTINGS
    _WORKER_DEM_DATASET = rasterio.open(dem_path)
    _WORKER_DEM_REFERENCE = _dataset_grid_reference(_WORKER_DEM_DATASET)
    _WORKER_CLASSIFICATION_SETTINGS = dict(classification_settings)
    atexit.register(_close_worker_dem_dataset)


def _pack_local_mask(mask: np.ndarray) -> bytes:
    return np.packbits(np.asarray(mask, dtype=np.uint8).reshape(-1), bitorder="little").tobytes()


def _unpack_local_mask(payload: bytes, shape: tuple[int, int]) -> np.ndarray:
    cell_count = int(shape[0] * shape[1])
    unpacked = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8),
        count=cell_count,
        bitorder="little",
    )
    return unpacked.reshape(shape).astype(bool, copy=False)


def _classify_polygon_task_local(
    task: dict[str, Any],
    dem_dataset: Any,
    dem_reference: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Calculate one polygon locally without mutating shared output arrays."""
    landslide_id = str(task["landslide_id"])
    feature_index = int(task["feature_index"])
    geometry_wkb = task.get("geometry_wkb")
    base_result = {
        "landslide_id": landslide_id,
        "feature_index": feature_index,
    }
    if geometry_wkb is None:
        return {**base_result, "skipped": True, "skip_reason": "empty_geometry"}

    geometry = from_wkb(geometry_wkb)
    if geometry is None or geometry.is_empty:
        return {**base_result, "skipped": True, "skip_reason": "empty_geometry"}

    window = clamped_geometry_window(
        tuple(float(value) for value in geometry.bounds),
        dem_reference,
    )
    if window is None:
        return {**base_result, "skipped": True, "skip_reason": "outside_dem_extent"}

    row_off = int(window.row_off)
    col_off = int(window.col_off)
    out_shape = (int(window.height), int(window.width))
    local_transform = window_transform(window, dem_reference["transform"])
    dem_window = dem_dataset.read(1, window=window, masked=True)
    dem_values = dem_window.filled(np.nan).astype(np.float64)
    polygon_mask = rasterize(
        [(geometry, 1)],
        out_shape=out_shape,
        transform=local_transform,
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=bool(settings["all_touched"]),
    ).astype(bool)
    valid_dem_window = (~np.ma.getmaskarray(dem_window)) & np.isfinite(dem_values)
    valid_polygon_dem = polygon_mask & valid_dem_window
    valid_cell_count = int(np.count_nonzero(valid_polygon_dem))
    min_valid_dem_cells = int(settings["min_valid_dem_cells"])
    if valid_cell_count < min_valid_dem_cells:
        return {
            **base_result,
            "skipped": True,
            "skip_reason": f"too_few_valid_dem_cells<{min_valid_dem_cells}",
            "valid_dem_cell_count": valid_cell_count,
        }

    values = dem_values[valid_polygon_dem]
    low_threshold = float(
        np.nanpercentile(values, float(settings["depositional_percentile"]))
    )
    high_threshold = float(
        np.nanpercentile(values, float(settings["high_percentile"]))
    )
    if low_threshold >= high_threshold:
        return {
            **base_result,
            "skipped": True,
            "skip_reason": "threshold_overlap_or_insufficient_elevation_range",
            "valid_dem_cell_count": valid_cell_count,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        }

    local_depositional = valid_polygon_dem & (dem_values <= low_threshold)
    local_non_depositional = valid_polygon_dem & (dem_values >= high_threshold)
    local_uncertain = (
        valid_polygon_dem
        & (dem_values > low_threshold)
        & (dem_values < high_threshold)
    )
    depositional_count = int(np.count_nonzero(local_depositional))
    non_depositional_count = int(np.count_nonzero(local_non_depositional))
    uncertain_count = int(np.count_nonzero(local_uncertain))
    return {
        **base_result,
        "skipped": False,
        "window": (row_off, col_off, out_shape[0], out_shape[1]),
        "shape": out_shape,
        "depositional_mask": _pack_local_mask(local_depositional),
        "non_depositional_mask": _pack_local_mask(local_non_depositional),
        "uncertain_mask": _pack_local_mask(local_uncertain),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "valid_dem_cell_count": valid_cell_count,
        "depositional_cell_count": depositional_count,
        "non_depositional_cell_count": non_depositional_count,
        "uncertain_cell_count": uncertain_count,
    }


def _run_polygon_worker(task: dict[str, Any]) -> dict[str, Any]:
    if (
        _WORKER_DEM_DATASET is None
        or _WORKER_DEM_REFERENCE is None
        or _WORKER_CLASSIFICATION_SETTINGS is None
    ):
        raise RuntimeError("Step 1 polygon worker was not initialized.")
    try:
        return _classify_polygon_task_local(
            task,
            _WORKER_DEM_DATASET,
            _WORKER_DEM_REFERENCE,
            _WORKER_CLASSIFICATION_SETTINGS,
        )
    except Exception as exc:
        raise RuntimeError(
            "Polygon classification failed for "
            f"{task.get('landslide_id')} (feature {task.get('feature_index')}): {exc}"
        ) from exc


def _iter_parallel_polygon_results(
    tasks: list[dict[str, Any]],
    dem_path: str,
    settings: dict[str, Any],
    worker_count: int,
) -> Iterator[dict[str, Any]]:
    """Yield deterministic results while keeping only a bounded task queue."""
    queue_limit = max(worker_count, worker_count * 2)
    executor = ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_initialize_polygon_worker,
        initargs=(dem_path, settings),
    )
    pending: dict[Future[dict[str, Any]], int] = {}
    completed: dict[int, dict[str, Any]] = {}
    next_submit = 0
    next_yield = 0
    try:
        while next_yield < len(tasks):
            while (
                next_submit < len(tasks)
                and len(pending) + len(completed) < queue_limit
            ):
                future = executor.submit(_run_polygon_worker, tasks[next_submit])
                pending[future] = next_submit
                next_submit += 1

            if next_yield in completed:
                result = completed.pop(next_yield)
                next_yield += 1
                yield result
                continue

            if not pending:
                raise RuntimeError(
                    "Polygon worker queue ended before all results were returned."
                )
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                task_index = pending.pop(future)
                completed[task_index] = future.result()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def classify_internal_landslide_zones_per_polygon(
    landslides: gpd.GeoDataFrame,
    dem_path: str,
    dem_reference: dict[str, Any],
    depositional_percentile: float,
    non_depositional_percentile: float,
    nodata_value: int,
    min_valid_dem_cells: int,
    all_touched: bool,
    logger: logging.Logger,
    working_dir: Path | None = None,
    parallel_processing: bool = True,
    max_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Classify high-confidence internal landslide zones by elevation percentiles.

    The lower-elevation portion of each mapped landslide is treated as the most likely
    final accumulation zone. The upper-elevation portion is treated as source or
    upper-transport terrain where material initiated or continued moving rather than
    accumulating. Middle-elevation terrain and outside-polygon cells stay unlabeled.
    """
    depositional_mask = allocate_grid_array(
        dem_reference["shape"], "bool", False, "depositional", working_dir
    )
    non_depositional_mask = allocate_grid_array(
        dem_reference["shape"], "bool", False, "non_depositional", working_dir
    )
    uncertain_mask = allocate_grid_array(
        dem_reference["shape"], "bool", False, "uncertain", working_dir
    )
    landslide_id_nodata = int(nodata_value)
    landslide_id_array = allocate_grid_array(
        dem_reference["shape"], "int32", landslide_id_nodata, "landslide_id", working_dir
    )
    per_landslide_rows: list[dict[str, Any]] = []
    low_thresholds: list[float] = []
    high_thresholds: list[float] = []
    skipped_rows: list[dict[str, Any]] = []
    high_percentile = 100.0 - float(non_depositional_percentile)

    tasks = [
        {
            "landslide_id": str(landslide_id),
            "feature_index": int(feature_index),
            "geometry_wkb": None if geometry is None else geometry.wkb,
        }
        for geometry, landslide_id, feature_index in zip(
            landslides.geometry,
            landslides["_landslide_id"],
            landslides["_feature_index"],
            strict=True,
        )
    ]
    worker_count = min(
        max(1, int(max_workers)),
        max(1, len(tasks)),
        max(1, int(os.cpu_count() or 1)),
    )
    use_parallel = bool(parallel_processing) and worker_count > 1
    settings = {
        "depositional_percentile": float(depositional_percentile),
        "high_percentile": float(high_percentile),
        "min_valid_dem_cells": int(min_valid_dem_cells),
        "all_touched": bool(all_touched),
    }
    logger.info(
        "Polygon classification execution: mode=%s workers=%d polygons=%d.",
        "multiprocessing" if use_parallel else "serial",
        worker_count if use_parallel else 1,
        len(tasks),
    )

    try:
        if use_parallel:
            result_iterator: Iterator[dict[str, Any]] = _iter_parallel_polygon_results(
                tasks,
                dem_path,
                settings,
                worker_count,
            )
        else:
            dem_dataset = rasterio.open(dem_path)
            serial_reference = _dataset_grid_reference(dem_dataset)

            def serial_results() -> Iterator[dict[str, Any]]:
                try:
                    for task in tasks:
                        yield _classify_polygon_task_local(
                            task,
                            dem_dataset,
                            serial_reference,
                            settings,
                        )
                finally:
                    dem_dataset.close()

            result_iterator = serial_results()

        progress_interval = max(1, len(tasks) // 20)
        for result_number, result in enumerate(result_iterator, start=1):
            landslide_id = str(result["landslide_id"])
            feature_index = int(result["feature_index"])
            if bool(result["skipped"]):
                skipped_rows.append(
                    {
                        "landslide_id": landslide_id,
                        "feature_index": feature_index,
                        "skip_reason": str(result["skip_reason"]),
                        "valid_dem_cell_count": int(
                            result.get("valid_dem_cell_count", 0)
                        ),
                        "low_threshold": float(
                            result.get("low_threshold", np.nan)
                        ),
                        "high_threshold": float(
                            result.get("high_threshold", np.nan)
                        ),
                    }
                )
            else:
                row_off, col_off, window_height, window_width = (
                    int(value) for value in result["window"]
                )
                row_stop = row_off + window_height
                col_stop = col_off + window_width
                shape = (window_height, window_width)
                local_depositional = _unpack_local_mask(
                    result["depositional_mask"], shape
                )
                local_non_depositional = _unpack_local_mask(
                    result["non_depositional_mask"], shape
                )
                local_uncertain = _unpack_local_mask(
                    result["uncertain_mask"], shape
                )
                valid_polygon_dem = (
                    local_depositional | local_non_depositional | local_uncertain
                )

                depositional_mask[row_off:row_stop, col_off:col_stop] |= (
                    local_depositional
                )
                non_depositional_mask[row_off:row_stop, col_off:col_stop] |= (
                    local_non_depositional
                )
                uncertain_mask[row_off:row_stop, col_off:col_stop] |= local_uncertain
                landslide_id_window = landslide_id_array[
                    row_off:row_stop, col_off:col_stop
                ]
                landslide_id_window[valid_polygon_dem] = feature_index

                low_threshold = float(result["low_threshold"])
                high_threshold = float(result["high_threshold"])
                valid_cell_count = int(result["valid_dem_cell_count"])
                depositional_count = int(result["depositional_cell_count"])
                non_depositional_count = int(
                    result["non_depositional_cell_count"]
                )
                uncertain_count = int(result["uncertain_cell_count"])
                low_thresholds.append(low_threshold)
                high_thresholds.append(high_threshold)
                per_landslide_rows.append(
                    {
                        "landslide_id": landslide_id,
                        "feature_index": feature_index,
                        "low_threshold_elevation": low_threshold,
                        "high_threshold_elevation": high_threshold,
                        "valid_dem_cell_count": valid_cell_count,
                        "depositional_cell_count": depositional_count,
                        "non_depositional_cell_count": non_depositional_count,
                        "uncertain_cell_count": uncertain_count,
                        "depositional_percentage": (
                            100.0 * depositional_count / valid_cell_count
                        ),
                        "non_depositional_percentage": (
                            100.0 * non_depositional_count / valid_cell_count
                        ),
                        "uncertain_percentage": (
                            100.0 * uncertain_count / valid_cell_count
                        ),
                        "skipped": False,
                        "skip_reason": "",
                    }
                )

            if result_number % progress_interval == 0 or result_number == len(tasks):
                logger.info(
                    "Polygon classification progress: %d/%d (%.1f%%).",
                    result_number,
                    len(tasks),
                    100.0 * result_number / max(1, len(tasks)),
                )
    except Exception:
        for array in (
            depositional_mask,
            non_depositional_mask,
            uncertain_mask,
            landslide_id_array,
        ):
            close_grid_array(array)
        raise

    if not np.any(depositional_mask):
        for array in (
            depositional_mask,
            non_depositional_mask,
            uncertain_mask,
            landslide_id_array,
        ):
            close_grid_array(array)
        raise ValueError(
            "Depositional zone classification produced zero cells. Check DEM coverage, CRS, "
            "polygon geometry, raster resolution, or the selected low percentile."
        )
    
    if not np.any(non_depositional_mask):
        for array in (
            depositional_mask,
            non_depositional_mask,
            uncertain_mask,
            landslide_id_array,
        ):
            close_grid_array(array)
        raise ValueError(
            "Non-depositional zone classification produced zero cells. Check DEM coverage, CRS, "
            "polygon geometry, raster resolution, or the selected high percentile."
        )

    for skipped in skipped_rows:
        per_landslide_rows.append(
            {
                "landslide_id": skipped["landslide_id"],
                "feature_index": int(skipped["feature_index"]),
                "low_threshold_elevation": float(skipped.get("low_threshold", np.nan)),
                "high_threshold_elevation": float(skipped.get("high_threshold", np.nan)),
                "valid_dem_cell_count": int(skipped.get("valid_dem_cell_count", 0)),
                "depositional_cell_count": 0,
                "non_depositional_cell_count": 0,
                "uncertain_cell_count": 0,
                "depositional_percentage": np.nan,
                "non_depositional_percentage": np.nan,
                "uncertain_percentage": np.nan,
                "skipped": True,
                "skip_reason": str(skipped["skip_reason"]),
            }
        )

    diagnostics = {
        "processed_polygons": int(len(per_landslide_rows) - len(skipped_rows)),
        "skipped_polygons": int(len(skipped_rows)),
        "low_threshold_min": float(np.min(low_thresholds)) if low_thresholds else None,
        "low_threshold_max": float(np.max(low_thresholds)) if low_thresholds else None,
        "low_threshold_mean": float(np.mean(low_thresholds)) if low_thresholds else None,
        "low_threshold_median": float(np.median(low_thresholds)) if low_thresholds else None,
        "high_threshold_min": float(np.min(high_thresholds)) if high_thresholds else None,
        "high_threshold_max": float(np.max(high_thresholds)) if high_thresholds else None,
        "high_threshold_mean": float(np.mean(high_thresholds)) if high_thresholds else None,
        "high_threshold_median": float(np.median(high_thresholds)) if high_thresholds else None,
        "depositional_cells": int(np.count_nonzero(depositional_mask)),
        "non_depositional_cells": int(np.count_nonzero(non_depositional_mask)),
        "uncertain_cells": int(np.count_nonzero(uncertain_mask)),
        "depositional_percentile": float(depositional_percentile),
        "non_depositional_percentile": float(non_depositional_percentile),
        "computed_high_threshold_percentile": float(high_percentile),
        "parallel_processing_requested": bool(parallel_processing),
        "parallel_processing_used": bool(use_parallel),
        "polygon_worker_count": int(worker_count if use_parallel else 1),
    }
    processed_rows = [row for row in per_landslide_rows if not bool(row["skipped"])]
    diagnostics["depositional_percentage_deviation_gt_5pp_polygons"] = int(
        sum(abs(float(row["depositional_percentage"]) - depositional_percentile) > 5.0 for row in processed_rows)
    )
    diagnostics["non_depositional_percentage_deviation_gt_5pp_polygons"] = int(
        sum(
            abs(float(row["non_depositional_percentage"]) - non_depositional_percentile) > 5.0
            for row in processed_rows
        )
    )
    logger.info(
        "Per-polygon internal-zone classification complete: processed=%d skipped=%d depositional_cells=%d non_depositional_cells=%d uncertain_cells=%d.",
        diagnostics["processed_polygons"],
        diagnostics["skipped_polygons"],
        diagnostics["depositional_cells"],
        diagnostics["non_depositional_cells"],
        diagnostics["uncertain_cells"],
    )
    return depositional_mask, non_depositional_mask, uncertain_mask, landslide_id_array, per_landslide_rows, diagnostics


def build_final_label_raster(
    depositional_mask: np.ndarray,
    non_depositional_mask: np.ndarray,
    uncertain_mask: np.ndarray,
    nodata_value: int,
    output_dtype: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Build final label raster for internal landslide zones.
    
    - Initialize as NoData
    - Write non-depositional/source-transport as 0
    - Write depositional as 1
    - Keep uncertain and outside-polygon as NoData
    """
    effective_depositional = np.asarray(depositional_mask, dtype=bool)
    effective_non_depositional = np.asarray(non_depositional_mask & ~effective_depositional, dtype=bool)
    effective_uncertain = np.asarray(uncertain_mask & ~(effective_depositional | effective_non_depositional), dtype=bool)

    label = np.full(depositional_mask.shape, nodata_value, dtype=np.dtype(output_dtype))
    label[effective_non_depositional] = 0
    label[effective_depositional] = 1
    
    counts = {
        "total_raster_cells": int(label.size),
        "depositional_cells": int(np.count_nonzero(effective_depositional)),
        "non_depositional_cells": int(np.count_nonzero(effective_non_depositional)),
        "uncertain_cells": int(np.count_nonzero(effective_uncertain)),
        "valid_non_depositional_cells": int(np.count_nonzero(label == 0)),
        "nodata_cells": int(np.count_nonzero(label == nodata_value)),
        "depositional_non_depositional_overlap_cells": int(np.count_nonzero(depositional_mask & non_depositional_mask)),
        "uncertain_labeled_overlap_cells": int(np.count_nonzero(uncertain_mask & (depositional_mask | non_depositional_mask))),
    }
    
    return label, effective_depositional, effective_non_depositional, effective_uncertain, counts


def mask_to_array(mask: np.ndarray, nodata_value: int, output_dtype: str) -> np.ndarray:
    out = np.full(mask.shape, nodata_value, dtype=np.dtype(output_dtype))
    out[mask] = 1
    return out


def iter_grid_windows(width: int, height: int, window_size: int = 1024) -> Any:
    for row_off in range(0, height, window_size):
        window_height = min(window_size, height - row_off)
        for col_off in range(0, width, window_size):
            window_width = min(window_size, width - col_off)
            yield Window(col_off, row_off, window_width, window_height)


def raster_output_profile(
    dem_reference: dict[str, Any],
    output_dtype: str,
    nodata_value: int,
) -> dict[str, Any]:
    profile = dem_reference["profile"].copy()
    profile.update(
        driver="GTiff",
        dtype=output_dtype,
        count=1,
        nodata=nodata_value,
        compress="lzw",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    return profile


def write_resolved_outputs_windowed(
    output_paths: dict[str, Path],
    depositional_mask: np.ndarray,
    non_depositional_mask: np.ndarray,
    uncertain_mask: np.ndarray,
    landslide_id_array: np.ndarray,
    dem_reference: dict[str, Any],
    nodata_value: int,
    output_dtype: str,
) -> dict[str, int]:
    """Resolve overlaps and write all rasters without materializing full output arrays."""
    label_profile = raster_output_profile(dem_reference, output_dtype, nodata_value)
    id_profile = raster_output_profile(dem_reference, "int32", nodata_value)
    counts = {
        "total_raster_cells": int(dem_reference["width"] * dem_reference["height"]),
        "depositional_cells": 0,
        "non_depositional_cells": 0,
        "uncertain_cells": 0,
        "valid_non_depositional_cells": 0,
        "nodata_cells": 0,
        "depositional_non_depositional_overlap_cells": 0,
        "uncertain_labeled_overlap_cells": 0,
        "landslide_id_cells": 0,
        "landslide_id_unique_values": 0,
    }
    unique_ids: set[int] = set()

    with ExitStack() as stack:
        dep_dst = stack.enter_context(rasterio.open(output_paths["depositional"], "w", **label_profile))
        nondep_dst = stack.enter_context(
            rasterio.open(output_paths["non_depositional"], "w", **label_profile)
        )
        id_dst = stack.enter_context(rasterio.open(output_paths["landslide_id"], "w", **id_profile))
        label_dst = stack.enter_context(rasterio.open(output_paths["label"], "w", **label_profile))

        for window in iter_grid_windows(dem_reference["width"], dem_reference["height"]):
            row_start = int(window.row_off)
            row_stop = row_start + int(window.height)
            col_start = int(window.col_off)
            col_stop = col_start + int(window.width)
            selection = np.s_[row_start:row_stop, col_start:col_stop]

            raw_dep = np.asarray(depositional_mask[selection], dtype=bool)
            raw_nondep = np.asarray(non_depositional_mask[selection], dtype=bool)
            raw_uncertain = np.asarray(uncertain_mask[selection], dtype=bool)
            effective_dep = raw_dep
            effective_nondep = raw_nondep & ~effective_dep
            effective_uncertain = raw_uncertain & ~(effective_dep | effective_nondep)

            label = np.full(raw_dep.shape, nodata_value, dtype=np.dtype(output_dtype))
            label[effective_nondep] = 0
            label[effective_dep] = 1
            dep_array = mask_to_array(effective_dep, nodata_value, output_dtype)
            nondep_array = mask_to_array(effective_nondep, nodata_value, output_dtype)
            id_array = np.asarray(landslide_id_array[selection], dtype=np.int32)

            dep_dst.write(dep_array, 1, window=window)
            nondep_dst.write(nondep_array, 1, window=window)
            id_dst.write(id_array, 1, window=window)
            label_dst.write(label, 1, window=window)

            dep_count = int(np.count_nonzero(effective_dep))
            nondep_count = int(np.count_nonzero(effective_nondep))
            uncertain_count = int(np.count_nonzero(effective_uncertain))
            counts["depositional_cells"] += dep_count
            counts["non_depositional_cells"] += nondep_count
            counts["uncertain_cells"] += uncertain_count
            counts["valid_non_depositional_cells"] += nondep_count
            counts["depositional_non_depositional_overlap_cells"] += int(
                np.count_nonzero(raw_dep & raw_nondep)
            )
            counts["uncertain_labeled_overlap_cells"] += int(
                np.count_nonzero(raw_uncertain & (raw_dep | raw_nondep))
            )
            valid_ids = id_array[id_array != nodata_value]
            counts["landslide_id_cells"] += int(valid_ids.size)
            unique_ids.update(int(value) for value in np.unique(valid_ids))

    counts["landslide_id_unique_values"] = len(unique_ids)
    counts["nodata_cells"] = (
        counts["total_raster_cells"]
        - counts["depositional_cells"]
        - counts["non_depositional_cells"]
    )
    return counts


def write_run_config(
    output_path: Path,
    config: dict[str, Any],
    dem_reference: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "config": config,
        "dem_grid": {
            "width": int(dem_reference["width"]),
            "height": int(dem_reference["height"]),
            "resolution": [float(value) for value in dem_reference["res"]],
            "crs": str(dem_reference["crs"]),
            "transform": [float(value) for value in tuple(dem_reference["transform"])[:6]],
        },
        "diagnostics": diagnostics,
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def create_summary_csv(
    output_path: Path,
    class_counts: dict[str, int],
    per_landslide_rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> None:
    total_cells = max(1, int(class_counts["total_raster_cells"]))
    
    summary_rows = [
        {
            "record_type": "summary",
            "metric": "depositional_percentile",
            "value": float(diagnostics.get("depositional_percentile", 15)),
            "percentage_of_total_cells": np.nan,
            "notes": "Lower-elevation polygon percentage assigned label 1.",
        },
        {
            "record_type": "summary",
            "metric": "non_depositional_percentile",
            "value": float(diagnostics.get("non_depositional_percentile", 30)),
            "percentage_of_total_cells": np.nan,
            "notes": "Upper-elevation polygon percentage assigned label 0.",
        },
        {
            "record_type": "summary",
            "metric": "computed_high_threshold_percentile",
            "value": float(diagnostics.get("computed_high_threshold_percentile", 70)),
            "percentage_of_total_cells": np.nan,
            "notes": "Percentile threshold used for the upper source/transport zone.",
        },
        {
            "record_type": "summary",
            "metric": "parallel_processing_used",
            "value": bool(diagnostics.get("parallel_processing_used", False)),
            "percentage_of_total_cells": np.nan,
            "notes": "Whether polygon-local classification used worker processes.",
        },
        {
            "record_type": "summary",
            "metric": "polygon_worker_count",
            "value": int(diagnostics.get("polygon_worker_count", 1)),
            "percentage_of_total_cells": np.nan,
            "notes": "Effective number of polygon-classification worker processes.",
        },
        {
            "record_type": "summary",
            "metric": "methodology_note",
            "value": "",
            "percentage_of_total_cells": np.nan,
            "notes": (
                "Negative labels are derived from the upper portion of mapped landslide polygons, "
                "not from outside-polygon terrain. This contrasts terrain where material accumulated "
                "with terrain where material initiated or moved through without final deposition."
            ),
        },
        {
            "record_type": "summary",
            "metric": "total_raster_cells",
            "value": int(class_counts["total_raster_cells"]),
            "percentage_of_total_cells": 100.0,
            "notes": "",
        },
        {
            "record_type": "summary",
            "metric": "depositional_cells",
            "value": int(class_counts["depositional_cells"]),
            "percentage_of_total_cells": 100.0 * float(class_counts["depositional_cells"]) / float(total_cells),
            "notes": "Label value 1 - lowest elevation depositional zone.",
        },
        {
            "record_type": "summary",
            "metric": "non_depositional_cells",
            "value": int(class_counts.get("non_depositional_cells", 0)),
            "percentage_of_total_cells": 100.0 * float(class_counts.get("non_depositional_cells", 0)) / float(total_cells),
            "notes": "Label value 0 - highest elevation source/transport zone.",
        },
        {
            "record_type": "summary",
            "metric": "uncertain_cells",
            "value": int(class_counts.get("uncertain_cells", 0)),
            "percentage_of_total_cells": 100.0 * float(class_counts.get("uncertain_cells", 0)) / float(total_cells),
            "notes": "Middle elevation zone within polygons - excluded from training.",
        },
        {
            "record_type": "summary",
            "metric": "landslide_id_cells",
            "value": int(class_counts.get("landslide_id_cells", 0)),
            "percentage_of_total_cells": 100.0 * float(class_counts.get("landslide_id_cells", 0)) / float(total_cells),
            "notes": "Valid DEM cells inside mapped landslide polygons with a numeric landslide group ID.",
        },
        {
            "record_type": "summary",
            "metric": "landslide_id_unique_values",
            "value": int(class_counts.get("landslide_id_unique_values", 0)),
            "percentage_of_total_cells": np.nan,
            "notes": "Number of unique numeric landslide IDs burned into landslide_id_raster.tif.",
        },
        {
            "record_type": "summary",
            "metric": "nodata_cells",
            "value": int(class_counts["nodata_cells"]),
            "percentage_of_total_cells": 100.0 * float(class_counts["nodata_cells"]) / float(total_cells),
            "notes": "Outside mapped landslides or uncertain zones.",
        },
        {
            "record_type": "summary",
            "metric": "skipped_polygons",
            "value": int(diagnostics.get("skipped_polygons", 0)),
            "percentage_of_total_cells": np.nan,
            "notes": "Polygons skipped during per-feature percentile classification.",
        },
        {
            "record_type": "summary",
            "metric": "multipart_feature_count",
            "value": int(diagnostics.get("multipart_feature_count", 0)),
            "percentage_of_total_cells": np.nan,
            "notes": "Features whose disconnected polygon parts share one elevation distribution and group ID.",
        },
        {
            "record_type": "summary",
            "metric": "depositional_percentage_deviation_gt_5pp_polygons",
            "value": int(diagnostics.get("depositional_percentage_deviation_gt_5pp_polygons", 0)),
            "percentage_of_total_cells": np.nan,
            "notes": "Processed polygons whose realized depositional share differs from the target by over 5 percentage points.",
        },
        {
            "record_type": "summary",
            "metric": "non_depositional_percentage_deviation_gt_5pp_polygons",
            "value": int(diagnostics.get("non_depositional_percentage_deviation_gt_5pp_polygons", 0)),
            "percentage_of_total_cells": np.nan,
            "notes": "Processed polygons whose realized non-depositional share differs from the target by over 5 percentage points.",
        },
        {
            "record_type": "summary",
            "metric": "depositional_non_depositional_overlap_cells",
            "value": int(class_counts.get("depositional_non_depositional_overlap_cells", 0)),
            "percentage_of_total_cells": np.nan,
            "notes": "Cells resolved by depositional priority after polygon overlap.",
        },
        {
            "record_type": "summary",
            "metric": "uncertain_labeled_overlap_cells",
            "value": int(class_counts.get("uncertain_labeled_overlap_cells", 0)),
            "percentage_of_total_cells": np.nan,
            "notes": "Cells where uncertain status was superseded by a labeled class after polygon overlap.",
        },
    ]

    per_landslide_df = pd.DataFrame(per_landslide_rows)
    if not per_landslide_df.empty:
        per_landslide_df.insert(0, "record_type", "per_landslide")
        per_landslide_df["metric"] = ""
        per_landslide_df["value"] = np.nan
        per_landslide_df["percentage_of_total_cells"] = np.nan
        per_landslide_df["notes"] = ""

    summary_df = pd.DataFrame(summary_rows)
    output_df = pd.concat([summary_df, per_landslide_df], ignore_index=True, sort=False)
    output_df.to_csv(output_path, index=False)


def run_qc_checks(
    output_paths: dict[str, Path],
    dem_reference: dict[str, Any],
    nodata_value: int,
    class_counts: dict[str, int],
    logger: logging.Logger,
) -> None:
    errors: list[str] = []
    expected_values = {
        "depositional": {1},
        "non_depositional": {1},
        "label": {0, 1},
    }
    keys_to_check = ("depositional", "non_depositional", "label")

    for key in keys_to_check:
        if key not in output_paths or not output_paths[key].exists():
            errors.append(f"QC failed: required raster output '{key}' was not written.")
            continue
        with rasterio.open(output_paths[key]) as src:
            if src.count != 1:
                errors.append(f"{src.name} must contain exactly one raster band.")
            if src.width != dem_reference["width"] or src.height != dem_reference["height"]:
                errors.append(f"{src.name} dimensions do not match the DEM grid.")
            if src.transform != dem_reference["transform"]:
                errors.append(f"{src.name} transform does not match the DEM grid.")
            if src.crs != dem_reference["crs"]:
                errors.append(f"{src.name} CRS does not match the DEM grid.")
            if src.nodata != nodata_value:
                errors.append(f"{src.name} NoData metadata does not equal {nodata_value}.")
            unique_values: set[int | float] = set()
            for _, window in src.block_windows(1):
                arr = src.read(1, window=window)
                unique_values.update(np.unique(arr[arr != nodata_value]).tolist())
            if not unique_values.issubset(expected_values[key]):
                errors.append(
                    f"{src.name} contains unexpected data values {sorted(unique_values)}; "
                    f"expected subset of {sorted(expected_values[key])} plus NoData."
                )

    landslide_id_path = output_paths.get("landslide_id")
    if landslide_id_path is None or not landslide_id_path.exists():
        errors.append("QC failed: landslide_id_raster.tif was not written.")
    else:
        with rasterio.open(landslide_id_path) as src:
            if src.count != 1:
                errors.append(f"{src.name} must contain exactly one raster band.")
            if src.width != dem_reference["width"] or src.height != dem_reference["height"]:
                errors.append(f"{src.name} dimensions do not match the DEM grid.")
            if src.transform != dem_reference["transform"]:
                errors.append(f"{src.name} transform does not match the DEM grid.")
            if src.crs != dem_reference["crs"]:
                errors.append(f"{src.name} CRS does not match the DEM grid.")
            if src.nodata != nodata_value:
                errors.append(f"{src.name} NoData metadata does not equal {nodata_value}.")
            valid_id_count = 0
            minimum_id: int | None = None
            for _, window in src.block_windows(1):
                arr = src.read(1, window=window)
                valid_ids = arr[arr != nodata_value]
                valid_id_count += int(valid_ids.size)
                if valid_ids.size:
                    window_minimum = int(np.min(valid_ids))
                    minimum_id = window_minimum if minimum_id is None else min(minimum_id, window_minimum)
            if valid_id_count == 0:
                errors.append("QC failed: landslide_id_raster.tif has no valid landslide IDs.")
            elif minimum_id is not None and minimum_id <= 0:
                errors.append("QC failed: landslide_id_raster.tif contains non-positive valid IDs.")

    if class_counts["depositional_cells"] <= 0:
        errors.append("QC failed: depositional cell count is zero.")
    if class_counts.get("non_depositional_cells", 0) <= 0:
        errors.append("QC failed: non-depositional cell count is zero.")
    if class_counts.get("uncertain_cells", 0) <= 0:
        errors.append("QC failed: uncertain cell count is zero or not provided.")

    nodata_total = int(class_counts["nodata_cells"])
    if nodata_total < 0:
        errors.append("QC failed: NoData cell count is negative.")

    if errors:
        raise RuntimeError("QC checks failed:\n- " + "\n- ".join(errors))
    logger.info("QC checks passed for all raster outputs.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare per-pixel ML-DFI training labels from landslide polygons and a DEM using internal landslide zone classification."
    )
    parser.add_argument("--config-json", help="Optional Step 1 JSON config file.")
    parser.add_argument("--landslide-polygon-path")
    parser.add_argument("--dem-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--landslide-id-field")
    parser.add_argument("--depositional-percentile", type=float)
    parser.add_argument("--non-depositional-percentile", type=float)
    parser.add_argument("--nodata-value", type=int)
    parser.add_argument("--output-pixel-type")
    parser.add_argument("--min-valid-dem-cells", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument(
        "--all-touched",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include every pixel touched by a polygon boundary.",
    )
    parser.add_argument(
        "--parallel-processing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Process independent polygon DEM windows in spawned worker processes.",
    )
    return parser.parse_args()


def resolve_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    runtime = dict(CONFIG)
    if args.config_json:
        config_path = Path(args.config_json).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Step 1 JSON config does not exist: {config_path}")
        runtime.update(load_json_config(config_path))
    overrides = {
        "landslide_polygon_path": args.landslide_polygon_path,
        "dem_path": args.dem_path,
        "output_dir": args.output_dir,
        "landslide_id_field": args.landslide_id_field,
        "depositional_percentile": args.depositional_percentile,
        "non_depositional_percentile": args.non_depositional_percentile,
        "nodata_value": args.nodata_value,
        "output_pixel_type": args.output_pixel_type,
        "min_valid_dem_cells": args.min_valid_dem_cells,
        "max_workers": args.max_workers,
    }
    for key, value in overrides.items():
        if value is not None:
            runtime[key] = value
    if args.all_touched is not None:
        runtime["all_touched"] = bool(args.all_touched)
    if args.parallel_processing is not None:
        runtime["parallel_processing"] = bool(args.parallel_processing)
    for key in PATH_CONFIG_KEYS:
        value = runtime.get(key)
        if value is None or not str(value).strip():
            continue
        runtime[key] = resolve_config_path(
            str(value).strip(),
            Path.cwd(),
            vector=(key == "landslide_polygon_path"),
        )
    return runtime


def main() -> None:
    args = parse_args()
    logger: logging.Logger | None = None
    staging_dir: Path | None = None
    staged_log_path: Path | None = None
    output_dir: Path | None = None
    working_arrays: list[np.ndarray] = []

    try:
        config = resolve_runtime_config(args)
        validate_inputs(config)
        output_dir = Path(str(config["output_dir"])).expanduser().resolve()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=".step1-stage-", dir=output_dir.parent))
        logger, staged_log_path = setup_logging(staging_dir)
        logger.info("Starting Step 1 inventory label preparation.")
        logger.info("Input landslide polygons: %s", config["landslide_polygon_path"])
        logger.info("Input DEM: %s", config["dem_path"])
        logger.info("Validated outputs will be published to: %s", output_dir)
        logger.info("Staging directory: %s", staging_dir)
        logger.info("Depositional percentile: %.2f", float(config["depositional_percentile"]))
        logger.info(
            "Non-depositional upper-zone percentage: %.2f",
            float(config["non_depositional_percentile"]),
        )
        logger.info(
            "Computed high-elevation threshold percentile: %.2f",
            100.0 - float(config["non_depositional_percentile"]),
        )
        logger.info(
            "Polygon multiprocessing requested: %s; maximum workers: %d",
            bool(config["parallel_processing"]),
            int(config["max_workers"]),
        )
        if "PROJ" in CONFIGURED_GEOSPATIAL_DIRS:
            logger.info("Using PROJ data directory: %s", CONFIGURED_GEOSPATIAL_DIRS["PROJ"])
        if "GDAL" in CONFIGURED_GEOSPATIAL_DIRS:
            logger.info("Using GDAL data directory: %s", CONFIGURED_GEOSPATIAL_DIRS["GDAL"])

        output_dtype = output_dtype_name(config)
        nodata_value = int(config["nodata_value"])

        with rasterio.open(str(config["dem_path"])) as dem_ds:
            if dem_ds.count < 1:
                raise ValueError("DEM raster has no band 1.")
            if dem_ds.crs is None:
                raise ValueError("DEM raster has no CRS.")
            dem_reference = {
                "profile": dem_ds.profile.copy(),
                "transform": dem_ds.transform,
                "crs": dem_ds.crs,
                "width": dem_ds.width,
                "height": dem_ds.height,
                "shape": (dem_ds.height, dem_ds.width),
                "bounds": dem_ds.bounds,
                "res": dem_ds.res,
            }

        logger.info(
            "DEM grid: width=%d height=%d resolution=(%.6f, %.6f) CRS=%s",
            dem_reference["width"],
            dem_reference["height"],
            float(dem_reference["res"][0]),
            float(dem_reference["res"][1]),
            dem_reference["crs"],
        )

        landslides, vector_diagnostics = load_and_reproject_vectors(
            landslide_polygon_path=str(config["landslide_polygon_path"]),
            dem_crs=dem_reference["crs"],
            landslide_id_field=str(config.get("landslide_id_field", "")),
            logger=logger,
        )
        logger.info("Loaded %d landslide polygons.", len(landslides))

        depositional_mask, non_depositional_mask, uncertain_mask, landslide_id_array, per_landslide_rows, classification_diagnostics = (
            classify_internal_landslide_zones_per_polygon(
                landslides=landslides,
                dem_path=str(config["dem_path"]),
                dem_reference=dem_reference,
                depositional_percentile=float(config["depositional_percentile"]),
                non_depositional_percentile=float(config["non_depositional_percentile"]),
                nodata_value=nodata_value,
                min_valid_dem_cells=int(config["min_valid_dem_cells"]),
                all_touched=bool(config["all_touched"]),
                logger=logger,
                working_dir=staging_dir,
                parallel_processing=bool(config["parallel_processing"]),
                max_workers=int(config["max_workers"]),
            )
        )
        working_arrays = [
            depositional_mask,
            non_depositional_mask,
            uncertain_mask,
            landslide_id_array,
        ]

        output_paths = {
            "depositional": staging_dir / MASK_OUTPUTS["depositional"],
            "non_depositional": staging_dir / MASK_OUTPUTS["non_depositional"],
            "landslide_id": staging_dir / MASK_OUTPUTS["landslide_id"],
            "label": staging_dir / MASK_OUTPUTS["label"],
            "summary": staging_dir / MASK_OUTPUTS["summary"],
            "run_config": staging_dir / MASK_OUTPUTS["run_config"],
            "log": staged_log_path,
        }

        logger.info("Writing aligned raster outputs in bounded-memory windows.")
        class_counts = write_resolved_outputs_windowed(
            output_paths=output_paths,
            depositional_mask=depositional_mask,
            non_depositional_mask=non_depositional_mask,
            uncertain_mask=uncertain_mask,
            landslide_id_array=landslide_id_array,
            dem_reference=dem_reference,
            nodata_value=nodata_value,
            output_dtype=output_dtype,
        )
        for array in working_arrays:
            close_grid_array(array)
        working_arrays.clear()
        del depositional_mask, non_depositional_mask, uncertain_mask, landslide_id_array
        gc.collect()

        combined_diagnostics = {}
        combined_diagnostics.update(vector_diagnostics)
        combined_diagnostics.update(classification_diagnostics)
        combined_diagnostics["depositional_percentile"] = float(config["depositional_percentile"])
        combined_diagnostics["non_depositional_percentile"] = float(config["non_depositional_percentile"])

        create_summary_csv(
            output_paths["summary"],
            class_counts=class_counts,
            per_landslide_rows=sorted(per_landslide_rows, key=lambda item: int(item["feature_index"])),
            diagnostics=combined_diagnostics,
        )
        write_run_config(
            output_paths["run_config"],
            config=config,
            dem_reference=dem_reference,
            diagnostics={**combined_diagnostics, **class_counts},
        )
        run_qc_checks(output_paths, dem_reference, nodata_value, class_counts, logger)

        dep_deviation_count = int(
            combined_diagnostics.get("depositional_percentage_deviation_gt_5pp_polygons", 0)
        )
        nondep_deviation_count = int(
            combined_diagnostics.get("non_depositional_percentage_deviation_gt_5pp_polygons", 0)
        )
        if dep_deviation_count or nondep_deviation_count:
            logger.warning(
                "Discrete pixels or tied DEM elevations caused class shares to differ by more than "
                "5 percentage points in %d depositional and %d non-depositional polygons.",
                dep_deviation_count,
                nondep_deviation_count,
            )
        logger.info("Skipped polygons: %d", int(combined_diagnostics.get("skipped_polygons", 0)))
        logger.info("Depositional cells: %d", class_counts["depositional_cells"])
        logger.info("Non-depositional cells: %d", class_counts.get("non_depositional_cells", 0))
        logger.info("Uncertain cells: %d", class_counts.get("uncertain_cells", 0))
        logger.info(
            "Resolved depositional/non-depositional overlap cells: %d",
            class_counts.get("depositional_non_depositional_overlap_cells", 0),
        )
        logger.info(
            "Resolved uncertain/labeled overlap cells: %d",
            class_counts.get("uncertain_labeled_overlap_cells", 0),
        )
        logger.info("NoData cells: %d", class_counts["nodata_cells"])
        logger.info("QC passed; publishing the complete Step 1 output bundle.")
        logger.info("Output label raster: %s", output_dir / MASK_OUTPUTS["label"])
        logger.info("Output landslide ID raster: %s", output_dir / MASK_OUTPUTS["landslide_id"])
        logger.info("Output summary CSV: %s", output_dir / MASK_OUTPUTS["summary"])
        logger.info("Resolved run config: %s", output_dir / MASK_OUTPUTS["run_config"])
        logger.info("Completed successfully.")

        close_file_log_handlers(logger)
        final_paths = publish_output_bundle(output_paths, output_dir, logger)
        logger.info("Published %d validated Step 1 outputs to %s", len(final_paths), output_dir)
        shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir = None
    except Exception as exc:
        if logger is not None:
            logger.exception("Step 1 inventory label preparation failed: %s", exc)
            close_file_log_handlers(logger)
        else:
            print(f"Step 1 inventory label preparation failed: {exc}", file=sys.stderr)
        for array in working_arrays:
            try:
                close_grid_array(array)
            except Exception:
                pass
        working_arrays.clear()
        gc.collect()
        if staged_log_path is not None and staged_log_path.exists() and output_dir is not None:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                failed_log = output_dir / "processing_log.failed.txt"
                failed_log_tmp = output_dir / ".processing_log.failed.tmp"
                shutil.copy2(staged_log_path, failed_log_tmp)
                os.replace(failed_log_tmp, failed_log)
                print(f"Failure log: {failed_log}", file=sys.stderr)
            except OSError as log_exc:
                print(f"Could not preserve the Step 1 failure log: {log_exc}", file=sys.stderr)
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
