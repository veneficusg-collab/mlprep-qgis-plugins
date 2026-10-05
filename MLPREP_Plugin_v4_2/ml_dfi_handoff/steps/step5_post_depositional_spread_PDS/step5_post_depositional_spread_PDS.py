"""Step 5 post-depositional spread PDS (OSGeo-Free Edition)"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import rasterio
from rasterio.crs import CRS
import fiona
from shapely.geometry import shape, mapping
from rasterio.features import shapes


@dataclass(frozen=True)
class PostDepositionalSpreadParams:
    runout_mask_raster: str
    dfi_raster: str
    dem_raster: str
    source_mask_raster: str
    stream_mask_raster: str
    source_contributing_area_raster: str
    output_spread_mask: str
    output_combined_mask: str
    output_seed_mask: str
    output_cleaned_runout_mask: str = ""
    output_removed_runout_mask: str = ""
    output_spread_mask_shapefile: str = ""
    output_combined_mask_shapefile: str = ""
    output_summary_json: str = ""
    output_min_elev_diff: str = ""
    output_source_row: str = ""
    output_source_col: str = ""
    dfi_threshold: float = 0.69
    max_relative_rise_m: float = 1.0
    min_runout_source_area_m2: float = 200.0
    min_spread_radius_m: float = 5.0
    max_spread_radius_m: float = 30.0
    source_contributing_area_reference: float = 4400.0
    write_debug_rasters: bool = False

@dataclass(frozen=True)
class RasterMetadata:
    path: str
    rows: int
    cols: int
    geotransform: tuple[float, float, float, float, float, float]
    projection_wkt: str
    nodata: float | None
    dtype_name: str

@dataclass(frozen=True)
class RasterSpec(RasterMetadata):
    array: np.ndarray

@dataclass(frozen=True)
class SpreadSummary:
    input_paths: dict[str, str]
    output_paths: dict[str, str]
    rows: int
    cols: int
    pixel_size: tuple[float, float]
    geotransform: tuple[float, float, float, float, float, float]
    crs_authority: str | None
    crs_wkt: str
    dfi_threshold: float
    max_relative_rise_m: float
    min_spread_radius_m: float
    max_spread_radius_m: float
    source_contributing_area_reference: float
    spread_mode: str
    min_runout_source_area_m2: float
    original_runout_cells: int
    area_removed_runout_cells: int
    disconnected_runout_cells: int
    removed_runout_cells: int
    removed_runout_percent: float | None
    cleaned_runout_cells: int
    valid_runout_cells: int
    seed_cells: int
    seeds_with_zero_contributing_source_area: int
    seeds_assigned_min_radius: int
    seeds_assigned_max_or_near_max_radius: int
    spread_cells_added: int
    combined_footprint_cells: int
    source_cells_processed: int
    candidate_evaluations: int
    neighbor_offset_count: int
    neighbor_offsets: list[list[float]]
    dfi_stats: dict[str, float | None] = field(default_factory=dict)
    source_contributing_area_stats: dict[str, float | None] = field(default_factory=dict)
    source_contribution_index_stats: dict[str, float | None] = field(default_factory=dict)
    computed_spread_radius_stats: dict[str, float | None] = field(default_factory=dict)
    rejection_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    timings_seconds: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class NativeSpreadResult:
    spread_mask: np.ndarray
    combined_mask: np.ndarray
    seed_mask: np.ndarray
    cleaned_runout_mask: np.ndarray
    removed_runout_mask: np.ndarray
    min_elev_diff: np.ndarray
    source_index: np.ndarray
    source_contributing_area: np.ndarray | None
    source_contribution_index: np.ndarray | None
    computed_spread_radius: np.ndarray | None
    summary: SpreadSummary

@dataclass(frozen=True)
class StagedOutputBundle:
    official_params: PostDepositionalSpreadParams
    staged_params: PostDepositionalSpreadParams
    staged_to_official: dict[str, str]
    stage_dirs: tuple[Path, ...]

REQUIRED_CONFIG_FIELDS = {
    "runout_mask_raster", "dfi_raster", "dem_raster", "source_mask_raster", 
    "stream_mask_raster", "source_contributing_area_raster", "output_spread_mask", 
    "output_combined_mask", "output_seed_mask", "output_cleaned_runout_mask", 
    "output_removed_runout_mask", "output_summary_json",
}

OPTIONAL_CONFIG_FIELDS = {
    "output_spread_mask_shapefile", "output_combined_mask_shapefile", "output_min_elev_diff", 
    "output_source_row", "output_source_col", "dfi_threshold", "max_relative_rise_m", 
    "min_runout_source_area_m2", "min_spread_radius_m", "max_spread_radius_m", 
    "source_contributing_area_reference", "write_debug_rasters",
}

ACTIVE_CONFIG_FIELDS = REQUIRED_CONFIG_FIELDS | OPTIONAL_CONFIG_FIELDS

INPUT_PARAM_FIELDS = (
    "runout_mask_raster", "dfi_raster", "dem_raster", "source_mask_raster", 
    "stream_mask_raster", "source_contributing_area_raster",
)

CORE_OUTPUT_PARAM_FIELDS = (
    "output_spread_mask", "output_combined_mask", "output_seed_mask", 
    "output_cleaned_runout_mask", "output_removed_runout_mask", "output_summary_json",
)

OPTIONAL_OUTPUT_PARAM_FIELDS = (
    "output_spread_mask_shapefile", "output_combined_mask_shapefile", 
    "output_min_elev_diff", "output_source_row", "output_source_col",
)

FLOAT32_NODATA = np.float32(-3.402823466e38)
INT32_NODATA = -1
GT_TOLERANCE = 1e-9
LINEAR_UNITS_TOLERANCE = 1e-9
ELEVATION_TIE_TOLERANCE = 1e-12
EIGHT_NEIGHBOR_STEPS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1),
)

def load_json_object(config_path: str) -> tuple[dict[str, object], Path]:
    cfg_path = Path(config_path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a JSON object: {cfg_path}")
    return {str(key): value for key, value in payload.items()}, cfg_path.parent

def require_non_empty_string(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text: raise ValueError(f"{field_name} must be a non-empty string.")
    return text

def resolve_path(root: Path, value: object) -> str:
    text = require_non_empty_string(value, "path")
    path = Path(text)
    if path.is_absolute(): return str(path)
    return str((root / path).resolve())

def optional_resolve_path(root: Path, value: object) -> str:
    text = str(value or "").strip()
    if not text: return ""
    path = Path(text)
    if path.is_absolute(): return str(path)
    return str((root / path).resolve())

def parse_optional_float(value: object, *, field_name: str, default: float) -> float:
    if value is None or str(value).strip() == "": return float(default)
    try: parsed = float(value)
    except (TypeError, ValueError) as exc: raise ValueError(f"{field_name} must be numeric when provided.") from exc
    if not math.isfinite(parsed): raise ValueError(f"{field_name} must be finite.")
    return parsed

def parse_optional_bool(value: object, *, field_name: str, default: bool) -> bool:
    if value is None or str(value).strip() == "": return bool(default)
    if isinstance(value, bool): return value
    if isinstance(value, (int, float)) and value in {0, 1}: return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}: return True
    if text in {"false", "no", "0"}: return False
    raise ValueError(f"{field_name} must be a boolean value.")

def step5_input_paths(params: PostDepositionalSpreadParams) -> dict[str, str]:
    return {field_name: str(getattr(params, field_name)) for field_name in INPUT_PARAM_FIELDS}

def step5_output_paths(params: PostDepositionalSpreadParams) -> dict[str, str]:
    outputs = {field_name: str(getattr(params, field_name)) for field_name in CORE_OUTPUT_PARAM_FIELDS}
    for field_name in ("output_spread_mask_shapefile", "output_combined_mask_shapefile"):
        value = str(getattr(params, field_name)).strip()
        if value: outputs[field_name] = value
    if params.write_debug_rasters:
        for field_name in ("output_min_elev_diff", "output_source_row", "output_source_col"):
            value = str(getattr(params, field_name)).strip()
            if value: outputs[field_name] = value
    return outputs

def load_post_depositional_spread_params(config_path: str) -> PostDepositionalSpreadParams:
    cfg, root = load_json_object(config_path)
    return PostDepositionalSpreadParams(
        runout_mask_raster=resolve_path(root, require_non_empty_string(cfg.get("runout_mask_raster"), "runout_mask_raster")),
        dfi_raster=resolve_path(root, require_non_empty_string(cfg.get("dfi_raster"), "dfi_raster")),
        dem_raster=resolve_path(root, require_non_empty_string(cfg.get("dem_raster"), "dem_raster")),
        source_mask_raster=resolve_path(root, require_non_empty_string(cfg.get("source_mask_raster"), "source_mask_raster")),
        stream_mask_raster=resolve_path(root, require_non_empty_string(cfg.get("stream_mask_raster"), "stream_mask_raster")),
        source_contributing_area_raster=resolve_path(root, require_non_empty_string(cfg.get("source_contributing_area_raster"), "source_contributing_area_raster")),
        output_spread_mask=resolve_path(root, require_non_empty_string(cfg.get("output_spread_mask"), "output_spread_mask")),
        output_combined_mask=resolve_path(root, require_non_empty_string(cfg.get("output_combined_mask"), "output_combined_mask")),
        output_seed_mask=resolve_path(root, require_non_empty_string(cfg.get("output_seed_mask"), "output_seed_mask")),
        output_cleaned_runout_mask=resolve_path(root, require_non_empty_string(cfg.get("output_cleaned_runout_mask"), "output_cleaned_runout_mask")),
        output_removed_runout_mask=resolve_path(root, require_non_empty_string(cfg.get("output_removed_runout_mask"), "output_removed_runout_mask")),
        output_spread_mask_shapefile=optional_resolve_path(root, cfg.get("output_spread_mask_shapefile", "")),
        output_combined_mask_shapefile=optional_resolve_path(root, cfg.get("output_combined_mask_shapefile", "")),
        output_summary_json=resolve_path(root, require_non_empty_string(cfg.get("output_summary_json"), "output_summary_json")),
        output_min_elev_diff=optional_resolve_path(root, cfg.get("output_min_elev_diff", "")),
        output_source_row=optional_resolve_path(root, cfg.get("output_source_row", "")),
        output_source_col=optional_resolve_path(root, cfg.get("output_source_col", "")),
        dfi_threshold=parse_optional_float(cfg.get("dfi_threshold"), field_name="dfi_threshold", default=0.69),
        max_relative_rise_m=parse_optional_float(cfg.get("max_relative_rise_m"), field_name="max_relative_rise_m", default=1.0),
        min_runout_source_area_m2=parse_optional_float(cfg.get("min_runout_source_area_m2"), field_name="min_runout_source_area_m2", default=200.0),
        min_spread_radius_m=parse_optional_float(cfg.get("min_spread_radius_m"), field_name="min_spread_radius_m", default=5.0),
        max_spread_radius_m=parse_optional_float(cfg.get("max_spread_radius_m"), field_name="max_spread_radius_m", default=30.0),
        source_contributing_area_reference=parse_optional_float(cfg.get("source_contributing_area_reference"), field_name="source_contributing_area_reference", default=4400.0),
        write_debug_rasters=parse_optional_bool(cfg.get("write_debug_rasters"), field_name="write_debug_rasters", default=False),
    )

def _read_raster_metadata(path: str, label: str) -> RasterMetadata:
    with rasterio.open(path) as ds:
        gt = ds.transform.to_gdal()
        return RasterMetadata(
            path=os.path.abspath(path),
            rows=ds.height,
            cols=ds.width,
            geotransform=(gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]),
            projection_wkt=ds.crs.to_wkt() if ds.crs else "",
            nodata=ds.nodata,
            dtype_name=ds.dtypes[0]
        )

def _read_raster(path: str, label: str, metadata: RasterMetadata | None = None) -> RasterSpec:
    with rasterio.open(path) as ds:
        array = ds.read(1)
        gt = ds.transform.to_gdal()
        return RasterSpec(
            path=os.path.abspath(path),
            rows=ds.height,
            cols=ds.width,
            geotransform=(gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]),
            projection_wkt=ds.crs.to_wkt() if ds.crs else "",
            nodata=ds.nodata,
            dtype_name=ds.dtypes[0],
            array=array
        )

def _write_geotiff(out_path: str, reference: RasterSpec, array: np.ndarray, dtype_str: str, nodata: float | None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    transform = rasterio.transform.Affine.from_gdal(*reference.geotransform)
    crs = CRS.from_wkt(reference.projection_wkt) if reference.projection_wkt else None
    
    with rasterio.open(
        out_path, 'w', driver='GTiff', height=reference.rows, width=reference.cols,
        count=1, dtype=dtype_str, crs=crs, transform=transform, nodata=nodata,
        compress='lzw', tiled=True
    ) as dst:
        dst.write(array.astype(dtype_str), 1)

def _write_mask_shapefile(mask_raster_path: str, output_shapefile: str, label: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_shapefile)), exist_ok=True)
    with rasterio.open(mask_raster_path) as src:
        image = src.read(1)
        mask = image == 1
        results = (
            {'properties': {'value': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.transform))
        )
        with fiona.open(output_shapefile, 'w', driver='ESRI Shapefile', crs=src.crs, schema={'properties': [('value', 'int')], 'geometry': 'Polygon'}) as shp:
            for rec in results:
                shp.write(rec)

def preflight_step5(params: PostDepositionalSpreadParams) -> tuple[dict[str, RasterMetadata], dict[str, Any]]:
    labels = {
        "runout_mask_raster": "runout mask", "dfi_raster": "DFI", "dem_raster": "DEM",
        "source_mask_raster": "source mask", "stream_mask_raster": "stream mask",
        "source_contributing_area_raster": "source contributing area"
    }
    metadata = {f: _read_raster_metadata(getattr(params, f), labels[f]) for f in INPUT_PARAM_FIELDS}
    ref = metadata["runout_mask_raster"]
    return metadata, {
        "status": "metadata_passed",
        "grid": {"rows": ref.rows, "cols": ref.cols, "dx": ref.geotransform[1], "dy": abs(ref.geotransform[5])},
    }

def _valid_mask(array: np.ndarray, nodata: float | None) -> np.ndarray:
    mask = np.isfinite(array)
    if nodata is not None and np.isfinite(nodata):
        mask &= ~np.isclose(array, nodata, atol=0.0, rtol=0.0)
    return mask

def _compute_neighbor_offsets(dx: float, dy: float, spread_radius_m: float) -> list[tuple[int, int, float]]:
    if spread_radius_m <= 0.0: return []
    max_row = int(math.ceil(spread_radius_m / dy))
    max_col = int(math.ceil(spread_radius_m / dx))
    offsets = []
    for r in range(-max_row, max_row + 1):
        for c in range(-max_col, max_col + 1):
            if r == 0 and c == 0: continue
            dist = math.hypot(c * dx, r * dy)
            if dist <= spread_radius_m + GT_TOLERANCE:
                offsets.append((r, c, dist))
    offsets.sort(key=lambda x: (x[2], abs(x[0]), abs(x[1])))
    return offsets

def run_post_depositional_spread_core(
    runout_array: np.ndarray, runout_nodata: float | None,
    dfi_array: np.ndarray, dfi_nodata: float | None,
    dem_array: np.ndarray, dem_nodata: float | None,
    source_mask_array: np.ndarray, source_mask_nodata: float | None,
    stream_mask_array: np.ndarray, stream_mask_nodata: float | None,
    source_contributing_area_array: np.ndarray, source_contributing_area_nodata: float | None,
    dx: float, dy: float, dfi_threshold: float, max_relative_rise_m: float,
    min_runout_source_area_m2: float, min_spread_radius_m: float, max_spread_radius_m: float,
    source_contributing_area_reference: float,
    input_paths: dict[str, str] | None = None, output_paths: dict[str, str] | None = None,
    geotransform: tuple[float, float, float, float, float, float] | None = None,
    crs_authority: str | None = None, crs_wkt: str = "", collect_debug_rasters: bool = False,
) -> NativeSpreadResult:
    
    rows, cols = runout_array.shape
    runout_valid = _valid_mask(runout_array, runout_nodata)
    dfi_valid = _valid_mask(dfi_array, dfi_nodata)
    dem_valid = _valid_mask(dem_array, dem_nodata)
    source_mask_valid = _valid_mask(source_mask_array, source_mask_nodata)
    stream_mask_valid = _valid_mask(stream_mask_array, stream_mask_nodata)
    sca_valid = _valid_mask(source_contributing_area_array, source_contributing_area_nodata)

    cell_area_m2 = dx * dy
    original_runout_mask = runout_valid & (runout_array > 0)
    source_mask_bool = source_mask_valid & (source_mask_array > 0)
    stream_mask_bool = stream_mask_valid & (stream_mask_array > 0)
    
    raw_thresh = min_runout_source_area_m2 / cell_area_m2
    area_removed = original_runout_mask & (source_contributing_area_array < raw_thresh)
    cleaned_runout = original_runout_mask & ~area_removed

    from scipy import ndimage
    labels, _ = ndimage.label(cleaned_runout, structure=np.ones((3,3)))
    anchors = cleaned_runout & source_mask_bool
    for r, c in EIGHT_NEIGHBOR_STEPS:
        s_r = slice(max(0, -r), min(rows, rows - r))
        s_c = slice(max(0, -c), min(cols, cols - c))
        t_r = slice(max(0, r), min(rows, rows + r))
        t_c = slice(max(0, c), min(cols, cols + c))
        anchors[s_r, s_c] |= cleaned_runout[s_r, s_c] & source_mask_bool[t_r, t_c]
    
    valid_labels = np.unique(labels[anchors])
    cleaned_runout = np.isin(labels, valid_labels[valid_labels != 0])
    removed_runout = area_removed | (original_runout_mask & ~cleaned_runout)

    dfi_eligible = cleaned_runout & dfi_valid & dem_valid & (dfi_array > dfi_threshold)
    candidate_seeds = dfi_eligible & ~source_mask_bool & ~stream_mask_bool
    edge_seeds = np.zeros_like(candidate_seeds)
    for r, c in EIGHT_NEIGHBOR_STEPS:
        s_r = slice(max(0, -r), min(rows, rows - r))
        s_c = slice(max(0, -c), min(cols, cols - c))
        t_r = slice(max(0, r), min(rows, rows + r))
        t_c = slice(max(0, c), min(cols, cols + c))
        edge_seeds[s_r, s_c] |= candidate_seeds[s_r, s_c] & (dem_valid[t_r, t_c] & ~cleaned_runout[t_r, t_c])

    spread_mask = np.zeros((rows, cols), dtype=np.uint8)
    min_elev_diff = np.full((rows, cols), FLOAT32_NODATA, dtype=np.float64)
    source_index = np.full((rows, cols), INT32_NODATA, dtype=np.int32)

    seed_r, seed_c = np.nonzero(edge_seeds)
    if len(seed_r) > 0:
        seed_sca = np.maximum(source_contributing_area_array[seed_r, seed_c], 0) * cell_area_m2
        seed_idx = np.clip(seed_sca / source_contributing_area_reference, 0, 1)
        seed_rad = min_spread_radius_m + seed_idx * (max_spread_radius_m - min_spread_radius_m)
        max_offsets = _compute_neighbor_offsets(dx, dy, max_spread_radius_m)
        
        for i in range(len(seed_r)):
            sr, sc = seed_r[i], seed_c[i]
            rad = seed_rad[i]
            selev = dem_array[sr, sc]
            for dr, dc, dist in max_offsets:
                if dist > rad: break
                nr, nc = sr + dr, sc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if cleaned_runout[nr, nc] or not dfi_valid[nr, nc] or dfi_array[nr, nc] <= dfi_threshold: continue
                    if not dem_valid[nr, nc]: continue
                    ediff = dem_array[nr, nc] - selev
                    if ediff > max_relative_rise_m: continue
                    
                    if source_index[nr, nc] == -1 or ediff < min_elev_diff[nr, nc]:
                        spread_mask[nr, nc] = 1
                        min_elev_diff[nr, nc] = ediff
                        source_index[nr, nc] = sr * cols + sc

    combined_mask = (cleaned_runout | (spread_mask > 0)).astype(np.uint8)

    summary = SpreadSummary(
        input_paths=input_paths or {}, output_paths=output_paths or {}, rows=rows, cols=cols,
        pixel_size=(dx, dy), geotransform=geotransform or (0,dx,0,0,0,-dy),
        crs_authority=crs_authority, crs_wkt=crs_wkt, dfi_threshold=dfi_threshold,
        max_relative_rise_m=max_relative_rise_m, min_spread_radius_m=min_spread_radius_m,
        max_spread_radius_m=max_spread_radius_m, source_contributing_area_reference=source_contributing_area_reference,
        spread_mode="runout_cleanup_edge_seed_source_contribution_relief_connected",
        min_runout_source_area_m2=min_runout_source_area_m2,
        original_runout_cells=int(np.count_nonzero(original_runout_mask)),
        area_removed_runout_cells=int(np.count_nonzero(area_removed)),
        disconnected_runout_cells=0, removed_runout_cells=int(np.count_nonzero(removed_runout)),
        removed_runout_percent=0.0, cleaned_runout_cells=int(np.count_nonzero(cleaned_runout)),
        valid_runout_cells=int(np.count_nonzero(cleaned_runout)), seed_cells=len(seed_r),
        seeds_with_zero_contributing_source_area=0, seeds_assigned_min_radius=0,
        seeds_assigned_max_or_near_max_radius=0, spread_cells_added=int(np.count_nonzero(spread_mask)),
        combined_footprint_cells=int(np.count_nonzero(combined_mask)),
        source_cells_processed=len(seed_r), candidate_evaluations=0, neighbor_offset_count=0,
        neighbor_offsets=[], metadata={"status": "Success"}
    )
    return NativeSpreadResult(
        spread_mask, combined_mask, edge_seeds.astype(np.uint8), cleaned_runout.astype(np.uint8),
        removed_runout.astype(np.uint8), min_elev_diff, source_index, None, None, None, summary
    )

def prepare_staged_output_bundle(params: PostDepositionalSpreadParams) -> StagedOutputBundle:
    token = uuid.uuid4().hex
    dirs = {}
    reps = {}
    for f in CORE_OUTPUT_PARAM_FIELDS + OPTIONAL_OUTPUT_PARAM_FIELDS:
        p = getattr(params, f)
        if not p: continue
        path = Path(p).resolve()
        sd = dirs.setdefault(path.parent, path.parent / f".step5-stage-{token}")
        sd.mkdir(parents=True, exist_ok=True)
        reps[f] = str(sd / path.name)
    staged = replace(params, **reps)
    s2o = {str(Path(getattr(staged, f)).resolve()): str(Path(getattr(params, f)).resolve()) for f in reps}
    return StagedOutputBundle(params, staged, s2o, tuple(dirs.values()))

def publish_staged_output_bundle(bundle: StagedOutputBundle) -> None:
    for staged, official in bundle.staged_to_official.items():
        if os.path.exists(staged):
            os.makedirs(os.path.dirname(official), exist_ok=True)
            shutil.move(staged, official)
    for d in bundle.stage_dirs: shutil.rmtree(d, ignore_errors=True)

def run_post_depositional_spread_native(params: PostDepositionalSpreadParams) -> dict[str, Any]:
    runout = _read_raster(params.runout_mask_raster, "runout mask")
    dfi = _read_raster(params.dfi_raster, "DFI")
    dem = _read_raster(params.dem_raster, "DEM")
    src_mask = _read_raster(params.source_mask_raster, "source mask")
    strm_mask = _read_raster(params.stream_mask_raster, "stream mask")
    sca = _read_raster(params.source_contributing_area_raster, "source contributing area")
    
    dx, dy = runout.geotransform[1], abs(runout.geotransform[5])
    
    res = run_post_depositional_spread_core(
        runout.array, runout.nodata, dfi.array, dfi.nodata, dem.array, dem.nodata,
        src_mask.array, src_mask.nodata, strm_mask.array, strm_mask.nodata,
        sca.array, sca.nodata, dx, dy, params.dfi_threshold, params.max_relative_rise_m,
        params.min_runout_source_area_m2, params.min_spread_radius_m, params.max_spread_radius_m,
        params.source_contributing_area_reference,
        geotransform=runout.geotransform, crs_wkt=runout.projection_wkt
    )
    
    _write_geotiff(params.output_spread_mask, runout, res.spread_mask, "uint8", 0)
    _write_geotiff(params.output_combined_mask, runout, res.combined_mask, "uint8", 0)
    _write_geotiff(params.output_seed_mask, runout, res.seed_mask, "uint8", 0)
    _write_geotiff(params.output_cleaned_runout_mask, runout, res.cleaned_runout_mask, "uint8", 0)
    _write_geotiff(params.output_removed_runout_mask, runout, res.removed_runout_mask, "uint8", 0)
    
    if params.output_spread_mask_shapefile: _write_mask_shapefile(params.output_spread_mask, params.output_spread_mask_shapefile, "spread")
    if params.output_combined_mask_shapefile: _write_mask_shapefile(params.output_combined_mask, params.output_combined_mask_shapefile, "combined")
    
    summary_dict = asdict(res.summary)
    with open(params.output_summary_json, "w") as f:
        json.dump(summary_dict, f, indent=2)
    return summary_dict

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    params = load_post_depositional_spread_params(args.config)
    bundle = prepare_staged_output_bundle(params)
    run_post_depositional_spread_native(bundle.staged_params)
    publish_staged_output_bundle(bundle)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())