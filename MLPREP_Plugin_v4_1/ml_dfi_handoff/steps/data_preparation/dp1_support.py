from __future__ import annotations

import csv
import importlib
import logging
import math
import os
import time
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
from osgeo import gdal, ogr, osr
from pyproj import CRS as PyProjCRS
import rasterio
from rasterio.windows import Window as RasterioWindow
from shapely import intersects_xy, wkt as shapely_wkt
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from dp1_models import ContourNode

LOG = logging.getLogger("step1_aobs_native")
_step1_core_engine: Any = None
CONTOUR_INTERVAL_M = 2.0


@dataclass
class Params:
    landslide_polygons: str
    polygon_id_field: str
    dem: str
    output_gpkg: str
    output_prefix: str = "step1"
    buffer_min_m: float = 10.0
    buffer_pixel_multiple: float = 2.0
    boundary_sample_step_m: float = 10.0
    crown_percentile_high: float = 90.0
    endpoint_snap_search_radius_px: int = 4
    qc_percentile_high: float = 90.0
    enable_early_polygon_filter: bool = True
    complexity_threshold: float = 1.0
    early_exclusion_area_threshold_m2: float = 250.0
    write_csv_summary: bool = True
    write_optional_layers: bool = True
    progress_every: int = 100
    log_level: str = "INFO"
    enable_parallel: bool = True
    max_workers: Optional[int] = 4
    worker_start_method: str = "spawn"
    preserve_order: bool = True
    fail_fast: bool = False
    worker_chunk_size: int = 1


@dataclass(frozen=True)
class LocalRasterWindow:
    array: np.ndarray
    valid_mask: np.ndarray
    origin_x: float
    origin_y: float
    pixel_width: float
    pixel_height: float
    row_off: int
    col_off: int
    projection_wkt: str = ""


@dataclass(frozen=True)
class LayerSpec:
    name: str
    rows: Sequence[Dict[str, Any]]
    columns: Sequence[str]
    geometry_type: int


@dataclass(frozen=True)
class ContourExtractionResult:
    nodes: tuple[ContourNode, ...]
    contour_generation_s: float
    node_extraction_s: float
    total_s: float


def configure_logging(level: str) -> None:
    LOG.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    if not LOG.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        LOG.addHandler(handler)


def log(msg: str, level: int = logging.INFO) -> None:
    LOG.log(level, msg)


def timer_start() -> float:
    return time.perf_counter()


def log_timing(label: str, t0: float) -> float:
    elapsed = float(time.perf_counter() - t0)
    log(f"[timing] {label}: {elapsed:.3f}s")
    return elapsed


def resolve_output_gpkg(path_value: str, prefix: str) -> str:
    out = str(path_value).strip()
    if not out:
        raise ValueError("output_gpkg/workspace is required.")
    if out.lower().endswith(".gpkg"):
        out_dir = os.path.dirname(out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        return out
    os.makedirs(out, exist_ok=True)
    return os.path.join(out, f"{prefix}.gpkg")


def ensure_qgis_runtime_only() -> None:
    executable = os.path.abspath(sys.executable)
    prefix = os.path.abspath(sys.prefix)

    def normalize(value: str) -> str:
        return os.path.normcase(os.path.abspath(value))

    expected_venv = normalize(
        str(Path(__file__).resolve().parents[2] / ".venv-dp1")
    )
    if normalize(prefix) != expected_venv:
        raise RuntimeError(
            "DP1 must run from its dedicated frozen QGIS-based environment. "
            "Use steps\\data_preparation\\launch_dp1.py --config <config.json>. "
            f"Detected executable={executable}; prefix={prefix}"
        )

    roots = {
        normalize(value)
        for value in (
            os.environ.get("QGIS_PREFIX_PATH", ""),
            os.environ.get("OSGEO4W_ROOT", ""),
            os.path.dirname(os.path.dirname(executable)),
            expected_venv,
            sys.prefix,
            sys.base_prefix,
        )
        if str(value).strip()
    }

    def origin_is_allowed(origin: str) -> bool:
        normalized_origin = normalize(origin)
        for root in roots:
            try:
                if os.path.commonpath([normalized_origin, root]) == root:
                    return True
            except (OSError, ValueError):
                continue
        return False

    offenders: list[str] = []
    for module_name in ("qgis", "osgeo", "numpy", "shapely", "rasterio", "pyproj", "matplotlib"):
        module = importlib.import_module(module_name)
        origin = os.path.abspath(str(getattr(module, "__file__", "")))
        valid_origin = bool(origin) and origin_is_allowed(origin)
        if not valid_origin:
            offenders.append(f"{module_name}={origin or '<unknown>'}")
    if offenders:
        raise RuntimeError(
            "QGIS-only runtime check failed; module origins are outside allowed QGIS roots. "
            + "; ".join(offenders)
        )


def load_step1_core_engine():
    global _step1_core_engine
    if _step1_core_engine is not None:
        return _step1_core_engine
    import dp1_engine as core_dp1

    _step1_core_engine = core_dp1
    return _step1_core_engine


def ensure_projected_meter_crs(crs_obj: Any) -> None:
    try:
        crs = PyProjCRS.from_user_input(crs_obj)
    except Exception as exc:
        raise RuntimeError(f"Could not parse CRS: {crs_obj}") from exc
    if not crs.is_projected:
        raise RuntimeError(f"DEM CRS must be projected. Received: {crs.to_string()}")
    axis_info = list(crs.axis_info or [])
    if axis_info:
        units = {str(a.unit_name or "").lower() for a in axis_info}
        if not any(("metre" in u) or ("meter" in u) for u in units):
            raise RuntimeError(f"DEM CRS units must be meters. Received units: {sorted(units)}")


def split_vector_source(source: str) -> Tuple[str, str]:
    raw = str(source).strip()
    if not raw:
        raise RuntimeError("Vector layer source is empty.")
    marker = "|layername="
    lower = raw.lower()
    idx = lower.find(marker)
    if idx < 0:
        return os.path.abspath(raw), ""
    path = raw[:idx]
    layer_name = raw[idx + len(marker) :].strip()
    return os.path.abspath(path), layer_name


def set_axis_mapping_if_supported(srs: Optional[osr.SpatialReference]) -> None:
    if srs is None:
        return
    if hasattr(srs, "SetAxisMappingStrategy") and hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
        try:
            srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        except Exception:
            pass


def open_vector_layer(source: str) -> Tuple[Any, Any, str]:
    vector_path, layer_name = split_vector_source(source)
    ds = ogr.Open(vector_path, 0)
    if ds is None:
        raise RuntimeError(f"Cannot open vector dataset: {vector_path}")
    if layer_name:
        layer = ds.GetLayerByName(layer_name)
        if layer is None:
            raise RuntimeError(f"Layer '{layer_name}' not found in vector dataset: {vector_path}")
    else:
        layer = ds.GetLayer(0)
        if layer is None:
            raise RuntimeError(f"No layers found in vector dataset: {vector_path}")
        layer_name = str(layer.GetName() or "")
    return ds, layer, layer_name


def spatial_ref_from_pyproj(crs: Any) -> osr.SpatialReference:
    srs = osr.SpatialReference()
    srs.ImportFromWkt(str(crs.to_wkt()))
    set_axis_mapping_if_supported(srs)
    return srs


def shapely_from_ogr(ogr_geom: Any) -> Any:
    if ogr_geom is None:
        return None
    geometry_wkt = ogr_geom.ExportToWkt()
    if not geometry_wkt:
        return None
    return shapely_wkt.loads(str(geometry_wkt))


def read_landslide_rows(
    source: str,
    polygon_id_field: str,
    target_crs: Any,
    target_crs_wkt: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Any]:
    _ds, layer, layer_name = open_vector_layer(source)
    layer_defn = layer.GetLayerDefn()
    if layer_defn.GetFieldIndex(str(polygon_id_field)) < 0:
        raise RuntimeError(
            f"polygon_id_field '{polygon_id_field}' is missing in landslide polygons ({source}; layer={layer_name})."
        )
    source_srs = layer.GetSpatialRef()
    if source_srs is None:
        raise RuntimeError("Landslide polygons have no CRS.")
    source_srs = source_srs.Clone()
    set_axis_mapping_if_supported(source_srs)
    if str(target_crs_wkt or "").strip():
        target_srs = osr.SpatialReference()
        target_srs.ImportFromWkt(str(target_crs_wkt))
        set_axis_mapping_if_supported(target_srs)
    else:
        target_srs = spatial_ref_from_pyproj(target_crs)
    transform = None
    try:
        same_crs = bool(source_srs.IsSame(target_srs))
    except Exception:
        same_crs = False
    if not same_crs:
        try:
            target_desc = PyProjCRS.from_user_input(target_crs).to_string()
        except Exception:
            target_desc = str(target_crs)
        log(f"Reprojecting landslide polygons to DEM CRS: {target_desc}")
        transform = osr.CoordinateTransformation(source_srs, target_srs)
    rows: List[Dict[str, Any]] = []
    layer.ResetReading()
    for feat in layer:
        pid_raw = feat.GetField(str(polygon_id_field))
        geom_ref = feat.GetGeometryRef()
        geom = None
        if geom_ref is not None:
            geom = geom_ref.Clone()
            if transform is not None:
                err = int(geom.Transform(transform))
                if err != 0:
                    geom = None
        rows.append({"polygon_id": str(pid_raw), "geometry": shapely_from_ogr(geom)})
    if not rows:
        raise RuntimeError("Input landslide layer is empty.")
    return rows, target_crs


def normalize_polygon_geometry(geom: Any) -> Optional[Tuple[Any, Polygon]]:
    if geom is None or geom.is_empty:
        return None
    g = geom
    if not g.is_valid:
        try:
            g = g.buffer(0)
        except Exception:
            return None
    if g is None or g.is_empty or not g.is_valid:
        return None
    if isinstance(g, Polygon):
        return g, g
    if isinstance(g, MultiPolygon):
        parts = [p for p in g.geoms if isinstance(p, Polygon) and (not p.is_empty) and p.area > 0]
        if not parts:
            return None
        return g, max(parts, key=lambda p_: p_.area)
    if isinstance(g, GeometryCollection):
        polys: List[Polygon] = []
        for part in g.geoms:
            if isinstance(part, Polygon) and (not part.is_empty) and part.area > 0:
                polys.append(part)
            elif isinstance(part, MultiPolygon):
                polys.extend([p for p in part.geoms if isinstance(p, Polygon) and (not p.is_empty) and p.area > 0])
        if not polys:
            return None
        if len(polys) == 1:
            return polys[0], polys[0]
        mp = MultiPolygon(polys)
        return mp, max(polys, key=lambda p_: p_.area)
    return None


class GdalTransform:
    def __init__(self, a: float, e: float) -> None:
        self.a = float(a)
        self.e = float(e)


class GdalRasterDataset:
    def __init__(self, raster_path: str, label: str = "Raster") -> None:
        try:
            self._ds = rasterio.open(raster_path, "r")
        except Exception as exc:
            raise RuntimeError(f"Cannot open {label}: {raster_path}") from exc
        if int(self._ds.count) < 1:
            raise RuntimeError(f"{label} band 1 is unavailable: {raster_path}")
        self.path = os.path.abspath(raster_path)
        self.label = str(label)
        self._gt = tuple(float(v) for v in self._ds.transform.to_gdal())
        if abs(self._gt[2]) > 1e-12 or abs(self._gt[4]) > 1e-12:
            raise RuntimeError(f"Rotated/sheared geotransforms are unsupported in Step 1 ({label}).")
        self.transform = GdalTransform(self._gt[1], self._gt[5])
        self.xsize = int(self._ds.width)
        self.ysize = int(self._ds.height)
        xmin = self._gt[0]
        ymax = self._gt[3]
        xmax = xmin + self._gt[1] * self.xsize
        ymin = ymax + self._gt[5] * self.ysize
        self.bounds = (float(xmin), float(ymin), float(xmax), float(ymax))
        self.nodata = self._ds.nodata
        wkt = str(self._ds.crs.to_wkt() if self._ds.crs is not None else "")
        self.crs_wkt = wkt
        self.crs = PyProjCRS.from_wkt(wkt) if wkt else None

    @property
    def geotransform(self) -> Tuple[float, float, float, float, float, float]:
        return (
            float(self._gt[0]),
            float(self._gt[1]),
            float(self._gt[2]),
            float(self._gt[3]),
            float(self._gt[4]),
            float(self._gt[5]),
        )

    def sample(self, coords: Sequence[Tuple[float, float]], indexes: int = 1) -> List[np.ndarray]:
        if int(indexes) != 1:
            raise RuntimeError("Step 1 DEM reader only supports band index 1.")
        return [
            np.asarray(value, dtype=np.float64)
            for value in self._ds.sample([(float(x), float(y)) for x, y in coords], indexes=1)
        ]

    def read_window(self, bounds: Tuple[float, float, float, float]) -> Optional[LocalRasterWindow]:
        minx, miny, maxx, maxy = [float(v) for v in bounds]
        col0 = int(math.floor((minx - self._gt[0]) / self._gt[1]))
        col1 = int(math.ceil((maxx - self._gt[0]) / self._gt[1]))
        row0 = int(math.floor((maxy - self._gt[3]) / self._gt[5]))
        row1 = int(math.ceil((miny - self._gt[3]) / self._gt[5]))
        col0 = max(0, min(self.xsize, col0))
        col1 = max(0, min(self.xsize, col1))
        row0 = max(0, min(self.ysize, row0))
        row1 = max(0, min(self.ysize, row1))
        width = int(col1 - col0)
        height = int(row1 - row0)
        if width <= 0 or height <= 0:
            return None
        try:
            window_factory = cast(Any, RasterioWindow)
            arr = self._ds.read(
                1,
                window=window_factory(int(col0), int(row0), int(width), int(height)),
            )
        except Exception:
            return None
        arr_f = np.asarray(arr, dtype=np.float64)
        valid_mask = np.isfinite(arr_f)
        if self.nodata is not None:
            valid_mask &= ~np.isclose(arr_f, float(self.nodata), atol=1e-8, rtol=0.0)
        origin_x = float(self._gt[0] + col0 * self._gt[1])
        origin_y = float(self._gt[3] + row0 * self._gt[5])
        return LocalRasterWindow(
            array=arr_f,
            valid_mask=np.asarray(valid_mask, dtype=bool),
            origin_x=origin_x,
            origin_y=origin_y,
            pixel_width=float(self._gt[1]),
            pixel_height=float(self._gt[5]),
            row_off=int(row0),
            col_off=int(col0),
            projection_wkt=str(self._ds.crs.to_wkt() if self._ds.crs is not None else ""),
        )

    def __enter__(self) -> "GdalRasterDataset":
        return self

    @property
    def closed(self) -> bool:
        return bool(self._ds.closed)

    def close(self) -> None:
        if not self.closed:
            self._ds.close()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


GdalDemDataset = GdalRasterDataset


def open_dem_dataset(dem_path: str) -> GdalRasterDataset:
    return GdalRasterDataset(dem_path, "DEM")


def open_raster_dataset(path: str, label: str) -> GdalRasterDataset:
    return GdalRasterDataset(path, label)


def validate_matching_raster_grid(reference: GdalRasterDataset, other: GdalRasterDataset, label: str) -> None:
    if int(reference.xsize) != int(other.xsize) or int(reference.ysize) != int(other.ysize):
        raise ValueError(
            f"{label} size mismatch: expected {reference.xsize}x{reference.ysize}, "
            f"got {other.xsize}x{other.ysize} ({other.path})."
        )
    if any(abs(float(reference.geotransform[i]) - float(other.geotransform[i])) > 1e-9 for i in range(6)):
        raise ValueError(f"{label} geotransform mismatch with DEM grid: {other.path}")
    if reference.crs is None or other.crs is None:
        raise ValueError(f"{label} must have a valid CRS: {other.path}")
    ref_srs = spatial_ref_from_pyproj(reference.crs)
    oth_srs = spatial_ref_from_pyproj(other.crs)
    if not bool(ref_srs.IsSame(oth_srs)):
        raise ValueError(f"{label} CRS mismatch with DEM grid: {other.path}")


def densify_boundary_points(poly: Polygon, step_m: float) -> List[Point]:
    ring = LineString(poly.exterior.coords)
    if ring.is_empty or ring.length <= 0:
        return []
    n_pts = max(3, int(math.ceil(float(ring.length) / float(step_m))) + 1)
    distances = np.linspace(0.0, float(ring.length), n_pts, dtype=np.float64)
    return [ring.interpolate(float(d)) for d in distances]


def sample_dem_points(dem_ds: GdalDemDataset, points: Sequence[Point], nodata: Optional[float]) -> List[Tuple[Point, Optional[float]]]:
    if not points:
        return []
    coords = [(float(pt.x), float(pt.y)) for pt in points]
    samples = list(dem_ds.sample(coords, indexes=1))
    out: List[Tuple[Point, Optional[float]]] = []
    for pt, arr in zip(points, samples):
        z: Optional[float] = None
        try:
            val = float(np.asarray(arr).reshape(-1)[0])
            if np.isfinite(val) and (nodata is None or not np.isclose(val, float(nodata), atol=1e-8, rtol=0.0)):
                z = float(val)
        except Exception:
            z = None
        out.append((pt, z))
    return out


def detect_toe(boundary_samples: Sequence[Tuple[Point, Optional[float]]]) -> Tuple[Optional[Point], Optional[float], str]:
    return load_step1_core_engine().detect_toe(boundary_samples)


def detect_crown(boundary_samples: Sequence[Tuple[Point, Optional[float]]], percentile_high: float) -> Tuple[Optional[Point], Optional[float], str]:
    return load_step1_core_engine().detect_crown(boundary_samples, percentile_high=float(percentile_high))


def complexity_ratio(poly: Polygon) -> Optional[float]:
    try:
        ombb = poly.minimum_rotated_rectangle
        if ombb is None or ombb.is_empty:
            return None
        return load_step1_core_engine().complexity_ratio_from_perimeters(float(poly.length), float(ombb.length))
    except Exception:
        return None


def window_cell_center_xy(window: LocalRasterWindow, row: int, col: int) -> Tuple[float, float]:
    x = float(window.origin_x) + (float(col) + 0.5) * float(window.pixel_width)
    y = float(window.origin_y) + (float(row) + 0.5) * float(window.pixel_height)
    return float(x), float(y)


def mask_polygon_to_window(poly: Any, window: LocalRasterWindow) -> np.ndarray:
    rows, cols = window.array.shape
    xs = float(window.origin_x) + (np.arange(cols, dtype=np.float64) + 0.5) * float(window.pixel_width)
    ys = float(window.origin_y) + (np.arange(rows, dtype=np.float64) + 0.5) * float(window.pixel_height)
    xx, yy = np.meshgrid(xs, ys)
    return np.asarray(intersects_xy(poly, xx, yy), dtype=bool)


def point_to_local_cell(window: LocalRasterWindow, point: Point) -> Tuple[int, int]:
    col = int(math.floor((float(point.x) - float(window.origin_x)) / float(window.pixel_width)))
    row = int(math.floor((float(point.y) - float(window.origin_y)) / float(window.pixel_height)))
    return int(row), int(col)


def snap_point_to_mask(point: Point, window: LocalRasterWindow, mask: np.ndarray, search_radius_px: int) -> Optional[Tuple[int, int]]:
    rows, cols = window.array.shape
    row0, col0 = point_to_local_cell(window, point)
    row0 = max(0, min(rows - 1, row0))
    col0 = max(0, min(cols - 1, col0))
    if bool(mask[row0, col0]):
        return int(row0), int(col0)
    best: Optional[Tuple[float, int, int]] = None
    for radius in range(1, int(search_radius_px) + 1):
        r_min = max(0, row0 - radius)
        r_max = min(rows - 1, row0 + radius)
        c_min = max(0, col0 - radius)
        c_max = min(cols - 1, col0 + radius)
        for row in range(r_min, r_max + 1):
            for col in range(c_min, c_max + 1):
                if not bool(mask[row, col]):
                    continue
                x, y = window_cell_center_xy(window, row, col)
                d2 = float((x - float(point.x)) ** 2 + (y - float(point.y)) ** 2)
                cand = (d2, int(row), int(col))
                if best is None or cand < best:
                    best = cand
        if best is not None:
            return int(best[1]), int(best[2])
    return None


def _create_local_masked_raster(
    window: LocalRasterWindow,
    mask: np.ndarray,
    nodata_value: float = -9999.0,
) -> Any:
    rows, cols = window.array.shape
    arr = np.ascontiguousarray(window.array, dtype=np.float32)
    valid = np.asarray(window.valid_mask, dtype=bool) & np.asarray(mask, dtype=bool)
    arr = arr.copy()
    arr[~valid] = np.float32(nodata_value)
    driver = gdal.GetDriverByName("MEM")
    if driver is None:
        raise RuntimeError("GDAL MEM raster driver is unavailable for Step 1 contour generation.")
    dataset = driver.Create("", int(cols), int(rows), 1, gdal.GDT_Float32)
    if dataset is None:
        raise RuntimeError("Could not create in-memory Step 1 contour raster.")
    dataset.SetGeoTransform(
        (
            float(window.origin_x),
            float(window.pixel_width),
            0.0,
            float(window.origin_y),
            0.0,
            float(window.pixel_height),
        )
    )
    if str(window.projection_wkt).strip():
        dataset.SetProjection(str(window.projection_wkt))
    band = dataset.GetRasterBand(1)
    if band is None:
        raise RuntimeError("In-memory Step 1 contour raster has no band 1.")
    band.SetNoDataValue(float(nodata_value))
    result = band.WriteRaster(
        0,
        0,
        int(cols),
        int(rows),
        arr.tobytes(order="C"),
        buf_xsize=int(cols),
        buf_ysize=int(rows),
        buf_type=gdal.GDT_Float32,
    )
    if int(result) != 0:
        raise RuntimeError("Could not populate in-memory Step 1 contour raster.")
    band.FlushCache()
    return dataset


def _iter_line_parts(geom: BaseGeometry) -> List[LineString]:
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom] if geom.length > 0.0 else []
    if isinstance(geom, MultiLineString):
        return [part for part in geom.geoms if isinstance(part, LineString) and part.length > 0.0]
    if isinstance(geom, GeometryCollection):
        out: List[LineString] = []
        for part in geom.geoms:
            if isinstance(part, LineString) and part.length > 0.0:
                out.append(part)
            elif isinstance(part, MultiLineString):
                out.extend([sub for sub in part.geoms if isinstance(sub, LineString) and sub.length > 0.0])
        return out
    return []


def extract_contour_nodes_from_buffered_window(
    window: LocalRasterWindow,
    buffer_mask: np.ndarray,
    landslide_polygon: Polygon,
    polygon_id: str,
    contour_interval_m: float = CONTOUR_INTERVAL_M,
) -> ContourExtractionResult:
    total_t0 = time.perf_counter()
    if float(contour_interval_m) <= 0.0:
        raise ValueError("contour_interval_m must be > 0.")
    if not np.any(np.asarray(buffer_mask, dtype=bool) & np.asarray(window.valid_mask, dtype=bool)):
        return ContourExtractionResult(
            nodes=(),
            contour_generation_s=0.0,
            node_extraction_s=0.0,
            total_s=float(time.perf_counter() - total_t0),
        )

    raster_ds = _create_local_masked_raster(window, buffer_mask)
    try:
        band = raster_ds.GetRasterBand(1)
        if band is None:
            raise RuntimeError("In-memory Step 1 contour raster has no band 1.")

        mem_drv = ogr.GetDriverByName("Memory") or ogr.GetDriverByName("MEM")
        if mem_drv is None:
            raise RuntimeError("OGR Memory driver is unavailable for Step 1 contour generation.")
        mem_ds = mem_drv.CreateDataSource(f"step1_contours_{uuid.uuid4().hex}")
        if mem_ds is None:
            raise RuntimeError("Could not create in-memory contour datasource for Step 1.")

        srs = None
        if str(window.projection_wkt).strip():
            srs = osr.SpatialReference()
            srs.ImportFromWkt(str(window.projection_wkt))
            set_axis_mapping_if_supported(srs)
        layer = mem_ds.CreateLayer("contours", srs, ogr.wkbLineString)
        layer.CreateField(ogr.FieldDefn("elev_m", ogr.OFTReal))

        contour_t0 = time.perf_counter()
        result = gdal.ContourGenerateEx(
            band,
            layer,
            [
                f"LEVEL_INTERVAL={float(contour_interval_m)}",
                "LEVEL_BASE=0.0",
                "ELEV_FIELD=elev_m",
                "NODATA=-9999.0",
            ],
        )
        contour_generation_s = float(time.perf_counter() - contour_t0)
        if int(result) != 0:
            raise RuntimeError(f"GDAL contour generation failed for polygon {polygon_id}.")

        nodes: List[ContourNode] = []
        node_t0 = time.perf_counter()
        layer.ResetReading()
        for feat in layer:
            elev_val = feat.GetField("elev_m")
            try:
                elev_m = float(elev_val)
            except Exception:
                continue
            geom = shapely_from_ogr(feat.GetGeometryRef())
            if geom is None or geom.is_empty:
                continue
            clipped = geom.intersection(landslide_polygon)
            for part in _iter_line_parts(clipped):
                midpoint = part.interpolate(float(part.length) / 2.0)
                if midpoint.is_empty:
                    continue
                nodes.append(
                    ContourNode(
                        xy=(float(midpoint.x), float(midpoint.y)),
                        elevation_m=float(elev_m),
                        source_length_2d_m=float(part.length),
                        segment_xy=tuple((float(coord[0]), float(coord[1])) for coord in list(part.coords)),
                    )
                )
        deduped: Dict[Tuple[int, int, int], ContourNode] = {}
        for node in nodes:
            key = (
                int(round(float(node.xy[0]) * 1000.0)),
                int(round(float(node.xy[1]) * 1000.0)),
                int(round(float(node.elevation_m) * 1000.0)),
            )
            deduped[key] = node
        node_extraction_s = float(time.perf_counter() - node_t0)
        return ContourExtractionResult(
            nodes=tuple(
                sorted(
                    deduped.values(),
                    key=lambda item: (-float(item.elevation_m), float(item.xy[0]), float(item.xy[1])),
                )
            ),
            contour_generation_s=float(contour_generation_s),
            node_extraction_s=float(node_extraction_s),
            total_s=float(time.perf_counter() - total_t0),
        )
    finally:
        raster_ds = None


def infer_ogr_field_type(rows: Sequence[Dict[str, Any]], field_name: str) -> int:
    for row in rows:
        value = row.get(field_name)
        if value is None:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool):
            return ogr.OFTInteger
        if isinstance(value, (int, np.integer)):
            return ogr.OFTInteger64
        if isinstance(value, (float, np.floating)):
            return ogr.OFTReal
        return ogr.OFTString
    return ogr.OFTString


def set_feature_field(feature: Any, name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        feature.SetField(name, int(value))
    elif isinstance(value, (int, float, str)):
        feature.SetField(name, value)
    else:
        feature.SetField(name, str(value))


def coerce_geometry_for_layer(geom: Any, geometry_type: int) -> Any:
    if geom is None or geom.is_empty:
        return None
    if geometry_type == ogr.wkbPoint:
        return geom if isinstance(geom, Point) else None
    if geometry_type == ogr.wkbMultiLineString:
        if isinstance(geom, LineString):
            return MultiLineString([geom])
        if isinstance(geom, MultiLineString):
            return geom
        return None
    if geometry_type == ogr.wkbMultiPolygon:
        if isinstance(geom, Polygon):
            return MultiPolygon([geom])
        if isinstance(geom, MultiPolygon):
            return geom
        return None
    return geom


def write_gpkg_layers(output_gpkg: str, layer_specs: Sequence[LayerSpec], crs: Any) -> None:
    drv = ogr.GetDriverByName("GPKG")
    if drv is None:
        raise RuntimeError("GPKG driver is unavailable for Step 1 output writing.")
    if os.path.exists(output_gpkg):
        err = int(drv.DeleteDataSource(output_gpkg))
        if err != 0:
            raise RuntimeError(f"Failed deleting existing output gpkg: {output_gpkg}")
    ds = drv.CreateDataSource(output_gpkg)
    if ds is None:
        raise RuntimeError(f"Failed creating output gpkg: {output_gpkg}")
    out_srs = spatial_ref_from_pyproj(crs)
    try:
        for spec in layer_specs:
            layer = ds.CreateLayer(str(spec.name), out_srs, int(spec.geometry_type))
            if layer is None:
                raise RuntimeError(f"Failed creating output layer: {spec.name}")
            for col in spec.columns:
                field_type = infer_ogr_field_type(spec.rows, str(col))
                field_def = ogr.FieldDefn(str(col), field_type)
                if field_type == ogr.OFTReal:
                    field_def.SetWidth(24)
                    field_def.SetPrecision(8)
                elif field_type == ogr.OFTString:
                    field_def.SetWidth(254)
                if int(layer.CreateField(field_def)) != 0:
                    raise RuntimeError(f"Failed creating field {col!r} in output layer {spec.name!r}.")
            defn = layer.GetLayerDefn()
            transaction_started = int(layer.StartTransaction()) == 0
            try:
                for row in spec.rows:
                    feat = ogr.Feature(defn)
                    for col in spec.columns:
                        set_feature_field(feat, str(col), row.get(col))
                    geom = coerce_geometry_for_layer(row.get("geometry"), int(spec.geometry_type))
                    if geom is not None:
                        ogr_geom = ogr.CreateGeometryFromWkt(str(geom.wkt))
                        feat.SetGeometry(ogr_geom)
                    if int(layer.CreateFeature(feat)) != 0:
                        raise RuntimeError(f"Failed writing a feature to output layer {spec.name!r}.")
                    feat = None
                if transaction_started and int(layer.CommitTransaction()) != 0:
                    raise RuntimeError(f"Failed committing output layer {spec.name!r}.")
            except Exception:
                if transaction_started:
                    layer.RollbackTransaction()
                raise
    finally:
        ds = None


def write_qc_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "polygon_id",
                "crown_id",
                "polygon_area_m2",
                "complexity_ratio",
                "early_excluded",
                "path_method",
                "fallback_triggered",
                "diagnostic_reason",
                "H",
                "h_over_l",
                "L_2d",
                "L_3d",
                "a_obs_deg",
                "outside_mask_steps",
                "outside_mask_len_2d",
                "outside_mask_len_3d",
                "outside_mask_fraction_2d",
                "outside_mask_fraction_3d",
                "max_consecutive_outside_steps",
                "tortuosity_2d",
                "qc_flag",
                "qc_reason",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.get("polygon_id"),
                    row.get("crown_id"),
                    row.get("polygon_area_m2"),
                    row.get("complexity_ratio"),
                    row.get("early_excluded"),
                    row.get("path_method"),
                    row.get("fallback_triggered"),
                    row.get("diagnostic_reason"),
                    row.get("H"),
                    row.get("h_over_l"),
                    row.get("L_2d"),
                    row.get("L_3d"),
                    row.get("a_obs_deg"),
                    row.get("outside_mask_steps"),
                    row.get("outside_mask_len_2d"),
                    row.get("outside_mask_len_3d"),
                    row.get("outside_mask_fraction_2d"),
                    row.get("outside_mask_fraction_3d"),
                    row.get("max_consecutive_outside_steps"),
                    row.get("tortuosity_2d"),
                    row.get("qc_flag"),
                    row.get("qc_reason"),
                ]
            )


def validate_params(p: Params) -> None:
    landslide_path, _ = split_vector_source(p.landslide_polygons)
    if not os.path.isfile(landslide_path):
        raise RuntimeError(f"Landslide polygons not found: {p.landslide_polygons}")
    if not os.path.isfile(p.dem):
        raise RuntimeError(f"DEM not found: {p.dem}")
    if p.buffer_min_m <= 0 or p.buffer_pixel_multiple <= 0 or p.boundary_sample_step_m <= 0:
        raise ValueError("Buffer and boundary sampling parameters must be > 0.")
    if not (0 < p.crown_percentile_high < 100) or not (0 < p.qc_percentile_high < 100):
        raise ValueError("Percentile parameters must be in (0,100).")
    if p.endpoint_snap_search_radius_px <= 0:
        raise ValueError("endpoint_snap_search_radius_px must be > 0.")
    if p.early_exclusion_area_threshold_m2 <= 0:
        raise ValueError("early_exclusion_area_threshold_m2 must be > 0.")
    if p.progress_every <= 0:
        raise ValueError("progress_every must be > 0.")
    if p.max_workers is not None and int(p.max_workers) <= 0:
        raise ValueError("max_workers must be > 0 when provided.")
    if int(p.worker_chunk_size) <= 0:
        raise ValueError("worker_chunk_size must be > 0.")
    if not str(p.worker_start_method).strip():
        raise ValueError("worker_start_method is required.")
    if not p.output_prefix.strip():
        raise ValueError("output_prefix is required.")


def fmt_qc_reason(reasons: Set[str]) -> str:
    return load_step1_core_engine().format_qc_reason(reasons)
