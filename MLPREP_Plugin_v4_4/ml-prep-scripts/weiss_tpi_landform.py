#weiss_tpi_landform.py
#!/usr/bin/env python3
"""
Generate Weiss-style TPI landform classes from an input DEM GeoTIFF.

Implements Andrew D. Weiss-style TPI analysis with:
- Annulus-neighborhood TPI (small and large scale)
- Horn slope in degrees
- Optional standardized integer TPIs (100 units = 1 std dev)
- 6-class slope position output
- 10-class landform output using both small and large standardized TPIs
"""

from __future__ import annotations

import argparse
import math
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.windows import Window
from scipy.signal import convolve2d, fftconvolve

STD_INT_NODATA = np.int16(-32768)
# Preserve -32768 for nodata; valid standardized values clip to [-32767, 32767].
STD_INT_WRITE_MIN = int(np.iinfo(np.int16).min) + 1
STD_INT_WRITE_MAX = int(np.iinfo(np.int16).max)
CLASS_NODATA = np.uint8(0)
FLOAT32_NODATA_FALLBACK = float(np.finfo(np.float32).min)
TPI_TILE_MEMORY_WARN_MIB = 512.0
LANDFORM10_CLASS_NAMES: dict[int, str] = {
    1: "canyons, deeply incised streams",
    2: "midslope drainages, shallow valleys",
    3: "upland drainages, headwaters",
    4: "U-shape valleys",
    5: "plains",
    6: "open slopes",
    7: "upper slopes, mesas",
    8: "local ridges/hills in valleys",
    9: "midslope ridges, small hills in plains",
    10: "mt tops, high ridges",
}
SLOPE6_CLASS_NAMES: dict[int, str] = {
    1: "valleys <= -1.0 STDEV",
    2: "lower slopes > -1.0 STDEV and <= -0.5 STDEV",
    3: "flats > -0.5 STDEV and < 0.5 STDEV, slope <= 5 deg",
    4: "middle slope > -0.5 STDEV and < 0.5 STDEV, slope > 5 deg",
    5: "upper slope > 0.5 STDEV and <= 1 STDEV",
    6: "ridge > +1 STDEV",
}

HORN_DX_KERNEL = np.array(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    dtype=np.float32,
)
HORN_DY_KERNEL = np.array(
    [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
    dtype=np.float32,
)
NEIGHBOR_3X3 = np.ones((3, 3), dtype=np.float32)


@dataclass
class RunningStats:
    """Streaming statistics for valid raster cells."""

    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def update(self, values: np.ndarray, valid: np.ndarray) -> None:
        """Update running stats from values where valid is True."""
        if values.shape != valid.shape:
            raise ValueError("values and valid masks must have the same shape.")
        if not np.any(valid):
            return

        v = values[valid].astype(np.float64, copy=False)
        self.count += int(v.size)
        self.total += float(v.sum())
        self.total_sq += float((v * v).sum())

    def mean_std(self) -> tuple[float, float]:
        """Return population mean and std dev."""
        if self.count == 0:
            raise ValueError("No valid cells were available to compute statistics.")
        mean = self.total / self.count
        variance = max((self.total_sq / self.count) - (mean * mean), 0.0)
        std = math.sqrt(variance)
        if std == 0.0:
            raise ValueError("Standard deviation is zero; cannot standardize TPI.")
        return mean, std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate TPI, standardized TPI, slope, and Weiss-style "
            "6-class/10-class landform rasters from a DEM GeoTIFF."
        )
    )
    parser.add_argument("dem", type=Path, help="Input DEM GeoTIFF path.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: DEM parent folder).",
    )
    parser.add_argument(
        "--small-irad",
        type=int,
        default=30,
        help="Small-scale annulus inner radius in cells (default: 30).",
    )
    parser.add_argument(
        "--small-orad",
        type=int,
        default=60,
        help="Small-scale annulus outer radius in cells (default: 60).",
    )
    parser.add_argument(
        "--large-irad",
        type=int,
        default=372,
        help="Large-scale annulus inner radius in cells (default: 372).",
    )
    parser.add_argument(
        "--large-orad",
        type=int,
        default=402,
        help="Large-scale annulus outer radius in cells (default: 402).",
    )
    parser.add_argument(
        "--flat-threshold",
        type=float,
        default=5.0,
        help="Flat slope threshold in degrees (default: 5.0).",
    )
    parser.add_argument(
        "--slope-position-scale",
        choices=("small", "large"),
        default="small",
        help="Which TPI scale to use for 6-class slope position (default: small).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help=(
            "Base tile size in pixels for block processing (default: 1024). "
            "For TPI, effective tile size may be reduced per scale to honor "
            "--max-padded-side."
        ),
    )
    parser.add_argument(
        "--max-padded-side",
        type=int,
        default=1536,
        help=(
            "Maximum padded tile side for TPI FFT processing (default: 1536). "
            "Effective TPI tile size is reduced so tile + 2*outer_radius does "
            "not exceed this limit."
        ),
    )
    parser.add_argument(
        "--mean-zero-tolerance",
        type=float,
        default=None,
        help=(
            "Legacy alias for --mean-tol (ratio applied to std). "
            "If provided, it overrides --mean-tol."
        ),
    )
    parser.add_argument(
        "--require-mean-near-zero",
        action="store_true",
        help=(
            "Require each TPI mean to be near zero before standardization. "
            "If abs(mean) > (--mean-tol * std), abort unless --force is set."
        ),
    )
    parser.add_argument(
        "--mean-tol",
        type=float,
        default=0.1,
        help=(
            "Tolerance ratio for mean-near-zero check: require abs(mean) <= "
            "mean_tol * std when --require-mean-near-zero is enabled "
            "(default: 0.1)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Proceed even when --require-mean-near-zero fails. "
            "A warning is printed instead of aborting."
        ),
    )
    parser.add_argument(
        "--landform-classification",
        choices=("none", "6", "10", "both"),
        default="both",
        help=(
            "Select landform outputs to write: none, 6, 10, or both "
            "(default: both)."
        ),
    )
    parser.add_argument(
        "--landform10-out",
        type=Path,
        default=Path("landform_10class.tif"),
        help=(
            "Output path for 10-class landform raster. Relative paths are "
            "resolved under --out-dir (default: landform_10class.tif)."
        ),
    )
    parser.add_argument(
        "--reuse-existing-tpi",
        action="store_true",
        help=(
            "Reuse existing tpi_small.tif and tpi_large.tif in --out-dir when "
            "they match DEM dimensions/georeference."
        ),
    )
    parser.add_argument(
        "--no-write-stdint",
        "--no-standardize",
        dest="standardize",
        action="store_false",
        help=(
            "Skip writing standardized integer TPI rasters. "
            "Classification still uses standardized TPIs in memory. "
            "--no-standardize is retained as a backward-compatible alias."
        ),
    )
    parser.set_defaults(standardize=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dem.exists():
        raise FileNotFoundError(f"Input DEM not found: {args.dem}")
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be > 0.")
    if args.max_padded_side <= 0:
        raise ValueError("--max-padded-side must be > 0.")
    for name, irad, orad in [
        ("small", args.small_irad, args.small_orad),
        ("large", args.large_irad, args.large_orad),
    ]:
        if irad < 0:
            raise ValueError(f"{name} inner radius must be >= 0.")
        if orad <= 0:
            raise ValueError(f"{name} outer radius must be > 0.")
        if irad >= orad:
            raise ValueError(f"{name} inner radius must be < outer radius.")
    if args.flat_threshold < 0:
        raise ValueError("--flat-threshold must be >= 0.")
    if args.mean_tol < 0:
        raise ValueError("--mean-tol must be >= 0.")
    if args.mean_zero_tolerance is not None and args.mean_zero_tolerance < 0:
        raise ValueError("--mean-zero-tolerance must be >= 0.")


def create_annulus_kernel(irad: int, orad: int) -> np.ndarray:
    """Create a binary annulus kernel with irad < distance <= orad."""
    yy, xx = np.ogrid[-orad : orad + 1, -orad : orad + 1]
    dist2 = (xx * xx) + (yy * yy)
    ring = (dist2 <= (orad * orad)) & (dist2 > (irad * irad))
    if not np.any(ring):
        raise ValueError(f"Annulus kernel is empty for irad={irad}, orad={orad}.")
    return ring.astype(np.float32)


def effective_tpi_tile_size(base_tile_size: int, pad: int, max_padded_side: int) -> int:
    """
    Return TPI tile size constrained by padded FFT side length.

    Ensures tile_size + (2 * pad) <= max_padded_side.
    """
    max_core = max_padded_side - (2 * pad)
    if max_core <= 0:
        raise ValueError(
            "--max-padded-side is too small for the selected outer radius: "
            f"{max_padded_side} <= 2 * {pad}. Increase --max-padded-side or "
            "reduce annulus outer radius."
        )
    return min(base_tile_size, max_core)


def bytes_to_mib(num_bytes: int) -> float:
    """Convert byte count to MiB."""
    return num_bytes / (1024.0 * 1024.0)


def estimate_tpi_tile_memory(
    tile_size: int,
    pad: int,
    float_arrays: int = 6,
) -> tuple[int, int, int, int]:
    """
    Estimate padded tile dimensions and float32 memory usage.

    Returns:
        padded_side, padded_cells, per_float32_array_bytes, total_float32_bytes
    """
    padded_side = tile_size + (2 * pad)
    padded_cells = padded_side * padded_side
    per_float32_array_bytes = padded_cells * np.dtype(np.float32).itemsize
    total_float32_bytes = per_float32_array_bytes * float_arrays
    return padded_side, padded_cells, per_float32_array_bytes, total_float32_bytes


def print_tpi_memory_estimate(scale_name: str, tile_size: int, pad: int) -> None:
    """Print padded tile and float32 memory estimates for one TPI scale."""
    padded_side, padded_cells, one_array_b, total_b = estimate_tpi_tile_memory(
        tile_size=tile_size,
        pad=pad,
    )
    one_array_mib = bytes_to_mib(one_array_b)
    total_mib = bytes_to_mib(total_b)
    print(
        f"{scale_name} TPI padded tile: {padded_side}x{padded_side} "
        f"({padded_cells} cells)."
    )
    print(
        "  Estimated float32 memory: "
        f"{one_array_mib:.1f} MiB per array, ~{total_mib:.1f} MiB total "
        "(approximate; FFT workspace not included)."
    )
    if total_mib >= TPI_TILE_MEMORY_WARN_MIB:
        print(
            "Warning: high estimated TPI tile memory for "
            f"{scale_name.lower()} scale (~{total_mib:.1f} MiB). "
            "Consider reducing --tile-size or --max-padded-side."
        )


def checked_mean_std(stats: RunningStats, scale_name: str) -> tuple[float, float]:
    """Return mean/std with clearer error messages for common failure modes."""
    try:
        return stats.mean_std()
    except ValueError as err:
        msg = str(err)
        if "Standard deviation is zero" in msg:
            raise ValueError(
                f"{scale_name} TPI std == 0. DEM may be effectively flat at this scale; "
                "change annulus radii or verify DEM values."
            ) from err
        if "No valid cells were available" in msg:
            raise ValueError(
                f"{scale_name} TPI has no valid cells. Check DEM nodata/mask coverage "
                "and annulus settings."
            ) from err
        raise


def ensure_mean_near_zero_or_force(
    scale_name: str,
    mean: float,
    std: float,
    tolerance_ratio: float,
    force: bool,
) -> None:
    """
    Enforce mean-near-zero requirement, with optional force override.

    Raises:
        ValueError when requirement fails and force is False.
    """
    max_abs_mean = tolerance_ratio * std
    if abs(mean) <= max_abs_mean:
        return

    msg = (
        f"{scale_name} TPI mean is not near zero for standardization: "
        f"abs(mean)={abs(mean):.6f} > {max_abs_mean:.6f} "
        f"(mean_tol={tolerance_ratio:.3f} * std={std:.6f})."
    )
    if force:
        print(f"Warning: {msg} Proceeding because --force is set.")
        return
    raise ValueError(f"{msg} Re-run with --force to proceed.")


def resolve_output_path(base_dir: Path, out_path: Path) -> Path:
    """Resolve an output path; relative paths are anchored under base_dir."""
    return out_path if out_path.is_absolute() else (base_dir / out_path)


def raster_matches_reference(candidate: Path, ref: DatasetReader) -> bool:
    """Return True if candidate raster matches reference dimensions/georeference."""
    if not candidate.exists():
        return False
    try:
        with rasterio.open(candidate) as src:
            return (
                src.count == 1
                and src.width == ref.width
                and src.height == ref.height
                and src.transform == ref.transform
                and src.crs == ref.crs
            )
    except Exception:
        return False


def compute_raster_stats(src: DatasetReader, tile_size: int) -> RunningStats:
    """Compute streaming stats from an existing raster using valid finite cells."""
    stats = RunningStats()
    for win in iter_windows(src.height, src.width, tile_size):
        ma = src.read(1, window=win, masked=True)
        arr = np.asarray(ma.data, dtype=np.float32)
        valid = (~np.ma.getmaskarray(ma)) & np.isfinite(arr)
        stats.update(arr, valid)
    return stats


def iter_windows(height: int, width: int, tile_size: int) -> Iterator[Window]:
    """Yield raster windows in row-major order."""
    for row_off in range(0, height, tile_size):
        win_h = min(tile_size, height - row_off)
        for col_off in range(0, width, tile_size):
            win_w = min(tile_size, width - col_off)
            yield Window.from_slices(
                rows=(row_off, row_off + win_h),
                cols=(col_off, col_off + win_w),
            )


def padded_window(win: Window, pad: int) -> Window:
    """Return core window expanded by pad on all sides."""
    row_off = int(win.row_off)
    col_off = int(win.col_off)
    height = int(win.height)
    width = int(win.width)
    return Window.from_slices(
        rows=(row_off - pad, row_off + height + pad),
        cols=(col_off - pad, col_off + width + pad),
        boundless=True,
    )


def as_valid_mask(masked: np.ma.MaskedArray) -> np.ndarray:
    """Return boolean mask of valid finite cells."""
    data = np.asarray(masked.data)
    return (~np.ma.getmaskarray(masked)) & np.isfinite(data)


def fill_float_nodata(arr: np.ndarray, valid: np.ndarray, nodata: float) -> np.ndarray:
    """Fill invalid cells with float nodata value."""
    out = np.array(arr, dtype=np.float32, copy=True)
    if np.isnan(nodata):
        out[~valid] = np.nan
    else:
        out[~valid] = nodata
    return out


def resolve_float_nodata(src_nodata: float | None) -> float:
    """Resolve output float nodata to a finite float32 value."""
    if src_nodata is None or not np.isfinite(src_nodata):
        return FLOAT32_NODATA_FALLBACK

    nodata32 = float(np.float32(src_nodata))
    if not np.isfinite(nodata32):
        return FLOAT32_NODATA_FALLBACK
    return nodata32


def output_profile(src: DatasetReader, dtype: str, nodata: float | int | None) -> dict:
    """Create output GTiff profile preserving georeference."""
    out_dtype = np.dtype(dtype)
    if np.issubdtype(out_dtype, np.floating):
        profile_nodata: float | int = resolve_float_nodata(
            float(nodata) if nodata is not None else None
        )
    else:
        if nodata is None:
            raise ValueError("Integer output profile requires a finite nodata value.")
        profile_nodata = int(nodata)

    profile = src.profile.copy()
    profile.update(
        count=1,
        dtype=dtype,
        nodata=profile_nodata,
        compress="deflate",
        BIGTIFF="IF_SAFER",
    )
    if np.issubdtype(out_dtype, np.floating):
        profile.update(predictor=3)
    elif out_dtype.itemsize > 1:
        profile.update(predictor=2)
    return profile


def compute_tpi_tile(
    dem_tile: np.ma.MaskedArray,
    kernel: np.ndarray,
    pad: int,
    core_h: int,
    core_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute TPI for one padded tile and return core tile + valid mask."""
    dem_data = dem_tile.filled(0.0).astype(np.float32, copy=False)
    valid = as_valid_mask(dem_tile)
    valid_f = valid.astype(np.float32, copy=False)

    conv_sum = fftconvolve(dem_data, kernel, mode="same").astype(np.float32, copy=False)
    conv_count = fftconvolve(valid_f, kernel, mode="same").astype(np.float32, copy=False)

    mean_annulus = np.zeros_like(conv_sum, dtype=np.float32)
    np.divide(conv_sum, conv_count, out=mean_annulus, where=(conv_count > 0.0))

    tpi = np.full(dem_data.shape, np.nan, dtype=np.float32)
    good = valid & (conv_count > 0.0)
    tpi[good] = dem_data[good] - mean_annulus[good]

    yslice = slice(pad, pad + core_h)
    xslice = slice(pad, pad + core_w)
    return tpi[yslice, xslice], good[yslice, xslice]


def compute_slope_tile(
    dem_tile: np.ma.MaskedArray,
    xres: float,
    yres: float,
    pad: int,
    core_h: int,
    core_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Horn slope for one padded tile and return core slope + valid mask.

    Valid slope requires a fully valid 3x3 neighborhood around each center cell.
    """
    dem_data = dem_tile.filled(0.0).astype(np.float32, copy=False)
    valid = as_valid_mask(dem_tile)

    gx = convolve2d(dem_data, HORN_DX_KERNEL, mode="same", boundary="fill", fillvalue=0).astype(
        np.float32, copy=False
    )
    gy = convolve2d(dem_data, HORN_DY_KERNEL, mode="same", boundary="fill", fillvalue=0).astype(
        np.float32, copy=False
    )

    valid_count = convolve2d(
        valid.astype(np.float32, copy=False),
        NEIGHBOR_3X3,
        mode="same",
        boundary="fill",
        fillvalue=0,
    )
    # Conservative mask: keep slope only where all 9 cells in the 3x3 are valid.
    good = valid & (valid_count == 9.0)

    slope = np.full(dem_data.shape, np.nan, dtype=np.float32)
    if np.any(good):
        dzdx = gx / (8.0 * xres)
        dzdy = gy / (8.0 * yres)
        slope_all = np.degrees(np.arctan(np.sqrt((dzdx * dzdx) + (dzdy * dzdy))))
        slope[good] = slope_all[good].astype(np.float32, copy=False)

    yslice = slice(pad, pad + core_h)
    xslice = slice(pad, pad + core_w)
    return slope[yslice, xslice], good[yslice, xslice]


def compute_tpi_raster(
    src: DatasetReader,
    out_path: Path,
    kernel: np.ndarray,
    pad: int,
    tile_size: int,
    float_nodata: float,
) -> RunningStats:
    """Compute a TPI raster with tiled FFT convolution and return running stats."""
    stats = RunningStats()
    profile = output_profile(src, dtype="float32", nodata=float_nodata)

    with rasterio.open(out_path, "w", **profile) as dst:
        for win in iter_windows(src.height, src.width, tile_size):
            pwin = padded_window(win, pad)
            dem_tile = src.read(1, window=pwin, boundless=True, masked=True)

            core_h = int(win.height)
            core_w = int(win.width)
            tpi_core, valid_core = compute_tpi_tile(dem_tile, kernel, pad, core_h, core_w)

            stats.update(tpi_core, valid_core)
            out = fill_float_nodata(tpi_core, valid_core, float_nodata)
            dst.write(out.astype(np.float32, copy=False), 1, window=win)

    return stats


def compute_slope_raster(
    src: DatasetReader,
    out_path: Path,
    tile_size: int,
    float_nodata: float,
    xres: float,
    yres: float,
) -> None:
    """Compute Horn slope in degrees and write float32 raster."""
    pad = 1
    profile = output_profile(src, dtype="float32", nodata=float_nodata)

    with rasterio.open(out_path, "w", **profile) as dst:
        for win in iter_windows(src.height, src.width, tile_size):
            pwin = padded_window(win, pad)
            dem_tile = src.read(1, window=pwin, boundless=True, masked=True)

            core_h = int(win.height)
            core_w = int(win.width)
            slope_core, valid_core = compute_slope_tile(dem_tile, xres, yres, pad, core_h, core_w)

            out = fill_float_nodata(slope_core, valid_core, float_nodata)
            dst.write(out.astype(np.float32, copy=False), 1, window=win)


def to_std_units(tpi: np.ndarray, valid: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Standardize TPI and convert to integer units (100 units = 1 std dev).

    Uses symmetric rounding to nearest integer for both positive and
    negative values. Returns int32 so large magnitudes can be clipped
    safely at write-time when exporting int16 rasters.
    """
    out = np.zeros(tpi.shape, dtype=np.int32)
    if not np.any(valid):
        return out
    z100 = ((tpi[valid] - mean) / std) * 100.0
    rounded = np.where(z100 >= 0.0, np.floor(z100 + 0.5), np.ceil(z100 - 0.5))
    out[valid] = rounded.astype(np.int32, copy=False)
    return out


def classify_slope_position_6(
    tp: np.ndarray,
    slope_deg: np.ndarray,
    valid: np.ndarray,
    flat_threshold: float,
) -> np.ndarray:
    """Classify 6-class slope position using standardized TPI units and slope."""
    cls = np.zeros(tp.shape, dtype=np.uint8)

    m = valid & (tp <= -100)
    cls[m] = 1

    m = valid & (tp > -100) & (tp <= -50)
    cls[m] = 2

    m = valid & (tp > -50) & (tp < 50) & (slope_deg <= flat_threshold)
    cls[m] = 3

    m = valid & (tp > -50) & (tp < 50) & (slope_deg > flat_threshold)
    cls[m] = 4

    m = valid & (tp >= 50) & (tp < 100)
    cls[m] = 5

    m = valid & (tp >= 100)
    cls[m] = 6

    return cls


def classify_landform_10class(
    tp_small_std: np.ndarray,
    tp_large_std: np.ndarray,
    slope_deg: np.ndarray,
    valid: np.ndarray,
    flat_threshold: float,
) -> np.ndarray:
    """
    Classify Weiss-style 10-class landforms from small/large standardized TPIs.

    Uses the requested condition order exactly, including central-zone split:
      class 5 when slope <= flat_threshold
      class 6 when slope >= flat_threshold + 1
    """
    cls = np.full(tp_small_std.shape, STD_INT_NODATA, dtype=np.int16)
    steep_threshold = flat_threshold + 1.0

    m = (
        valid
        & (tp_small_std > -100)
        & (tp_small_std < 100)
        & (tp_large_std > -100)
        & (tp_large_std < 100)
        & (slope_deg <= flat_threshold)
    )
    cls[m] = 5

    m = (
        valid
        & (tp_small_std > -100)
        & (tp_small_std < 100)
        & (tp_large_std > -100)
        & (tp_large_std < 100)
        & (slope_deg >= steep_threshold)
    )
    cls[m] = 6

    m = (
        valid
        & (tp_small_std > -100)
        & (tp_small_std < 100)
        & (tp_large_std >= 100)
    )
    cls[m] = 7

    m = (
        valid
        & (tp_small_std > -100)
        & (tp_small_std < 100)
        & (tp_large_std <= -100)
    )
    cls[m] = 4

    m = (
        valid
        & (tp_small_std <= -100)
        & (tp_large_std > -100)
        & (tp_large_std < 100)
    )
    cls[m] = 2

    m = (
        valid
        & (tp_small_std >= 100)
        & (tp_large_std > -100)
        & (tp_large_std < 100)
    )
    cls[m] = 9

    m = valid & (tp_small_std <= -100) & (tp_large_std >= 100)
    cls[m] = 3

    m = valid & (tp_small_std <= -100) & (tp_large_std <= -100)
    cls[m] = 1

    m = valid & (tp_small_std >= 100) & (tp_large_std >= 100)
    cls[m] = 10

    m = valid & (tp_small_std >= 100) & (tp_large_std <= -100)
    cls[m] = 8

    return cls


def write_standardized_and_landforms(
    tpi_small_path: Path,
    tpi_large_path: Path,
    slope_path: Path,
    tpi_small_mean: float,
    tpi_small_std: float,
    tpi_large_mean: float,
    tpi_large_std: float,
    out_tpi_small_stdint: Path,
    out_tpi_large_stdint: Path,
    out_slope6: Path,
    out_landform10: Path,
    tile_size: int,
    flat_threshold: float,
    slope_position_scale: str,
    write_stdint: bool,
    write_6class: bool,
    write_10class: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Standardize TPIs, optionally write stdint rasters, and write class rasters.

    Returns:
        hist6, hist10 (None when that output is disabled).
        - hist6 includes nodata at index 0, classes 1..6 at indices 1..6.
        - hist10 includes nodata at index 0, classes 1..10 at indices 1..10.
    """
    hist6 = np.zeros(7, dtype=np.int64) if write_6class else None
    hist10 = np.zeros(11, dtype=np.int64) if write_10class else None
    clipped_small_low = 0
    clipped_small_high = 0
    clipped_large_low = 0
    clipped_large_high = 0

    with ExitStack() as stack:
        src_s = stack.enter_context(rasterio.open(tpi_small_path))
        src_l = stack.enter_context(rasterio.open(tpi_large_path))
        src_sl = stack.enter_context(rasterio.open(slope_path))

        if src_s.width != src_l.width or src_s.height != src_l.height:
            raise ValueError("Small and large TPI rasters have different dimensions.")
        if src_s.width != src_sl.width or src_s.height != src_sl.height:
            raise ValueError("Slope raster dimensions do not match TPI rasters.")
        if src_s.transform != src_l.transform or src_s.transform != src_sl.transform:
            raise ValueError("Input rasters have mismatched geotransforms.")
        if src_s.crs != src_l.crs or src_s.crs != src_sl.crs:
            raise ValueError("Input rasters have mismatched CRS.")

        std_profile = output_profile(src_s, dtype="int16", nodata=int(STD_INT_NODATA))
        class_profile = output_profile(src_s, dtype="uint8", nodata=int(CLASS_NODATA))
        landform10_profile = output_profile(src_s, dtype="int16", nodata=int(STD_INT_NODATA))

        dst_s_std = None
        dst_l_std = None
        if write_stdint:
            dst_s_std = stack.enter_context(rasterio.open(out_tpi_small_stdint, "w", **std_profile))
            dst_l_std = stack.enter_context(rasterio.open(out_tpi_large_stdint, "w", **std_profile))

        dst_6 = None
        if write_6class:
            dst_6 = stack.enter_context(rasterio.open(out_slope6, "w", **class_profile))

        dst_10 = None
        if write_10class:
            dst_10 = stack.enter_context(rasterio.open(out_landform10, "w", **landform10_profile))

        for win in iter_windows(src_s.height, src_s.width, tile_size):
            s_ma = src_s.read(1, window=win, masked=True)
            l_ma = src_l.read(1, window=win, masked=True)
            sl_ma = src_sl.read(1, window=win, masked=True)

            s = np.asarray(s_ma.data, dtype=np.float32)
            large = np.asarray(l_ma.data, dtype=np.float32)
            sl = np.asarray(sl_ma.data, dtype=np.float32)

            valid_s = (~np.ma.getmaskarray(s_ma)) & np.isfinite(s)
            valid_l = (~np.ma.getmaskarray(l_ma)) & np.isfinite(large)
            valid_sl = (~np.ma.getmaskarray(sl_ma)) & np.isfinite(sl)

            tp_s = to_std_units(s, valid_s, tpi_small_mean, tpi_small_std)
            tp_l = to_std_units(large, valid_l, tpi_large_mean, tpi_large_std)

            if write_stdint and dst_s_std is not None and dst_l_std is not None:
                s_std = np.full(s.shape, STD_INT_NODATA, dtype=np.int16)
                l_std = np.full(large.shape, STD_INT_NODATA, dtype=np.int16)

                if np.any(valid_s):
                    tp_s_valid = tp_s[valid_s]
                    clipped_small_low += int(np.count_nonzero(tp_s_valid < STD_INT_WRITE_MIN))
                    clipped_small_high += int(np.count_nonzero(tp_s_valid > STD_INT_WRITE_MAX))
                    clipped = np.clip(
                        tp_s_valid,
                        STD_INT_WRITE_MIN,
                        STD_INT_WRITE_MAX,
                    )
                    s_std[valid_s] = clipped.astype(np.int16, copy=False)

                if np.any(valid_l):
                    tp_l_valid = tp_l[valid_l]
                    clipped_large_low += int(np.count_nonzero(tp_l_valid < STD_INT_WRITE_MIN))
                    clipped_large_high += int(np.count_nonzero(tp_l_valid > STD_INT_WRITE_MAX))
                    clipped = np.clip(
                        tp_l_valid,
                        STD_INT_WRITE_MIN,
                        STD_INT_WRITE_MAX,
                    )
                    l_std[valid_l] = clipped.astype(np.int16, copy=False)

                dst_s_std.write(s_std, 1, window=win)
                dst_l_std.write(l_std, 1, window=win)

            if write_6class and dst_6 is not None and hist6 is not None:
                if slope_position_scale == "small":
                    tp_for_6 = tp_s
                    valid_6 = valid_s & valid_sl
                else:
                    tp_for_6 = tp_l
                    valid_6 = valid_l & valid_sl

                cls6 = classify_slope_position_6(tp_for_6, sl, valid_6, flat_threshold)
                dst_6.write(cls6.astype(np.uint8, copy=False), 1, window=win)
                hist6 += np.bincount(cls6.ravel(), minlength=7)

            if write_10class and dst_10 is not None and hist10 is not None:
                valid_10 = valid_s & valid_l & valid_sl
                cls10 = classify_landform_10class(tp_s, tp_l, sl, valid_10, flat_threshold)
                dst_10.write(cls10.astype(np.int16, copy=False), 1, window=win)

                hist10[0] += int(np.count_nonzero(cls10 == STD_INT_NODATA))
                valid_cls = cls10 != STD_INT_NODATA
                if np.any(valid_cls):
                    hist10 += np.bincount(
                        cls10[valid_cls].astype(np.int64, copy=False),
                        minlength=11,
                    )

    if write_stdint:
        total_small = clipped_small_low + clipped_small_high
        total_large = clipped_large_low + clipped_large_high
        if total_small > 0:
            print(
                "Warning: clipped "
                f"{total_small} small-scale standardized TPI cells to int16 range "
                f"[{STD_INT_WRITE_MIN}, {STD_INT_WRITE_MAX}] "
                f"(low={clipped_small_low}, high={clipped_small_high})."
            )
        if total_large > 0:
            print(
                "Warning: clipped "
                f"{total_large} large-scale standardized TPI cells to int16 range "
                f"[{STD_INT_WRITE_MIN}, {STD_INT_WRITE_MAX}] "
                f"(low={clipped_large_low}, high={clipped_large_high})."
            )

    return hist6, hist10


def print_histogram_slope6(name: str, hist: np.ndarray) -> None:
    """Print 6-class slope-position counts with class names."""
    print(f"{name} histogram:")
    print(f"  0 (nodata): {int(hist[0])}")
    for class_id in range(1, 7):
        class_name = SLOPE6_CLASS_NAMES[class_id]
        print(f"  {class_id} ({class_name}): {int(hist[class_id])}")


def print_histogram_landform10(name: str, hist: np.ndarray) -> None:
    """Print 10-class landform counts, including int16 nodata bucket."""
    print(f"{name} histogram:")
    print(f"  nodata ({int(STD_INT_NODATA)}): {int(hist[0])}")
    for class_id in range(1, 11):
        class_name = LANDFORM10_CLASS_NAMES[class_id]
        print(f"  {class_id} ({class_name}): {int(hist[class_id])}")


def main() -> None:
    args = parse_args()
    if args.mean_zero_tolerance is not None:
        args.mean_tol = args.mean_zero_tolerance
    validate_args(args)

    write_6class = args.landform_classification in ("6", "both")
    write_10class = args.landform_classification in ("10", "both")

    out_dir = args.out_dir if args.out_dir is not None else args.dem.resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    tpi_small_path = out_dir / "tpi_small.tif"
    tpi_large_path = out_dir / "tpi_large.tif"
    tpi_small_stdint_path = out_dir / "tpi_small_stdint.tif"
    tpi_large_stdint_path = out_dir / "tpi_large_stdint.tif"
    slope_path = out_dir / "slope_deg.tif"
    slope6_path = out_dir / "Geomorphology.tif"
    landform10_path = resolve_output_path(out_dir, args.landform10_out)
    landform10_path.parent.mkdir(parents=True, exist_ok=True)

    small_kernel = create_annulus_kernel(args.small_irad, args.small_orad)
    large_kernel = create_annulus_kernel(args.large_irad, args.large_orad)

    small_stats: RunningStats
    large_stats: RunningStats

    with rasterio.open(args.dem) as src:
        if src.count != 1:
            raise ValueError("Input DEM must be single-band.")
        if src.transform.a == 0 or src.transform.e == 0:
            raise ValueError("Invalid geotransform cell size.")

        xres = abs(float(src.transform.a))
        yres = abs(float(src.transform.e))

        float_nodata = resolve_float_nodata(src.nodata)
        small_tpi_tile = effective_tpi_tile_size(
            base_tile_size=args.tile_size,
            pad=args.small_orad,
            max_padded_side=args.max_padded_side,
        )
        large_tpi_tile = effective_tpi_tile_size(
            base_tile_size=args.tile_size,
            pad=args.large_orad,
            max_padded_side=args.max_padded_side,
        )

        if small_tpi_tile < args.tile_size:
            padded = small_tpi_tile + (2 * args.small_orad)
            print(
                "Reducing small-scale TPI tile size from "
                f"{args.tile_size} to {small_tpi_tile} to keep padded side "
                f"{padded} <= --max-padded-side {args.max_padded_side}."
            )
        if large_tpi_tile < args.tile_size:
            padded = large_tpi_tile + (2 * args.large_orad)
            print(
                "Reducing large-scale TPI tile size from "
                f"{args.tile_size} to {large_tpi_tile} to keep padded side "
                f"{padded} <= --max-padded-side {args.max_padded_side}."
            )
        for scale_name, pad in [("small", args.small_orad), ("large", args.large_orad)]:
            kernel_diameter = (2 * pad) + 1
            if kernel_diameter > min(src.width, src.height):
                print(
                    "Warning: "
                    f"{scale_name}-scale kernel diameter ({kernel_diameter}) exceeds "
                    "the DEM minimum dimension; edge effects and slower processing are likely."
                )
        print_tpi_memory_estimate("Small-scale", small_tpi_tile, args.small_orad)
        print_tpi_memory_estimate("Large-scale", large_tpi_tile, args.large_orad)

        print("Computing slope (degrees)...")
        compute_slope_raster(
            src=src,
            out_path=slope_path,
            tile_size=args.tile_size,
            float_nodata=float_nodata,
            xres=xres,
            yres=yres,
        )

        if args.reuse_existing_tpi and raster_matches_reference(tpi_small_path, src):
            print(f"Reusing existing small-scale TPI raster: {tpi_small_path}")
            with rasterio.open(tpi_small_path) as src_tpi_small:
                small_stats = compute_raster_stats(src_tpi_small, args.tile_size)
        else:
            print("Computing small-scale TPI...")
            small_stats = compute_tpi_raster(
                src=src,
                out_path=tpi_small_path,
                kernel=small_kernel,
                pad=args.small_orad,
                tile_size=small_tpi_tile,
                float_nodata=float_nodata,
            )

        if args.reuse_existing_tpi and raster_matches_reference(tpi_large_path, src):
            print(f"Reusing existing large-scale TPI raster: {tpi_large_path}")
            with rasterio.open(tpi_large_path) as src_tpi_large:
                large_stats = compute_raster_stats(src_tpi_large, args.tile_size)
        else:
            print("Computing large-scale TPI...")
            large_stats = compute_tpi_raster(
                src=src,
                out_path=tpi_large_path,
                kernel=large_kernel,
                pad=args.large_orad,
                tile_size=large_tpi_tile,
                float_nodata=float_nodata,
            )

    small_mean, small_std = checked_mean_std(small_stats, "Small-scale")
    large_mean, large_std = checked_mean_std(large_stats, "Large-scale")

    if args.require_mean_near_zero:
        ensure_mean_near_zero_or_force(
            "Small-scale",
            small_mean,
            small_std,
            args.mean_tol,
            args.force,
        )
        ensure_mean_near_zero_or_force(
            "Large-scale",
            large_mean,
            large_std,
            args.mean_tol,
            args.force,
        )

    print(f"Small TPI mean/std: {small_mean:.6f} / {small_std:.6f}")
    print(f"Large TPI mean/std: {large_mean:.6f} / {large_std:.6f}")

    hist6: np.ndarray | None = None
    hist10: np.ndarray | None = None
    if args.standardize or write_6class or write_10class:
        print("Writing standardized TPIs and requested landform rasters...")
        hist6, hist10 = write_standardized_and_landforms(
            tpi_small_path=tpi_small_path,
            tpi_large_path=tpi_large_path,
            slope_path=slope_path,
            tpi_small_mean=small_mean,
            tpi_small_std=small_std,
            tpi_large_mean=large_mean,
            tpi_large_std=large_std,
            out_tpi_small_stdint=tpi_small_stdint_path,
            out_tpi_large_stdint=tpi_large_stdint_path,
            out_slope6=slope6_path,
            out_landform10=landform10_path,
            tile_size=args.tile_size,
            flat_threshold=args.flat_threshold,
            slope_position_scale=args.slope_position_scale,
            write_stdint=args.standardize,
            write_6class=write_6class,
            write_10class=write_10class,
        )
    else:
        print(
            "Skipping standardized/class outputs "
            "(--no-write-stdint/--no-standardize and classification=none)."
        )

    if write_6class and hist6 is not None:
        print_histogram_slope6("Slope position 6-class", hist6)
    if write_10class and hist10 is not None:
        print_histogram_landform10("Landform 10-class", hist10)

    print("Outputs:")
    print(f"  slope: {slope_path}")
    print(f"  tpi small: {tpi_small_path}")
    print(f"  tpi large: {tpi_large_path}")
    if args.standardize:
        print(f"  tpi small stdint: {tpi_small_stdint_path}")
        print(f"  tpi large stdint: {tpi_large_stdint_path}")
    if write_6class:
        print(f"  slope position 6-class: {slope6_path}")
    if write_10class:
        print(f"  landform 10-class: {landform10_path}")

    print("Done.")


if __name__ == "__main__":
    main()
