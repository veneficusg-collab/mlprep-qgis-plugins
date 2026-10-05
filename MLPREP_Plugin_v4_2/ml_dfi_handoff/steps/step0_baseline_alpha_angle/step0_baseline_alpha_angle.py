"""Create a source-slope-based baseline alpha raster from slope in degrees.

This Step 0 product is a baseline alpha prior for the dynamic-alpha DFI runout
workflow. Its fixed lookup is calibrated from clean observed alpha angles
summarized against mean slope in matched upper-30-percent source areas. The
input slope raster is never modified.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

# Rasterio wheels carry matching PROJ/GDAL data. External GIS applications,
# especially TauDEM, can leave incompatible process-wide overrides behind.
for _gis_override in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(_gis_override, None)

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

LOGGER = logging.getLogger("step0_baseline_alpha_angle")

DEFAULT_NODATA = -9999.0
CLASS_NODATA = -9999
# Empirical positive-alpha values are whole-degree priors from clean DP1
# observed alpha within mean-source-slope bins.
# Class 0 remains a deliberate non-initiation policy: source cells below
# 10 degrees receive alpha NoData because no alpha angle is assigned.
ALPHA_VALUES = np.asarray(
    [np.nan, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 22.0, 25.0, 28.0, 29.0, 30.0],
    dtype=np.float32,
)
UPPER_BOUNDS = np.asarray(
    [10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45],
    dtype=np.float64,
)
LOOKUP_TABLE: tuple[dict[str, Any], ...] = (
    {
        "class_id": 0,
        "interval": "slope < 10",
        "alpha_degrees": None,
        "calibration_n": 15,
        "observed_alpha_median_degrees": 9.268,
        "assignment_basis": "non_source_nodata_policy_override",
    },
    {
        "class_id": 1,
        "interval": "10 <= slope < 12",
        "alpha_degrees": 9.0,
        "calibration_n": None,
        "observed_alpha_median_degrees": None,
        "assignment_basis": "configured_lookup_table",
    },
    {
        "class_id": 2,
        "interval": "12 <= slope < 14",
        "alpha_degrees": 11.0,
        "calibration_n": None,
        "observed_alpha_median_degrees": None,
        "assignment_basis": "configured_lookup_table",
    },
    {
        "class_id": 3,
        "interval": "14 <= slope < 16",
        "alpha_degrees": 13.0,
        "calibration_n": None,
        "observed_alpha_median_degrees": None,
        "assignment_basis": "configured_lookup_table",
    },
    {
        "class_id": 4,
        "interval": "16 <= slope < 18",
        "alpha_degrees": 15.0,
        "calibration_n": None,
        "observed_alpha_median_degrees": None,
        "assignment_basis": "configured_lookup_table",
    },
    {
        "class_id": 5,
        "interval": "18 <= slope < 20",
        "alpha_degrees": 17.0,
        "calibration_n": None,
        "observed_alpha_median_degrees": None,
        "assignment_basis": "configured_lookup_table",
    },
    {
        "class_id": 6,
        "interval": "20 <= slope < 25",
        "alpha_degrees": 19.0,
        "calibration_n": 55,
        "observed_alpha_median_degrees": 18.793,
        "assignment_basis": "rounded_upper_interval_median_alpha",
    },
    {
        "class_id": 7,
        "interval": "25 <= slope < 30",
        "alpha_degrees": 22.0,
        "calibration_n": 108,
        "observed_alpha_median_degrees": 21.898,
        "assignment_basis": "rounded_upper_interval_median_alpha",
    },
    {
        "class_id": 8,
        "interval": "30 <= slope < 35",
        "alpha_degrees": 25.0,
        "calibration_n": 110,
        "observed_alpha_median_degrees": 25.252,
        "assignment_basis": "rounded_upper_interval_median_alpha",
    },
    {
        "class_id": 9,
        "interval": "35 <= slope < 40",
        "alpha_degrees": 28.0,
        "calibration_n": 115,
        "observed_alpha_median_degrees": 27.874,
        "assignment_basis": "rounded_upper_interval_median_alpha",
    },
    {
        "class_id": 10,
        "interval": "40 <= slope < 45",
        "alpha_degrees": 29.0,
        "calibration_n": 65,
        "observed_alpha_median_degrees": 28.835,
        "assignment_basis": "rounded_upper_interval_median_alpha",
    },
    {
        "class_id": 11,
        "interval": "slope >= 45",
        "alpha_degrees": 30.0,
        "calibration_n": 19,
        "observed_alpha_median_degrees": [30.218, 29.895],
        "assignment_basis": "rounded_upper_interval_median_alpha",
    },
)
CALIBRATION: dict[str, Any] = {
    "response": "clean DP1 observed alpha angle (a_obs_deg)",
    "predictor": "mean source-area slope in degrees",
    "source_area": "matched upper-30-percent non-depositional/source footprint",
    "valid_observation_count": 564,
    "association": {"pearson_r": 0.7769277464, "spearman_rho": 0.7618945631},
    "assignment_method": (
        "Classes 1-5 use an assigned alpha one degree below the lower slope "
        "interval bound. Classes 6-11 use the rounded median alpha angle of "
        "the slope interval from the generated slope-alpha correlation line."
    ),
    "policy_exception": (
        "Class 0 is written as alpha NoData to mark cells with slope below "
        "10 degrees as non-source or very low initiation-potential cells, "
        "rather than applying its observed median or using alpha 0."
    ),
    "application_note": (
        "Calibration uses source-area mean slope observations, while raster "
        "generation assigns the prior by local slope cell."
    ),
}


@dataclass(frozen=True)
class Step0Params:
    """Inputs and output options for Step 0 baseline alpha raster generation."""

    slope: Path
    output_alpha: Path
    output_class: Path | None = None
    metadata_out: Path | None = None
    nodata: float = DEFAULT_NODATA
    overwrite: bool = False


def classify_source_slope(
    slope_degrees: np.ndarray,
    valid_mask: np.ndarray,
    *,
    alpha_nodata: float = DEFAULT_NODATA,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify valid slope cells with lower-inclusive, upper-exclusive bins."""
    if slope_degrees.shape != valid_mask.shape:
        raise ValueError("Slope values and valid mask must have identical shapes.")

    alpha = np.full(slope_degrees.shape, alpha_nodata, dtype=np.float32)
    slope_class = np.full(slope_degrees.shape, CLASS_NODATA, dtype=np.int16)

    # searchsorted with side="right" places exact lower bounds in their new
    # class: 10 -> class 1, 12 -> class 2, and 45 -> final class 11.
    valid_classes = np.searchsorted(
        UPPER_BOUNDS,
        slope_degrees[valid_mask],
        side="right",
    ).astype(np.int16)
    slope_class[valid_mask] = valid_classes
    positive_alpha = np.isfinite(ALPHA_VALUES[valid_classes])
    valid_rows, valid_cols = np.nonzero(valid_mask)
    alpha[
        valid_rows[positive_alpha],
        valid_cols[positive_alpha],
    ] = ALPHA_VALUES[valid_classes[positive_alpha]]
    return alpha, slope_class


@dataclass
class RunningStatistics:
    count: int = 0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    total: float = 0.0

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        self.count += int(values.size)
        self.minimum = min(self.minimum, float(np.min(values)))
        self.maximum = max(self.maximum, float(np.max(values)))
        self.total += float(np.sum(values, dtype=np.float64))

    def as_dict(self) -> dict[str, float | None]:
        if self.count == 0:
            return {"min": None, "max": None, "mean": None}
        return {
            "min": float(self.minimum),
            "max": float(self.maximum),
            "mean": float(self.total / self.count),
        }


def _temporary_output_path(target: Path) -> Path:
    return target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp{target.suffix}")


def _resolve_metadata_path(params: Step0Params) -> Path:
    if params.metadata_out is not None:
        return params.metadata_out
    return params.output_alpha.with_name(f"{params.output_alpha.stem}_metadata.json")


def _validate_output_paths(params: Step0Params, metadata_path: Path) -> None:
    input_path = params.slope.resolve()
    targets = [params.output_alpha, metadata_path]
    if params.output_class is not None:
        targets.append(params.output_class)

    resolved_targets = [path.resolve() for path in targets]
    if len(set(resolved_targets)) != len(resolved_targets):
        raise ValueError("Output alpha, class, and metadata paths must be distinct.")
    if input_path in resolved_targets:
        raise ValueError("Step 0 outputs must not overwrite the input slope raster.")
    for path in targets:
        if path.exists() and not params.overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Pass --overwrite to replace it."
            )


def _validate_alpha_nodata(value: float) -> None:
    if not np.isfinite(value):
        raise ValueError("--nodata must be a finite numeric value.")
    assigned_alpha_values = ALPHA_VALUES[np.isfinite(ALPHA_VALUES)]
    if np.any(assigned_alpha_values == np.float32(value)):
        raise ValueError(
            "--nodata cannot equal an assigned positive alpha value."
        )


def run_step0(params: Step0Params) -> dict[str, Any]:
    """Generate baseline alpha products using bounded-memory raster windows."""
    if not params.slope.exists():
        raise FileNotFoundError(f"Input slope raster does not exist: {params.slope}")
    _validate_alpha_nodata(params.nodata)
    metadata_path = _resolve_metadata_path(params)
    _validate_output_paths(params, metadata_path)

    targets = [params.output_alpha, metadata_path]
    if params.output_class is not None:
        targets.append(params.output_class)
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)

    alpha_temp = _temporary_output_path(params.output_alpha)
    class_temp = _temporary_output_path(params.output_class) if params.output_class is not None else None
    metadata_temp = _temporary_output_path(metadata_path)
    temporary_paths = [alpha_temp, metadata_temp]
    if class_temp is not None:
        temporary_paths.append(class_temp)

    warnings: list[str] = []
    try:
        with rasterio.open(params.slope) as src:
            if src.count != 1:
                raise ValueError(
                    f"Input slope raster must be single-band; found {src.count} bands."
                )
            if src.crs is None:
                raise ValueError("Input slope raster must define a CRS.")
            if src.transform is None:
                raise ValueError("Input slope raster must define an affine transform.")
            if src.width <= 0 or src.height <= 0:
                raise ValueError("Input slope raster must have positive width and height.")
            if not np.issubdtype(np.dtype(src.dtypes[0]), np.number):
                raise ValueError("Input slope raster must contain numeric values.")

            alpha_profile = src.profile.copy()
            alpha_profile.update(
                driver="GTiff",
                count=1,
                dtype="float32",
                nodata=float(params.nodata),
                compress="LZW",
                BIGTIFF="IF_SAFER",
            )
            class_profile = src.profile.copy()
            class_profile.update(
                driver="GTiff",
                count=1,
                dtype="int16",
                nodata=CLASS_NODATA,
                compress="LZW",
                BIGTIFF="IF_SAFER",
            )

            total_cell_count = int(src.width * src.height)
            valid_count = 0
            alpha_valid_count = 0
            class_counts = np.zeros(len(ALPHA_VALUES), dtype=np.int64)
            slope_stats = RunningStatistics()
            alpha_stats = RunningStatistics()
            window_size = 1024
            window_factory = cast(Any, Window)

            with rasterio.open(alpha_temp, "w", **alpha_profile) as alpha_dst:
                class_context = (
                    rasterio.open(class_temp, "w", **class_profile)
                    if class_temp is not None
                    else None
                )
                try:
                    for row_off in range(0, src.height, window_size):
                        height = min(window_size, src.height - row_off)
                        for col_off in range(0, src.width, window_size):
                            width = min(window_size, src.width - col_off)
                            window = window_factory(col_off, row_off, width, height)
                            input_data = src.read(1, window=window, masked=True)
                            slope = np.asarray(input_data.data, dtype=np.float64)
                            valid_mask = ~np.ma.getmaskarray(input_data) & np.isfinite(slope)
                            alpha, slope_class = classify_source_slope(
                                slope,
                                valid_mask,
                                alpha_nodata=params.nodata,
                            )
                            alpha_dst.write(alpha, 1, window=window)
                            if class_context is not None:
                                class_context.write(slope_class, 1, window=window)

                            valid_slope = slope[valid_mask]
                            slope_stats.update(valid_slope)
                            block_valid_count = int(valid_slope.size)
                            valid_count += block_valid_count
                            if block_valid_count:
                                class_counts += np.bincount(
                                    slope_class[valid_mask],
                                    minlength=len(ALPHA_VALUES),
                                ).astype(np.int64)
                            alpha_valid_mask = valid_mask & (slope_class != 0)
                            valid_alpha = alpha[alpha_valid_mask]
                            alpha_stats.update(valid_alpha)
                            alpha_valid_count += int(valid_alpha.size)
                        LOGGER.info(
                            "Processed raster rows %d-%d of %d",
                            row_off + 1,
                            row_off + height,
                            src.height,
                        )
                finally:
                    if class_context is not None:
                        class_context.close()

            if slope_stats.count == 0:
                warnings.append(
                    "No valid finite slope cells were available; output contains NoData only."
                )
            else:
                if slope_stats.minimum < 0:
                    warnings.append(
                        "Valid slope values below 0 degrees were found and classified "
                        "using the lookup table."
                    )
                if slope_stats.maximum > 90:
                    warnings.append(
                        "Valid slope values above 90 degrees were found and classified "
                        "using the lookup table."
                    )

            class_rows: list[dict[str, Any]] = []
            for entry in LOOKUP_TABLE:
                class_id = int(entry["class_id"])
                count = int(class_counts[class_id])
                percentage = (100.0 * count / valid_count) if valid_count else 0.0
                class_rows.append(
                    {
                        **entry,
                        "cell_count": count,
                        "percentage_of_valid_cells": percentage,
                    }
                )

            input_nodata_count = int(total_cell_count - valid_count)
            alpha_nodata_count = int(total_cell_count - alpha_valid_count)
            metadata: dict[str, Any] = {
                "step": "step0",
                "purpose": (
                    "Source-slope-based baseline alpha prior for the dynamic-alpha "
                    "DFI runout workflow."
                ),
                "input_slope_path": str(params.slope.resolve()),
                "output_alpha_path": str(params.output_alpha.resolve()),
                "output_class_path": (
                    str(params.output_class.resolve())
                    if params.output_class is not None
                    else None
                ),
                "metadata_path": str(metadata_path.resolve()),
                "processing": {
                    "mode": "windowed",
                    "window_size_pixels": window_size,
                    "temporary_output_commit": True,
                },
                "raster": {
                    "shape": [src.height, src.width],
                    "height": src.height,
                    "width": src.width,
                    "crs": str(src.crs),
                    "transform": list(src.transform)[:6],
                    "resolution": list(src.res),
                    "bounds": list(src.bounds),
                },
                "lookup_table": list(LOOKUP_TABLE),
                "calibration": CALIBRATION,
                "statistics": {
                    "input_slope_degrees": slope_stats.as_dict(),
                    "output_alpha_degrees": alpha_stats.as_dict(),
                },
                "total_cell_count": total_cell_count,
                "valid_cell_count": valid_count,
                "input_nodata_cell_count": input_nodata_count,
                "non_source_class_zero_cell_count": int(class_counts[0]),
                "alpha_nodata_cell_count": alpha_nodata_count,
                "nodata_cell_count": alpha_nodata_count,
                "cell_count_per_class": {
                    str(row["class_id"]): row["cell_count"] for row in class_rows
                },
                "percentage_per_class": {
                    str(row["class_id"]): row["percentage_of_valid_cells"]
                    for row in class_rows
                },
                "class_summary": class_rows,
                "nodata": {
                    "alpha": float(params.nodata),
                    "class": CLASS_NODATA if params.output_class is not None else None,
                },
                "warnings": warnings,
            }

        metadata_temp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if class_temp is not None and params.output_class is not None:
            os.replace(class_temp, params.output_class)
        os.replace(alpha_temp, params.output_alpha)
        os.replace(metadata_temp, metadata_path)
        return metadata
    except Exception:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        raise


def _resolve_config_path(config_dir: Path, value: object, field_name: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Step 0 config field '{field_name}' must be a non-empty path.")
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()


def load_step0_config(config_path: Path) -> dict[str, Any]:
    resolved_config_path = config_path.expanduser().resolve()
    with resolved_config_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Step 0 config root must be a JSON object: {resolved_config_path}")
    section = payload.get("step0", payload)
    if not isinstance(section, dict):
        raise ValueError("Step 0 config section must be a JSON object.")
    allowed = {
        "slope",
        "output_alpha",
        "output_class",
        "metadata_out",
        "nodata",
        "overwrite",
    }
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"Unknown Step 0 config keys: {', '.join(unknown)}")

    config = dict(section)
    for field_name in ("slope", "output_alpha"):
        config[field_name] = _resolve_config_path(
            resolved_config_path.parent, config.get(field_name), field_name
        )
    for field_name in ("output_class", "metadata_out"):
        value = str(config.get(field_name, "") or "").strip()
        config[field_name] = (
            _resolve_config_path(resolved_config_path.parent, value, field_name)
            if value
            else None
        )
    return config


def parse_args(argv: Sequence[str] | None = None) -> Step0Params:
    parser = argparse.ArgumentParser(
        description=(
            "Step 0: generate a source-slope-based baseline alpha-angle raster "
            "from a slope raster in degrees."
        )
    )
    parser.add_argument("--config", type=Path, help="Optional Step 0 JSON config file.")
    parser.add_argument("--slope", type=Path, help="Input slope raster.")
    parser.add_argument(
        "--output-alpha",
        type=Path,
        help="Output alpha-angle raster in degrees.",
    )
    parser.add_argument(
        "--output-class",
        type=Path,
        default=None,
        help="Optional output slope-class raster.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help=(
            "Optional JSON metadata output. Defaults beside --output-alpha as "
            "<output_alpha_stem>_metadata.json."
        ),
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=None,
        help="NoData value for the output alpha raster (default: -9999).",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow replacement of existing output files.",
    )
    args = parser.parse_args(argv)
    config: dict[str, Any] = {}
    if args.config is not None:
        config.update(load_step0_config(args.config))

    path_overrides = {
        "slope": args.slope,
        "output_alpha": args.output_alpha,
        "output_class": args.output_class,
        "metadata_out": args.metadata_out,
    }
    for key, value in path_overrides.items():
        if value is not None:
            config[key] = value.expanduser().resolve()
    if args.nodata is not None:
        config["nodata"] = args.nodata
    if args.overwrite is not None:
        config["overwrite"] = args.overwrite

    if not config.get("slope"):
        parser.error("--slope is required unless supplied by --config.")
    if not config.get("output_alpha"):
        parser.error("--output-alpha is required unless supplied by --config.")
    return Step0Params(
        slope=Path(config["slope"]),
        output_alpha=Path(config["output_alpha"]),
        output_class=None if config.get("output_class") is None else Path(config["output_class"]),
        metadata_out=None if config.get("metadata_out") is None else Path(config["metadata_out"]),
        nodata=float(config.get("nodata", DEFAULT_NODATA)),
        overwrite=bool(config.get("overwrite", False)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        params = parse_args(argv)
        metadata = run_step0(params)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Alpha raster written: %s", metadata["output_alpha_path"])
    if metadata["output_class_path"] is not None:
        LOGGER.info("Class raster written: %s", metadata["output_class_path"])
    LOGGER.info("Metadata written: %s", metadata["metadata_path"])
    for warning in metadata["warnings"]:
        LOGGER.warning("%s", warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

