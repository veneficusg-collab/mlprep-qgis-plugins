import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import CRS as PyprojCRS
import rasterio
from rasterio.transform import xy as transform_xy


CONFIG = {
    "label_raster_path": r"",
    "predictor_rasters": {
        "slope": r"",
        "plan_curvature": r"",
        "profile_curvature": r"",
        "tpi": r"",
        "flow_accumulation": r"",
        "distance_to_drainage": r"",
        "local_relief": r"",
        "proximity_to_stream": r"",
        "ndvi": r"",
    },
    "group_rasters": {
        "landslide_id": r"",
    },
    "categorical_predictors": [],
    "output_dir": r"",
    "output_table_name": "ml_dfi_pixel_samples",
    "output_format": "csv",
    "label_nodata_value": -9999,
    "include_coordinates": True,
    "include_row_col": True,
    "compress_outputs": True,
}


LOG_NAME = "step2_extract_pixel_samples"
OUTPUTS = {
    "log": "processing_log.txt",
    "alignment_report": "predictor_alignment_report.csv",
    "multicollinearity_summary": "multicollinearity_summary.csv",
    "multicollinearity_pearson_matrix": "multicollinearity_pearson_matrix.csv",
    "multicollinearity_pearson_heatmap": "multicollinearity_pearson_heatmap.png",
    "multicollinearity_vif_tolerance": "multicollinearity_vif_tolerance.png",
    "run_config": "run_config.json",
}
MULTICOLLINEARITY_MAX_SAMPLE_ROWS = 100_000
MULTICOLLINEARITY_RANDOM_SEED = 42
PEARSON_ABS_THRESHOLD = 0.70
VIF_THRESHOLD = 5.0
TOLERANCE_THRESHOLD = 0.20
MULTICOLLINEARITY_DROP_PRIORITY = {
    "proximity_to_stream": 100,
}
MANAGED_STATIC_OUTPUT_NAMES = set(OUTPUTS.values()) | {"processing_log.failed.txt"}


def setup_logging(output_dir: Path) -> tuple[logging.Logger, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / OUTPUTS["log"]
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


def close_file_log_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)


def publish_output_bundle(
    staged_paths: dict[str, Path],
    output_dir: Path,
    managed_names: set[str],
) -> dict[str, Path]:
    """Publish a complete validated bundle and restore prior files on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(tempfile.mkdtemp(prefix=".step2-backup-", dir=output_dir.parent))
    final_paths = {key: output_dir / path.name for key, path in staged_paths.items()}
    backed_up: dict[str, Path] = {}
    published: set[str] = set()
    try:
        for name in sorted(managed_names):
            final_path = output_dir / name
            if final_path.exists() and final_path.is_file():
                backup_path = backup_dir / name
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(final_path, backup_path)
                backed_up[name] = backup_path

        for key, staged_path in staged_paths.items():
            if not staged_path.exists():
                raise FileNotFoundError(f"Validated staged output is missing: {staged_path}")
            os.replace(staged_path, final_paths[key])
            published.add(key)
    except Exception as publish_error:
        rollback_errors: list[str] = []
        for key in published:
            final_path = final_paths[key]
            if final_path.exists():
                try:
                    final_path.unlink()
                except OSError as exc:
                    rollback_errors.append(f"remove {final_path}: {exc}")
        for name, backup_path in backed_up.items():
            if backup_path.exists():
                try:
                    os.replace(backup_path, output_dir / name)
                except OSError as exc:
                    rollback_errors.append(f"restore {name}: {exc}")
        if rollback_errors:
            raise RuntimeError(
                "Step 2 publication failed and rollback was incomplete. "
                f"Previous outputs remain in {backup_dir}. "
                f"Problems: {'; '.join(rollback_errors)}"
            ) from publish_error
        shutil.rmtree(backup_dir, ignore_errors=True)
        raise
    else:
        shutil.rmtree(backup_dir, ignore_errors=True)
    return final_paths


def validate_config(config: dict[str, Any]) -> None:
    label_raster_path = str(config.get("label_raster_path", "")).strip()
    output_dir = str(config.get("output_dir", "")).strip()
    output_table_name = str(config.get("output_table_name", "")).strip()
    output_format = str(config.get("output_format", "csv")).strip().lower()

    if not label_raster_path:
        raise ValueError("CONFIG['label_raster_path'] is required.")
    if not Path(label_raster_path).exists():
        raise FileNotFoundError(f"Label raster does not exist: {label_raster_path}")
    if not output_dir:
        raise ValueError("CONFIG['output_dir'] is required.")
    if not output_table_name:
        raise ValueError("CONFIG['output_table_name'] must not be empty.")
    if (
        Path(output_table_name).name != output_table_name
        or output_table_name in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", output_table_name)
    ):
        raise ValueError(
            "output_table_name must be a filename stem containing only letters, "
            "numbers, periods, underscores, or hyphens."
        )
    if output_format not in {"csv", "parquet", "both"}:
        raise ValueError("output_format must be one of: csv, parquet, both.")
    if "drop_rows_with_any_predictor_nodata" in config:
        raise ValueError(
            "drop_rows_with_any_predictor_nodata is no longer configurable. "
            "Step 2 always drops rows with predictor NoData, NaN, or infinite values."
        )

    predictor_rasters = config.get("predictor_rasters", {})
    if not isinstance(predictor_rasters, dict):
        raise ValueError("predictor_rasters must be a dictionary of predictor_name -> raster_path.")
    group_rasters = config.get("group_rasters", {})
    if not isinstance(group_rasters, dict):
        raise ValueError("group_rasters must be a dictionary of group_name -> raster_path.")
    configured_group_paths = [
        str(path).strip()
        for path in group_rasters.values()
        if path is not None and str(path).strip()
    ]
    if not configured_group_paths:
        raise ValueError(
            "At least one group raster is required in group_rasters. "
            "Step 2 training samples must carry a group column such as landslide_id "
            "so Step 3 can use grouped validation and reduce leakage from spatial autocorrelation."
        )
    if not str(group_rasters.get("landslide_id", "")).strip():
        raise ValueError(
            "group_rasters.landslide_id is required. "
            "This workflow uses landslide_id as the grouping column for Step 3 stratified group k-fold validation."
        )
    categorical_predictors = config.get("categorical_predictors", [])
    if not isinstance(categorical_predictors, list) or any(
        not isinstance(value, str) or not value.strip()
        for value in categorical_predictors
    ):
        raise ValueError("categorical_predictors must be a list of predictor names.")

    int(config.get("label_nodata_value", -9999))


def sanitize_column_name(name: str, used_names: set[str]) -> str:
    sanitized = str(name).strip().lower()
    sanitized = sanitized.replace(" ", "_")
    sanitized = re.sub(r"[^a-z0-9_]+", "", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        sanitized = "field"
    if sanitized[0].isdigit():
        sanitized = f"f_{sanitized}"

    base_name = sanitized
    suffix = 2
    while sanitized in used_names:
        sanitized = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(sanitized)
    return sanitized


def open_reference_label_raster(
    label_raster_path: str,
    config_label_nodata_value: int,
    logger: logging.Logger,
) -> tuple[dict[str, Any], np.ma.MaskedArray]:
    with rasterio.open(label_raster_path) as src:
        if src.count < 1:
            raise ValueError(f"Label raster has no band 1: {label_raster_path}")
        if src.crs is None:
            raise ValueError(f"Label raster has no CRS: {label_raster_path}")
        label_array = src.read(1, masked=True)
        label_nodata = src.nodata if src.nodata is not None else config_label_nodata_value
        reference = {
            "path": str(Path(label_raster_path).resolve()),
            "transform": src.transform,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "shape": (src.height, src.width),
            "bounds": src.bounds,
            "dtype": src.dtypes[0],
            "nodata": label_nodata,
            "profile": src.profile.copy(),
        }
    logger.info(
        "Label raster grid: width=%d height=%d CRS=%s dtype=%s nodata=%s",
        reference["width"],
        reference["height"],
        reference["crs"],
        reference["dtype"],
        reference["nodata"],
    )
    return reference, label_array


def collect_predictor_paths(
    predictor_rasters: dict[str, Any],
    group_rasters: dict[str, Any],
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    used_names: set[str] = {"sample_id", "row", "col", "x", "y", "target"}

    def collect(
        source: dict[str, Any],
        kind: str,
    ) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for raw_name, raw_path in source.items():
            path_text = "" if raw_path is None else str(raw_path).strip()
            if not path_text:
                logger.warning("Ignoring blank %s raster path for '%s'.", kind, raw_name)
                continue
            path = Path(path_text)
            if not path.exists():
                raise FileNotFoundError(f"{kind.capitalize()} raster does not exist for '{raw_name}': {path}")
            column_name = sanitize_column_name(str(raw_name), used_names)
            collected.append(
                {
                    "name": str(raw_name),
                    "column_name": column_name,
                    "path": str(path.resolve()),
                    "kind": kind,
                }
            )
        return collected

    predictor_specs = collect(predictor_rasters, "predictor")
    if not predictor_specs:
        raise ValueError("No valid predictor rasters were provided after filtering blank paths.")
    group_specs = collect(group_rasters, "group")
    if not group_specs:
        raise ValueError(
            "No valid group rasters were provided after filtering blank paths. "
            "At least one group column is mandatory for grouped training/validation."
        )
    return predictor_specs, group_specs


def crs_authority(value: Any) -> tuple[str, str] | None:
    try:
        authority = value.to_authority()
    except Exception:
        authority = None
    if authority is not None:
        return authority
    try:
        return PyprojCRS.from_wkt(value.to_wkt()).to_authority()
    except Exception:
        return None


def crs_values_match(left: Any, right: Any) -> bool:
    if left == right:
        return True
    if left is None or right is None:
        return False
    left_authority = crs_authority(left)
    right_authority = crs_authority(right)
    return left_authority is not None and left_authority == right_authority


def check_raster_alignment(
    reference: dict[str, Any],
    raster_spec: dict[str, Any],
) -> dict[str, Any]:
    with rasterio.open(raster_spec["path"]) as src:
        if src.count < 1:
            raise ValueError(f"Raster has no band 1: {raster_spec['path']}")
        crs_match = crs_values_match(src.crs, reference["crs"])
        transform_match = bool(src.transform == reference["transform"])
        width_match = bool(src.width == reference["width"])
        height_match = bool(src.height == reference["height"])
        status = "OK" if crs_match and transform_match and width_match and height_match else "ERROR"

        return {
            "name": raster_spec["name"],
            "column_name": raster_spec["column_name"],
            "path": raster_spec["path"],
            "crs_match": crs_match,
            "transform_match": transform_match,
            "width_match": width_match,
            "height_match": height_match,
            "nodata_value": src.nodata,
            "dtype": src.dtypes[0],
            "status": status,
        }


def extract_raster_values_at_points(
    raster_path: str,
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read only raster blocks that contain requested row/column positions."""
    values = np.full(rows.size, np.nan, dtype=np.float64)
    nodata_rows = np.zeros(rows.size, dtype=bool)
    nonfinite_rows = np.zeros(rows.size, dtype=bool)
    if rows.size == 0:
        return values, nodata_rows, nonfinite_rows

    with rasterio.open(raster_path) as src:
        if src.count < 1:
            raise ValueError(f"Raster has no band 1: {raster_path}")
        block_height, block_width = src.block_shapes[0]
        if block_width == src.width and block_height < min(64, src.height):
            # Strip-organized GeoTIFFs commonly expose one raster row per block.
            # Coalescing strips avoids thousands of tiny RasterIO calls.
            block_height = min(512, src.height)
            block_width = src.width
        block_column_count = int(np.ceil(src.width / block_width))
        block_keys = (
            (rows.astype(np.int64) // block_height) * block_column_count
            + (cols.astype(np.int64) // block_width)
        )
        order = np.argsort(block_keys, kind="stable")
        ordered_keys = block_keys[order]
        boundaries = np.flatnonzero(np.diff(ordered_keys)) + 1
        groups = np.split(order, boundaries)

        for point_indices in groups:
            key = int(block_keys[point_indices[0]])
            block_row = key // block_column_count
            block_col = key % block_column_count
            row_off = block_row * block_height
            col_off = block_col * block_width
            window_height = min(block_height, src.height - row_off)
            window_width = min(block_width, src.width - col_off)
            window = rasterio.windows.Window(
                col_off,
                row_off,
                window_width,
                window_height,
            )
            array = src.read(1, window=window, masked=True)
            local_rows = rows[point_indices] - row_off
            local_cols = cols[point_indices] - col_off
            extracted = array[local_rows, local_cols]
            extracted_mask = np.ma.getmaskarray(extracted)
            extracted_values = np.asarray(extracted.data, dtype=np.float64)
            raw_nonfinite = ~np.isfinite(extracted_values)
            values[point_indices] = extracted_values
            nodata_rows[point_indices] = extracted_mask
            nonfinite_rows[point_indices] = raw_nonfinite & ~extracted_mask

    values[nodata_rows | nonfinite_rows] = np.nan
    return values, nodata_rows, nonfinite_rows


def extract_valid_label_indices(
    label_array: np.ma.MaskedArray,
    label_nodata_value: int | float,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    label_values = np.asarray(label_array.data, dtype=np.float64)
    mask = np.ma.getmaskarray(label_array).copy()
    if label_nodata_value is not None and np.isfinite(float(label_nodata_value)):
        mask |= np.isclose(label_values, float(label_nodata_value))
    valid_targets_mask = (~mask) & np.isin(label_values, [0.0, 1.0])
    unexpected_values_mask = (~mask) & (~np.isin(label_values, [0.0, 1.0]))
    unexpected_count = int(np.count_nonzero(unexpected_values_mask))
    if unexpected_count:
        unexpected_examples = np.unique(label_values[unexpected_values_mask])[:10]
        raise ValueError(
            "Label raster contains "
            f"{unexpected_count} unexpected non-NoData cells outside {{0, 1}}. "
            f"Example values: {unexpected_examples.tolist()}"
        )

    rows, cols = np.nonzero(valid_targets_mask)
    targets = label_values[rows, cols].astype(np.int8, copy=False)
    counts = {
        "total_raster_cells": int(label_values.size),
        "valid_label_cells": int(rows.size),
        "positive_cells_available": int(np.count_nonzero(targets == 1)),
        "negative_cells_available": int(np.count_nonzero(targets == 0)),
        "unexpected_label_value_cells": unexpected_count,
    }
    return rows.astype(np.int32), cols.astype(np.int32), targets, counts


def extract_predictor_values(
    rows: np.ndarray,
    cols: np.ndarray,
    raster_specs: list[dict[str, Any]],
    logger: logging.Logger,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    list[dict[str, Any]],
]:
    values_by_column: dict[str, np.ndarray] = {}
    invalid_by_column: dict[str, dict[str, np.ndarray]] = {}
    per_raster_invalid_rows: list[dict[str, Any]] = []

    for raster_spec in raster_specs:
        logger.info("Extracting %s values from %s.", raster_spec["kind"], raster_spec["path"])
        extracted_values, nodata_rows, nonfinite_rows = (
            extract_raster_values_at_points(
                raster_spec["path"],
                rows,
                cols,
            )
        )
        values_by_column[raster_spec["column_name"]] = extracted_values
        invalid_by_column[raster_spec["column_name"]] = {
            "nodata": nodata_rows,
            "nonfinite": nonfinite_rows,
        }
        per_raster_invalid_rows.append(
            {
                "name": raster_spec["name"],
                "column_name": raster_spec["column_name"],
                "kind": raster_spec["kind"],
                "nodata_row_count": int(np.count_nonzero(nodata_rows)),
                "nonfinite_row_count": int(np.count_nonzero(nonfinite_rows)),
            }
        )

    return values_by_column, invalid_by_column, per_raster_invalid_rows


def calculate_vif_tolerance(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    columns = list(data.columns)
    if not columns:
        return pd.DataFrame(columns=["predictor", "VIF", "tolerance"])
    if len(columns) == 1:
        return pd.DataFrame(
            [{"predictor": columns[0], "VIF": 1.0, "tolerance": 1.0}],
            columns=["predictor", "VIF", "tolerance"],
        )

    matrix = data.to_numpy(dtype=np.float64)
    for index, predictor in enumerate(columns):
        y = matrix[:, index]
        x_other = np.delete(matrix, index, axis=1)
        design = np.column_stack([np.ones(x_other.shape[0], dtype=np.float64), x_other])
        try:
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            fitted = design @ coefficients
            residual_sum_squares = float(np.sum((y - fitted) ** 2))
            total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
            if total_sum_squares <= 0:
                r_squared = 1.0
            else:
                r_squared = max(0.0, min(1.0, 1.0 - residual_sum_squares / total_sum_squares))
            tolerance = max(0.0, 1.0 - r_squared)
            vif = np.inf if tolerance <= 0 else 1.0 / tolerance
        except np.linalg.LinAlgError:
            tolerance = 0.0
            vif = np.inf
        rows.append({"predictor": predictor, "VIF": float(vif), "tolerance": float(tolerance)})
    return pd.DataFrame(rows, columns=["predictor", "VIF", "tolerance"])


def standardize_for_collinearity(values: pd.DataFrame) -> pd.DataFrame:
    standardized = values.copy()
    for column in standardized.columns:
        series = pd.to_numeric(standardized[column], errors="coerce").astype(float)
        std = float(series.std(ddof=0))
        if std <= 0 or not np.isfinite(std):
            standardized[column] = 0.0
        else:
            standardized[column] = (series - float(series.mean())) / std
    return standardized


def choose_predictor_to_drop(
    first: str,
    second: str,
    mean_abs_corr: pd.Series,
    vif: pd.Series,
    predictor_order: dict[str, int],
) -> str:
    first_priority = MULTICOLLINEARITY_DROP_PRIORITY.get(first, 0)
    second_priority = MULTICOLLINEARITY_DROP_PRIORITY.get(second, 0)
    if first_priority != second_priority:
        return first if first_priority > second_priority else second

    first_vif = float(vif.get(first, np.nan))
    second_vif = float(vif.get(second, np.nan))
    if np.isfinite(first_vif) and np.isfinite(second_vif) and not np.isclose(first_vif, second_vif):
        return first if first_vif > second_vif else second

    first_mean_corr = float(mean_abs_corr.get(first, 0.0))
    second_mean_corr = float(mean_abs_corr.get(second, 0.0))
    if not np.isclose(first_mean_corr, second_mean_corr):
        return first if first_mean_corr > second_mean_corr else second

    return first if predictor_order[first] > predictor_order[second] else second


def write_pearson_heatmap(
    pearson_matrix: pd.DataFrame,
    output_path: Path,
    dropped_predictors: set[str],
) -> None:
    if pearson_matrix.empty:
        return
    labels = list(pearson_matrix.columns)
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.8), max(6.0, len(labels) * 0.7)))
    image = ax.imshow(pearson_matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Step 2 Retained Predictor Pearson Correlation Matrix")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for y_index, row_name in enumerate(labels):
        for x_index, column_name in enumerate(labels):
            value = pearson_matrix.iloc[y_index, x_index]
            text_color = "white" if abs(float(value)) >= 0.65 else "black"
            ax.text(x_index, y_index, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)
        if row_name in dropped_predictors:
            ax.get_yticklabels()[y_index].set_color("#b00020")
            ax.get_yticklabels()[y_index].set_fontweight("bold")
            ax.get_xticklabels()[y_index].set_color("#b00020")
            ax.get_xticklabels()[y_index].set_fontweight("bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson r")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_vif_tolerance_plot(
    vif_table: pd.DataFrame,
    output_path: Path,
    dropped_predictors: set[str],
) -> None:
    if vif_table.empty:
        return
    plot_df = vif_table.copy()
    labels = plot_df["predictor"].astype(str).tolist()
    x = np.arange(len(labels))
    colors = ["#b00020" if label in dropped_predictors else "#2f6f9f" for label in labels]
    finite_vif = pd.to_numeric(plot_df["VIF"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    max_finite_vif = float(finite_vif.max()) if finite_vif.notna().any() else VIF_THRESHOLD
    vif_cap = max(VIF_THRESHOLD * 1.25, max_finite_vif * 1.1)
    vif_values = pd.to_numeric(plot_df["VIF"], errors="coerce").replace([np.inf, -np.inf], vif_cap).fillna(0.0)
    tolerance_values = pd.to_numeric(plot_df["tolerance"], errors="coerce").fillna(0.0)

    fig, axes = plt.subplots(2, 1, figsize=(max(9.0, len(labels) * 0.8), 8.0), sharex=True)
    axes[0].bar(x, vif_values, color=colors)
    axes[0].axhline(VIF_THRESHOLD, color="#c1121f", linestyle="--", linewidth=1.4, label=f"VIF threshold = {VIF_THRESHOLD:g}")
    axes[0].set_ylabel("VIF")
    axes[0].set_title("Step 2 Retained Predictor VIF and Tolerance Screening")
    axes[0].legend(frameon=False)

    axes[1].bar(x, tolerance_values, color=colors)
    axes[1].axhline(
        TOLERANCE_THRESHOLD,
        color="#c1121f",
        linestyle="--",
        linewidth=1.4,
        label=f"Tolerance threshold = {TOLERANCE_THRESHOLD:g}",
    )
    axes[1].set_ylabel("Tolerance")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_multicollinearity_screening(
    predictor_values: dict[str, np.ndarray],
    predictor_specs: list[dict[str, Any]],
    output_dir: Path,
    logger: logging.Logger,
    configured_categorical_predictors: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    logger.info(
        "Starting predictor multicollinearity screening with thresholds: |Pearson r| >= %.2f, VIF >= %.1f, tolerance <= %.2f.",
        PEARSON_ABS_THRESHOLD,
        VIF_THRESHOLD,
        TOLERANCE_THRESHOLD,
    )
    if not predictor_specs:
        raise ValueError("No predictor rasters were configured.")

    row_count = len(predictor_values[predictor_specs[0]["column_name"]])
    if len(predictor_specs) < 2 or row_count == 0:
        summary_path = output_dir / OUTPUTS["multicollinearity_summary"]
        pd.DataFrame(
            [
                {
                    "predictor": spec["column_name"],
                    "status": "retained",
                    "test_role": "not_enough_predictors_for_test",
                    "drop_reason": "",
                }
                for spec in predictor_specs
            ]
        ).to_csv(summary_path, index=False)
        return predictor_specs, {"summary_path": summary_path, "dropped_predictors": []}

    sample_indices = np.arange(row_count, dtype=np.int64)
    if sample_indices.size > MULTICOLLINEARITY_MAX_SAMPLE_ROWS:
        rng = np.random.default_rng(MULTICOLLINEARITY_RANDOM_SEED)
        sample_indices = np.sort(rng.choice(sample_indices, size=MULTICOLLINEARITY_MAX_SAMPLE_ROWS, replace=False))
    values_by_predictor: dict[str, np.ndarray] = {}
    predictor_roles: dict[str, str] = {}
    valid_fraction_by_predictor: dict[str, float] = {}
    constant_predictors: set[str] = set()
    continuous_predictors: list[str] = []

    for spec in predictor_specs:
        column = spec["column_name"]
        values = np.asarray(
            predictor_values[column][sample_indices],
            dtype=np.float64,
        ).copy()
        values[~np.isfinite(values)] = np.nan
        values_by_predictor[column] = values
        finite_values = values[np.isfinite(values)]
        valid_fraction_by_predictor[column] = float(finite_values.size / values.size) if values.size else 0.0
        if finite_values.size < 3 or np.nanstd(finite_values) <= 0:
            predictor_roles[column] = "constant_or_insufficient_valid_values"
            constant_predictors.add(column)
        elif column in configured_categorical_predictors:
            predictor_roles[column] = "configured_categorical_skipped"
        else:
            predictor_roles[column] = "continuous_tested"
            continuous_predictors.append(column)

    continuous_df = pd.DataFrame({column: values_by_predictor[column] for column in continuous_predictors})
    complete_continuous_df = continuous_df.dropna(axis=0, how="any") if not continuous_df.empty else continuous_df
    if len(complete_continuous_df) < 3 or len(continuous_predictors) < 2:
        logger.warning(
            "Multicollinearity screening skipped automatic dropping because fewer than two continuous predictors had enough complete sampled rows."
        )
        retained_specs = [spec for spec in predictor_specs if spec["column_name"] not in constant_predictors]
        summary_rows = []
        for spec in predictor_specs:
            column = spec["column_name"]
            status = "dropped" if column in constant_predictors else "retained"
            summary_rows.append(
                {
                    "predictor": column,
                    "status": status,
                    "test_role": predictor_roles.get(column, "not_tested"),
                    "drop_reason": "constant_or_insufficient_valid_values" if column in constant_predictors else "",
                    "valid_fraction_in_screening_sample": valid_fraction_by_predictor.get(column, np.nan),
                }
            )
        summary_path = output_dir / OUTPUTS["multicollinearity_summary"]
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        return retained_specs, {"summary_path": summary_path, "dropped_predictors": sorted(constant_predictors)}

    standardized_df = standardize_for_collinearity(complete_continuous_df)
    selected = list(standardized_df.columns)
    dropped_reasons: dict[str, str] = {column: "constant_or_insufficient_valid_values" for column in constant_predictors}
    predictor_order = {spec["column_name"]: index for index, spec in enumerate(predictor_specs)}

    while len(selected) >= 2:
        current = standardized_df[selected]
        corr = current.corr(method="pearson")
        abs_corr_values = corr.abs().to_numpy(dtype=float, copy=True)
        np.fill_diagonal(abs_corr_values, 0.0)
        abs_corr = pd.DataFrame(abs_corr_values, index=corr.index, columns=corr.columns)
        max_corr = float(abs_corr.max().max())
        current_vif = calculate_vif_tolerance(current)
        vif_series = current_vif.set_index("predictor")["VIF"]
        mean_abs_corr = abs_corr.replace(0.0, np.nan).mean(axis=1).fillna(0.0)

        if max_corr >= PEARSON_ABS_THRESHOLD:
            pair_indices = np.argwhere(abs_corr.to_numpy(dtype=float) == max_corr)[0]
            first = selected[int(pair_indices[0])]
            second = selected[int(pair_indices[1])]
            drop = choose_predictor_to_drop(first, second, mean_abs_corr, vif_series, predictor_order)
            kept = second if drop == first else first
            dropped_reasons[drop] = (
                f"high Pearson correlation with {kept}: |r|={max_corr:.3f} >= {PEARSON_ABS_THRESHOLD:.2f}"
            )
            selected.remove(drop)
            logger.info("Dropped predictor '%s' during Pearson screening: %s", drop, dropped_reasons[drop])
            continue

        current_vif = calculate_vif_tolerance(current)
        finite_or_inf_vif = pd.to_numeric(current_vif["VIF"], errors="coerce")
        highest_vif_index = int(finite_or_inf_vif.replace(np.inf, np.finfo(np.float64).max).idxmax())
        highest_vif_row = current_vif.loc[highest_vif_index]
        highest_vif = float(highest_vif_row["VIF"])
        lowest_tolerance = float(pd.to_numeric(current_vif["tolerance"], errors="coerce").min())
        if highest_vif >= VIF_THRESHOLD or lowest_tolerance <= TOLERANCE_THRESHOLD:
            drop = str(highest_vif_row["predictor"])
            dropped_reasons[drop] = (
                f"high VIF/low tolerance: VIF={highest_vif:.3f}, tolerance={float(highest_vif_row['tolerance']):.3f}"
            )
            selected.remove(drop)
            logger.info("Dropped predictor '%s' during VIF/TOL screening: %s", drop, dropped_reasons[drop])
            continue

        break

    final_pearson = standardized_df[selected].corr(method="pearson") if selected else pd.DataFrame()
    final_vif = calculate_vif_tolerance(standardized_df[selected]) if selected else pd.DataFrame()
    final_vif_by_predictor = final_vif.set_index("predictor") if not final_vif.empty else pd.DataFrame()
    final_max_abs_corr = final_pearson.abs().copy()
    if not final_max_abs_corr.empty:
        final_max_abs_corr_values = final_max_abs_corr.to_numpy(dtype=float, copy=True)
        np.fill_diagonal(final_max_abs_corr_values, 0.0)
        final_max_abs_corr = pd.DataFrame(
            final_max_abs_corr_values,
            index=final_max_abs_corr.index,
            columns=final_max_abs_corr.columns,
        )

    dropped_predictors = set(dropped_reasons)
    retained_specs = [spec for spec in predictor_specs if spec["column_name"] not in dropped_predictors]
    summary_rows = []
    for spec in predictor_specs:
        column = spec["column_name"]
        status = "dropped" if column in dropped_predictors else "retained"
        summary_rows.append(
            {
                "predictor": column,
                "status": status,
                "test_role": predictor_roles.get(column, "not_tested"),
                "drop_reason": dropped_reasons.get(column, ""),
                "screening_sample_rows": int(len(complete_continuous_df)) if column in continuous_predictors else int(sample_indices.size),
                "valid_fraction_in_screening_sample": valid_fraction_by_predictor.get(column, np.nan),
                "max_abs_pearson_final_if_retained": (
                    float(final_max_abs_corr.loc[column].max())
                    if column in final_max_abs_corr.index
                    else np.nan
                ),
                "VIF_final_if_retained": (
                    float(final_vif_by_predictor.loc[column, "VIF"])
                    if column in final_vif_by_predictor.index
                    else np.nan
                ),
                "tolerance_final_if_retained": (
                    float(final_vif_by_predictor.loc[column, "tolerance"])
                    if column in final_vif_by_predictor.index
                    else np.nan
                ),
                "pearson_abs_threshold": PEARSON_ABS_THRESHOLD,
                "VIF_threshold": VIF_THRESHOLD,
                "tolerance_threshold": TOLERANCE_THRESHOLD,
            }
        )

    summary_path = output_dir / OUTPUTS["multicollinearity_summary"]
    pearson_path = output_dir / OUTPUTS["multicollinearity_pearson_matrix"]
    heatmap_path = output_dir / OUTPUTS["multicollinearity_pearson_heatmap"]
    vif_plot_path = output_dir / OUTPUTS["multicollinearity_vif_tolerance"]
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    final_pearson.to_csv(pearson_path)
    write_pearson_heatmap(final_pearson, heatmap_path, set())
    write_vif_tolerance_plot(final_vif, vif_plot_path, set())

    if not retained_specs:
        raise ValueError("Multicollinearity screening removed all predictors. Review thresholds or predictor set.")
    logger.info(
        "Multicollinearity screening retained %d/%d predictors. Dropped: %s",
        len(retained_specs),
        len(predictor_specs),
        ", ".join(sorted(dropped_predictors)) if dropped_predictors else "none",
    )
    logger.info("Multicollinearity summary path: %s", summary_path)
    logger.info("Pearson matrix path: %s", pearson_path)
    logger.info("Pearson heatmap path: %s", heatmap_path)
    logger.info("VIF/TOL plot path: %s", vif_plot_path)
    return retained_specs, {
        "summary_path": summary_path,
        "pearson_matrix_path": pearson_path,
        "pearson_heatmap_path": heatmap_path,
        "vif_tolerance_plot_path": vif_plot_path,
        "dropped_predictors": sorted(dropped_predictors),
    }


def remove_invalid_predictor_rows(
    rows: np.ndarray,
    cols: np.ndarray,
    targets: np.ndarray,
    predictor_values: dict[str, np.ndarray],
    group_values: dict[str, np.ndarray],
    predictor_nodata_rows: np.ndarray,
    predictor_nonfinite_rows: np.ndarray,
    group_nodata_rows: np.ndarray,
    group_nonfinite_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    keep_mask = ~(
        predictor_nodata_rows
        | predictor_nonfinite_rows
        | group_nodata_rows
        | group_nonfinite_rows
    )

    filtered_predictors: dict[str, np.ndarray] = {}
    for column_name, values in predictor_values.items():
        filtered_predictors[column_name] = values[keep_mask]

    filtered_groups: dict[str, np.ndarray] = {}
    for column_name, values in group_values.items():
        filtered_groups[column_name] = values[keep_mask]

    stats = {
        "rows_before_predictor_filtering": int(rows.size),
        "rows_dropped_due_to_predictor_nodata": int(np.count_nonzero(predictor_nodata_rows)),
        "rows_dropped_due_to_nonfinite_predictors": int(np.count_nonzero(predictor_nonfinite_rows)),
        "rows_dropped_due_to_group_nodata": int(np.count_nonzero(group_nodata_rows)),
        "rows_dropped_due_to_nonfinite_groups": int(np.count_nonzero(group_nonfinite_rows)),
        "rows_dropped_total": int(np.count_nonzero(~keep_mask)),
        "rows_after_predictor_filtering": int(np.count_nonzero(keep_mask)),
    }
    return (
        rows[keep_mask],
        cols[keep_mask],
        targets[keep_mask],
        filtered_predictors,
        filtered_groups,
        stats,
    )


def aggregate_invalid_rows(
    invalid_by_column: dict[str, dict[str, np.ndarray]],
    columns: list[str],
    row_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    nodata_rows = np.zeros(row_count, dtype=bool)
    nonfinite_rows = np.zeros(row_count, dtype=bool)
    for column in columns:
        invalid = invalid_by_column[column]
        nodata_rows |= invalid["nodata"]
        nonfinite_rows |= invalid["nonfinite"]
    return nodata_rows, nonfinite_rows


def validate_group_values(
    group_values: dict[str, np.ndarray],
    invalid_by_column: dict[str, dict[str, np.ndarray]],
    group_specs: list[dict[str, Any]],
) -> None:
    errors: list[str] = []
    for spec in group_specs:
        column = spec["column_name"]
        values = np.asarray(group_values[column], dtype=np.float64)
        invalid = invalid_by_column[column]["nodata"] | invalid_by_column[column]["nonfinite"]
        if np.any(invalid):
            errors.append(
                f"{column} has {int(np.count_nonzero(invalid))} missing or non-finite values "
                "at labeled cells"
            )
            continue
        if column == "landslide_id":
            invalid_ids = (values <= 0) | (~np.isclose(values, np.round(values)))
            if np.any(invalid_ids):
                examples = np.unique(values[invalid_ids])[:10]
                errors.append(
                    f"landslide_id has {int(np.count_nonzero(invalid_ids))} non-positive or "
                    f"non-integer values; examples: {examples.tolist()}"
                )
        if np.unique(values).size < 2:
            errors.append(f"{column} contains fewer than two distinct groups")
    if errors:
        raise ValueError(
            "Required group raster validation failed. Step 3 grouped validation cannot "
            "proceed safely:\n- " + "\n- ".join(errors)
        )


def summarize_extracted_samples(targets: np.ndarray) -> tuple[np.ndarray, dict[str, int | float]]:
    positive_indices = np.flatnonzero(targets == 1)
    negative_indices = np.flatnonzero(targets == 0)
    if positive_indices.size == 0:
        raise ValueError("No positive depositional pixels remain after filtering.")
    if negative_indices.size == 0:
        raise ValueError("No negative non-depositional pixels remain after filtering.")
    selected = np.arange(targets.size, dtype=np.int64)

    summary = {
        "positive_samples_exported": int(positive_indices.size),
        "negative_samples_exported": int(negative_indices.size),
        "final_sample_count": int(selected.size),
        "positive_percentage": float(100.0 * positive_indices.size / selected.size),
        "negative_percentage": float(100.0 * negative_indices.size / selected.size),
    }
    return selected, summary


def build_sample_dataframe(
    selected_indices: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    targets: np.ndarray,
    predictor_values: dict[str, np.ndarray],
    group_values: dict[str, np.ndarray],
    predictor_specs: list[dict[str, Any]],
    group_specs: list[dict[str, Any]],
    transform: Any,
    include_coordinates: bool,
    include_row_col: bool,
) -> pd.DataFrame:
    data: dict[str, Any] = {
        "sample_id": np.arange(1, selected_indices.size + 1, dtype=np.int64),
    }

    selected_rows = rows[selected_indices]
    selected_cols = cols[selected_indices]
    selected_targets = targets[selected_indices].astype(np.int8, copy=False)

    if include_row_col:
        data["row"] = selected_rows.astype(np.int32, copy=False)
        data["col"] = selected_cols.astype(np.int32, copy=False)
    if include_coordinates:
        xs, ys = transform_xy(transform, selected_rows, selected_cols, offset="center")
        data["x"] = np.asarray(xs, dtype=np.float64)
        data["y"] = np.asarray(ys, dtype=np.float64)

    data["target"] = selected_targets

    for spec in group_specs:
        selected_values = group_values[spec["column_name"]][selected_indices]
        if np.all(~np.isfinite(selected_values) | np.isclose(selected_values, np.round(selected_values))):
            data[spec["column_name"]] = pd.array(
                [None if not np.isfinite(value) else int(round(float(value))) for value in selected_values],
                dtype="Int64",
            )
        else:
            data[spec["column_name"]] = selected_values.astype(np.float64, copy=False)

    for spec in predictor_specs:
        data[spec["column_name"]] = predictor_values[spec["column_name"]][selected_indices].astype(
            np.float64,
            copy=False,
        )

    ordered_columns = ["sample_id"]
    if include_row_col:
        ordered_columns.extend(["row", "col"])
    if include_coordinates:
        ordered_columns.extend(["x", "y"])
    ordered_columns.append("target")
    ordered_columns.extend(spec["column_name"] for spec in group_specs)
    ordered_columns.extend(spec["column_name"] for spec in predictor_specs)

    return pd.DataFrame(data, columns=ordered_columns)


def write_output_table(
    sample_df: pd.DataFrame,
    output_dir: Path,
    output_table_name: str,
    output_format: str,
    compress_outputs: bool,
    logger: logging.Logger,
) -> dict[str, Path | None]:
    csv_path = output_dir / f"{output_table_name}.csv"
    parquet_path = output_dir / f"{output_table_name}.parquet"
    written_paths: dict[str, Path | None] = {"csv": None, "parquet": None, "primary": None}

    should_write_csv = output_format in {"csv", "both"}
    should_write_parquet = output_format in {"parquet", "both"}
    parquet_failed = False

    if should_write_parquet:
        parquet_compression = "snappy" if compress_outputs else None
        try:
            sample_df.to_parquet(parquet_path, index=False, compression=parquet_compression)
            written_paths["parquet"] = parquet_path
            written_paths["primary"] = parquet_path
            logger.info("Wrote Parquet output: %s", parquet_path)
        except Exception as exc:
            parquet_failed = True
            logger.warning(
                "Could not write Parquet output (%s). Falling back to CSV where needed.",
                exc,
            )

    if should_write_csv or parquet_failed:
        if compress_outputs and output_format == "csv":
            logger.info("CSV compression was not applied so the output keeps the standard .csv filename.")
        sample_df.to_csv(csv_path, index=False)
        written_paths["csv"] = csv_path
        if written_paths["primary"] is None:
            written_paths["primary"] = csv_path
        logger.info("Wrote CSV output: %s", csv_path)

    if written_paths["primary"] is None:
        raise RuntimeError("No output table was written.")
    return written_paths


def write_summary_csv(
    output_path: Path,
    label_counts: dict[str, int],
    predictor_filter_stats: dict[str, int],
    sample_summary: dict[str, int | float],
    predictor_specs: list[dict[str, Any]],
    multicollinearity_outputs: dict[str, Any],
    output_table_path: Path,
) -> None:
    row = {
        "total_raster_cells": int(label_counts["total_raster_cells"]),
        "valid_label_cells": int(label_counts["valid_label_cells"]),
        "positive_depositional_cells_available": int(label_counts["positive_cells_available"]),
        "negative_non_depositional_cells_available": int(label_counts["negative_cells_available"]),
        "positive_samples_exported": int(sample_summary["positive_samples_exported"]),
        "negative_samples_exported": int(sample_summary["negative_samples_exported"]),
        "final_sample_count": int(sample_summary["final_sample_count"]),
        "positive_percentage": float(sample_summary["positive_percentage"]),
        "negative_percentage": float(sample_summary["negative_percentage"]),
        "predictor_names": ";".join(spec["column_name"] for spec in predictor_specs),
        "predictors_dropped_by_multicollinearity": ";".join(
            map(str, multicollinearity_outputs.get("dropped_predictors", []))
        ),
        "multicollinearity_summary_path": str(multicollinearity_outputs.get("summary_path", "")),
        "rows_dropped_due_to_predictor_nodata": int(predictor_filter_stats["rows_dropped_due_to_predictor_nodata"]),
        "rows_dropped_due_to_nonfinite_predictors": int(
            predictor_filter_stats["rows_dropped_due_to_nonfinite_predictors"]
        ),
        "rows_dropped_due_to_group_nodata": int(
            predictor_filter_stats["rows_dropped_due_to_group_nodata"]
        ),
        "rows_dropped_due_to_nonfinite_groups": int(
            predictor_filter_stats["rows_dropped_due_to_nonfinite_groups"]
        ),
        "rows_dropped_total": int(predictor_filter_stats["rows_dropped_total"]),
        "output_file_path": str(output_table_path),
    }
    pd.DataFrame([row]).to_csv(output_path, index=False)


def write_predictor_alignment_report(output_path: Path, alignment_rows: list[dict[str, Any]]) -> None:
    report_df = pd.DataFrame(alignment_rows)
    report_df.to_csv(output_path, index=False)


def run_qc_checks(
    sample_df: pd.DataFrame,
    predictor_specs: list[dict[str, Any]],
    group_specs: list[dict[str, Any]],
    output_paths: dict[str, Path | None],
    alignment_rows: list[dict[str, Any]],
) -> None:
    errors: list[str] = []

    primary_output = output_paths.get("primary")
    if primary_output is None or not Path(primary_output).exists():
        errors.append("Output table was not written.")
    if "target" not in sample_df.columns:
        errors.append("Output table is missing the target column.")
    predictor_columns = [spec["column_name"] for spec in predictor_specs]
    if not predictor_columns:
        errors.append("No predictor columns were configured.")
    for predictor_column in predictor_columns:
        if predictor_column not in sample_df.columns:
            errors.append(f"Predictor column missing from output table: {predictor_column}")

    group_columns = [spec["column_name"] for spec in group_specs]
    if not group_columns:
        errors.append("No group columns were configured. At least one group column is mandatory.")
    for group_column in group_columns:
        if group_column not in sample_df.columns:
            errors.append(f"Group column missing from output table: {group_column}")
            continue
        values = pd.to_numeric(sample_df[group_column], errors="coerce").to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(values)
        if not bool(np.all(finite_mask)):
            errors.append(f"Group column contains missing or non-finite values: {group_column}")
        unique_group_count = int(np.unique(values[finite_mask]).size)
        if unique_group_count < 2:
            errors.append(
                f"Group column must contain at least two groups for grouped validation: {group_column}"
            )

    unique_targets = set(pd.unique(sample_df["target"]).tolist()) if "target" in sample_df.columns else set()
    if unique_targets - {0, 1}:
        errors.append(f"Unexpected target values present: {sorted(unique_targets)}")
    if 1 not in unique_targets:
        errors.append("QC failed: no positive target samples were exported.")
    if 0 not in unique_targets:
        errors.append("QC failed: no negative target samples were exported.")

    for predictor_column in predictor_columns:
        values = pd.to_numeric(sample_df[predictor_column], errors="coerce").to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(values)):
            errors.append(f"Predictor column contains NaN or infinite values after filtering: {predictor_column}")

    if any(row["status"] != "OK" for row in alignment_rows):
        errors.append("At least one predictor or group raster failed alignment checks.")

    if errors:
        raise RuntimeError("QC checks failed:\n- " + "\n- ".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pixel-based ML-DFI training samples from a label raster and aligned predictors."
    )
    parser.add_argument("--config-json", help="Optional JSON config. Uses the extract_pixel_samples section if present.")
    parser.add_argument("--label-raster-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-table-name")
    parser.add_argument("--output-format")
    parser.add_argument("--label-nodata-value", type=int)
    parser.add_argument("--disable-coordinates", action="store_true")
    parser.add_argument("--disable-row-col", action="store_true")
    parser.add_argument("--disable-compression", action="store_true")
    return parser.parse_args()


def resolve_runtime_path(root: Path, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    return str(path.resolve() if path.is_absolute() else (root / path).resolve())


def resolve_step2_config_paths(config: dict[str, Any], root: Path) -> dict[str, Any]:
    resolved = dict(config)
    for key in ("label_raster_path", "output_dir"):
        resolved[key] = resolve_runtime_path(root, resolved.get(key, ""))
    for key in ("predictor_rasters", "group_rasters"):
        values = resolved.get(key, {})
        if not isinstance(values, dict):
            continue
        resolved[key] = {
            str(name): resolve_runtime_path(root, path)
            for name, path in values.items()
        }
    return resolved


def resolve_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    runtime = dict(CONFIG)
    runtime["predictor_rasters"] = dict(CONFIG["predictor_rasters"])
    runtime["group_rasters"] = dict(CONFIG["group_rasters"])

    if args.config_json:
        config_path = Path(args.config_json).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config JSON does not exist: {config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            loaded_config = json.load(file)
        loaded_section = loaded_config.get("extract_pixel_samples", loaded_config)
        if not isinstance(loaded_section, dict):
            raise ValueError("Config JSON must contain an object or an extract_pixel_samples object.")
        if "optional_group_rasters" in loaded_section:
            if "group_rasters" in loaded_section:
                raise ValueError(
                    "Use only group_rasters; do not provide both group_rasters "
                    "and the legacy optional_group_rasters key."
                )
            loaded_section = dict(loaded_section)
            loaded_section["group_rasters"] = loaded_section.pop(
                "optional_group_rasters"
            )
        unknown_keys = sorted(set(loaded_section) - set(CONFIG))
        if unknown_keys:
            raise ValueError(
                "Unknown Step 2 config key(s): " + ", ".join(unknown_keys)
            )
        loaded_section = resolve_step2_config_paths(dict(loaded_section), config_path.parent)
        for key, value in loaded_section.items():
            if key == "predictor_rasters" and isinstance(value, dict):
                runtime["predictor_rasters"] = dict(value)
            elif key == "group_rasters" and isinstance(value, dict):
                runtime["group_rasters"] = dict(value)
            else:
                runtime[key] = value

    overrides = {
        "label_raster_path": args.label_raster_path,
        "output_dir": args.output_dir,
        "output_table_name": args.output_table_name,
        "output_format": args.output_format,
        "label_nodata_value": args.label_nodata_value,
    }
    for key, value in overrides.items():
        if value is not None:
            runtime[key] = value

    if args.disable_coordinates:
        runtime["include_coordinates"] = False
    if args.disable_row_col:
        runtime["include_row_col"] = False
    if args.disable_compression:
        runtime["compress_outputs"] = False
    return resolve_step2_config_paths(runtime, Path.cwd())


def resolve_categorical_predictor_columns(
    configured_names: list[str],
    predictor_specs: list[dict[str, Any]],
) -> set[str]:
    requested = {name.strip().lower() for name in configured_names}
    matched: set[str] = set()
    columns: set[str] = set()
    for spec in predictor_specs:
        candidates = {
            str(spec["name"]).strip().lower(),
            str(spec["column_name"]).strip().lower(),
        }
        matching_names = requested & candidates
        if matching_names:
            matched |= matching_names
            columns.add(spec["column_name"])
    unknown = sorted(requested - matched)
    if unknown:
        raise ValueError(
            "categorical_predictors contains names that are not configured predictors: "
            + ", ".join(unknown)
        )
    return columns


def write_run_config(
    output_path: Path,
    config: dict[str, Any],
    reference: dict[str, Any],
    configured_predictor_specs: list[dict[str, Any]],
    retained_predictor_specs: list[dict[str, Any]],
    group_specs: list[dict[str, Any]],
    multicollinearity_outputs: dict[str, Any],
    final_output_names: list[str],
) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "step2_extract_pixel_samples",
        "behavior": {
            "alignment": "strict_manual_precondition",
            "extraction": "all_valid_labeled_rows",
            "predictor_invalid_rows": "always_drop",
            "group_validation": "required_complete_positive_integer_landslide_id",
        },
        "config": config,
        "reference_grid": {
            "path": reference["path"],
            "crs": str(reference["crs"]),
            "transform": list(reference["transform"]),
            "width": reference["width"],
            "height": reference["height"],
            "nodata": reference["nodata"],
        },
        "configured_predictors": [
            spec["column_name"] for spec in configured_predictor_specs
        ],
        "retained_predictors": [
            spec["column_name"] for spec in retained_predictor_specs
        ],
        "dropped_predictors": multicollinearity_outputs.get(
            "dropped_predictors",
            [],
        ),
        "group_columns": [spec["column_name"] for spec in group_specs],
        "output_files": sorted(final_output_names),
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    args = parse_args()
    logger: logging.Logger | None = None
    staging_dir: Path | None = None
    staged_log_path: Path | None = None
    output_dir: Path | None = None

    try:
        config = resolve_runtime_config(args)
        validate_config(config)
        output_dir = Path(str(config["output_dir"])).expanduser().resolve()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.step2-stage-",
                dir=output_dir.parent,
            )
        )
        logger, staged_log_path = setup_logging(staging_dir)

        if "PROJ" in CONFIGURED_GEOSPATIAL_DIRS:
            logger.info("Using PROJ data directory: %s", CONFIGURED_GEOSPATIAL_DIRS["PROJ"])
        if "GDAL" in CONFIGURED_GEOSPATIAL_DIRS:
            logger.info("Using GDAL data directory: %s", CONFIGURED_GEOSPATIAL_DIRS["GDAL"])

        logger.info("Starting Step 2 pixel sample extraction.")
        logger.info("Label raster: %s", config["label_raster_path"])
        logger.info("Output directory: %s", output_dir)
        logger.info("Output format: %s", config["output_format"])
        logger.info("Extraction mode: all valid labeled rows after predictor filtering.")

        predictor_specs, group_specs = collect_predictor_paths(
            predictor_rasters=dict(config["predictor_rasters"]),
            group_rasters=dict(config["group_rasters"]),
            logger=logger,
        )
        configured_predictor_specs = list(predictor_specs)
        categorical_predictor_columns = resolve_categorical_predictor_columns(
            list(config["categorical_predictors"]),
            predictor_specs,
        )
        logger.info("Predictor rasters retained: %d", len(predictor_specs))
        logger.info("Required group rasters retained: %d", len(group_specs))
        logger.info(
            "Configured categorical predictors excluded from Pearson/VIF screening: %s",
            ", ".join(sorted(categorical_predictor_columns)) or "none",
        )

        reference, label_array = open_reference_label_raster(
            label_raster_path=str(config["label_raster_path"]),
            config_label_nodata_value=int(config["label_nodata_value"]),
            logger=logger,
        )

        alignment_rows: list[dict[str, Any]] = []
        alignment_errors: list[str] = []
        for predictor_spec in predictor_specs:
            alignment_row = check_raster_alignment(reference, predictor_spec)
            alignment_rows.append(alignment_row)
            logger.info(
                "Predictor alignment %s: status=%s crs_match=%s transform_match=%s width_match=%s height_match=%s",
                predictor_spec["name"],
                alignment_row["status"],
                alignment_row["crs_match"],
                alignment_row["transform_match"],
                alignment_row["width_match"],
                alignment_row["height_match"],
            )
            if alignment_row["status"] != "OK":
                alignment_errors.append(
                    f"Predictor raster '{predictor_spec['name']}' is not aligned with label raster. "
                    "Manually align predictors before running Step 2."
                )

        for group_spec in group_specs:
            group_alignment = check_raster_alignment(reference, group_spec)
            alignment_rows.append(group_alignment)
            logger.info(
                "Group alignment %s: status=%s crs_match=%s transform_match=%s width_match=%s height_match=%s",
                group_spec["name"],
                group_alignment["status"],
                group_alignment["crs_match"],
                group_alignment["transform_match"],
                group_alignment["width_match"],
                group_alignment["height_match"],
            )
            if group_alignment["status"] != "OK":
                alignment_errors.append(
                    f"Required group raster '{group_spec['name']}' is not aligned with label raster. "
                    "Manually align group rasters before running Step 2."
                )

        alignment_report_path = staging_dir / OUTPUTS["alignment_report"]
        write_predictor_alignment_report(alignment_report_path, alignment_rows)

        if alignment_errors:
            raise ValueError("\n".join(alignment_errors))

        rows, cols, targets, label_counts = extract_valid_label_indices(
            label_array=label_array,
            label_nodata_value=reference["nodata"],
            logger=logger,
        )
        logger.info("Valid label cells: %d", label_counts["valid_label_cells"])
        logger.info("Available positive cells: %d", label_counts["positive_cells_available"])
        logger.info("Available negative cells: %d", label_counts["negative_cells_available"])

        all_predictor_values, all_predictor_invalid, predictor_invalid_summary = (
            extract_predictor_values(
                rows=rows,
                cols=cols,
                raster_specs=predictor_specs,
                logger=logger,
            )
        )
        predictor_specs, multicollinearity_outputs = run_multicollinearity_screening(
            predictor_values=all_predictor_values,
            predictor_specs=predictor_specs,
            output_dir=staging_dir,
            logger=logger,
            configured_categorical_predictors=categorical_predictor_columns,
        )
        if multicollinearity_outputs["dropped_predictors"]:
            logger.info(
                "Predictors retained after multicollinearity filtering: %s",
                ", ".join(spec["column_name"] for spec in predictor_specs),
            )

        # Scientific note:
        # This step builds a pixel-based supervised learning table for ML-DFI. The target is
        # depositional terrain favorability, not landslide initiation susceptibility. Positive
        # samples represent observed depositional-zone pixels, and negative samples represent
        # upper source/transport-zone pixels inside mapped landslides.
        predictor_values = {
            spec["column_name"]: all_predictor_values[spec["column_name"]]
            for spec in predictor_specs
        }
        retained_predictor_columns = [
            spec["column_name"] for spec in predictor_specs
        ]
        predictor_nodata_rows, predictor_nonfinite_rows = aggregate_invalid_rows(
            all_predictor_invalid,
            retained_predictor_columns,
            rows.size,
        )

        group_values, group_invalid_by_column, group_invalid_summary = extract_predictor_values(
            rows=rows,
            cols=cols,
            raster_specs=group_specs,
            logger=logger,
        )
        validate_group_values(group_values, group_invalid_by_column, group_specs)
        group_columns = [spec["column_name"] for spec in group_specs]
        group_nodata_rows, group_nonfinite_rows = aggregate_invalid_rows(
            group_invalid_by_column,
            group_columns,
            rows.size,
        )

        for invalid_summary in predictor_invalid_summary + group_invalid_summary:
            logger.info(
                "%s '%s': nodata_rows=%d nonfinite_rows=%d",
                invalid_summary["kind"],
                invalid_summary["name"],
                invalid_summary["nodata_row_count"],
                invalid_summary["nonfinite_row_count"],
            )

        rows, cols, targets, predictor_values, group_values, predictor_filter_stats = remove_invalid_predictor_rows(
            rows=rows,
            cols=cols,
            targets=targets,
            predictor_values=predictor_values,
            group_values=group_values,
            predictor_nodata_rows=predictor_nodata_rows,
            predictor_nonfinite_rows=predictor_nonfinite_rows,
            group_nodata_rows=group_nodata_rows,
            group_nonfinite_rows=group_nonfinite_rows,
        )
        logger.info("Rows after predictor filtering: %d", predictor_filter_stats["rows_after_predictor_filtering"])

        if rows.size == 0:
            raise ValueError("All rows were removed during predictor filtering. Check predictor NoData coverage.")

        selected_indices, sample_summary = summarize_extracted_samples(targets)

        # Scientific note:
        # Later model validation should not use random pixel splitting because nearby pixels are
        # spatially autocorrelated. Step 2 requires grouped samples so Step 3 can keep whole
        # landslides separated during validation.
        sample_df = build_sample_dataframe(
            selected_indices=selected_indices,
            rows=rows,
            cols=cols,
            targets=targets,
            predictor_values=predictor_values,
            group_values=group_values,
            predictor_specs=predictor_specs,
            group_specs=group_specs,
            transform=reference["transform"],
            include_coordinates=bool(config["include_coordinates"]),
            include_row_col=bool(config["include_row_col"]),
        )

        output_paths = write_output_table(
            sample_df=sample_df,
            output_dir=staging_dir,
            output_table_name=str(config["output_table_name"]),
            output_format=str(config["output_format"]).lower(),
            compress_outputs=bool(config["compress_outputs"]),
            logger=logger,
        )

        summary_csv_path = staging_dir / f"{str(config['output_table_name'])}_summary.csv"
        final_multicollinearity_outputs = dict(multicollinearity_outputs)
        for key, value in list(final_multicollinearity_outputs.items()):
            if key.endswith("_path") and value:
                final_multicollinearity_outputs[key] = output_dir / Path(value).name
        write_summary_csv(
            output_path=summary_csv_path,
            label_counts=label_counts,
            predictor_filter_stats=predictor_filter_stats,
            sample_summary=sample_summary,
            predictor_specs=predictor_specs,
            multicollinearity_outputs=final_multicollinearity_outputs,
            output_table_path=(
                output_dir / Path(output_paths["primary"]).name
                if output_paths["primary"] is not None
                else output_dir / summary_csv_path.name
            ),
        )

        run_qc_checks(
            sample_df=sample_df,
            predictor_specs=predictor_specs,
            group_specs=group_specs,
            output_paths=output_paths,
            alignment_rows=alignment_rows,
        )

        staged_paths: dict[str, Path] = {
            "alignment_report": alignment_report_path,
            "summary": summary_csv_path,
        }
        for key in ("csv", "parquet"):
            path = output_paths.get(key)
            if path is not None:
                staged_paths[key] = Path(path)
        for key, value in multicollinearity_outputs.items():
            if key.endswith("_path") and value:
                staged_paths[f"multicollinearity_{key}"] = Path(value)

        run_config_path = staging_dir / OUTPUTS["run_config"]
        final_output_names = [
            path.name for path in staged_paths.values()
        ] + [OUTPUTS["run_config"], OUTPUTS["log"]]
        write_run_config(
            output_path=run_config_path,
            config=config,
            reference=reference,
            configured_predictor_specs=configured_predictor_specs,
            retained_predictor_specs=predictor_specs,
            group_specs=group_specs,
            multicollinearity_outputs=multicollinearity_outputs,
            final_output_names=final_output_names,
        )
        staged_paths["run_config"] = run_config_path

        logger.info("Finished Step 2 successfully.")
        logger.info(
            "Output table path: %s",
            output_dir / Path(output_paths["primary"]).name,
        )
        logger.info("Total samples: %d", int(sample_summary["final_sample_count"]))
        logger.info("Positive samples: %d", int(sample_summary["positive_samples_exported"]))
        logger.info("Negative samples: %d", int(sample_summary["negative_samples_exported"]))
        logger.info("Number of predictors: %d", len(predictor_specs))
        logger.info("Summary CSV path: %s", output_dir / summary_csv_path.name)
        logger.info("Alignment report path: %s", output_dir / alignment_report_path.name)
        logger.info(
            "Multicollinearity summary path: %s",
            output_dir / Path(multicollinearity_outputs["summary_path"]).name,
        )
        logger.info("Processing log path: %s", output_dir / OUTPUTS["log"])

        close_file_log_handlers(logger)
        staged_paths["log"] = staged_log_path
        managed_names = set(MANAGED_STATIC_OUTPUT_NAMES)
        table_name = str(config["output_table_name"])
        managed_names |= {
            f"{table_name}.csv",
            f"{table_name}.parquet",
            f"{table_name}_summary.csv",
        }
        publish_output_bundle(staged_paths, output_dir, managed_names)
        shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir = None
    except Exception as exc:
        if logger is not None:
            logger.exception("Step 2 pixel sample extraction failed: %s", exc)
            close_file_log_handlers(logger)
            if staged_log_path is not None and staged_log_path.exists() and output_dir is not None:
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    failed_log_temp = output_dir / ".processing_log.failed.tmp"
                    shutil.copy2(staged_log_path, failed_log_temp)
                    os.replace(
                        failed_log_temp,
                        output_dir / "processing_log.failed.txt",
                    )
                except OSError:
                    pass
        else:
            print(f"Step 2 pixel sample extraction failed: {exc}", file=sys.stderr)
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
