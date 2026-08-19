import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from contextlib import ExitStack
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

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window


class PreprocessedXGBClassifier:
    """Compatibility wrapper needed when loading Step 3 models saved from __main__."""

    def __init__(self, preprocessor: Any, estimator: Any) -> None:
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.transformed_feature_names_: list[str] = []

    def fit(self, x: pd.DataFrame, y: pd.Series, **fit_kwargs: Any) -> "PreprocessedXGBClassifier":
        eval_set = fit_kwargs.pop("eval_set", None)
        x_transformed = self.preprocessor.fit_transform(x)
        self.transformed_feature_names_ = [str(name) for name in self.preprocessor.get_feature_names_out()]
        if eval_set is not None:
            transformed_eval_set = []
            for eval_x, eval_y in eval_set:
                transformed_eval_set.append((self.preprocessor.transform(eval_x), eval_y))
            fit_kwargs["eval_set"] = transformed_eval_set
        self.estimator.fit(x_transformed, y, **fit_kwargs)
        return self

    def set_params(self, **params: Any) -> "PreprocessedXGBClassifier":
        self.estimator.set_params(**params)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self.preprocessor.transform(x))

    def get_booster(self) -> Any:
        return self.estimator.get_booster()


CONFIG = {
    "model_path": r"",
    "metadata_path": r"",
    "predictor_config_json": r"",
    "reference_raster_path": r"",
    "output_dir": r"",
    "output_probability_raster": "ml_dfi_probability_full_valid_predictor_domain.tif",
    "output_summary_json": "ml_dfi_probability_full_valid_predictor_domain_summary.json",
    "output_nodata": -9999.0,
    "window_height": 256,
    "compress": "lzw",
    "bigtiff": "IF_SAFER",
    "write_full_domain_shap": True,
    "full_domain_shap_sample_size": 60000,
    "full_domain_shap_max_display_features": 12,
    "full_domain_shap_random_seed": 42,
}

DFI_PROBABILITY_GROUPS = {
    "low_dfi": "P < 0.30",
    "medium_dfi": "0.30 <= P < 0.70",
    "high_dfi": "P >= 0.70",
}


LOG_NAME = "apply_ml_dfi_model_to_rasters"
STATIC_APPLICATION_OUTPUT_NAMES = {
    "apply_ml_dfi_model_to_rasters_log.txt",
    "apply_ml_dfi_model_to_rasters_log.failed.txt",
    str(CONFIG["output_probability_raster"]),
    str(CONFIG["output_summary_json"]),
    "full_domain_shap_feature_importance.csv",
    "full_domain_shap_sample.csv",
    "full_domain_shap_feature_importance.png",
    "full_domain_shap_summary_beeswarm.png",
    "full_domain_shap_dependence_plots.png",
}


def setup_logging(output_dir: Path) -> tuple[logging.Logger, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "apply_ml_dfi_model_to_rasters_log.txt"
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
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(tempfile.mkdtemp(prefix=".step3-application-backup-", dir=output_dir.parent))
    final_paths = {key: output_dir / path.name for key, path in staged_paths.items()}
    backed_up: dict[str, Path] = {}
    published: set[str] = set()
    try:
        for name in sorted(managed_names):
            final_path = output_dir / name
            if final_path.exists() and final_path.is_file():
                backup_path = backup_dir / name
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
                "Step 3 raster publication failed and rollback was incomplete. "
                f"Previous files remain in {backup_dir}. Problems: {'; '.join(rollback_errors)}"
            ) from publish_error
        shutil.rmtree(backup_dir, ignore_errors=True)
        raise
    else:
        shutil.rmtree(backup_dir, ignore_errors=True)
    return final_paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_output_filename(value: Any, key: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"CONFIG['{key}'] must not be empty.")
    path = Path(text)
    if path.is_absolute() or path.name != text or text in {".", ".."}:
        raise ValueError(f"{key} must be a filename, not a path.")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a trained ML-DFI model directly to aligned predictor rasters and write a probability raster."
    )
    parser.add_argument("--config-json", type=Path, help="Optional raster-application JSON config.")
    parser.add_argument("--model-path")
    parser.add_argument("--metadata-path")
    parser.add_argument("--predictor-config-json")
    parser.add_argument("--reference-raster-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-probability-raster")
    parser.add_argument("--window-height", type=int)
    parser.add_argument("--full-domain-shap-sample-size", type=int)
    parser.add_argument("--disable-full-domain-shap", action="store_true")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(CONFIG)
    if args.config_json is not None:
        config_path = args.config_json.expanduser().resolve()
        loaded = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(loaded, dict):
            raise ValueError("Raster-application config must contain a JSON object.")
        section = loaded.get("apply_ml_dfi_model", loaded)
        if not isinstance(section, dict):
            raise ValueError("apply_ml_dfi_model must contain a JSON object.")
        unknown = sorted(set(section) - set(CONFIG))
        if unknown:
            raise ValueError(
                "Unknown Step 3 raster-application config keys: " + ", ".join(unknown)
            )
        config.update(section)
        for key in (
            "model_path",
            "metadata_path",
            "predictor_config_json",
            "reference_raster_path",
            "output_dir",
        ):
            value = str(config.get(key, "") or "").strip()
            if not value:
                continue
            path = Path(value).expanduser()
            config[key] = str(
                path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()
            )
    overrides = {
        "model_path": args.model_path,
        "metadata_path": args.metadata_path,
        "predictor_config_json": args.predictor_config_json,
        "reference_raster_path": args.reference_raster_path,
        "output_dir": args.output_dir,
        "output_probability_raster": args.output_probability_raster,
        "window_height": args.window_height,
        "full_domain_shap_sample_size": args.full_domain_shap_sample_size,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.disable_full_domain_shap:
        config["write_full_domain_shap"] = False
    for key in (
        "model_path",
        "metadata_path",
        "predictor_config_json",
        "reference_raster_path",
        "output_dir",
    ):
        value = str(config.get(key, "") or "").strip()
        if value:
            config[key] = str(Path(value).expanduser().resolve())
    return config


def validate_config(config: dict[str, Any]) -> None:
    unknown = sorted(set(config) - set(CONFIG))
    if unknown:
        raise ValueError(
            "Unknown Step 3 raster-application config keys: " + ", ".join(unknown)
        )
    required = ["model_path", "metadata_path", "predictor_config_json", "reference_raster_path", "output_dir"]
    for key in required:
        value = str(config.get(key, "")).strip()
        if not value:
            raise ValueError(f"CONFIG['{key}'] is required.")
        if key != "output_dir" and not Path(value).exists():
            raise FileNotFoundError(f"{key} does not exist: {value}")
    if int(config.get("window_height", 256)) < 1:
        raise ValueError("window_height must be >= 1.")
    if bool(config.get("write_full_domain_shap", True)) and int(config.get("full_domain_shap_sample_size", 60000)) < 1:
        raise ValueError("full_domain_shap_sample_size must be >= 1.")
    if int(config.get("full_domain_shap_max_display_features", 12)) < 1:
        raise ValueError("full_domain_shap_max_display_features must be >= 1.")
    nodata_value = float(config.get("output_nodata", -9999.0))
    if not np.isfinite(nodata_value):
        raise ValueError("output_nodata must be finite.")
    validate_output_filename(config.get("output_probability_raster"), "output_probability_raster")
    validate_output_filename(config.get("output_summary_json"), "output_summary_json")


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    predictor_columns = metadata.get("predictor_columns")
    if not isinstance(predictor_columns, list) or not predictor_columns:
        raise ValueError("Model metadata does not contain a non-empty predictor_columns list.")
    categorical_columns = metadata.get("categorical_predictor_columns", [])
    if not isinstance(categorical_columns, list):
        raise ValueError("Model metadata categorical_predictor_columns must be a list.")
    unknown_categorical = sorted(set(map(str, categorical_columns)) - set(map(str, predictor_columns)))
    if unknown_categorical:
        raise ValueError(
            "Model metadata lists categorical predictors that are not model predictors: "
            + ", ".join(unknown_categorical)
        )
    return metadata


def get_model_categorical_levels(
    model: Any,
    metadata: dict[str, Any],
    categorical_columns: list[str],
) -> dict[str, set[str]]:
    configured = metadata.get("categorical_levels", {})
    if isinstance(configured, dict) and all(
        column in configured and isinstance(configured[column], list)
        for column in categorical_columns
    ):
        return {
            column: {str(value) for value in configured[column]}
            for column in categorical_columns
        }
    preprocessor = getattr(model, "preprocessor", None)
    if preprocessor is None and hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("The saved model does not expose its fitted categorical preprocessor.")
    try:
        transformer = preprocessor.named_transformers_["categorical"]
        encoder = transformer.named_steps["onehot"]
        return {
            column: {str(value) for value in levels.tolist()}
            for column, levels in zip(
                categorical_columns,
                encoder.categories_,
                strict=True,
            )
        }
    except Exception as exc:
        raise ValueError(
            "Could not recover fitted categorical levels from the model."
        ) from exc


def prepare_predictor_frame(
    frame: pd.DataFrame,
    categorical_columns: list[str],
    categorical_levels: dict[str, set[str]],
) -> pd.DataFrame:
    if not categorical_columns:
        return frame
    prepared = frame.copy()
    for column in categorical_columns:
        values = prepared[column]
        not_missing = values.notna()
        normalized = values.astype(object)
        normalized.loc[not_missing] = values.loc[not_missing].astype(str)
        normalized.loc[~not_missing] = np.nan
        observed = set(normalized.loc[not_missing].astype(str).unique())
        allowed = categorical_levels.get(column, set())
        unexpected = sorted(observed - allowed)
        if unexpected:
            raise ValueError(
                f"Predictor raster '{column}' contains categorical values not seen during training: "
                + ", ".join(unexpected[:12])
            )
        prepared[column] = normalized
    return prepared


def load_predictor_rasters(config_json_path: Path, predictor_columns: list[str]) -> dict[str, Path]:
    loaded = json.loads(config_json_path.read_text(encoding="utf-8"))
    section = loaded.get("extract_pixel_samples", loaded)
    predictor_rasters = section.get("predictor_rasters", {})
    if not isinstance(predictor_rasters, dict):
        raise ValueError("Predictor config JSON does not contain an extract_pixel_samples.predictor_rasters object.")
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for predictor in predictor_columns:
        path_text = str(predictor_rasters.get(predictor, "")).strip()
        if not path_text:
            missing.append(predictor)
            continue
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            path = (config_json_path.parent / path).resolve()
        else:
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Predictor raster for '{predictor}' does not exist: {path}")
        resolved[predictor] = path
    if missing:
        raise ValueError(f"Predictor config JSON is missing model predictors: {missing}")
    return resolved


def iter_windows(width: int, height: int, window_height: int) -> Iterator[Window]:
    for row_off in range(0, height, window_height):
        current_height = min(window_height, height - row_off)
        yield Window(col_off=0, row_off=row_off, width=width, height=current_height)


def validate_alignment(reference_path: Path, predictor_paths: dict[str, Path]) -> dict[str, Any]:
    with rasterio.open(reference_path) as ref:
        reference = {
            "width": ref.width,
            "height": ref.height,
            "transform": ref.transform,
            "crs": ref.crs,
            "profile": ref.profile.copy(),
        }
        for name, path in predictor_paths.items():
            with rasterio.open(path) as src:
                if src.crs != ref.crs:
                    raise ValueError(f"Predictor raster '{name}' CRS does not match the reference raster.")
                if src.transform != ref.transform:
                    raise ValueError(f"Predictor raster '{name}' transform does not match the reference raster.")
                if src.width != ref.width or src.height != ref.height:
                    raise ValueError(f"Predictor raster '{name}' dimensions do not match the reference raster.")
    return reference


def update_domain_shap_sample(
    current_sample: pd.DataFrame | None,
    batch_df: pd.DataFrame,
    probabilities: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    group_sample_size: int,
) -> pd.DataFrame:
    if group_sample_size < 1 or batch_df.empty:
        return pd.DataFrame() if current_sample is None else current_sample
    rows = np.asarray(row_indices, dtype=np.int64)
    cols = np.asarray(col_indices, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    probability_groups = np.select(
        [probability_array < 0.30, probability_array < 0.70],
        ["low_dfi", "medium_dfi"],
        default="high_dfi",
    )
    row_hash = rows.astype(np.uint64, copy=False) * np.uint64(1_140_071_481_932_319_845)
    col_hash = cols.astype(np.uint64, copy=False) * np.uint64(1_402_945_737_069_566_093)
    sample_keys = np.bitwise_xor(row_hash, col_hash)
    group_samples: list[pd.DataFrame] = []
    for group_name in DFI_PROBABILITY_GROUPS:
        candidate_indices = np.flatnonzero(probability_groups == group_name)
        if len(candidate_indices) > group_sample_size:
            local_keys = sample_keys[candidate_indices]
            keep = np.argpartition(local_keys, group_sample_size - 1)[:group_sample_size]
            candidate_indices = candidate_indices[keep]
        if len(candidate_indices):
            group_df = batch_df.iloc[candidate_indices].copy()
            group_df["domain_row"] = rows[candidate_indices]
            group_df["domain_col"] = cols[candidate_indices]
            group_df["predicted_dfi_probability"] = probability_array[candidate_indices]
            group_df["dfi_probability_group"] = group_name
            group_df["__sample_key"] = sample_keys[candidate_indices]
        else:
            group_df = pd.DataFrame()
        if current_sample is not None and not current_sample.empty:
            previous = current_sample.loc[
                current_sample["dfi_probability_group"] == group_name
            ]
            group_df = pd.concat([previous, group_df], ignore_index=True)
        if len(group_df) > group_sample_size:
            group_df = group_df.nsmallest(group_sample_size, "__sample_key")
        if not group_df.empty:
            group_samples.append(group_df)
    return pd.concat(group_samples, ignore_index=True) if group_samples else pd.DataFrame()


def get_model_feature_names(model: Any, predictor_columns: list[str]) -> list[str]:
    if hasattr(model, "transformed_feature_names_") and getattr(model, "transformed_feature_names_"):
        return [str(name) for name in getattr(model, "transformed_feature_names_")]
    if hasattr(model, "preprocessor"):
        try:
            return [str(name) for name in model.preprocessor.get_feature_names_out()]
        except Exception:
            return predictor_columns
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        try:
            return [str(name) for name in model.named_steps["preprocessor"].get_feature_names_out()]
        except Exception:
            return predictor_columns
    return predictor_columns


def source_predictor_from_model_feature(feature_name: str, predictor_columns: list[str]) -> str:
    base_name = str(feature_name).split("__", 1)[-1]
    for predictor in sorted(map(str, predictor_columns), key=len, reverse=True):
        if base_name == predictor or base_name.startswith(f"{predictor}_"):
            return predictor
    return base_name


def write_full_domain_shap_plots(
    output_dir: Path,
    shap_feature_df: pd.DataFrame,
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    max_display_features: int,
    random_seed: int,
    logger: logging.Logger,
) -> dict[str, str]:
    paths = {
        "feature_importance_plot_path": str(output_dir / "full_domain_shap_feature_importance.png"),
        "summary_plot_path": str(output_dir / "full_domain_shap_summary_beeswarm.png"),
    }
    predictor_rows = shap_feature_df.loc[shap_feature_df["level"] == "predictor"].copy()
    predictor_rows = predictor_rows.sort_values("mean_abs_shap", ascending=False).head(max_display_features)
    if not predictor_rows.empty:
        try:
            fig, ax = plt.subplots(figsize=(9, max(4, len(predictor_rows) * 0.38)))
            ax.barh(predictor_rows["feature"], predictor_rows["mean_abs_shap"], color="#D55E00")
            ax.invert_yaxis()
            ax.set_xlabel("Mean absolute SHAP contribution to XGBoost model margin")
            ax.set_ylabel("Predictor")
            ax.set_title("Full-Domain SHAP Feature Importance for ML-DFI Raster")
            fig.tight_layout()
            fig.savefig(output_dir / "full_domain_shap_feature_importance.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            logger.warning("Could not write full-domain SHAP feature importance plot: %s", exc)
            plt.close()

    feature_rows = shap_feature_df.loc[shap_feature_df["level"] == "model_feature"].copy()
    feature_rows = feature_rows.sort_values("mean_abs_shap", ascending=False).head(max_display_features)
    if feature_rows.empty:
        return paths
    try:
        row_indices = [feature_names.index(str(feature)) for feature in feature_rows["feature"]]
        rng = np.random.default_rng(int(random_seed))
        fig, ax = plt.subplots(figsize=(10, max(5, len(row_indices) * 0.42)))
        color_handle = None
        for display_row, feature_index in enumerate(reversed(row_indices)):
            jitter = rng.normal(0.0, 0.08, size=shap_values.shape[0])
            color_handle = ax.scatter(
                shap_values[:, feature_index],
                np.full(shap_values.shape[0], display_row, dtype=float) + jitter,
                c=feature_values[:, feature_index],
                cmap="coolwarm",
                alpha=0.45,
                s=10,
                linewidths=0,
            )
        ax.axvline(0.0, color="#444444", linewidth=1)
        ax.set_yticks(np.arange(len(row_indices)))
        ax.set_yticklabels(list(reversed(feature_rows["feature"].astype(str).tolist())))
        ax.set_xlabel("SHAP contribution to XGBoost model margin")
        ax.set_ylabel("Transformed model feature")
        ax.set_title("Full-Domain SHAP Contribution Summary for ML-DFI Raster")
        if color_handle is not None:
            colorbar = fig.colorbar(color_handle, ax=ax)
            colorbar.set_label("Transformed feature value")
        fig.tight_layout()
        fig.savefig(output_dir / "full_domain_shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning("Could not write full-domain SHAP beeswarm plot: %s", exc)
        plt.close()
    return paths


def write_full_domain_shap_dependence_plots(
    output_dir: Path,
    shap_feature_df: pd.DataFrame,
    domain_sample: pd.DataFrame,
    predictor_shap_by_name: dict[str, np.ndarray],
    max_display_features: int,
    random_seed: int,
    logger: logging.Logger,
) -> str | None:
    predictor_rows = shap_feature_df.loc[shap_feature_df["level"] == "predictor"].copy()
    predictor_rows = predictor_rows.sort_values("mean_abs_shap", ascending=False).head(max_display_features)
    plotted_features = [
        str(feature)
        for feature in predictor_rows["feature"].tolist()
        if str(feature) in domain_sample.columns and str(feature) in predictor_shap_by_name
    ]
    if not plotted_features:
        return None

    plot_path = output_dir / "full_domain_shap_dependence_plots.png"
    try:
        rng = np.random.default_rng(int(random_seed))
        column_count = min(3, len(plotted_features))
        row_count = int(math.ceil(len(plotted_features) / column_count))
        fig, axes = plt.subplots(row_count, column_count, figsize=(5.4 * column_count, 3.9 * row_count), squeeze=False)
        for axis_index, predictor in enumerate(plotted_features):
            ax = axes[axis_index // column_count][axis_index % column_count]
            raw_values = domain_sample[predictor]
            numeric_values = pd.to_numeric(raw_values, errors="coerce")
            shap_values = np.asarray(predictor_shap_by_name[predictor], dtype=float)
            is_numeric = numeric_values.notna().mean() >= 0.9 and numeric_values.nunique(dropna=True) > 12

            if is_numeric:
                ax.scatter(
                    numeric_values,
                    shap_values,
                    color="#D55E00",
                    alpha=0.45,
                    s=12,
                    linewidths=0,
                )
                ax.set_xlabel(f"{predictor} value")
            else:
                labels = raw_values.where(raw_values.notna(), "missing").astype(str)
                label_order = sorted(labels.unique())
                positions = {label: index for index, label in enumerate(label_order)}
                x_values = labels.map(positions).astype(float).to_numpy()
                x_values = x_values + rng.normal(0.0, 0.06, size=len(x_values))
                ax.scatter(
                    x_values,
                    shap_values,
                    color="#D55E00",
                    alpha=0.45,
                    s=12,
                    linewidths=0,
                )
                ax.set_xticks(np.arange(len(label_order)))
                ax.set_xticklabels(label_order, rotation=35, ha="right")
                ax.set_xlabel(f"{predictor} category")

            ax.axhline(0.0, color="#444444", linewidth=1)
            ax.set_ylabel("SHAP contribution to XGBoost model margin")
            ax.set_title(f"SHAP Dependence: {predictor}")
            ax.grid(alpha=0.2)

        for axis_index in range(len(plotted_features), row_count * column_count):
            axes[axis_index // column_count][axis_index % column_count].axis("off")
        fig.suptitle("Full-Domain SHAP Dependence for ML-DFI Raster", y=1.01)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(plot_path)
    except Exception as exc:
        logger.warning("Could not write full-domain SHAP dependence plots: %s", exc)
        plt.close()
        return None


def export_full_domain_shap_analysis(
    model: Any,
    domain_sample: pd.DataFrame | None,
    predictor_columns: list[str],
    output_dir: Path,
    config: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    if not bool(config.get("write_full_domain_shap", True)):
        return {"enabled": False, "reason": "disabled"}
    if domain_sample is None or domain_sample.empty:
        return {"enabled": False, "reason": "no_valid_domain_sample"}
    if "__sample_key" in domain_sample.columns:
        domain_sample = domain_sample.drop(columns=["__sample_key"]).reset_index(drop=True)
    if not hasattr(model, "preprocessor") or not hasattr(model, "get_booster"):
        return {"enabled": False, "reason": "xgboost_preprocessor_or_booster_unavailable"}

    try:
        from xgboost import DMatrix

        transformed = np.asarray(model.preprocessor.transform(domain_sample[predictor_columns]), dtype=float)
        feature_names = get_model_feature_names(model, predictor_columns)
        contribution_matrix = np.asarray(
            model.get_booster().predict(
                DMatrix(transformed, feature_names=feature_names),
                pred_contribs=True,
            ),
            dtype=float,
        )
        contribution_matrix = np.squeeze(contribution_matrix)
        if contribution_matrix.ndim != 2 or contribution_matrix.shape[1] != transformed.shape[1] + 1:
            raise ValueError(
                "Expected SHAP contribution matrix with one column per transformed feature plus bias; "
                f"received shape {contribution_matrix.shape} for {transformed.shape[1]} transformed features."
            )

        shap_values = contribution_matrix[:, :-1]
        bias_values = contribution_matrix[:, -1]
        model_feature_rows: list[dict[str, Any]] = []
        source_predictors: list[str] = []
        for feature_index, feature_name in enumerate(feature_names):
            feature_shap = shap_values[:, feature_index]
            source_predictor = source_predictor_from_model_feature(str(feature_name), predictor_columns)
            source_predictors.append(source_predictor)
            model_feature_rows.append(
                {
                    "level": "model_feature",
                    "feature": str(feature_name),
                    "source_predictor": source_predictor,
                    "mean_abs_shap": float(np.mean(np.abs(feature_shap))),
                    "mean_shap": float(np.mean(feature_shap)),
                    "std_shap": float(np.std(feature_shap)),
                    "sample_count": int(len(domain_sample)),
                }
            )

        predictor_rows: list[dict[str, Any]] = []
        predictor_shap_by_name: dict[str, np.ndarray] = {}
        for predictor in dict.fromkeys(source_predictors):
            feature_indices = [index for index, source in enumerate(source_predictors) if source == predictor]
            predictor_shap = np.sum(shap_values[:, feature_indices], axis=1)
            predictor_shap_by_name[predictor] = predictor_shap
            predictor_rows.append(
                {
                    "level": "predictor",
                    "feature": predictor,
                    "source_predictor": predictor,
                    "mean_abs_shap": float(np.mean(np.abs(predictor_shap))),
                    "mean_shap": float(np.mean(predictor_shap)),
                    "std_shap": float(np.std(predictor_shap)),
                    "sample_count": int(len(domain_sample)),
                }
            )
        shap_feature_df = pd.concat([pd.DataFrame(predictor_rows), pd.DataFrame(model_feature_rows)], ignore_index=True)
        shap_feature_df = shap_feature_df.sort_values(["level", "mean_abs_shap"], ascending=[True, False]).reset_index(drop=True)
        shap_feature_df["rank"] = (
            shap_feature_df.groupby("level")["mean_abs_shap"].rank(method="first", ascending=False).astype(int)
        )

        output_path = output_dir / "full_domain_shap_feature_importance.csv"
        shap_feature_df[
            ["level", "feature", "source_predictor", "mean_abs_shap", "mean_shap", "std_shap", "sample_count", "rank"]
        ].to_csv(output_path, index=False)
        sample_path = output_dir / "full_domain_shap_sample.csv"
        domain_sample.to_csv(sample_path, index=False)
        plot_paths = write_full_domain_shap_plots(
            output_dir,
            shap_feature_df,
            shap_values,
            transformed,
            feature_names,
            int(config["full_domain_shap_max_display_features"]),
            int(config["full_domain_shap_random_seed"]),
            logger,
        )
        dependence_plot_path = write_full_domain_shap_dependence_plots(
            output_dir,
            shap_feature_df,
            domain_sample,
            predictor_shap_by_name,
            int(config["full_domain_shap_max_display_features"]),
            int(config["full_domain_shap_random_seed"]),
            logger,
        )
        if dependence_plot_path is not None:
            plot_paths["dependence_plot_path"] = dependence_plot_path
        logger.info("Wrote full-domain SHAP analysis from %d sampled predicted pixels.", len(domain_sample))
        group_counts = (
            domain_sample["dfi_probability_group"].value_counts().reindex(DFI_PROBABILITY_GROUPS.keys(), fill_value=0).astype(int).to_dict()
            if "dfi_probability_group" in domain_sample.columns
            else {}
        )
        return {
            "enabled": True,
            "method": "xgboost_pred_contribs",
            "sampling_method": "deterministic_stratified_by_predicted_dfi_probability",
            "probability_groups": DFI_PROBABILITY_GROUPS,
            "sample_group_counts": group_counts,
            "sample_count": int(len(domain_sample)),
            "importance_path": str(output_path),
            "sample_path": str(sample_path),
            "mean_bias_contribution": float(np.mean(bias_values)),
            "contribution_output": "xgboost_raw_margin",
            **plot_paths,
        }
    except Exception as exc:
        logger.warning("Could not export full-domain SHAP analysis: %s", exc)
        return {"enabled": False, "reason": str(exc)}


def predict_raster(
    model: Any,
    predictor_paths: dict[str, Path],
    predictor_columns: list[str],
    reference: dict[str, Any],
    output_raster_path: Path,
    nodata_value: float,
    window_height: int,
    compress: str,
    bigtiff: str,
    full_domain_shap_sample_size: int,
    categorical_columns: list[str],
    categorical_levels: dict[str, set[str]],
    logger: logging.Logger,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    profile = reference["profile"]
    profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "float32",
            "nodata": float(nodata_value),
            "compress": compress,
            "bigtiff": bigtiff,
        }
    )

    window_count = int(math.ceil(int(reference["height"]) / int(window_height)))
    total_predicted = 0
    total_valid_predictor_pixels = 0
    probability_sum = 0.0
    min_probability = np.inf
    max_probability = -np.inf
    full_domain_shap_sample: pd.DataFrame | None = None
    full_domain_shap_group_sample_size = (
        int(math.ceil(int(full_domain_shap_sample_size) / len(DFI_PROBABILITY_GROUPS)))
        if int(full_domain_shap_sample_size) > 0
        else 0
    )

    with ExitStack() as stack:
        datasets = {
            name: stack.enter_context(rasterio.open(path))
            for name, path in predictor_paths.items()
        }
        dst = stack.enter_context(rasterio.open(output_raster_path, "w", **profile))
        for index, window in enumerate(
            iter_windows(
                int(reference["width"]),
                int(reference["height"]),
                int(window_height),
            ),
            start=1,
        ):
            valid_mask: np.ndarray | None = None

            block_values: dict[str, np.ndarray] = {}
            for name in predictor_columns:
                array = datasets[name].read(1, window=window, masked=True)
                array_values = np.asarray(array.data, dtype=np.float32)
                finite_mask = np.isfinite(array_values)
                current_valid = (~np.ma.getmaskarray(array)) & finite_mask
                valid_mask = current_valid if valid_mask is None else (valid_mask & current_valid)
                block_values[name] = array_values

            if valid_mask is None:
                raise RuntimeError("No predictor arrays were loaded for raster prediction.")

            total_valid_predictor_pixels += int(np.count_nonzero(valid_mask))
            output_block = np.full((int(window.height), int(window.width)), float(nodata_value), dtype=np.float32)

            if np.any(valid_mask):
                local_rows, local_cols = np.nonzero(valid_mask)
                data = {
                    name: block_values[name][valid_mask]
                    for name in predictor_columns
                }
                predictors_df = pd.DataFrame(data, columns=predictor_columns)
                predictors_df = prepare_predictor_frame(
                    predictors_df,
                    categorical_columns,
                    categorical_levels,
                )
                probabilities = model.predict_proba(predictors_df)[:, 1].astype(np.float32, copy=False)
                if np.any((probabilities < 0) | (probabilities > 1) | ~np.isfinite(probabilities)):
                    raise ValueError(f"Model returned invalid probabilities in window {index}.")
                output_block[valid_mask] = probabilities
                total_predicted += int(probabilities.size)
                probability_sum += float(np.sum(probabilities, dtype=np.float64))
                min_probability = min(min_probability, float(np.min(probabilities)))
                max_probability = max(max_probability, float(np.max(probabilities)))
                if full_domain_shap_group_sample_size > 0:
                    full_domain_shap_sample = update_domain_shap_sample(
                        full_domain_shap_sample,
                        predictors_df,
                        probabilities,
                        int(window.row_off) + local_rows,
                        int(window.col_off) + local_cols,
                        full_domain_shap_group_sample_size,
                    )

            dst.write(output_block, 1, window=window)
            logger.info(
                "Predicted window %d/%d: row_off=%d height=%d valid_pixels=%d cumulative_predicted=%d",
                index,
                window_count,
                int(window.row_off),
                int(window.height),
                int(np.count_nonzero(valid_mask)),
                total_predicted,
            )

    if total_predicted == 0:
        raise ValueError(
            "No pixels had a complete, finite predictor stack; no DFI probabilities were produced."
        )
    mean_probability = float(probability_sum / total_predicted)
    stats = {
        "predicted_pixels": total_predicted,
        "valid_predictor_pixels": total_valid_predictor_pixels,
        "probability_min": min_probability,
        "probability_max": max_probability,
        "probability_mean": mean_probability,
    }
    return stats, full_domain_shap_sample


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    logger: logging.Logger | None = None
    staging_dir: Path | None = None
    staged_log_path: Path | None = None
    official_output_dir: Path | None = None

    try:
        validate_config(config)
        official_output_dir = Path(str(config["output_dir"])).expanduser().resolve()
        official_output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{official_output_dir.name}.step3-application-stage-",
                dir=official_output_dir.parent,
            )
        )
        output_dir = staging_dir
        logger, staged_log_path = setup_logging(output_dir)

        if "PROJ" in CONFIGURED_GEOSPATIAL_DIRS:
            logger.info("Using PROJ data directory: %s", CONFIGURED_GEOSPATIAL_DIRS["PROJ"])
        if "GDAL" in CONFIGURED_GEOSPATIAL_DIRS:
            logger.info("Using GDAL data directory: %s", CONFIGURED_GEOSPATIAL_DIRS["GDAL"])

        model_path = Path(str(config["model_path"])).resolve()
        metadata_path = Path(str(config["metadata_path"])).resolve()
        reference_raster_path = Path(str(config["reference_raster_path"])).resolve()
        predictor_config_json = Path(str(config["predictor_config_json"])).resolve()
        output_raster_path = output_dir / str(config["output_probability_raster"])
        output_summary_path = output_dir / str(config["output_summary_json"])

        metadata = load_metadata(metadata_path)
        predictor_columns = [str(column) for column in metadata["predictor_columns"]]
        categorical_columns = [
            str(column)
            for column in metadata.get("categorical_predictor_columns", [])
        ]
        model_sha256 = sha256_file(model_path)
        metadata_sha256 = sha256_file(metadata_path)
        expected_model_sha256 = str(metadata.get("model_sha256") or "").strip().lower()
        if expected_model_sha256 and expected_model_sha256 != model_sha256.lower():
            raise ValueError(
                "The model file SHA-256 does not match model_metadata.json. "
                "Use the model and metadata from the same validated training bundle."
            )
        logger.warning(
            "Loading a joblib model executes serialized Python objects. Use only a trusted Step 3 model artifact."
        )
        logger.info("Loading model: %s", model_path)
        model = joblib.load(model_path)
        categorical_levels = get_model_categorical_levels(
            model,
            metadata,
            categorical_columns,
        )
        predictor_paths = load_predictor_rasters(predictor_config_json, predictor_columns)
        reference = validate_alignment(reference_raster_path, predictor_paths)

        logger.info("Reference raster: %s", reference_raster_path)
        logger.info("Predictor columns: %s", ", ".join(predictor_columns))
        logger.info(
            "Categorical predictor contract: %s",
            ", ".join(categorical_columns) or "none",
        )

        stats, full_domain_shap_sample = predict_raster(
            model=model,
            predictor_paths=predictor_paths,
            predictor_columns=predictor_columns,
            reference=reference,
            output_raster_path=output_raster_path,
            nodata_value=float(config["output_nodata"]),
            window_height=int(config["window_height"]),
            compress=str(config["compress"]),
            bigtiff=str(config["bigtiff"]),
            full_domain_shap_sample_size=(
                int(config["full_domain_shap_sample_size"])
                if bool(config.get("write_full_domain_shap", True))
                else 0
            ),
            categorical_columns=categorical_columns,
            categorical_levels=categorical_levels,
            logger=logger,
        )
        full_domain_shap_summary = export_full_domain_shap_analysis(
            model=model,
            domain_sample=full_domain_shap_sample,
            predictor_columns=predictor_columns,
            output_dir=output_dir,
            config=config,
            logger=logger,
        )

        summary = {
            "model_path": str(model_path),
            "model_sha256": model_sha256,
            "metadata_path": str(metadata_path),
            "metadata_sha256": metadata_sha256,
            "predictor_config_json": str(predictor_config_json),
            "reference_raster_path": str(reference_raster_path),
            "output_probability_raster": output_raster_path.name,
            "output_probability_raster_sha256": sha256_file(output_raster_path),
            "predictor_columns": predictor_columns,
            "categorical_predictor_columns": categorical_columns,
            "categorical_levels": {
                column: sorted(levels)
                for column, levels in categorical_levels.items()
            },
            "full_domain_shap": full_domain_shap_summary,
            **stats,
        }
        for key in (
            "importance_path",
            "sample_path",
            "feature_importance_plot_path",
            "summary_plot_path",
            "dependence_plot_path",
        ):
            path_text = full_domain_shap_summary.get(key)
            if path_text:
                full_domain_shap_summary[key] = Path(str(path_text)).name
        output_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        logger.info("Finished ML-DFI raster prediction successfully.")
        logger.info("Official output probability raster: %s", official_output_dir / output_raster_path.name)
        logger.info("Predicted pixels: %d", stats["predicted_pixels"])
        logger.info("Probability range: %.6f to %.6f", stats["probability_min"], stats["probability_max"])
        logger.info("Mean probability: %.6f", stats["probability_mean"])
        logger.info("Official summary JSON: %s", official_output_dir / output_summary_path.name)
        close_file_log_handlers(logger)
        staged_paths = {
            path.name: path
            for path in output_dir.iterdir()
            if path.is_file()
        }
        managed_names = STATIC_APPLICATION_OUTPUT_NAMES | {
            output_raster_path.name,
            output_summary_path.name,
        }
        published_paths = publish_output_bundle(
            staged_paths,
            official_output_dir,
            managed_names,
        )
        shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir = None
        final_raster_path = published_paths[output_raster_path.name]
        final_summary_path = published_paths[output_summary_path.name]
        final_log_path = published_paths["apply_ml_dfi_model_to_rasters_log.txt"]

        print("ML-DFI raster prediction complete")
        print(f"Output probability raster: {final_raster_path}")
        print(f"Predicted pixels: {stats['predicted_pixels']}")
        print(f"Probability range: {stats['probability_min']:.6f} to {stats['probability_max']:.6f}")
        print(f"Mean probability: {stats['probability_mean']:.6f}")
        print(f"Summary JSON: {final_summary_path}")
        print(f"Processing log: {final_log_path}")
    except Exception as exc:
        if logger is not None:
            logger.exception("ML-DFI raster prediction from rasters failed: %s", exc)
            close_file_log_handlers(logger)
            if (
                staged_log_path is not None
                and staged_log_path.exists()
                and official_output_dir is not None
            ):
                official_output_dir.mkdir(parents=True, exist_ok=True)
                failed_log = official_output_dir / "apply_ml_dfi_model_to_rasters_log.failed.txt"
                failed_log_tmp = official_output_dir / ".apply_ml_dfi_model_to_rasters_log.failed.tmp"
                shutil.copy2(staged_log_path, failed_log_tmp)
                os.replace(failed_log_tmp, failed_log)
        else:
            print(f"ML-DFI raster prediction from rasters failed: {exc}", file=sys.stderr)
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
