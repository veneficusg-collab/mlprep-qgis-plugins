import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


CONFIG = {
    "input_sample_table": r"",
    "output_dir": r"",
    "input_format": "auto",
    "target_column": "target",
    "id_columns": ["sample_id", "row", "col", "x", "y"],
    "group_columns": ["landslide_id"],
    "drop_columns": ["sample_role", "split"],
    "categorical_columns": ["geomorphology"],
    "auto_detect_categorical": False,
    "model_type": "xgboost",
    "random_seed": 42,
    "missing_predictor_strategy": "xgboost_native_else_median_impute",
    "class_imbalance_strategy": "auto_scale_pos_weight",
    "enable_hyperparameter_tuning": True,
    "tuning_trials": 100,
    "tuning_metric": "PR_AUC",
    "xgboost_params": {
        "n_estimators": 800,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.0,
        "reg_lambda": 3.0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "early_stopping_rounds": 100,
    },
    "random_forest_params": {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
    },
    "probability_threshold": "optimize_f1",
    "fixed_threshold": 0.5,
    "write_feature_importance": True,
    "write_predictions": True,
    "write_model": True,
    "write_plots": True,
    "write_explainability": True,
    "shap_sample_size": 5000,
    "shap_max_display_features": 12,
    "partial_dependence_sample_size": 2500,
    "partial_dependence_grid_resolution": 25,
    "partial_dependence_max_categories": 12,
}


LOG_NAME = "step3_train_ml_dfi_model"
OUTPUTS = {
    "model": "trained_ml_dfi_model.pkl",
    "metadata": "model_metadata.json",
    "validation_predictions": "validation_predictions.csv",
    "training_predictions": "training_predictions.csv",
    "metrics": "model_metrics.csv",
    "threshold_analysis": "threshold_analysis.csv",
    "partial_dependence": "partial_dependence.csv",
    "tuning_trials": "hyperparameter_tuning_trials.csv",
    "log": "processing_log.txt",
}

NEGATIVE_TARGET_LABEL = "Source/transport non-depositional (target=0)"
POSITIVE_TARGET_LABEL = "Depositional terrain (target=1)"
VALIDATION_MODE = "stratified_group_kfold"
VALIDATION_WORKFLOW = "holdout_then_stratified_group_kfold"
VALIDATION_GROUP_COLUMN = "landslide_id"
VALIDATION_FOLD_COUNT = 5
HOLDOUT_FRACTION = 0.30
PERMUTATION_IMPORTANCE_REPEATS = 5
PLOT_COLORS = {
    "final_training": "#0072B2",
    "holdout_validation": "#D55E00",
    "cv_mean": "#009E73",
}
SPLIT_COLUMN = "split"
NEVER_TRAIN_SPLIT_VALUES = {"test", "holdout", "external"}
OBSOLETE_OUTPUT_NAMES = {
    "final_training_predictions.csv",
    "feature_importance.csv",
    "final_training_confusion_matrix.png",
    "final_training_feature_importance.png",
    "final_training_permutation_feature_importance.csv",
    "final_training_shap_feature_importance.csv",
    "final_training_shap_feature_importance.png",
    "final_training_shap_summary_beeswarm.png",
    "final_training_shap_dependence_plots.png",
    "shap_feature_importance.csv",
    "shap_dependence_plots.png",
    "final_training_partial_dependence_plots.png",
    "holdout_validation_partial_dependence_plots.png",
    "shap_feature_importance.png",
    "shap_summary_beeswarm.png",
    "feature_importance.png",
    "confusion_matrix.png",
    "probability_histogram.png",
    "auc_comparison_folds_training_holdout_line.png",
}
MANAGED_OUTPUT_NAMES = set(OUTPUTS.values()) | OBSOLETE_OUTPUT_NAMES | {
    "processing_log.failed.txt",
    "pr_curve.png",
    "roc_curve.png",
    "calibration_curve.png",
    "partial_dependence_plots.png",
    "pr_auc_comparison_folds_training_holdout_line.png",
    "final_training_partial_dependence.csv",
    "holdout_validation_partial_dependence.csv",
    "final_training_probability_histogram.png",
    "holdout_validation_probability_histogram.png",
    "holdout_validation_confusion_matrix.png",
    "holdout_validation_feature_importance.png",
    "holdout_validation_permutation_feature_importance.csv",
    "holdout_validation_shap_feature_importance.csv",
    "holdout_validation_shap_feature_importance.png",
    "holdout_validation_shap_summary_beeswarm.png",
    "holdout_validation_shap_dependence_plots.png",
}


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
    """Close file handlers before moving a staged log on Windows."""
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
    """Publish a validated bundle and restore the previous files on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(tempfile.mkdtemp(prefix=".step3-backup-", dir=output_dir.parent))
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
                "Step 3 publication failed and rollback was incomplete. Previous files "
                f"remain in {backup_dir}. Problems: {'; '.join(rollback_errors)}"
            ) from publish_error
        shutil.rmtree(backup_dir, ignore_errors=True)
        raise
    else:
        shutil.rmtree(backup_dir, ignore_errors=True)
    return final_paths


def make_artifact_paths_relative(value: Any, output_dir: Path) -> Any:
    """Represent generated artifacts relative to their output bundle."""
    if isinstance(value, dict):
        return {key: make_artifact_paths_relative(item, output_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [make_artifact_paths_relative(item, output_dir) for item in value]
    if not isinstance(value, str) or not value:
        return value
    candidate = Path(value)
    if not candidate.is_absolute():
        return value
    try:
        return candidate.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    input_sample_table = str(config.get("input_sample_table", "")).strip()
    output_dir = str(config.get("output_dir", "")).strip()
    input_format = str(config.get("input_format", "auto")).strip().lower()
    model_type = str(config.get("model_type", "xgboost")).strip().lower()
    probability_threshold = str(config.get("probability_threshold", "optimize_f1")).strip().lower()
    tuning_metric = str(config.get("tuning_metric", "PR_AUC")).strip()

    if not input_sample_table:
        raise ValueError("CONFIG['input_sample_table'] is required.")
    if not Path(input_sample_table).exists():
        raise FileNotFoundError(f"Input sample table does not exist: {input_sample_table}")
    if not output_dir:
        raise ValueError("CONFIG['output_dir'] is required.")
    if input_format not in {"auto", "csv", "parquet"}:
        raise ValueError("input_format must be one of: auto, csv, parquet.")
    if model_type not in {"xgboost", "random_forest"}:
        raise ValueError("model_type must be one of: xgboost, random_forest.")
    removed_validation_keys = {
        "validation_mode",
        "fallback_validation_mode",
        "split_column",
        "train_value",
        "validation_value",
        "final_training_split_values",
        "never_train_split_values",
        "group_column",
        "tuning_validation_mode",
        "final_model_scope",
        "validation_workflow",
        "holdout_fraction",
        "n_splits",
    }
    stale_validation_keys = sorted(key for key in removed_validation_keys if key in config)
    if stale_validation_keys:
        raise ValueError(
            "Step 3 validation is fixed to a 70% grouped training split, a 30% grouped "
            "holdout validation split, and 5-fold stratified group CV inside training. "
            "Remove these config keys: "
            + ", ".join(stale_validation_keys)
            + "."
        )
    removed_sample_weight_keys = {
        "use_sample_weight",
        "sample_weight_column",
        "sample_weight_role",
        "sample_weight_includes_class_balance",
    }
    stale_sample_weight_keys = sorted(key for key in removed_sample_weight_keys if key in config)
    if stale_sample_weight_keys:
        raise ValueError(
            "Step 3 no longer supports sample-weight configuration. Class imbalance is handled with "
            "class_imbalance_strategy='auto_scale_pos_weight'. Remove these config keys: "
            + ", ".join(stale_sample_weight_keys)
            + "."
        )
    known_keys = set(CONFIG) | removed_validation_keys | removed_sample_weight_keys
    unknown_keys = sorted(set(config) - known_keys)
    if unknown_keys:
        raise ValueError(f"Unknown Step 3 config keys: {', '.join(unknown_keys)}")
    if probability_threshold not in {"fixed", "optimize_f1", "optimize_recall_precision_balance", "default_0.5"}:
        raise ValueError(
            "probability_threshold must be one of: fixed, optimize_f1, "
            "optimize_recall_precision_balance, default_0.5."
        )
    if tuning_metric not in {"PR_AUC", "ROC_AUC", "F1"}:
        raise ValueError("tuning_metric must be one of: PR_AUC, ROC_AUC, F1.")
    if bool(config.get("enable_hyperparameter_tuning", False)) and model_type != "xgboost":
        raise ValueError("Hyperparameter tuning currently supports model_type='xgboost' only.")
    if str(config.get("class_imbalance_strategy", "")).strip() != "auto_scale_pos_weight":
        raise ValueError("class_imbalance_strategy must be 'auto_scale_pos_weight'.")
    if not isinstance(config.get("xgboost_params"), dict):
        raise ValueError("xgboost_params must be a JSON object.")
    if not isinstance(config.get("random_forest_params"), dict):
        raise ValueError("random_forest_params must be a JSON object.")

    if int(config.get("tuning_trials", 50)) < 1:
        raise ValueError("tuning_trials must be at least 1.")
    fixed_threshold = float(config.get("fixed_threshold", 0.5))
    if not 0 <= fixed_threshold <= 1:
        raise ValueError("fixed_threshold must be between 0 and 1.")
    for key in (
        "shap_sample_size",
        "shap_max_display_features",
        "partial_dependence_sample_size",
        "partial_dependence_grid_resolution",
        "partial_dependence_max_categories",
    ):
        if int(config.get(key, 1)) < 1:
            raise ValueError(f"{key} must be at least 1.")
    int(config.get("random_seed", 42))

    for key in (
        "id_columns",
        "group_columns",
        "drop_columns",
        "categorical_columns",
    ):
        if not isinstance(config.get(key, []), list):
            raise ValueError(f"{key} must be a list.")
    if VALIDATION_GROUP_COLUMN not in set(map(str, config.get("group_columns", []))):
        raise ValueError(
            f"group_columns must contain the required validation group '{VALIDATION_GROUP_COLUMN}'."
        )

    xgb = dict(config["xgboost_params"])
    positive_xgb_values = (
        "n_estimators",
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "reg_lambda",
        "early_stopping_rounds",
    )
    for key in positive_xgb_values:
        if key in xgb and float(xgb[key]) <= 0:
            raise ValueError(f"xgboost_params.{key} must be greater than zero.")
    for key in ("subsample", "colsample_bytree"):
        if key in xgb and not 0 < float(xgb[key]) <= 1:
            raise ValueError(f"xgboost_params.{key} must be in (0, 1].")
    if "reg_alpha" in xgb and float(xgb["reg_alpha"]) < 0:
        raise ValueError("xgboost_params.reg_alpha must be non-negative.")
    if xgb.get("objective", "binary:logistic") != "binary:logistic":
        raise ValueError("xgboost_params.objective must be 'binary:logistic'.")


def infer_input_format(input_path: Path, input_format: str) -> str:
    fmt = input_format.lower()
    if fmt != "auto":
        return fmt
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    raise ValueError(f"Could not infer input format from file extension: {input_path}")


def load_sample_table(input_sample_table: str, input_format: str) -> pd.DataFrame:
    input_path = Path(input_sample_table).resolve()
    fmt = infer_input_format(input_path, input_format)
    if fmt == "csv":
        return pd.read_csv(input_path)
    if fmt == "parquet":
        return pd.read_parquet(input_path)
    raise ValueError(f"Unsupported input format: {fmt}")


def identify_predictor_columns(
    df: pd.DataFrame,
    config: dict[str, Any],
    logger: logging.Logger,
) -> tuple[list[str], dict[str, list[str]]]:
    target_column = str(config["target_column"])
    excluded = {
        target_column,
        "sample_weight",
        SPLIT_COLUMN,
        *map(str, config.get("id_columns", [])),
        *map(str, config.get("group_columns", [])),
        *map(str, config.get("drop_columns", [])),
    }
    predictor_columns: list[str] = []
    removed = {
        "excluded_by_role": sorted(column for column in df.columns if column in excluded),
        "non_numeric_columns": [],
        "categorical_columns": [],
    }
    configured_categorical = set(map(str, config.get("categorical_columns", [])))
    auto_detect_categorical = bool(config.get("auto_detect_categorical", False))
    missing_configured_categorical = sorted(configured_categorical - set(map(str, df.columns)))
    if missing_configured_categorical:
        raise ValueError(
            "Configured categorical predictors are missing from the input table: "
            + ", ".join(missing_configured_categorical)
        )

    for column in df.columns:
        if column in excluded:
            continue
        if column in configured_categorical:
            predictor_columns.append(column)
            removed["categorical_columns"].append(column)
            logger.info("Predictor column '%s' will be treated as categorical from CONFIG.", column)
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            predictor_columns.append(column)
            continue

        original_non_missing = df[column].notna()
        converted = pd.to_numeric(df[column], errors="coerce")
        introduced_missing = converted.isna() & original_non_missing
        if not introduced_missing.any():
            predictor_columns.append(column)
            logger.info("Predictor column '%s' will be converted to numeric.", column)
        elif auto_detect_categorical:
            predictor_columns.append(column)
            removed["categorical_columns"].append(column)
            logger.info("Predictor column '%s' will be treated as categorical.", column)
        else:
            removed["non_numeric_columns"].append(column)

    if not predictor_columns:
        raise ValueError("No valid predictor columns were found after excluding target, ID, group, split, and drop columns.")
    return predictor_columns, removed


def get_predictor_type_columns(
    df: pd.DataFrame,
    predictor_columns: list[str],
    config: dict[str, Any],
) -> tuple[list[str], list[str]]:
    configured_categorical = set(map(str, config.get("categorical_columns", [])))
    auto_detect_categorical = bool(config.get("auto_detect_categorical", False))
    categorical_columns: list[str] = []
    numeric_columns: list[str] = []

    for column in predictor_columns:
        if column in configured_categorical:
            categorical_columns.append(column)
            continue
        if auto_detect_categorical and (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
            or isinstance(df[column].dtype, pd.CategoricalDtype)
        ):
            converted = pd.to_numeric(df[column], errors="coerce")
            introduced_missing = converted.isna() & df[column].notna()
            if introduced_missing.any():
                categorical_columns.append(column)
                continue
        numeric_columns.append(column)

    return numeric_columns, categorical_columns


def clean_training_table(
    df: pd.DataFrame,
    predictor_columns: list[str],
    config: dict[str, Any],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, int]]:
    target_column = str(config["target_column"])
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the input table.")
    missing_predictors = [column for column in predictor_columns if column not in df.columns]
    if missing_predictors:
        raise ValueError(f"Predictor columns are missing from the input table: {missing_predictors}")

    cleaned = df.copy()
    stats = {
        "rows_before_cleaning": int(len(cleaned)),
        "rows_removed_missing_target": 0,
        "rows_removed_invalid_target": 0,
        "rows_removed_invalid_predictors": 0,
        "missing_predictor_values_retained": 0,
        "missing_categorical_values_retained": 0,
        "non_finite_predictor_values_removed": 0,
        "duplicate_rows_removed": 0,
        "rows_after_cleaning": 0,
    }

    target_numeric = pd.to_numeric(cleaned[target_column], errors="coerce")
    missing_target = target_numeric.isna()
    stats["rows_removed_missing_target"] = int(missing_target.sum())
    cleaned = cleaned.loc[~missing_target].copy()
    target_numeric = target_numeric.loc[~missing_target]

    invalid_target = ~target_numeric.isin([0, 1])
    stats["rows_removed_invalid_target"] = int(invalid_target.sum())
    cleaned = cleaned.loc[~invalid_target].copy()
    target_numeric = target_numeric.loc[~invalid_target]
    cleaned[target_column] = target_numeric.astype(np.int8)

    invalid_predictor_mask = np.zeros(len(cleaned), dtype=bool)
    missing_predictor_value_count = 0
    missing_categorical_value_count = 0
    non_finite_predictor_value_count = 0
    numeric_predictor_columns, categorical_predictor_columns = get_predictor_type_columns(cleaned, predictor_columns, config)
    for predictor_column in predictor_columns:
        if predictor_column in categorical_predictor_columns:
            values = cleaned[predictor_column]
            missing_count = int(values.isna().sum())
            missing_categorical_value_count += missing_count
            if pd.api.types.is_numeric_dtype(values):
                numeric_array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
                non_finite_column_mask = ~np.isfinite(numeric_array) & ~pd.isna(numeric_array)
            else:
                non_finite_column_mask = np.zeros(len(cleaned), dtype=bool)
            invalid_count = int(np.count_nonzero(non_finite_column_mask))
            non_finite_predictor_value_count += invalid_count
            if missing_count:
                logger.warning(
                    "Categorical predictor '%s' has %d missing values. They will be retained for one-hot encoding.",
                    predictor_column,
                    missing_count,
                )
            if invalid_count:
                logger.warning("Categorical predictor '%s' has %d infinite values that will be removed.", predictor_column, invalid_count)
            invalid_predictor_mask |= non_finite_column_mask
            string_values = values.astype(object)
            not_missing = ~values.isna()
            string_values.loc[not_missing] = values.loc[not_missing].astype(str)
            string_values.loc[~not_missing] = np.nan
            cleaned[predictor_column] = string_values
            continue

        numeric_values = pd.to_numeric(cleaned[predictor_column], errors="coerce")
        numeric_array = numeric_values.to_numpy(dtype=np.float64)
        missing_column_mask = numeric_values.isna().to_numpy()
        non_finite_column_mask = ~np.isfinite(numeric_array) & ~missing_column_mask
        invalid_column_mask = non_finite_column_mask
        missing_count = int(np.count_nonzero(missing_column_mask))
        invalid_count = int(np.count_nonzero(invalid_column_mask))
        missing_predictor_value_count += missing_count
        non_finite_predictor_value_count += invalid_count
        if missing_count:
            logger.warning(
                "Predictor '%s' has %d missing values. They will be retained for XGBoost native missing handling "
                "or median imputation in scikit-learn models.",
                predictor_column,
                missing_count,
            )
        if invalid_count:
            logger.warning("Predictor '%s' has %d infinite or non-finite rows that will be removed.", predictor_column, invalid_count)
        invalid_predictor_mask |= invalid_column_mask
        cleaned[predictor_column] = numeric_values

    stats["rows_removed_invalid_predictors"] = int(np.count_nonzero(invalid_predictor_mask))
    stats["missing_predictor_values_retained"] = int(missing_predictor_value_count)
    stats["missing_categorical_values_retained"] = int(missing_categorical_value_count)
    stats["non_finite_predictor_values_removed"] = int(non_finite_predictor_value_count)
    cleaned = cleaned.loc[~invalid_predictor_mask].copy()

    before_duplicates = len(cleaned)
    cleaned = cleaned.drop_duplicates().copy()
    stats["duplicate_rows_removed"] = int(before_duplicates - len(cleaned))
    stats["rows_after_cleaning"] = int(len(cleaned))

    if cleaned.empty:
        raise ValueError("No rows remain after cleaning.")
    if cleaned[target_column].nunique(dropna=True) < 2:
        raise ValueError("Both target classes 0 and 1 must exist after cleaning.")
    return cleaned.reset_index(drop=True), stats


def validate_group_column(df: pd.DataFrame) -> None:
    """Require complete positive integer landslide IDs before grouped splitting."""
    if VALIDATION_GROUP_COLUMN not in df.columns:
        raise ValueError(
            f"Required validation group column '{VALIDATION_GROUP_COLUMN}' is missing."
        )
    raw = df[VALIDATION_GROUP_COLUMN]
    numeric = pd.to_numeric(raw, errors="coerce")
    missing_or_invalid = numeric.isna()
    numeric_array = numeric.to_numpy(dtype=np.float64)
    non_finite = ~np.isfinite(numeric_array)
    non_integer = np.isfinite(numeric_array) & (numeric_array != np.floor(numeric_array))
    non_positive = np.isfinite(numeric_array) & (numeric_array <= 0)
    invalid = missing_or_invalid.to_numpy() | non_finite | non_integer | non_positive
    if np.any(invalid):
        raise ValueError(
            f"'{VALIDATION_GROUP_COLUMN}' must contain complete positive integer IDs; "
            f"found {int(np.count_nonzero(invalid))} invalid rows."
        )
    df[VALIDATION_GROUP_COLUMN] = numeric.astype(np.int64)


def create_group_kfold_splits(
    df: pd.DataFrame,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    group_column = VALIDATION_GROUP_COLUMN
    target_column = str(config["target_column"])
    if group_column not in df.columns:
        raise ValueError(f"stratified group validation requires group_column '{group_column}' to exist.")
    groups = df[group_column].astype(str)
    unique_group_count = int(groups.nunique(dropna=False))
    n_splits = min(VALIDATION_FOLD_COUNT, unique_group_count)
    if n_splits < 2:
        raise ValueError("stratified group validation requires at least two unique groups.")

    splits: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    # StratifiedGroupKFold keeps whole landslides/groups out of both train and
    # validation while also trying to preserve depositional/non-depositional
    # class balance in each fold.
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(config["random_seed"]))
    for fold_index, (train_idx, validation_idx) in enumerate(
        splitter.split(df, df[target_column], groups), start=1
    ):
        train_df = df.iloc[train_idx].copy()
        validation_df = df.iloc[validation_idx].copy()
        train_groups = set(groups.iloc[train_idx])
        validation_groups = set(groups.iloc[validation_idx])
        overlap = train_groups & validation_groups
        if overlap:
            raise ValueError(f"Group leakage detected in fold {fold_index}: {sorted(overlap)[:5]}")
        if train_df[target_column].nunique(dropna=True) < 2:
            raise ValueError(
                f"Stratified group fold {fold_index} has only one class in training. "
                "Use fewer folds or a group column with better class coverage."
            )
        if validation_df[target_column].nunique(dropna=True) < 2:
            raise ValueError(
                f"Stratified group fold {fold_index} has only one class in validation. "
                "Use fewer folds or inspect landslide_id class coverage."
            )
        if logger is not None:
            train_counts = train_df[target_column].value_counts().sort_index().to_dict()
            validation_counts = validation_df[target_column].value_counts().sort_index().to_dict()
            logger.info(
                "Stratified group fold %d class counts: train=%s, validation=%s.",
                fold_index,
                train_counts,
                validation_counts,
            )
        splits.append((f"fold_{fold_index}", train_df.reset_index(drop=True), validation_df.reset_index(drop=True)))
    return splits


def create_stratified_group_holdout_split(
    df: pd.DataFrame,
    config: dict[str, Any],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    group_column = VALIDATION_GROUP_COLUMN
    target_column = str(config["target_column"])
    if group_column not in df.columns:
        raise ValueError(f"Holdout validation requires group_column '{group_column}' to exist.")

    groups = df[group_column].astype(str)
    unique_group_count = int(groups.nunique(dropna=False))
    n_bins = min(10, unique_group_count)
    if n_bins < 2:
        raise ValueError("Holdout validation requires at least two unique groups.")
    holdout_bin_count = max(1, min(n_bins - 1, int(round(HOLDOUT_FRACTION * n_bins))))

    splitter = StratifiedGroupKFold(
        n_splits=n_bins,
        shuffle=True,
        random_state=int(config["random_seed"]) + 1009,
    )
    holdout_indices: list[np.ndarray] = []
    for fold_index, (_, validation_idx) in enumerate(
        splitter.split(df, df[target_column], groups),
        start=1,
    ):
        if fold_index <= holdout_bin_count:
            holdout_indices.append(validation_idx)
    if not holdout_indices:
        raise ValueError("Could not create holdout validation groups.")

    holdout_idx = np.sort(np.concatenate(holdout_indices))
    holdout_mask = np.zeros(len(df), dtype=bool)
    holdout_mask[holdout_idx] = True
    training_df = df.loc[~holdout_mask].copy().reset_index(drop=True)
    holdout_df = df.loc[holdout_mask].copy().reset_index(drop=True)

    training_groups = set(training_df[group_column].astype(str))
    holdout_groups = set(holdout_df[group_column].astype(str))
    overlap = training_groups & holdout_groups
    if overlap:
        raise ValueError(f"Group leakage detected in holdout split: {sorted(overlap)[:5]}")
    if training_df[target_column].nunique(dropna=True) < 2:
        raise ValueError("Holdout workflow training split has only one target class.")
    if holdout_df[target_column].nunique(dropna=True) < 2:
        raise ValueError("Holdout workflow validation split has only one target class.")

    training_counts = training_df[target_column].value_counts().sort_index().to_dict()
    holdout_counts = holdout_df[target_column].value_counts().sort_index().to_dict()
    summary = {
        "requested_holdout_fraction": HOLDOUT_FRACTION,
        "actual_holdout_fraction": float(len(holdout_df) / len(df)),
        "training_sample_count": int(len(training_df)),
        "holdout_sample_count": int(len(holdout_df)),
        "training_group_count": int(len(training_groups)),
        "holdout_group_count": int(len(holdout_groups)),
        "training_class_counts": {str(key): int(value) for key, value in training_counts.items()},
        "holdout_class_counts": {str(key): int(value) for key, value in holdout_counts.items()},
        "split_method": f"{holdout_bin_count}_of_{n_bins}_stratified_group_bins",
    }
    logger.info(
        "Created grouped holdout split: training=%d rows, holdout=%d rows, actual_holdout_fraction=%.3f.",
        len(training_df),
        len(holdout_df),
        summary["actual_holdout_fraction"],
    )
    logger.info("Holdout workflow training class counts: %s.", training_counts)
    logger.info("Holdout workflow validation class counts: %s.", holdout_counts)
    return training_df, holdout_df, summary


def create_stratified_group_kfold_splits(
    df: pd.DataFrame,
    config: dict[str, Any],
    logger: logging.Logger,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    group_column = VALIDATION_GROUP_COLUMN
    target_column = str(config["target_column"])
    if group_column not in df.columns:
        raise ValueError(
            "Tuning validation requires stratified group k-fold, but "
            f"'{group_column}' is missing from the sample table."
        )

    groups = df[group_column].astype(str)
    unique_group_count = int(groups.nunique(dropna=False))
    n_splits = min(VALIDATION_FOLD_COUNT, unique_group_count)
    if n_splits < 2:
        raise ValueError(
            "Tuning validation requires stratified group k-fold, but fewer than two unique "
            f"'{group_column}' groups are available."
        )

    splits: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(config["random_seed"]))
    for fold_index, (train_idx, validation_idx) in enumerate(
        splitter.split(df, df[target_column], groups), start=1
    ):
        train_df = df.iloc[train_idx].copy()
        validation_df = df.iloc[validation_idx].copy()
        train_groups = set(groups.iloc[train_idx])
        validation_groups = set(groups.iloc[validation_idx])
        overlap = train_groups & validation_groups
        if overlap:
            raise ValueError(f"Group leakage detected in tuning fold {fold_index}: {sorted(overlap)[:5]}")
        if train_df[target_column].nunique(dropna=True) < 2 or validation_df[target_column].nunique(dropna=True) < 2:
            logger.warning("Skipping tuning fold %d because train or validation side has only one class.", fold_index)
            continue
        splits.append((f"tuning_fold_{fold_index}", train_df.reset_index(drop=True), validation_df.reset_index(drop=True)))

    if not splits:
        raise ValueError("No usable stratified group tuning folds were created.")
    return splits


def resolve_tuning_splits(
    tuning_df: pd.DataFrame,
    config: dict[str, Any],
    logger: logging.Logger,
) -> tuple[str, list[tuple[str, pd.DataFrame, pd.DataFrame]]]:
    return VALIDATION_MODE, create_stratified_group_kfold_splits(tuning_df, config, logger)


def build_preprocessor(
    predictor_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    passthrough_numeric_missing: bool,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = []
    if not passthrough_numeric_missing:
        numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
    numeric_transformer: Any = Pipeline(numeric_steps) if numeric_steps else "passthrough"
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", numeric_transformer, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_transformer, categorical_columns))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


class PreprocessedXGBClassifier:
    def __init__(self, preprocessor: ColumnTransformer, estimator: Any) -> None:
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

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.estimator.feature_importances_

    @property
    def best_iteration(self) -> Any:
        return getattr(self.estimator, "best_iteration", None)

    @property
    def best_ntree_limit(self) -> Any:
        return getattr(self.estimator, "best_ntree_limit", None)

    def get_booster(self) -> Any:
        return self.estimator.get_booster()


def build_model(
    config: dict[str, Any],
    y_train: pd.Series,
    predictor_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    n_estimators_override: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    model_type = str(config["model_type"]).lower()
    random_seed = int(config["random_seed"])

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for model_type='xgboost'. Install xgboost or use model_type='random_forest'."
            ) from exc

        params = deepcopy(config["xgboost_params"])
        params.pop("early_stopping_rounds", None)
        if n_estimators_override is not None:
            params["n_estimators"] = int(n_estimators_override)
        positive_count = int(np.count_nonzero(y_train == 1))
        negative_count = int(np.count_nonzero(y_train == 0))
        if (
            str(config["class_imbalance_strategy"]) == "auto_scale_pos_weight"
            and positive_count > 0
        ):
            params.setdefault("scale_pos_weight", negative_count / positive_count)
        params.setdefault("random_state", random_seed)
        params.setdefault("n_jobs", -1)
        preprocessor = build_preprocessor(predictor_columns, numeric_columns, categorical_columns, passthrough_numeric_missing=True)
        return PreprocessedXGBClassifier(preprocessor, XGBClassifier(**params)), params

    if model_type == "random_forest":
        params = deepcopy(config["random_forest_params"])
        params.setdefault("random_state", random_seed)
        preprocessor = build_preprocessor(predictor_columns, numeric_columns, categorical_columns, passthrough_numeric_missing=False)
        model = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("random_forest", RandomForestClassifier(**params)),
            ]
        )
        return model, params

    raise ValueError(f"Unsupported model_type: {model_type}")


def train_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame | None,
    predictor_columns: list[str],
    config: dict[str, Any],
    logger: logging.Logger,
    n_estimators_override: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    target_column = str(config["target_column"])
    x_train = train_df[predictor_columns]
    y_train = train_df[target_column]
    numeric_columns, categorical_columns = get_predictor_type_columns(train_df, predictor_columns, config)
    logger.info(
        "Predictor preprocessing columns: numeric=%d, categorical=%d.",
        len(numeric_columns),
        len(categorical_columns),
    )
    model, model_params = build_model(
        config,
        y_train,
        predictor_columns,
        numeric_columns,
        categorical_columns,
        n_estimators_override,
    )
    model_type = str(config["model_type"]).lower()

    fit_kwargs: dict[str, Any] = {}
    if model_type == "xgboost" and validation_df is not None and not validation_df.empty:
        x_validation = validation_df[predictor_columns]
        y_validation = validation_df[target_column]
        fit_kwargs["eval_set"] = [(x_validation, y_validation)]
        fit_kwargs["verbose"] = False
        early_stopping_rounds = deepcopy(config["xgboost_params"]).get("early_stopping_rounds")
        if early_stopping_rounds is not None:
            try:
                model.set_params(early_stopping_rounds=early_stopping_rounds)
                model_params["early_stopping_rounds"] = early_stopping_rounds
            except Exception:
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        try:
            model.fit(x_train, y_train, **fit_kwargs)
            return model, model_params
        except TypeError:
            model, model_params = build_model(
                config,
                y_train,
                predictor_columns,
                numeric_columns,
                categorical_columns,
                n_estimators_override,
            )
            if early_stopping_rounds is not None:
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            try:
                model.fit(x_train, y_train, **fit_kwargs)
                return model, model_params
            except TypeError:
                logger.warning("XGBoost early stopping is not compatible with this installed version. Training without it.")
                fit_kwargs.pop("early_stopping_rounds", None)
                fit_kwargs.pop("eval_set", None)
                fit_kwargs.pop("verbose", None)
                model, model_params = build_model(
                    config,
                    y_train,
                    predictor_columns,
                    numeric_columns,
                    categorical_columns,
                    n_estimators_override,
                )

    model.fit(x_train, y_train, **fit_kwargs)
    return model, model_params


def get_xgboost_best_n_estimators(model: Any, logger: logging.Logger) -> int | None:
    """Return the validated boosting round count from an early-stopped XGBoost model."""
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        try:
            return max(1, int(best_iteration) + 1)
        except Exception:
            logger.warning("Could not parse XGBoost best_iteration value: %s", best_iteration)

    best_ntree_limit = getattr(model, "best_ntree_limit", None)
    if best_ntree_limit is not None:
        try:
            return max(1, int(best_ntree_limit))
        except Exception:
            logger.warning("Could not parse XGBoost best_ntree_limit value: %s", best_ntree_limit)

    try:
        booster = model.get_booster()
        booster_best_iteration = booster.attr("best_iteration")
        if booster_best_iteration is not None:
            return max(1, int(booster_best_iteration) + 1)
    except Exception:
        return None
    return None


def choose_final_xgboost_n_estimators(
    validated_n_estimators: list[int],
    config: dict[str, Any],
    logger: logging.Logger,
) -> tuple[int | None, dict[str, Any]]:
    if str(config["model_type"]).lower() != "xgboost":
        return None, {"enabled": False, "reason": "model_type_is_not_xgboost"}
    if not validated_n_estimators:
        return None, {
            "enabled": False,
            "reason": "no_validated_best_iteration_available",
            "configured_n_estimators": int(config["xgboost_params"]["n_estimators"]),
        }

    selected = int(round(float(np.median(validated_n_estimators))))
    selected = max(1, selected)
    strategy = "single_validation_split" if len(validated_n_estimators) == 1 else "median_across_validation_folds"
    logger.info(
        "Validated XGBoost boosting rounds from early stopping: %s; final production n_estimators=%d using %s.",
        validated_n_estimators,
        selected,
        strategy,
    )
    return selected, {
        "enabled": True,
        "selection_strategy": strategy,
        "validation_best_n_estimators": [int(value) for value in validated_n_estimators],
        "final_n_estimators": selected,
        "configured_n_estimators": int(config["xgboost_params"]["n_estimators"]),
    }


def suggest_xgboost_params(
    trial: Any,
    scale_pos_weight_reference: float | None = None,
    tune_scale_pos_weight: bool = True,
) -> dict[str, Any]:
    if scale_pos_weight_reference is None or not np.isfinite(scale_pos_weight_reference) or scale_pos_weight_reference <= 0:
        scale_pos_weight_low = 0.5
        scale_pos_weight_high = 10.0
    else:
        scale_pos_weight_low = max(0.1, float(scale_pos_weight_reference) * 0.25)
        scale_pos_weight_high = max(scale_pos_weight_low * 1.01, float(scale_pos_weight_reference) * 4.0)

    params = {
        "n_estimators": 2000,
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 15.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 20.0),
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
    }
    if tune_scale_pos_weight:
        params["scale_pos_weight"] = trial.suggest_float(
            "scale_pos_weight",
            scale_pos_weight_low,
            scale_pos_weight_high,
            log=True,
        )
    return params


def score_tuning_predictions(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    metric_name: str,
    logger: logging.Logger,
) -> float:
    y_array = np.asarray(y_true, dtype=int)
    try:
        if metric_name == "PR_AUC":
            return float(average_precision_score(y_array, probabilities))
        if metric_name == "ROC_AUC":
            return float(roc_auc_score(y_array, probabilities))
        if metric_name == "F1":
            predicted = (probabilities >= 0.5).astype(int)
            return float(f1_score(y_array, predicted, zero_division=0))
    except Exception as exc:
        logger.warning("Tuning metric %s is undefined for this fold: %s", metric_name, exc)
    return np.nan


def write_tuning_trials_csv(study: Any, output_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "trial_number": trial.number,
            "state": str(trial.state),
            "value": trial.value,
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        row.update({f"user_{key}": value for key, value in trial.user_attrs.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def prepare_tuning_fold_cache(
    tuning_splits: list[tuple[str, pd.DataFrame, pd.DataFrame]],
    predictor_columns: list[str],
    config: dict[str, Any],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """Fit fold-local preprocessing once so Optuna trials can reuse matrices."""
    target_column = str(config["target_column"])
    cached_folds: list[dict[str, Any]] = []
    for split_name, train_df, validation_df in tuning_splits:
        numeric_columns, categorical_columns = get_predictor_type_columns(
            train_df,
            predictor_columns,
            config,
        )
        preprocessor = build_preprocessor(
            predictor_columns,
            numeric_columns,
            categorical_columns,
            passthrough_numeric_missing=True,
        )
        x_train = preprocessor.fit_transform(train_df[predictor_columns])
        x_validation = preprocessor.transform(validation_df[predictor_columns])
        cached_folds.append(
            {
                "name": split_name,
                "x_train": x_train,
                "y_train": train_df[target_column].to_numpy(dtype=np.int8),
                "x_validation": x_validation,
                "y_validation": validation_df[target_column].to_numpy(dtype=np.int8),
            }
        )
    logger.info(
        "Cached fold-local preprocessing for %d Optuna folds; preprocessing will not be refit per trial.",
        len(cached_folds),
    )
    return cached_folds


def fit_cached_xgboost_fold(
    cached_fold: dict[str, Any],
    xgboost_params: dict[str, Any],
    random_seed: int,
) -> tuple[Any, np.ndarray]:
    from xgboost import XGBClassifier

    params = deepcopy(xgboost_params)
    params.setdefault("random_state", random_seed)
    params.setdefault("n_jobs", -1)
    estimator = XGBClassifier(**params)
    estimator.fit(
        cached_fold["x_train"],
        cached_fold["y_train"],
        eval_set=[(cached_fold["x_validation"], cached_fold["y_validation"])],
        verbose=False,
    )
    probabilities = estimator.predict_proba(cached_fold["x_validation"])[:, 1].astype(float)
    return estimator, probabilities


def run_hyperparameter_tuning(
    config: dict[str, Any],
    tuning_df: pd.DataFrame,
    predictor_columns: list[str],
    output_dir: Path,
    logger: logging.Logger,
    tuning_source: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not bool(config.get("enable_hyperparameter_tuning", False)):
        return config, {"enabled": False, "reason": "disabled"}
    if str(config["model_type"]).lower() != "xgboost":
        raise ValueError("Hyperparameter tuning currently supports model_type='xgboost' only.")

    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "optuna is required when enable_hyperparameter_tuning=True. "
            "Install optuna or set enable_hyperparameter_tuning=False."
        ) from exc

    metric_name = str(config["tuning_metric"])
    n_trials = int(config["tuning_trials"])
    target_column = str(config["target_column"])
    positive_count = int(np.count_nonzero(tuning_df[target_column] == 1))
    negative_count = int(np.count_nonzero(tuning_df[target_column] == 0))
    scale_pos_weight_reference = negative_count / positive_count if positive_count > 0 else None
    tune_scale_pos_weight = str(config["class_imbalance_strategy"]) == "auto_scale_pos_weight"
    logger.info(
        "Starting XGBoost hyperparameter tuning with Optuna: trials=%d, metric=%s, validation_mode=%s, source=%s, "
        "scale_pos_weight_reference=%s, tune_scale_pos_weight=%s.",
        n_trials,
        metric_name,
        VALIDATION_MODE,
        tuning_source,
        scale_pos_weight_reference,
        tune_scale_pos_weight,
    )
    tuning_validation_mode_used, tuning_splits = resolve_tuning_splits(
        tuning_df,
        config,
        logger,
    )
    cached_folds = prepare_tuning_fold_cache(
        tuning_splits,
        predictor_columns,
        config,
        logger,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=int(config["random_seed"]))
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(10, max(5, n_trials // 10)),
        n_warmup_steps=1,
        interval_steps=1,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial: Any) -> float:
        trial_params = deepcopy(config["xgboost_params"])
        trial_params.update(suggest_xgboost_params(trial, scale_pos_weight_reference, tune_scale_pos_weight))

        fold_scores: list[float] = []
        fold_best_n_estimators: list[int] = []
        for fold_index, cached_fold in enumerate(cached_folds, start=1):
            split_name = str(cached_fold["name"])
            try:
                estimator, probabilities = fit_cached_xgboost_fold(
                    cached_fold,
                    trial_params,
                    int(config["random_seed"]),
                )
                score = score_tuning_predictions(
                    cached_fold["y_validation"],
                    probabilities,
                    metric_name,
                    logger,
                )
                if np.isfinite(score):
                    fold_scores.append(float(score))
                best_n_estimators = get_xgboost_best_n_estimators(estimator, logger)
                if best_n_estimators is not None:
                    fold_best_n_estimators.append(best_n_estimators)
            except Exception as exc:
                logger.warning("Optuna trial %d failed on %s: %s", trial.number, split_name, exc)
                return 0.0
            if fold_scores:
                trial.report(float(np.nanmean(fold_scores)), step=fold_index)
                if trial.should_prune():
                    raise optuna.TrialPruned(
                        f"Pruned after {fold_index} folds with interim {metric_name}="
                        f"{float(np.nanmean(fold_scores)):.6f}."
                    )

        if not fold_scores:
            return 0.0
        trial.set_user_attr("mean_best_n_estimators", float(np.mean(fold_best_n_estimators)) if fold_best_n_estimators else np.nan)
        trial.set_user_attr("fold_count", len(fold_scores))
        return float(np.nanmean(fold_scores))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, gc_after_trial=True)
    trials_path = output_dir / OUTPUTS["tuning_trials"]
    write_tuning_trials_csv(study, trials_path)
    completed_trial_count = sum(
        1 for trial in study.trials if str(trial.state.name) == "COMPLETE"
    )
    pruned_trial_count = sum(
        1 for trial in study.trials if str(trial.state.name) == "PRUNED"
    )

    best_params = deepcopy(config["xgboost_params"])
    best_params.update(
        {
            "n_estimators": 2000,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            **study.best_params,
        }
    )
    tuned_config = deepcopy(config)
    tuned_config["xgboost_params"] = best_params
    logger.info(
        "Finished Optuna tuning. Best %s=%.6f from trial %d.",
        metric_name,
        float(study.best_value),
        int(study.best_trial.number),
    )
    return tuned_config, {
        "enabled": True,
        "library": "optuna",
        "tuning_trials_requested": n_trials,
        "tuning_trials_started": len(study.trials),
        "tuning_trials_completed": completed_trial_count,
        "tuning_trials_pruned": pruned_trial_count,
        "fold_preprocessing_cache": True,
        "pruner": "MedianPruner",
        "tuning_metric": metric_name,
        "tuning_validation_mode_requested": VALIDATION_MODE,
        "tuning_validation_mode_used": tuning_validation_mode_used,
        "tuning_source": tuning_source,
        "best_trial_number": int(study.best_trial.number),
        "best_metric_value": float(study.best_value),
        "best_params": best_params,
        "trials_path": trials_path.name,
    }


def predict_probabilities(model: Any, df: pd.DataFrame, predictor_columns: list[str]) -> np.ndarray:
    probabilities = model.predict_proba(df[predictor_columns])
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Model predict_proba did not return two-class probabilities.")
    return probabilities[:, 1].astype(float)


def analyze_thresholds(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y_array = np.asarray(y_true, dtype=int)
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        predicted = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_array, predicted, labels=[0, 1]).ravel()
        precision = precision_score(y_array, predicted, zero_division=0)
        recall = recall_score(y_array, predicted, zero_division=0)
        f1 = f1_score(y_array, predicted, zero_division=0)
        false_positive_rate = fp / (fp + tn) if (fp + tn) else np.nan
        true_positive_rate = tp / (tp + fn) if (tp + fn) else np.nan
        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "F1": f1,
                "false_positive_rate": false_positive_rate,
                "true_positive_rate": true_positive_rate,
                "predicted_positive_count": int(np.count_nonzero(predicted == 1)),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(
    threshold_analysis: pd.DataFrame,
    config: dict[str, Any],
) -> float:
    mode = str(config["probability_threshold"]).lower()
    if mode == "fixed":
        return float(config["fixed_threshold"])
    if mode == "default_0.5":
        return 0.5
    if threshold_analysis.empty:
        return 0.5
    if mode == "optimize_f1":
        sorted_df = threshold_analysis.sort_values(["F1", "threshold"], ascending=[False, True])
        return float(sorted_df.iloc[0]["threshold"])
    if mode == "optimize_recall_precision_balance":
        analysis = threshold_analysis.copy()
        analysis["precision_recall_gap"] = (analysis["precision"] - analysis["recall"]).abs()
        sorted_df = analysis.sort_values(["precision_recall_gap", "F1"], ascending=[True, False])
        return float(sorted_df.iloc[0]["threshold"])
    return 0.5


def safe_metric(
    metric_name: str,
    func: Any,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predicted: np.ndarray,
    logger: logging.Logger,
) -> float:
    try:
        if metric_name in {"PR_AUC", "ROC_AUC", "brier_score", "log_loss"}:
            return float(func(y_true, probabilities))
        return float(func(y_true, predicted))
    except Exception as exc:
        logger.warning("Metric %s is undefined for this split: %s", metric_name, exc)
        return np.nan


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    y_array = np.asarray(y_true, dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)
    bin_ids = np.minimum(
        np.floor(np.clip(probability_array, 0.0, 1.0) * n_bins).astype(int),
        n_bins - 1,
    )
    error = 0.0
    for bin_id in range(n_bins):
        selection = bin_ids == bin_id
        if not np.any(selection):
            continue
        observed = float(np.mean(y_array[selection]))
        predicted = float(np.mean(probability_array[selection]))
        error += float(np.mean(selection)) * abs(observed - predicted)
    return float(error)


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    split_name: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    # PR-AUC is especially important because depositional pixels may be imbalanced
    # relative to valid non-depositional pixels.
    y_array = np.asarray(y_true, dtype=int)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_array, predicted, labels=[0, 1]).ravel()

    def safe_log_loss(y: np.ndarray, p: np.ndarray) -> float:
        return float(log_loss(y, p, labels=[0, 1]))

    return {
        "split": split_name,
        "PR_AUC": safe_metric("PR_AUC", average_precision_score, y_array, probabilities, predicted, logger),
        "ROC_AUC": safe_metric("ROC_AUC", roc_auc_score, y_array, probabilities, predicted, logger),
        "F1": safe_metric("F1", lambda y, pred: f1_score(y, pred, zero_division=0), y_array, probabilities, predicted, logger),
        "precision": safe_metric(
            "precision", lambda y, pred: precision_score(y, pred, zero_division=0), y_array, probabilities, predicted, logger
        ),
        "recall": safe_metric(
            "recall", lambda y, pred: recall_score(y, pred, zero_division=0), y_array, probabilities, predicted, logger
        ),
        "accuracy": safe_metric("accuracy", accuracy_score, y_array, probabilities, predicted, logger),
        "balanced_accuracy": safe_metric("balanced_accuracy", balanced_accuracy_score, y_array, probabilities, predicted, logger),
        "brier_score": safe_metric("brier_score", brier_score_loss, y_array, probabilities, predicted, logger),
        "log_loss": safe_metric("log_loss", safe_log_loss, y_array, probabilities, predicted, logger),
        "expected_calibration_error": expected_calibration_error(y_array, probabilities),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "threshold": float(threshold),
    }


def export_predictions(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    output_path: Path | None,
    config: dict[str, Any],
    split_name: str,
    fold_name: str | None = None,
) -> pd.DataFrame:
    target_column = str(config["target_column"])
    columns = [
        column
        for column in [*config.get("id_columns", []), *config.get("group_columns", [])]
        if column in df.columns
    ]
    output_df = df[columns].copy() if columns else pd.DataFrame(index=df.index)
    output_df[target_column] = df[target_column].to_numpy()
    output_df["predicted_probability"] = probabilities
    output_df["predicted_class"] = (probabilities >= threshold).astype(int)
    output_df["threshold"] = threshold
    output_df["split"] = split_name
    if fold_name is not None:
        output_df["fold"] = fold_name
    if output_path is not None:
        output_df.to_csv(output_path, index=False)
    return output_df


def get_model_feature_names(model: Any, predictor_columns: list[str]) -> list[str]:
    if hasattr(model, "transformed_feature_names_") and getattr(model, "transformed_feature_names_"):
        return [str(name) for name in getattr(model, "transformed_feature_names_")]
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        try:
            return [str(name) for name in model.named_steps["preprocessor"].get_feature_names_out()]
        except Exception:
            return predictor_columns
    return predictor_columns


def get_categorical_levels(
    model: Any,
    categorical_columns: list[str],
) -> dict[str, list[str]]:
    if not categorical_columns:
        return {}
    preprocessor = getattr(model, "preprocessor", None)
    if preprocessor is None and hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor")
    if preprocessor is None:
        return {}
    try:
        transformer = preprocessor.named_transformers_["categorical"]
        encoder = transformer.named_steps["onehot"]
        return {
            column: [str(value) for value in levels.tolist()]
            for column, levels in zip(
                categorical_columns,
                encoder.categories_,
                strict=True,
            )
        }
    except Exception:
        return {}


def sample_explainability_rows(df: pd.DataFrame, sample_size: int, random_seed: int) -> pd.DataFrame:
    if len(df) <= int(sample_size):
        return df.reset_index(drop=True).copy()
    return df.sample(n=int(sample_size), random_state=int(random_seed)).reset_index(drop=True).copy()


def source_predictor_from_model_feature(feature_name: str, predictor_columns: list[str]) -> str:
    base_name = str(feature_name).split("__", 1)[-1]
    for predictor in sorted(map(str, predictor_columns), key=len, reverse=True):
        if base_name == predictor or base_name.startswith(f"{predictor}_"):
            return predictor
    return base_name


def write_shap_plots(
    shap_feature_df: pd.DataFrame,
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    max_display_features: int,
    random_seed: int,
    importance_plot_path: Path,
    summary_plot_path: Path,
    dataset_label: str,
    logger: logging.Logger,
) -> None:
    predictor_rows = shap_feature_df.loc[shap_feature_df["level"] == "predictor"].copy()
    predictor_rows = predictor_rows.sort_values("mean_abs_shap", ascending=False).head(max_display_features)
    if not predictor_rows.empty:
        try:
            plt.figure(figsize=(9, max(4, len(predictor_rows) * 0.38)))
            plt.barh(predictor_rows["feature"], predictor_rows["mean_abs_shap"], color="#35618f")
            plt.gca().invert_yaxis()
            plt.xlabel("Mean absolute SHAP contribution to XGBoost model margin")
            plt.ylabel("Predictor")
            plt.title(f"SHAP Feature Importance for Depositional Favorability\n{dataset_label}")
            plt.tight_layout()
            plt.savefig(importance_plot_path, dpi=150)
            plt.close()
        except Exception as exc:
            logger.warning("Could not write SHAP feature importance plot: %s", exc)
            plt.close()

    feature_rows = shap_feature_df.loc[shap_feature_df["level"] == "model_feature"].copy()
    feature_rows = feature_rows.sort_values("mean_abs_shap", ascending=False).head(max_display_features)
    if feature_rows.empty:
        return
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
        ax.set_title(f"SHAP Contribution Summary for Depositional Favorability\n{dataset_label}")
        if color_handle is not None:
            colorbar = fig.colorbar(color_handle, ax=ax)
            colorbar.set_label("Transformed feature value")
        fig.tight_layout()
        fig.savefig(summary_plot_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        logger.warning("Could not write SHAP contribution summary plot: %s", exc)
        plt.close()


def write_shap_dependence_plots(
    shap_feature_df: pd.DataFrame,
    shap_sample: pd.DataFrame,
    predictor_shap_by_name: dict[str, np.ndarray],
    max_display_features: int,
    random_seed: int,
    dependence_plot_path: Path,
    dataset_label: str,
    logger: logging.Logger,
) -> Path | None:
    predictor_rows = shap_feature_df.loc[shap_feature_df["level"] == "predictor"].copy()
    predictor_rows = predictor_rows.sort_values("mean_abs_shap", ascending=False).head(max_display_features)
    plotted_predictors = [
        str(feature)
        for feature in predictor_rows["feature"].tolist()
        if str(feature) in predictor_shap_by_name and str(feature) in shap_sample.columns
    ]
    if not plotted_predictors:
        return None
    try:
        rng = np.random.default_rng(int(random_seed))
        column_count = min(3, len(plotted_predictors))
        row_count = int(math.ceil(len(plotted_predictors) / column_count))
        fig, axes = plt.subplots(row_count, column_count, figsize=(5.4 * column_count, 3.9 * row_count), squeeze=False)
        for axis_index, predictor in enumerate(plotted_predictors):
            ax = axes[axis_index // column_count][axis_index % column_count]
            y_values = np.asarray(predictor_shap_by_name[predictor], dtype=float)
            raw_values = shap_sample[predictor].reset_index(drop=True)
            numeric_values = pd.to_numeric(raw_values, errors="coerce")
            use_numeric = numeric_values.notna().mean() >= 0.9 and numeric_values.nunique(dropna=True) > 12
            if use_numeric:
                color_handle = ax.scatter(
                    numeric_values.to_numpy(dtype=float),
                    y_values,
                    c=numeric_values.to_numpy(dtype=float),
                    cmap="viridis",
                    alpha=0.45,
                    s=11,
                    linewidths=0,
                )
                fig.colorbar(color_handle, ax=ax, fraction=0.046, pad=0.04).set_label(f"{predictor} value")
                ax.set_xlabel(predictor)
            else:
                labels = raw_values.where(raw_values.notna(), "missing").astype(str)
                ordered_labels = labels.value_counts().index.tolist()[:12]
                label_lookup = {label: index for index, label in enumerate(ordered_labels)}
                positions = labels.map(label_lookup)
                keep_mask = positions.notna().to_numpy()
                x_values = positions[keep_mask].to_numpy(dtype=float)
                jitter = rng.normal(0.0, 0.08, size=x_values.size)
                ax.scatter(x_values + jitter, y_values[keep_mask], color="#D55E00", alpha=0.45, s=11, linewidths=0)
                ax.set_xticks(np.arange(len(ordered_labels)))
                ax.set_xticklabels(ordered_labels, rotation=35, ha="right")
                ax.set_xlabel(predictor)
            ax.axhline(0.0, color="#444444", linewidth=1)
            ax.set_ylabel("SHAP contribution to model margin")
            ax.set_title(f"SHAP Dependence: {predictor}")
            ax.grid(alpha=0.2)
        for axis_index in range(len(plotted_predictors), row_count * column_count):
            axes[axis_index // column_count][axis_index % column_count].axis("off")
        fig.suptitle(f"SHAP Dependence for Depositional Favorability\n{dataset_label}", y=1.01)
        fig.tight_layout()
        fig.savefig(dependence_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return dependence_plot_path
    except Exception as exc:
        logger.warning("Could not write SHAP dependence plots: %s", exc)
        plt.close()
        return None


def export_xgboost_shap_analysis(
    model: Any,
    sample_df: pd.DataFrame,
    predictor_columns: list[str],
    output_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    dataset_key: str,
    dataset_label: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    if str(config["model_type"]).lower() != "xgboost":
        logger.info("Skipping SHAP contribution export because final model type is not XGBoost.")
        return {"enabled": False, "reason": "model_type_is_not_xgboost"}
    if not hasattr(model, "preprocessor") or not hasattr(model, "get_booster"):
        logger.warning("Skipping SHAP contribution export because XGBoost preprocessing hooks are unavailable.")
        return {"enabled": False, "reason": "xgboost_preprocessor_or_booster_unavailable"}

    try:
        from xgboost import DMatrix

        shap_sample = sample_explainability_rows(
            sample_df,
            int(config["shap_sample_size"]),
            int(config["random_seed"]),
        )
        transformed = np.asarray(model.preprocessor.transform(shap_sample[predictor_columns]), dtype=float)
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
                "Expected SHAP contribution matrix with one column per transformed feature plus the bias term; "
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
                    "sample_count": int(len(shap_sample)),
                }
            )

        model_feature_df = pd.DataFrame(model_feature_rows)
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
                    "sample_count": int(len(shap_sample)),
                }
            )
        predictor_df = pd.DataFrame(predictor_rows)
        shap_feature_df = pd.concat([predictor_df, model_feature_df], ignore_index=True, sort=False)
        shap_feature_df = shap_feature_df.sort_values(["level", "mean_abs_shap"], ascending=[True, False]).reset_index(drop=True)
        shap_feature_df["rank"] = (
            shap_feature_df.groupby("level")["mean_abs_shap"].rank(method="first", ascending=False).astype(int)
        )
        shap_feature_df[
            ["level", "feature", "source_predictor", "mean_abs_shap", "mean_shap", "std_shap", "sample_count", "rank"]
        ].to_csv(output_path, index=False)
        importance_plot_path = output_dir / f"{dataset_key}_shap_feature_importance.png"
        summary_plot_path = output_dir / f"{dataset_key}_shap_summary_beeswarm.png"
        dependence_plot_path = output_dir / f"{dataset_key}_shap_dependence_plots.png"
        write_shap_plots(
            shap_feature_df,
            shap_values,
            transformed,
            feature_names,
            int(config["shap_max_display_features"]),
            int(config["random_seed"]),
            importance_plot_path,
            summary_plot_path,
            dataset_label,
            logger,
        )
        dependence_path = write_shap_dependence_plots(
            shap_feature_df,
            shap_sample,
            predictor_shap_by_name,
            int(config["shap_max_display_features"]),
            int(config["random_seed"]),
            dependence_plot_path,
            dataset_label,
            logger,
        )
        logger.info(
            "Wrote XGBoost SHAP contribution summary from %d sampled rows; mean bias contribution=%.6f.",
            len(shap_sample),
            float(np.mean(bias_values)),
        )
        return {
            "enabled": True,
            "method": "xgboost_pred_contribs",
            "sample_count": int(len(shap_sample)),
            "importance_path": str(output_path),
            "feature_importance_plot_path": str(importance_plot_path),
            "summary_plot_path": str(summary_plot_path),
            "dependence_plot_path": str(dependence_path) if dependence_path is not None else None,
            "mean_bias_contribution": float(np.mean(bias_values)),
            "contribution_output": "xgboost_raw_margin",
        }
    except Exception as exc:
        logger.warning("Could not export XGBoost SHAP contribution analysis: %s", exc)
        return {"enabled": False, "reason": str(exc)}


def write_partial_dependence_plot(
    partial_dependence_df: pd.DataFrame,
    predictor_columns: list[str],
    plot_path: Path,
    dataset_label: str,
    logger: logging.Logger,
) -> None:
    plotted_features = [column for column in predictor_columns if column in set(partial_dependence_df["feature"])]
    if not plotted_features:
        return
    try:
        column_count = min(3, len(plotted_features))
        row_count = int(math.ceil(len(plotted_features) / column_count))
        fig, axes = plt.subplots(row_count, column_count, figsize=(5.2 * column_count, 3.8 * row_count), squeeze=False)
        for axis_index, feature in enumerate(plotted_features):
            ax = axes[axis_index // column_count][axis_index % column_count]
            feature_df = partial_dependence_df.loc[partial_dependence_df["feature"] == feature].copy()
            feature_kind = str(feature_df["feature_kind"].iloc[0])
            if feature_kind == "categorical":
                positions = np.arange(len(feature_df))
                ax.bar(positions, feature_df["mean_predicted_probability"], color="#5f8f6f")
                ax.set_xticks(positions)
                ax.set_xticklabels(feature_df["feature_value_label"], rotation=35, ha="right")
                ax.set_xlabel(f"Forced category for {feature}")
            else:
                ax.plot(feature_df["feature_value_numeric"], feature_df["mean_predicted_probability"], marker="o", linewidth=1.5)
                ax.set_xlabel(f"Forced value of {feature}")
            ax.set_ylabel("Mean predicted probability of depositional terrain")
            ax.set_ylim(0, 1)
            ax.set_title(f"Partial Dependence: {feature}")
            ax.grid(alpha=0.2)
        for axis_index in range(len(plotted_features), row_count * column_count):
            axes[axis_index // column_count][axis_index % column_count].axis("off")
        fig.suptitle(f"Partial Dependence for Final ML-DFI Model\n{dataset_label}", y=1.01)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning("Could not write partial dependence plots: %s", exc)
        plt.close()


def write_combined_partial_dependence_plot(
    partial_dependence_frames: list[pd.DataFrame],
    predictor_columns: list[str],
    plot_path: Path,
    logger: logging.Logger,
) -> Path | None:
    valid_frames = [frame for frame in partial_dependence_frames if frame is not None and not frame.empty]
    if not valid_frames:
        return None
    combined_df = pd.concat(valid_frames, ignore_index=True)
    if combined_df.empty:
        return None
    plotted_features = [column for column in predictor_columns if column in set(combined_df["feature"])]
    if not plotted_features:
        return None
    try:
        column_count = min(3, len(plotted_features))
        row_count = int(math.ceil(len(plotted_features) / column_count))
        fig, axes = plt.subplots(row_count, column_count, figsize=(5.4 * column_count, 3.9 * row_count), squeeze=False)
        for axis_index, feature in enumerate(plotted_features):
            ax = axes[axis_index // column_count][axis_index % column_count]
            feature_df = combined_df.loc[combined_df["feature"] == feature].copy()
            feature_kind = str(feature_df["feature_kind"].iloc[0])
            if feature_kind == "categorical":
                labels = sorted(feature_df["feature_value_label"].astype(str).unique())
                positions = np.arange(len(labels))
                position_lookup = {label: index for index, label in enumerate(labels)}
                for dataset_key, dataset_df in feature_df.groupby("dataset_key", sort=False):
                    dataset_df = dataset_df.copy()
                    dataset_df["plot_position"] = dataset_df["feature_value_label"].astype(str).map(position_lookup)
                    dataset_df = dataset_df.sort_values("plot_position")
                    ax.plot(
                        dataset_df["plot_position"],
                        dataset_df["mean_predicted_probability"],
                        marker="o",
                        linewidth=1.8,
                        color=PLOT_COLORS.get(str(dataset_key), "#333333"),
                        label=str(dataset_df["dataset_label"].iloc[0]),
                    )
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=35, ha="right")
                ax.set_xlabel(f"Forced category for {feature}")
            else:
                for dataset_key, dataset_df in feature_df.groupby("dataset_key", sort=False):
                    dataset_df = dataset_df.sort_values("feature_value_numeric")
                    ax.plot(
                        dataset_df["feature_value_numeric"],
                        dataset_df["mean_predicted_probability"],
                        marker="o",
                        linewidth=1.8,
                        color=PLOT_COLORS.get(str(dataset_key), "#333333"),
                        label=str(dataset_df["dataset_label"].iloc[0]),
                    )
                ax.set_xlabel(f"Forced value of {feature}")
            ax.set_ylabel("Mean predicted probability of depositional terrain")
            ax.set_ylim(0, 1)
            ax.set_title(f"Partial Dependence: {feature}")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        for axis_index in range(len(plotted_features), row_count * column_count):
            axes[axis_index // column_count][axis_index % column_count].axis("off")
        fig.suptitle("Partial Dependence for Final ML-DFI Model\nFull Training vs Holdout Validation", y=1.01)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return plot_path
    except Exception as exc:
        logger.warning("Could not write combined partial dependence plots: %s", exc)
        plt.close()
        return None


def export_partial_dependence_analysis(
    model: Any,
    sample_df: pd.DataFrame,
    predictor_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    output_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    dataset_key: str,
    dataset_label: str,
    logger: logging.Logger,
    write_plot: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    evaluation_df = sample_explainability_rows(
        sample_df,
        int(config["partial_dependence_sample_size"]),
        int(config["random_seed"]),
    )
    rows: list[dict[str, Any]] = []
    grid_resolution = int(config["partial_dependence_grid_resolution"])
    max_categories = int(config["partial_dependence_max_categories"])

    for feature in predictor_columns:
        base_predictors = evaluation_df[predictor_columns].copy()
        if feature in categorical_columns:
            categories = evaluation_df[feature].dropna().value_counts().head(max_categories).index.tolist()
            for category in categories:
                forced_predictors = base_predictors.copy()
                forced_predictors[feature] = category
                probabilities = predict_probabilities(model, forced_predictors, predictor_columns)
                rows.append(
                    {
                        "feature": feature,
                        "feature_kind": "categorical",
                        "feature_value_numeric": np.nan,
                        "feature_value_label": str(category),
                        "mean_predicted_probability": float(np.mean(probabilities)),
                        "sample_count": int(len(forced_predictors)),
                        "dataset_key": dataset_key,
                        "dataset_label": dataset_label,
                    }
                )
            continue

        numeric_values = pd.to_numeric(evaluation_df[feature], errors="coerce")
        numeric_values = numeric_values[np.isfinite(numeric_values.to_numpy(dtype=float))]
        if numeric_values.nunique(dropna=True) < 2:
            logger.warning("Skipping partial dependence for '%s' because it has fewer than two finite numeric values.", feature)
            continue
        grid_values = np.unique(
            np.quantile(numeric_values.to_numpy(dtype=float), np.linspace(0.05, 0.95, grid_resolution))
        )
        for grid_value in grid_values:
            forced_predictors = base_predictors.copy()
            forced_predictors[feature] = float(grid_value)
            probabilities = predict_probabilities(model, forced_predictors, predictor_columns)
            rows.append(
                {
                    "feature": feature,
                    "feature_kind": "numeric",
                    "feature_value_numeric": float(grid_value),
                    "feature_value_label": f"{float(grid_value):.6g}",
                    "mean_predicted_probability": float(np.mean(probabilities)),
                    "sample_count": int(len(forced_predictors)),
                    "dataset_key": dataset_key,
                    "dataset_label": dataset_label,
                }
            )

    partial_dependence_df = pd.DataFrame(rows)
    if partial_dependence_df.empty:
        logger.warning("No usable partial dependence rows were created.")
        return {"enabled": False, "reason": "no_usable_partial_dependence_rows"}, partial_dependence_df
    partial_dependence_df.to_csv(output_path, index=False)
    plot_path = output_dir / f"{dataset_key}_partial_dependence_plots.png"
    if write_plot:
        write_partial_dependence_plot(partial_dependence_df, predictor_columns, plot_path, dataset_label, logger)
    logger.info(
        "Wrote partial dependence summary for %d predictors from %d sampled rows.",
        int(partial_dependence_df["feature"].nunique()),
        len(evaluation_df),
    )
    return (
        {
            "enabled": True,
            "sample_count": int(len(evaluation_df)),
            "predictor_count": int(partial_dependence_df["feature"].nunique()),
            "data_path": str(output_path),
            "plot_path": str(plot_path) if write_plot else None,
        },
        partial_dependence_df,
    )


def write_metadata_json(metadata: dict[str, Any], output_path: Path) -> None:
    def clean_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean_value(item) for item in value]
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    output_path.write_text(json.dumps(clean_value(metadata), indent=2), encoding="utf-8")


def write_metrics_csv(metrics_rows: list[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_path, index=False)
    return metrics_df


def write_threshold_analysis_csv(threshold_analysis: pd.DataFrame, output_path: Path) -> None:
    threshold_analysis.to_csv(output_path, index=False)


def save_figure(fig: Any, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_overlay_pr_curve(
    output_dir: Path,
    evaluation_sets: list[dict[str, Any]],
    logger: logging.Logger,
) -> Path | None:
    output_path = output_dir / "pr_curve.png"
    try:
        fig, ax = plt.subplots(figsize=(8.5, 6.2))
        for evaluation_set in evaluation_sets:
            y_array = np.asarray(evaluation_set["y_true"], dtype=int)
            probabilities = np.asarray(evaluation_set["probabilities"], dtype=float)
            precision, recall, _ = precision_recall_curve(y_array, probabilities)
            pr_auc = average_precision_score(y_array, probabilities)
            prevalence = float(np.mean(y_array == 1))
            color = PLOT_COLORS.get(str(evaluation_set["key"]), "#333333")
            ax.plot(
                recall,
                precision,
                color=color,
                linewidth=2.2,
                label=f"{evaluation_set['label']} PR-AUC = {pr_auc:.3f}",
            )
            ax.axhline(
                prevalence,
                color=color,
                linestyle="--",
                alpha=0.45,
                linewidth=1.2,
                label=f"{evaluation_set['label']} prevalence = {prevalence:.3f}",
            )
        ax.set_xlabel(f"Recall for {POSITIVE_TARGET_LABEL}")
        ax.set_ylabel(f"Precision for {POSITIVE_TARGET_LABEL}")
        ax.set_title("Precision-Recall Curves for Depositional Favorability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8.5)
        save_figure(fig, output_path)
        return output_path
    except Exception as exc:
        logger.warning("Could not write overlaid PR curve plot: %s", exc)
        plt.close()
        return None


def write_overlay_roc_curve(
    output_dir: Path,
    evaluation_sets: list[dict[str, Any]],
    logger: logging.Logger,
) -> Path | None:
    output_path = output_dir / "roc_curve.png"
    try:
        fig, ax = plt.subplots(figsize=(8.5, 6.2))
        for evaluation_set in evaluation_sets:
            y_array = np.asarray(evaluation_set["y_true"], dtype=int)
            probabilities = np.asarray(evaluation_set["probabilities"], dtype=float)
            fpr, tpr, _ = roc_curve(y_array, probabilities)
            roc_auc = roc_auc_score(y_array, probabilities)
            color = PLOT_COLORS.get(str(evaluation_set["key"]), "#333333")
            ax.plot(
                fpr,
                tpr,
                color=color,
                linewidth=2.2,
                label=f"{evaluation_set['label']} ROC-AUC = {roc_auc:.3f}",
            )
        ax.plot([0, 1], [0, 1], color="#666666", linestyle="--", label="No-skill baseline")
        ax.set_xlabel(f"False positive rate for {POSITIVE_TARGET_LABEL}")
        ax.set_ylabel(f"True positive rate for {POSITIVE_TARGET_LABEL}")
        ax.set_title("ROC Curves for Depositional Favorability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8.5)
        save_figure(fig, output_path)
        return output_path
    except Exception as exc:
        logger.warning("Could not write overlaid ROC curve plot: %s", exc)
        plt.close()
        return None


def write_overlay_calibration_curve(
    output_dir: Path,
    evaluation_sets: list[dict[str, Any]],
    logger: logging.Logger,
) -> Path | None:
    output_path = output_dir / "calibration_curve.png"
    try:
        fig, ax = plt.subplots(figsize=(7.4, 6.2))
        ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", label="Perfect calibration")
        for evaluation_set in evaluation_sets:
            y_array = np.asarray(evaluation_set["y_true"], dtype=int)
            probabilities = np.asarray(evaluation_set["probabilities"], dtype=float)
            observed, predicted = calibration_curve(
                y_array,
                probabilities,
                n_bins=10,
                strategy="quantile",
            )
            ece = expected_calibration_error(y_array, probabilities)
            color = PLOT_COLORS.get(str(evaluation_set["key"]), "#333333")
            ax.plot(
                predicted,
                observed,
                marker="o",
                linewidth=2,
                color=color,
                label=f"{evaluation_set['label']} (ECE={ece:.3f})",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted depositional probability")
        ax.set_ylabel("Observed depositional fraction")
        ax.set_title("Probability Calibration")
        ax.grid(alpha=0.2)
        ax.legend()
        save_figure(fig, output_path)
        return output_path
    except Exception as exc:
        logger.warning("Could not write calibration curve: %s", exc)
        plt.close()
        return None


def write_confusion_matrix_plot(
    output_dir: Path,
    evaluation_set: dict[str, Any],
    threshold: float,
    logger: logging.Logger,
) -> Path | None:
    output_path = output_dir / f"{evaluation_set['key']}_confusion_matrix.png"
    try:
        y_array = np.asarray(evaluation_set["y_true"], dtype=int)
        probabilities = np.asarray(evaluation_set["probabilities"], dtype=float)
        predicted = (probabilities >= threshold).astype(int)
        matrix = confusion_matrix(y_array, predicted, labels=[0, 1])
        class_tick_labels = ["Non-depositional\n(target=0)", "Depositional\n(target=1)"]
        fig, ax = plt.subplots(figsize=(7.6, 6.2))
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax, label="Pixel sample count", fraction=0.046, pad=0.04)
        ax.set_title(f"Confusion Matrix: {evaluation_set['label']}\nDiagnostic Threshold {threshold:.2f}")
        ax.set_xlabel("Predicted target class")
        ax.set_ylabel("Observed target class")
        ax.set_xticks([0, 1], class_tick_labels)
        ax.set_yticks([0, 1], class_tick_labels)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                text_color = "white" if matrix[row, col] > matrix.max() / 2 else "black"
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=text_color)
        save_figure(fig, output_path)
        return output_path
    except Exception as exc:
        logger.warning("Could not write %s confusion matrix plot: %s", evaluation_set["label"], exc)
        plt.close()
        return None


def write_probability_histogram(
    output_dir: Path,
    evaluation_set: dict[str, Any],
    threshold: float,
    logger: logging.Logger,
) -> Path | None:
    output_path = output_dir / f"{evaluation_set['key']}_probability_histogram.png"
    try:
        y_array = np.asarray(evaluation_set["y_true"], dtype=int)
        probabilities = np.asarray(evaluation_set["probabilities"], dtype=float)
        fig, ax = plt.subplots(figsize=(8.5, 5.8))
        ax.hist(probabilities[y_array == 0], bins=30, alpha=0.7, color="#56B4E9", label=NEGATIVE_TARGET_LABEL)
        ax.hist(probabilities[y_array == 1], bins=30, alpha=0.7, color="#E69F00", label=POSITIVE_TARGET_LABEL)
        ax.axvline(threshold, color="#222222", linestyle="--", label=f"Diagnostic threshold = {threshold:.2f}")
        ax.set_xlabel(f"Predicted probability of {POSITIVE_TARGET_LABEL}")
        ax.set_ylabel("Pixel sample count")
        ax.set_title(f"Predicted Probability Distribution by Observed Class\n{evaluation_set['label']}")
        ax.legend()
        save_figure(fig, output_path)
        return output_path
    except Exception as exc:
        logger.warning("Could not write %s probability histogram plot: %s", evaluation_set["label"], exc)
        plt.close()
        return None


def write_permutation_feature_importance(
    model: Any,
    evaluation_set: dict[str, Any],
    predictor_columns: list[str],
    target_column: str,
    output_dir: Path,
    config: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, str] | None:
    csv_path = output_dir / f"{evaluation_set['key']}_permutation_feature_importance.csv"
    plot_path = output_dir / f"{evaluation_set['key']}_feature_importance.png"
    try:
        data_df = evaluation_set["data_df"]
        y_array = data_df[target_column].astype(int).to_numpy()
        base_probabilities = predict_probabilities(model, data_df, predictor_columns)
        base_pr_auc = average_precision_score(y_array, base_probabilities)
        rng = np.random.default_rng(int(config["random_seed"]))
        rows: list[dict[str, Any]] = []
        for feature in predictor_columns:
            drops: list[float] = []
            feature_values = data_df[feature].to_numpy(copy=True)
            for _ in range(PERMUTATION_IMPORTANCE_REPEATS):
                permuted_df = data_df.copy()
                permuted_df[feature] = rng.permutation(feature_values)
                permuted_probabilities = predict_probabilities(model, permuted_df, predictor_columns)
                permuted_pr_auc = average_precision_score(y_array, permuted_probabilities)
                drops.append(float(base_pr_auc - permuted_pr_auc))
            rows.append(
                {
                    "feature": feature,
                    "importance_mean_pr_auc_drop": float(np.mean(drops)),
                    "importance_std_pr_auc_drop": float(np.std(drops)),
                    "baseline_pr_auc": float(base_pr_auc),
                    "importance_type": "permutation_mean_pr_auc_drop",
                    "dataset": evaluation_set["key"],
                    "sample_count": len(data_df),
                    "n_repeats": PERMUTATION_IMPORTANCE_REPEATS,
                }
            )
        importance_df = pd.DataFrame(rows).sort_values("importance_mean_pr_auc_drop", ascending=False)
        importance_df["rank"] = np.arange(1, len(importance_df) + 1)
        importance_df.to_csv(csv_path, index=False)

        plot_df = importance_df.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8.5, max(4.2, len(plot_df) * 0.38)))
        ax.barh(
            plot_df["feature"],
            plot_df["importance_mean_pr_auc_drop"],
            xerr=plot_df["importance_std_pr_auc_drop"],
            color=PLOT_COLORS.get(str(evaluation_set["key"]), "#555555"),
            alpha=0.85,
        )
        ax.axvline(0.0, color="#444444", linewidth=1)
        ax.set_xlabel("Mean decrease in PR-AUC after predictor permutation")
        ax.set_ylabel("Predictor")
        ax.set_title(f"Permutation Feature Importance\n{evaluation_set['label']}")
        ax.grid(axis="x", alpha=0.2)
        save_figure(fig, plot_path)
        return {"data_path": str(csv_path), "plot_path": str(plot_path)}
    except Exception as exc:
        logger.warning("Could not write %s permutation feature importance: %s", evaluation_set["label"], exc)
        plt.close()
        return None


def write_pr_auc_comparison_plot(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    logger: logging.Logger,
) -> Path | None:
    output_path = output_dir / "pr_auc_comparison_folds_training_holdout_line.png"
    try:
        rows: list[dict[str, Any]] = []
        for split_name in metrics_df["split"].astype(str):
            if split_name.startswith("fold_") and split_name.endswith("_threshold_optimized"):
                fold_number = int(split_name.split("_")[1])
                pr_auc = float(metrics_df.loc[metrics_df["split"] == split_name, "PR_AUC"].iloc[0])
                rows.append({"order": fold_number, "label": f"Fold {fold_number}", "PR_AUC": pr_auc})
        extra_rows = [
            (VALIDATION_FOLD_COUNT + 1, "CV Mean", "cv_mean_threshold_optimized"),
            (VALIDATION_FOLD_COUNT + 2, "Full Training", "final_training"),
            (VALIDATION_FOLD_COUNT + 3, "Holdout Validation", "holdout_validation_threshold_optimized"),
        ]
        for order, label, split_name in extra_rows:
            match = metrics_df.loc[metrics_df["split"].astype(str) == split_name]
            if not match.empty:
                rows.append({"order": order, "label": label, "PR_AUC": float(match["PR_AUC"].iloc[0])})
        if not rows:
            return None
        plot_df = pd.DataFrame(rows).sort_values("order").reset_index(drop=True)
        x_values = np.arange(len(plot_df))
        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        ax.plot(x_values, plot_df["PR_AUC"], marker="o", linewidth=2.4, color="#0072B2")
        for x_value, (_, row) in zip(x_values, plot_df.iterrows(), strict=False):
            ax.annotate(
                f"{row['PR_AUC']:.3f}",
                (x_value, row["PR_AUC"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        ax.set_xticks(x_values)
        ax.set_xticklabels(plot_df["label"], rotation=25, ha="right")
        ax.set_ylabel("PR-AUC")
        ax.set_xlabel("Evaluation dataset")
        ax.set_title("PR-AUC Comparison: CV Folds, CV Mean, Full Training, and Holdout Validation")
        y_min = max(0.0, float(plot_df["PR_AUC"].min()) - 0.05)
        y_max = min(1.0, float(plot_df["PR_AUC"].max()) + 0.05)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", alpha=0.25)
        save_figure(fig, output_path)
        return output_path
    except Exception as exc:
        logger.warning("Could not write PR-AUC comparison plot: %s", exc)
        plt.close()
        return None


def write_default_validation_plots(
    output_dir: Path,
    evaluation_sets: list[dict[str, Any]],
    threshold: float,
    metrics_df: pd.DataFrame,
    model: Any,
    predictor_columns: list[str],
    config: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"enabled": True, "datasets": {}}
    pr_path = write_overlay_pr_curve(output_dir, evaluation_sets, logger)
    roc_path = write_overlay_roc_curve(output_dir, evaluation_sets, logger)
    calibration_path = write_overlay_calibration_curve(output_dir, evaluation_sets, logger)
    pr_auc_path = write_pr_auc_comparison_plot(output_dir, metrics_df, logger)
    if pr_path is not None:
        summary["overlay_pr_curve"] = str(pr_path)
    if roc_path is not None:
        summary["overlay_roc_curve"] = str(roc_path)
    if calibration_path is not None:
        summary["calibration_curve"] = str(calibration_path)
    if pr_auc_path is not None:
        summary["pr_auc_comparison"] = str(pr_auc_path)

    target_column = str(config["target_column"])
    for evaluation_set in evaluation_sets:
        dataset_outputs: dict[str, Any] = {}
        confusion_path = None
        if (
            str(evaluation_set["key"]) == "holdout_validation"
            and bool(config.get("write_feature_importance", True))
        ):
            confusion_path = write_confusion_matrix_plot(output_dir, evaluation_set, threshold, logger)
        histogram_path = write_probability_histogram(output_dir, evaluation_set, threshold, logger)
        permutation_outputs = None
        if str(evaluation_set["key"]) == "holdout_validation":
            permutation_outputs = write_permutation_feature_importance(
                model,
                evaluation_set,
                predictor_columns,
                target_column,
                output_dir,
                config,
                logger,
            )
        if confusion_path is not None:
            dataset_outputs["confusion_matrix"] = str(confusion_path)
        if histogram_path is not None:
            dataset_outputs["probability_histogram"] = str(histogram_path)
        if permutation_outputs is not None:
            dataset_outputs["feature_importance"] = permutation_outputs
        summary["datasets"][str(evaluation_set["key"])] = dataset_outputs
    return summary


def save_model(model: Any, output_path: Path) -> None:
    joblib.dump(model, output_path)


def run_qc_checks(
    output_paths: dict[str, Path | None],
    predictor_columns: list[str],
    validation_predictions: pd.DataFrame | None,
    final_model: Any,
    cleaned_df: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    if bool(config["write_model"]) and (output_paths["model"] is None or not output_paths["model"].exists()):
        raise FileNotFoundError("Model file was not written.")
    for key in ("metadata", "metrics"):
        path = output_paths[key]
        if path is None or not path.exists():
            raise FileNotFoundError(f"Required output was not written: {key}")
    if not predictor_columns:
        raise ValueError("Predictor column list is empty.")
    for key in ("training_predictions", "validation_predictions"):
        path = output_paths.get(key)
        if path is None or not path.exists():
            continue
        df = pd.read_csv(path)
        if "predicted_probability" not in df.columns:
            raise ValueError(f"{path} is missing predicted_probability column.")
        if not df["predicted_probability"].between(0, 1).all():
            raise ValueError(f"Predicted probabilities outside [0, 1] in {path}.")
    if validation_predictions is not None and not validation_predictions.empty:
        required = {str(config["target_column"]), "predicted_probability", "predicted_class"}
        missing = required - set(validation_predictions.columns)
        if missing:
            raise ValueError(f"Validation predictions are missing required columns: {sorted(missing)}")
    sample = cleaned_df[predictor_columns].head(min(5, len(cleaned_df)))
    if len(sample) == 0:
        raise ValueError("No sample rows available for final model predict_proba QC.")
    probabilities = final_model.predict_proba(sample)[:, 1]
    if not np.all((probabilities >= 0) & (probabilities <= 1)):
        raise ValueError("Final model predict_proba returned probabilities outside [0, 1].")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a probabilistic ML-DFI depositional favorability classifier.")
    parser.add_argument("--config-json", help="Optional JSON config file. Uses the whole object as Step 3 config.")
    parser.add_argument("--input-sample-table")
    parser.add_argument("--output-dir")
    parser.add_argument("--input-format")
    parser.add_argument("--model-type")
    parser.add_argument("--probability-threshold")
    parser.add_argument("--fixed-threshold", type=float)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--enable-hyperparameter-tuning", action="store_true")
    parser.add_argument("--disable-hyperparameter-tuning", action="store_true")
    parser.add_argument("--tuning-trials", type=int)
    parser.add_argument("--tuning-metric")
    parser.add_argument("--disable-feature-importance", action="store_true")
    parser.add_argument("--disable-predictions", action="store_true")
    parser.add_argument("--disable-model", action="store_true")
    parser.add_argument("--disable-plots", action="store_true")
    parser.add_argument("--disable-explainability", action="store_true")
    return parser.parse_args()


def resolve_runtime_path(root: Path, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    return str(path.resolve() if path.is_absolute() else (root / path).resolve())


def resolve_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    runtime = deepcopy(CONFIG)
    if args.config_json:
        config_path = Path(args.config_json).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config JSON does not exist: {config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            loaded_config = json.load(file)
        if not isinstance(loaded_config, dict):
            raise ValueError("Config JSON must contain an object.")
        for path_key in ("input_sample_table", "output_dir"):
            if path_key in loaded_config:
                loaded_config[path_key] = resolve_runtime_path(
                    config_path.parent, loaded_config[path_key]
                )
        for key, value in loaded_config.items():
            runtime[key] = value
    overrides = {
        "input_sample_table": args.input_sample_table,
        "output_dir": args.output_dir,
        "input_format": args.input_format,
        "model_type": args.model_type,
        "probability_threshold": args.probability_threshold,
        "fixed_threshold": args.fixed_threshold,
        "random_seed": args.random_seed,
        "tuning_trials": args.tuning_trials,
        "tuning_metric": args.tuning_metric,
    }
    for key, value in overrides.items():
        if value is not None:
            runtime[key] = value
    if args.enable_hyperparameter_tuning:
        runtime["enable_hyperparameter_tuning"] = True
    if args.disable_hyperparameter_tuning:
        runtime["enable_hyperparameter_tuning"] = False
    if args.disable_feature_importance:
        runtime["write_feature_importance"] = False
    if args.disable_predictions:
        runtime["write_predictions"] = False
    if args.disable_model:
        runtime["write_model"] = False
    if args.disable_plots:
        runtime["write_plots"] = False
    if args.disable_explainability:
        runtime["write_explainability"] = False
    runtime["input_sample_table"] = resolve_runtime_path(
        Path.cwd(), runtime.get("input_sample_table")
    )
    runtime["output_dir"] = resolve_runtime_path(Path.cwd(), runtime.get("output_dir"))
    return runtime


def resolve_validation_splits(
    cleaned_df: pd.DataFrame,
    config: dict[str, Any],
    logger: logging.Logger,
) -> tuple[str, list[tuple[str, pd.DataFrame, pd.DataFrame]], pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    holdout_training_df, holdout_validation_df, holdout_summary = create_stratified_group_holdout_split(
        cleaned_df,
        config,
        logger,
    )
    logger.info(
        "Creating %s validation splits inside the 70%% training pool using group column '%s'.",
        VALIDATION_MODE,
        VALIDATION_GROUP_COLUMN,
    )
    splits = create_group_kfold_splits(holdout_training_df, config, logger)
    return VALIDATION_WORKFLOW, splits, holdout_training_df, holdout_validation_df, holdout_summary


def append_summary_metric_rows(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fold_rows = [row for row in metrics_rows if str(row.get("split", "")).startswith("fold_")]
    if not fold_rows:
        return metrics_rows
    metric_names = [
        "PR_AUC",
        "ROC_AUC",
        "F1",
        "precision",
        "recall",
        "accuracy",
        "balanced_accuracy",
        "brier_score",
        "log_loss",
        "expected_calibration_error",
    ]
    threshold_groups: dict[str, list[dict[str, Any]]] = {}
    for row in fold_rows:
        split_name = str(row.get("split", ""))
        if split_name.endswith("_threshold_0.5"):
            suffix = "threshold_0.5"
        elif split_name.endswith("_threshold_optimized"):
            suffix = "threshold_optimized"
        else:
            suffix = "all_thresholds"
        threshold_groups.setdefault(suffix, []).append(row)

    for suffix, rows in threshold_groups.items():
        mean_name = "cv_mean" if suffix == "all_thresholds" else f"cv_mean_{suffix}"
        std_name = "cv_std" if suffix == "all_thresholds" else f"cv_std_{suffix}"
        for summary_name, function in ((mean_name, np.nanmean), (std_name, np.nanstd)):
            row = {"split": summary_name}
            for metric_name in metric_names:
                row[metric_name] = float(function([item[metric_name] for item in rows]))
            for name in ("TP", "FP", "TN", "FN"):
                row[name] = np.nan
            row["threshold"] = np.nan
            metrics_rows.append(row)
    return metrics_rows


def select_summary_metric_row(metrics_df: pd.DataFrame, preferred_suffix: str) -> pd.Series:
    if metrics_df.empty:
        return pd.Series(dtype=object)
    split_values = metrics_df["split"].astype(str)
    candidates = [
        f"holdout_validation_{preferred_suffix}",
        f"cv_mean_{preferred_suffix}",
        f"validation_{preferred_suffix}",
        "cv_mean",
        "validation",
        "final_training",
    ]
    for candidate in candidates:
        if candidate in split_values.values:
            return metrics_df.loc[split_values == candidate].iloc[0]
    return metrics_df.iloc[0]


def select_independent_metric_row(metrics_df: pd.DataFrame, preferred_suffix: str) -> pd.Series:
    if metrics_df.empty:
        return pd.Series(dtype=object)
    split_values = metrics_df["split"].astype(str)
    candidates = [
        f"holdout_validation_{preferred_suffix}",
        f"cv_mean_{preferred_suffix}",
        f"validation_{preferred_suffix}",
        "holdout_validation",
        "cv_mean",
        "validation",
    ]
    for candidate in candidates:
        if candidate in split_values.values:
            return metrics_df.loc[split_values == candidate].iloc[0]
    return pd.Series(dtype=object)


def select_named_metric_row(metrics_df: pd.DataFrame, split_name: str) -> pd.Series:
    if metrics_df.empty:
        return pd.Series(dtype=object)
    split_values = metrics_df["split"].astype(str)
    if split_name in split_values.values:
        return metrics_df.loc[split_values == split_name].iloc[0]
    return pd.Series(dtype=object)


def filter_final_training_rows(
    df: pd.DataFrame,
    config: dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    if SPLIT_COLUMN not in df.columns:
        return df.reset_index(drop=True)

    working = df.copy()
    split_values = working[SPLIT_COLUMN].fillna("").astype(str).str.strip().str.lower()

    forbidden_mask = split_values.isin(NEVER_TRAIN_SPLIT_VALUES)
    forbidden_count = int(forbidden_mask.sum())
    if forbidden_count:
        logger.info(
            "Excluding %d rows from final model fitting because '%s' is one of %s.",
            forbidden_count,
            SPLIT_COLUMN,
            sorted(NEVER_TRAIN_SPLIT_VALUES),
        )
        working = working.loc[~forbidden_mask].copy()

    if working.empty:
        raise ValueError(
            "No rows remain for final model fitting after excluding explicit external/test/holdout rows."
        )
    return working.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    config = resolve_runtime_config(args)
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
                prefix=f".{official_output_dir.name}.step3-stage-",
                dir=official_output_dir.parent,
            )
        )
        output_dir = staging_dir
        logger, staged_log_path = setup_logging(output_dir)
        logger.info("Starting Step 3 ML-DFI model training.")
        logger.info("Input sample table: %s", config["input_sample_table"])
        logger.info("Official output directory: %s", official_output_dir)
        logger.info("Model type: %s", config["model_type"])

        raw_df = load_sample_table(str(config["input_sample_table"]), str(config["input_format"]))
        predictor_columns, removed_columns = identify_predictor_columns(raw_df, config, logger)
        cleaned_df, cleaning_stats = clean_training_table(raw_df, predictor_columns, config, logger)
        validate_group_column(cleaned_df)
        numeric_predictor_columns, categorical_predictor_columns = get_predictor_type_columns(
            cleaned_df,
            predictor_columns,
            config,
        )
        logger.info("Rows after cleaning: %d", len(cleaned_df))
        logger.info("Predictor columns retained: %d", len(predictor_columns))
        logger.info(
            "Predictor type split: numeric=%d, categorical=%d.",
            len(numeric_predictor_columns),
            len(categorical_predictor_columns),
        )

        validation_mode_used, validation_splits, final_training_df, holdout_validation_df, holdout_summary = resolve_validation_splits(
            cleaned_df=cleaned_df,
            config=config,
            logger=logger,
        )
        target_column = str(config["target_column"])
        final_fit_df = final_training_df
        final_model_scope = (
            "holdout_training_split_after_group_cv"
            if holdout_validation_df is not None
            else "all_cleaned_samples_after_group_cv"
        )
        logger.info(
            "Final model scope is %s; the saved model will be fit on %d rows after validation is estimated by grouped CV.",
            final_model_scope,
            len(final_fit_df),
        )
        final_fit_df = filter_final_training_rows(final_fit_df, config, logger)
        logger.info("Rows retained for final model fitting after split safeguards: %d", len(final_fit_df))

        tuning_source = "holdout_training_split" if holdout_validation_df is not None else "all_cleaned_samples"
        tuning_df = final_training_df
        config, tuning_summary = run_hyperparameter_tuning(
            config=config,
            tuning_df=tuning_df,
            predictor_columns=predictor_columns,
            output_dir=output_dir,
            logger=logger,
            tuning_source=tuning_source,
        )

        metrics_rows: list[dict[str, Any]] = []
        validation_prediction_frames: list[pd.DataFrame] = []
        threshold_source_targets: pd.Series | np.ndarray | None = None
        threshold_source_probabilities: np.ndarray | None = None
        validated_xgboost_n_estimators: list[int] = []

        if validation_splits:
            for split_name, train_df, validation_df in validation_splits:
                # This model predicts depositional favorability, not landslide initiation susceptibility
                # and not the full landslide runout footprint.
                # Positive samples are observed depositional pixels; negative samples are upper
                # source/transport-zone pixels inside mapped landslides.
                model, _ = train_model(train_df, validation_df, predictor_columns, config, logger)
                best_n_estimators = get_xgboost_best_n_estimators(model, logger)
                if best_n_estimators is not None:
                    validated_xgboost_n_estimators.append(best_n_estimators)
                    logger.info(
                        "Validation split '%s' selected XGBoost n_estimators=%d from early stopping.",
                        split_name,
                        best_n_estimators,
                    )
                elif str(config["model_type"]).lower() == "xgboost":
                    logger.warning(
                        "Validation split '%s' did not expose an XGBoost early-stopping best iteration; "
                        "the final model may use the configured n_estimators.",
                        split_name,
                    )
                validation_probabilities = predict_probabilities(model, validation_df, predictor_columns)

                if threshold_source_targets is None:
                    threshold_source_targets = validation_df[target_column]
                    threshold_source_probabilities = validation_probabilities
                else:
                    threshold_source_targets = pd.concat(
                        [pd.Series(threshold_source_targets), validation_df[target_column]], ignore_index=True
                    )
                    threshold_source_probabilities = np.concatenate([threshold_source_probabilities, validation_probabilities])

                validation_prediction_frames.append(
                    export_predictions(
                        validation_df,
                        validation_probabilities,
                        0.5,
                        None,
                        config,
                        "validation",
                        split_name,
                    )
                )

            threshold_analysis = analyze_thresholds(threshold_source_targets, threshold_source_probabilities)
            selected_threshold = choose_threshold(threshold_analysis, config)

            for frame in validation_prediction_frames:
                frame["threshold"] = selected_threshold
                frame["predicted_class"] = (frame["predicted_probability"] >= selected_threshold).astype(int)

            for frame in validation_prediction_frames:
                split_label = str(frame["fold"].iloc[0]) if "fold" in frame.columns else str(frame["split"].iloc[0])
                metrics_rows.append(
                    compute_metrics(
                        frame[target_column],
                        frame["predicted_probability"].to_numpy(dtype=float),
                        0.5,
                        f"{split_label}_threshold_0.5",
                        logger,
                    )
                )
                metrics_rows.append(
                    compute_metrics(
                        frame[target_column],
                        frame["predicted_probability"].to_numpy(dtype=float),
                        selected_threshold,
                        f"{split_label}_threshold_optimized",
                        logger,
                    )
                )
            metrics_rows = append_summary_metric_rows(metrics_rows)
        else:
            # The final ML-DFI raster should use predicted probability values, not only binary classes.
            threshold_analysis = pd.DataFrame()
            selected_threshold = 0.5 if str(config["probability_threshold"]) != "fixed" else float(config["fixed_threshold"])
            logger.info("Validation mode is train_only; validation metrics will not be computed.")

        final_xgboost_n_estimators, xgboost_early_stopping_summary = choose_final_xgboost_n_estimators(
            validated_xgboost_n_estimators,
            config,
            logger,
        )
        final_model, final_model_params = train_model(
            final_fit_df,
            None,
            predictor_columns,
            config,
            logger,
            n_estimators_override=final_xgboost_n_estimators,
        )

        final_probabilities = predict_probabilities(final_model, final_fit_df, predictor_columns)
        final_metrics = compute_metrics(
            final_fit_df[target_column],
            final_probabilities,
            selected_threshold,
            "final_training",
            logger,
        )
        metrics_rows.append(final_metrics)

        holdout_probabilities: np.ndarray | None = None
        holdout_prediction_frame: pd.DataFrame | None = None
        if holdout_validation_df is not None:
            holdout_probabilities = predict_probabilities(final_model, holdout_validation_df, predictor_columns)
            metrics_rows.append(
                compute_metrics(
                    holdout_validation_df[target_column],
                    holdout_probabilities,
                    0.5,
                    "holdout_validation_threshold_0.5",
                    logger,
                )
            )
            metrics_rows.append(
                compute_metrics(
                    holdout_validation_df[target_column],
                    holdout_probabilities,
                    selected_threshold,
                    "holdout_validation_threshold_optimized",
                    logger,
                )
            )

        evaluation_sets: list[dict[str, Any]] = [
            {
                "key": "final_training",
                "label": "Full Training Dataset",
                "data_df": final_fit_df,
                "y_true": final_fit_df[target_column],
                "probabilities": final_probabilities,
            }
        ]
        if holdout_validation_df is not None and holdout_probabilities is not None:
            evaluation_sets.append(
                {
                    "key": "holdout_validation",
                    "label": "Holdout Validation Dataset",
                    "data_df": holdout_validation_df,
                    "y_true": holdout_validation_df[target_column],
                    "probabilities": holdout_probabilities,
                }
            )

        output_paths: dict[str, Path | None] = {
            "model": output_dir / OUTPUTS["model"] if bool(config["write_model"]) else None,
            "metadata": output_dir / OUTPUTS["metadata"],
            "validation_predictions": output_dir / OUTPUTS["validation_predictions"] if bool(config["write_predictions"]) else None,
            "training_predictions": output_dir / OUTPUTS["training_predictions"] if bool(config["write_predictions"]) else None,
            "metrics": output_dir / OUTPUTS["metrics"],
            "threshold_analysis": output_dir / OUTPUTS["threshold_analysis"],
            "partial_dependence": output_dir / OUTPUTS["partial_dependence"] if bool(config["write_explainability"]) else None,
        }

        if bool(config["write_model"]):
            save_model(final_model, output_paths["model"])

        if bool(config["write_predictions"]):
            training_predictions = export_predictions(
                final_fit_df,
                final_probabilities,
                selected_threshold,
                output_paths["training_predictions"],
                config,
                "final_training",
            )
            training_predictions["prediction_context"] = "final_model_training_prediction"
            training_predictions.to_csv(output_paths["training_predictions"], index=False)

            if holdout_validation_df is not None and holdout_probabilities is not None:
                holdout_prediction_frame = export_predictions(
                    holdout_validation_df,
                    holdout_probabilities,
                    selected_threshold,
                    None,
                    config,
                    "holdout_validation",
                    "holdout_validation",
                )
                validation_prediction_frames.append(holdout_prediction_frame)

            if validation_prediction_frames:
                validation_predictions = pd.concat(validation_prediction_frames, ignore_index=True)
                validation_predictions["prediction_context"] = "out_of_fold_or_holdout_validation_prediction"
                validation_predictions.to_csv(output_paths["validation_predictions"], index=False)
            else:
                validation_predictions = None
                output_paths["validation_predictions"] = None
        else:
            training_predictions = None
            validation_predictions = None

        metrics_df = write_metrics_csv(metrics_rows, output_paths["metrics"])
        write_threshold_analysis_csv(threshold_analysis, output_paths["threshold_analysis"])

        if bool(config["write_explainability"]):
            shap_dataset_summaries: dict[str, Any] = {}
            partial_dependence_dataset_summaries: dict[str, Any] = {}
            partial_dependence_frames: list[pd.DataFrame] = []
            holdout_explainability_set = next(
                (item for item in evaluation_sets if str(item["key"]) == "holdout_validation"),
                evaluation_sets[0],
            )
            holdout_key = str(holdout_explainability_set["key"])
            shap_output_path = output_dir / f"{holdout_key}_shap_feature_importance.csv"
            shap_dataset_summaries[holdout_key] = export_xgboost_shap_analysis(
                final_model,
                holdout_explainability_set["data_df"],
                predictor_columns,
                shap_output_path,
                output_dir,
                config,
                holdout_key,
                str(holdout_explainability_set["label"]),
                logger,
            )
            for evaluation_set in evaluation_sets:
                dataset_key = str(evaluation_set["key"])
                dataset_label = str(evaluation_set["label"])
                partial_dependence_output_path = output_dir / f"{dataset_key}_partial_dependence.csv"
                (
                    partial_dependence_dataset_summaries[dataset_key],
                    partial_dependence_frame,
                ) = export_partial_dependence_analysis(
                    final_model,
                    evaluation_set["data_df"],
                    predictor_columns,
                    numeric_predictor_columns,
                    categorical_predictor_columns,
                    partial_dependence_output_path,
                    output_dir,
                    config,
                    dataset_key,
                    dataset_label,
                    logger,
                    write_plot=False,
                )
                if (
                    bool(partial_dependence_dataset_summaries[dataset_key].get("enabled"))
                    and not partial_dependence_frame.empty
                ):
                    partial_dependence_frames.append(partial_dependence_frame)
            combined_partial_dependence_path = output_paths["partial_dependence"]
            combined_partial_dependence_plot_path = output_dir / "partial_dependence_plots.png"
            if partial_dependence_frames and combined_partial_dependence_path is not None:
                combined_partial_dependence_df = pd.concat(partial_dependence_frames, ignore_index=True)
                combined_partial_dependence_df.to_csv(combined_partial_dependence_path, index=False)
                combined_plot_path = write_combined_partial_dependence_plot(
                    partial_dependence_frames,
                    predictor_columns,
                    combined_partial_dependence_plot_path,
                    logger,
                )
            else:
                combined_plot_path = None
            shap_summary = {
                "enabled": any(bool(item.get("enabled")) for item in shap_dataset_summaries.values()),
                "primary_dataset": holdout_key,
                "datasets": shap_dataset_summaries,
            }
            partial_dependence_summary = {
                "enabled": any(bool(item.get("enabled")) for item in partial_dependence_dataset_summaries.values()),
                "data_path": str(combined_partial_dependence_path) if combined_partial_dependence_path is not None else None,
                "plot_path": str(combined_plot_path) if combined_plot_path is not None else None,
                "datasets": partial_dependence_dataset_summaries,
            }
        else:
            shap_summary = {"enabled": False, "reason": "disabled"}
            partial_dependence_summary = {"enabled": False, "reason": "disabled"}

        if bool(config["write_plots"]):
            validation_plot_summary = write_default_validation_plots(
                output_dir,
                evaluation_sets,
                selected_threshold,
                metrics_df,
                final_model,
                predictor_columns,
                config,
                logger,
            )
        else:
            validation_plot_summary = {"enabled": False, "reason": "disabled"}

        validation_count = int(sum(len(frame) for frame in validation_prediction_frames))
        cv_validation_count = int(
            sum(
                len(frame)
                for frame in validation_prediction_frames
                if "split" in frame.columns and str(frame["split"].iloc[0]) == "validation"
            )
        )
        holdout_validation_count = int(len(holdout_validation_df)) if holdout_validation_df is not None else 0
        training_count = int(len(final_fit_df))
        class_counts = final_fit_df[target_column].value_counts().sort_index().to_dict()
        independent_fixed_threshold_row = select_independent_metric_row(metrics_df, "threshold_0.5")
        independent_optimized_threshold_row = select_independent_metric_row(metrics_df, "threshold_optimized")
        final_training_row = select_named_metric_row(metrics_df, "final_training")
        input_sample_path = Path(str(config["input_sample_table"])).resolve()
        model_sha256 = (
            sha256_file(output_paths["model"])
            if output_paths["model"] is not None and output_paths["model"].exists()
            else None
        )
        metadata = {
            "workflow": "step3_train_ml_dfi_model",
            "model_type": config["model_type"],
            "input_sample_table": str(input_sample_path),
            "input_sample_table_sha256": sha256_file(input_sample_path),
            "model_sha256": model_sha256,
            "resolved_config": config,
            "artifact_files": sorted(
                path.name for path in output_dir.iterdir() if path.is_file()
            ),
            "target_column": target_column,
            "predictor_columns": predictor_columns,
            "numeric_predictor_columns": numeric_predictor_columns,
            "categorical_predictor_columns": categorical_predictor_columns,
            "categorical_levels": get_categorical_levels(
                final_model,
                categorical_predictor_columns,
            ),
            "removed_columns": removed_columns,
            "cleaning_stats": cleaning_stats,
            "training_sample_count": training_count,
            "validation_sample_count": validation_count,
            "cv_validation_prediction_count": cv_validation_count,
            "holdout_validation_sample_count": holdout_validation_count,
            "positive_count": int(class_counts.get(1, 0)),
            "negative_count": int(class_counts.get(0, 0)),
            "class_imbalance_strategy": config["class_imbalance_strategy"],
            "missing_predictor_strategy": config.get("missing_predictor_strategy", "xgboost_native_else_median_impute"),
            "final_model_scope": final_model_scope,
            "selected_probability_threshold": selected_threshold,
            "random_seed": int(config["random_seed"]),
            "model_parameters": final_model_params,
            "hyperparameter_tuning": tuning_summary,
            "xgboost_early_stopping": xgboost_early_stopping_summary,
            "explainability": {
                "shap_feature_importance": shap_summary,
                "partial_dependence": partial_dependence_summary,
            },
            "validation_plots": validation_plot_summary,
            "validation_workflow": VALIDATION_WORKFLOW,
            "holdout_validation": holdout_summary,
            "validation_mode": VALIDATION_MODE,
            "validation_mode_used": validation_mode_used,
            "validation_group_column": VALIDATION_GROUP_COLUMN,
            "validation_fold_count": len(validation_splits),
            "reporting_guidance": {
                "independent_performance_priority_rows": [
                    "holdout_validation_threshold_0.5",
                    "holdout_validation_threshold_optimized",
                    "cv_mean_threshold_0.5",
                    "cv_mean_threshold_optimized",
                    "validation_threshold_0.5",
                    "validation_threshold_optimized",
                ],
                "diagnostic_only_rows": ["final_training"],
                "note": (
                    "When holdout validation is enabled, report final independent performance from holdout_validation rows. "
                    "Use CV rows to describe model selection/tuning stability. Treat final_training metrics as model-fit "
                    "diagnostics only because they are computed on the same rows used to fit the final saved model."
                ),
            },
            "date_time_created": datetime.now().isoformat(timespec="seconds"),
            "scientific_framing": (
                "Predicted probabilities are ML-derived Depositional Favorability Index values. "
                "They do not represent landslide initiation susceptibility or full runout footprints."
            ),
        }
        metadata = make_artifact_paths_relative(metadata, output_dir)
        write_metadata_json(metadata, output_paths["metadata"])

        run_qc_checks(
            output_paths=output_paths,
            predictor_columns=predictor_columns,
            validation_predictions=validation_predictions,
            final_model=final_model,
            cleaned_df=final_fit_df,
            config=config,
        )
        logger.info(
            "Independent performance should be reported from holdout_validation rows in model_metrics.csv. "
            "CV rows describe tuning/model-selection stability; the final_training row is diagnostic only."
        )

        fixed_threshold_summary_row = independent_fixed_threshold_row
        optimized_threshold_summary_row = independent_optimized_threshold_row
        logger.info("Finished Step 3 successfully.")
        close_file_log_handlers(logger)
        staged_paths = {
            path.name: path
            for path in output_dir.iterdir()
            if path.is_file()
        }
        published_paths = publish_output_bundle(
            staged_paths,
            official_output_dir,
            MANAGED_OUTPUT_NAMES,
        )
        shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir = None
        output_paths = {
            key: (
                published_paths.get(path.name)
                if path is not None
                else None
            )
            for key, path in output_paths.items()
        }
        log_path = published_paths[OUTPUTS["log"]]
        print("Step 3 ML-DFI model training completed successfully.")
        print(f"Model type: {config['model_type']}")
        print(f"Input sample table: {Path(str(config['input_sample_table'])).resolve()}")
        print(f"Output model path: {output_paths['model']}")
        print(f"Number of predictors: {len(predictor_columns)}")
        print(f"Training sample count: {training_count}")
        print(f"Validation CV folds: {len(validation_splits)}")
        if holdout_validation_count:
            print(f"Holdout validation sample count: {holdout_validation_count}")
        else:
            print(f"Validation sample count: {validation_count}")
        if not optimized_threshold_summary_row.empty:
            print(f"Independent validation PR-AUC: {optimized_threshold_summary_row.get('PR_AUC', np.nan)}")
            print(f"Independent validation ROC-AUC: {optimized_threshold_summary_row.get('ROC_AUC', np.nan)}")
            print(f"Independent validation Brier score: {optimized_threshold_summary_row.get('brier_score', np.nan)}")
            print(f"Independent validation F1 at threshold 0.5: {fixed_threshold_summary_row.get('F1', np.nan)}")
            print(f"Independent validation F1 at optimized threshold: {optimized_threshold_summary_row.get('F1', np.nan)}")
        else:
            print("Independent validation metrics: not available")
        print(f"Diagnostic final-training PR-AUC: {final_training_row.get('PR_AUC', np.nan)}")
        print(f"Diagnostic final-training ROC-AUC: {final_training_row.get('ROC_AUC', np.nan)}")
        print(f"Selected threshold: {selected_threshold}")
        print(f"Metadata path: {output_paths['metadata']}")
        print(f"Metrics path: {output_paths['metrics']}")
        print(f"Processing log path: {log_path}")
    except Exception as exc:
        if logger is not None:
            logger.exception("Step 3 ML-DFI model training failed: %s", exc)
            close_file_log_handlers(logger)
            if (
                staged_log_path is not None
                and staged_log_path.exists()
                and official_output_dir is not None
            ):
                official_output_dir.mkdir(parents=True, exist_ok=True)
                failed_log = official_output_dir / "processing_log.failed.txt"
                failed_log_tmp = official_output_dir / ".processing_log.failed.tmp"
                shutil.copy2(staged_log_path, failed_log_tmp)
                os.replace(failed_log_tmp, failed_log)
        else:
            print(f"Step 3 ML-DFI model training failed: {exc}", file=sys.stderr)
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
