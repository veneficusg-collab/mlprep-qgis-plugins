import importlib.util
import json
import logging
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = (
    ROOT / "steps" / "step3_train_ml_dfi_model" / "step3_train_ml_dfi_model.py"
)
APPLICATION_SCRIPT = (
    ROOT
    / "steps"
    / "step3_train_ml_dfi_model"
    / "apply_ml_dfi_model_to_rasters.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAINING = _load_module("step3_training_test_module", TRAINING_SCRIPT)
APPLICATION = _load_module("step3_application_test_module", APPLICATION_SCRIPT)


class _CategoricalRasterModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        assert all(isinstance(value, str) for value in frame["geomorphology"])
        positive = np.where(frame["geomorphology"] == "1.0", 0.8, 0.2)
        return np.column_stack([1.0 - positive, positive])


def _write_test_raster(path: Path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:32651",
        transform=from_origin(0, 10, 5, 5),
        nodata=-9999.0,
    ) as dst:
        dst.write(values.astype(np.float32), 1)


def test_raster_categorical_values_match_training_string_contract() -> None:
    frame = pd.DataFrame(
        {
            "slope": np.array([12.5, 22.0], dtype=np.float32),
            "geomorphology": np.array([1.0, 2.0], dtype=np.float32),
        }
    )

    prepared = APPLICATION.prepare_predictor_frame(
        frame,
        ["geomorphology"],
        {"geomorphology": {"1.0", "2.0"}},
    )

    assert prepared["geomorphology"].tolist() == ["1.0", "2.0"]
    assert frame["geomorphology"].dtype == np.float32


def test_unseen_raster_category_fails_clearly() -> None:
    frame = pd.DataFrame({"geomorphology": [1.0, 7.0]})

    with pytest.raises(ValueError, match="not seen during training"):
        APPLICATION.prepare_predictor_frame(
            frame,
            ["geomorphology"],
            {"geomorphology": {"1.0", "2.0"}},
        )


def test_windowed_raster_prediction_applies_categorical_contract(tmp_path: Path) -> None:
    slope_path = tmp_path / "slope.tif"
    geomorphology_path = tmp_path / "geomorphology.tif"
    output_path = tmp_path / "probability.tif"
    _write_test_raster(slope_path, np.array([[10, 20], [30, 40]], dtype=np.float32))
    _write_test_raster(
        geomorphology_path,
        np.array([[1, 2], [2, 1]], dtype=np.float32),
    )
    with rasterio.open(slope_path) as reference:
        reference_info = {
            "width": reference.width,
            "height": reference.height,
            "transform": reference.transform,
            "crs": reference.crs,
            "profile": reference.profile.copy(),
        }

    stats, sample = APPLICATION.predict_raster(
        model=_CategoricalRasterModel(),
        predictor_paths={
            "slope": slope_path,
            "geomorphology": geomorphology_path,
        },
        predictor_columns=["slope", "geomorphology"],
        reference=reference_info,
        output_raster_path=output_path,
        nodata_value=-9999.0,
        window_height=1,
        compress="lzw",
        bigtiff="IF_SAFER",
        full_domain_shap_sample_size=0,
        categorical_columns=["geomorphology"],
        categorical_levels={"geomorphology": {"1.0", "2.0"}},
        logger=logging.getLogger("step3-raster-category-test"),
    )

    with rasterio.open(output_path) as result:
        values = result.read(1)
    np.testing.assert_allclose(values, np.array([[0.8, 0.2], [0.2, 0.8]]))
    assert stats["predicted_pixels"] == 4
    assert sample is None


@pytest.mark.parametrize(
    "groups",
    [
        [1, np.nan],
        [1, 0],
        [1, -2],
        [1, 2.5],
        [1, np.inf],
    ],
)
def test_invalid_landslide_groups_fail_before_splitting(groups) -> None:
    frame = pd.DataFrame({"landslide_id": groups, "target": [0, 1]})

    with pytest.raises(ValueError, match="positive integer IDs"):
        TRAINING.validate_group_column(frame)


def test_unknown_training_config_key_fails(tmp_path: Path) -> None:
    sample_path = tmp_path / "samples.csv"
    sample_path.write_text("target,landslide_id,slope\n0,1,10\n1,2,20\n", encoding="utf-8")
    config = deepcopy(TRAINING.CONFIG)
    config["input_sample_table"] = str(sample_path)
    config["output_dir"] = str(tmp_path / "outputs")
    config["unexpected_typo"] = True

    with pytest.raises(ValueError, match="Unknown Step 3 config keys"):
        TRAINING.validate_config(config)


def test_application_output_name_cannot_escape_bundle(tmp_path: Path) -> None:
    config = dict(APPLICATION.CONFIG)
    for key in (
        "model_path",
        "metadata_path",
        "predictor_config_json",
        "reference_raster_path",
    ):
        path = tmp_path / f"{key}.dat"
        path.write_text("placeholder", encoding="utf-8")
        config[key] = str(path)
    config["output_dir"] = str(tmp_path / "outputs")
    config["output_probability_raster"] = "../escaped.tif"

    with pytest.raises(ValueError, match="must be a filename"):
        APPLICATION.validate_config(config)


def test_domain_shap_sampler_is_bounded_and_deterministic() -> None:
    count = 300
    frame = pd.DataFrame({"slope": np.arange(count, dtype=np.float32)})
    probabilities = np.concatenate(
        [
            np.full(100, 0.1),
            np.full(100, 0.5),
            np.full(100, 0.9),
        ]
    )
    rows = np.arange(count)
    cols = np.arange(count)[::-1]

    first = APPLICATION.update_domain_shap_sample(
        None,
        frame,
        probabilities,
        rows,
        cols,
        group_sample_size=20,
    ).sort_values(["dfi_probability_group", "__sample_key"]).reset_index(drop=True)
    second = APPLICATION.update_domain_shap_sample(
        None,
        frame,
        probabilities,
        rows,
        cols,
        group_sample_size=20,
    ).sort_values(["dfi_probability_group", "__sample_key"]).reset_index(drop=True)

    assert first.groupby("dfi_probability_group").size().to_dict() == {
        "high_dfi": 20,
        "low_dfi": 20,
        "medium_dfi": 20,
    }
    pd.testing.assert_frame_equal(first, second)


def test_training_publication_failure_restores_previous_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "official"
    staging_dir.mkdir()
    output_dir.mkdir()
    staged_paths = {
        "first": staging_dir / "first.txt",
        "second": staging_dir / "second.txt",
    }
    staged_paths["first"].write_text("new-first", encoding="utf-8")
    staged_paths["second"].write_text("new-second", encoding="utf-8")
    (output_dir / "first.txt").write_text("old-first", encoding="utf-8")
    (output_dir / "second.txt").write_text("old-second", encoding="utf-8")
    real_replace = TRAINING.os.replace

    def fail_on_second(source, destination) -> None:
        if Path(source) == staged_paths["second"]:
            raise OSError("simulated publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(TRAINING.os, "replace", fail_on_second)

    with pytest.raises(OSError, match="simulated publication failure"):
        TRAINING.publish_output_bundle(
            staged_paths,
            output_dir,
            {"first.txt", "second.txt"},
        )

    assert (output_dir / "first.txt").read_text(encoding="utf-8") == "old-first"
    assert (output_dir / "second.txt").read_text(encoding="utf-8") == "old-second"
    assert not list(tmp_path.glob(".step3-backup-*"))


def test_expected_calibration_error_is_zero_for_perfect_bins() -> None:
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])

    assert TRAINING.expected_calibration_error(targets, probabilities) == pytest.approx(0.0)


def test_prepare_tuning_fold_cache_reuses_fold_local_preprocessing() -> None:
    train = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "slope": [10.0, 20.0, 30.0, 40.0],
            "geomorphology": ["1.0", "2.0", "1.0", "2.0"],
        }
    )
    validation = pd.DataFrame(
        {
            "target": [0, 1],
            "slope": [15.0, 35.0],
            "geomorphology": ["1.0", "2.0"],
        }
    )
    config = deepcopy(TRAINING.CONFIG)
    logger = logging.getLogger("step3-cache-test")

    cached = TRAINING.prepare_tuning_fold_cache(
        [("fold_1", train, validation)],
        ["slope", "geomorphology"],
        config,
        logger,
    )

    assert len(cached) == 1
    assert cached[0]["x_train"].shape == (4, 3)
    assert cached[0]["x_validation"].shape == (2, 3)


def test_cached_xgboost_fold_fit_is_runnable() -> None:
    pytest.importorskip("xgboost")
    cached_fold = {
        "name": "fold_1",
        "x_train": np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
        "y_train": np.array([0, 0, 1, 1], dtype=np.int8),
        "x_validation": np.array([[0.5], [2.5]], dtype=np.float32),
        "y_validation": np.array([0, 1], dtype=np.int8),
    }
    estimator, probabilities = TRAINING.fit_cached_xgboost_fold(
        cached_fold,
        {
            "n_estimators": 20,
            "max_depth": 2,
            "learning_rate": 0.1,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "early_stopping_rounds": 5,
            "n_jobs": 1,
        },
        random_seed=42,
    )

    assert estimator is not None
    assert probabilities.shape == (2,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_small_training_run_publishes_one_clean_bundle(tmp_path: Path) -> None:
    rows = []
    sample_id = 1
    for group_id in range(1, 31):
        for target in (0, 1):
            rows.append(
                {
                    "sample_id": sample_id,
                    "landslide_id": group_id,
                    "target": target,
                    "slope": float(group_id + target),
                }
            )
            sample_id += 1
    sample_path = tmp_path / "samples.csv"
    pd.DataFrame(rows).to_csv(sample_path, index=False)
    output_dir = tmp_path / "model_training_validation"
    output_dir.mkdir()
    stale_duplicate = output_dir / "final_training_predictions.csv"
    stale_duplicate.write_text("stale\n", encoding="utf-8")

    config = deepcopy(TRAINING.CONFIG)
    config.update(
        {
            "input_sample_table": str(sample_path),
            "output_dir": str(output_dir),
            "id_columns": ["sample_id"],
            "group_columns": ["landslide_id"],
            "drop_columns": [],
            "categorical_columns": [],
            "auto_detect_categorical": False,
            "model_type": "random_forest",
            "enable_hyperparameter_tuning": False,
            "random_forest_params": {
                "n_estimators": 10,
                "max_depth": 3,
                "min_samples_leaf": 1,
                "class_weight": "balanced_subsample",
                "n_jobs": 1,
            },
            "write_feature_importance": False,
            "write_predictions": True,
            "write_model": True,
            "write_plots": False,
            "write_explainability": False,
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(TRAINING_SCRIPT),
            "--config-json",
            str(config_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_dir / "trained_ml_dfi_model.pkl").exists()
    assert (output_dir / "model_metadata.json").exists()
    assert (output_dir / "model_metrics.csv").exists()
    assert (output_dir / "training_predictions.csv").exists()
    assert (output_dir / "validation_predictions.csv").exists()
    assert not stale_duplicate.exists()
    assert not list(tmp_path.glob(".model_training_validation.step3-stage-*"))
