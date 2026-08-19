from __future__ import annotations

import json
from pathlib import Path

import pytest

import workflow_config as workflow


def write_local_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "local_config.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_deep_merge_does_not_mutate_shared_defaults():
    shared = {"nested": {"kept": 1, "changed": 2}, "items": [1, 2]}
    local = {"nested": {"changed": 3}, "items": [4]}

    merged = workflow.deep_merge(shared, local)

    assert merged == {"nested": {"kept": 1, "changed": 3}, "items": [4]}
    assert shared == {"nested": {"kept": 1, "changed": 2}, "items": [1, 2]}
    assert local == {"nested": {"changed": 3}, "items": [4]}


def test_replace_marker_removes_stale_mapping_entries():
    shared = {
        "predictors": {
            "slope": "shared-slope",
            "stale_predictor": "shared-stale",
        }
    }
    local = {
        "predictors": {
            "$replace": True,
            "slope": "local-slope",
        }
    }

    assert workflow.deep_merge(shared, local) == {
        "predictors": {
            "slope": "local-slope",
        }
    }


def test_local_paths_and_worker_overrides_are_applied_in_memory(tmp_path):
    local_path = write_local_config(
        tmp_path,
        {
            "schema_version": 1,
            "runtime": {
                "workers": {
                    "step3_train_ml_dfi_model": 3,
                }
            },
            "step_overrides": {
                "step0_baseline_alpha_angle": {
                    "slope": "study/slope.tif",
                    "output_alpha": "results/alpha.tif",
                }
            },
        },
    )
    shared_text_before = workflow.DEFAULT_MASTER_CONFIG.read_text(encoding="utf-8")

    loaded = workflow.load_workflow_config(
        workflow.DEFAULT_MASTER_CONFIG,
        local_path,
        use_auto_local=False,
    )
    step0, _ = workflow.effective_step_config(
        loaded,
        "step0_baseline_alpha_angle",
    )
    step3, _ = workflow.effective_step_config(
        loaded,
        "step3_train_ml_dfi_model",
    )

    assert step0["slope"] == str((tmp_path / "study/slope.tif").resolve())
    assert step0["output_alpha"] == str((tmp_path / "results/alpha.tif").resolve())
    assert step3["xgboost_params"]["n_jobs"] == 3
    assert step3["random_forest_params"]["n_jobs"] == 3
    assert workflow.DEFAULT_MASTER_CONFIG.read_text(encoding="utf-8") == shared_text_before


def test_runtime_paths_and_commands_are_exported_safely(tmp_path):
    local_path = write_local_config(
        tmp_path,
        {
            "schema_version": 1,
            "runtime": {
                "python_executable": "portable-python",
                "osgeo4w_root": "runtime/osgeo",
                "mpiexec": "portable-mpiexec",
                "taudem_dll_dir": "runtime/taudem",
                "taudem_proj_dir": "runtime/proj",
                "step4_executable": "runtime/step4.exe",
                "workers": {
                    "step4_deposition_zone": 6,
                },
            },
            "step_overrides": {},
        },
    )
    loaded = workflow.load_workflow_config(
        workflow.DEFAULT_MASTER_CONFIG,
        local_path,
        use_auto_local=False,
    )
    effective_path = tmp_path / "step4.json"

    command, environment = workflow.build_step_invocation(
        loaded,
        "step4",
        effective_path,
        ["--dry-run"],
    )

    assert environment["MLDFI_PYTHON"] == "portable-python"
    assert environment["MPIEXEC"] == "portable-mpiexec"
    assert environment["OSGEO4W_ROOT"] == str((tmp_path / "runtime/osgeo").resolve())
    assert environment["TAUDEM_DLL_DIR"] == str((tmp_path / "runtime/taudem").resolve())
    assert environment["TAUDEM_PROJ_DIR"] == str((tmp_path / "runtime/proj").resolve())
    assert "--processes 6" in command
    assert "--exe" in command
    assert "--dry-run" in command


def test_master_runner_rejects_per_step_config_bypass():
    loaded = workflow.load_workflow_config(
        workflow.DEFAULT_MASTER_CONFIG,
        use_auto_local=False,
    )

    with pytest.raises(ValueError, match="Do not pass a per-step config"):
        workflow.build_step_invocation(
            loaded,
            "step0",
            Path("effective.json"),
            ["--config", "other.json"],
        )


def test_unknown_or_misspelled_local_runtime_setting_fails(tmp_path):
    local_path = write_local_config(
        tmp_path,
        {
            "schema_version": 1,
            "runtime": {
                "worker_count": 8,
            },
            "step_overrides": {},
        },
    )

    with pytest.raises(ValueError, match="Unsupported runtime settings"):
        workflow.load_workflow_config(
            workflow.DEFAULT_MASTER_CONFIG,
            local_path,
            use_auto_local=False,
        )


def test_every_registered_step_builds_an_effective_config():
    loaded = workflow.load_workflow_config(
        workflow.DEFAULT_MASTER_CONFIG,
        use_auto_local=False,
    )

    for step in workflow.STEP_SPECS:
        effective, base_path = workflow.effective_step_config(loaded, step)
        assert effective
        assert base_path.is_file()


def test_model_application_uses_centralized_step2_predictor_config(tmp_path):
    local_path = write_local_config(
        tmp_path,
        {
            "schema_version": 1,
            "runtime": {},
            "step_overrides": {
                "step2_extract_pixel_samples": {
                    "extract_pixel_samples": {
                        "predictor_rasters": {
                            "$replace": True,
                            "slope": "inputs/slope.tif",
                        }
                    }
                }
            },
        },
    )
    loaded = workflow.load_workflow_config(
        workflow.DEFAULT_MASTER_CONFIG,
        local_path,
        use_auto_local=False,
    )

    application_path, _ = workflow.write_effective_config(
        loaded,
        "step3_model_application",
    )
    application = json.loads(application_path.read_text(encoding="utf-8"))
    predictor_config_path = Path(
        application["apply_ml_dfi_model"]["predictor_config_json"]
    )
    predictors = json.loads(predictor_config_path.read_text(encoding="utf-8"))

    assert predictors["extract_pixel_samples"]["predictor_rasters"] == {
        "slope": str((tmp_path / "inputs/slope.tif").resolve())
    }
