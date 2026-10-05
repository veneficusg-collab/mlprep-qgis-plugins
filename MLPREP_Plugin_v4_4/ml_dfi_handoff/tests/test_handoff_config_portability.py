from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    script = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_step0_config_paths_are_relative_to_config(tmp_path: Path) -> None:
    module = load_script(
        "portable_step0",
        "steps/step0_baseline_alpha_angle/step0_baseline_alpha_angle.py",
    )
    config_dir = tmp_path / "study"
    config_dir.mkdir()
    config_path = config_dir / "step0.json"
    config_path.write_text(
        json.dumps(
            {
                "slope": "inputs/slope.tif",
                "output_alpha": "outputs/alpha.tif",
            }
        ),
        encoding="utf-8",
    )

    config = module.load_step0_config(config_path)

    assert config["slope"] == (config_dir / "inputs/slope.tif").resolve()
    assert config["output_alpha"] == (config_dir / "outputs/alpha.tif").resolve()


def test_step2_nested_paths_are_relative_to_config_root(tmp_path: Path) -> None:
    module = load_script(
        "portable_step2",
        "steps/step2_extract_pixel_samples/step2_extract_pixel_samples.py",
    )
    config = {
        "label_raster_path": "inputs/label.tif",
        "predictor_rasters": {"slope": "inputs/slope.tif"},
        "group_rasters": {"landslide_id": "inputs/id.tif"},
        "output_dir": "outputs/step2",
    }

    resolved = module.resolve_step2_config_paths(config, tmp_path)

    assert resolved["label_raster_path"] == str((tmp_path / "inputs/label.tif").resolve())
    assert resolved["predictor_rasters"]["slope"] == str(
        (tmp_path / "inputs/slope.tif").resolve()
    )
    assert resolved["group_rasters"]["landslide_id"] == str(
        (tmp_path / "inputs/id.tif").resolve()
    )
    assert resolved["output_dir"] == str((tmp_path / "outputs/step2").resolve())


def test_model_application_paths_are_relative_to_config(tmp_path: Path) -> None:
    module = load_script(
        "portable_model_application",
        "steps/step3_train_ml_dfi_model/apply_ml_dfi_model_to_rasters.py",
    )
    config_dir = tmp_path / "application"
    config_dir.mkdir()
    config_path = config_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "apply_ml_dfi_model": {
                    "model_path": "model/model.pkl",
                    "metadata_path": "model/metadata.json",
                    "predictor_config_json": "predictors/config.json",
                    "reference_raster_path": "inputs/reference.tif",
                    "output_dir": "outputs",
                }
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config_json=config_path,
        model_path=None,
        metadata_path=None,
        predictor_config_json=None,
        reference_raster_path=None,
        output_dir=None,
        output_probability_raster=None,
        window_height=None,
        full_domain_shap_sample_size=None,
        disable_full_domain_shap=False,
    )

    config = module.resolve_config(args)

    assert config["model_path"] == str((config_dir / "model/model.pkl").resolve())
    assert config["metadata_path"] == str((config_dir / "model/metadata.json").resolve())
    assert config["predictor_config_json"] == str(
        (config_dir / "predictors/config.json").resolve()
    )
    assert config["reference_raster_path"] == str(
        (config_dir / "inputs/reference.tif").resolve()
    )
    assert config["output_dir"] == str((config_dir / "outputs").resolve())
