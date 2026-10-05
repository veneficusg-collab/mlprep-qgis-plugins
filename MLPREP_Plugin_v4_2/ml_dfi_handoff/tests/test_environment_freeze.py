import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_conda_environment_uses_exact_python_and_pip_lock():
    payload = yaml.safe_load((ROOT / "environment.yml").read_text(encoding="utf-8"))
    dependencies = payload["dependencies"]

    assert "python=3.13.12" in dependencies
    assert "pip=25.3" in dependencies
    pip_section = next(item["pip"] for item in dependencies if isinstance(item, dict))
    assert "-r requirements.lock.txt" in pip_section


def test_important_standard_packages_are_exactly_pinned():
    expected = {
        "numpy": "2.4.2",
        "pandas": "3.0.1",
        "rasterio": "1.5.0",
        "geopandas": "1.1.3",
        "shapely": "2.1.2",
        "scikit-learn": "1.8.0",
        "xgboost": "3.2.0",
        "optuna": "4.8.0",
        "joblib": "1.5.3",
    }
    pins = {}
    for line in (ROOT / "requirements.lock.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.lower()] = version

    for name, version in expected.items():
        assert pins[name] == version


def test_native_lock_records_qgis_gdal_proj_and_python_builds():
    lock = (ROOT / "environment.osgeo4w.lock.txt").read_text(encoding="utf-8")

    assert "qgis-ltr=3.40.7-1" in lock
    assert "gdal=3.10.3-2" in lock
    assert "proj=9.6.0-2" in lock
    assert "python3-core=3.12.10-1" in lock


def test_dp1_overlay_is_exactly_locked():
    lock = (
        ROOT / "steps" / "data_preparation" / "requirements.lock.txt"
    ).read_text(encoding="utf-8")

    assert "numpy==2.5.1" in lock
    assert "shapely==2.1.2" in lock
    assert "rasterio==1.5.0" in lock


def test_dp1_launcher_uses_qgis_python_directly():
    launcher = (
        ROOT / "steps" / "data_preparation" / "launch_dp1.py"
    ).read_text(encoding="utf-8")

    assert "QGIS_PREFIX_PATH" in launcher
    assert ".venv-dp1" in launcher


def test_dp1_moved_layout_keeps_workspace_relative_paths():
    step_root = ROOT / "steps" / "data_preparation"
    payload = json.loads(
        (step_root / "config.default.json").read_text(encoding="utf-8")
    )["dp1"]

    assert (step_root / payload["landslide_polygons"]).resolve() == (
        ROOT
        / "Sample Data"
        / "Input"
        / "Shapefile"
        / "PRSTM"
        / "Makilala_LI_PRSTM51N.shp"
    ).resolve()
    assert (step_root / payload["dem"]).resolve() == (
        ROOT
        / "Sample Data"
        / "Input"
        / "Raster"
        / "Utils"
        / "PRSTM"
        / "Makilala_DEM_Buff_PRSTM51N.tif"
    ).resolve()
    assert "STEP_ROOT.parents[1]" in (
        step_root / "launch_dp1.py"
    ).read_text(encoding="utf-8")
    assert 'step_root.parents[1] / ".venv-dp1"' in (
        step_root / "run_dp1.py"
    ).read_text(encoding="utf-8")


def test_standard_setup_enforces_frozen_python_and_pip():
    setup = (ROOT / "setup_environment.py").read_text(encoding="utf-8")

    assert 'REQUIRED_PYTHON = "3.13.12"' in setup
    assert 'REQUIRED_PIP = "25.3"' in setup
