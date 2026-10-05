import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT
    / "steps"
    / "step1_prepare_inventory_labels"
    / "step1_prepare_inventory_labels.py"
)
LOCAL_RUNTIME_PYTHON = ROOT / ".venv-local" / "Scripts" / "python.exe"
RUNTIME_PYTHON = LOCAL_RUNTIME_PYTHON if LOCAL_RUNTIME_PYTHON.exists() else Path(sys.executable)
SPEC = importlib.util.spec_from_file_location("step1_prepare_inventory_labels", SCRIPT_PATH)
STEP1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(STEP1)


def _write_dem(path: Path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:32651",
        transform=from_origin(0, values.shape[0], 1, 1),
        nodata=-9999,
    ) as dst:
        dst.write(values.astype(np.float32), 1)


def _write_inventory(path: Path, geometry=None) -> None:
    if geometry is None:
        geometry = box(0, 0, 10, 10)
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "name": "inventory",
                "crs": {"type": "name", "properties": {"name": "EPSG:32651"}},
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"landslide_id": "LS_1"},
                        "geometry": mapping(geometry),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_config(
    path: Path,
    output_dir: str = "outputs",
    *,
    parallel_processing: bool = True,
    max_workers: int = 4,
) -> None:
    path.write_text(
        json.dumps(
            {
                "landslide_polygon_path": "inventory.geojson",
                "dem_path": "dem.tif",
                "output_dir": output_dir,
                "landslide_id_field": "landslide_id",
                "depositional_percentile": 15,
                "non_depositional_percentile": 30,
                "nodata_value": -9999,
                "output_pixel_type": "int16",
                "min_valid_dem_cells": 5,
                "all_touched": False,
                "parallel_processing": parallel_processing,
                "max_workers": max_workers,
            }
        ),
        encoding="utf-8",
    )


def _write_multi_inventory(path: Path) -> None:
    geometries = [
        box(0, 10, 12, 20),
        box(8, 10, 20, 20),
        box(0, 0, 10, 10),
        box(10, 0, 20, 10),
    ]
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "name": "inventory",
                "crs": {"type": "name", "properties": {"name": "EPSG:32651"}},
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"landslide_id": f"LS_{index}"},
                        "geometry": mapping(geometry),
                    }
                    for index, geometry in enumerate(geometries, start=1)
                ],
            }
        ),
        encoding="utf-8",
    )


def test_json_paths_are_resolved_relative_to_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    loaded = STEP1.load_json_config(config_path)

    assert loaded["dem_path"] == str((tmp_path / "dem.tif").resolve())
    assert loaded["output_dir"] == str((tmp_path / "outputs").resolve())
    assert loaded["landslide_polygon_path"] == str((tmp_path / "inventory.geojson").resolve())


def test_overlap_priority_and_expected_class_counts() -> None:
    dep = np.array([[False, True], [False, False]])
    nondep = np.array([[True, True], [False, False]])
    uncertain = np.array([[False, False], [True, False]])

    label, _, _, _, counts = STEP1.build_final_label_raster(
        dep, nondep, uncertain, -9999, "int16"
    )

    assert label.tolist() == [[0, 1], [-9999, -9999]]
    assert counts["depositional_non_depositional_overlap_cells"] == 1


def test_duplicate_configured_landslide_ids_fail() -> None:
    gdf = gpd.GeoDataFrame(
        {"landslide_id": ["same", "same"]},
        geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
        crs="EPSG:32651",
    )
    logger = logging.getLogger("step1-test-duplicate-id")

    with pytest.raises(ValueError, match="duplicate"):
        STEP1.attach_landslide_ids(gdf, "landslide_id", logger)


def test_config_driven_run_publishes_validated_bundle(tmp_path: Path) -> None:
    _write_dem(tmp_path / "dem.tif", np.arange(100, dtype=np.float32).reshape(10, 10))
    _write_inventory(tmp_path / "inventory.geojson")
    config_path = tmp_path / "config.json"
    _write_config(config_path)

    result = subprocess.run(
        [str(RUNTIME_PYTHON), str(SCRIPT_PATH), "--config-json", str(config_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    output_dir = tmp_path / "outputs"
    expected = {
        "depositional_zone_raster.tif",
        "non_depositional_zone_raster.tif",
        "landslide_id_raster.tif",
        "ml_dfi_label_raster.tif",
        "label_summary.csv",
        "run_config.json",
        "processing_log.txt",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
    with rasterio.open(output_dir / "ml_dfi_label_raster.tif") as src:
        label = src.read(1)
        assert int(np.count_nonzero(label == 1)) == 15
        assert int(np.count_nonzero(label == 0)) == 30
        assert src.nodata == -9999
    assert not list(tmp_path.glob(".step1-stage-*"))


def test_failed_run_preserves_existing_official_output(tmp_path: Path) -> None:
    _write_dem(tmp_path / "dem.tif", np.ones((10, 10), dtype=np.float32))
    _write_inventory(tmp_path / "inventory.geojson")
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    existing = output_dir / "ml_dfi_label_raster.tif"
    existing.write_bytes(b"previous-valid-output")

    result = subprocess.run(
        [str(RUNTIME_PYTHON), str(SCRIPT_PATH), "--config-json", str(config_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert existing.read_bytes() == b"previous-valid-output"
    assert (output_dir / "processing_log.failed.txt").exists()


def test_parallel_and_serial_runs_are_cell_identical(tmp_path: Path) -> None:
    dem_values = np.arange(400, dtype=np.float32).reshape(20, 20)
    _write_dem(tmp_path / "dem.tif", dem_values)
    _write_multi_inventory(tmp_path / "inventory.geojson")
    serial_config = tmp_path / "serial.json"
    parallel_config = tmp_path / "parallel.json"
    _write_config(
        serial_config,
        "serial_outputs",
        parallel_processing=False,
        max_workers=2,
    )
    _write_config(
        parallel_config,
        "parallel_outputs",
        parallel_processing=True,
        max_workers=2,
    )

    serial_result = subprocess.run(
        [str(RUNTIME_PYTHON), str(SCRIPT_PATH), "--config-json", str(serial_config)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    parallel_result = subprocess.run(
        [str(RUNTIME_PYTHON), str(SCRIPT_PATH), "--config-json", str(parallel_config)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert serial_result.returncode == 0, serial_result.stdout + serial_result.stderr
    assert parallel_result.returncode == 0, parallel_result.stdout + parallel_result.stderr
    raster_names = (
        "depositional_zone_raster.tif",
        "non_depositional_zone_raster.tif",
        "landslide_id_raster.tif",
        "ml_dfi_label_raster.tif",
    )
    for raster_name in raster_names:
        with rasterio.open(tmp_path / "serial_outputs" / raster_name) as serial_ds:
            serial_array = serial_ds.read(1)
        with rasterio.open(tmp_path / "parallel_outputs" / raster_name) as parallel_ds:
            parallel_array = parallel_ds.read(1)
        np.testing.assert_array_equal(parallel_array, serial_array)

    serial_summary = pd.read_csv(tmp_path / "serial_outputs" / "label_summary.csv")
    parallel_summary = pd.read_csv(tmp_path / "parallel_outputs" / "label_summary.csv")
    serial_polygons = serial_summary[
        serial_summary["record_type"] == "per_landslide"
    ].reset_index(drop=True)
    parallel_polygons = parallel_summary[
        parallel_summary["record_type"] == "per_landslide"
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        parallel_polygons,
        serial_polygons,
        check_dtype=False,
    )

    parallel_receipt = json.loads(
        (tmp_path / "parallel_outputs" / "run_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert parallel_receipt["diagnostics"]["parallel_processing_used"] is True
    assert parallel_receipt["diagnostics"]["polygon_worker_count"] == 2


def test_publication_failure_rolls_back_complete_previous_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

    real_replace = STEP1.os.replace

    def fail_on_second_staged_file(source, destination) -> None:
        if Path(source) == staged_paths["second"]:
            raise OSError("simulated publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(STEP1.os, "replace", fail_on_second_staged_file)
    logger = logging.getLogger("step1-test-publication-rollback")

    with pytest.raises(OSError, match="simulated publication failure"):
        STEP1.publish_output_bundle(staged_paths, output_dir, logger)

    assert (output_dir / "first.txt").read_text(encoding="utf-8") == "old-first"
    assert (output_dir / "second.txt").read_text(encoding="utf-8") == "old-second"
    assert not list(tmp_path.glob(".step1-backup-*"))
