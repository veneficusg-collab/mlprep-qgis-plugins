import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT
    / "steps"
    / "step2_extract_pixel_samples"
    / "step2_extract_pixel_samples.py"
)
LOCAL_RUNTIME_PYTHON = ROOT / ".venv-local" / "Scripts" / "python.exe"
RUNTIME_PYTHON = (
    LOCAL_RUNTIME_PYTHON if LOCAL_RUNTIME_PYTHON.exists() else Path(sys.executable)
)
SPEC = importlib.util.spec_from_file_location(
    "step2_extract_pixel_samples",
    SCRIPT_PATH,
)
STEP2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(STEP2)


def _write_raster(
    path: Path,
    values: np.ndarray,
    *,
    nodata: float | int | None,
    transform=None,
) -> None:
    if transform is None:
        transform = from_origin(0, values.shape[0], 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=str(values.dtype),
        crs="EPSG:32651",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(values, 1)


def _make_inputs(tmp_path: Path, *, invalid_group: bool = False) -> tuple[Path, Path, Path]:
    label = np.full((6, 6), -9999, dtype=np.int16)
    points = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 1),
        (4, 4, 0),
        (4, 5, 1),
        (5, 4, 0),
        (5, 5, 1),
    ]
    for row, col, target in points:
        label[row, col] = target

    predictor = np.arange(36, dtype=np.float32).reshape(6, 6)
    predictor[0, 0] = -9999
    groups = np.zeros((6, 6), dtype=np.int32)
    groups[:3, :] = 1
    groups[3:, :] = 2
    if invalid_group:
        groups[0, 1] = -9999

    label_path = tmp_path / "labels.tif"
    predictor_path = tmp_path / "predictor.tif"
    group_path = tmp_path / "groups.tif"
    _write_raster(label_path, label, nodata=-9999)
    _write_raster(predictor_path, predictor, nodata=-9999)
    _write_raster(group_path, groups, nodata=-9999)
    return label_path, predictor_path, group_path


def _write_config(
    path: Path,
    label_path: Path,
    predictor_path: Path,
    group_path: Path,
    output_dir: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "extract_pixel_samples": {
                    "label_raster_path": str(label_path),
                    "predictor_rasters": {"slope": str(predictor_path)},
                    "group_rasters": {"landslide_id": str(group_path)},
                    "categorical_predictors": [],
                    "output_dir": str(output_dir),
                    "output_table_name": "samples",
                    "output_format": "csv",
                    "label_nodata_value": -9999,
                    "include_coordinates": True,
                    "include_row_col": True,
                    "compress_outputs": True,
                }
            }
        ),
        encoding="utf-8",
    )


def _run(config_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(RUNTIME_PYTHON),
            str(SCRIPT_PATH),
            "--config-json",
            str(config_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_successful_run_filters_predictor_nodata_and_removes_stale_outputs(
    tmp_path: Path,
) -> None:
    label, predictor, groups = _make_inputs(tmp_path)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    stale_plot = output_dir / STEP2.OUTPUTS["multicollinearity_pearson_heatmap"]
    stale_plot.write_bytes(b"stale")
    config_path = tmp_path / "config.json"
    _write_config(config_path, label, predictor, groups, output_dir)

    result = _run(config_path)

    assert result.returncode == 0, result.stdout + result.stderr
    samples = pd.read_csv(output_dir / "samples.csv")
    assert len(samples) == 7
    assert set(samples["target"]) == {0, 1}
    assert set(samples["landslide_id"]) == {1, 2}
    assert np.isfinite(samples["slope"]).all()
    assert not stale_plot.exists()
    assert (output_dir / "samples_summary.csv").exists()
    assert (output_dir / STEP2.OUTPUTS["run_config"]).exists()
    assert not list(tmp_path.glob(".outputs.step2-stage-*"))


def test_alignment_failure_preserves_previous_official_output(tmp_path: Path) -> None:
    label, predictor, groups = _make_inputs(tmp_path)
    misaligned = tmp_path / "misaligned.tif"
    with rasterio.open(predictor) as src:
        values = src.read(1)
    _write_raster(
        misaligned,
        values,
        nodata=-9999,
        transform=from_origin(0.5, 6, 1, 1),
    )
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    official = output_dir / "samples.csv"
    official.write_text("last-good-output\n", encoding="utf-8")
    config_path = tmp_path / "config.json"
    _write_config(config_path, label, misaligned, groups, output_dir)

    result = _run(config_path)

    assert result.returncode == 1
    assert official.read_text(encoding="utf-8") == "last-good-output\n"
    assert (output_dir / "processing_log.failed.txt").exists()
    assert not (output_dir / STEP2.OUTPUTS["alignment_report"]).exists()


def test_invalid_landslide_group_fails_before_publication(tmp_path: Path) -> None:
    label, predictor, groups = _make_inputs(tmp_path, invalid_group=True)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    official = output_dir / "samples.csv"
    official.write_text("last-good-output\n", encoding="utf-8")
    config_path = tmp_path / "config.json"
    _write_config(config_path, label, predictor, groups, output_dir)

    result = _run(config_path)

    assert result.returncode == 1
    assert "Required group raster validation failed" in result.stdout
    assert official.read_text(encoding="utf-8") == "last-good-output\n"


def test_uint8_masked_label_does_not_use_out_of_range_fill_value() -> None:
    label = np.ma.array(
        np.array([[0, 1, 255]], dtype=np.uint8),
        mask=np.array([[False, False, True]]),
    )
    logger = logging.getLogger("step2-test-label")

    rows, cols, targets, counts = STEP2.extract_valid_label_indices(
        label,
        -9999,
        logger,
    )

    assert rows.tolist() == [0, 0]
    assert cols.tolist() == [0, 1]
    assert targets.tolist() == [0, 1]
    assert counts["valid_label_cells"] == 2


def test_configured_label_nodata_is_used_when_array_has_no_mask() -> None:
    label = np.ma.array(
        np.array([[0, 1, 255]], dtype=np.uint8),
        mask=False,
    )

    rows, cols, targets, counts = STEP2.extract_valid_label_indices(
        label,
        255,
        logging.getLogger("step2-test-fallback-nodata"),
    )

    assert rows.tolist() == [0, 0]
    assert cols.tolist() == [0, 1]
    assert targets.tolist() == [0, 1]
    assert counts["unexpected_label_value_cells"] == 0


def test_sparse_block_extraction_matches_requested_cells(tmp_path: Path) -> None:
    values = np.arange(100, dtype=np.float32).reshape(10, 10)
    values[4, 4] = -9999
    raster_path = tmp_path / "values.tif"
    _write_raster(raster_path, values, nodata=-9999)
    rows = np.array([0, 4, 9], dtype=np.int32)
    cols = np.array([1, 4, 8], dtype=np.int32)

    extracted, nodata, nonfinite = STEP2.extract_raster_values_at_points(
        str(raster_path),
        rows,
        cols,
    )

    assert extracted[0] == pytest.approx(1.0)
    assert np.isnan(extracted[1])
    assert extracted[2] == pytest.approx(98.0)
    assert nodata.tolist() == [False, True, False]
    assert not nonfinite.any()
