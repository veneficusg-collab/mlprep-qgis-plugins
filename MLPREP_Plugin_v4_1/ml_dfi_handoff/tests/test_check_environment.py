from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from scripts import check_environment as check


def _write_raster(
    path: Path,
    *,
    transform=None,
    crs: str = "EPSG:32651",
) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform or from_origin(500000, 800000, 5, 5),
        nodata=-9999,
    ) as dataset:
        dataset.write(np.ones((2, 3), dtype=np.float32), 1)


def test_step2_contract_contains_labels_predictors_groups_and_output(tmp_path):
    payload = {
        "extract_pixel_samples": {
            "label_raster_path": str(tmp_path / "label.tif"),
            "predictor_rasters": {
                "slope": str(tmp_path / "slope.tif"),
                "twi": str(tmp_path / "twi.tif"),
            },
            "group_rasters": {
                "landslide_id": str(tmp_path / "groups.tif"),
            },
            "output_dir": str(tmp_path / "output"),
        }
    }

    contract = check.collect_step_contract(
        "step2_extract_pixel_samples",
        payload,
        {"step2_extract_pixel_samples": payload},
    )

    assert len(contract.inputs) == 4
    assert len(contract.raster_group) == 4
    assert contract.outputs == [
        check.Artifact(
            "Step 2 output folder",
            (tmp_path / "output").resolve(),
            "directory",
        )
    ]


def test_missing_input_is_a_blocking_failure(tmp_path, capsys):
    contract = check.StepContract(
        "step0",
        inputs=[
            check.Artifact("slope raster", tmp_path / "missing.tif", "raster")
        ],
    )
    report = check.Report()

    check.check_input_paths(report, [contract])

    assert report.failures == 1
    assert "slope raster is missing" in capsys.readouterr().out


def test_output_write_probe_is_removed_and_disk_check_passes(tmp_path):
    output_dir = tmp_path / "new" / "output"
    contract = check.StepContract(
        "step",
        outputs=[
            check.Artifact("output folder", output_dir, "directory"),
        ],
    )
    report = check.Report()

    check.check_output_safety(report, [contract], min_free_gb=0)

    assert report.failures == 0
    assert not list(tmp_path.glob(".mldfi-write-check-*"))


def test_disk_threshold_can_fail_before_processing(tmp_path):
    contract = check.StepContract(
        "step",
        outputs=[
            check.Artifact("output folder", tmp_path, "directory"),
        ],
    )
    report = check.Report()

    check.check_output_safety(report, [contract], min_free_gb=10**9)

    assert report.failures == 1


def test_projected_metric_aligned_rasters_pass(tmp_path, monkeypatch):
    monkeypatch.delenv("PROJ_DATA", raising=False)
    monkeypatch.delenv("PROJ_LIB", raising=False)
    reference = tmp_path / "reference.tif"
    predictor = tmp_path / "predictor.tif"
    _write_raster(reference)
    _write_raster(predictor)
    contract = check.StepContract(
        "step2",
        raster_group=[
            check.Artifact("reference", reference, "raster"),
            check.Artifact("predictor", predictor, "raster"),
        ],
    )
    report = check.Report()

    check.check_spatial_compatibility(report, [contract])

    assert report.failures == 0


def test_transform_mismatch_is_a_blocking_failure(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("PROJ_DATA", raising=False)
    monkeypatch.delenv("PROJ_LIB", raising=False)
    reference = tmp_path / "reference.tif"
    shifted = tmp_path / "shifted.tif"
    _write_raster(reference)
    _write_raster(shifted, transform=from_origin(500002, 800000, 5, 5))
    contract = check.StepContract(
        "step2",
        raster_group=[
            check.Artifact("reference", reference, "raster"),
            check.Artifact("shifted", shifted, "raster"),
        ],
    )
    report = check.Report()

    check.check_spatial_compatibility(report, [contract])

    assert report.failures == 1
    assert "raster alignment mismatch" in capsys.readouterr().out
