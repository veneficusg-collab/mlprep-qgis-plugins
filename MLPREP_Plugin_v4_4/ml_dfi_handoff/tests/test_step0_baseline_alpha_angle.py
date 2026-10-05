from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "steps"
    / "step0_baseline_alpha_angle"
    / "step0_baseline_alpha_angle.py"
)
SPEC = importlib.util.spec_from_file_location("step0_baseline_alpha_angle_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
STEP0 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STEP0
SPEC.loader.exec_module(STEP0)


def test_classification_uses_documented_lower_inclusive_breaks() -> None:
    slope = np.asarray([[9.9, 10.0, 12.0, 20.0, 45.0]], dtype=np.float64)
    valid = np.ones(slope.shape, dtype=bool)

    alpha, slope_class = STEP0.classify_source_slope(slope, valid)

    np.testing.assert_array_equal(slope_class, [[0, 1, 2, 6, 11]])
    np.testing.assert_array_equal(alpha, [[-9999.0, 9.0, 11.0, 19.0, 30.0]])


def test_windowed_run_writes_expected_outputs(tmp_path: Path) -> None:
    slope_path = tmp_path / "slope.tif"
    alpha_path = tmp_path / "alpha.tif"
    class_path = tmp_path / "class.tif"
    metadata_path = tmp_path / "metadata.json"
    slope = np.asarray(
        [
            [-9999.0, 0.0, 10.0],
            [12.0, 25.0, 45.0],
        ],
        dtype=np.float32,
    )
    with rasterio.open(
        slope_path,
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=1,
        dtype="float32",
        crs=(
            'LOCAL_CS["Test grid",LOCAL_DATUM["Test datum",32767],'
            'UNIT["metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
        ),
        transform=from_origin(0.0, 10.0, 5.0, 5.0),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(slope, 1)

    metadata = STEP0.run_step0(
        STEP0.Step0Params(
            slope=slope_path,
            output_alpha=alpha_path,
            output_class=class_path,
            metadata_out=metadata_path,
        )
    )

    with rasterio.open(alpha_path) as dataset:
        alpha = dataset.read(1)
    with rasterio.open(class_path) as dataset:
        slope_class = dataset.read(1)

    np.testing.assert_array_equal(
        alpha,
        [[-9999.0, -9999.0, 9.0], [11.0, 22.0, 30.0]],
    )
    np.testing.assert_array_equal(
        slope_class,
        [[-9999, 0, 1], [2, 7, 11]],
    )
    assert metadata["step"] == "step0"
    assert metadata["processing"]["mode"] == "windowed"
    assert metadata["valid_cell_count"] == 5
    assert metadata["non_source_class_zero_cell_count"] == 1
    assert not list(tmp_path.glob(".*.tmp*"))
