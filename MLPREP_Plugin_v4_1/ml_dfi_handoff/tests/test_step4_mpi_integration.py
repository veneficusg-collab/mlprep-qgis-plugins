import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "steps" / "step4_deposition_zone" / "run_step4.bat"
EXE = (
    ROOT
    / "steps"
    / "step4_deposition_zone"
    / "cpp_mpi_port"
    / "build_manual"
    / "step4_deposition_zone_mpi.exe"
)


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_STEP4_MPI_INTEGRATION") != "1",
    reason="Set RUN_STEP4_MPI_INTEGRATION=1 to run the external MPI/GDAL integration test.",
)


def _write_raster(path: Path, array: np.ndarray, nodata: float) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[1],
        height=array.shape[0],
        count=1,
        dtype=array.dtype,
        crs="EPSG:32651",
        transform=from_origin(500000.0, 800000.0, 5.0, 5.0),
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)


def _config(root: Path, processes: int) -> Path:
    rows, cols = 40, 60
    dem = np.repeat((100.0 - np.arange(cols, dtype=np.float32))[None, :], rows, axis=0)
    flow = np.zeros((rows, cols), dtype=np.float32)
    source = np.zeros((rows, cols), dtype=np.float32)
    source[10, 0] = 1.0
    dfi = np.full((rows, cols), 0.69, dtype=np.float32)
    alpha = np.full((rows, cols), 10.0, dtype=np.float32)

    inputs = root / "inputs"
    outputs = root / f"n{processes}"
    inputs.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)
    for name, array in {
        "dem.tif": dem,
        "flow.tif": flow,
        "source.tif": source,
        "dfi.tif": dfi,
        "alpha.tif": alpha,
    }.items():
        path = inputs / name
        if not path.exists():
            _write_raster(path, array, -9999.0)

    config = {
        "dem_fel_raster": str(inputs / "dem.tif"),
        "dinf_flow_raster": str(inputs / "flow.tif"),
        "source_raster": str(inputs / "source.tif"),
        "dfi_raster": str(inputs / "dfi.tif"),
        "alpha_raster": str(inputs / "alpha.tif"),
        "proportion_threshold": 0.2,
        "dfi_mid": 0.69,
        "alpha_gain_per_meter": 0.03333,
        "write_debug_rasters": False,
        "output_dynamic_alpha": str(outputs / "dynamic_alpha.tif"),
        "output_beta_angle": str(outputs / "beta.tif"),
        "output_dfs": str(outputs / "dfs.tif"),
        "output_mask": str(outputs / "mask.tif"),
        "output_depositional_mask": str(outputs / "deposition.tif"),
        "output_summary_json": str(outputs / "summary.json"),
    }
    path = root / f"config_n{processes}.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def _run(config: Path, processes: int) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    command = subprocess.list2cmdline(
        [
            str(RUNNER),
            "--config",
            str(config),
            "--processes",
            str(processes),
            "--exe",
            str(EXE),
        ]
    )
    return subprocess.run(
        f"cmd.exe /d /s /c call {command}",
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
        timeout=120,
    )


def test_mpi_process_counts_match_and_planimetric_cap_is_enforced(tmp_path: Path) -> None:
    results: dict[int, dict[str, np.ndarray]] = {}
    for processes in (1, 2):
        config = _config(tmp_path, processes)
        completed = _run(config, processes)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        output_dir = tmp_path / f"n{processes}"
        arrays: dict[str, np.ndarray] = {}
        for name in ("dynamic_alpha", "beta", "dfs", "mask", "deposition"):
            with rasterio.open(output_dir / f"{name}.tif") as src:
                arrays[name] = src.read(1)
        results[processes] = arrays

        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["engine"]["executable_sha256"]
        assert summary["config"]["sha256"]
        assert summary["parameters"]["max_planimetric_flow_path_length_m"] == 200.0
        assert not list(output_dir.glob(".step4-stage-*"))

    for name in results[1]:
        np.testing.assert_array_equal(results[1][name], results[2][name])

    runout = results[1]["mask"] > 0
    assert int(np.count_nonzero(runout)) == 41
    assert float(np.max(results[1]["dfs"][runout])) == pytest.approx(200.0)


def _rasters_are_exactly_equal(left: Path, right: Path) -> bool:
    with rasterio.open(left) as left_ds, rasterio.open(right) as right_ds:
        if (
            left_ds.width != right_ds.width
            or left_ds.height != right_ds.height
            or left_ds.transform != right_ds.transform
            or left_ds.crs != right_ds.crs
        ):
            return False
        for _, window in left_ds.block_windows(1):
            if not np.array_equal(left_ds.read(1, window=window), right_ds.read(1, window=window)):
                return False
    return True


@pytest.mark.skipif(
    os.environ.get("RUN_STEP4_FULL_REGRESSION") != "1",
    reason="Set RUN_STEP4_FULL_REGRESSION=1 to run the full Makilala process-count regression.",
)
def test_full_default_n1_n8_and_retained_baseline_match(tmp_path: Path) -> None:
    default_path = (
        ROOT / "steps" / "step4_deposition_zone" / "config.default.json"
    )
    default = json.loads(default_path.read_text(encoding="utf-8"))
    config_dir = default_path.parent
    for key in (
        "dem_fel_raster",
        "dinf_flow_raster",
        "source_raster",
        "dfi_raster",
        "alpha_raster",
    ):
        default[key] = str((config_dir / default[key]).resolve())

    output_names = {
        "output_dynamic_alpha": "dynamic_alpha.tif",
        "output_beta_angle": "beta.tif",
        "output_dfs": "dfs.tif",
        "output_mask": "mask.tif",
        "output_depositional_mask": "deposition.tif",
        "output_summary_json": "summary.json",
    }
    run_dirs: dict[int, Path] = {}
    for processes in (1, 8):
        run_dir = tmp_path / f"full_n{processes}"
        run_dir.mkdir()
        run_dirs[processes] = run_dir
        config = dict(default)
        for key, name in output_names.items():
            config[key] = str(run_dir / name)
        config_path = tmp_path / f"full_n{processes}.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        completed = _run(config_path, processes)
        assert completed.returncode == 0, completed.stdout + completed.stderr

    official_paths = {
        key: (config_dir / value).resolve()
        for key, value in default.items()
        if key in output_names
    }
    for key, name in output_names.items():
        if not name.endswith(".tif"):
            continue
        n1 = run_dirs[1] / name
        n8 = run_dirs[8] / name
        assert _rasters_are_exactly_equal(n1, n8), f"n1/n8 mismatch: {key}"
        assert _rasters_are_exactly_equal(n8, official_paths[key]), f"retained baseline mismatch: {key}"
