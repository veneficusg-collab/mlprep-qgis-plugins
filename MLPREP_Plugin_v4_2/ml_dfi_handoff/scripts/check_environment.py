#!/usr/bin/env python
"""Fail-fast environment and configured-data preflight for the ML-DFI workflow."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


def configure_geospatial_environment() -> dict[str, str]:
    """Use GDAL/PROJ data bundled with the active environment, not global GIS overrides."""

    site_roots = [
        Path(sys.prefix) / "Lib" / "site-packages",
        Path(sys.base_prefix) / "Lib" / "site-packages",
    ]
    candidates = {
        "PROJ": (
            ("rasterio/proj_data", "pyogrio/proj_data", "fiona/proj_data", "pyproj/proj_dir/share/proj"),
            ("PROJ_LIB", "PROJ_DATA"),
        ),
        "GDAL": (
            ("rasterio/gdal_data", "fiona/gdal_data"),
            ("GDAL_DATA",),
        ),
    }
    configured: dict[str, str] = {}
    for label, (relative_paths, variable_names) in candidates.items():
        selected: Path | None = None
        for root in site_roots:
            for relative in relative_paths:
                candidate = root / relative
                if candidate.exists():
                    selected = candidate
                    break
            if selected is not None:
                break
        if selected is None:
            continue
        value = str(selected)
        for variable_name in variable_names:
            os.environ[variable_name] = value
        configured[label] = value
    return configured


CONFIGURED_GEOSPATIAL_DIRS = configure_geospatial_environment()


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import workflow_config  # noqa: E402


REQUIRED_PYTHON = "3.13.12"
REQUIRED_PACKAGES = {
    "numpy": ("numpy", "2.4.2"),
    "numba": ("numba", "0.65.1"),
    "pandas": ("pandas", "3.0.1"),
    "scipy": ("scipy", "1.17.1"),
    "rasterio": ("rasterio", "1.5.0"),
    "geopandas": ("geopandas", "1.1.3"),
    "shapely": ("shapely", "2.1.2"),
    "pyproj": ("pyproj", "3.7.2"),
    "pyogrio": ("pyogrio", "0.12.1"),
    "sklearn": ("scikit-learn", "1.8.0"),
    "xgboost": ("xgboost", "3.2.0"),
    "optuna": ("optuna", "4.8.0"),
    "joblib": ("joblib", "1.5.3"),
    "matplotlib": ("matplotlib", "3.10.9"),
}
TAUDEM_CLI_NAMES = (
    "pitremove.exe",
    "DinfFlowDir.exe",
    "AreaDinf.exe",
)


@dataclass(frozen=True)
class Artifact:
    label: str
    path: Path
    kind: str


@dataclass
class StepContract:
    name: str
    inputs: list[Artifact] = field(default_factory=list)
    outputs: list[Artifact] = field(default_factory=list)
    raster_group: list[Artifact] = field(default_factory=list)
    vector_raster_pairs: list[tuple[Artifact, Artifact]] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class RasterMetadata:
    path: Path
    crs: Any
    transform: tuple[float, ...]
    width: int
    height: int


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0
        self.passes = 0

    def add(self, status: str, section: str, message: str, detail: str = "") -> None:
        if status == "FAIL":
            self.failures += 1
        elif status == "WARN":
            self.warnings += 1
        else:
            self.passes += 1
        suffix = f" - {detail}" if detail else ""
        print(f"[{status}] {section}: {message}{suffix}")

    def passed(self, section: str, message: str, detail: str = "") -> None:
        self.add("PASS", section, message, detail)

    def warn(self, section: str, message: str, detail: str = "") -> None:
        self.add("WARN", section, message, detail)

    def fail(self, section: str, message: str, detail: str = "") -> None:
        self.add("FAIL", section, message, detail)

    def finish(self) -> int:
        print()
        print(
            "Preflight summary: "
            f"{self.passes} passed, {self.warnings} warning(s), "
            f"{self.failures} failure(s)."
        )
        if self.failures:
            print(
                "Environment check FAILED. Correct the failures above before "
                "starting the workflow."
            )
            return 1
        print("Environment check PASSED.")
        return 0


def _raw_path(value: Any) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return Path()
    return Path(raw.split("|", 1)[0]).expanduser().resolve()


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    node: Any = payload
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(".".join(keys))
        node = node[key]
    return node


def _artifact(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    label: str,
    kind: str,
) -> Artifact:
    value = _nested(payload, *keys)
    if not str(value or "").strip():
        raise ValueError(f"Required path is empty: {'.'.join(keys)}")
    return Artifact(label, _raw_path(value), kind)


def _optional_artifact(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    label: str,
    kind: str,
) -> Artifact | None:
    try:
        value = _nested(payload, *keys)
    except KeyError:
        return None
    if not str(value or "").strip():
        return None
    return Artifact(label, _raw_path(value), kind)


def _mapping_artifacts(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    label_prefix: str,
    kind: str,
) -> list[Artifact]:
    values = _nested(payload, *keys)
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{'.'.join(keys)} must contain at least one path.")
    return [
        Artifact(f"{label_prefix} {name}", _raw_path(value), kind)
        for name, value in values.items()
        if str(value or "").strip()
    ]


def collect_step_contract(
    step: str,
    payload: dict[str, Any],
    all_payloads: dict[str, dict[str, Any]],
) -> StepContract:
    contract = StepContract(step)
    if step == "data_preparation":
        vector = _artifact(
            payload, ("dp1", "landslide_polygons"), "landslide inventory", "vector"
        )
        dem = _artifact(payload, ("dp1", "dem"), "DEM", "raster")
        contract.inputs.extend((vector, dem))
        contract.outputs.append(
            _artifact(payload, ("dp1", "output_dir"), "DP1 output folder", "directory")
        )
        contract.raster_group.append(dem)
        contract.vector_raster_pairs.append((vector, dem))
    elif step == "step0_baseline_alpha_angle":
        slope = _artifact(payload, ("slope",), "slope raster", "raster")
        contract.inputs.append(slope)
        for key, label in (
            ("output_alpha", "alpha raster"),
            ("output_class", "alpha class raster"),
            ("metadata_out", "metadata"),
        ):
            contract.outputs.append(_artifact(payload, (key,), label, "file"))
        contract.raster_group.append(slope)
    elif step == "step1_prepare_inventory_labels":
        vector = _artifact(
            payload, ("landslide_polygon_path",), "clean landslide inventory", "vector"
        )
        dem = _artifact(payload, ("dem_path",), "DEM", "raster")
        contract.inputs.extend((vector, dem))
        contract.outputs.append(
            _artifact(payload, ("output_dir",), "Step 1 output folder", "directory")
        )
        contract.raster_group.append(dem)
        contract.vector_raster_pairs.append((vector, dem))
    elif step == "step2_extract_pixel_samples":
        label = _artifact(
            payload,
            ("extract_pixel_samples", "label_raster_path"),
            "label raster",
            "raster",
        )
        predictors = _mapping_artifacts(
            payload,
            ("extract_pixel_samples", "predictor_rasters"),
            "predictor",
            "raster",
        )
        groups = _mapping_artifacts(
            payload,
            ("extract_pixel_samples", "group_rasters"),
            "group raster",
            "raster",
        )
        contract.inputs.extend((label, *predictors, *groups))
        contract.outputs.append(
            _artifact(
                payload,
                ("extract_pixel_samples", "output_dir"),
                "Step 2 output folder",
                "directory",
            )
        )
        contract.raster_group.extend((label, *predictors, *groups))
    elif step == "step3_train_ml_dfi_model":
        contract.inputs.append(
            _artifact(payload, ("input_sample_table",), "sample table", "table")
        )
        contract.outputs.append(
            _artifact(payload, ("output_dir",), "training output folder", "directory")
        )
    elif step == "step3_model_application":
        app = ("apply_ml_dfi_model",)
        for key, label, kind in (
            ("model_path", "trained model", "model"),
            ("metadata_path", "model metadata", "json"),
            ("reference_raster_path", "reference raster", "raster"),
        ):
            contract.inputs.append(_artifact(payload, (*app, key), label, kind))
        contract.outputs.append(
            _artifact(
                payload,
                (*app, "output_dir"),
                "model-application output folder",
                "directory",
            )
        )
        reference = contract.inputs[-1]
        step2 = all_payloads["step2_extract_pixel_samples"]
        predictors = _mapping_artifacts(
            step2,
            ("extract_pixel_samples", "predictor_rasters"),
            "application predictor",
            "raster",
        )
        contract.inputs.extend(predictors)
        contract.raster_group.extend((reference, *predictors))
    elif step == "step4_deposition_zone":
        input_fields = (
            ("dem_fel_raster", "filled DEM"),
            ("dinf_flow_raster", "D-Infinity direction raster"),
            ("source_raster", "source raster"),
            ("dfi_raster", "DFI raster"),
            ("alpha_raster", "alpha raster"),
        )
        required = [
            _artifact(payload, (key,), label, "raster")
            for key, label in input_fields
        ]
        contract.inputs.extend(required)
        for key, label in (
            (
                "observed_full_landslide_footprint_raster",
                "observed footprint raster",
            ),
            ("observed_depositional_zone_raster", "observed deposition raster"),
            ("validation_domain_raster", "validation-domain raster"),
        ):
            optional = _optional_artifact(payload, (key,), label, "raster")
            if optional:
                contract.inputs.append(optional)
                required.append(optional)
        for key in (
            "output_dynamic_alpha",
            "output_beta_angle",
            "output_dfs",
            "output_mask",
            "output_depositional_mask",
            "output_summary_json",
            "output_validation_summary_csv",
            "output_validation_summary_json",
        ):
            optional = _optional_artifact(payload, (key,), key, "file")
            if optional:
                contract.outputs.append(optional)
        contract.raster_group.extend(required)
    elif step == "step5_post_depositional_spread_PDS":
        inputs = [
            _artifact(payload, (key,), label, "raster")
            for key, label in (
                ("runout_mask_raster", "runout mask"),
                ("dfi_raster", "DFI raster"),
                ("dem_raster", "DEM"),
                ("source_mask_raster", "source mask"),
                ("stream_mask_raster", "stream mask"),
                (
                    "source_contributing_area_raster",
                    "source contributing-area raster",
                ),
            )
        ]
        contract.inputs.extend(inputs)
        for key, value in payload.items():
            if key.startswith("output_") and str(value or "").strip():
                contract.outputs.append(Artifact(key, _raw_path(value), "file"))
        contract.raster_group.extend(inputs)
    elif step == "step6_landslide_damming_potential_LDP":
        inputs = [
            _artifact(payload, (key,), label, "raster")
            for key, label in (
                ("dem_fel_raster", "filled DEM"),
                ("dinf_flow_raster", "D-Infinity direction raster"),
                ("runout_mask_raster", "Step 5 combined runout mask"),
                ("stream_raster", "D-Infinity stream raster"),
                ("slope_raster", "slope raster"),
            )
        ]
        contract.inputs.extend(inputs)
        for key, label, kind in (
            ("stream_vector", "stream vector", "vector"),
        ):
            optional = _optional_artifact(payload, (key,), label, kind)
            if optional:
                contract.inputs.append(optional)
                if kind == "raster":
                    inputs.append(optional)
                elif key == "stream_vector":
                    contract.vector_raster_pairs.append((optional, inputs[0]))
        for key, value in payload.items():
            if key.startswith("output_") and str(value or "").strip():
                contract.outputs.append(Artifact(key, _raw_path(value), "file"))
        contract.raster_group.extend(inputs)
    else:
        raise ValueError(f"No environment-check contract is defined for {step}.")
    return contract


def check_python_and_packages(report: Report) -> None:
    section = "Python"
    actual_python = ".".join(str(value) for value in sys.version_info[:3])
    if actual_python == REQUIRED_PYTHON:
        report.passed(section, f"Python {actual_python}")
    else:
        report.fail(
            section,
            f"Python version mismatch: expected {REQUIRED_PYTHON}, found {actual_python}",
            f"interpreter={sys.executable}",
        )

    for module_name, (distribution, expected) in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
            actual = importlib.metadata.version(distribution)
        except Exception as exc:
            report.fail(
                "Packages",
                f"{distribution} is unavailable or cannot import",
                str(exc),
            )
            continue
        if actual == expected:
            report.passed("Packages", f"{distribution} {actual}")
        else:
            report.fail(
                "Packages",
                f"{distribution} version mismatch",
                f"expected={expected}, actual={actual}",
            )

    try:
        from pyproj import CRS as PyprojCRS
        from rasterio.crs import CRS as RasterioCRS

        PyprojCRS.from_epsg(4326)
        RasterioCRS.from_epsg(4326)
    except Exception as exc:
        overrides = ", ".join(
            f"{name}={os.environ[name]}"
            for name in ("PROJ_DATA", "PROJ_LIB", "GDAL_DATA")
            if os.environ.get(name)
        )
        report.fail(
            "Packages",
            "GDAL/PROJ CRS lookup smoke test failed",
            (
                f"{exc}. Remove incompatible global GIS variables from the "
                "standard Python session; keep TauDEM PROJ settings scoped to "
                f"Step 4. Active overrides: {overrides or 'none'}"
            ),
        )
    else:
        report.passed("Packages", "GDAL/PROJ CRS lookup smoke test")

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report.fail("Packages", "pip dependency consistency check failed", str(exc))
        return
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode == 0:
        report.passed("Packages", "pip dependency consistency", output)
    else:
        report.fail("Packages", "pip dependency consistency failed", output)


def _resolve_command(value: str, fallback: str) -> Path | None:
    raw = value.strip()
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        found = shutil.which(raw)
        return Path(found).resolve() if found else None
    found = shutil.which(fallback)
    return Path(found).resolve() if found else None


def _resolve_runtime_path(
    workflow: workflow_config.LoadedWorkflowConfig,
    key: str,
) -> Path | None:
    value = str(workflow.payload["runtime"].get(key, "") or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    root = (
        workflow.local_path.parent
        if workflow.local_path is not None
        else workflow.default_path.parent
    )
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _check_qgis(
    report: Report,
    workflow: workflow_config.LoadedWorkflowConfig,
) -> Path | None:
    root = _resolve_runtime_path(workflow, "osgeo4w_root")
    if root is None:
        environment_value = os.environ.get("OSGEO4W_ROOT", "").strip()
        root = Path(environment_value).resolve() if environment_value else None
    if root is None:
        discovered = shutil.which("python-qgis-ltr.bat")
        if discovered:
            root = Path(discovered).resolve().parent.parent
    if root is None:
        report.fail(
            "Native",
            "QGIS/OSGeo4W root is not configured",
            "Set runtime.osgeo4w_root in config/local_config.json.",
        )
        return None
    launcher = root / "bin" / "python-qgis-ltr.bat"
    if launcher.is_file():
        report.passed("Native", "QGIS LTR Python launcher", str(launcher))
        return root
    report.fail(
        "Native",
        "QGIS LTR Python launcher is missing",
        str(launcher),
    )
    return None


def check_native_runtime(
    report: Report,
    workflow: workflow_config.LoadedWorkflowConfig,
    selected_steps: list[str],
    *,
    require_taudem_cli: bool,
) -> None:
    needs_qgis = any(
        step in {
            "data_preparation",
            "step4_deposition_zone",
            "step5_post_depositional_spread_PDS",
        }
        for step in selected_steps
    )
    qgis_root = _check_qgis(report, workflow) if needs_qgis else None

    if "data_preparation" in selected_steps and qgis_root is not None:
        environment = workflow_config.build_runtime_environment(workflow)
        environment["OSGEO4W_ROOT"] = str(qgis_root)
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "steps" / "data_preparation" / "launch_dp1.py"),
                    "--check-environment",
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            report.fail(
                "Native",
                "DP1 QGIS/GDAL environment check could not run",
                str(exc),
            )
            completed = None
        if completed is not None:
            output = (completed.stdout or completed.stderr).strip()
            if completed.returncode == 0:
                report.passed("Native", "DP1 QGIS/GDAL geometry smoke check")
            else:
                report.fail(
                    "Native",
                    "DP1 QGIS/GDAL environment check failed",
                    output or f"exit code {completed.returncode}",
                )

    if "step4_deposition_zone" not in selected_steps:
        return

    runtime = dict(workflow.payload["runtime"])
    mpi_value = str(runtime.get("mpiexec", "") or os.environ.get("MPIEXEC", ""))
    mpi = _resolve_command(mpi_value, "mpiexec")
    if mpi:
        report.passed("Native", "MPI launcher", str(mpi))
    else:
        report.fail(
            "Native",
            "MPI launcher was not found",
            "Configure runtime.mpiexec or add mpiexec to PATH.",
        )

    engine = _resolve_runtime_path(workflow, "step4_executable")
    if engine is None:
        engine = (
            ROOT
            / "steps"
            / "step4_deposition_zone"
            / "cpp_mpi_port"
            / "build_manual"
            / "step4_deposition_zone_mpi.exe"
        )
    if engine.is_file():
        report.passed("Native", "Step 4 MPI engine", str(engine))
    else:
        report.fail("Native", "Step 4 MPI engine is missing", str(engine))

    cli_paths = {
        Path(found).resolve()
        for name in TAUDEM_CLI_NAMES
        if (found := shutil.which(name))
    }
    dll_dir = _resolve_runtime_path(workflow, "taudem_dll_dir")
    if dll_dir is None:
        environment_value = os.environ.get("TAUDEM_DLL_DIR", "").strip()
        if environment_value:
            dll_dir = Path(environment_value).expanduser().resolve()
    if dll_dir is None:
        local_runtime = (
            ROOT / "steps" / "step4_deposition_zone" / "cpp_mpi_port" / "runtime"
        )
        found_gdal = shutil.which("gdal.dll")
        if (local_runtime / "gdal.dll").is_file():
            dll_dir = local_runtime
        elif found_gdal:
            dll_dir = Path(found_gdal).resolve().parent
        else:
            dll_dir = next(
                (
                    executable.parent
                    for executable in cli_paths
                    if (executable.parent / "gdal.dll").is_file()
                ),
                None,
            )
    if dll_dir is None or not (dll_dir / "gdal.dll").is_file():
        report.fail(
            "Native",
            "TauDEM-compatible GDAL runtime is unavailable",
            "Configure runtime.taudem_dll_dir with a folder containing gdal.dll.",
        )
    else:
        report.passed(
            "Native",
            "TauDEM-compatible GDAL runtime",
            str(dll_dir / "gdal.dll"),
        )

    proj_dir = _resolve_runtime_path(workflow, "taudem_proj_dir")
    if proj_dir is not None:
        if (proj_dir / "proj.db").is_file():
            report.passed("Native", "TauDEM PROJ data", str(proj_dir))
        else:
            report.fail(
                "Native",
                "Configured TauDEM PROJ data is invalid",
                f"proj.db not found in {proj_dir}",
            )

    if dll_dir and dll_dir.is_dir():
        for name in TAUDEM_CLI_NAMES:
            candidate = dll_dir / name
            if candidate.is_file():
                cli_paths.add(candidate.resolve())
    if cli_paths:
        report.passed(
            "Native",
            "TauDEM command-line executable availability",
            ", ".join(str(path) for path in sorted(cli_paths)),
        )
    elif require_taudem_cli:
        report.fail(
            "Native",
            "No standalone TauDEM command-line executable was found",
            "Install TauDEM or add its executable folder to PATH.",
        )
    else:
        report.warn(
            "Native",
            "No standalone TauDEM command-line executable was found",
            "The active Step 4 runtime does not call one; it uses the bundled MPI engine.",
        )


def check_input_paths(report: Report, contracts: list[StepContract]) -> None:
    seen: set[Path] = set()
    for contract in contracts:
        for artifact in contract.inputs:
            if artifact.path in seen:
                continue
            seen.add(artifact.path)
            if artifact.path.is_file():
                size_mb = artifact.path.stat().st_size / (1024 * 1024)
                report.passed(
                    "Inputs",
                    f"{contract.name}: {artifact.label}",
                    f"{artifact.path} ({size_mb:.1f} MiB)",
                )
            else:
                report.fail(
                    "Inputs",
                    f"{contract.name}: {artifact.label} is missing",
                    str(artifact.path),
                )


def _nearest_existing_directory(path: Path) -> Path | None:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if candidate.is_file():
        candidate = candidate.parent
    return candidate if candidate.is_dir() else None


def output_directories(contracts: list[StepContract]) -> dict[Path, list[str]]:
    directories: dict[Path, list[str]] = {}
    for contract in contracts:
        for artifact in contract.outputs:
            directory = (
                artifact.path
                if artifact.kind == "directory"
                else artifact.path.parent
            )
            directories.setdefault(directory, []).append(
                f"{contract.name}: {artifact.label}"
            )
    return directories


def check_output_safety(
    report: Report,
    contracts: list[StepContract],
    *,
    min_free_gb: float,
) -> None:
    inputs = {
        artifact.path
        for contract in contracts
        for artifact in contract.inputs
    }
    for contract in contracts:
        for artifact in contract.outputs:
            if artifact.path in inputs:
                report.fail(
                    "Outputs",
                    f"{contract.name}: output collides with an input",
                    str(artifact.path),
                )

    checked_anchors: dict[str, Path] = {}
    for intended, labels in output_directories(contracts).items():
        anchor = _nearest_existing_directory(intended)
        label = ", ".join(labels)
        if anchor is None:
            report.fail(
                "Outputs",
                "No existing parent is available for output creation",
                f"{intended} ({label})",
            )
            continue
        probe_name = ""
        try:
            descriptor, probe_name = tempfile.mkstemp(
                prefix=".mldfi-write-check-",
                dir=anchor,
            )
            os.close(descriptor)
            Path(probe_name).unlink()
        except OSError as exc:
            if probe_name:
                Path(probe_name).unlink(missing_ok=True)
            report.fail(
                "Outputs",
                "Output location is not writable",
                f"{intended}; tested at {anchor}: {exc}",
            )
            continue
        report.passed(
            "Outputs",
            "Output location is writable",
            f"{intended}; probe parent={anchor}",
        )
        device_key = f"{anchor.anchor.lower()}:{anchor.stat().st_dev}"
        checked_anchors.setdefault(device_key, anchor)

    for anchor in checked_anchors.values():
        try:
            usage = shutil.disk_usage(anchor)
        except OSError as exc:
            report.fail(
                "Disk",
                "Free-space query failed",
                f"{anchor}: {exc}",
            )
            continue
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        if free_gb >= min_free_gb:
            report.passed(
                "Disk",
                f"{free_gb:.1f} GiB free",
                f"{anchor.anchor or anchor} of {total_gb:.1f} GiB",
            )
        else:
            report.fail(
                "Disk",
                "Insufficient free space",
                (
                    f"{anchor.anchor or anchor}: {free_gb:.1f} GiB available, "
                    f"{min_free_gb:.1f} GiB required"
                ),
            )


def _read_raster_metadata(path: Path) -> RasterMetadata:
    import rasterio

    with rasterio.open(path) as dataset:
        return RasterMetadata(
            path=path,
            crs=dataset.crs,
            transform=tuple(dataset.transform),
            width=dataset.width,
            height=dataset.height,
        )


def _pyproj_crs(value: Any) -> Any:
    from pyproj import CRS

    return CRS.from_user_input(value)


def _check_projected_metric(
    report: Report,
    step: str,
    metadata: RasterMetadata,
) -> None:
    try:
        crs = _pyproj_crs(metadata.crs)
    except Exception as exc:
        report.fail("CRS", f"{step}: unreadable raster CRS", f"{metadata.path}: {exc}")
        return
    units = {
        str(axis.unit_name or "").lower()
        for axis in crs.axis_info
    }
    metric = bool(units) and all(
        unit in {"metre", "meter", "metres", "meters"} for unit in units
    )
    if crs.is_projected and metric:
        report.passed(
            "CRS",
            f"{step}: projected metric CRS",
            crs.to_string(),
        )
    else:
        report.fail(
            "CRS",
            f"{step}: projected metric CRS is required",
            f"{metadata.path}; crs={crs.to_string()}, units={sorted(units)}",
        )


def check_spatial_compatibility(
    report: Report,
    contracts: list[StepContract],
) -> None:
    try:
        import pyogrio
        import pyproj  # noqa: F401
        import rasterio  # noqa: F401
    except Exception as exc:
        report.fail(
            "CRS",
            "Spatial compatibility checks cannot run",
            f"Required geospatial import failed: {exc}",
        )
        return

    metadata_cache: dict[Path, RasterMetadata] = {}

    def metadata(artifact: Artifact) -> RasterMetadata | None:
        if not artifact.path.is_file():
            return None
        if artifact.path in metadata_cache:
            return metadata_cache[artifact.path]
        try:
            result = _read_raster_metadata(artifact.path)
        except Exception as exc:
            report.fail(
                "CRS",
                f"{artifact.label} cannot be opened as a raster",
                f"{artifact.path}: {exc}",
            )
            return None
        metadata_cache[artifact.path] = result
        return result

    for contract in contracts:
        raster_metadata = [
            item
            for artifact in contract.raster_group
            if (item := metadata(artifact)) is not None
        ]
        if not raster_metadata:
            continue
        reference = raster_metadata[0]
        _check_projected_metric(report, contract.name, reference)
        mismatches: list[str] = []
        for other in raster_metadata[1:]:
            try:
                same_crs = _pyproj_crs(reference.crs) == _pyproj_crs(other.crs)
            except Exception:
                same_crs = False
            same_transform = reference.transform == other.transform
            same_size = (
                reference.width == other.width
                and reference.height == other.height
            )
            if not (same_crs and same_transform and same_size):
                mismatches.append(
                    f"{other.path.name}: crs={same_crs}, "
                    f"transform={same_transform}, size={same_size}"
                )
        if mismatches:
            report.fail(
                "CRS",
                f"{contract.name}: raster alignment mismatch",
                "; ".join(mismatches),
            )
        elif len(raster_metadata) > 1:
            report.passed(
                "CRS",
                f"{contract.name}: {len(raster_metadata)} rasters align exactly",
                reference.path.name,
            )

        for vector, raster in contract.vector_raster_pairs:
            raster_meta = metadata(raster)
            if raster_meta is None or not vector.path.is_file():
                continue
            try:
                info = pyogrio.read_info(vector.path)
                vector_crs = _pyproj_crs(info["crs"])
                raster_crs = _pyproj_crs(raster_meta.crs)
            except Exception as exc:
                report.fail(
                    "CRS",
                    f"{contract.name}: vector CRS could not be read",
                    f"{vector.path}: {exc}",
                )
                continue
            if vector_crs == raster_crs:
                report.passed(
                    "CRS",
                    f"{contract.name}: vector and raster CRS match",
                    vector_crs.to_string(),
                )
            else:
                report.fail(
                    "CRS",
                    f"{contract.name}: vector and raster CRS mismatch",
                    (
                        f"vector={vector_crs.to_string()}, "
                        f"raster={raster_crs.to_string()}"
                    ),
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=workflow_config.DEFAULT_MASTER_CONFIG,
        help="Shared master config.",
    )
    parser.add_argument(
        "--local-config",
        type=Path,
        help="Optional local override. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--no-local",
        action="store_true",
        help="Ignore config/local_config.json.",
    )
    parser.add_argument(
        "--step",
        action="append",
        help="Check only one step or alias. Repeat to check multiple steps.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=10.0,
        help="Minimum free space required on each output filesystem.",
    )
    parser.add_argument(
        "--require-taudem-cli",
        action="store_true",
        help="Make absence of standalone TauDEM utilities a blocking failure.",
    )
    return parser


def _selected_steps(values: Iterable[str] | None) -> list[str]:
    if not values:
        return list(workflow_config.STEP_SPECS)
    selected: list[str] = []
    for value in values:
        canonical = workflow_config.canonical_step_name(value)
        if canonical not in selected:
            selected.append(canonical)
    return selected


def main() -> int:
    args = build_parser().parse_args()
    report = Report()
    if args.min_free_gb < 0:
        report.fail("Arguments", "--min-free-gb cannot be negative")
        return report.finish()

    print(f"Workspace: {ROOT}")
    print(f"Interpreter: {sys.executable}")
    print()
    check_python_and_packages(report)

    try:
        selected = _selected_steps(args.step)
        workflow = workflow_config.load_workflow_config(
            args.config,
            args.local_config,
            use_auto_local=not args.no_local,
        )
        payloads = {
            step: workflow_config.effective_step_config(workflow, step)[0]
            for step in workflow_config.STEP_SPECS
        }
        contracts = [
            collect_step_contract(step, payloads[step], payloads)
            for step in selected
        ]
    except Exception as exc:
        report.fail("Configuration", "Central configuration is invalid", str(exc))
        return report.finish()

    report.passed(
        "Configuration",
        "Central configuration loaded",
        (
            f"steps={', '.join(selected)}; "
            f"local={workflow.local_path or 'not present'}"
        ),
    )
    check_native_runtime(
        report,
        workflow,
        selected,
        require_taudem_cli=args.require_taudem_cli,
    )
    check_input_paths(report, contracts)
    check_output_safety(
        report,
        contracts,
        min_free_gb=args.min_free_gb,
    )
    check_spatial_compatibility(report, contracts)
    return report.finish()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nEnvironment check cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"[FAIL] Unexpected environment-check error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
