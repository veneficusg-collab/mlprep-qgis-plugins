"""Wrapper for the Step 4 C++/MPI routing executable.

This keeps the Python config/reporting workflow while delegating the heavy
dynamic-alpha raster routing to cpp_mpi_port/build_manual/step4_deposition_zone_mpi.exe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .step4_common import (
        ALPHA_ANGLE_VALID_MAX,
        ALPHA_ANGLE_VALID_MIN,
        FIXED_ALPHA_MAX,
        FIXED_ALPHA_MIN,
        FIXED_FAIL_ON_UNPROCESSED_CELLS,
        FIXED_MAX_SOURCE_FRACTION,
        FIXED_MIN_DFI_VALID_FRACTION,
        TAUDEM_FLOAT_NODATA,
        DynamicAlphaCoreResult,
        RasterMetadata,
        RasterSpec,
        Step4Params,
        _ensure_parent_dir,
        assert_projected_metric_grid,
        build_debug_output_paths,
        load_step4_params,
        read_raster,
        read_raster_metadata,
        require_osgeo,
        run_validation_metrics,
        srs_from_wkt,
        step4_output_paths,
        validate_alignment,
        valid_mask,
        write_validation_summaries,
        gdal,
    )
except ImportError:
    from step4_common import (  # type: ignore[no-redef]
        ALPHA_ANGLE_VALID_MAX,
        ALPHA_ANGLE_VALID_MIN,
        FIXED_ALPHA_MAX,
        FIXED_ALPHA_MIN,
        FIXED_FAIL_ON_UNPROCESSED_CELLS,
        FIXED_MAX_SOURCE_FRACTION,
        FIXED_MIN_DFI_VALID_FRACTION,
        TAUDEM_FLOAT_NODATA,
        DynamicAlphaCoreResult,
        RasterMetadata,
        RasterSpec,
        Step4Params,
        _ensure_parent_dir,
        assert_projected_metric_grid,
        build_debug_output_paths,
        load_step4_params,
        read_raster,
        read_raster_metadata,
        require_osgeo,
        run_validation_metrics,
        srs_from_wkt,
        step4_output_paths,
        validate_alignment,
        valid_mask,
        write_validation_summaries,
        gdal,
    )


DEFAULT_EXE = Path(__file__).resolve().parent / "cpp_mpi_port" / "build_manual" / "step4_deposition_zone_mpi.exe"
TAUDEM_DLL_ENV = "TAUDEM_DLL_DIR"
TAUDEM_PROJ_ENV = "TAUDEM_PROJ_DIR"
MPIEXEC_ENV = "MPIEXEC"


@dataclass(frozen=True)
class StagedOutputBundle:
    params: Step4Params
    staged_to_official: dict[str, str]
    stage_dirs: tuple[Path, ...]


OUTPUT_PARAM_FIELDS = (
    "output_dynamic_alpha",
    "output_beta_angle",
    "output_dfs",
    "output_mask",
    "output_depositional_mask",
    "output_summary_json",
    "output_validation_summary_csv",
    "output_validation_summary_json",
)


def _resolve(path: str) -> str:
    return os.path.abspath(path) if path else ""


def _load_cpp_outputs(params: Step4Params) -> tuple[DynamicAlphaCoreResult, dict[str, Any]]:
    fel = read_raster(_resolve(params.dem_fel_raster), "DEMfel")
    ang = read_raster(_resolve(params.dinf_flow_raster), "DinfFlow")
    source = read_raster(_resolve(params.source_raster), "source")
    alpha = read_raster(_resolve(params.alpha_raster), "alpha")
    dfi = read_raster(_resolve(params.dfi_raster), "DFI")
    dynamic_alpha = read_raster(_resolve(params.output_dynamic_alpha), "dynamic_alpha")
    beta_angle = read_raster(_resolve(params.output_beta_angle), "beta_angle")
    dfs = read_raster(_resolve(params.output_dfs), "dfs")
    mask = read_raster(_resolve(params.output_mask), "runout_mask")
    deposition = read_raster(_resolve(params.output_depositional_mask), "depositional_mask")
    for label, raster in (
        ("DinfFlow", ang),
        ("source", source),
        ("alpha", alpha),
        ("DFI", dfi),
        ("dynamic_alpha", dynamic_alpha),
        ("beta_angle", beta_angle),
        ("DFS", dfs),
        ("runout_mask", mask),
        ("depositional_mask", deposition),
    ):
        validate_alignment(fel, raster, label)

    terrain_valid = valid_mask(fel.array, fel.nodata) & valid_mask(ang.array, ang.nodata)
    alpha_valid = terrain_valid & valid_mask(alpha.array, alpha.nodata) & np.isfinite(alpha.array)
    source_marked = terrain_valid & valid_mask(source.array, source.nodata) & (source.array > 0)
    source_cells = source_marked & alpha_valid
    dfi_valid = valid_mask(dfi.array, dfi.nodata)
    runout = valid_mask(mask.array, mask.nodata) & (mask.array > 0)
    depositional = valid_mask(deposition.array, deposition.nodata) & (deposition.array > 0)

    terrain_valid_count = int(np.count_nonzero(terrain_valid))
    remaining_dependency_cells = 0
    remaining_dependency_fraction = 0.0

    result = DynamicAlphaCoreResult(
        dynamic_alpha_out=np.asarray(dynamic_alpha.array, dtype=np.float32),
        beta_angle_out=np.asarray(beta_angle.array, dtype=np.float32),
        dfs_out=np.asarray(dfs.array, dtype=np.float32),
        mask_out=np.asarray(runout, dtype=np.uint8),
        depositional_mask_out=np.asarray(depositional, dtype=np.uint8),
        accepted_beta_out=np.asarray(beta_angle.array, dtype=np.float32),
        parent_dynamic_alpha_out=None,
        valid_cells=terrain_valid_count,
        terrain_valid_cells=terrain_valid_count,
        alpha_valid_cells=int(np.count_nonzero(alpha_valid)),
        source_cells=int(np.count_nonzero(source_cells)),
        non_source_terrain_valid_cells=int(np.count_nonzero(terrain_valid & (~source_cells))),
        runout_cells=int(np.count_nonzero(runout)),
        pure_depositional_cells=int(np.count_nonzero(depositional)),
        processed_cells=0,
        remaining_dependency_cells=remaining_dependency_cells,
        remaining_dependency_fraction=remaining_dependency_fraction,
        candidate_evaluations=0,
        accepted_candidate_propagations=0,
        source_cells_without_valid_alpha=int(np.count_nonzero(source_marked & (~alpha_valid))),
        terrain_valid_cells_without_valid_dfi=int(np.count_nonzero(terrain_valid & (~dfi_valid))),
        candidate_cells_skipped_missing_dfi=0,
        dfi_clamped_count=0,
    )

    context = {
        "fel": fel,
        "source": source,
        "alpha": alpha,
        "dfi": dfi,
        "terrain_valid": terrain_valid,
        "source_cells": source_cells,
        "source_marked": source_marked,
        "runout": runout,
        "depositional": depositional,
    }
    return result, context


def resolve_mpiexec(value: str | None) -> str:
    raw = value or os.environ.get(MPIEXEC_ENV) or ""
    if raw:
        path = Path(raw).expanduser()
        if path.is_file():
            return str(path.resolve())
        found = shutil.which(raw)
        if found:
            return found
        raise FileNotFoundError(f"MPI launcher does not exist and is not on PATH: {raw}")
    found = shutil.which("mpiexec")
    if found:
        return found
    raise FileNotFoundError(
        "Could not find mpiexec. Pass --mpiexec, set "
        f"{MPIEXEC_ENV}, or add the MPI launcher to PATH."
    )


def _build_command(
    params: Step4Params,
    exe: Path,
    processes: int,
    mpiexec: str,
    engine_stats_json: str = "",
) -> list[str]:
    cmd = [
        mpiexec,
        "-n",
        str(processes),
        str(exe),
        "--fel",
        _resolve(params.dem_fel_raster),
        "--ang",
        _resolve(params.dinf_flow_raster),
        "--source",
        _resolve(params.source_raster),
        "--dfi",
        _resolve(params.dfi_raster),
        "--alpha",
        _resolve(params.alpha_raster),
        "--out-alpha",
        _resolve(params.output_dynamic_alpha),
        "--out-beta",
        _resolve(params.output_beta_angle),
        "--out-dfs",
        _resolve(params.output_dfs),
        "--out-mask",
        _resolve(params.output_mask),
        "--out-deposition",
        _resolve(params.output_depositional_mask),
        "--threshold",
        str(float(params.proportion_threshold)),
        "--dfi-mid",
        str(float(params.dfi_mid)),
        "--alpha-gain-per-meter",
        str(float(params.alpha_gain_per_meter)),
    ]
    debug_paths = build_debug_output_paths(params.output_dynamic_alpha, params.write_debug_rasters)
    if debug_paths["parent_dynamic_alpha"]:
        cmd.extend(["--out-parent-alpha", _resolve(debug_paths["parent_dynamic_alpha"])])
    if engine_stats_json:
        cmd.extend(["--stats-json", _resolve(engine_stats_json)])
    return cmd


def _load_engine_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"C++/MPI engine did not write its statistics receipt: {path}")
    with path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    if not isinstance(stats, dict) or int(stats.get("schema_version", 0)) != 1:
        raise RuntimeError(f"Unsupported or invalid Step 4 engine statistics receipt: {path}")
    return stats


def _validate_cpp_output_metadata(params: Step4Params) -> dict[str, dict[str, Any]]:
    reference = read_raster_metadata(_resolve(params.dem_fel_raster), "DEMfel")
    output_paths = {
        "dynamic_alpha_raster": params.output_dynamic_alpha,
        "beta_angle_raster": params.output_beta_angle,
        "dfs_raster": params.output_dfs,
        "runout_mask": params.output_mask,
        "depositional_mask": params.output_depositional_mask,
    }
    debug_paths = build_debug_output_paths(params.output_dynamic_alpha, params.write_debug_rasters)
    if debug_paths["parent_dynamic_alpha"]:
        output_paths["parent_dynamic_alpha_raster"] = debug_paths["parent_dynamic_alpha"]

    artifacts: dict[str, dict[str, Any]] = {}
    for label, path in output_paths.items():
        metadata = read_raster_metadata(_resolve(path), label)
        validate_alignment(reference, metadata, label)
        file_path = Path(path)
        size = file_path.stat().st_size
        if size <= 0:
            raise RuntimeError(f"Step 4 output is empty: {file_path}")
        artifacts[label] = {
            "staged_path": str(file_path.resolve()),
            "size_bytes": int(size),
            "rows": int(metadata.rows),
            "cols": int(metadata.cols),
        }
    return artifacts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_taudem_dll_dir(value: str | None) -> Path:
    """Resolve and validate the TauDEM runtime DLL folder used by the C++ executable."""
    candidates: list[Path] = []
    if value:
        candidates.append(Path(value))
    elif os.environ.get(TAUDEM_DLL_ENV):
        candidates.append(Path(str(os.environ[TAUDEM_DLL_ENV])))
    else:
        local_runtime = Path(__file__).resolve().parent / "cpp_mpi_port" / "runtime"
        candidates.append(local_runtime)
        found_gdal = shutil.which("gdal.dll")
        if found_gdal:
            candidates.append(Path(found_gdal).parent)

    checked: list[str] = []
    for candidate in candidates:
        dll_dir = candidate.expanduser().resolve()
        checked.append(str(dll_dir))
        if dll_dir.is_dir() and (dll_dir / "gdal.dll").is_file():
            return dll_dir
    raise FileNotFoundError(
        "Could not find a TauDEM-compatible runtime folder containing gdal.dll. "
        f"Checked: {checked}. Pass --taudem-dll-dir or set {TAUDEM_DLL_ENV}."
    )


def resolve_taudem_proj_dir(dll_dir: Path) -> Path | None:
    """Resolve an optional PROJ data directory without assuming an installation root."""

    configured = str(os.environ.get(TAUDEM_PROJ_ENV, "")).strip()
    if configured:
        resolved = Path(configured).expanduser().resolve()
        if resolved.is_dir() and (resolved / "proj.db").is_file():
            return resolved
        raise FileNotFoundError(
            f"{TAUDEM_PROJ_ENV} does not identify a PROJ data folder containing proj.db: "
            f"{Path(configured).expanduser()}"
        )

    candidates: list[Path] = []
    for environment_name in ("PROJ_DATA", "PROJ_LIB"):
        environment_value = str(os.environ.get(environment_name, "")).strip()
        if environment_value:
            candidates.append(Path(environment_value))
    candidates.append(dll_dir.parent / "share" / "proj")
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_dir() and (resolved / "proj.db").is_file():
            return resolved
    return None


def _required_input_paths(params: Step4Params) -> dict[str, str]:
    return {
        "DEMfel": params.dem_fel_raster,
        "DinfFlow": params.dinf_flow_raster,
        "source": params.source_raster,
        "DFI": params.dfi_raster,
        "alpha": params.alpha_raster,
    }


def _optional_validation_input_paths(params: Step4Params) -> dict[str, str]:
    return {
        "observed_full_landslide_footprint": params.observed_full_landslide_footprint_raster,
        "observed_depositional_zone": params.observed_depositional_zone_raster,
        "validation_domain": params.validation_domain_raster,
    }


def _check_input_paths_exist(params: Step4Params) -> None:
    missing: list[str] = []
    for label, path in {**_required_input_paths(params), **_optional_validation_input_paths(params)}.items():
        if path and not Path(path).exists():
            missing.append(f"{label}: {path}")
    if missing:
        raise FileNotFoundError("Missing Step 4 input raster(s):\n  " + "\n  ".join(missing))


def _ensure_output_parent_dirs(params: Step4Params) -> list[str]:
    parent_dirs: list[str] = []
    for output in step4_output_paths(params).values():
        if not output:
            continue
        _ensure_parent_dir(output)
        parent = str(Path(output).resolve().parent)
        if parent not in parent_dirs:
            parent_dirs.append(parent)
    return parent_dirs


def prepare_staged_output_bundle(params: Step4Params) -> StagedOutputBundle:
    """Create same-filesystem staging paths for every configured output."""

    token = uuid.uuid4().hex
    stage_dirs_by_parent: dict[Path, Path] = {}
    replacements: dict[str, str] = {}
    staged_to_official: dict[str, str] = {}

    for field_name in OUTPUT_PARAM_FIELDS:
        official_raw = str(getattr(params, field_name) or "")
        if not official_raw:
            continue
        official = Path(official_raw).resolve()
        stage_dir = stage_dirs_by_parent.get(official.parent)
        if stage_dir is None:
            stage_dir = official.parent / f".step4-stage-{token}"
            stage_dir.mkdir(parents=True, exist_ok=False)
            stage_dirs_by_parent[official.parent] = stage_dir
        staged = stage_dir / official.name
        replacements[field_name] = str(staged)
        staged_to_official[str(staged)] = str(official)

    staged_params = replace(params, **replacements)
    official_debug = build_debug_output_paths(params.output_dynamic_alpha, params.write_debug_rasters)
    staged_debug = build_debug_output_paths(
        staged_params.output_dynamic_alpha,
        staged_params.write_debug_rasters,
    )
    for key, official in official_debug.items():
        staged = staged_debug.get(key, "")
        if official and staged:
            staged_to_official[str(Path(staged).resolve())] = str(Path(official).resolve())

    return StagedOutputBundle(
        params=staged_params,
        staged_to_official=staged_to_official,
        stage_dirs=tuple(stage_dirs_by_parent.values()),
    )


def cleanup_staged_output_bundle(bundle: StagedOutputBundle) -> None:
    for stage_dir in bundle.stage_dirs:
        shutil.rmtree(stage_dir, ignore_errors=True)


def _sidecar_paths(path: Path) -> list[Path]:
    return sorted(
        candidate
        for candidate in path.parent.glob(path.name + ".*")
        if candidate.is_file()
    )


def publish_staged_output_bundle(bundle: StagedOutputBundle) -> None:
    """Publish all outputs with rollback if any replacement fails."""

    pairs = [(Path(staged), Path(official)) for staged, official in bundle.staged_to_official.items()]
    missing = [str(staged) for staged, _ in pairs if not staged.is_file()]
    if missing:
        raise RuntimeError("Step 4 staged output bundle is incomplete:\n  " + "\n  ".join(missing))

    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for pair_index, (staged, official) in enumerate(pairs):
            existing_paths = ([official] if official.exists() else []) + _sidecar_paths(official)
            for existing_index, existing in enumerate(existing_paths):
                backup = staged.parent / f".backup-{pair_index}-{existing_index}-{existing.name}"
                os.replace(existing, backup)
                backups.append((backup, existing))

            os.replace(staged, official)
            published.append(official)
            staged_prefix = str(staged)
            for staged_sidecar in _sidecar_paths(staged):
                suffix = str(staged_sidecar)[len(staged_prefix) :]
                official_sidecar = Path(str(official) + suffix)
                os.replace(staged_sidecar, official_sidecar)
                published.append(official_sidecar)
    except Exception:
        for published_path in reversed(published):
            try:
                published_path.unlink(missing_ok=True)
            except OSError:
                pass
        rollback_errors: list[str] = []
        for backup, original in reversed(backups):
            try:
                os.replace(backup, original)
            except OSError as exc:
                rollback_errors.append(f"{original}: {exc}")
        if rollback_errors:
            raise RuntimeError(
                "Step 4 publication failed and rollback was incomplete:\n  "
                + "\n  ".join(rollback_errors)
            )
        raise


def _official_command(command: list[str], bundle: StagedOutputBundle) -> list[str]:
    reverse_map = {
        str(Path(staged).resolve()): str(Path(official).resolve())
        for staged, official in bundle.staged_to_official.items()
    }
    official: list[str] = []
    skip_next = False
    for value in command:
        if skip_next:
            skip_next = False
            continue
        if value == "--stats-json":
            skip_next = True
            continue
        official.append(reverse_map.get(str(Path(value).resolve()), value) if value else value)
    return official


def _assert_projected_crs_in_meters(reference: RasterSpec) -> tuple[float, float]:
    dx, dy = assert_projected_metric_grid(reference)
    srs = srs_from_wkt(reference.projection_wkt)
    linear_units = float(srs.GetLinearUnits())
    if abs(linear_units - 1.0) > 1e-9:
        unit_name = srs.GetLinearUnitsName() or "unknown"
        raise ValueError(
            "Step 4 requires projected CRS rasters with meter units. "
            f"Reference raster unit is {unit_name!r} with conversion factor {linear_units:g}."
        )
    return dx, dy


def _read_and_validate_input_rasters(params: Step4Params) -> tuple[dict[str, RasterMetadata], dict[str, Any]]:
    fel = read_raster_metadata(_resolve(params.dem_fel_raster), "DEMfel")
    rasters: dict[str, RasterMetadata] = {
        "fel": fel,
        "ang": read_raster_metadata(_resolve(params.dinf_flow_raster), "DinfFlow"),
        "source": read_raster_metadata(_resolve(params.source_raster), "source"),
        "dfi": read_raster_metadata(_resolve(params.dfi_raster), "DFI"),
        "alpha": read_raster_metadata(_resolve(params.alpha_raster), "alpha"),
    }

    validate_alignment(fel, rasters["ang"], "D-infinity angle")
    validate_alignment(fel, rasters["source"], "source")
    validate_alignment(fel, rasters["dfi"], "DFI")
    validate_alignment(fel, rasters["alpha"], "alpha")
    dx, dy = _assert_projected_crs_in_meters(fel)

    optional_rasters: dict[str, RasterMetadata] = {}
    for label, path in _optional_validation_input_paths(params).items():
        if not path:
            continue
        optional = read_raster_metadata(_resolve(path), label)
        validate_alignment(fel, optional, label)
        optional_rasters[label] = optional

    rasters.update(optional_rasters)
    return rasters, {"dx": dx, "dy": dy, "optional_validation_raster_count": len(optional_rasters)}


def run_preflight_checks(params: Step4Params) -> dict[str, Any]:
    """Validate paths and grid metadata before launching MPI."""

    require_osgeo()
    if gdal is not None:
        gdal.UseExceptions()

    _check_input_paths_exist(params)
    output_parent_dirs = _ensure_output_parent_dirs(params)
    rasters, grid_summary = _read_and_validate_input_rasters(params)
    fel = rasters["fel"]

    return {
        "status": "metadata_passed",
        "warnings": [],
        "grid": {
            "cols": int(fel.cols),
            "rows": int(fel.rows),
            "dx": float(grid_summary["dx"]),
            "dy": float(grid_summary["dy"]),
            "projected_crs_units": "meter",
            "optional_validation_raster_count": int(grid_summary["optional_validation_raster_count"]),
        },
        "counts": {},
        "fractions": {},
        "ranges": {},
        "engine_value_validation": "required_before_publication",
        "output_parent_dirs": output_parent_dirs,
    }


def merge_engine_validation(
    preflight_summary: dict[str, Any],
    engine_stats: dict[str, Any],
) -> dict[str, Any]:
    counts = dict(engine_stats.get("counts", {}))
    fractions = dict(engine_stats.get("fractions", {}))
    ranges = dict(engine_stats.get("ranges", {}))
    warnings: list[str] = []
    missing_alpha = int(counts.get("source_cells_without_valid_alpha", 0))
    source_fraction = float(fractions.get("source_fraction", 0.0))
    dfi_fraction = float(fractions.get("dfi_valid_fraction_over_terrain", 0.0))
    if missing_alpha > 0:
        warnings.append(
            f"Excluded {missing_alpha} source-marked cells from initiation because alpha is missing or NoData."
        )
    if source_fraction > FIXED_MAX_SOURCE_FRACTION:
        warnings.append(
            f"Initiating source cells cover {source_fraction:.2%} of terrain-valid cells, "
            f"above fixed max_source_fraction={FIXED_MAX_SOURCE_FRACTION:.2%}."
        )
    if dfi_fraction < FIXED_MIN_DFI_VALID_FRACTION:
        warnings.append(
            f"DFI valid coverage over terrain-valid cells is {dfi_fraction:.2%}, "
            f"below fixed min_dfi_valid_fraction={FIXED_MIN_DFI_VALID_FRACTION:.2%}."
        )
    merged = dict(preflight_summary)
    merged["status"] = "passed"
    merged["warnings"] = warnings
    merged["counts"] = counts
    merged["fractions"] = fractions
    merged["ranges"] = {
        **ranges,
        "alpha_valid_range_degrees": [ALPHA_ANGLE_VALID_MIN, ALPHA_ANGLE_VALID_MAX],
        "source_alpha_required_range_degrees": [FIXED_ALPHA_MIN, FIXED_ALPHA_MAX],
    }
    merged["engine_value_validation"] = "passed"
    return merged


def _make_summary(
    *,
    params: Step4Params,
    engine_stats: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    command: list[str],
    config_path: Path,
    executable_path: Path,
    returncode: int,
    engine_wall_seconds: float,
    wrapper_elapsed_seconds: float,
    preflight_summary: dict[str, Any],
    validation_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    counts = dict(engine_stats.get("counts", {}))
    ranges = dict(engine_stats.get("ranges", {}))
    source_alpha_stats = dict(ranges.get("source_alpha_degrees", {}))
    dynamic_alpha_stats = dict(ranges.get("dynamic_alpha_degrees", {}))
    terrain_cells = int(counts.get("terrain_valid_cells", 0))
    source_cells = int(counts.get("source_cells", 0))

    return {
        "step": {
            "name": "step4_deposition_zone_mpi_wrapper",
            "description": "Python config/reporting wrapper around the C++/MPI Step 4 routing executable.",
        },
        "engine": {
            "mode": "cpp_mpi",
            "executable": str(executable_path.resolve()),
            "executable_sha256": _sha256(executable_path),
            "processes": int(command[2]) if len(command) > 2 and command[2].isdigit() else None,
            "returncode": int(returncode),
            "engine_wall_seconds": float(engine_wall_seconds),
            "wrapper_total_elapsed_seconds": float(wrapper_elapsed_seconds),
            "engine_timing_seconds": engine_stats.get("timing_seconds", {}),
            "command": command,
        },
        "config": {
            "path": str(config_path.resolve()),
            "sha256": _sha256(config_path),
        },
        "inputs": {
            "dem_fel_raster": params.dem_fel_raster,
            "dinf_flow_raster": params.dinf_flow_raster,
            "source_raster": params.source_raster,
            "dfi_raster": params.dfi_raster,
            "alpha_raster": params.alpha_raster,
            "validation_domain_raster": params.validation_domain_raster,
        },
        "parameters": {
            "proportion_threshold": float(params.proportion_threshold),
            "path_mode": "flow",
            "angle_units_contract": "TauDEM radians",
            "alpha_rule": "path_dependent_dynamic_alpha",
            "alpha_gain_per_meter": float(params.alpha_gain_per_meter),
            "dfi_mid": float(params.dfi_mid),
            "max_planimetric_flow_path_length_m": 200.0,
        },
        "counts": {
            **counts,
            "non_source_terrain_valid_cells": max(0, terrain_cells - source_cells),
        },
        "alpha_stats": {
            "source_alpha_unclamped": source_alpha_stats,
            "dynamic_alpha_final": dynamic_alpha_stats,
        },
        "outputs": {
            "dynamic_alpha_raster": _resolve(params.output_dynamic_alpha),
            "beta_angle_raster": _resolve(params.output_beta_angle),
            "dfs_raster": _resolve(params.output_dfs),
            "runout_mask": _resolve(params.output_mask),
            "depositional_mask": _resolve(params.output_depositional_mask),
            "summary_json": _resolve(params.output_summary_json),
            "validation_summary_csv": _resolve(params.output_validation_summary_csv),
            "validation_summary_json": _resolve(params.output_validation_summary_json),
        },
        "artifacts": {
            label: {
                "size_bytes": int(details["size_bytes"]),
                "rows": int(details["rows"]),
                "cols": int(details["cols"]),
            }
            for label, details in artifacts.items()
        },
        "preflight": preflight_summary,
        "validation": validation_summary,
        "notes": [
            "The C++/MPI executable performs the heavy routing; this wrapper handles config and reporting.",
            "Input value validation and routing counters are emitted by the distributed C++ engine.",
            "The 200 m cap is accumulated planimetric D-Infinity flow-path length, not three-dimensional terrain-surface distance.",
        ],
    }


def run_mpi_wrapper(
    config: str,
    processes: int,
    exe: str,
    dry_run: bool = False,
    taudem_dll_dir: str | None = None,
    mpiexec: str | None = None,
) -> dict[str, Any]:
    wrapper_start = time.perf_counter()
    config_path = Path(config).resolve()
    params = load_step4_params(config)

    exe_path = Path(exe).resolve()
    if not exe_path.exists():
        raise FileNotFoundError(f"C++/MPI executable not found: {exe_path}")

    preflight_summary = run_preflight_checks(params)

    mpiexec_path = resolve_mpiexec(mpiexec)
    env = os.environ.copy()
    dll_dir = resolve_taudem_dll_dir(taudem_dll_dir)
    env["PATH"] = str(dll_dir) + os.pathsep + env.get("PATH", "")
    proj_dir = resolve_taudem_proj_dir(dll_dir)
    if proj_dir is not None:
        env["PROJ_LIB"] = str(proj_dir)
        env["PROJ_DATA"] = str(proj_dir)

    if dry_run:
        cmd = _build_command(params, exe_path, processes, mpiexec_path)
        return {
            "dry_run": True,
            "command": cmd,
            "taudem_dll_dir": str(dll_dir),
            "mpiexec": mpiexec_path,
            "path_prefix": str(dll_dir),
            "taudem_dll_env": TAUDEM_DLL_ENV,
            "taudem_proj_dir": str(proj_dir) if proj_dir is not None else "",
            "taudem_proj_env": TAUDEM_PROJ_ENV,
            "preflight": preflight_summary,
        }

    bundle = prepare_staged_output_bundle(params)
    try:
        staged_params = bundle.params
        engine_stats_path = bundle.stage_dirs[0] / "step4_engine_stats.json"
        cmd = _build_command(
            staged_params,
            exe_path,
            processes,
            mpiexec_path,
            str(engine_stats_path),
        )
        start = time.perf_counter()
        process = subprocess.Popen(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="")
        returncode = process.wait()
        elapsed = time.perf_counter() - start
        if returncode != 0:
            raise RuntimeError(f"C++/MPI Step 4 failed with return code {returncode}.")

        engine_stats = _load_engine_stats(engine_stats_path)
        preflight_summary = merge_engine_validation(preflight_summary, engine_stats)
        for warning in preflight_summary.get("warnings", []):
            print(f"Validation warning: {warning}", file=sys.stderr)
        artifacts = _validate_cpp_output_metadata(staged_params)

        validation_summary: dict[str, Any] | None = None
        validation_requested = any(
            (
                params.observed_full_landslide_footprint_raster,
                params.observed_depositional_zone_raster,
                params.validation_domain_raster,
                params.output_validation_summary_csv,
                params.output_validation_summary_json,
            )
        )
        if validation_requested:
            core_result, context = _load_cpp_outputs(staged_params)
            validation_summary = run_validation_metrics(
                staged_params,
                context["fel"],
                core_result,
                context["source"],
            )
            write_validation_summaries(staged_params, validation_summary)

        summary = _make_summary(
            params=params,
            engine_stats=engine_stats,
            artifacts=artifacts,
            command=_official_command(cmd, bundle),
            config_path=config_path,
            executable_path=exe_path,
            returncode=returncode,
            engine_wall_seconds=elapsed,
            wrapper_elapsed_seconds=time.perf_counter() - wrapper_start,
            preflight_summary=preflight_summary,
            validation_summary=validation_summary,
        )
        _ensure_parent_dir(staged_params.output_summary_json)
        with open(staged_params.output_summary_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        if (
            FIXED_FAIL_ON_UNPROCESSED_CELLS
            and int(engine_stats.get("counts", {}).get("remaining_dependency_cells", 0)) > 0
        ):
            raise RuntimeError(
                "Remaining dependency cells exceed the configured allowance. "
                "remaining_dependency_cells="
                f"{int(engine_stats.get('counts', {}).get('remaining_dependency_cells', 0))}."
            )
        publish_staged_output_bundle(bundle)
        return summary
    finally:
        cleanup_staged_output_bundle(bundle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step 4 through the C++/MPI routing executable.")
    parser.add_argument("--config", help="Path to Step 4 JSON config.")
    parser.add_argument("--processes", "-n", type=int, default=8, help="MPI process count.")
    parser.add_argument("--exe", default=str(DEFAULT_EXE), help="Path to step4_deposition_zone_mpi.exe.")
    parser.add_argument(
        "--mpiexec",
        default=None,
        help=f"Path or command name for mpiexec. Resolution order: this argument, {MPIEXEC_ENV}, then PATH.",
    )
    parser.add_argument(
        "--taudem-dll-dir",
        default=None,
        help=(
            "Folder containing TauDEM runtime DLLs such as gdal.dll. "
            f"Resolution order: this argument, {TAUDEM_DLL_ENV}, the workspace runtime folder, then PATH."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved command without running.")
    parser.add_argument(
        "--check-environment",
        action="store_true",
        help="Check Python, osgeo, MPI, executable, and GDAL runtime availability without reading data.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.check_environment:
        errors: list[str] = []
        report: dict[str, Any] = {
            "python": sys.executable,
            "executable": str(Path(args.exe).expanduser().resolve()),
        }
        try:
            require_osgeo()
            report["osgeo"] = "available"
        except Exception as exc:
            errors.append(str(exc))
            report["osgeo"] = "missing"
        try:
            report["mpiexec"] = resolve_mpiexec(args.mpiexec)
        except Exception as exc:
            errors.append(str(exc))
        try:
            dll_dir = resolve_taudem_dll_dir(args.taudem_dll_dir)
            report["taudem_dll_dir"] = str(dll_dir)
            proj_dir = resolve_taudem_proj_dir(dll_dir)
            report["taudem_proj_dir"] = (
                str(proj_dir) if proj_dir is not None else "not explicitly resolved"
            )
        except Exception as exc:
            errors.append(str(exc))
        if not Path(args.exe).expanduser().resolve().is_file():
            errors.append(f"C++/MPI executable not found: {Path(args.exe).expanduser().resolve()}")
        report["errors"] = errors
        print(json.dumps(report, indent=2))
        return 1 if errors else 0
    if not args.config:
        raise SystemExit("--config is required unless --check-environment is used.")
    if args.processes < 1:
        raise SystemExit("--processes must be >= 1")
    summary = run_mpi_wrapper(args.config, args.processes, args.exe, args.dry_run, args.taudem_dll_dir, args.mpiexec)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

