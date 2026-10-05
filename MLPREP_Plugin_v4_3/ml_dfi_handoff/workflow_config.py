#!/usr/bin/env python
"""Load, validate, inspect, or run one workflow step from centralized config."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_MASTER_CONFIG = ROOT / "config" / "default_config.json"
LOCAL_CONFIG_NAME = "local_config.json"
RUNTIME_DIRECTORY_NAME = ".runtime"
REPLACE_OBJECT_KEY = "$replace"


@dataclass(frozen=True)
class StepSpec:
    launcher: Path
    config_argument: str


@dataclass(frozen=True)
class LoadedWorkflowConfig:
    default_path: Path
    local_path: Path | None
    payload: dict[str, Any]


STEP_SPECS = {
    "data_preparation": StepSpec(
        ROOT / "steps" / "data_preparation" / "launch_dp1.py",
        "--config",
    ),
    "step0_baseline_alpha_angle": StepSpec(
        ROOT / "steps" / "step0_baseline_alpha_angle" / "run_step0.bat",
        "--config",
    ),
    "step1_prepare_inventory_labels": StepSpec(
        ROOT / "steps" / "step1_prepare_inventory_labels" / "run_step1.bat",
        "--config-json",
    ),
    "step2_extract_pixel_samples": StepSpec(
        ROOT / "steps" / "step2_extract_pixel_samples" / "run_step2.bat",
        "--config-json",
    ),
    "step3_train_ml_dfi_model": StepSpec(
        ROOT / "steps" / "step3_train_ml_dfi_model" / "run_step3.bat",
        "--config-json",
    ),
    "step3_model_application": StepSpec(
        ROOT / "steps" / "step3_train_ml_dfi_model" / "run_apply_model.bat",
        "--config-json",
    ),
    "step4_deposition_zone": StepSpec(
        ROOT / "steps" / "step4_deposition_zone" / "run_step4.bat",
        "--config",
    ),
    "step5_post_depositional_spread_PDS": StepSpec(
        ROOT / "steps" / "step5_post_depositional_spread_PDS" / "run_step5.bat",
        "--config",
    ),
    "step6_landslide_damming_potential_LDP": StepSpec(
        ROOT
        / "steps"
        / "step6_landslide_damming_potential_LDP"
        / "run_step6.bat",
        "--config",
    ),
}

STEP_ALIASES = {
    "dp1": "data_preparation",
    "data_preparation": "data_preparation",
    "step0": "step0_baseline_alpha_angle",
    "step1": "step1_prepare_inventory_labels",
    "step2": "step2_extract_pixel_samples",
    "step3": "step3_train_ml_dfi_model",
    "apply_model": "step3_model_application",
    "step3_application": "step3_model_application",
    "step4": "step4_deposition_zone",
    "step5": "step5_post_depositional_spread_PDS",
    "step6": "step6_landslide_damming_potential_LDP",
}

PATH_PATTERNS: dict[str, tuple[tuple[str, ...], ...]] = {
    "data_preparation": (
        ("dp1", "landslide_polygons"),
        ("dp1", "dem"),
        ("dp1", "output_dir"),
    ),
    "step0_baseline_alpha_angle": (
        ("slope",),
        ("output_alpha",),
        ("output_class",),
        ("metadata_out",),
    ),
    "step1_prepare_inventory_labels": (
        ("landslide_polygon_path",),
        ("dem_path",),
        ("output_dir",),
    ),
    "step2_extract_pixel_samples": (
        ("extract_pixel_samples", "label_raster_path"),
        ("extract_pixel_samples", "predictor_rasters", "*"),
        ("extract_pixel_samples", "group_rasters", "*"),
        ("extract_pixel_samples", "output_dir"),
    ),
    "step3_train_ml_dfi_model": (
        ("input_sample_table",),
        ("output_dir",),
    ),
    "step3_model_application": (
        ("apply_ml_dfi_model", "model_path"),
        ("apply_ml_dfi_model", "metadata_path"),
        ("apply_ml_dfi_model", "predictor_config_json"),
        ("apply_ml_dfi_model", "reference_raster_path"),
        ("apply_ml_dfi_model", "output_dir"),
    ),
    "step4_deposition_zone": (
        ("dem_fel_raster",),
        ("dinf_flow_raster",),
        ("source_raster",),
        ("dfi_raster",),
        ("alpha_raster",),
        ("output_dynamic_alpha",),
        ("output_beta_angle",),
        ("output_dfs",),
        ("output_mask",),
        ("output_depositional_mask",),
        ("output_summary_json",),
        ("observed_full_landslide_footprint_raster",),
        ("observed_depositional_zone_raster",),
        ("validation_domain_raster",),
        ("output_validation_summary_csv",),
        ("output_validation_summary_json",),
    ),
    "step5_post_depositional_spread_PDS": (
        ("runout_mask_raster",),
        ("dfi_raster",),
        ("dem_raster",),
        ("source_mask_raster",),
        ("stream_mask_raster",),
        ("source_contributing_area_raster",),
        ("output_spread_mask",),
        ("output_combined_mask",),
        ("output_seed_mask",),
        ("output_cleaned_runout_mask",),
        ("output_removed_runout_mask",),
        ("output_spread_mask_shapefile",),
        ("output_combined_mask_shapefile",),
        ("output_summary_json",),
        ("output_min_elev_diff",),
        ("output_source_row",),
        ("output_source_col",),
    ),
    "step6_landslide_damming_potential_LDP": (
        ("dem_fel_raster",),
        ("dinf_flow_raster",),
        ("runout_mask_raster",),
        ("stream_raster",),
        ("slope_raster",),
        ("stream_vector",),
        ("source_raster",),
        ("depositional_mask_raster",),
        ("output_stream_intersection_mask",),
        ("output_inflow_angle",),
        ("output_stream_angle",),
        ("output_runout_approach_angle",),
        ("output_channel_slope",),
        ("output_candidate_damming_potential_class",),
        ("output_damming_potential_class",),
        ("output_damming_potential_score",),
        ("output_height_above_stream_raster",),
        ("output_valley_floor_mask_raster",),
        ("output_raw_valley_width_raster",),
        ("output_corrected_valley_width_raster",),
        ("output_reference_volume_domain_class",),
        ("output_reference_volume_combined_class",),
        ("output_reference_volume_combined_score",),
        ("output_candidate_points",),
        ("output_summary_json",),
    ),
}

WORKER_PATHS = {
    "data_preparation": ("dp1", "parallel", "max_workers"),
    "step1_prepare_inventory_labels": ("max_workers",),
}

RUNTIME_ENVIRONMENT_KEYS = {
    "python_executable": "MLDFI_PYTHON",
    "osgeo4w_root": "OSGEO4W_ROOT",
    "mpiexec": "MPIEXEC",
    "taudem_dll_dir": "TAUDEM_DLL_DIR",
    "taudem_proj_dir": "TAUDEM_PROJ_DIR",
}

PATH_RUNTIME_KEYS = {
    "osgeo4w_root",
    "taudem_dll_dir",
    "taudem_proj_dir",
    "step4_executable",
}


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input."""
    if override.get(REPLACE_OBJECT_KEY) is True:
        return {
            key: copy.deepcopy(value)
            for key, value in override.items()
            if key != REPLACE_OBJECT_KEY
        }
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def canonical_step_name(value: str) -> str:
    normalized = value.strip()
    canonical = STEP_ALIASES.get(normalized, normalized)
    if canonical not in STEP_SPECS:
        valid = ", ".join(STEP_SPECS)
        raise ValueError(f"Unknown workflow step {value!r}. Valid steps: {valid}")
    return canonical


def _validate_master_payload(payload: dict[str, Any], label: str) -> None:
    allowed = {"schema_version", "runtime", "steps", "step_overrides"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unsupported keys in {label}: {', '.join(unknown)}")
    if payload.get("schema_version") != 1:
        raise ValueError(f"{label} must declare schema_version = 1.")
    for key in ("runtime", "steps", "step_overrides"):
        if not isinstance(payload.get(key, {}), dict):
            raise ValueError(f"{label}.{key} must be a JSON object.")


def _validate_local_payload(payload: dict[str, Any], label: str) -> None:
    allowed = {"schema_version", "runtime", "step_overrides"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            f"Unsupported keys in {label}: {', '.join(unknown)}. "
            "Local config cannot replace the shared step registry."
        )
    if payload.get("schema_version", 1) != 1:
        raise ValueError(f"{label} must declare schema_version = 1.")
    for key in ("runtime", "step_overrides"):
        if not isinstance(payload.get(key, {}), dict):
            raise ValueError(f"{label}.{key} must be a JSON object.")


def _validate_workers(runtime: dict[str, Any]) -> None:
    workers = runtime.get("workers", {})
    if not isinstance(workers, dict):
        raise ValueError("runtime.workers must be a JSON object.")
    for raw_step, value in workers.items():
        step = str(raw_step)
        if step not in STEP_SPECS:
            raise ValueError(
                f"runtime.workers must use an exact centralized step name, not {step!r}."
            )
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"runtime.workers.{raw_step} must be null or an integer >= 1.")
        if step not in {
            "data_preparation",
            "step1_prepare_inventory_labels",
            "step3_train_ml_dfi_model",
            "step4_deposition_zone",
        }:
            raise ValueError(f"Worker override is not supported for {step}.")


def _validate_runtime(runtime: dict[str, Any]) -> None:
    allowed = {*RUNTIME_ENVIRONMENT_KEYS, "step4_executable", "workers"}
    unknown = sorted(set(runtime) - allowed)
    if unknown:
        raise ValueError(f"Unsupported runtime settings: {', '.join(unknown)}")
    for key in allowed - {"workers"}:
        value = runtime.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"runtime.{key} must be a string or null.")
    _validate_workers(runtime)


def load_workflow_config(
    default_path: Path = DEFAULT_MASTER_CONFIG,
    local_path: Path | None = None,
    *,
    use_auto_local: bool = True,
) -> LoadedWorkflowConfig:
    default_path = default_path.expanduser().resolve()
    shared = read_json_object(default_path, "Shared workflow config")
    _validate_master_payload(shared, "shared workflow config")

    resolved_local = local_path.expanduser().resolve() if local_path else None
    if resolved_local is None and use_auto_local:
        candidate = default_path.parent / LOCAL_CONFIG_NAME
        if candidate.is_file():
            resolved_local = candidate.resolve()

    payload = copy.deepcopy(shared)
    if resolved_local is not None:
        local = read_json_object(resolved_local, "Local workflow config")
        _validate_local_payload(local, "local workflow config")
        payload["runtime"] = deep_merge(
            dict(payload.get("runtime", {})),
            dict(local.get("runtime", {})),
        )
        payload["step_overrides"] = deep_merge(
            dict(payload.get("step_overrides", {})),
            dict(local.get("step_overrides", {})),
        )

    unknown_steps = sorted(set(payload["steps"]) - set(STEP_SPECS))
    if unknown_steps:
        raise ValueError(f"Unknown steps in shared registry: {', '.join(unknown_steps)}")
    missing_steps = sorted(set(STEP_SPECS) - set(payload["steps"]))
    if missing_steps:
        raise ValueError(f"Missing steps in shared registry: {', '.join(missing_steps)}")
    unknown_overrides = sorted(set(payload["step_overrides"]) - set(STEP_SPECS))
    if unknown_overrides:
        raise ValueError(f"Unknown step override sections: {', '.join(unknown_overrides)}")
    _validate_runtime(dict(payload["runtime"]))

    return LoadedWorkflowConfig(default_path, resolved_local, payload)


def _resolve_path_value(root: Path, value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    raw = value.strip()
    base, separator, suffix = raw.partition("|")
    path = Path(base).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    return f"{resolved}|{suffix}" if separator else str(resolved)


def _apply_path_pattern(
    node: Any,
    pattern: tuple[str, ...],
    root: Path,
) -> None:
    if not pattern or not isinstance(node, dict):
        return
    key = pattern[0]
    if key == "*":
        if len(pattern) == 1:
            for child_key, value in list(node.items()):
                node[child_key] = _resolve_path_value(root, value)
        else:
            for value in node.values():
                _apply_path_pattern(value, pattern[1:], root)
        return
    if key not in node:
        return
    if len(pattern) == 1:
        node[key] = _resolve_path_value(root, node[key])
    else:
        _apply_path_pattern(node[key], pattern[1:], root)


def resolve_step_paths(
    step: str,
    payload: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    resolved = copy.deepcopy(payload)
    for pattern in PATH_PATTERNS[step]:
        _apply_path_pattern(resolved, pattern, root)
    return resolved


def _set_nested_value(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = payload
    for key in path[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot apply runtime override at {'.'.join(path)}.")
        node = child
    node[path[-1]] = value


def effective_step_config(
    workflow: LoadedWorkflowConfig,
    step_name: str,
) -> tuple[dict[str, Any], Path]:
    step = canonical_step_name(step_name)
    raw_base_path = workflow.payload["steps"][step]
    if not isinstance(raw_base_path, str) or not raw_base_path.strip():
        raise ValueError(f"steps.{step} must identify a base JSON config.")
    base_path = Path(raw_base_path).expanduser()
    if not base_path.is_absolute():
        base_path = workflow.default_path.parent / base_path
    base_path = base_path.resolve()

    base = resolve_step_paths(
        step,
        read_json_object(base_path, f"Base config for {step}"),
        base_path.parent,
    )
    override = workflow.payload["step_overrides"].get(step, {})
    if not isinstance(override, dict):
        raise ValueError(f"step_overrides.{step} must be a JSON object.")
    override_root = (
        workflow.local_path.parent
        if workflow.local_path is not None
        else workflow.default_path.parent
    )
    effective = deep_merge(
        base,
        resolve_step_paths(step, override, override_root),
    )

    workers = workflow.payload["runtime"].get("workers", {})
    worker_value = workers.get(step)
    if worker_value is not None:
        if step in WORKER_PATHS:
            _set_nested_value(effective, WORKER_PATHS[step], int(worker_value))
        elif step == "step3_train_ml_dfi_model":
            _set_nested_value(
                effective,
                ("xgboost_params", "n_jobs"),
                int(worker_value),
            )
            _set_nested_value(
                effective,
                ("random_forest_params", "n_jobs"),
                int(worker_value),
            )
    if step == "step3_model_application":
        centralized_predictor_config = (
            workflow.default_path.parent
            / RUNTIME_DIRECTORY_NAME
            / "step2_extract_pixel_samples.effective.json"
        )
        _set_nested_value(
            effective,
            ("apply_ml_dfi_model", "predictor_config_json"),
            str(centralized_predictor_config.resolve()),
        )
    return effective, base_path


def write_effective_config(
    workflow: LoadedWorkflowConfig,
    step_name: str,
) -> tuple[Path, Path]:
    step = canonical_step_name(step_name)
    if step == "step3_model_application":
        write_effective_config(workflow, "step2_extract_pixel_samples")
    effective, base_path = effective_step_config(workflow, step)
    runtime_dir = workflow.default_path.parent / RUNTIME_DIRECTORY_NAME
    runtime_dir.mkdir(parents=True, exist_ok=True)
    output_path = runtime_dir / f"{step}.effective.json"
    temporary_path = runtime_dir / f".{step}.effective.tmp"
    temporary_path.write_text(
        json.dumps(effective, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path, base_path


def _resolve_runtime_value(key: str, value: Any, root: Path) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    if key in PATH_RUNTIME_KEYS or any(separator in raw for separator in ("/", "\\")):
        path = Path(raw).expanduser()
        return str(path.resolve() if path.is_absolute() else (root / path).resolve())
    return raw


def build_runtime_environment(workflow: LoadedWorkflowConfig) -> dict[str, str]:
    environment = os.environ.copy()
    runtime = dict(workflow.payload["runtime"])
    runtime_root = (
        workflow.local_path.parent
        if workflow.local_path is not None
        else workflow.default_path.parent
    )
    for config_key, environment_key in RUNTIME_ENVIRONMENT_KEYS.items():
        value = _resolve_runtime_value(config_key, runtime.get(config_key), runtime_root)
        if value:
            environment[environment_key] = value
    return environment


def _has_option(arguments: list[str], *options: str) -> bool:
    return any(
        argument in options
        or any(argument.startswith(f"{option}=") for option in options)
        for argument in arguments
    )


def build_step_invocation(
    workflow: LoadedWorkflowConfig,
    step_name: str,
    effective_config_path: Path,
    extra_arguments: list[str],
) -> tuple[str, dict[str, str]]:
    step = canonical_step_name(step_name)
    spec = STEP_SPECS[step]
    if _has_option(extra_arguments, "--config", "--config-json"):
        raise ValueError(
            "Do not pass a per-step config argument through the master runner. "
            "Use step_overrides in local_config.json."
        )
    batch_arguments = [
        str(spec.launcher),
        spec.config_argument,
        str(effective_config_path),
    ]
    runtime = dict(workflow.payload["runtime"])
    runtime_root = (
        workflow.local_path.parent
        if workflow.local_path is not None
        else workflow.default_path.parent
    )
    if step == "step4_deposition_zone":
        processes = runtime.get("workers", {}).get(step)
        if processes is not None and not _has_option(extra_arguments, "--processes", "-n"):
            batch_arguments.extend(["--processes", str(processes)])
        executable = _resolve_runtime_value(
            "step4_executable",
            runtime.get("step4_executable"),
            runtime_root,
        )
        if executable and not _has_option(extra_arguments, "--exe"):
            batch_arguments.extend(["--exe", executable])
    batch_arguments.extend(extra_arguments)

    if spec.launcher.suffix.lower() == ".py":
        command = subprocess.list2cmdline(
            [
                sys.executable,
                *batch_arguments,
            ]
        )
        return command, build_runtime_environment(workflow)

    command_processor = os.environ.get("COMSPEC") or shutil.which("cmd.exe")
    if not command_processor:
        raise RuntimeError("Windows command processor cmd.exe was not found.")
    command = (
        f"{subprocess.list2cmdline([command_processor])} /d /s /c "
        f"call {subprocess.list2cmdline(batch_arguments)}"
    )
    return command, build_runtime_environment(workflow)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_MASTER_CONFIG,
        help="Shared master config. Defaults to config/default_config.json.",
    )
    parser.add_argument(
        "--local-config",
        type=Path,
        help="Optional local override. Defaults to config/local_config.json when present.",
    )
    parser.add_argument(
        "--no-local",
        action="store_true",
        help="Ignore an automatically discovered local_config.json.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List centralized workflow step names.")

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate master/local config and referenced base configs.",
    )
    validate_parser.add_argument("step", nargs="?", help="Optional step or alias.")

    show_parser = subparsers.add_parser(
        "show",
        help="Print one fully merged effective step config.",
    )
    show_parser.add_argument("step", help="Step name or alias.")

    run_parser = subparsers.add_parser(
        "run",
        help="Generate the effective config and launch one step.",
    )
    run_parser.add_argument("step", help="Step name or alias.")
    run_parser.add_argument(
        "extra_arguments",
        nargs=argparse.REMAINDER,
        help="Additional non-config arguments passed to the step launcher.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.no_local and args.local_config is not None:
        raise SystemExit("--no-local cannot be combined with --local-config.")
    workflow = load_workflow_config(
        args.config,
        args.local_config,
        use_auto_local=not args.no_local,
    )

    if args.command == "list":
        for step in STEP_SPECS:
            print(step)
        return 0

    if args.command == "validate":
        selected = (
            [canonical_step_name(args.step)]
            if args.step
            else list(STEP_SPECS)
        )
        for step in selected:
            _, base_path = effective_step_config(workflow, step)
            print(f"OK {step}: {base_path}")
        print(
            "Local override: "
            + (str(workflow.local_path) if workflow.local_path else "not present")
        )
        return 0

    if args.command == "show":
        step = canonical_step_name(args.step)
        effective, _ = effective_step_config(workflow, step)
        print(json.dumps(effective, indent=2))
        return 0

    if args.command == "run":
        step = canonical_step_name(args.step)
        extra_arguments = list(args.extra_arguments)
        if extra_arguments[:1] == ["--"]:
            extra_arguments = extra_arguments[1:]
        effective_path, base_path = write_effective_config(workflow, step)
        command, environment = build_step_invocation(
            workflow,
            step,
            effective_path,
            extra_arguments,
        )
        print(f"Shared config: {workflow.default_path}")
        print(
            "Local override: "
            + (str(workflow.local_path) if workflow.local_path else "not present")
        )
        print(f"Base step config: {base_path}")
        print(f"Effective config: {effective_path}")
        print(f"Launching: {step}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        return int(completed.returncode)

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
