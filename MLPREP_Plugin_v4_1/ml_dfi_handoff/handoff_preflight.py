#!/usr/bin/env python
"""Read-only portability checks for the ML-DFI handoff folders."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


ROOT = Path(__file__).resolve().parent
STEP_DIRS = (
    "steps/data_preparation",
    "steps/step0_baseline_alpha_angle",
    "steps/step1_prepare_inventory_labels",
    "steps/step2_extract_pixel_samples",
    "steps/step3_train_ml_dfi_model",
    "steps/step4_deposition_zone",
    "steps/step5_post_depositional_spread_PDS",
    "steps/step6_landslide_damming_potential_LDP",
)
REQUIRED_FILES = (
    "environment.yml",
    "environment.osgeo4w.lock.txt",
    "ENVIRONMENT.md",
    "NATIVE_DEPENDENCIES.md",
    "requirements.txt",
    "requirements.lock.txt",
    "setup_environment.py",
    "setup_environment.ps1",
    "scripts/check_environment.py",
    "workflow_config.py",
    "run_workflow_step.ps1",
    "config/default_config.json",
    "config/local_config.example.json",
    "config/schema_notes.md",
    "steps/data_preparation/launch_dp1.py",
    "steps/data_preparation/setup_environment.py",
    "steps/step0_baseline_alpha_angle/run_step0.bat",
    "steps/step1_prepare_inventory_labels/run_step1.bat",
    "steps/step2_extract_pixel_samples/run_step2.bat",
    "steps/step3_train_ml_dfi_model/run_step3.bat",
    "steps/step3_train_ml_dfi_model/run_apply_model.bat",
    "steps/step4_deposition_zone/run_step4.bat",
    "steps/step5_post_depositional_spread_PDS/run_step5.bat",
    "steps/step6_landslide_damming_potential_LDP/run_step6.bat",
    "steps/step4_deposition_zone/cpp_mpi_port/build_manual/"
    "step4_deposition_zone_mpi.exe",
)
CENTRAL_CONFIGS = (
    "config/default_config.json",
    "config/local_config.example.json",
)
WINDOWS_DRIVE_LITERAL = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
USER_HOME_LITERAL = re.compile(
    r"(?<![A-Za-z0-9_])/(?:Users|home)/[^/\s\"'`<>]+",
    re.IGNORECASE,
)
LEGACY_WORKSPACE_LITERAL = re.compile(r"\bCodes_V1\b", re.IGNORECASE)
TEXT_SUFFIXES = {
    ".bat",
    ".cfg",
    ".cmd",
    ".cpp",
    ".cxx",
    ".ini",
    ".h",
    ".hpp",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
}
EXCLUDED_DIRECTORY_NAMES = {
    ".agents",
    ".codex",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".runtime",
    "Sample Data",
    "__pycache__",
    "analysis_outputs",
    "archive",
    "build",
    "build_manual",
    "dist",
    "node_modules",
    "third_party",
}
EXCLUDED_FILE_NAMES = {"local_config.json"}
PACKAGE_EXCLUDED_DIRECTORY_NAMES = {
    ".agents",
    ".codex",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".runtime",
    "Sample Data",
    "__pycache__",
    "analysis_outputs",
    "archive",
    "dist",
    "node_modules",
    "outputs",
}
PACKAGE_EXCLUDED_FILE_NAMES = {
    ".env",
    "local_config.json",
    "symbology-style.db",
}
PACKAGE_FORBIDDEN_ENDINGS = (
    ".7z",
    ".aux.xml",
    ".bak",
    ".cpg",
    ".dbf",
    ".gpkg",
    ".joblib",
    ".key",
    ".log",
    ".onnx",
    ".ovr",
    ".p12",
    ".pem",
    ".pickle",
    ".pkl",
    ".prof",
    ".pfx",
    ".pyc",
    ".rar",
    ".shp",
    ".shx",
    ".tfw",
    ".tif",
    ".tiff",
    ".tmp",
    ".zip",
)
PACKAGE_SECRET_NAME_PREFIXES = (
    "credentials",
    "id_ed25519",
    "id_rsa",
    "secret",
)
PACKAGE_ALLOWED_EXECUTABLES = {
    PurePosixPath(
        "steps/step4_deposition_zone/cpp_mpi_port/build_manual/"
        "step4_deposition_zone_mpi.exe"
    )
}


def iter_values(value: Any, key_path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from iter_values(child, (*key_path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_values(child, (*key_path, str(index)))
    else:
        yield key_path, value


def is_input_path(key_path: tuple[str, ...], value: str) -> bool:
    del value
    key = key_path[-1].lower() if key_path else ""
    if (
        key.startswith("output")
        or key.endswith(("_out", "_format", "_mode", "_method", "_field"))
    ):
        return False
    return any(
        token in key
        for token in (
            "input",
            "path",
            "raster",
            "polygon",
            "sample_table",
            "model_file",
            "predictor_config",
            "dem",
        )
    )


def is_absolute_path(value: str) -> bool:
    """Recognize Windows and POSIX absolute paths on either host OS."""
    return (
        PureWindowsPath(value).is_absolute()
        or PurePosixPath(value).is_absolute()
    )


def is_excluded(file_path: Path) -> bool:
    relative_parts = file_path.relative_to(ROOT).parts
    return file_path.name in EXCLUDED_FILE_NAMES or any(
        part in EXCLUDED_DIRECTORY_NAMES or part.startswith(".venv")
        for part in relative_parts[:-1]
    )


def package_exclusion_reason(relative_path: str | Path) -> str | None:
    """Return why a repository file must not enter a handoff archive."""
    normalized = str(relative_path).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    path = PurePosixPath(normalized)
    parts = path.parts
    lower_name = path.name.lower()

    if any(part.startswith(".venv") for part in parts):
        return "virtual environment"
    excluded_directories = {
        directory.lower() for directory in PACKAGE_EXCLUDED_DIRECTORY_NAMES
    }
    if any(part.lower() in excluded_directories for part in parts[:-1]):
        return "excluded workspace directory"
    if lower_name in PACKAGE_EXCLUDED_FILE_NAMES or lower_name.startswith(".env."):
        return "local or sensitive file"
    if lower_name.startswith(PACKAGE_SECRET_NAME_PREFIXES):
        return "credential-like filename"
    if lower_name.endswith(PACKAGE_FORBIDDEN_ENDINGS):
        return "generated, data, model, log, or archive artifact"
    if lower_name.endswith(".exe") and path not in PACKAGE_ALLOWED_EXECUTABLES:
        return "unreviewed executable"
    return None


def git_handoff_candidates() -> list[Path]:
    """List tracked and non-ignored untracked files from the working tree."""
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [
        Path(item.decode("utf-8"))
        for item in completed.stdout.split(b"\0")
        if item
    ]


def iter_active_text_files():
    for file_path in ROOT.rglob("*"):
        if (
            file_path.is_file()
            and file_path.suffix.lower() in TEXT_SUFFIXES
            and not is_excluded(file_path)
        ):
            yield file_path


def find_machine_specific_text(text: str) -> list[tuple[int, str]]:
    findings: list[tuple[int, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if (
            WINDOWS_DRIVE_LITERAL.search(line)
            or USER_HOME_LITERAL.search(line)
            or LEGACY_WORKSPACE_LITERAL.search(line)
        ):
            findings.append((line_number, line.strip()))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Also report configured default input files that do not exist.",
    )
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    python_count = 0
    config_count = 0
    package_eligible_count = 0
    package_excluded_count = 0

    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"Missing required handoff file: {relative}")

    for relative in CENTRAL_CONFIGS:
        config = ROOT / relative
        try:
            payload = json.loads(config.read_text(encoding="utf-8"))
            config_count += 1
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid JSON in {relative}: {exc}")
            continue
        for key_path, value in iter_values(payload):
            if isinstance(value, str) and value.strip() and is_absolute_path(value):
                errors.append(
                    f"Machine-bound absolute path in {relative} "
                    f"at {'.'.join(key_path)}: {value}"
                )

    for directory_name in STEP_DIRS:
        directory = ROOT / directory_name
        if not directory.is_dir():
            errors.append(f"Missing handoff folder: {directory_name}")
            continue

        for source in directory.rglob("*.py"):
            if "__pycache__" in source.parts:
                continue
            try:
                compile(source.read_bytes(), str(source), "exec")
                python_count += 1
            except (OSError, SyntaxError) as exc:
                errors.append(
                    f"Python syntax/read failure in {source.relative_to(ROOT)}: {exc}"
                )

        for config in directory.glob("config*.json"):
            try:
                payload = json.loads(config.read_text(encoding="utf-8"))
                config_count += 1
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"Invalid JSON in {config.relative_to(ROOT)}: {exc}")
                continue

            for key_path, value in iter_values(payload):
                if not isinstance(value, str) or not value.strip():
                    continue
                if is_absolute_path(value):
                    errors.append(
                        f"Machine-bound absolute path in {config.relative_to(ROOT)} "
                        f"at {'.'.join(key_path)}: {value}"
                    )
                if args.check_inputs and config.name == "config.default.json":
                    if is_input_path(key_path, value):
                        candidate = Path(value)
                        if not candidate.is_absolute():
                            candidate = (config.parent / candidate).resolve()
                        if not candidate.exists():
                            warnings.append(
                                f"Configured input is absent: "
                                f"{config.relative_to(ROOT)} {'.'.join(key_path)} "
                                f"-> {candidate}"
                            )

    for file_path in iter_active_text_files():
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            errors.append(f"Cannot read {file_path.relative_to(ROOT)}: {exc}")
            continue
        for line_number, line in find_machine_specific_text(text):
            errors.append(
                "Machine/legacy path reference in active file: "
                f"{file_path.relative_to(ROOT)}:{line_number}: {line}"
            )

    if (ROOT / ".git").exists():
        try:
            candidates = git_handoff_candidates()
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"Cannot inspect Git handoff candidates: {exc}")
        else:
            for candidate in candidates:
                if not (ROOT / candidate).is_file():
                    continue
                if package_exclusion_reason(candidate):
                    package_excluded_count += 1
                else:
                    package_eligible_count += 1

    print(f"Checked {len(STEP_DIRS)} handoff folders.")
    print(f"Parsed {config_count} JSON configs and compiled {python_count} Python files.")
    if package_eligible_count or package_excluded_count:
        print(
            "Package policy found "
            f"{package_eligible_count} eligible and "
            f"{package_excluded_count} excluded working-tree files."
        )
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"Handoff preflight FAILED with {len(errors)} error(s).")
        return 1
    print(f"Handoff preflight PASSED with {len(warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
