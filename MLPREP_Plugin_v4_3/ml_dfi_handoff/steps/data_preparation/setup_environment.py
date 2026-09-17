#!/usr/bin/env python
"""Validate the frozen DP1 QGIS LTR/OSGeo4W environment."""

from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path

from launch_dp1 import _configured_osgeo4w_root, run_qgis


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
VENV = WORKSPACE_ROOT / ".venv-dp1"
VENV_PYTHON = VENV / "Scripts" / "python.exe"
REQUIRED_PYTHON = "3.12.10"
REQUIRED_PIP = "25.0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--osgeo4w-root",
        type=Path,
        help="Override the configured OSGeo4W root for this check.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate an existing DP1 environment without installing packages.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = (
        args.osgeo4w_root.expanduser().resolve()
        if args.osgeo4w_root
        else _configured_osgeo4w_root()
    )
    base_python = root / "apps" / "Python312" / "python.exe"
    if not base_python.is_file():
        raise RuntimeError(f"OSGeo4W Python was not found under {root}.")
    if not VENV_PYTHON.is_file():
        if args.check_only:
            raise RuntimeError(f"DP1 environment is missing: {VENV}")
        subprocess.run(
            [
                str(base_python),
                "-m",
                "venv",
                "--system-site-packages",
                str(VENV),
            ],
            check=True,
        )

    actual_python = subprocess.check_output(
        [str(VENV_PYTHON), "-c", "import platform; print(platform.python_version())"],
        text=True,
    ).strip()
    if actual_python != REQUIRED_PYTHON:
        raise RuntimeError(
            f"DP1 requires Python {REQUIRED_PYTHON}; found {actual_python}."
        )
    if not args.check_only:
        subprocess.run(
            [str(VENV_PYTHON), "-m", "pip", "install", f"pip=={REQUIRED_PIP}"],
            check=True,
        )
        subprocess.run(
            [
                str(VENV_PYTHON),
                "-m",
                "pip",
                "install",
                "-r",
                str(Path(__file__).resolve().parent / "requirements.lock.txt"),
            ],
            check=True,
        )
        subprocess.run([str(VENV_PYTHON), "-m", "pip", "check"], check=True)

    print(
        f"DP1 base runtime: QGIS/OSGeo4W Python {REQUIRED_PYTHON}; "
        f"setup host: {platform.platform()}"
    )
    return run_qgis(["--check-environment"], root)


if __name__ == "__main__":
    raise SystemExit(main())
