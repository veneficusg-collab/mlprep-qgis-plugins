#!/usr/bin/env python
"""Create the frozen standard ML-DFI Python environment."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / "Scripts" / "python.exe"
REQUIRED_PYTHON = "3.13.12"
REQUIRED_PIP = "25.3"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def base_python(explicit: str) -> list[str]:
    if explicit:
        return [explicit]
    if shutil.which("py"):
        return ["py", "-3.13"]
    if shutil.which("python"):
        return ["python"]
    raise RuntimeError(
        f"Python was not found. Install 64-bit Python {REQUIRED_PYTHON} "
        "or pass --python-executable."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dependency-set",
        choices=("lock", "minimum"),
        default="lock",
    )
    parser.add_argument("--python-executable", default="")
    parser.add_argument("--force-reinstall", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not VENV_PYTHON.is_file():
        run([*base_python(args.python_executable), "-m", "venv", str(VENV)])

    actual_python = subprocess.check_output(
        [str(VENV_PYTHON), "-c", "import platform; print(platform.python_version())"],
        text=True,
    ).strip()
    if actual_python != REQUIRED_PYTHON:
        raise RuntimeError(
            f"The standard environment requires Python {REQUIRED_PYTHON}; "
            f"found {actual_python} in {VENV}."
        )

    requirements = ROOT / (
        "requirements.lock.txt"
        if args.dependency_set == "lock"
        else "requirements.txt"
    )
    run([str(VENV_PYTHON), "-m", "pip", "install", f"pip=={REQUIRED_PIP}"])
    install = [
        str(VENV_PYTHON),
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements),
    ]
    if args.force_reinstall:
        install.append("--force-reinstall")
    run(install)
    run([str(VENV_PYTHON), "-m", "pip", "check"])
    run(
        [
            str(VENV_PYTHON),
            "-c",
            (
                "import geopandas, matplotlib, numba, numpy, optuna, pandas, "
                "rasterio, sklearn, shapely, xgboost; "
                "print('Standard ML-DFI environment is ready.')"
            ),
        ]
    )
    print(f"Environment: {VENV}")
    print(f"Dependencies: {requirements}")
    print(f"Host used to launch setup: {platform.platform()} / {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

