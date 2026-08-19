#!/usr/bin/env python
"""Launch DP1 through the configured QGIS LTR/OSGeo4W Python runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


STEP_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = STEP_ROOT.parents[1]


def _configured_osgeo4w_root() -> Path:
    environment_value = os.environ.get("OSGEO4W_ROOT", "").strip()
    if environment_value:
        return Path(environment_value).expanduser().resolve()

    local_config = WORKSPACE_ROOT / "config" / "local_config.json"
    if local_config.is_file():
        payload: Any = json.loads(local_config.read_text(encoding="utf-8"))
        value = str(payload.get("runtime", {}).get("osgeo4w_root", "")).strip()
        if value:
            path = Path(value).expanduser()
            return (
                path.resolve()
                if path.is_absolute()
                else (local_config.parent / path).resolve()
            )
    raise RuntimeError(
        "Configure runtime.osgeo4w_root in config/local_config.json or set "
        "OSGEO4W_ROOT."
    )


def run_qgis(step_arguments: list[str], osgeo4w_root: Path | None = None) -> int:
    root = osgeo4w_root or _configured_osgeo4w_root()
    venv_python = WORKSPACE_ROOT / ".venv-dp1" / "Scripts" / "python.exe"
    if not venv_python.is_file():
        raise RuntimeError(
            "The dedicated DP1 environment is missing. Run "
            "steps/data_preparation/setup_environment.py first."
        )

    qgis_prefix = root / "apps" / "qgis-ltr"
    python_root = root / "apps" / "Python312"
    environment = os.environ.copy()
    environment["OSGEO4W_ROOT"] = str(root)
    environment["PATH"] = os.pathsep.join(
        str(path)
        for path in (
            qgis_prefix / "bin",
            root / "apps" / "qt5" / "bin",
            python_root / "Scripts",
            root / "bin",
            Path(environment.get("PATH", "")),
        )
        if str(path)
    )
    environment["QGIS_PREFIX_PATH"] = qgis_prefix.as_posix()
    environment["GDAL_DATA"] = str(root / "apps" / "gdal" / "share" / "gdal")
    environment["GDAL_DRIVER_PATH"] = str(
        root / "apps" / "gdal" / "lib" / "gdalplugins"
    )
    environment["PROJ_DATA"] = str(root / "share" / "proj")
    environment["QT_PLUGIN_PATH"] = os.pathsep.join(
        [
            str(qgis_prefix / "qtplugins"),
            str(root / "apps" / "qt5" / "plugins"),
        ]
    )
    environment["PYTHONHOME"] = str(python_root)
    environment["PYTHONPATH"] = str(qgis_prefix / "python")
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["GDAL_FILENAME_IS_UTF8"] = "YES"
    environment["VSI_CACHE"] = "TRUE"
    environment["VSI_CACHE_SIZE"] = "1000000"
    completed = subprocess.run(
        [str(venv_python), str(STEP_ROOT / "run_dp1.py"), *step_arguments],
        cwd=WORKSPACE_ROOT,
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def main() -> int:
    arguments = sys.argv[1:]
    if not arguments:
        completed = subprocess.run(
            [
                sys.executable,
                str(WORKSPACE_ROOT / "workflow_config.py"),
                "run",
                "data_preparation",
            ],
            cwd=WORKSPACE_ROOT,
            check=False,
        )
        return int(completed.returncode)
    return run_qgis(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
