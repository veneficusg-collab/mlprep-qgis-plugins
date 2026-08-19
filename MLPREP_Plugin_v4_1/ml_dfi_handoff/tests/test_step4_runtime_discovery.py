from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STEP4_DIR = ROOT / "steps" / "step4_deposition_zone"


def load_wrapper():
    module_name = "portable_step4_runtime_wrapper"
    sys.path.insert(0, str(STEP4_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            STEP4_DIR / "step4_deposition_zone_mpi_wrapper.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(STEP4_DIR))


def test_mpi_launcher_can_be_resolved_by_command_name(monkeypatch):
    module = load_wrapper()
    monkeypatch.delenv(module.MPIEXEC_ENV, raising=False)
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda command: "resolved-mpiexec" if command == "portable-mpiexec" else None,
    )

    assert module.resolve_mpiexec("portable-mpiexec") == "resolved-mpiexec"


def test_taudem_runtime_and_sibling_proj_are_discovered(tmp_path, monkeypatch):
    module = load_wrapper()
    runtime_dir = tmp_path / "taudem" / "bin"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "gdal.dll").touch()
    proj_dir = runtime_dir.parent / "share" / "proj"
    proj_dir.mkdir(parents=True)
    (proj_dir / "proj.db").touch()
    for environment_name in (
        module.TAUDEM_DLL_ENV,
        module.TAUDEM_PROJ_ENV,
        "PROJ_DATA",
        "PROJ_LIB",
    ):
        monkeypatch.delenv(environment_name, raising=False)

    resolved_runtime = module.resolve_taudem_dll_dir(str(runtime_dir))

    assert resolved_runtime == runtime_dir.resolve()
    assert module.resolve_taudem_proj_dir(resolved_runtime) == proj_dir.resolve()


def test_invalid_explicit_proj_directory_fails_clearly(tmp_path, monkeypatch):
    module = load_wrapper()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    missing_proj = tmp_path / "missing-proj"
    monkeypatch.setenv(module.TAUDEM_PROJ_ENV, str(missing_proj))
    monkeypatch.delenv("PROJ_DATA", raising=False)
    monkeypatch.delenv("PROJ_LIB", raising=False)

    with pytest.raises(FileNotFoundError, match=module.TAUDEM_PROJ_ENV):
        module.resolve_taudem_proj_dir(runtime_dir)
