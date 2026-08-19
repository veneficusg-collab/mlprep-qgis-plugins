# ML-DFI Handoff Guide

## Handoff Scope

Transfer these workflow folders together:

```text
steps/data_preparation
steps/step0_baseline_alpha_angle
steps/step1_prepare_inventory_labels
steps/step2_extract_pixel_samples
steps/step3_train_ml_dfi_model
steps/step4_deposition_zone
steps/step5_post_depositional_spread_PDS
steps/step6_landslide_damming_potential_LDP
```

Also transfer `config/`, `workflow_config.py`, `run_workflow_step.ps1`, the root
requirements files, `setup_environment.ps1`, and this guide. Do not transfer
`config/local_config.json`, `config/.runtime/`, virtual environments, Python
caches, test caches, archives, or previous run outputs. They are
machine-specific, exploratory, or reproducible.

Create the deliverable with the repository packager after committing reviewed
changes:

```powershell
python .\scripts\create_handoff_package.py --overwrite
```

The packager runs the preflight first, includes only tracked or non-ignored
source files, applies a denylist for datasets, models, logs, local settings,
credentials, archives, and unreviewed executables, and writes
`PACKAGE_MANIFEST.txt` with SHA256 checksums into the ZIP. It follows the
handoff scope above, including the centrally configured Step 6 source.

## Supported Platform

The tested platform is Windows 11 Pro 25H2 x64, build 26200.8875. Step 4 uses
a Windows x64 C++/Microsoft MPI executable. Exact Python, QGIS/OSGeo4W, GDAL,
PROJ, TauDEM, MPI, and Visual C++ runtime versions are recorded in
`ENVIRONMENT.md` and `NATIVE_DEPENDENCIES.md`.

## Folder Layout

The recipient may place the workspace anywhere. Active JSON paths are resolved
relative to the JSON file containing them, not relative to the terminal's
working directory. The supplied defaults expect this optional sibling layout:

```text
workspace/
|-- config/
|-- Sample Data/
|-- steps/
|   |-- data_preparation/
|   |-- step0_baseline_alpha_angle/
|   |-- step1_prepare_inventory_labels/
|   |-- step2_extract_pixel_samples/
|   |-- step3_train_ml_dfi_model/
|   |-- step4_deposition_zone/
|   |-- step5_post_depositional_spread_PDS/
|   `-- step6_landslide_damming_potential_LDP/
|-- scripts/
|-- tests/
`-- workflow_config.py
```

The recipient copies `config/local_config.example.json` to
`config/local_config.json` and edits that one file. Relative study paths resolve
from `config/`. Absolute paths are accepted only in the ignored local file when
needed. Output directories are created by the steps; input files are never
modified.

## Environment Setup

Install 64-bit Python 3.13.12, then create the standard Steps 0-3 environment:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_environment.ps1
```

The lock file is used by default for exact recreation. Use
`-DependencySet minimum` to install the shorter direct dependency list.
Conda/Mamba users may instead run `conda env create -f environment.yml` from
the workspace root. See `ENVIRONMENT.md` before installation.

Data preparation requires QGIS LTR/OSGeo4W because `qgis` and `osgeo` are
native bindings. Install the tested QGIS LTR/OSGeo4W builds in
`environment.osgeo4w.lock.txt`, set `runtime.osgeo4w_root`
in `config/local_config.json`, then run:

```powershell
python .\steps\data_preparation\setup_environment.py
python .\workflow_config.py run data_preparation --check-environment
```

Environment variables remain supported, but the ignored local config is the
recommended handoff interface.

Step 4 requires:

- QGIS LTR/OSGeo4W Python for the wrapper and raster preflight
- Microsoft MPI runtime (`mpiexec.exe`)
- the supplied `cpp_mpi_port/build_manual/step4_deposition_zone_mpi.exe`
- a TauDEM-compatible `gdal.dll`

The tested non-Python versions and binary checksums are in
`NATIVE_DEPENDENCIES.md`. TauDEM and Microsoft MPI are installed separately;
neither is a pip dependency.

Set the Step 4 executable/runtime locations under `runtime` in
`config/local_config.json`. Leave `step4_executable` empty to use the bundled
executable. Verify without running:

```powershell
python .\workflow_config.py run step4 --check-environment
```

Step 5 also uses QGIS/OSGeo4W:

```powershell
python .\workflow_config.py run step5 --check-environment
python .\workflow_config.py run step5 --dry-run
```

## Running The Workflow

Create the workstation/study override and validate the merged configuration:

```powershell
Copy-Item .\config\local_config.example.json .\config\local_config.json
python .\workflow_config.py validate
```

Run each step through the centralized config:

```powershell
python .\workflow_config.py run data_preparation
python .\workflow_config.py run step0
python .\workflow_config.py run step1
python .\workflow_config.py run step2
python .\workflow_config.py run step3
python .\workflow_config.py run apply_model
python .\workflow_config.py run step4
python .\workflow_config.py run step5
python .\workflow_config.py run step6
```

The optional PowerShell helper selects the workspace Python automatically:

```powershell
.\run_workflow_step.ps1 validate
.\run_workflow_step.ps1 run step0
```

Running an individual step launcher without arguments also uses the master
configuration. Passing an explicit per-step config remains supported for named
experiments.

## Input Contracts

- Use projected rasters with horizontal and vertical units in meters.
- All raster inputs within a step must match CRS, dimensions, transform,
  resolution, and pixel origin unless the step documentation explicitly says
  otherwise.
- Step 1 requires a clean polygon inventory and writes both the binary label
  raster and required `landslide_id` group raster.
- Step 2 requires predictors to be manually aligned before execution and always
  drops labeled rows containing predictor NoData. Any grid mismatch fails the
  run before extraction.
- Step 3 group validation requires the Step 2 `landslide_id` column.
- Step 4 requires aligned DEM, D-Infinity direction, source, DFI, and alpha
  rasters. Its preflight fails before MPI starts when the contract is broken.
- Step 5 requires aligned runout, DFI, DEM, binary source, binary stream, and
  D-Infinity weighted source-contributing-area rasters.
- Step 6 requires an aligned Step 5 combined runout mask, D-Infinity stream
  mask, D-Infinity flow direction, slope, and DEM.

## Recipient Checklist

1. Run `python .\handoff_preflight.py` after extracting the handoff.
2. Rebuild `.venv` and `.venv-dp1`; never reuse copied environments. Install
   the frozen QGIS LTR/OSGeo4W runtime separately for DP1 and native GIS steps.
3. Copy only `config/local_config.example.json` to the ignored
   `config/local_config.json`, then edit study and machine settings there.
4. Run `python .\workflow_config.py validate`.
5. Run `python .\scripts\check_environment.py`. Resolve every `FAIL` before
   processing; warnings should be reviewed.
6. Repeat the targeted check before each official run, for example
   `python .\scripts\check_environment.py --step step4`.
7. Run the DP1, Step 4, and Step 5 environment checks through the master runner.
8. Confirm raster alignment and polygon CRS before a full run.
9. Run on a small clipped dataset before processing the full domain.
10. Keep input and output folders separate; do not point an output at an input.
11. Review each step's summary, QC, and diagnostic outputs before continuing.

Use `python .\handoff_preflight.py` to check the shared handoff and use each
step's dry-run/preflight through `workflow_config.py` after adapting the local
study paths.

The centralized environment checker uses a 10 GiB minimum free-space floor by
default. Override it only when a justified storage estimate is available:

```powershell
python .\scripts\check_environment.py --step step2 --min-free-gb 25
```

Standalone TauDEM command-line tools are reported even though active Step 4
uses the bundled MPI engine and a TauDEM-compatible `gdal.dll`. Add
`--require-taudem-cli` only when the workstation must also run TauDEM utilities
outside this workflow.

The Step 4 prebuilt executable is retained for normal use. Rebuilding it is
optional and requires Visual Studio Build Tools, Microsoft MPI development
files, and a compatible GDAL DLL. The required TauDEM source subset is included
under `cpp_mpi_port/third_party/taudem`. The build script accepts overrides
through parameters or environment variables and contains no desktop-specific
source path. For a build-time GDAL override, set `TAUDEM_GDAL_DLL` to the DLL
file.
