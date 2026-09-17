# Step 4 Deposition Zone C++/MPI Port

This folder contains the active C++/MPI Step 4 routing core. It was validated
against the historical Python reference now archived at
`..\archive\historical_reference\old_step4_deposition_zone.py`. That file is not
a production fallback.

## What Is Ported

- TauDEM D-Infinity neighbor numbering and flow-proportion formula
- flow-path distance routing
- binary source raster initiation
- alpha raster source initialization
- DFI-driven dynamic alpha adjustment
- fixed 200 m source-to-cell planimetric D-Infinity flow-path length cap
- beta acceptance rule, `beta >= dynamic alpha`
- highest-beta / lower-alpha tie-breaking
- dynamic alpha, beta, DFS, runout mask, and depositional mask outputs

## Requirements

### Runtime Requirements

These are required to run the already-built C++/MPI executable through the
Python wrapper.

- Windows 64-bit.
- Microsoft MPI runtime. The wrapper resolves `mpiexec` from `--mpiexec`, the
  `MPIEXEC` environment variable, or `PATH`.
- The compiled executable:
  `cpp_mpi_port\build_manual\step4_deposition_zone_mpi.exe`
- TauDEM/GDAL runtime DLL folder containing `gdal.dll`.
  Supply it through `--taudem-dll-dir`, `TAUDEM_DLL_DIR`, the workspace
  `cpp_mpi_port\runtime` folder, or `PATH`.
- A compatible PROJ database. Set `TAUDEM_PROJ_DIR` when it cannot be derived
  as `share\proj` beside the configured TauDEM runtime.
- Python with the packages used by the Step 4 wrapper:
  `numpy` and `osgeo.gdal`.
- The shared Python helper module beside the wrapper:
  `step4_common.py`
- A valid Step 4 JSON config, for example `config.default.json`.
- Input rasters listed in the config:
  DEM/fel, D-Infinity flow direction, source, DFI, and alpha.
- Writable output folders for the rasters and summary JSON in the config.

Recommended PowerShell environment before running:

```powershell
$env:OSGEO4W_ROOT = "<osgeo4w-installation-folder>"
$env:MPIEXEC = "<mpi-launcher>"
$env:TAUDEM_DLL_DIR = "<taudem-gdal-runtime-folder>"
$env:TAUDEM_PROJ_DIR = "<proj-data-folder>"
```

Recommended wrapper command:

```powershell
python step4_deposition_zone_mpi_wrapper.py `
  --config config.default.json
```

The wrapper defaults to 8 MPI processes. To override it:

```powershell
python step4_deposition_zone_mpi_wrapper.py `
  --config config.default.json `
  --processes 4
```

### Build Requirements

These are required only if you rebuild the C++ executable.

- Visual Studio Build Tools 2022.
- MSVC v143 C++ build tools.
- Windows 10 or Windows 11 SDK.
- C++ CMake tools for Windows, optional for CMake builds.
- C++17 compiler support.
- Microsoft MPI headers and import library.
  This port uses the local copy under:
  `cpp_mpi_port\third_party\msmpi`
- The required TauDEM source subset is included under
  `third_party\taudem\src` with its GPL license. An alternative compatible
  source tree may still be passed with `-TaudemSrc` or `TAUDEM_SRC_DIR`.
- GDAL DLL used to generate the import library.
  Pass it with `-GdalDll` or set `TAUDEM_GDAL_DLL`.
- The compatibility headers under:
  `cpp_mpi_port\gdal_compat`

## Build

Build from this folder with CMake. The default uses the local TauDEM subset.

```powershell
cmake -S . -B build
cmake --build build --config Release
```

The executable can also be rebuilt with the manual MSVC script:

```powershell
.\cpp_mpi_port\build_manual_msvc.ps1
```

The build requires:

- C++17 compiler
- MPI, for example Microsoft MPI on Windows
- GDAL development files
- the locally included TauDEM source subset
- the TauDEM/GDAL runtime DLL folder

The build script compiles to a candidate executable, preserves the previous
runtime if compilation fails, and removes object/import-library intermediates
after success. The prebuilt executable is retained for handoff use.

## Run

```powershell
mpiexec -n 8 .\build_manual\step4_deposition_zone_mpi.exe `
  --fel "path\to\fel.tif" `
  --ang "path\to\ang.tif" `
  --source "path\to\source.tif" `
  --dfi "path\to\dfi.tif" `
  --alpha "path\to\alpha.tif" `
  --out-alpha "path\to\dynamic_alpha.tif" `
  --out-beta "path\to\beta_angle.tif" `
  --out-dfs "path\to\dfs.tif" `
  --out-mask "path\to\runout_mask.tif" `
  --out-deposition "path\to\depositional_mask.tif" `
  --threshold 0.2 `
  --dfi-mid 0.69 `
  --alpha-gain-per-meter 0.03333
```

## Python Wrapper

From the Step 4 folder, use:

```powershell
python step4_deposition_zone_mpi_wrapper.py --config config.default.json
```

The wrapper defaults to 8 MPI processes. Pass `--processes` only when you want a
different process count.

If Microsoft MPI is installed somewhere else, pass the launcher explicitly:

```powershell
python step4_deposition_zone_mpi_wrapper.py `
  --config config.default.json `
  --mpiexec "$env:MPIEXEC"
```

If TauDEM is installed somewhere else, pass the DLL folder explicitly:

```powershell
python step4_deposition_zone_mpi_wrapper.py `
  --config config.default.json `
  --processes 8 `
  --taudem-dll-dir "$env:TAUDEM_DLL_DIR"
```

Or set it once for the current PowerShell session:

```powershell
$env:TAUDEM_DLL_DIR = "<taudem-gdal-runtime-folder>"
python step4_deposition_zone_mpi_wrapper.py --config config.default.json --processes 8
```

Use `--dry-run` to print the resolved `mpiexec` command without running:

```powershell
python step4_deposition_zone_mpi_wrapper.py --config config.default.json --dry-run
```

The wrapper reads the JSON config using `step4_common.py`, performs metadata
preflight, and runs the C++/MPI executable. The engine emits a machine-readable
statistics receipt, avoiding a second full-array scan for normal runs. Optional
observed-raster validation still reads the required validation arrays.

Before launching MPI, the wrapper checks:

- required and configured optional input raster paths exist
- DEM, D-Infinity, source, DFI, alpha, and validation rasters align
- the reference CRS is projected in meters
- output paths are unique and do not collide with inputs
- output parent directories can be created or already exist

The distributed engine validates source, alpha, D-Infinity angle, and DFI pixel
values while reading its partitions. It returns ranges, coverage, routing
counters, and timings in JSON. All outputs are written to sibling staging
directories and are published together with rollback protection only after the
engine and wrapper checks pass.

The DLL folder must contain `gdal.dll`. The wrapper temporarily prefixes this
folder onto `PATH` for the C++ executable, so you normally do not need to edit
your permanent Windows environment variables.

## Validation Plan

Use the archived historical Python script as the reference behavior:

1. Run the Python script on a small test raster.
2. Run this executable on the same inputs and parameters.
3. Compare these rasters pixel-by-pixel:
   - dynamic alpha
   - beta angle
   - DFS/path distance
   - runout mask
   - depositional mask
4. Then repeat with `mpiexec -n 1`, `-n 2`, and larger process counts.

The current implementation uses TauDEM's row-partition and border-sharing
classes. The compact report `latest_validation_report.json` records the latest
validated behavior.

## Process Count Validation

The current optimized executable was tested on the full Makilala grid with
`n=1` and `n=8`. Dynamic alpha, beta, DFS, runout, and pure deposition rasters
were exactly identical to each other and to the retained production baseline:
`6,910,670` runout cells and `645,620` pure depositional cells. A focused
synthetic `n=1`/`n=2` test also verifies process-count equality and the exact
200 m planimetric cap. See `latest_validation_report.json`.





