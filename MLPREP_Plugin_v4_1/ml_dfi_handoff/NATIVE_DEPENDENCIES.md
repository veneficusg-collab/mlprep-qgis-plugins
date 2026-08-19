# Non-Python Dependencies

These dependencies are installed and versioned separately from pip and
`environment.yml`.

| Dependency | Tested version | Used by |
|---|---:|---|
| Windows | Windows 11 Pro 25H2 x64, build 26200.8875 | Entire supplied handoff |
| QGIS LTR | 3.40.7-Bratislava (`3.40.7-1`) | DP1, Steps 4-5 |
| GDAL (OSGeo4W) | 3.10.3 (`3.10.3-2`) | QGIS/OSGeo raster and vector I/O |
| PROJ (OSGeo4W) | 9.6.0 (`9.6.0-2`) | QGIS/OSGeo CRS operations |
| Microsoft MPI | 10.1.12498.52, x64 | Step 4 distributed routing |
| TauDEM | 5.5.0 | Tested Step 4-compatible GDAL runtime/source family |
| Microsoft Visual C++ runtime | 14.50.35719.0, x64 | Step 4 executable |

## QGIS And OSGeo4W

Install QGIS LTR through OSGeo4W with Python support. Configure the installation
root in the ignored `config/local_config.json` under
`runtime.osgeo4w_root`. Exact tested OSGeo4W package builds are recorded in
`environment.osgeo4w.lock.txt`.

QGIS/OSGeo4W provides its own Python, GDAL, PROJ, NumPy, and compiled extension
modules. Do not point its launchers at the standard Python 3.13 environment.

## Microsoft MPI

Step 4 requires the **Microsoft MPI 10.1.12498.52 x64 runtime**. Configure
`runtime.mpiexec` in `config/local_config.json`, or make `mpiexec.exe`
available on `PATH`.

The tested `mpiexec.exe` SHA-256 is:

```text
161D18AC9F66D28ADD3AEC8719F834297D0DC08CE07EC536C8AED5A9172FCA8E
```

The MPI SDK, headers, and import libraries are needed only to rebuild Step 4.
They are not needed to run the bundled executable.

## TauDEM

TauDEM is not a pip dependency. The tested native installation was **TauDEM
5.5.0**. Normal Step 4 execution uses the bundled MPI executable but still
needs a compatible folder containing `gdal.dll` and matching PROJ data.
Configure these ignored local settings:

```json
{
  "runtime": {
    "taudem_dll_dir": "<taudem-runtime-folder>",
    "taudem_proj_dir": "<matching-proj-data-folder>"
  }
}
```

The tested TauDEM runtime `gdal.dll` identifies as GDAL `3.10.3` and has this
SHA-256:

```text
C01A6299CA4BF0D96C72180CE4166338A2A8653967014E3E8E484CF2047429C6
```

The small TauDEM-compatible source subset required to rebuild the routing core
is vendored under
`steps/step4_deposition_zone/cpp_mpi_port/third_party/taudem`.
Its exact upstream commit was not preserved, so the tested binary and
behavioral validation report remain the authoritative production artifacts.

## Step 4 Binary

The supplied Windows x64 executable is:

```text
steps/step4_deposition_zone/cpp_mpi_port/build_manual/step4_deposition_zone_mpi.exe
```

Its tested SHA-256 is:

```text
DC93B9F567C89F818246CEDF02E7FA9C99699A9E11A79E5F5091538516BFF66B
```

Run `python .\workflow_config.py run step4 --check-environment` before a
production run. Rebuilding additionally requires Visual Studio Build Tools
2022, MSVC v143, a Windows SDK, Microsoft MPI development files, and compatible
GDAL development/runtime files.
