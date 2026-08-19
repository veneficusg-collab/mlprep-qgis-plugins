# Reproducible Software Environment

## Tested Platform

The workflow was tested on **Windows 11 Pro 25H2, x64, build
26200.8875**. Windows compatibility APIs may report this as NT 10.0. The
provided batch files, PowerShell setup scripts, Microsoft MPI runtime, and
Step 4 executable are Windows-specific.

Three runtime boundaries are intentional:

| Workflow scope | Python | Environment source |
|---|---:|---|
| Steps 0-3 and workspace utilities | 3.13.12 | `environment.yml` and `requirements.lock.txt` |
| Data preparation (DP1) | 3.12.10 | QGIS LTR/OSGeo4W and `.venv-dp1` |
| Steps 4-5 native GIS wrappers | 3.12.10 | QGIS LTR/OSGeo4W |

Step 4 additionally requires Microsoft MPI and a TauDEM-compatible GDAL
runtime. Those are documented separately in `NATIVE_DEPENDENCIES.md`.

## Standard Python Environment

Use Conda or Mamba from the workspace root:

```powershell
conda env create -f environment.yml
conda activate mldfi-standard
python -m pip check
```

Alternatively, use the tested virtual-environment installer:

```powershell
.\setup_environment.ps1
```

After configuring `config/local_config.json`, validate the complete installed
environment and current study paths:

```powershell
python .\scripts\check_environment.py
```

The command fails on version drift, broken imports, incompatible GDAL/PROJ
environment variables, missing required native runtimes or inputs, unwritable
outputs, CRS/grid mismatches, and insufficient disk space. Use `--step` to
check only the next workflow stage.

The setup script requires Python `3.13.12`, pins pip to `25.3`, and installs
the complete transitive lock in `requirements.lock.txt`. The shorter
`requirements.txt` records direct project dependencies but is less suitable
for a reproducible handoff.

Important versions in the standard runtime are:

| Package | Version |
|---|---:|
| Python | 3.13.12 |
| pip | 25.3 |
| NumPy | 2.4.2 |
| pandas | 3.0.1 |
| SciPy | 1.17.1 |
| rasterio | 1.5.0 |
| GeoPandas | 1.1.3 |
| Shapely | 2.1.2 |
| pyproj | 3.7.2 |
| pyogrio | 0.12.1 |
| scikit-learn | 1.8.0 |
| XGBoost | 3.2.0 |
| Optuna | 4.8.0 |
| joblib | 1.5.3 |
| Numba | 0.65.1 |
| Matplotlib | 3.10.9 |

The tested rasterio wheel embeds GDAL `3.12.1` and PROJ `9.7.1`. The tested
pyproj wheel reports embedded PROJ `9.5.1`. These wheel-local native libraries
are separate from the QGIS/OSGeo4W GDAL and PROJ runtime below.

For integrity checking, the tested SHA-256 of `requirements.lock.txt` is:

```text
9E6BB86CB1ACAFE52C9C305DC28F3C7832EE80F85FA82146964AF0BFBBFA15F2
```

## QGIS And DP1 Environment

DP1 and the native GIS wrappers use the QGIS LTR/OSGeo4W runtime:

| Component | Tested version |
|---|---:|
| QGIS LTR | 3.40.7-Bratislava (`3.40.7-1`) |
| Python | 3.12.10 (`python3-core 3.12.10-1`) |
| GDAL | 3.10.3 (`3.10.3-2`) |
| PROJ | 9.6.0 (`9.6.0-2`) |
| NumPy | 1.26.4 |
| pandas | 2.2.2 |
| SciPy | 1.13.0 |
| GeoPandas | 1.0.1 |
| Shapely | 2.0.6 |
| rasterio | 1.5.0 |
| pyproj | 3.7.0 |
| Matplotlib | 3.10.0 |

The exact OSGeo4W package build identifiers are in
`environment.osgeo4w.lock.txt`. DP1 adds this exact isolated wheel overlay:

| DP1 local package | Version |
|---|---:|
| pip | 25.0.1 |
| NumPy | 2.5.1 |
| Shapely | 2.1.2 |
| rasterio | 1.5.0 |
| pyproj | 3.7.2 |
| Matplotlib | 3.10.9 |

The complete DP1 transitive overlay is frozen in
`steps/data_preparation/requirements.lock.txt`. It replaces the corresponding
Python packages only inside `.venv-dp1` while inheriting QGIS and GDAL bindings
from OSGeo4W. This boundary is deliberate: the tested native QGIS Shapely
2.0.6 stack collides with the QGIS-loaded GEOS DLL during DP1 geometry
creation, while the isolated Shapely 2.1.2 wheel passes the geometry smoke
test.

DP1 must be launched through `steps/data_preparation/launch_dp1.py`, which initializes
the QGIS/GDAL runtime and then executes `.venv-dp1`. Do not import unrelated
OSGeo4W SciPy, pandas, GeoPandas, or PyArrow packages from this mixed runtime;
DP1 does not use them.

The tested SHA-256 of `steps/data_preparation/requirements.lock.txt` is:

```text
7D20876921A24BFCAAD996BAE684B6CCF1F1A42072EA4A9A1161E806ECCC2816
```

Create and verify DP1 after installing the documented QGIS LTR build:

```powershell
python .\steps\data_preparation\setup_environment.py
python .\workflow_config.py run data_preparation --check-environment
```

## Reproducibility Notes

- Conda/Mamba was not installed on the test workstation. `environment.yml`
  mirrors the tested Python/pip lock but was validated structurally rather than
  installed there.
- QGIS/OSGeo4W, TauDEM, and Microsoft MPI are not installed through pip.
- Rebuild `.venv` and `.venv-dp1` on each workstation; never transfer copied
  environments. DP1 also requires the separately installed QGIS LTR runtime.
- Historical OSGeo4W builds may require an archived installer or package
  mirror. A newer QGIS build should be treated as a new environment and pass
  all preflight and representative raster tests before production use.
