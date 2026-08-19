# Step 00: DP1 Observed Alpha-Angle Data Preparation

## Purpose

Step 00 transfers the DP1 observed-alpha workflow into this Codes_V2 workspace
as the initial data-preparation step before `step0_baseline_alpha_angle`.
It measures observed landslide mobility from inventory polygons and a DEM by
estimating each landslide's crown, toe, travel-path length, elevation drop, and
observed alpha angle.

The main calculation is:

```text
h_over_l = H / L
a_obs = atan(H / L)
```

where `H` is the vertical drop from crown to toe, and `L` is the mapped
three-dimensional travel-path length. Lower observed alpha generally means a
longer-reaching, more mobile landslide; higher observed alpha generally means a
shorter, steeper runout.

## Implementation

The DP1 runtime is contained directly in this folder:

```text
steps/data_preparation/
|-- run_dp1.py
|-- config.default.json
|-- step00.md
|-- dp1_native.py
|-- dp1_worker.py
|-- dp1_support.py
|-- dp1_engine.py
`-- dp1_models.py
```

This keeps the step independent inside Codes_V2. It does not import scripts,
contracts, or helpers from any older workspace.

The sole runtime launcher is:

```text
steps/data_preparation/launch_dp1.py
```

## Required Inputs

- `landslide_polygons`: mapped landslide polygon inventory.
- `id_field`: unique polygon identifier field.
- `dem`: projected DEM raster in meters.
- `output_dir`: folder for final DP1 outputs.

The default config is:

```text
steps/data_preparation/config.default.json
```

## Main Outputs

The workflow writes `dp1.gpkg` in the configured output folder. Important layers
include observed alpha results, crown/toe diagnostics, runout-path diagnostics,
and clean/accepted landslide features such as `dp1_landslide_polygons_clean`.

Diagnostic outputs should be inspected before using DP1 results as evidence for
later alpha-angle or runout calibration decisions.

## Run Command

DP1 requires its dedicated `.venv-dp1` overlay on the frozen QGIS
LTR/OSGeo4W environment. Always invoke it through the local launcher so the
required OSGeo4W DLL and QGIS paths are initialized before Python starts.

```powershell
python .\steps\data_preparation\setup_environment.py
python .\steps\data_preparation\launch_dp1.py --config .\steps\data_preparation\config.default.json
```

Set `OSGEO4W_ROOT` to the local OSGeo4W installation folder first. Do not copy
`.venv-dp1` to another desktop; rebuild it there.

Validate the complete environment without starting a production run:

```powershell
python .\steps\data_preparation\launch_dp1.py --check-environment
```

## Runtime Dependency Contract

The transferred step has two dependency layers:

- All local source dependencies are included directly in this folder.
- Runtime GIS bindings must come from QGIS/OSGeo Python: `qgis`, `osgeo`,
  `numpy`, and related QGIS Processing modules.

The DP1 runtime uses `.venv-dp1`, based on QGIS LTR 3.40.7 and OSGeo4W Python
3.12.10. It inherits QGIS and GDAL/OGR/OSR from OSGeo4W while loading the exact
NumPy, Shapely, Rasterio, PyProj, and Matplotlib wheel overlay frozen in
`requirements.lock.txt`. It does not use `.venv`/`.venv-local` or Windows
user-site packages. Exact versions are listed in `../../ENVIRONMENT.md` and
`../../environment.osgeo4w.lock.txt`.

The launcher removes user-site package folders from `sys.path` and sets
`PYTHONNOUSERSITE=1` before loading DP1 modules. Spawned worker processes inherit
that setting. This prevents packages under the Windows user profile or another
workspace from silently replacing the OSGeo4W versions.

The runtime check rejects every interpreter except `.venv-dp1`, validates the
origin of every required package against `.venv-dp1` and the configured
QGIS LTR/OSGeo4W root, and
fails when Python, QGIS, GDAL, PROJ, or the tested Python package versions have
drifted from the frozen contract.

The wrapper prepends this folder to `sys.path`, so every `dp1_*` source import
resolves to the local Codes_V2 implementation first.

The transferred code was pruned to the active DP1 runtime only. Legacy
post-processing analyses and shared helpers for unrelated steps are not carried
in this folder. The remaining modules preserve deliberate boundaries between
the routing engine, GIS helpers, multiprocessing worker, and run orchestration.

## Method Notes

- Crown and toe are inferred from sampled landslide-boundary elevations.
- Before DEM extraction and routing, DP1 computes polygon area and the
  complexity ratio `polygon perimeter / minimum-rotated-rectangle perimeter`.
  A polygon is skipped early when either condition is true: its complexity
  ratio is greater than `complexity_threshold`, or its area is less than
  `early_exclusion_area_threshold_m2` (default `250 m2`). The exclusion
  remains visible in the QC CSV as
  `early_excluded_complex_or_small_polygon`, together with the specific
  `complexity_ratio_exceeds_threshold` and/or
  `polygon_area_below_threshold` reason.
- The path engine uses fixed local 2 m contour-midpoint routing.
- Every crown-to-contour, contour-to-contour, and contour-to-toe segment must
  remain within its landslide polygon. Tortuosity smoothing is subject to the
  same containment requirement.
- Final path geometry is independently compared with the landslide polygon.
  Actual outside-path step counts, lengths, and fractions are recorded, and
  any excursion beyond floating-point tolerance adds
  `path_outside_landslide`, making the crown ineligible for the clean layer.
- Complex and small polygons are excluded before DEM extraction and routing.
  Other physically inconsistent polygons remain explicitly flagged rather than
  silently accepted.
- Per-polygon processing uses four persistent worker processes by default.
  Tasks are scheduled individually so unusually expensive polygons do not
  stall a fixed batch. Each process reuses its read-only DEM handle.
- Local masked DEMs used for contour generation stay in GDAL memory; DP1 no
  longer creates a temporary contour TIFF or per-polygon scratch directory.
- GeoPackage features are written in transactions after all workers finish.
  Workers never write concurrently to the output GeoPackage.
