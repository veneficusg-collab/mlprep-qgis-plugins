# Step 1: Prepare Inventory Labels

## What This Step Does

`Step 1` converts mapped landslide polygons and a DEM into ML-DFI training labels using only internal landslide zones.

For each landslide polygon, DEM cells are ranked by elevation:

1. Lowest 15% of valid polygon DEM cells = depositional zone, label `1`
2. Highest 30% of valid polygon DEM cells = source/upper-transport non-depositional zone, label `0`
3. Middle 55% of valid polygon DEM cells = uncertain, `NoData`
4. Outside mapped landslide polygons = `NoData`

This step does not use outside-polygon terrain as negative training data.

## Scientific Rationale

The goal is to train ML-DFI on high-confidence contrasts inside observed landslides.

The lower-elevation part of a mapped landslide is interpreted as the most likely depositional area because material commonly accumulates near the distal and lower portion of the mapped feature.

The upper-elevation part is interpreted as source to upper-transport terrain, where material initiated or continued moving downslope rather than accumulating.

The middle zone is excluded because it can contain mixed transport, erosion, minor accumulation, or transitional deposition.

## Main Script

- [step1_prepare_inventory_labels.py](step1_prepare_inventory_labels.py)

## Inputs

Required:

- `landslide_polygon_path`: mapped landslide polygon inventory
  - use a clean inventory before running Step 1
  - polygons should be valid, deduplicated, correctly projected, and free of
    obvious slivers, multipart mistakes, topology errors, and non-landslide
    artifacts
- `dem_path`: DEM used for per-polygon elevation ranking
- `output_dir`: output folder

Configuration can be supplied with `--config-json`. Relative paths inside the
JSON file are resolved from the JSON file's folder, so the workflow does not
depend on the current PowerShell working directory. Command-line values remain
available and take priority over JSON values.

Main options:

- `depositional_percentile`: default `15`
  - lower-elevation percentage assigned label `1`
- `non_depositional_percentile`: default `30`
  - upper-elevation percentage assigned label `0`
  - internally uses threshold `100 - non_depositional_percentile`, so the default is `P70`
- `landslide_id_field`: optional existing polygon ID field
  - configured IDs must be unique
- `nodata_value`: default `-9999`
- `output_pixel_type`: default `int16`
- `min_valid_dem_cells`: default `5`
- `all_touched`: rasterization behavior
- `parallel_processing`: default `true`
  - processes independent polygon DEM windows in spawned worker processes
- `max_workers`: default `4`
  - upper limit; effective workers are also capped by CPU count and polygon count

## Outputs

The script writes DEM-aligned rasters using the DEM grid, transform, extent, and
CRS. Output location is controlled entirely by `output_dir`.

Core rasters:

- `depositional_zone_raster.tif`
  - `1` for the lowest 15% zone
  - `NoData` elsewhere
- `non_depositional_zone_raster.tif`
  - `1` for the highest 30% source/transport zone
  - `NoData` elsewhere
- `landslide_id_raster.tif`
  - one-based numeric landslide feature ID for valid DEM cells inside mapped landslide polygons
  - `NoData` outside mapped landslides or where the polygon has no valid DEM support
  - intended for Step 2 `group_rasters.landslide_id` and Step 3 grouped validation
- `ml_dfi_label_raster.tif`
  - `1` = depositional terrain
  - `0` = source/transport non-depositional terrain
  - `NoData` = uncertain or outside mapped landslides

Diagnostics:

- `label_summary.csv`
  - class counts, percentile settings, skipped polygons, and per-landslide thresholds
- `processing_log.txt`
- `run_config.json`
  - resolved paths, parameter values, DEM grid metadata, and run diagnostics

All outputs are first generated in a temporary staging directory. Step 1 runs
QC against that complete staged bundle and publishes it only after QC passes.
If publication fails, the previous official bundle is restored.

## Final Label Logic

The final raster is initialized as `NoData`.

Then:

1. non-depositional/source-transport cells are written as `0`
2. depositional cells are written as `1`
3. uncertain middle-zone cells remain `NoData`
4. outside-polygon cells remain `NoData`

If overlapping polygons mark the same pixel differently, depositional labels take priority over non-depositional labels, and labeled classes take priority over uncertain status.

## Quality Control

Before running Step 1, inspect and clean the landslide inventory. The script can
check raster-label consistency after processing, but it cannot determine whether
the mapped polygons themselves are scientifically correct.

Step 1 checks that:

- all rasters match the DEM shape, transform, and CRS
- depositional and non-depositional mask rasters contain only `1` and `NoData`
- the landslide ID raster contains positive numeric IDs and `NoData`
- the label raster contains only `0`, `1`, and `NoData`
- depositional cell count is greater than zero
- non-depositional cell count is greater than zero
- uncertain cell count is greater than zero
- the final label raster is readable
- inventory geometries are polygons or multipolygons
- configured landslide IDs are unique

The summary records the realized class percentages for every polygon. Small
polygons and tied DEM elevations can cause the pixel percentages to differ from
the requested 15% and 30%; Step 1 warns when the difference exceeds five
percentage points. Multipart features are retained as one landslide group and
reported because their parts share one elevation distribution.

## Runtime And Memory

DEM reads remain limited to each polygon's bounding window. With
`parallel_processing=true`, Step 1 uses a bounded `ProcessPoolExecutor` queue.
Each spawned worker opens the DEM once and reuses that read-only handle for all
polygons assigned to it. Workers perform only local DEM-window reading,
rasterization, percentile calculation, and mask classification.

Worker masks are bit-packed before transfer to limit inter-process memory and
serialization overhead. The parent process merges results in original polygon
order, preserving deterministic overlap handling. Workers never mutate shared
masks and never write output rasters.

Production runs use disk-backed parent masks and write/QC outputs in
bounded-memory windows. A single-polygon inventory automatically uses the
serial path because process startup would provide no benefit. Multiprocessing
can be disabled with `--no-parallel-processing` for diagnosis. C++ and MPI are
not used by Step 1.

## Example Usage

```powershell
.\steps\step1_prepare_inventory_labels\run_step1.bat
```

The launcher finds the workspace `.venv` and can be called from any directory.
Config paths resolve relative to the config file.

Command-line-only execution remains supported:

```powershell
python .\steps\step1_prepare_inventory_labels\step1_prepare_inventory_labels.py --landslide-polygon-path "<inventory.gpkg>|layername=inventory" --dem-path "<dem.tif>" --output-dir "<step1-output-folder>" --depositional-percentile 15 --non-depositional-percentile 30
```

## Dependencies

- `rasterio`
- `geopandas`
- `shapely`
- `numpy`
- `pandas`

## Folder Contents

- [step1_prepare_inventory_labels.py](step1_prepare_inventory_labels.py)
- [step1.md](step1.md)
