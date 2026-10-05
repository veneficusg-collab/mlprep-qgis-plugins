# Step 5: Post-Depositional Spread And Runout Cleanup PDS

## Introduction

Step 5 is an optional post-processing step after Step 4. It first cleans the
Step 4 dynamic-alpha runout mask by removing runout cells with negligible
potential source contribution, then expands the cleaned runout mask into a
broader possible depositional footprint using a precomputed D-Infinity source
contributing-area raster, local DFI, DEM validity, and seed-relative relief
constraints.

In plain geologic terms, Step 4 estimates where the runout can travel. Step 5
then asks two questions: which Step 4 runout cells are too weakly supported by
potential source area to matter at provincial mapping scale, and near the edge
of the remaining runout, where might material spread laterally into a fan-like
deposit?

This is not a full sediment-transport model. It is a terrain-conditioned GIS
cleanup and expansion tool. It uses the existing Step 4 runout mask, DFI, DEM,
source mask, stream mask, and a TauDEM D-Infinity weighted contributing-area
raster to decide which negligible-source runout cells should be removed and
where limited spreading is plausible.

The PDS radius is controlled by a D-Infinity source contributing-area raster
computed before Step 5. That raster should be generated with TauDEM contributing
area / specific catchment area using a binary source mask as the weight/load
grid. The source mask must use `1 = source` and `0 = non-source`. The resulting
contributing-area raster is treated as a D-Infinity-weighted count of
contributing source cells. Step 5 converts that raw count to square meters by
multiplying by cell area before computing spread radius. It is a terrain-based
proxy for possible source magnitude, coalescence, and effective material
contribution; final spread still remains constrained by DFI, DEM validity,
radius, seed-relative relief, and connectivity filters.

Runout cleanup uses the same converted source contributing area. By default,
runout cells with less than `200 m2` of contributing source area are removed
before seed identification. This threshold is based on raster resolution and
provincial mapping scale: source areas smaller than this are treated as
negligible for producing meaningful mapped runout.

After the area threshold, Step 5 applies a source-connectivity cleanup. Only
retained runout components that touch the binary source mask, or are
8-neighbor-adjacent to it, are kept. This prevents isolated downstream runout
islands from surviving just because their local contributing-area raster value
is high.

The implementation is standalone in this Codes_V2 workspace and is consolidated
into one main script. It does not import scripts or config helpers from the
older `Codes` workspace.

## Main Files

- `step5_post_depositional_spread_PDS.py`
- `config.default.json`
- `step5.md`

Verify the QGIS/OSGeo environment without opening inputs:

```powershell
.\steps\step5_post_depositional_spread_PDS\run_step5.bat --check-environment
```

Validate the config, file accessibility, raster headers, exact alignment, CRS,
pixel size, and output-path contract without loading full raster arrays or
writing outputs:

```powershell
.\steps\step5_post_depositional_spread_PDS\run_step5.bat --config .\steps\step5_post_depositional_spread_PDS\config.default.json --dry-run
```

Run through the portable QGIS launcher:

```powershell
.\steps\step5_post_depositional_spread_PDS\run_step5.bat --config .\steps\step5_post_depositional_spread_PDS\config.default.json
```

Set `OSGEO4W_ROOT` to the local OSGeo4W installation folder, or make
`python-qgis-ltr.bat` available on `PATH`. Relative input and output paths
resolve from the config file.

## Inputs And Outputs

Required input and core-output paths:

- `runout_mask_raster`: Step 4 runout footprint.
- `dfi_raster`: ML-DFI probability raster used to identify depositional seed cells.
- `dem_raster`: filled DEM used for elevation-validity and seed-relative relief checks.
- `source_mask_raster`: binary source mask. Value `1` is source and `0` is non-source. Any other valid value is rejected.
- `stream_mask_raster`: required stream mask. Value `0` is treated as non-stream; any positive valid value is treated as stream and excluded from seed selection.
- `source_contributing_area_raster`: required TauDEM D-Infinity weighted source-cell-count raster generated from the binary source mask as the weight/load grid.
- `output_spread_mask`: raster of newly added spread cells.
- `output_combined_mask`: raster combining the cleaned Step 4 runout mask and added spread.
- `output_seed_mask`: raster showing which cells were used as spread seeds.
- `output_cleaned_runout_mask`: cleaned Step 4 runout mask after removing cells below the minimum source-area threshold.
- `output_removed_runout_mask`: audit mask showing Step 4 runout cells removed by source-area or source-connectivity cleanup.
- `output_summary_json`: run parameters, counts, statistics, warnings, validation metadata, and timing information.

The default config uses shared aligned DFI, DEM, and source-mask rasters under
`Sample Data/Input/Raster/Shared Inputs`. Step-specific stream-mask and source
contributing-area rasters remain under `Sample Data/Input/Raster/Step5 Inputs`.

Optional outputs:

- shapefiles for spread and combined masks, if configured.
- debug rasters for elevation difference and source provenance.

The default outputs are written under:

```text
Sample Data/Output/Latest Runs/Step5
```

Core default outputs:

- `post_depositional_spread_mask.tif`
- `post_depositional_combined_mask.tif`
- `post_depositional_seed_mask.tif`
- `post_depositional_cleaned_runout_mask.tif`
- `post_depositional_removed_runout_mask.tif`
- `post_depositional_spread_summary.json`

Optional debug rasters are enabled by default:

- `post_depositional_min_elev_diff.tif`
- `post_depositional_source_row.tif`
- `post_depositional_source_col.tif`
- converted source contributing area at seed cells, in square meters
- source contribution index at seed cells
- computed spread radius at seed cells

Important parameters:

- `dfi_threshold`: controls which Step 4 runout-edge cells can become spread seeds.
- `max_relative_rise_m`: maximum allowed elevation rise from the edge seed cell to a candidate spread cell.
- `min_runout_source_area_m2`: cleanup threshold for deleting Step 4 runout cells with negligible converted source contributing area.
- `min_spread_radius_m` and `max_spread_radius_m`: bounds on how far spreading can extend.
- `source_contributing_area_reference`: converted source contributing area, in square meters, where the radius score reaches 1.0. The default is `4400 m2`, based on a maximum typical coseismic landslide area of about `11,000 m2` and an assumed `40%` source/contributing-area fraction.
- `write_debug_rasters`: controls optional inspection rasters.

Removed public options:

- `spread_radius_m`
- `spread_half_angle_deg`
- `allow_recursive_spread`
- `terminal_slope_min_deg`
- `terminal_slope_max_deg`
- `plan_curvature_raster`
- `terminal_influence_length_m`
- `min_valid_terminal_path_length_m`
- `source_susceptibility_raster`
- `source_area_reference_m2`
- `source_area_weight`
- `high_fraction_weight`

These were removed because spread radius is now controlled by weighted source
contributing area and spread is no longer directionally confined by plan
curvature.

## Full Data Flow

```text
Step 4 runout mask + terrain rasters
    -> source-area cleanup of negligible runout cells
    -> source-connected component cleanup
    -> cleaned-runout edge seed detection and raster checks
    -> source contributing-area radius lookup
    -> all-direction connected spread with seed-relative relief filtering
    -> spread/combined masks
    -> summaries/debug rasters
```

Input:

- The script starts with the Step 4 runout mask and terrain rasters that describe DFI, elevation, D-Infinity routing, and precomputed source contributing area.

Preprocessing:

- A metadata-only preflight checks that all required rasters open and have exact
  dimensions, geotransform, and CRS alignment before full arrays are read.
- The script rejects geographic CRS, rotated grids, or mismatched raster dimensions.
- The runout and source masks must be binary `0/1`; DFI must be within `[0, 1]`;
  the stream mask and contributing-area raster must be non-negative.
- NoData or non-finite contributing-area values at any Step 4 runout cell are
  rejected. They are not interpreted as zero source contribution.
- The raw weighted contributing source-cell count is converted to square meters.
- Step 4 runout cells with converted source contributing area below `min_runout_source_area_m2` are removed before seed detection.
- Remaining runout cells are filtered to keep only components connected to the source mask.

## Seed Cell Selection Criteria

A cell is used as a Step 5 spread seed only when it passes all of the following
base criteria:

1. It belongs to the cleaned Step 4 runout mask.
2. It has valid DEM and DFI values.
3. Its DFI value is greater than `dfi_threshold`.
4. It is not marked as a source cell in `source_mask_raster`.
5. It is not marked as a stream cell in `stream_mask_raster`.
6. It is on the edge of the cleaned Step 4 footprint, meaning at least one adjacent 8-neighbor cell is valid and outside the cleaned runout mask.

The seed-selection stage does not apply an elevation-difference filter. The
relative relief control is applied downstream when candidate spread cells are
evaluated against each identified seed cell.

Main computation:

- For each edge seed cell, the script reads the raw D-Infinity-weighted source-cell count at that seed.
- The raw weighted count is multiplied by raster cell area to estimate contributing source area in square meters.
- The converted source area is normalized by `source_contributing_area_reference` into a 0-1 source contribution index.
- The source contribution index scales spread radius between `min_spread_radius_m` and `max_spread_radius_m`.
- Candidate spread cells are evaluated in all directions inside the computed radius.
- A candidate can join the spread only if it satisfies DFI and DEM validity, is not more than `max_relative_rise_m` above its edge seed cell, and is connected to the seed through eligible spread cells.

## Equations

Cell area:

```text
cell_area_m2 = dx * dy
```

Raw D-Infinity weighted source-cell count at seed cell `s`:

```text
raw_source_count_s = source_contributing_area_raster[s]
```

Converted contributing source area:

```text
source_area_s_m2 = raw_source_count_s * cell_area_m2
```

Runout cleanup:

```text
source_area_c_m2 = source_contributing_area_raster[c] * cell_area_m2

area_removed_runout_c =
    runout_mask_c == 1
    AND source_area_c_m2 < min_runout_source_area_m2

area_cleaned_runout_c =
    runout_mask_c == 1
    AND NOT area_removed_runout_c

source_connected_runout_c =
    area_cleaned_runout_c == 1
    AND c belongs to an 8-connected area-cleaned runout component
        that touches or is 8-neighbor-adjacent to source_mask == 1

disconnected_runout_c =
    area_cleaned_runout_c == 1
    AND source_connected_runout_c == 0

cleaned_runout_c = source_connected_runout_c

removed_runout_c =
    area_removed_runout_c
    OR disconnected_runout_c
```

Source contribution index:

```text
source_contribution_index_s =
    min(source_area_s_m2 / source_contributing_area_reference, 1.0)
```

Default reference-area basis:

```text
source_contributing_area_reference = 11000 m2 * 0.40 = 4400 m2
```

Spread radius:

```text
computed_spread_radius_s =
    min_spread_radius_m
    + source_contribution_index_s
      * (max_spread_radius_m - min_spread_radius_m)
```

With the current 5 m raster resolution:

```text
cell_area_m2 = 5 * 5 = 25
source_area_s_m2 = raw_source_count_s * 25
```

Seed eligibility:

```text
edge_s =
    exists adjacent valid non-cleaned-runout cell n

seed_s =
    cleaned_runout_s == 1
    AND DFI_s > dfi_threshold
    AND source_mask_s == 0
    AND stream_mask_s == 0
    AND edge_s
```

Candidate spread-cell eligibility:

```text
candidate_c =
    distance(c, s) <= computed_spread_radius_s
    AND cleaned_runout_c == 0
    AND DFI_c > dfi_threshold
    AND DEM_c is valid
    AND DEM_c <= DEM_s + max_relative_rise_m
    AND c is connected to s through eligible candidate cells
```

Final masks:

```text
spread_mask_c = 1 if candidate_c is accepted by at least one seed
combined_mask_c = cleaned_runout_mask_c OR spread_mask_c
```

Postprocessing:

- The newly added spread mask is written separately.
- The spread mask is combined with the cleaned Step 4 runout mask.
- Optional shapefiles and debug rasters are created for GIS inspection.
- Every output is first written to a hidden staging directory on the same
  filesystem and checked for completeness and raster alignment.
- The staged bundle replaces the official bundle only after validation. If
  publication fails, the previous official files are restored. The summary JSON
  is published last and therefore acts as the completion marker.

Output:

- `output_spread_mask` shows only the added depositional spread.
- `output_combined_mask` shows cleaned Step 4 runout plus post-depositional spread.
- `output_seed_mask` shows which cleaned Step 4 edge cells controlled spreading.
- `output_cleaned_runout_mask` shows the Step 4 runout cells retained after source-area cleanup.
- `output_removed_runout_mask` shows Step 4 runout cells deleted because converted source contributing area is below `min_runout_source_area_m2` or because the remaining component is not source-connected.

Validation:

- Inspect whether seed cells are located along plausible cleaned-runout margins.
- Inspect the removed-runout mask to confirm that cleanup mainly removes negligible-source or source-disconnected, overextended runout cells.
- Check that source contribution index and spread radius look geologically reasonable.
- Check that added spread does not climb more than the configured rise above its edge seed cells.
- Review the summary JSON and debug rasters when spread looks too wide, too narrow, or misdirected.

## Scientific, GIS, And Coding Assumptions

Scientific assumptions:

- Depositional spreading is initiated along margins of the cleaned runout footprint, not across the entire source-to-path area.
- Runout cells supported by less than 200 m2 of contributing source area are negligible at provincial mapping scale and can be removed before spread.
- A retained runout component must remain spatially connected to the source mask after the area cleanup.
- DFI can help identify cells where deposition or spreading is more plausible.
- Larger source contributing-area values imply larger possible source contribution and coalescence potential.
- Post-depositional spread can occur laterally in all directions around an edge seed, but should not climb more than the configured rise above that seed.
- The method is a deterministic GIS approximation, not a full physics-based debris-flow deposition model.

GIS/data assumptions:

- All rasters are exactly aligned to the Step 4 runout mask grid.
- Rasters use a projected metric CRS.
- The source contributing-area raster is generated upstream with TauDEM D-Infinity routing.
- Source mask values must be binary: `1 = source`, `0 = non-source`.
- Source contributing-area values are non-negative weighted source-cell counts.
- Source contributing-area NoData is allowed outside the Step 4 runout footprint,
  but not at a Step 4 runout cell.

Coding/workflow assumptions:

- Step 5 never modifies Step 4 files; it writes separate post-processing outputs.
- `write_debug_rasters` affects only extra inspection outputs, not the core spread calculation.
- Seed cells are DFI-qualified non-source, non-stream edge pixels of the cleaned Step 4 footprint; interior runout cells, source cells, and stream cells do not initiate post-depositional spreading.
- Removed config options are intentionally rejected so old fixed-radius settings do not silently change the current method.
- Unknown config fields, non-finite parameters, duplicate output paths, and
  input/output path collisions are rejected.
- Accepted spread cells never become new seed cells.

## Runtime And Memory Design

The optimized implementation keeps the scientific rules unchanged while
reducing Python and memory overhead:

- Input rasters retain their native NumPy dtypes instead of all being promoted
  to `float64`.
- Seed source-area scores and radii are calculated in vectorized one-dimensional
  arrays.
- Candidate offsets are precomputed once, then filtered with NumPy for each
  seed.
- Per-seed connectivity reuses compact local Boolean buffers and fixed integer
  queues instead of constructing Python dictionaries and sets repeatedly.
- The winning seed for each spread cell is stored as one flattened integer
  index. Row, column, and elevation debug rasters are derived one at a time
  during output writing.
- Full-grid source-area, source-index, and radius debug arrays are allocated
  only when `write_debug_rasters` is enabled.
- Source-connected runout components use SciPy labeling when available, with a
  deterministic NumPy propagation fallback for environments whose SciPy
  labeling extension is unavailable or incompatible.
- The summary records raster-read, core-compute, and output-write timings for
  future profiling.
