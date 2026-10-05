# Step 5 Cleanup Only

## Purpose

This standalone Python tool cleans the Step 4 runout mask and applies a minimal
fixed buffer without running the full terrain-conditioned Step 5 spreading
workflow. It is intended for the time-constrained implementation stage.

The tool is self-contained inside this folder. It does not import scripts from
the full Step 5 folder or from the older `Codes` workspace.

## What It Does

The cleanup is applied in this order:

1. Convert the configured minimum source area from square metres to the raw
   weighted-source-cell-count units stored in the source-contributing-area
   raster.
2. Remove runout cells below that threshold.
3. Divide the remaining runout into 8-neighbor connected components.
4. Remove components with no source overlap or 8-neighbor source contact.
5. Keep components with at least one source overlap or side contact.
6. Keep components with at least two diagonal-only source anchors.
7. Directionally validate components with exactly one diagonal-only source
   anchor:
   - at least one diagonally adjacent source cell must route a qualifying
     TauDEM D-Infinity branch into the anchor;
   - routing then starts at that anchor;
   - only component cells reachable downstream through qualifying D-Infinity
     branches are retained.
8. Identify simplified edge seeds from the cleaned runout. A seed must be a
   non-source runout cell touching at least one valid outside-runout cell in an
   8-neighbor window.
9. Apply a fixed `edge_seed_buffer_m` around every seed and combine the added
   cells with the cleaned runout to create the final depositional-zone mask.

This targeted rule fixes the case where one corner-touching runout cell keeps
an entire component even though the adjacent source does not flow into it. It
does not impose the stricter directional trace on components with stronger
source contact, limiting unnecessary pruning.

## Fixed Edge Buffer

The default buffer is `5 m`. Buffer distance is measured between cell centers
using map-unit Euclidean distance. On the current 5 m square grid, the four
orthogonal neighbors are 5 m away and are eligible; diagonal neighbors are
approximately 7.07 m away and are not included.

Added spread cells must be inside the valid raster domain and must not be:

- part of the final cleaned-runout mask; or
- marked as a source cell.

The conservative envelope may cover a cell rejected by the cleanup. Such a
cell remains `0` in the cleaned-runout mask and `1` in the removed-runout mask,
but is also `1` in the spread and combined masks because it lies within 5 m of
an eligible seed. Keeping these products separate prevents the buffer from
being mistaken for a reversal of the cleanup. The combined output thickens
narrow one-cell traces; it does not delete their centerline cells.

## D-Infinity Rule

The flow-direction raster must use TauDEM radians in the range `0..2*pi`.
TauDEM can divide flow between two neighboring cells. A branch is followed
when its calculated proportion is greater than or equal to
`min_dinf_flow_proportion`.

The default is `0.2`, matching the active Step 4 branch threshold. There is no
extra lateral-cell tolerance: both legitimate TauDEM split-flow recipients are
already evaluated, while unrelated neighboring cells are not admitted.

## Source-Area Rule

The default minimum contributing source area is `200 m2`:

```text
raw threshold = minimum source area / cell area
```

For a 5 m grid, cell area is `25 m2`, so the raw threshold is `8`. Values below
8 are removed; exactly 8 is retained.

## Required Inputs

All inputs must be single-band rasters with identical rows, columns, affine
transform, projected CRS, and pixel alignment.

- `runout_mask_raster`: binary Step 4 runout mask (`1` runout, `0` background).
- `source_mask_raster`: binary source mask (`1` source, `0` non-source).
- `source_contributing_area_raster`: non-negative TauDEM D-Infinity weighted
  source-cell count generated from the source mask.
- `dinf_flow_raster`: TauDEM D-Infinity direction in radians.

The default configuration uses:

```text
Sample Data/Output/Latest Runs/Step4/step4_runout_mask.tif
Sample Data/Input/Raster/Shared Inputs/Makilala_EIL_PRSTM51N.tif
Sample Data/Input/Raster/Step5 Inputs/Makilala_SWCA_PRSTM51N.tif
Sample Data/Input/Raster/Shared Inputs/Makilala_DFD_PRSTM51N.tif
```

## Outputs

Each run writes to the configured `output_dir`:

- `post_depositional_cleaned_runout_mask.tif`: final retained runout.
- `post_depositional_removed_runout_mask.tif`: all cells removed by any rule.
- `post_depositional_directionally_removed_runout_mask.tif`: cells removed
  specifically by the single-diagonal-anchor D-Infinity validation.
- `post_depositional_edge_seed_mask.tif`: simplified non-source edge seeds.
- `post_depositional_spread_mask.tif`: valid cells added by the fixed buffer.
- `post_depositional_combined_mask.tif`: cleaned runout plus fixed-buffer
  spread. This is the final depositional-zone footprint from this tool.
- `post_depositional_cleanup_summary.json`: input paths, parameters, grid
  metadata, runtime, and counts for every cleanup stage.

Use a new, descriptive output folder for each run. The default config includes
a dated run identifier and does not overwrite existing output unless
`--overwrite` is supplied.

## Run

From the workspace root:

```powershell
python .\steps\step5_cleanup_only\step5_cleanup_only.py
```

The Windows launcher is equivalent:

```powershell
.\steps\step5_cleanup_only\run_step5_cleanup_only.bat
```

Use another configuration:

```powershell
python .\steps\step5_cleanup_only\step5_cleanup_only.py `
  --config .\path\to\cleanup_config.json
```

Replace outputs in the configured folder:

```powershell
python .\steps\step5_cleanup_only\step5_cleanup_only.py --overwrite
```

Paths in a configuration resolve relative to that configuration file.

## Dependencies

The script uses Python, NumPy, Rasterio, and SciPy as declared in the workspace
`requirements.txt`. It does not require QGIS. Dependencies are intentionally
not bundled or downloaded by this folder; install them on the destination
desktop using the workspace setup instructions.

## What It Does Not Do

- No variable-radius or terrain-conditioned post-depositional spreading.
- No DFI thresholding.
- No stream buffering.
- No additional alpha-angle, beta-angle, or runout-length cap.
- No DEM-based monotonic-elevation rule.
- No source-identity propagation for components with stronger contact.

The source-contributing-area raster proves that some source contribution is
upstream, but it does not identify a particular source. The targeted
D-Infinity check establishes explicit source-to-anchor and downstream routing
only for the weak single-diagonal-anchor components. This is a conservative
correction, not a complete sediment-transport model.

## Makilala Validation Run

The default configuration was validated on 2026-09-17. The new output is under
`Sample Data/Output/Step5_CleanupOnly/step5_cleanup_dinf_buffer5m_20260917`.

- Original Step 4 runout: 6,921,500 cells.
- Removed below the 200 m2 source-area threshold: 2,604,188 cells.
- Removed as source-disconnected: 5,979 cells.
- Single-diagonal-anchor components checked: 388 components / 1,216 cells.
- Components accepting D-Infinity entry: 238; rejecting entry: 150.
- Directionally retained weak-component cells: 593.
- Directionally removed cells: 623.
- Final cleaned runout: 4,310,710 cells.
- Simplified non-source edge seeds: 204,076 cells.
- Cells added by the fixed 5 m buffer: 158,449 cells / 3.961225 km2.
- Added envelope cells that overlap rejected runout: 48,414 cells.
- Final combined depositional zone: 4,469,159 cells / 111.728975 km2.

For the 183 multi-cell sequences that motivated this correction, 62 were
removed completely, 74 were trimmed to their routed subset, and 47 were fully
retained because D-Infinity supported every cell. The validation confirmed
exact grid alignment, binary output values, complete cleaned/removed accounting,
and no overlap between the cleaned and removed masks.

An independent reconstruction matched the seed, spread, and combined masks
cell for cell. The spread contains no source or NoData cells. Of 9,165 strict
straight one-cell runout cells, 8,844 (96.5%) gained perpendicular width and
6,009 (65.56%) gained width on both sides.

## Tests

Run the focused tests from the workspace root:

```powershell
python -m pytest tests\test_step5_cleanup_only.py -q
```

The tests cover the area threshold, correct and incorrect diagonal inflow,
downstream-only retention, disconnected islands, exact-threshold behavior,
simplified seed selection, 5 m buffer geometry, prevention of source spreading,
separate treatment of removed runout inside the conservative envelope, and
invalid input handling.
