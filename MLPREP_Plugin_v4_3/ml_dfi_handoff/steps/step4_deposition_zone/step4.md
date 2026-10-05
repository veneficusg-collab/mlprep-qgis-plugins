# Step 4: Dynamic-Alpha Deposition-Zone Routing

Step 4 is the Codes_V2 dynamic-alpha terrain-factor runtime. The active runtime
uses a Python config/reporting wrapper around a C++/MPI routing executable. It
turns a source raster plus alpha-angle input into a terrain-following runout and
deposition-zone simulation.

In plain terms, it does this:

1. Finds the cells where runout is allowed to start.
2. Gives each source cell a starting alpha angle.
3. Follows TauDEM-style D-Infinity flow routing downslope.
4. Adjusts the carried alpha angle at each accepted downslope cell using DFI.
5. Computes the beta angle from the original source cell to the candidate cell.
6. Keeps the path going only while `beta >= dynamic_alpha`.

The result is a runout mask, plus supporting rasters that explain how the model
reached that footprint. The active wrapper does not import modules from the
original Objective 4 `Codes` workspace.

## Files

- `step4_deposition_zone_mpi_wrapper.py` - active Python CLI wrapper.
- `step4_common.py` - shared config, raster validation, and reporting helpers.
- `cpp_mpi_port/step4_deposition_zone_mpi.cpp` - C++/MPI routing core.
- `cpp_mpi_port/third_party/taudem` - the small licensed TauDEM source subset
  required to rebuild the core locally.
- `archive/historical_reference/old_step4_deposition_zone.py` - historical
  Python reference only, not an active fallback.
- `config.default.json` - default Codes_V2 run config.

Regenerable build/cache artifacts are intentionally not maintained. The active
prebuilt executable is kept at
`cpp_mpi_port/build_manual/step4_deposition_zone_mpi.exe`.

## How To Run

Verify native dependencies without opening raster inputs:

```powershell
.\steps\step4_deposition_zone\run_step4.bat --check-environment
```

For a config/raster preflight without launching MPI:

```powershell
.\steps\step4_deposition_zone\run_step4.bat --config .\steps\step4_deposition_zone\config.default.json --dry-run
```

For a real run:

```powershell
.\steps\step4_deposition_zone\run_step4.bat --config .\steps\step4_deposition_zone\config.default.json --processes 8
```

Make `mpiexec.exe` available on `PATH`, pass it with `--mpiexec`, or set the
`MPIEXEC` environment variable.

Set `OSGEO4W_ROOT` to the local QGIS/OSGeo4W installation folder, unless
`python-qgis-ltr.bat` is already on `PATH`. Set `TAUDEM_DLL_DIR` to the folder
containing the TauDEM-compatible runtime `gdal.dll`. `TAUDEM_GDAL_DLL` is used
only when rebuilding the executable and must point to the DLL file itself.
Config paths resolve relative to the config file.

## Required Inputs

- `dem_fel_raster`
  Filled DEM used for elevations and path geometry.
- `dinf_flow_raster`
  TauDEM D-Infinity flow-direction raster in radians.
- `source_raster`
  Raster showing where runout can begin.
- `dfi_raster`
  Terrain-factor raster used to adjust alpha as the path moves.
- `alpha_raster`
  Required raster of baseline alpha angles.

## Default Config Inputs

The default `config.default.json` uses the handoff production chain:

- DEM: `Sample Data/Input/Raster/Shared Inputs/Makilala_DEM_Fill_PRSTM51N.tif`
- D-Infinity flow: `Sample Data/Input/Raster/Shared Inputs/Makilala_DFD_PRSTM51N.tif`
- Source mask: `Sample Data/Input/Raster/Shared Inputs/Makilala_EIL_PRSTM51N.tif`
- DFI: `Sample Data/Input/Raster/Shared Inputs/Makilala_DFI_PRSTM51N.tif`
- Alpha: `Sample Data/Input/Raster/Step4 Inputs/Makilala_AlphaRaster_PRSTM51N.tif`

This makes the required dependency chain explicit: Step 3 prepares the DFI
probability raster, Step 0 prepares the source-slope alpha raster, and both are
required to generate the Step 4 depositional-zone map.

Because Step 4 resolves relative paths from the config file location, paths
inside `config.default.json` use `../Sample Data/...`.

## Deprecated Alpha-Build Path

The standalone runtime no longer builds alpha from slope units, CSV tables, or
Step 1 prediction contracts. Prepare an aligned `alpha_raster` before Step 4.

## Main Outputs

Step 4 always writes the complete diagnostic raster bundle. `output_mode` is no
longer a config field.

- `output_dynamic_alpha`
  Final dynamic alpha angle at each accepted runout cell.
- `output_beta_angle`
  Final winning beta angle at each accepted runout cell.
- `output_dfs`
  Downslope flow-path distance from the winning source cell.
- `output_mask`
  Binary runout mask.
- `output_depositional_mask`
  Binary mask of accepted runout cells that are not source cells.
- `output_summary_json`
  Run receipt with inputs, parameters, counts, and output paths.

When `write_debug_rasters = true`, Step 4 additionally writes the parent
dynamic-alpha raster. Accepted beta is already represented by
`output_beta_angle`, and DFI is never clamped.

When DFI is fixed across comparisons, keep output names short and identify only
the alpha input and gain rate you changed, for example:

```text
sourceslope_gain_per_m_0p03333.tif
sourceslope_gain_per_m_0p2.tif
```

## The Full Runtime, Start To Finish

### 1. Config Loading

The config loader reads paths and scalar parameters, normalizes them, and
enforces these rules before raster work:

- unknown config fields fail instead of being silently ignored
- numeric parameters must be finite
- `proportion_threshold` and `dfi_mid` must be between `0` and `1`
- `alpha_gain_per_meter` must be `>= 0`
- `alpha_raster` is required and must point to an aligned alpha-angle raster
- output paths must be unique and must not overwrite input paths

### Fixed Non-Configurable Defaults

These are embedded in the active Step 4 runtime and are intentionally not
accepted in the JSON config. If an old config includes any of these keys, the
run fails with a clear message.

| Setting | Fixed value | Failure or warning behavior |
| --- | ---: | --- |
| `alpha_min` | `6.0` | Initiating source alpha below 6 degrees fails the run; dynamic alpha cannot fall below this value. |
| `alpha_max` | `72.0` | Initiating source alpha above 72 degrees fails the run; dynamic alpha cannot rise above this value. |
| `clamp_dfi_to_unit_interval` | disabled | Valid DFI values must already be within `0` to `1`; values like `-0.1` or `1.2` fail the run. |
| `dfi_deadband` | `0.05` | DFI departures from `dfi_mid` smaller than 0.05 do not change alpha. |
| `beta_tie_tolerance` | `1e-6` | Winning-path beta ties are resolved with this fixed tolerance. |
| `clamp_initial_source_alpha` | `false` | Source alpha is not clamped. The run fails if initiating source alpha is outside fixed `alpha_min`/`alpha_max`. |
| `fail_on_unprocessed_cells` | `true` | The run fails when unresolved routing cells exceed the fixed allowance. |
| `max_unprocessed_fraction` | `0.0` | No unresolved routing cells are allowed. |
| `require_binary_source_raster` | `true` | The source raster must contain only `0`, `1`, or NoData. |
| `max_source_fraction` | `0.30` | A warning is logged if initiating source cells exceed 30% of terrain-valid cells. |
| `min_dfi_valid_fraction` | `0.95` | A warning is logged if valid DFI coverage is below 95% of terrain-valid cells. |

### 2. Fast Metadata Preflight

The Python wrapper reads only raster metadata before launching MPI. It verifies
paths, dimensions, geotransforms, CRS, projected meter units, and output
locations without loading five complete raster arrays. Distributed pixel-value
validation is then performed once by the C++ engine while each rank reads its
own partition.

### 3. Alpha Preparation

The runtime reads `alpha_raster` directly. The alpha raster must contain
baseline alpha angles in degrees and must be aligned with the DEM reference
grid.

### 4. Alignment Checks

Before the model runs, all rasters must match the DEM exactly in:

- rows and columns
- geotransform
- CRS

Step 4 is strict here. If one input is on a slightly different grid or has
non-matching CRS metadata, align it first.

### 5. Terrain Validity And Source Validity

Inside the dynamic-alpha engine block, the routing domain is defined like this:

- `terrain_valid = DEM valid AND D-Infinity valid`
- `valid = terrain_valid`

This means routing is allowed anywhere the DEM and flow-direction raster are
valid.

Candidate source membership is defined like this:

- `source_marked_cells = terrain_valid AND source defined AND source > 0`

Alpha input values must be valid alpha angles:

- finite alpha values on valid terrain must be between `1` and `90` degrees
- alpha NoData is allowed
- finite invalid values such as `0`, negative values, or values `> 90` fail the run
- initiating source alpha must also be within the fixed runtime bounds `6` to `72` degrees

Initiating source membership requires a valid alpha:

- `initiating_source_cells = source_marked_cells AND alpha defined AND 1 <= alpha <= 90`

A source-marked cell with alpha NoData is excluded from initiation because it
does not have an alpha value. It remains part of valid terrain and can still
receive runout propagated from an upstream initiating source.

Alpha is required only at source cells. Downstream runout cells do not need
their own alpha value.

The runtime will fail if:

- there are no valid terrain cells
- any finite alpha value on valid terrain is outside `1` to `90` degrees
- any initiating source alpha is outside the fixed `6` to `72` degree bounds
- the source raster contains values other than `0`, `1`, or NoData
- there are no initiating source cells after excluding source-marked cells with missing alpha
- there are no valid DFI cells at all
- any finite valid DFI value is outside `0` to `1`
- any terrain-valid cells remain unprocessed after routing

## What The Engine Stores Per Cell

For each accepted runout cell, the engine keeps one winning state:

- `beta_angle`
- `dfs`
- `dynamic_alpha`
- winning source elevation

Only `dynamic_alpha`, `beta_angle`, `dfs`, `runout_mask`, and
`pure_depositional_mask` are written as standard outputs. The source elevation
is retained only as internal routing state.

This is important:

- Step 4 uses a single winning-path state per cell.
- A source cell starts with its own baseline state.
- If a stronger upstream path reaches that source cell later, that source cell can be overtaken.
- After that, the cell propagates downstream using the winning upstream state.

So a source cell can also become a runout cell of an upstream source, and the
stronger path wins.

## The Core Propagation Logic

### 1. Source Initialization

Every source cell starts with:

- `dynamic_alpha = alpha[source_cell]`
- `dfs = 0`
- source elevation = DEM at the source cell

This is the seed state for each starting path.

### 2. Topological Evaluation Order

The engine computes an upstream dependency count using D-Infinity proportions
and then processes cells in a queue. This makes sure a cell is evaluated only
after its upstream contributors have had a chance to send candidate paths into
it.

### 3. Candidate Generation

For each cell, the engine inspects its upstream neighbors.

For each upstream neighbor:

- it computes the D-Infinity proportion flowing from that upstream cell into the current cell
- it rejects that candidate if the proportion is `<= 0`
- it rejects that candidate if the proportion is below `proportion_threshold`
- it rejects that candidate if the current cell has invalid DFI
- it rejects that candidate if the upstream cell does not currently hold a valid winning path state

### 4. Travel Distance

If the candidate is still alive, the engine computes:

```text
d = upstream_dfs + one_cell_travel_distance
```

The one-cell travel distance depends on grid spacing and whether the move is
cardinal or diagonal.

Step 4 applies a fixed planimetric D-Infinity flow-path length cap:

```text
d <= 200 m
```

If a candidate cell would place the accumulated source-to-cell planimetric
flow-path length above `200 m`, that candidate is rejected and propagation does
not continue through that cell. The step length uses `dx` and `dy`; it does not
include elevation change as a third distance component. This is consistent with
the map-view centerline lengths used to calibrate the cap.

### 5. Beta Angle

The engine uses the winning source elevation carried by the upstream path, not
the elevation of the current cell or a local source override:

```text
beta = degrees(atan((source_elevation - current_cell_elevation) / routed_distance))
```

This beta is always measured back to the source cell that currently owns the
path. Beta angle is NoData at source cells because beta is not meaningful at
zero flow-path distance.

### 6. Dynamic Alpha Update

At the candidate cell, the engine updates the upstream path alpha using the DFI
value of the candidate cell and the physical distance traveled into that cell:

```text
delta_alpha =
    alpha_gain_per_meter
    * (dfi_value - dfi_mid)
    * step_distance_m

alpha_candidate = clamp(alpha_parent + delta_alpha)
```

If `abs(dfi_value - dfi_mid) < 0.05`, no alpha change is applied.
The default `dfi_mid` is `0.69`, based on the optimized probability threshold
from the latest Step 3 model-training validation run.

The default is:

```text
alpha_gain_per_meter = 0.03333
```

This rate was selected after Step 4 sensitivity tests comparing `0.2` and
`0.03` per meter. A value of `0.03333` gives an accumulated distance multiplier
of approximately `1.0` over `30 m` (`0.03333 * 30 = 0.9999`) while retaining
continuous physical-distance scaling at the native raster resolution. At the
current `5 m` resolution, an orthogonal step contributes a multiplier of
approximately `0.16665`; a diagonal step uses its longer physical distance.

`alpha_gain_per_meter` replaces the former `alpha_step_gain` and
`alpha_gain_reference_distance_m` pair. Those two settings affected the update
only through their ratio and therefore could not be interpreted or calibrated
independently.

Conceptually:

- low DFI lowers dynamic alpha and tends to make continuation easier
- high DFI raises dynamic alpha and tends to make stopping easier

### 7. Alpha Floor And Cap

The adjusted alpha is always bounded by the fixed `alpha_min = 6` and
`alpha_max = 72`.

This means dynamic alpha cannot fall below 6 degrees or rise above 72 degrees.

### 8. Stopping Rule

This is the core stopping rule:

```text
continue only if beta >= dynamic_alpha
```

If `beta < dynamic_alpha`, that candidate path stops at the current cell.

If `beta >= dynamic_alpha`, the candidate is valid and competes to become the
winning path state for the current cell.

### 9. Winning Path Rule

If several valid upstream candidates reach the same cell, the engine chooses
one winner.

The rule is:

1. prefer the candidate with higher `beta`
2. if beta is tied within the fixed `beta_tie_tolerance = 1e-6`, prefer the candidate with lower `dynamic_alpha`
3. if still tied, the fixed upstream iteration order breaks the tie deterministically

The winning candidate overwrites the cell state:

- `beta_angle`
- `dfs`
- `dynamic_alpha`
- winning source elevation

When `write_debug_rasters = true`, the candidate's parent dynamic alpha is
also retained for the optional parent-alpha debug raster.

This applies to ordinary cells and to source cells.

## What Happens To Source Cells

For an initiating source cell, this is the current behavior:

1. A source cell always begins with its own baseline alpha state.
2. That baseline state is immediately eligible to propagate downslope.
3. If no stronger upstream path reaches the source cell, it keeps its own
   baseline routing state and source elevation.
4. If a stronger upstream path does reach it, the source cell is overtaken and now belongs to the upstream source in the output rasters.
5. That overtaken source cell still remains part of the runout mask.

This is the meaning of the current single winning-path model.

## Important Practical Rules

### Downstream Cells Do Not Need Alpha

This is allowed:

- source cell has valid positive alpha
- downstream non-source cell has no alpha value
- flow still continues through that downstream cell if terrain, DFI, and beta/dynamic-alpha rules permit it

### Invalid Alpha Fails, Missing Alpha Excludes Initiation

Finite alpha values must be between `1` and `90` degrees. If the alpha raster
contains `0`, negative values, or values greater than `90` on valid terrain,
Step 4 fails because the input is not a valid alpha-angle raster.

This source-marked cell will not initiate runout, but will not fail the run:

- source cell marked as `source > 0`
- the same cell has alpha NoData

It is excluded from the initiating source set and counted in the output
summary. It remains traversable terrain, so an upstream initiated path can
still move through or deposit on that cell. The run fails only if no
initiating source cells remain.

### DFI NoData Still Stops Routing

The current engine requires valid DFI at the candidate cell.

That means:

- `DFI = 0` is allowed and usually makes continuation easier
- finite DFI values must already be within `0` to `1`
- values below `0` or above `1` fail the run; Step 4 does not clamp DFI
- `DFI NoData` is not the same thing as `0`
- if a candidate cell has DFI NoData, the engine does not accept routing through that cell

### Source Footprint Size Matters

If your source raster is already extremely broad, Step 4 may add only a small
number of new downslope cells. In that case the model may still be working
correctly, but the source raster is already covering most of the terrain you
are interested in.

## Summary JSON

The summary JSON records:

- input paths
- parameter values
- config and executable SHA-256 hashes
- MPI and total-wrapper timings
- DFI range, valid coverage, and the fixed no-clamping policy
- final alpha statistics
- cell counts
- output dimensions and file sizes
- output paths
- the winning rule used by the engine

Useful counts include:

- `terrain_valid_cells`
- `alpha_valid_cells`
- `source_cells`
- `source_cells_without_valid_alpha`
- `non_source_terrain_valid_cells`
- `runout_cells`
- `pure_depositional_cells`
- `candidate_evaluations`
- `accepted_candidate_propagations`
- `candidate_cells_skipped_missing_dfi`
- fixed `max_planimetric_flow_path_length_m = 200`

## Safe Output Publication

MPI, optional validation, and summary generation write to hidden staging
directories beside the configured outputs. Step 4 validates the complete staged
bundle before replacing official files. Existing outputs and raster sidecars
are backed up during publication and restored if any replacement fails. A
failed or interrupted run therefore cannot leave a mixed old/new official
bundle.

## Output Meaning

### `dynamic_alpha`

The final winning dynamic alpha at each accepted runout cell.

### `beta_angle`

The final winning beta angle at each accepted runout cell. Source cells are
NoData because beta is not meaningful at zero flow-path distance.

### `dfs`

The routed downslope flow-path distance from the winning source cell.

### `runout_mask`

All accepted cells, including source cells and overtaken source cells.

### `pure_depositional_mask`

Accepted runout cells that are not source cells.

## Common Reasons A Run Looks Wrong

### Very Small Expansion Beyond Source Cells

Usually means one of these:

- the source raster is already huge
- many candidate cells fail the `beta >= dynamic_alpha` test
- DFI values are pushing alpha upward and stopping continuation
- DFI NoData blocks continuation

### Unexpected Alignment Error

Usually means one raster differs from the DEM in:

- grid dimensions
- geotransform
- CRS metadata

### Missing Runout From A Source-Marked Cell

Usually means one of these:

- the source-marked cell has missing alpha and was excluded from initiation
- the source cell cannot route into valid downslope terrain
- the downstream candidate cell has DFI NoData
- the candidate fails the `beta >= dynamic_alpha` test immediately

## Mental Model

The cleanest way to think about Step 4 is this:

- sources provide starting alpha
- D-Infinity provides where the path can go
- DFI changes how permissive the path is at each accepted cell
- beta measures whether the source still has enough relief to keep going
- one winning source path owns each accepted cell

That is the current maintained Step 4 behavior.
