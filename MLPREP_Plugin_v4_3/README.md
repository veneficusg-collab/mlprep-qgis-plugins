# MLPREP Detection Suite v4.3

**Status: superseded by [v4.4](../MLPREP_Plugin_v4_4/README.md).** Kept for reference.
[Back to the repository overview](../README.md).

v4.3 is the first version in which the runout stage uses the trained ML-DFI model. It adds
the NDVI input, computes the model's eight predictors from the DEM and NDVI during the run,
and passes the resulting depositional favourability index (DFI) to the routing engine.
Slope-unit generation, zonal statistics, PINN inference and the Landslide Detection tab
work as they do in v4.4.

This guide was written from the source code in this folder. For installation, the input
rasters and their units, and the Landslide Detection tab, use the
[v4.4 guide](../MLPREP_Plugin_v4_4/README.md); only the differences are listed here.

## Contents

- [Differences from v4.4](#differences-from-v44)
- [Changes from v4.2](#changes-from-v42)
- [Installation](#installation)
- [What Run Pipeline does](#what-run-pipeline-does)
- [Output files](#output-files)
- [Known issues](#known-issues)

## Differences from v4.4

| | v4.3 | v4.4 |
| --- | --- | --- |
| Runout sources | Every cell with a filtered susceptibility above 0.01 and a usable baseline alpha | Cells in slope units with susceptibility of at least 0.5 and a factor of safety of 1.5 or less |
| Distance to stream and to ridge | Straight-line distance | TauDEM flow-path distance |
| Stream threshold | 100 | 1500 |
| Smoothing of continuous predictors | None | 3 × 3 mean filter |
| Step 5 | Post-depositional spread, run with placeholder inputs | Cleanup: removes runout fed by under 200 m² of source area, adds a 5 m buffer |
| Step 6 (damming potential) | Requested, but never runs | Not requested |
| Final runout raster | `step5_pds_output/post_depositional_combined_mask.tif` | `PostDep_DepZone.tif` |
| Slope-unit output fields | No `Wetness` field | `Wetness` added |
| Per-parameter rasters (`objective2_rasters/`) | Not written | Written |
| Command-line runner (`web_runner.py`) | Not included | Included |
| Landslide detection CNN (`model_lsi/default/best.keras`) | Included | Not included |
| Sample data | Complete, 2.7 GB | Rasters over 50 MB excluded |

The filtered susceptibility is the PINN susceptibility of the slope unit, set to 0 where a
factor of safety recomputed outside the model (`FoS_DFI`) is 1.25 or more.

## Changes from v4.2

- New required input row, **NDVI Raster**.
- The trained ML-DFI model is included in `model_dfi/` and applied during the run. v4.2
  passed the filtered PINN susceptibility to the routing engine in place of a DFI.
- The baseline alpha raster comes from the Step 0 script instead of code inside the plugin.
- The routing engine's `--threshold` setting is 0.01 instead of 0.2.
- `Sample Data/` added, with inputs and outputs of the runout workflow for Makilala.

## Installation

Follow the [v4.4 installation steps](../MLPREP_Plugin_v4_4/README.md#installation), using
`MLPREP_Plugin_v4_3` wherever the folder name appears. The plugin is listed in QGIS as
**MLPREP Detection V4.3**.

All versions create and reuse the same Python environments in `~/.mlprep_envs`.

This folder is 2.8 GB, most of it `Sample Data/`. The sample data is not needed to run the
plugin.

## What Run Pipeline does

| Stage | What happens |
| --- | --- |
| 1. Slope units and zonal statistics | Same as v4.4 |
| 2. Seam cleanup | Same as v4.4 |
| 3. PINN inference | Same model as v4.4. Writes `Final_Hazard_Map.gpkg` and `overall_output/Overall_Output.tif`, which holds the filtered susceptibility |
| 4. Runout routing | Computes the predictors, applies the ML-DFI model, builds the source raster, and runs the routing engine |
| 5. Post-depositional spread | Spreads the edge of the runout footprint sideways, by 5 to 30 m, where the DFI is at least 0.69 and the ground rises no more than 1 m (see [Known issues](#known-issues)) |
| 6. Damming potential | Does not run (see [Known issues](#known-issues)) |

**Predictors computed in stage 4.** Topographic wetness index, slope, profile curvature,
plan curvature, NDVI, a six-class Weiss landform map, distance to ridge and distance to
stream.

**Routing.** As in v4.4, routing follows D-Infinity flow paths on 4 MPI processes and stops
when the angle from the source falls below the path's angle of reach. The baseline angle of
reach is looked up from the local slope, and the DFI adjusts it along the path (`--dfi-mid`
0.69, `--alpha-gain-per-meter` 0.03333). Cells where the model made no prediction are given
a DFI of 0.69, which leaves the angle unchanged.

The run ends with this line in the log, not with a completion message:

```
[Warning] Step 6 Script not found at ... Pipeline complete.
```

## Output files

Files that differ from the [v4.4 list](../MLPREP_Plugin_v4_4/README.md#output-files):

| File | Layer name in QGIS | Contains |
| --- | --- | --- |
| `ML_DFI_Depositional_Zone.tif` | ML-DFI Depositional Zone | Runout cells beyond the source cells |
| `ML_DFI_Full_Runout_Track.tif` | Not loaded | Runout footprint including the source cells |
| `step5_pds_output/post_depositional_combined_mask.tif` | PDS Combined Runout Footprint | Runout footprint after the spread step |
| `step5_pds_output/` | Not loaded | Seed, spread, cleaned and removed masks, with a JSON summary |
| `deposition_zone_output/` | Not loaded | TauDEM grids, the eight predictor rasters, `ml_dfi_probability.tif`, the source raster and the routing engine's outputs |

`Initial_DepZone.tif`, `PostDep_DepZone.tif`, `step5_cleanup_output/` and
`overall_output/objective2_rasters/` are not written by this version.

**Fields of `Final_Hazard_Map.gpkg`:** `Hazard_Susceptibility`, `FactorOfSafety`,
`Displacement`, `Cohesion`, `Internal_Friction`, `FoS_DFI`, `Susceptibility_DFI`. Their
meanings and units are in the
[v4.4 guide](../MLPREP_Plugin_v4_4/README.md#reading-the-results).

## Known issues

Specific to this version:

- **Step 5 runs on placeholder inputs.** The spread step needs a stream mask and a source
  contributing area. The plugin gives it a stream mask of all zeros and a contributing area
  of 4400 in every cell, a value the code describes as the maximum-spread fallback. The
  spread therefore does not respond to streams or to how much source area feeds each cell.
- **Step 6 never runs.** The plugin looks for
  `step6_landslide_damming_potential_PDS.py`, but the file in the folder is named
  `step6_landslide_damming_potential_LDP.py`. No damming-potential output is produced.
- **Distances are straight-line.** Distance to stream and distance to ridge are computed as
  straight-line distances with a stream threshold of 100. v4.4 changed both to match the
  data the ML-DFI model was trained on.
- **Sources are not refined.** Any cell with a filtered susceptibility above 0.01 starts a
  runout path.
- **Unused files.** `step5_cleanup_only/`, `merged_wrapper_step4_fix.py`,
  `diagnose_step4_inputs.py` and `test_step4_complete.py` are in the folder but are not
  called by the plugin.
- **Large files.** 31 files in `Sample Data/` are over 50 MB.

Shared with v4.4, and described in its [Known issues](../MLPREP_Plugin_v4_4/README.md#known-issues):
packages missing from the requirements files, the routing engine's macOS build being in
`build_mac/` instead of `build_manual/`, `roi_data/` and the preprocessing manifest not
being in the repository, macOS tool paths, the "Number of Landslides" axis label, and the
behaviour of Cancel, the progress bar and the DEM `...` button.

## Citation and licence

See the [repository overview](../README.md#citation).
