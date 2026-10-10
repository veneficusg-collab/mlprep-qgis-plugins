# MLPREP Detection Suite v4.2

**Status: superseded by [v4.4](../MLPREP_Plugin_v4_4/README.md).** Kept for reference.
[Back to the repository overview](../README.md).

> **This folder cannot run as stored.** `ml-prep-scripts/`, which holds the slope-unit and
> zonal-statistics engine, is empty in the repository. See [Known issues](#known-issues).

v4.2 has the same interface and stages as v4.1. The one change is in what the runout stage
receives: PINN inference computes a second factor of safety outside the model and uses it
to filter the susceptibility before routing.

This guide was written from the source code in this folder. For installation, the input
rasters and their units, and the Landslide Detection tab, use the
[v4.4 guide](../MLPREP_Plugin_v4_4/README.md); only the differences are listed here.

## Contents

- [Changes from v4.1](#changes-from-v41)
- [Differences from v4.4](#differences-from-v44)
- [Installation](#installation)
- [What Run Pipeline does](#what-run-pipeline-does)
- [Output files](#output-files)
- [Known issues](#known-issues)

## Changes from v4.1

`pinn_inference.py` is the only file with a change in behaviour:

- **`FoS_DFI`.** A factor of safety computed for each slope unit from the predicted
  cohesion and friction angle, the model's saturation ratio, and the slope unit's mean soil
  thickness, unit weight and slope. It is limited to the range 0.01 to 10.
- **`Susceptibility_DFI`.** The PINN susceptibility, set to 0 where `FoS_DFI` is 1.25 or
  more.
- **`Overall_Output.tif`** now holds `Susceptibility_DFI` instead of the unfiltered
  susceptibility. This is the raster the routing engine receives, so slope units with a
  `FoS_DFI` of 1.25 or more no longer start runout.

`merged_wrapper.py` differs from v4.1 in one default value that the plugin overrides, so
the routing settings are the same as in v4.1.

## Differences from v4.4

| | v4.2 | v4.4 |
| --- | --- | --- |
| EIL inputs | DEM + 10 rasters (no NDVI row) | DEM + 11 rasters |
| DFI used by the routing engine | Filtered PINN susceptibility | Output of the trained ML-DFI model |
| Runout sources | Every cell with a filtered susceptibility above 0 | Cells in slope units with susceptibility of at least 0.5 and a factor of safety of 1.5 or less |
| Routing engine `--threshold` | 0.2 | 0.01 |
| Step 5 | Post-depositional spread, run with placeholder inputs | Cleanup: removes runout fed by under 200 m² of source area, adds a 5 m buffer |
| Step 6 (damming potential) | Requested, but never runs | Not requested |
| Final runout raster | `step5_pds_output/post_depositional_combined_mask.tif` | `PostDep_DepZone.tif` |
| ML-DFI model (`model_dfi/`), sample data | Not included | Included |
| Landslide detection CNN (`model_lsi/default/best.keras`) | Included | Not included |

## Installation

1. Restore the engine first: `ml-prep-scripts/` must contain `main.go` and its companion
   files before the plugin can generate slope units.
2. Follow the [v4.4 installation steps](../MLPREP_Plugin_v4_4/README.md#installation),
   using `MLPREP_Plugin_v4_2` wherever the folder name appears. The plugin is listed in
   QGIS as **MLPREP Detection V4.2**.

`xgboost` is not needed for this version, because the ML-DFI model is not used.
`rasterstats` and `scikit-image` are still needed. All versions create and reuse the same
Python environments in `~/.mlprep_envs`.

## What Run Pipeline does

| Stage | What happens |
| --- | --- |
| 1. Slope units and zonal statistics | Same as v4.4, once the engine is restored |
| 2. Seam cleanup | Same as v4.4 |
| 3. PINN inference | Same model as v4.4. Writes `Final_Hazard_Map.gpkg` and `overall_output/Overall_Output.tif`, which holds the filtered susceptibility |
| 4. Runout routing | Fills pits and computes D-Infinity flow directions with TauDEM, looks up the baseline angle of reach from slope, and runs the routing engine with the filtered susceptibility as the DFI |
| 5. Post-depositional spread | Spreads the edge of the runout footprint sideways, by 5 to 30 m, where the DFI is at least 0.69 and the ground rises no more than 1 m (see [Known issues](#known-issues)) |
| 6. Damming potential | Does not run (see [Known issues](#known-issues)) |

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
| `deposition_zone_output/` | Not loaded | TauDEM grids, the alpha, source and DFI rasters, and the routing engine's outputs |

`Initial_DepZone.tif`, `PostDep_DepZone.tif`, `step5_cleanup_output/` and
`overall_output/objective2_rasters/` are not written by this version. Despite their names,
the two `ML_DFI_*` rasters are not produced with the ML-DFI model.

**Fields of `Final_Hazard_Map.gpkg`:** `Hazard_Susceptibility`, `FactorOfSafety`,
`Displacement`, `Cohesion`, `Internal_Friction`, `FoS_DFI`, `Susceptibility_DFI`. Their
meanings and units are in the
[v4.4 guide](../MLPREP_Plugin_v4_4/README.md#reading-the-results).

## Known issues

Specific to this version:

- **The slope-unit engine is missing.** `ml-prep-scripts/` was committed as a link to
  another repository (a git submodule entry) with no address recorded for it, so cloning
  leaves the folder empty and the first stage of a run fails. `MLPREP_Plugin_v4_1/` has an
  engine for the same set of inputs. Whether it is identical to the one v4.2 was developed
  with cannot be confirmed from this repository.
- **The DFI is a stand-in.** The routing engine is designed to take a depositional
  favourability index from the ML-DFI model. This version gives it the filtered PINN
  susceptibility instead.
- **Step 5 runs on placeholder inputs.** The plugin gives the spread step a stream mask of
  all zeros and a source contributing area of 4400 in every cell, a value the code describes
  as the maximum-spread fallback.
- **Step 6 never runs.** The plugin looks for
  `step6_landslide_damming_potential_PDS.py`, but the file in the folder is named
  `step6_landslide_damming_potential_LDP.py`. No damming-potential output is produced.

Shared with v4.4, and described in its [Known issues](../MLPREP_Plugin_v4_4/README.md#known-issues):
packages missing from the requirements files, the routing engine's macOS build being in
`build_mac/` instead of `build_manual/`, `roi_data/` and the preprocessing manifest not
being in the repository, macOS tool paths, the "Number of Landslides" axis label, and the
behaviour of Cancel, the progress bar and the DEM `...` button.

## Citation and licence

See the [repository overview](../README.md#citation).
