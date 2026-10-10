# MLPREP Detection Suite

QGIS plugins for earthquake-induced landslide (EIL) susceptibility mapping with a
physics-informed neural network (PINN), developed under the ML-PREP Project.

From one window the plugin generates slope units from a DEM, aggregates predictor rasters
over them, runs the trained PINN, and routes potential runout from the susceptible slopes.
A second tab detects landslides that have already happened from Sentinel-2 imagery.

This repository keeps four successive versions side by side, each in its own folder with
its own README.

> **Use `MLPREP_Plugin_v4_4/` on branch `MLPREP_Plugin_V4_3`.**
> The older branch `MLPREP_Plugin_V4_1` holds only two files of the v4.4 folder.

## Versions

| Version | Status | EIL inputs | Runout stage | Guide |
| --- | --- | --- | --- | --- |
| **v4.4** | Current | DEM + 11 rasters | ML-DFI model with predictors matched to its training data; sources refined by factor of safety; cleanup step | [MLPREP_Plugin_v4_4](MLPREP_Plugin_v4_4/README.md) |
| v4.3 | Superseded | DEM + 11 rasters | First version to apply the ML-DFI model; sources taken from filtered susceptibility | [MLPREP_Plugin_v4_3](MLPREP_Plugin_v4_3/README.md) |
| v4.2 | Superseded; slope-unit engine files are missing from the repository | DEM + 10 rasters | PINN susceptibility, filtered by a factor of safety, used in place of the ML-DFI model | [MLPREP_Plugin_v4_2](MLPREP_Plugin_v4_2/README.md) |
| v4.1 | Superseded | DEM + 10 rasters | PINN susceptibility used in place of the ML-DFI model | [MLPREP_Plugin_v4_1](MLPREP_Plugin_v4_1/README.md) |

The v4.1 README is the one written with that version and has been left as it was. The
other three were written later from the source code.

All four versions load the same trained PINN (`model/my_model.keras`), use the same
slope-unit settings, and share the same interface layout. They differ in how runout is
modelled after the PINN has run.

## What a run does

![The five stages of a v4.4 run, the files each writes and the layer each adds to QGIS](docs/images/run-sequence.png)

The figure shows v4.4. Stages 1 to 3 are the same in every version. Stages 4 and 5 are
where the versions differ; each version's README lists its own.

## Getting started

Installation, the input rasters and their units, a step-by-step run, the output files and
the error messages are all in the [v4.4 guide](MLPREP_Plugin_v4_4/README.md).

```bash
git clone --branch MLPREP_Plugin_V4_3 https://github.com/veneficusg-collab/mlprep-qgis-plugins.git
```

## What changed between versions

**v4.1 (August 2026).** First version in this repository. After the PINN runs, the routing
engine is given the PINN susceptibility raster in place of a depositional favourability
index (DFI), and every cell with susceptibility above 0 is a runout source. Its runout
stage has the limitations listed under
[Known issues in the v4.2 README](MLPREP_Plugin_v4_2/README.md#known-issues), apart from
the missing engine files.

**v4.2.** Inference also writes a second factor of safety (`FoS_DFI`) and a filtered
susceptibility (`Susceptibility_DFI`, zero where `FoS_DFI` is 1.25 or more). The filtered
raster is what the routing engine receives.

**v4.3.** Adds the NDVI input and the trained ML-DFI model. The plugin now computes the
DFI predictors from the DEM and NDVI and applies the model, instead of reusing
susceptibility as the DFI. Adds sample data for the runout workflow.

**v4.4 (October 2026).** Runout sources are refined by susceptibility and factor of safety.
The DFI predictors are recomputed to match the data the model was trained on. The
post-depositional spread step is replaced by a cleanup step, and the final footprint is
written as `PostDep_DepZone.tif`.

## Things that are the same in every version

- **Classes in the municipal summary.** Slope units are counted as High above a
  susceptibility of 0.66, Moderate from 0.33 to 0.66, and Low below 0.33.
- **Slope units.** `r.slopeunits.create` with `thresh` 5000 m², `areamin` 5000 m², `cvmin`
  0.15, `rf` 10 and a default of 10 iterations, on 3 × 3 tiles with a 500-pixel overlap.
- **Requirements.** QGIS 3, Go, GRASS GIS with `r.slopeunits`, GDAL, TauDEM, MPI and the
  compiled routing engine. The tool paths in the code target macOS with Homebrew.
- **Not included.** The administrative boundary shapefiles (`roi_data/`) and the
  preprocessing manifest that inference looks for are not in the repository.

## Repository layout

| Path | Contents |
| --- | --- |
| `MLPREP_Plugin_v4_4/` | Current plugin |
| `MLPREP_Plugin_v4_3/`, `MLPREP_Plugin_v4_2/`, `MLPREP_Plugin_v4_1/` | Earlier versions, kept for reference |
| `docs/images/` | Figures used in the READMEs |

An earlier prototype from March 2026, without slope-unit generation or runout, is in a
separate repository: [mlprep-qgis-plugin](https://github.com/veneficusg-collab/mlprep-qgis-plugin).

## Citation

If you use this plugin, please cite:

> Cubillas, J.E.D.; Gonzales, G.L.; Puspus, M.A.B.; et al. A QGIS Plugin for
> Earthquake-Induced Landslide Susceptibility Mapping Using Physics-Informed Neural
> Networks. Presented at the 3rd International Conference on AI Sensors and Transducers,
> Jeju, South Korea, 2–7 August 2026.

## Acknowledgements

Funded by the Department of Science and Technology–Philippine Council for Industry, Energy
and Emerging Technology Research and Development (DOST-PCIEERD) under the ML-PREP Project.
The landslide inventory and PGA maps used in development were provided by DOST-PHIVOLCS.

## Licence

No licence file has been added to this repository yet.
