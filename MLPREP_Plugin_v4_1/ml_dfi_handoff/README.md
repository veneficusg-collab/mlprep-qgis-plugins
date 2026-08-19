# ML-DFI Objective 4 Workspace

This workspace contains a staged geospatial Python workflow for pixel-based
ML-DFI depositional favorability modeling around the Makilala sample data.

## Current Pipeline

00. `steps/data_preparation`
   - Runs the transferred DP1 observed-alpha data-preparation workflow before
     source-slope alpha raster generation.
   - Measures per-landslide crown, toe, travel length, elevation drop,
     `h_over_l`, and observed alpha angle from landslide polygons and a DEM.
   - Requires a QGIS/OSGeo Python runtime and is self-contained inside Codes_V2;
     all DP1 source modules are copied locally under `steps/data_preparation`.
   - Active outputs are under `Sample Data/Output/Latest Runs/Step00_DP1`.

0. `steps/step0_baseline_alpha_angle`
   - Converts slope in degrees into a source-slope baseline alpha-angle raster.
   - Writes alpha, class, and metadata outputs used as Step 4 alpha inputs.
   - Active outputs are under `Sample Data/Output/Latest Runs/Step0`.

1. `steps/step1_prepare_inventory_labels`
   - Converts landslide inventory polygons and a DEM into raster training labels.
   - Final label values are `1` for the lower depositional zone, `0` for the
     upper source/transport zone, and `NoData` for middle-zone or outside-polygon cells.
   - Also writes `landslide_id_raster.tif` for downstream grouped validation.
   - Active outputs are under `Sample Data/Output/Latest Runs/Step1`.

2. `steps/step2_extract_pixel_samples`
   - Extracts all valid labeled pixels and aligned predictor values into a
     machine learning sample table.
   - Requires manual raster alignment and fails on any grid mismatch; Step 2
     does not resample or reproject inputs.
   - Carries the required `landslide_id` group raster into the sample table.
   - Active outputs are under `Sample Data/Output/Latest Runs/Step2`.

3. `steps/step3_train_ml_dfi_model`
   - Trains a probabilistic classifier for ML-derived Depositional Favorability
     Index directly from the Step 2 extracted sample table.
   - Handles class imbalance inside model training through XGBoost
     `scale_pos_weight` tuning or Random Forest class weights.
   - Uses a non-configurable grouped 70/30 training-holdout split, with
     five-fold stratified group CV by `landslide_id` inside training.
   - Training and raster-application outputs are separate bundles under
     `Sample Data/Output/Latest Runs/Step3`.

4. `steps/step4_deposition_zone`
   - Runs dynamic-alpha deposition-zone routing using DEM, D-Infinity flow,
     source, DFI, and alpha inputs.
   - The active runtime is `step4_deposition_zone_mpi_wrapper.py`, which
     launches the C++/MPI routing core and writes the full diagnostic output
     bundle.
   - Uses metadata-only Python preflight, distributed C++ value validation,
     and staged rollback-safe publication. The prebuilt executable and required
     TauDEM build-source subset are included for handoff.
   - The default config uses shared aligned DEM, D-Infinity flow, source, and
     DFI rasters under `Sample Data/Input/Raster/Shared Inputs`, plus the
     step-specific alpha raster under `Sample Data/Input/Raster/Step4 Inputs`.
   - Active outputs are under `Sample Data/Output/Latest Runs/Step4`.

5. `steps/step5_post_depositional_spread_PDS`
   - Runs post-depositional spread from the Step 4 runout mask using DFI, DEM,
     source mask, stream mask, and source contributing-area inputs.
   - Consolidated into Codes_V2 as a standalone script; it no longer imports config
     helpers or scripts from the older `Codes` workspace.
   - Active outputs are under `Sample Data/Output/Latest Runs/Step5`.

6. `steps/step6_landslide_damming_potential_LDP`
   - Screens stream cells where the Step 4 full runout mask intersects or
     approaches mapped stream channels.
   - Uses the mean runout approach direction from the upper 25 m runout window.
   - Uses a required slope raster to compute mean channel-adjacent slope over
     upper/lower 25 m windows around each candidate cell.
   - Preserves geometry-based damming potential outputs from inflow angle and
     channel slope.
   - Adds a separate reference-volume assessment using a rounded 2,600 m3
     delivered-deposit scenario derived from `Aref = 2,500 m2`, the Yunus
     shallow-landslide relation `Vsource = 0.54 * A^1.15`, and a 60% delivery
     fraction.
   - Uses corrected obstructable-valley width, not active wetted-channel width,
     with fixed thresholds `Wformation = 3.80 m` and `Wpossible = 18.78 m`.
   - Writes reference-volume valley-domain classes: `1` non-formation,
     `2` possible/uncertain obstruction, and `3` formation-domain condition met.
   - Writes combined reference-volume class/score outputs separately from the
     geometry-only class/score outputs. These are screening-level conditional
     potential outputs, not dam-formation probabilities.
   - Applies class buffers only to the higher damming-potential classes: 5 m for
     high and very-high; no buffer for low, moderate, or debris-flow connectivity favored.
   - Uses the aligned D-Infinity stream mask under
     `Sample Data/Input/Raster/Step6 Inputs`.
   - Active outputs are under `Sample Data/Output/Latest Runs/Step6`.

The active production chain for the final map is:

```text
Step 00 observed-alpha preparation
    -> Step 0 source-slope alpha + Step 3 ML-DFI probability
    -> Step 4 depositional-zone map
    -> Step 5 post-depositional spread
    -> Step 6 runout-based landslide damming potential screening
```

## Environment

Use 64-bit Python 3.13.12 for Steps 0-3. Copied virtual environments are not
portable and must be rebuilt on each desktop.

Suggested setup:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_environment.ps1
```

The setup script uses `requirements.lock.txt` by default for exact recreation.
Conda/Mamba users can use `environment.yml`. See `ENVIRONMENT.md`,
`NATIVE_DEPENDENCIES.md`, and `HANDOFF.md` for the complete tested software and
native dependency contract.

Step 00 DP1 uses a dedicated `.venv-dp1` overlay on QGIS LTR 3.40.7/OSGeo4W
Python 3.12.10. Launch it through `steps/data_preparation/launch_dp1.py`; it does not
use `.venv`, `.venv-local`, or Windows user-site packages. Step 4 and Step 5
PDS real raster execution also require GDAL/OGR/OSR
bindings from QGIS/OSGeo Python. Step 6 default raster-only
execution uses `rasterio`; optional Step 6 stream-vector direction and
candidate-point outputs require an environment with `osgeo` available too. Step
6 also requires an aligned slope raster; the default uses
`Sample Data/Input/Raster/Step6 Inputs/Makilala_Slope_LPF_PRSTM51N.tif`.

Before a Step 5 production run, use `run_step5.bat --config <config> --dry-run`
to validate its config, raster accessibility and alignment, projected metric
units, and output-path safety without loading full arrays or writing outputs.

Run the centralized fail-fast environment and data preflight before starting
the workflow, and again for the next step immediately before its official run:

```powershell
python .\scripts\check_environment.py
python .\scripts\check_environment.py --step step4
```

The checker validates exact Python/package versions, GDAL/PROJ lookup health,
QGIS, MPI, the Step 4 engine, TauDEM runtime and utilities, configured input
paths, output write permission, projected CRS and exact raster alignment, and
free output-disk space. It exits nonzero on blocking failures. The default disk
floor is 10 GiB and can be changed with `--min-free-gb`.

The handoff workflow is configured centrally. Create the ignored local override
once, then validate or launch any step:

```powershell
Copy-Item .\config\local_config.example.json .\config\local_config.json
python .\workflow_config.py validate
python .\workflow_config.py run data_preparation
python .\workflow_config.py run step0
```

Run the read-only portability check before transfer:

```powershell
python .\handoff_preflight.py
```

Create the source-only handoff archive after committing reviewed changes:

```powershell
python .\scripts\create_handoff_package.py --overwrite
```

The packager excludes local datasets, outputs, models, virtual environments,
logs, caches, local configuration, and archived experiments. See `OUTPUTS.md`
and `HANDOFF.md`.

## Diagnostics

Use `scripts/check_environment.py` for pre-run environment and configured-data
validation. The older workspace diagnostics additionally inspect retained
sample-run outputs:

```powershell
.\.venv-local\Scripts\python.exe .\tools\workspace_diagnostics.py
```

The checker is read-only. It verifies Python package imports, core script
syntax, sample data presence, Step 1 label raster readability, aligned predictor
grid compatibility, the existing Step 2 sample table, and Step 3 model outputs
when present. It also checks Step 4, Step 5, and Step 6 config dependencies.

## Configuration

The shared registry is `config/default_config.json`. Machine-specific runtime
locations, worker counts, and study input/output paths belong only in the
ignored `config/local_config.json`. See `config/schema_notes.md`.

Reviewed default JSON contracts still live beside their respective step scripts:

- `steps/data_preparation/config.default.json`
- `steps/step0_baseline_alpha_angle/config.default.json`
- `steps/step1_prepare_inventory_labels/config.default.json`
- `steps/step2_extract_pixel_samples/config.default.json`
- `steps/step3_train_ml_dfi_model/config.default.json`
- `steps/step4_deposition_zone/config.default.json`
- `steps/step5_post_depositional_spread_PDS/config.default.json`
- `steps/step6_landslide_damming_potential_LDP/config.default.json`

The plural `configs/` folder is only an archive for named experiment and
comparison configs; it is separate from the singular central `config/` folder.

Steps 1-3 accept JSON configuration as well as command-line overrides, resolve
relative paths from the JSON file's own folder, and publish official output
bundles only after staged QC succeeds. Step 0 and Steps 4-6 also use JSON
configuration records directly. Existing explicit per-step config commands
remain supported.

## Handoff Notes

- The repository was initialized after the project was copied into this
  workspace, so the first commit should avoid local caches and copied virtual
  environments.
- Generated logs may contain absolute paths from the desktop that produced the
  run. Keep them outside the source handoff; the packaging script excludes
  generated run artifacts.
- Step 3 keeps model training/validation and full-domain raster application in
  separate bundles. Its default training bundle is
  `Latest Runs/Step3/model_training_validation`.
- Future run outputs should be placed under `Sample Data/Output/Latest Runs`
  for active outputs. Older completed runs may be archived outside the
  workspace, or under `Sample Data/Output/Old Runs` only when they are still
  actively needed for comparison.
- Step 4 CRS alignment keeps raster size and transform strict, but now tolerates
  harmless CRS WKT precision differences after exact OSR equality and authority
  checks fail.
- Shared aligned rasters used by multiple later steps live under
  `Sample Data/Input/Raster/Shared Inputs`; step-specific inputs remain in their
  own `Step4 Inputs`, `Step5 Inputs`, or `Step6 Inputs` folders.
