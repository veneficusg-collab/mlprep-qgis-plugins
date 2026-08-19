# Central Workflow Configuration

## Purpose

`default_config.json` is the shared workflow registry. It points to the reviewed
default config beside each step and contains only portable runtime defaults.
Do not add desktop-specific paths to it.

For a workstation or study area:

```powershell
Copy-Item .\config\local_config.example.json .\config\local_config.json
```

Edit only `local_config.json`. It is ignored by Git and is merged in memory over
the shared defaults. Neither the master default nor the per-step defaults are
rewritten.

## Merge Rules

- Objects are merged recursively.
- Scalars and arrays in the local file replace the corresponding shared value.
- Put `"$replace": true` inside an object when its local entries must replace
  the complete shared object instead of merging with it. The Step 2 predictor
  map uses this to prevent stale shared predictors from carrying into a study.
- Omitted local fields retain their reviewed per-step defaults.
- Relative study paths in the local file resolve from the `config` folder.
- Empty optional paths remain disabled.
- Generated effective configs are written under `config/.runtime/`, which is
  ignored by Git.

## Runtime Settings

The `runtime` object contains settings that may differ by workstation:

- `python_executable`: optional Python command or executable for Steps 0-3.
- `osgeo4w_root`: OSGeo4W/QGIS installation folder used by DP1 and Steps 4-5.
- `mpiexec`: optional MPI command or executable used by Step 4.
- `taudem_dll_dir`: folder containing the TauDEM-compatible `gdal.dll`.
- `taudem_proj_dir`: optional folder containing the matching `proj.db`.
- `step4_executable`: optional replacement Step 4 C++/MPI executable. Leave it
  empty to use the bundled executable.
- `workers.data_preparation`: DP1 polygon worker count.
- `workers.step1_prepare_inventory_labels`: Step 1 polygon worker count.
- `workers.step3_train_ml_dfi_model`: XGBoost/random-forest thread count.
- `workers.step4_deposition_zone`: Step 4 MPI process count.

Use `null` for a worker setting to preserve that step's shared default.

## Step Overrides

`step_overrides` uses the exact step names printed by:

```powershell
python .\workflow_config.py list
```

Override only values that differ for the current study. Scientific parameters
not listed in the local example continue to come from each step's
`config.default.json`.

The model-training and model-application configurations are separate sections:

- `step3_train_ml_dfi_model`
- `step3_model_application`

The central runner automatically points model application to the generated
effective Step 2 predictor config. This prevents extraction and full-domain
application from using different predictor lists.

## Commands

Validate the centralized configuration without running a step:

```powershell
python .\workflow_config.py validate
```

Validate the installed environment and the paths produced by the merged
configuration:

```powershell
python .\scripts\check_environment.py
python .\scripts\check_environment.py --step step2 --min-free-gb 20
```

The environment checker performs temporary write probes in output filesystems
and removes them immediately. It does not create workflow outputs or process
input pixels.

Inspect a fully merged config:

```powershell
python .\workflow_config.py show step4
```

Run a step:

```powershell
python .\workflow_config.py run step0
python .\workflow_config.py run step4 --dry-run
python .\workflow_config.py run step5 --check-environment
```

Per-step config arguments cannot be passed through the central runner. Put
study-specific changes under `step_overrides` instead.

The optional PowerShell helper selects the workspace environment automatically:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_workflow_step.ps1 validate
.\run_workflow_step.ps1 run step0
```

## Compatibility

The existing per-step launchers and `config.default.json` files remain valid.
They are the tested base contracts and can still be invoked explicitly for
experiments. The centralized runner is the recommended handoff interface.
Running a per-step launcher without arguments also uses the centralized config.
