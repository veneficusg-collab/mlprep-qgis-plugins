# Step 2: Extract Pixel Samples

## What This Step Does

`Step 2` converts the Step 1 ML-DFI label raster and a set of manually aligned
predictor rasters into a machine-learning-ready pixel table.

Each exported row represents one valid raster cell. The script:

1. Reads the Step 1 `ml_dfi_label_raster.tif`.
2. Keeps only valid labeled cells:
   - `1` = observed depositional terrain
   - `0` = source/upper-transport non-depositional terrain inside mapped landslides
   - `NoData` = excluded from training
3. Verifies that every predictor raster matches the label raster exactly.
4. Screens continuous predictors for strong multicollinearity using Pearson
   correlation, VIF, and tolerance.
5. Automatically drops one predictor from a highly redundant pair or the
   highest-VIF predictor when the high multicollinearity threshold is breached.
6. Extracts predictor values at valid label pixels. Predictor values are read
   once in raster blocks and reused by screening and table construction.
7. Extracts the required `landslide_id` group raster.
8. Removes invalid rows based on predictor NoData and non-finite values.
9. Exports all valid labeled pixels after filtering.

This step creates the supervised learning table for the pixel-based ML-DFI classifier. It is strictly an extraction step; class imbalance handling and train/validation splitting are handled by Step 3.

## Scientific Meaning Of The Target

The `target` column in this step is not landslide initiation susceptibility.

It represents depositional terrain favorability:

- `target = 1` means the pixel falls inside an observed depositional-zone cell from Step 1
- `target = 0` means the pixel falls inside the upper source/transport zone from Step 1

This means the model learned from Step 2 is designed to discriminate depositional terrain from source/transport-dominant terrain inside mapped landslides.

## Scientific Caution

This table is pixel-based, so nearby samples are spatially autocorrelated. Step 2 therefore requires the `landslide_id` group column so Step 3 can use grouped validation and keep pixels from the same mapped landslide out of both training and validation at the same time.

## Main Script

Script path:

- [step2_extract_pixel_samples.py](step2_extract_pixel_samples.py)

## Inputs

The script accepts values from the `CONFIG` dictionary near the top of the script or from command-line arguments.

Required inputs:

- `label_raster_path`
  - usually the Step 1 output `ml_dfi_label_raster.tif`
- `predictor_rasters`
  - dictionary of predictor name to raster path
- `output_dir`

Required grouping input:

- `group_rasters`
  - `landslide_id` is required
  - used for landslide-grouped validation later

Other inputs:

- `output_table_name`
  - default `ml_dfi_pixel_samples`
- `output_format`
  - `csv`, `parquet`, or `both`
- `label_nodata_value`
  - default `-9999`
- `include_coordinates`
- `include_row_col`
- `compress_outputs`
- `categorical_predictors`
  - explicit predictor names to retain but exclude from Pearson/VIF screening

## Predictor Rasters

The script accepts any number of predictor rasters. Typical examples include:

- `slope`
- `plan_curvature`
- `profile_curvature`
- `tpi`
- `flow_accumulation`
- `distance_to_drainage`
- `local_relief`
- `proximity_to_stream`
- `ndvi`

Predictor paths that are blank are ignored with a warning. The script fails only if no valid predictor rasters remain after filtering.

## Multicollinearity Screening

Before full extraction, Step 2 samples predictor values from valid labeled cells
and screens continuous predictors for redundancy. The default thresholds are:

- absolute Pearson correlation `>= 0.70`
- VIF `>= 5`
- tolerance `<= 0.20`

If a Pearson threshold is breached, Step 2 drops one of the two highly
correlated predictors based on stronger redundancy evidence. If no Pearson pair
breaches but VIF/tolerance still breaches, Step 2 drops the highest-VIF
predictor. The screening repeats until the retained continuous predictors are
below the thresholds.

The reported Pearson matrix, VIF, and tolerance values are recomputed after all
automatic predictor drops, so the outputs describe the retained predictor set
that is actually used for extraction.

For the known high-correlation pair `dstgsurf` / `proximity_to_stream`, Step 2
prioritizes dropping `proximity_to_stream` and retaining `dstgsurf`.

Configured categorical rasters, such as `geomorphology`, are retained but
skipped from Pearson/VIF removal because treating class codes as continuous
distances can create misleading correlations. Step 2 does not infer categorical
status from the number of unique values; categorical predictors must be listed
explicitly in `categorical_predictors`.

## Required Group Rasters

The `landslide_id` group raster is mandatory. Step 2 fails if it is not configured, because Step 3 training and validation should be done by landslide group to reduce leakage from spatial autocorrelation.

By default, `steps/step2_extract_pixel_samples/config.default.json` points
`landslide_id` to the latest Step 1
output `Sample Data/Output/Latest Runs/Step1/landslide_id_raster.tif`. This carries the
mapped landslide group ID into Step 2 samples so Step 3 can use stratified
grouped validation without manual raster preparation.

This column is not the training target. It is the required grouping control for Step 3 `stratified_group_kfold` validation by `landslide_id`.

Every labeled cell must have a valid, finite, positive integer `landslide_id`.
Step 2 fails before publishing outputs if the group raster contains NoData,
non-finite, non-positive, or fractional IDs at labeled cells, or fewer than two
distinct landslide groups.

## Alignment Rules

This step is intentionally strict. The user must align predictor and group
rasters before running the script.

Every predictor raster must match the label raster exactly in:

- CRS
- transform
- width
- height
- extent

The script does not resample or reproject predictors.

If a predictor or group raster does not match, the script fails immediately.
It does not resample, reproject, or modify any input raster.

## Output Files

Core outputs:

The active sample output folder is:

- `Sample Data/Output/Latest Runs/Step2`

Predictor rasters are expected to be manually aligned before extraction. Step 2
does not write an aligned-predictor folder in the current workflow.

- `ml_dfi_pixel_samples.csv`
  - written when `output_format = csv`
- `ml_dfi_pixel_samples.parquet`
  - written when `output_format = parquet`
- both files
  - written when `output_format = both`, if Parquet support is available

Required columns in the main output table can include:

- `sample_id`
- `row`
- `col`
- `x`
- `y`
- `target`
- required group columns
- one predictor column per predictor raster

Additional report outputs:

- `ml_dfi_pixel_samples_summary.csv`
  - class counts, dropped rows, predictor list, output path
- `predictor_alignment_report.csv`
  - one row per predictor raster with alignment diagnostics
- `multicollinearity_summary.csv`
  - retained/dropped predictor status, post-drop Pearson/VIF/TOL values for
    retained predictors, and drop reason for removed predictors
- `multicollinearity_pearson_matrix.csv`
  - Pearson correlation matrix recomputed after automatic predictor dropping
- `multicollinearity_pearson_heatmap.png`
  - labeled Pearson correlation heatmap for retained predictors
- `multicollinearity_vif_tolerance.png`
  - post-drop VIF and tolerance bar diagrams with threshold lines
- `processing_log.txt`
  - full runtime log
- `run_config.json`
  - resolved run configuration, reference-grid metadata, retained predictors,
    and published output inventory

Outputs are first written and checked in a temporary staging directory. Step 2
publishes the complete bundle only after QC succeeds. A failed run leaves the
previous official output bundle unchanged and writes
`processing_log.failed.txt` when possible.

## Target Column

The final class column is named:

- `target`

Values are:

- `1` = depositional terrain
- `0` = source/upper-transport non-depositional terrain

## Whole-Domain DFI Raster

Step 2 no longer writes an unlabeled full-domain predictor table. That table can
be very large, and it duplicates work that is better handled during raster
application.

For whole-domain DFI generation, use Step 3's
`apply_ml_dfi_model_to_rasters.py`. That script reads the trained model and the
predictor rasters directly, streams raster blocks, runs `predict_proba`, and
writes the continuous ML-DFI probability GeoTIFF without creating a large
intermediate CSV.

## Extraction Scope

Step 2 exports every valid labeled row that remains after predictor filtering.
It does not balance classes, cap positives or negatives, shuffle samples, or
create train/validation splits. This keeps Step 2 focused on extraction and
preserves the full candidate sample pool for Step 3
grouped validation.

## Row Filtering Rules

The script removes rows where:

- label is `NoData`
- a predictor is `NoData`
- a predictor value is `NaN`
- a predictor value is infinite

Predictor NoData filtering is always enabled and is not configurable. This keeps
the exported Step 2 table strictly training-ready: every predictor value in
every exported row must be finite.

## Coordinates And Indices

If enabled:

- `include_row_col = true`
  - adds `row` and `col`
- `include_coordinates = true`
  - adds `x` and `y` from pixel centers based on the label raster transform

## Quality Control Performed

After extraction, the script checks that:

- the output table exists
- the output table includes `target`
- at least one predictor column exists
- both classes `0` and `1` are present
- no target values other than `0` and `1` exist
- predictor columns do not contain `NaN` or infinite values when predictor-drop filtering is enabled
- at least one group column is present
- group columns contain valid finite values for all exported rows
- group columns contain at least two distinct groups
- positive sample count is greater than zero
- negative sample count is greater than zero
- all predictor and group rasters passed alignment checks

## Example Usage

Using the `CONFIG` block:

1. Open the script.
2. Fill in the label raster path, predictor raster paths, required group rasters, and output directory.
3. Run the script with Python.

Command-line example:

```powershell
python .\steps\step2_extract_pixel_samples\step2_extract_pixel_samples.py --label-raster-path "<ml_dfi_label_raster.tif>" --output-dir "<step2-output-folder>"
```

Using the default JSON config:

```powershell
.\steps\step2_extract_pixel_samples\run_step2.bat
```

The launcher prefers the workspace runtime environment. Every relative path in a JSON config
is resolved from that config's folder, so execution does not depend on the
terminal's current directory. Predictor alignment is performed manually by the
user; Step 2 does not provide resampling or reprojection.

Pipeline extraction concept:

- Step 2 writes all valid labeled pixels after filtering
- Step 3 handles class imbalance during model training

## Output Column Order

The output dataframe is written in this order:

1. `sample_id`
2. `row`
3. `col`
4. `x`
5. `y`
6. `target`
7. required group columns
8. predictor columns

## Dependencies

The script uses:

- `rasterio`
- `numpy`
- `pandas`
- `matplotlib`
- `pyproj`
- `pathlib`
- `logging`
- `argparse`

If Parquet output is requested, Parquet support depends on the installed pandas backend. If Parquet writing fails, the script falls back to CSV with a warning.

## Folder Contents

- [step2_extract_pixel_samples.py](step2_extract_pixel_samples.py)
- [step2.md](step2.md)
