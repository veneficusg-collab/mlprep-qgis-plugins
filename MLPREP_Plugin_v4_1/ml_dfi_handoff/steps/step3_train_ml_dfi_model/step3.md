# Step 3: Train ML-DFI Model

## What This Step Does

`Step 3` trains a probabilistic classifier for ML-derived Depositional
Favorability Index.

The script reads the Step 2 extracted sample table, identifies predictor columns,
applies numeric/categorical preprocessing, trains a model, evaluates validation
performance, chooses a diagnostic classification threshold, and saves the final
production model. It also writes explainability outputs for the final model,
including XGBoost SHAP contribution summaries and partial dependence plots.

The saved model output is a probability of `target = 1`.

## Scientific Meaning Of The Prediction

The predicted probability is interpreted as ML-DFI:

- low probability = terrain is less favorable for landslide deposition
- high probability = terrain is more favorable for landslide deposition

This model does not predict landslide initiation susceptibility and does not
predict the full landslide runout footprint. It learns terrain conditions
associated with depositional-favorable cells.

## Main Script

Script path:

- [step3_train_ml_dfi_model.py](step3_train_ml_dfi_model.py)
- [apply_ml_dfi_model_to_rasters.py](apply_ml_dfi_model_to_rasters.py)

## Inputs

Required inputs:

- `input_sample_table`
  - default: `Sample Data/Output/Latest Runs/Step2/ml_dfi_pixel_samples.csv`
- `output_dir`
  - default: `Sample Data/Output/Latest Runs/Step3`

Step 3 expects the single Step 2 extracted sample table. Validation is fixed
and non-configurable: the script first creates a grouped 70/30 holdout split,
then runs grouped 5-fold CV inside the 70% training split.

```text
Full Step 2 sample table
|-- Training split, approximately 70%
|   |-- Optuna tuning and model-selection diagnostics use 5-fold
|       stratified group CV by landslide_id
|-- Held-out validation split, approximately 30%
    |-- used only after the final model is trained on the 70% training split
```

## Supported Models

- `xgboost`
  - preferred model
  - uses `xgboost.XGBClassifier`
  - validation runs use early stopping when validation data are available
  - the final production model is retrained on the grouped 70% training split
    using the validated best boosting length after grouped CV evaluation
- `random_forest`
  - baseline backup model
  - uses `sklearn.ensemble.RandomForestClassifier`

## Hyperparameter Tuning

XGBoost tuning is enabled by default with Optuna:

- `enable_hyperparameter_tuning = true`
- `tuning_trials = 100`
- `tuning_metric = "PR_AUC"`

Tuning validation uses stratified group k-fold by `landslide_id` inside the
training split. This is preferred for ML-DFI because nearby terrain pixels are
spatially autocorrelated and depositional samples can be class-imbalanced. If
`landslide_id` is missing or cannot produce valid folds, Step 3 fails clearly
instead of falling back to random pixel validation.

The 70% training / 30% holdout split, 5-fold CV count, and `landslide_id`
grouping are part of the Step 3 method and are not config options.

Optuna tunes these XGBoost parameters:

- `max_depth`: `2` to `8`
- `learning_rate`: `0.005` to `0.08`, log scale
- `subsample`: `0.5` to `1.0`
- `colsample_bytree`: `0.5` to `1.0`
- `min_child_weight`: `1.0` to `15.0`
- `gamma`: `0.0` to `5.0`
- `reg_alpha`: `0.0` to `10.0`
- `reg_lambda`: `1.0` to `20.0`
- `scale_pos_weight`: `0.25x` to `4x` of the training table's negative/positive class ratio, log scale

XGBoost tuning uses `n_estimators = 2000` as the upper limit and
`early_stopping_rounds = 100` to choose the effective boosting length.

## Output Files

The default model-training and validation output folder is:

- `Sample Data/Output/Latest Runs/Step3/model_training_validation`

Raster application should use a separate bundle such as:

- `Sample Data/Output/Latest Runs/Step3/domain_application_dfi`

Core outputs:

- `trained_ml_dfi_model.pkl`
- `model_metadata.json`
- `validation_predictions.csv`
- `training_predictions.csv`
- `model_metrics.csv`
- `threshold_analysis.csv`
- `partial_dependence.csv`
- `hyperparameter_tuning_trials.csv`
- `processing_log.txt`

For the fixed holdout-plus-CV workflow, `validation_predictions.csv` contains both
out-of-fold CV predictions from the 70% training split and final-model
predictions for the held-out 30% validation split. `training_predictions.csv`
contains one prediction per final training sample from the final production
model, not repeated fold-training predictions.

`model_metrics.csv` reports validation metrics twice for threshold-dependent
diagnostics: once at fixed threshold `0.5`, and once at the optimized diagnostic
threshold. PR-AUC, ROC-AUC, Brier score, probability histograms, and the
continuous probability raster remain the primary ML-DFI outputs.

Raster-application outputs:

- `ml_dfi_probability_full_valid_predictor_domain.tif`
  - continuous probability raster for `target = 1`
  - interpreted as ML-derived Depositional Favorability Index
  - `NoData` where the full predictor stack is incomplete
- `ml_dfi_probability_full_valid_predictor_domain_summary.json`
- `full_domain_shap_feature_importance.csv`
- `full_domain_shap_feature_importance.png`
- `full_domain_shap_summary_beeswarm.png`
- `full_domain_shap_dependence_plots.png`
- `full_domain_shap_sample.csv`
  - sampled full-domain prediction explainability outputs
  - deterministic stratified sample of up to 60,000 valid predicted cells:
    20,000 low DFI cells (`P < 0.30`), 20,000 medium DFI cells
    (`0.30 <= P < 0.70`), and 20,000 high DFI cells (`P >= 0.70`)
  - explains which predictors influence the mapped DFI probability surface
  - does not represent validation accuracy because the full raster domain does
    not have observed labels everywhere
- `apply_ml_dfi_model_to_rasters_log.txt`

Default validation plots:

- `pr_curve.png`
  - overlaid precision-recall curves for the full training dataset and held-out
    validation dataset
- `roc_curve.png`
  - overlaid ROC curves for the full training dataset and held-out validation
    dataset
- `calibration_curve.png`
  - overlaid reliability curves for the full training and held-out validation
    datasets, labeled with expected calibration error
- `holdout_validation_confusion_matrix.png`
  - primary thresholded classification result for unseen held-out groups
- `holdout_validation_feature_importance.png`
  - permutation feature importance measured as mean PR-AUC decrease after
    predictor shuffling in the held-out validation dataset
- `final_training_probability_histogram.png`
- `holdout_validation_probability_histogram.png`
  - diagnostic comparison of probability distributions in fitted training rows
    and unseen held-out rows
- `holdout_validation_shap_feature_importance.csv`
- `holdout_validation_shap_feature_importance.png`
- `holdout_validation_shap_summary_beeswarm.png`
- `holdout_validation_shap_dependence_plots.png`
  - predictor-value versus SHAP-contribution panels for the held-out validation
    sample
- `partial_dependence_plots.png`
  - one combined partial-dependence panel with full-training and held-out
    validation lines overlaid in contrasting colors
- `pr_auc_comparison_folds_training_holdout_line.png`
  - one line diagram showing PR-AUC for all CV folds, mean CV, full training,
    and held-out validation

The metric plots use depositional and non-depositional target labels on axes and
legends. The ROC plot reports ROC-AUC in the legend, and the PR plot reports
PR-AUC plus the positive-class prevalence baseline.

`model_metrics.csv` also reports expected calibration error. PR-AUC and ROC-AUC
measure discrimination, while Brier score, log loss, the calibration curve, and
expected calibration error diagnose whether DFI probability magnitudes are
well calibrated.

## Explainability Outputs

Explainability output is enabled by default:

- `write_explainability = true`
- `shap_sample_size = 5000`
- `partial_dependence_sample_size = 2500`
- `partial_dependence_grid_resolution = 25`

For the preferred XGBoost model, SHAP analysis uses final-model XGBoost
contribution values after the fitted preprocessing pipeline. The CSV includes:

- predictor-level global importance for interpretable ranking
- transformed model-feature rows for one-hot-expanded or numeric features
- mean absolute SHAP contribution, mean contribution, spread, rank, and sampled
  row count

The SHAP contribution axis is the XGBoost raw model margin, not a probability.
SHAP dependence plots show how each top predictor's observed value relates to
its SHAP contribution. Values above zero push the model toward higher
depositional probability; values below zero push it toward lower depositional
probability. Step 3 writes this plot for the held-out validation sample, while
the raster-application script writes the same style of plot for sampled valid
pixels across the full prediction domain. Full-domain SHAP sampling is
stratified by predicted DFI probability so low, medium, and high DFI areas are
represented evenly in the explainability outputs.
For non-XGBoost model types, the script logs that SHAP contribution export is
skipped and still writes partial dependence outputs.

Partial dependence is computed separately from sampled copies of the final
training table and held-out validation table, then shown in one overlaid plot.
Each predictor is forced across numeric quantiles or observed categorical
values, and the plotted y-axis is the mean predicted probability of depositional
terrain.

## Validation Caution

The fixed validation workflow uses `landslide_id` from Step 1 and Step 2
for both the initial holdout split and the 5-fold CV inside the training split.
This avoids landslide-level leakage while preserving depositional/non-
depositional class balance as much as the landslide group structure allows.

The held-out validation groups are not seen during Optuna tuning or final model
fitting. Inside the remaining training split, Step 3 keeps whole
`landslide_id` groups out of both training and validation within each CV fold.
Each split is checked for group leakage and for both target classes on the
training and validation sides.

Random pixel validation is intentionally not used because it can overestimate
model performance when nearby pixels are spatially autocorrelated.

## Class Imbalance Handling

Step 3 does not use a `sample_weight` column. Class imbalance is handled through
XGBoost `scale_pos_weight`.

When hyperparameter tuning is enabled, Optuna tunes `scale_pos_weight` around
the training table's negative/positive class ratio. If tuning is disabled,
`class_imbalance_strategy = "auto_scale_pos_weight"` computes
`scale_pos_weight` directly from the training labels.

## Missing Predictors

Rows with `inf` or `-inf` predictor values are still removed because those
values are unsafe for model training. Plain missing predictor values are kept:
XGBoost handles `NaN` values natively, while Random Forest uses median
imputation inside its model pipeline.

## Categorical Predictors

Step 3 uses a preprocessing pipeline that separates numeric and categorical
predictors. Numeric predictors are passed through for XGBoost or median-imputed
for scikit-learn models. Categorical predictors are imputed and one-hot encoded
before model training. By default, `geomorphology` is treated as categorical
even when it is stored as numeric class codes, so landform classes are not
treated as ordinal measurements. Categorical detection is explicit by default:
`auto_detect_categorical = false`.

The saved metadata records categorical columns and fitted class levels. During
raster application, numeric raster class codes are converted to the same string
representation used during training. Application fails clearly if a raster
contains a categorical class that the model never saw during training.

## Runtime And Output Safety

Optuna still evaluates the fixed five grouped folds, but each fold's
preprocessor is fitted once and its transformed training and validation
matrices are reused across trials. A median pruner can stop weak trials after
intermediate fold scores. XGBoost remains internally multithreaded; trials are
kept serial to avoid CPU oversubscription.

Training and raster application write into temporary sibling directories. The
official output bundle is replaced only after all configured outputs and QC
checks succeed. If publication fails, the preceding official files are
restored. Unknown configuration keys, missing or invalid `landslide_id` values,
unsafe output filenames, unaligned rasters, and unseen categorical raster
classes fail before publication.

Model files use Python joblib serialization. Load only model artifacts produced
by a trusted Step 3 run because loading an untrusted joblib file can execute
serialized Python code.

## Example Usage

Default XGBoost training:

```powershell
.\steps\step3_train_ml_dfi_model\run_step3.bat
```

The launcher uses the workspace `.venv`, and paths inside a JSON config resolve
relative to the config file.

Using the current Step 2 outputs:

```powershell
python .\steps\step3_train_ml_dfi_model\step3_train_ml_dfi_model.py --input-sample-table ".\Sample Data\Output\Latest Runs\Step2\ml_dfi_pixel_samples.csv" --output-dir ".\Sample Data\Output\Latest Runs\Step3" --model-type xgboost
```

Random Forest baseline:

```powershell
python .\steps\step3_train_ml_dfi_model\step3_train_ml_dfi_model.py --input-sample-table ".\Sample Data\Output\Latest Runs\Step2\ml_dfi_pixel_samples.csv" --output-dir ".\Sample Data\Output\Latest Runs\Step3_rf" --model-type random_forest --disable-hyperparameter-tuning
```

Skip explainability exports for a fast training-only rerun:

```powershell
python .\steps\step3_train_ml_dfi_model\step3_train_ml_dfi_model.py --input-sample-table ".\Sample Data\Output\Latest Runs\Step2\ml_dfi_pixel_samples.csv" --output-dir ".\Sample Data\Output\Latest Runs\Step3" --disable-explainability
```

Apply the trained model to the full valid Makilala predictor domain:

```powershell
Copy-Item .\steps\step3_train_ml_dfi_model\config.application.example.json .\my_application.json
.\steps\step3_train_ml_dfi_model\run_apply_model.bat --config-json .\my_application.json
```

The application script reads the predictor column order from
`model_metadata.json`, reads the Step 2 predictor raster configuration, streams
the aligned predictor rasters in windows, runs `predict_proba`, and writes a
GeoTIFF probability surface aligned to the Step 1 label raster. It does not
require a full-domain predictor CSV from Step 2.

Raster application has no built-in desktop paths. It requires an explicit
application config so the intended model, predictor rasters, and output folder
are visible before a domain run starts.

## Folder Contents

- [step3_train_ml_dfi_model.py](step3_train_ml_dfi_model.py)
- [apply_ml_dfi_model_to_rasters.py](apply_ml_dfi_model_to_rasters.py)
- [step3.md](step3.md)
