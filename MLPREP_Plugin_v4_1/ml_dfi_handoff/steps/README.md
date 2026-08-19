# Workflow Steps

This folder contains workflow components that are organized below a shared
`steps/` namespace.

The active sequence is:

```text
data_preparation/
step0_baseline_alpha_angle/
step1_prepare_inventory_labels/
step2_extract_pixel_samples/
step3_train_ml_dfi_model/
step4_deposition_zone/
step5_post_depositional_spread_PDS/
step6_landslide_damming_potential_LDP/
```

Folder placement does not change the public workflow keys. Run stages through
the root `workflow_config.py` command, for example
`python workflow_config.py run data_preparation` or
`python workflow_config.py run step4`.
