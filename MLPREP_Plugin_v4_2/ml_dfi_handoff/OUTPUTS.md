# Output Organization

Workflow outputs are local artifacts and are not included in source handoff
packages.

## Official Outputs

The most recently reviewed full run for each step belongs under:

```text
Sample Data/Output/Latest Runs/<StepName>/
```

Only one coherent output bundle per step should remain there. A run is official
only after its QC/summary files have been reviewed. Model training and domain
application remain separate subfolders under Step 3.

## Exploratory Outputs

Sensitivity tests, alternate-input comparisons, profilers, and one-off analyses
belong under:

```text
Sample Data/Output/Exploratory Runs/<purpose>/
```

Exploratory outputs must not overwrite official outputs. Promote a result by
rerunning it through the reviewed default config and publishing the complete
bundle to `Latest Runs`.

## Packaging

`Sample Data`, run outputs, trained models, local configs, caches, logs, and
archives are excluded from handoff packages. Use
`scripts/create_handoff_package.py`; it packages only Git-tracked source files
that pass the repository denylist and writes a checksum manifest into the ZIP.
