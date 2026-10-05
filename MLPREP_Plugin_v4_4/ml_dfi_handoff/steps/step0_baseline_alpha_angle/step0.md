# Step 0: Source-Slope Baseline Alpha Raster

Step 0 is the Codes_V2 copy of the DP5 source-slope baseline alpha-angle
raster generator. It creates the initial alpha-angle raster used by the
dynamic-alpha deposition-zone workflow.

This is a standalone raster conversion step: a slope raster in degrees is
classified into an empirical source-slope-based baseline alpha prior. The
script does not import modules from the original Objective 4 `Codes`
workspace.

## Files

- `step0_baseline_alpha_angle.py` - standalone CLI script.
- `config.default.json` - portable default run config.

## Inputs And Outputs

Required input:

- A single-band slope raster whose values are degrees.

Required output:

- An alpha-angle GeoTIFF in degrees (`float32`).

Optional outputs:

- A slope-class GeoTIFF (`int16`).
- A JSON metadata report. If no report path is provided, it is written beside
  the alpha raster as `<output_alpha_stem>_metadata.json`.

The alpha and optional class rasters inherit the source raster CRS, transform,
resolution, dimensions, and extent. GeoTIFF outputs are LZW-compressed.

## Default Config

The default config is a run record that uses workspace-root paths:

- slope:
  `Sample Data/Input/Raster/Predictors/Topographic/Filtered/Makilala_Slope_LPF_PRSTM51N.tif`
- alpha output: `Sample Data/Output/Latest Runs/Step0/NC_SourceSlope_BaselineAlpha_PRSTM51N.tif`
- class output: `Sample Data/Output/Latest Runs/Step0/NC_SourceSlope_BaselineAlpha_Class_PRSTM51N.tif`
- metadata output: `Sample Data/Output/Latest Runs/Step0/NC_SourceSlope_BaselineAlpha_Metadata.json`

Large rasters are processed in `1024 x 1024` windows, so memory use remains
bounded. Outputs are first written to temporary files in the destination
folder and replace the official files only after raster processing succeeds.
BigTIFF is enabled automatically when required.

## Calibration Basis

The lookup table is based on the clean landslide inventory. For each
landslide, the analysis compared:

- its observed alpha angle
- the mean slope in its upper source area

Slope had the strongest relationship with alpha among the tested terrain
factors (`564` valid landslides):

| Association | Value |
| --- | ---: |
| Pearson `r` | `0.777` |
| Spearman `rho` | `0.762` |

The slope ranges below are the Step 0 classes used to build the raster. Their
spacing is based on the relative behavior of the data points in the
slope-alpha correlation graph. At the lower end of the correlation line, the
data points are fewer and more scattered, so the slope intervals are closer
together. Toward the upper end of the correlation line, the data points are
more abundant and more defined, so wider slope intervals can be used while
still representing the observed trend.

Each positive class has one assigned whole-degree alpha value, but two
assignment rules are used:

- For classes `1` to `5`, the assigned alpha angle is set `1` degree lower
  than the lower end of the slope interval. This makes the starting alpha
  slightly permissive and allows runout to occur in the lower, more scattered
  part of the slope-alpha graph.
- For classes `6` to `11`, the assigned alpha angle is based on the median
  alpha angle of that slope interval from the generated slope-alpha
  correlation line, rounded to the nearest whole degree. This is used where
  the data points are more abundant and the trend is more defined.

Slopes below `10` degrees are written as alpha NoData. This is a workflow
rule: these cells are treated as non-source or very low initiation-potential
terrain, so Step 0 does not assign them an alpha angle at all.

Note: the table was developed using mean slope for each source area, but
Step 0 assigns alpha to individual raster cells. The resulting raster is a
starting alpha estimate for testing in Step 4.

## Lookup Table

The interval design follows the graph-based behavior described above. Classes
`1` to `5` use the one-degree-lower rule, while classes `6` to `11` use the
rounded median alpha angle from the generated correlation line.

| Class | Slope Interval (degrees) | Assigned Alpha |
| ---: | --- | ---: |
| 0 | `<10` | NoData |
| 1 | `10-12` | 9 |
| 2 | `12-14` | 11 |
| 3 | `14-16` | 13 |
| 4 | `16-18` | 15 |
| 5 | `18-20` | 17 |
| 6 | `20-25` | 19 |
| 7 | `25-30` | 22 |
| 8 | `30-35` | 25 |
| 9 | `35-40` | 28 |
| 10 | `40-45` | 29 |
| 11 | `>45` | 30 |

Implementation note: exact break values are assigned to the higher class. For
example, `12` degrees is class `2`, and `45` degrees is class `11`.

## Validation Of Upper-Interval Assigned Alpha

For classes `6` to `11`, the assigned alpha values are validated against the
median alpha angle for each upper slope interval. The assigned alpha is the
nearest whole-degree value of the median alpha.

| Class | Slope Interval (degrees) | Median Alpha From Correlation Line | Assigned Alpha | Validation |
| ---: | --- | ---: | ---: | --- |
| 6 | `20-25` | 18.793 | 19 | matches rounded median |
| 7 | `25-30` | 21.898 | 22 | matches rounded median |
| 8 | `30-35` | 25.252 | 25 | matches rounded median |
| 9 | `35-40` | 27.874 | 28 | matches rounded median |
| 10 | `40-45` | 28.835 | 29 | matches rounded median |
| 11 | `>45` | 29.895-30.218 | 30 | matches rounded median |

## NoData Behavior

Input masked cells, NaN values, non-finite values, and finite slopes below
`10` degrees are written as output alpha NoData.

Slopes below `10` degrees are still assigned class `0` in the optional class
raster for audit, but the alpha raster contains no alpha-angle value for them.
The alpha output uses NoData `-9999` by default, configurable with `--nodata`.
The optional class raster uses NoData `-9999` only for input NoData or
non-finite cells.

## Command Line

Run through the portable launcher. With no arguments it loads the config beside
the script, and config paths resolve relative to that config file:

```powershell
.\steps\step0_baseline_alpha_angle\run_step0.bat
```

Explicit command-line paths remain supported and are resolved relative to the
current terminal directory.

Pass `--overwrite` to replace existing outputs. Valid finite slopes outside
the physical range of 0 to 90 degrees are still classified by the fixed table
and are reported as warnings in the metadata JSON and console output.

## Environment Note

On Windows, older GIS applications can leave `PROJ_LIB`, `PROJ_DATA`, or
`GDAL_DATA` environment variables pointing at libraries that conflict with
rasterio. If the command reports a PROJ database version mismatch, run Step 0
from a clean terminal or remove those inherited overrides for that terminal
session before running the script.
