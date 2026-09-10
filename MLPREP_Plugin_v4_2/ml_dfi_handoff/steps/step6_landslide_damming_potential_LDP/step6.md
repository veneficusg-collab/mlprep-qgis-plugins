# Step 6: Landslide Damming Potential LDP

Step 6 is a standalone post-processing step after `step4_deposition_zone` and
`step5_post_depositional_spread_PDS`. It
screens stream cells where the cleaned and post-depositionally spread Step 5
runout footprint intersects
channels and may form a landslide dam.

This is a screening layer, not a definitive landslide-dam hazard map. It is
based mainly on runout/stream intersection geometry, mean upstream runout
approach direction, inflow angle, mean channel-adjacent slope, and a separate
DEM-derived obstructable-valley width reference-volume assessment.

## Required Inputs

The default config is wired to the current `Codes_V2` handoff layout:

- DEM/reference grid: `Sample Data/Input/Raster/Shared Inputs/Makilala_DEM_Fill_PRSTM51N.tif`
- D-Infinity flow direction: `Sample Data/Input/Raster/Shared Inputs/Makilala_DFD_PRSTM51N.tif`
- runout mask: `Sample Data/Output/Latest Runs/Step5/post_depositional_combined_mask.tif`
- required D-Infinity stream mask: `Sample Data/Input/Raster/Step6 Inputs/Makilala_DinfStreamMask_PRSTM51N.tif`
- required slope raster: `Sample Data/Input/Raster/Step6 Inputs/Makilala_Slope_LPF_PRSTM51N.tif`

The D-Infinity stream mask is the official Step 6 stream input because it
provided the better screening result during the Step 6 stream-mask comparison.
The D8 stream mask may still be used for experimental comparison runs, but it is
not the default input requirement.

The Step 5 combined runout mask is used instead of only the pure depositional
mask so Step 6
can flag possible blockage in source/transport or upstream portions of the
modeled runout path, where damming may be less visually obvious but still
important.

All required rasters must share CRS, transform, dimensions, resolution, and
extent.

Optional QA/input fields:

- `stream_vector`: stream polyline dataset. If provided, local stream direction
  is estimated from the nearest stream segment and requires `osgeo/OGR`.
- `output_candidate_points`: optional GeoPackage/Shapefile output. Leave blank
  for raster-only execution.

The default config leaves vector and candidate-point outputs blank so the step
can run with the standard `rasterio` stack.

The runout and stream rasters have a strict binary contract: every valid cell
must be `0` or `1`. D-Infinity directions must be radians within `0..2*pi`,
and slope values must be degrees within `0..90`. Step 6 fails before evaluation
when these value, alignment, or metric-CRS contracts are breached.

## Method

Step 6 evaluates candidate cells where:

```text
runout_mask_raster == 1
AND stream_raster == 1
```

Only direct runout-stream intersections are candidates. A nearby stream cell is
not evaluated unless it is itself part of the runout mask.

For each candidate stream cell, the script estimates:

- local stream-channel orientation;
- mean landslide/runout approach direction from the upper 25 m runout window;
- inflow angle between those two directions, normalized to `0..90` degrees; and
- mean channel slope from the required slope raster.

The angle-and-slope class is the geometry-only landslide-damming potential
classification. Valley width no longer modifies the geometry-only
`damming_potential_score` or class rasters. Instead, the corrected obstructable
valley width is evaluated separately against a fixed 2,600 m3 reference
delivered-deposit volume scenario.

## Mean Runout Approach Direction

Step 6 no longer uses only the candidate cell or nearest local runout direction.
For each candidate stream cell, it scans runout cells within
`runout_approach_window_m = 25` m upstream of the candidate. Runout cells are
treated as upstream when their D-Infinity direction points toward the candidate
cell within a 45 degree tolerance.

The output `step6_mean_runout_approach_angle_25m_deg.tif` stores the circular
mean of those upstream D-Infinity directions. If no upstream cells pass the
tolerance test, the script falls back to the best-aligned runout cell inside the
25 m window.

For one candidate cell, the runout-approach workflow is:

1. Start with a candidate cell where the Step 4 runout mask intersects the
   stream mask.
2. Search the surrounding runout cells within the configured upstream window
   distance, currently 25 m.
3. For each surrounding runout cell, read its D-Infinity flow-direction angle.
4. Check whether that surrounding cell's D-Infinity direction points toward the
   candidate cell within the fixed 45 degree tolerance.
5. Keep only the surrounding runout cells that pass that direction-to-candidate
   test.
6. Compute the circular mean of the kept D-Infinity direction angles. This
   circular mean becomes the candidate cell's main runout approach direction.
7. Estimate the local stream direction at the candidate cell.
8. Compute the inflow angle as the angle between the candidate cell's main
   runout approach direction and the local stream direction.
9. Use that inflow angle, together with mean channel slope, to assign the
   candidate damming-potential class.

The surrounding cells do not each receive a mean approach angle. Each
surrounding cell contributes its own D-Infinity direction. The candidate cell
receives the mean runout approach direction after those qualifying directions
are averaged.

## Mean Channel Slope

Step 6 no longer derives local channel slope from the DEM. The config requires a
`slope_raster` input.

For each candidate cell, the script samples slope values along the estimated
stream axis:

- up to `channel_slope_window_m = 25` m in one channel direction; and
- up to `25` m in the opposite channel direction.

The output `step6_mean_channel_slope_25m_deg.tif` stores the mean of valid slope
values from those upper/lower channel windows. This makes the slope test less
fragile than using only the candidate cell or a single neighboring DEM-derived
gradient.

## Obstructable Valley Width

Step 6 estimates `valley_floor_width_m` / `obstructable_valley_width_m`, not
active wetted-channel width. The width represents the horizontal span of low
valley-floor terrain that a landslide deposit would need to obstruct to impound
water.

The valley-floor module uses a threshold-elevation method:

1. Treat valid stream cells from `stream_raster` as downstream terminals.
2. Convert the D-Infinity flow direction raster to one deterministic primary
   downstream receiver per cell.
3. Trace each DEM cell along that primary path until it reaches a stream cell.
4. Compute `height_above_stream_m` as the DEM-cell elevation minus the
   downstream-associated stream-cell elevation.
5. Classify valley-floor cells where `height_above_stream_m <=
   valley_width_elevation_threshold_m`.
6. Select stream-profile anchor cells along the stream raster using
   `valley_width_local_profile_spacing_m`.
7. At those anchor cells, measure raw valley width along a profile
   perpendicular to local stream orientation.
8. Correct meander-inflated candidate widths by assigning the minimum valid raw
   anchor width within the `valley_width_meander_window_radius_m` moving
   window.

The default threshold is:

```text
valley_width_elevation_threshold_m = 10.0
```

This is a configurable screening parameter, not a universal physical
threshold. It should be checked against local hillshade, contours, imagery, and
known valley geometry.

The D-Infinity flow-path step is a deterministic primary-path approximation:
the script chooses the neighboring cell whose center-to-center direction best
matches the D-Infinity angle, preferring downslope or equal-elevation
neighbors. Cells that cannot be associated with a stream remain NoData in
`height_above_stream_m`.

The raw perpendicular-profile width is:

```text
raw_valley_width_m =
    left_valley_extent_m + right_valley_extent_m
```

Sampling uses physical coordinates and nearest-neighbor lookup on the
valley-floor mask. A profile is censored when one or both profile sides remain
inside valley-floor terrain out to the configured maximum profile distance.
Censored profiles are not treated as exact valley widths.

The meander correction is:

```text
corrected_valley_width(c) =
    minimum valid raw_valley_width(a)
    for stream-profile anchor cells a within
    valley_width_meander_window_radius_m
    of candidate stream cell c
```

The moving window is evaluated in map distance around the candidate stream
cell. The minimum is used because a stream bend can make a perpendicular profile
cut obliquely across the valley and overestimate the true obstructable span.
This anchor-based implementation follows the same threshold-elevation and
moving-window logic as Morgan et al. while avoiding repeated profile extraction
for every candidate runout/stream cell.

The implementation uses vectorized NumPy for primary D-Infinity receiver
construction, a Numba-compiled memoized height-above-stream trace, and
thread-parallel Numba candidate-cell evaluation. Valley-width neighborhood
queries use parallel, bounded-size chunks to avoid allocating the complete
candidate-neighbor list at once.

`numba_threads = 0` uses the active environment's Numba thread setting. Set a
positive value in a machine-local config to cap the candidate and spatial-query
worker count. `valley_width_assignment_chunk_size` controls only the number of
candidate points queried at once and does not change the scientific search
radius.

Very small negative height-above-stream results can occur where the
deterministic primary D-Infinity path must use an uphill fallback. Step 6
clamps those values to `0 m` and reports the affected-cell count in the summary.

## Equations

Candidate stream cells are evaluated where runout and stream cells overlap:

```text
candidate_c =
    runout_mask_c == 1
    AND stream_mask_c == 1
```

For each candidate stream cell `c`, the distance from a surrounding runout cell
`r` is:

```text
distance(r, c) =
    sqrt(((row_r - row_c) * dy)^2 + ((col_r - col_c) * dx)^2)
```

A surrounding runout cell is inside the approach window when:

```text
distance(r, c) <= runout_approach_window_m
```

The direction from surrounding runout cell `r` toward candidate stream cell `c`
is:

```text
target_angle(r -> c) =
    atan2((row_r - row_c) * dy, (col_c - col_r) * dx)
```

converted to degrees and normalized to `0..360`.

The D-Infinity flow angle of the surrounding runout cell is:

```text
flow_angle_r = degrees(dinf_flow_raster_r)
```

also normalized to `0..360`.

The angular difference used for the upstream test is:

```text
direction_difference_r =
    min(abs(flow_angle_r - target_angle(r -> c)),
        360 - abs(flow_angle_r - target_angle(r -> c)))
```

The runout cell contributes to the candidate approach direction when:

```text
direction_difference_r <= 45 degrees
```

The candidate runout approach angle is the circular mean of all contributing
runout-cell flow angles:

```text
runout_approach_angle_c =
    atan2(mean(sin(flow_angle_r)), mean(cos(flow_angle_r)))
```

If no runout cells pass the 45 degree upstream test, the script falls back to
the best-aligned runout cell in the 25 m window. Ties within 5 degrees are
averaged circularly.

The local stream angle is axial, meaning `0` and `180` degrees represent the
same channel orientation. The inflow angle is normalized to `0..90` degrees:

```text
delta =
    abs(runout_approach_angle_c - stream_angle_c) mod 180

inflow_angle_c =
    if delta > 90:
        180 - delta
    else:
        delta
```

Mean channel slope is sampled along the stream axis:

```text
channel_slope_c =
    mean(valid slope_raster samples within channel_slope_window_m
         in both directions along stream_angle_c)
```

With the current defaults, the class thresholds are:

```text
moderate_inflow_angle_deg = 30
high_inflow_angle_deg = 60
very_high_inflow_angle_deg = 75
low_channel_gradient_deg = 10
steep_channel_gradient_deg = 15
```

Candidate class assignment is:

```text
if slope is unavailable:
    inflow >= 75 -> class 4 very high damming potential
    inflow >= 60 -> class 3 high damming potential
    inflow >= 30 -> class 2 moderate damming potential
    otherwise    -> class 1 low damming potential

if slope is available:
    inflow >= 75 AND slope < 10 -> class 4 very high damming potential
    inflow >= 60 AND slope < 10 -> class 3 high damming potential
    inflow >= 60 AND slope >= 10 -> class 2 moderate damming potential
    30 <= inflow < 60 AND slope < 10 -> class 2 moderate damming potential
    inflow < 60 AND slope >= 10 -> class 5 debris-flow connectivity favored
    inflow < 30 AND slope < 10 -> class 1 low damming potential
```

The continuous damming-potential score is:

```text
angle_score =
    clamp((inflow_angle_c - moderate_inflow_angle_deg) / 60, 0, 1)
```

The channel-slope modifier is:

```text
if slope is unavailable:
    slope_modifier = 1.0
elif slope < low_channel_gradient_deg:
    slope_modifier = 1.0
elif slope >= steep_channel_gradient_deg:
    slope_modifier = 0.25
else:
    slope_modifier =
        1.0
        - 0.75
          * ((slope - low_channel_gradient_deg)
             / (steep_channel_gradient_deg - low_channel_gradient_deg))
```

Final score:

```text
damming_potential_score_c =
    clamp(angle_score * slope_modifier, 0, 1)
```

This score is geometry-only. It uses inflow angle and channel slope only.

## Reference-Volume Valley Assessment

Step 6 also evaluates whether the corrected obstructable valley width is
compatible with blockage by a fixed reference delivered-deposit volume. The
adopted reference scenario is:

```text
Aref = 2500.0 m2
Vsource = 0.54 * Aref^1.15
delivery_fraction = 0.60
Vdelivered_unrounded = Vsource * delivery_fraction
Vref = 2600.0 m3
```

With the default values:

```text
Vsource = 0.54 * 2500^1.15 = 4365.407 m3
Vdelivered_unrounded = 4365.407 * 0.60 = 2619.244 m3
Vref = 2600.0 m3
```

`Vref` is the rounded reference delivered-deposit volume. It is a screening
assumption, not a prediction of the actual source or delivered volume at every
candidate cell.

The fixed valley-width thresholds are:

```text
Wformation = sqrt(Vref / 180.3)
Wpossible = (Vref / 1.7)^(1 / 2.5)
```

For `Vref = 2600.0 m3`:

```text
Wformation = 3.80 m
Wpossible = 18.78 m
```

The corrected obstructable valley width is:

```text
Wv = corrected_valley_width_m
```

Reference-volume valley-domain classes are:

| Domain | Meaning |
| ---: | --- |
| 0 | not evaluated or indeterminate |
| 1 | non-formation domain; complete blockage unlikely |
| 2 | possible / uncertain obstruction domain |
| 3 | formation-domain condition met |

For uncensored valley-width measurements:

```text
if Wv <= Wformation:
    reference_domain_class = 3
elif Wformation < Wv <= Wpossible:
    reference_domain_class = 2
else:
    reference_domain_class = 1
```

For censored valley-width lower bounds:

```text
if measured_lower_bound_width_m > Wpossible:
    reference_domain_class = 1
else:
    reference_domain_class = 0
```

The combined reference-volume class keeps geometry as the gate:

```text
if geometry_baseline_class == 5:
    combined_class = 5
    combined_score = 0.0
elif reference_domain_class == 0:
    combined_class = 0
    combined_score = NoData
elif geometry_baseline_class == 1:
    combined_class = 1
    combined_score = 0.0
elif reference_domain_class == 1:
    combined_class = 1
    combined_score = 0.0
elif reference_domain_class == 2:
    combined_class = 2 if geometry_baseline_class == 2 else 3
    combined_score = geometry_baseline_score * 0.5
elif reference_domain_class == 3:
    combined_class = geometry_baseline_class
    combined_score = geometry_baseline_score
```

A narrow valley alone never creates a high damming class. Formation-domain
conditions only retain high or very-high potential where inflow angle and
channel gradient already support that geometry class. Class 5 is never promoted
by valley width or reference volume.

## Classes

`output_damming_potential_class` is a `uint8` GeoTIFF:

| Class | Meaning |
| ---: | --- |
| 0 | not evaluated / no candidate |
| 1 | low damming potential |
| 2 | moderate damming potential |
| 3 | high damming potential |
| 4 | very high damming potential |
| 5 | debris-flow connectivity favored |

High inflow angles around `60..90` degrees are interpreted as more favorable for
channel blockage. Low angles around `0..30` degrees are interpreted as more
favorable for downstream debris-flow connectivity.

The companion `output_damming_potential_score` is a transparent `0..1` score
based on inflow angle and the mean channel-slope modifier.

## Class Buffers

The candidate class raster records only evaluated stream/runout candidate cells.
The final damming-potential class raster applies buffers only to the higher
damming-potential classes:

| Candidate class | Buffer |
| ---: | ---: |
| 1 low damming potential | 0 m |
| 2 moderate damming potential | 0 m |
| 3 high damming potential | 5 m |
| 4 very high damming potential | 5 m |
| 5 debris-flow connectivity favored | 0 m |

When class 3 and class 4 buffers overlap, the higher damming-potential class
overwrites the lower one. Classes 1, 2, and 5 remain unbuffered.

## Outputs

Default outputs are written into a unique run folder:

```text
Sample Data/Output/Latest Runs/Step6
```

Default outputs include:

- `step6_runout_stream_intersection_mask.tif`
- `step6_runout_inflow_angle_deg.tif`
- `step6_runout_stream_angle_deg.tif`
- `step6_mean_runout_approach_angle_25m_deg.tif`
- `step6_mean_channel_slope_25m_deg.tif`
- `step6_candidate_damming_potential_class.tif`
- `step6_buffered_damming_potential_class.tif`
- `step6_runout_damming_potential_score.tif`
- `step6_height_above_stream_m.tif`
- `step6_valley_floor_mask.tif`
- `step6_raw_valley_width_m.tif`
- `step6_valley_floor_width_m.tif`
- `step6_vref2600_valley_domain.tif`
- `step6_vref2600_combined_damming_class.tif`
- `step6_vref2600_combined_damming_score.tif`
- `step6_runout_meanapproach25m_highveryhigh5m_summary.json`

GeoTIFF outputs preserve the DEM/reference grid and use LZW compression. Float
rasters use NoData `-9999.0`; byte rasters use `0` for not evaluated.

Step 6 writes and validates the complete output bundle in temporary staging
directories first. Official outputs are replaced only after every configured
raster, vector, and summary file is complete. A publication failure restores
the previous complete bundle. Output paths must be unique and cannot equal any
input path.

## Command Line

Run the smoke test:

```powershell
.\.venv\Scripts\python.exe .\steps\step6_landslide_damming_potential_LDP\step6_landslide_damming_potential_LDP.py --smoke-test
```

Run with the default config:

```powershell
.\.venv\Scripts\python.exe .\steps\step6_landslide_damming_potential_LDP\step6_landslide_damming_potential_LDP.py `
  --config .\steps\step6_landslide_damming_potential_LDP\config.default.json
```

Use `--overwrite` to replace existing Step 6 outputs.

## Limitations

Step 6 does not estimate landslide volume, dam height, channel width, discharge,
breach probability, or outburst-flood magnitude. Stream quality, slope-raster
quality, flow-direction quality, DEM resolution, vegetation/building artifacts,
broad floodplains, profile censoring, and the Step 4 runout mask strongly
affect the result.

The valley-width component estimates valley-floor or obstructable-valley width,
not active wetted-channel width. Results are conditional on an adopted 2,600 m3
reference delivered-deposit volume. The reference volume is not a prediction of
actual source or delivered volume for each individual landslide. The 60%
delivery fraction is a moderating scenario assumption and should be tested
against local observations where possible. The analysis does not estimate dam
height, dam stability, impounded-water volume, breach probability, or
outburst-flood magnitude.
