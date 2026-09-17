#!/usr/bin/env python3
"""
Step 4 input diagnostic for MLPREP_Plugin_v4_3.

WHY THIS EXISTS
---------------
step4_deposition_zone_mpi.cpp computes ELEVEN validation counters, but its
failure block only PRINTS SIX of them. The counter it hides is the one that
has been failing all along:

    local_source_without_valid_alpha

In the C++ (inside the per-cell validation loop):

    if (!sourceData->isNodata(x, y) && src > 0.0f) {
        if (!alpha_is_valid) {
            ++local_source_without_valid_alpha;      // <-- NEVER PRINTED
        } else {
            if (alpha_value < 6.0 || alpha_value > 72.0) {
                ++local_source_alpha_bounds;         // printed
            } else {
                ...
                ++local_sources;                     // "Initiating source cells"
            }
        }
    }

So a source cell only becomes an "initiating source cell" if alpha.tif is
valid AT THAT CELL. If alpha is NoData there, the cell silently vanishes into
an unprinted counter, "Initiating source cells" stays 0, and every printed
counter reads 0 -- which is exactly the log you have been staring at.

This script reads your existing rasters and reproduces all eleven counters,
including the hidden one. It is read-only and windowed, so it is fast and
will not exhaust RAM.

USAGE
-----
  python3 diagnose_step4_inputs.py "/path/to/Plugin_v4.3 Output 14/deposition_zone_output"

or edit OUTPUT_DIR below and run with no arguments.
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.windows import Window

OUTPUT_DIR = "/Users/ml-prepproject/Downloads/Plugin_v4.3 Output 14/deposition_zone_output"

# Constants copied verbatim from step4_deposition_zone_mpi.cpp
ALPHA_ANGLE_VALID_MIN = 1.0    # kAlphaAngleValidMin
ALPHA_ANGLE_VALID_MAX = 90.0   # kAlphaAngleValidMax
FIXED_ALPHA_MIN = 6.0          # kFixedAlphaMin
FIXED_ALPHA_MAX = 72.0         # kFixedAlphaMax
TWO_PI_EPS = 2.0 * np.pi + 1.0e-6
BINARY_TOL = 1.0e-7            # source_is_binary()

ROWS_PER_BLOCK = 512


def nodata_mask(arr, nd):
    """Mirror TauDEM's isNodata(): exact equality against the declared nodata.
    NaN does NOT equal nodata, so NaN cells count as real cells -- same as C++."""
    if nd is None:
        return np.zeros(arr.shape, dtype=bool)
    return arr == np.float32(nd)


def main(output_dir):
    paths = {
        "ang": os.path.join(output_dir, "ang.tif"),
        "fel": os.path.join(output_dir, "fel.tif"),
        "source": os.path.join(output_dir, "source_final.tif"),
        "dfi": os.path.join(output_dir, "dfi_final.tif"),
        "alpha": os.path.join(output_dir, "alpha.tif"),
    }
    for label, path in paths.items():
        if not os.path.exists(path):
            sys.exit(f"[FATAL] missing {label}: {path}")

    srcs = {k: rasterio.open(v) for k, v in paths.items()}
    ref = srcs["ang"]
    print(f"grid: {ref.width} x {ref.height}   crs={ref.crs}")
    for k, s in srcs.items():
        print(f"  {k:<7} dtype={str(s.dtypes[0]):<8} nodata={s.nodata}")
        if (s.width, s.height) != (ref.width, ref.height):
            print(f"  [!] {k} grid does not match ang.tif -- the C++ would exit 2, not 3.")

    c = dict(
        terrain=0, invalid_flow_angle=0,
        marked_sources=0, invalid_source=0,
        alpha_valid=0, invalid_alpha=0,
        sources=0, source_without_valid_alpha=0, source_alpha_bounds=0,
        valid_dfi=0, invalid_dfi=0,
        alpha_nodata_on_terrain=0,
    )
    alpha_at_src_min, alpha_at_src_max = np.inf, -np.inf

    for row0 in range(0, ref.height, ROWS_PER_BLOCK):
        h = min(ROWS_PER_BLOCK, ref.height - row0)
        w = Window(0, row0, ref.width, h)

        ang = srcs["ang"].read(1, window=w).astype(np.float32)
        fel = srcs["fel"].read(1, window=w).astype(np.float32)
        src = srcs["source"].read(1, window=w).astype(np.float32)
        dfi = srcs["dfi"].read(1, window=w).astype(np.float32)
        alp = srcs["alpha"].read(1, window=w).astype(np.float32)

        # C++: if (felData->isNodata || flowData->isNodata) continue;
        terrain = ~nodata_mask(fel, srcs["fel"].nodata) & ~nodata_mask(ang, srcs["ang"].nodata)
        c["terrain"] += int(terrain.sum())

        bad_flow = terrain & (~np.isfinite(ang) | (ang < 0.0) | (ang > TWO_PI_EPS))
        c["invalid_flow_angle"] += int(bad_flow.sum())

        src_present = terrain & ~nodata_mask(src, srcs["source"].nodata)
        not_binary = src_present & ~((np.abs(src) <= BINARY_TOL) |
                                     (np.abs(src - 1.0) <= BINARY_TOL))
        c["invalid_source"] += int(not_binary.sum())
        marked = src_present & (src > 0.0)
        c["marked_sources"] += int(marked.sum())

        alpha_present = terrain & ~nodata_mask(alp, srcs["alpha"].nodata)
        c["alpha_nodata_on_terrain"] += int((terrain & ~alpha_present).sum())
        alpha_bad = alpha_present & (~np.isfinite(alp) |
                                     (alp < ALPHA_ANGLE_VALID_MIN) |
                                     (alp > ALPHA_ANGLE_VALID_MAX))
        c["invalid_alpha"] += int(alpha_bad.sum())
        alpha_ok = alpha_present & ~alpha_bad
        c["alpha_valid"] += int(alpha_ok.sum())

        dfi_present = terrain & ~nodata_mask(dfi, srcs["dfi"].nodata)
        dfi_bad = dfi_present & (~np.isfinite(dfi) | (dfi < 0.0) | (dfi > 1.0))
        c["invalid_dfi"] += int(dfi_bad.sum())
        c["valid_dfi"] += int((dfi_present & ~dfi_bad).sum())

        no_alpha = marked & ~alpha_ok
        c["source_without_valid_alpha"] += int(no_alpha.sum())
        in_bounds = marked & alpha_ok & (alp >= FIXED_ALPHA_MIN) & (alp <= FIXED_ALPHA_MAX)
        out_bounds = marked & alpha_ok & ~in_bounds
        c["source_alpha_bounds"] += int(out_bounds.sum())
        c["sources"] += int(in_bounds.sum())

        sel = marked & alpha_ok
        if sel.any():
            alpha_at_src_min = min(alpha_at_src_min, float(alp[sel].min()))
            alpha_at_src_max = max(alpha_at_src_max, float(alp[sel].max()))

        del ang, fel, src, dfi, alp

    for s in srcs.values():
        s.close()

    print("\n=== counters the C++ computes (* = it never prints this one) ===")
    print(f"  Terrain-valid cells .................. {c['terrain']}")
    print(f"  Marked source cells (src > 0) ........ {c['marked_sources']}")
    print(f"  Initiating source cells .............. {c['sources']}")
    print(f"  invalid source cells (not 0.0/1.0) ... {c['invalid_source']}")
    print(f"* source cells w/o valid alpha ......... {c['source_without_valid_alpha']}")
    print(f"  source alpha outside [6, 72] ......... {c['source_alpha_bounds']}")
    print(f"  invalid alpha cells .................. {c['invalid_alpha']}")
    print(f"  alpha-valid cells .................... {c['alpha_valid']}")
    print(f"  alpha NoData on terrain .............. {c['alpha_nodata_on_terrain']}")
    print(f"  invalid D-Infinity angle cells ....... {c['invalid_flow_angle']}")
    print(f"  invalid DFI cells .................... {c['invalid_dfi']}")
    print(f"  valid DFI cells ...................... {c['valid_dfi']}")
    if np.isfinite(alpha_at_src_min):
        print(f"  alpha range at marked source cells ... "
              f"[{alpha_at_src_min:.3f}, {alpha_at_src_max:.3f}] degrees")

    print("\n=== verdict ===")
    if c["sources"] > 0 and c["invalid_source"] == 0 and c["invalid_alpha"] == 0 \
            and c["invalid_flow_angle"] == 0 and c["source_alpha_bounds"] == 0 \
            and c["invalid_dfi"] == 0 and c["valid_dfi"] > 0:
        print("  These inputs would PASS validation. The abort is elsewhere.")
        return

    if c["marked_sources"] == 0:
        print("  No source cells at all -- source_final.tif is empty or entirely NoData")
        print("  over the terrain. Check how it is written.")
    elif c["source_without_valid_alpha"] == c["marked_sources"]:
        print("  ROOT CAUSE CONFIRMED: every source cell sits where alpha.tif has no")
        print("  valid value, so the engine discards all of them into the counter it")
        print("  never prints. No amount of changing the SOURCE raster's dtype, NoData")
        print("  or pixel values can fix this -- the problem is alpha.tif's coverage.")
    elif c["source_without_valid_alpha"] > 0:
        print(f"  {c['source_without_valid_alpha']} of {c['marked_sources']} source cells")
        print("  lack a valid alpha. Restricting sources to alpha-valid cells will fix it.")
    if c["invalid_source"] > 0:
        print("  Source raster is not strictly binary. The engine requires values of")
        print("  EXACTLY 0.0 or 1.0 (tolerance 1e-7) -- probabilities, 0.99 and 1000.0")
        print("  all fail here. This is source_is_binary() in the C++.")
    if c["source_alpha_bounds"] > 0:
        print("  Some source cells have alpha outside [6, 72]. Sources must be placed")
        print("  only where alpha falls inside that window.")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else OUTPUT_DIR
    if not os.path.isdir(out):
        sys.exit(f"[FATAL] not a directory: {out}")
    main(out)
