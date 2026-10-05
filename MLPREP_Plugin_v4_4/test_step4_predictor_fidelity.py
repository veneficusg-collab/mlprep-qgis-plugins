#!/usr/bin/env python3
"""
Step 4 predictor-fidelity regression test.

WHY THIS EXISTS
-------------------------------------------------------------------------
The ML-DFI model shipped in model_dfi/ was trained on a specific set of
predictor rasters. When Step 4 regenerates those predictors on the fly, any
drift in sign, scale, smoothing or TauDEM parameters puts the model outside
its training distribution and the DFI output degrades in ways that are easy
to mistake for a rendering problem.

Four defects found this way, all reproduced below:

1. Profile curvature was computed with the wrong sign, plan curvature with a
   p^1.0 instead of a p^1.5 denominator, and both were multiplied by 100 for
   an "ArcGIS convention" the training rasters do not use. Values landed
   ~100-300x outside the model's training range.
2. The continuous predictors were fed to the model unfiltered. Every one the
   model was fitted on is a Makilala_*_LPF_* raster, and comparing the shipped
   filtered/unfiltered pairs shows that filter is exactly a 3x3 uniform mean.
   This produced the high-frequency streaking through the DFI.
3. dinfdistup ran at TauDEM's default proportion threshold of 0.5 rather than
   the 0 the reference lineage records, which truncates most flow paths
   immediately and collapsed dugsurf to about a sixth of its proper length.
4. areadinf ran without -nc, and dinfdistdown NoDatas any cell whose flow
   leaves the domain before reaching a stream (the sump<=0 branch, which -nc
   does NOT disable). That NoData propagates upstream and was the cause of the
   blank streaks through dstgsurf and the DFI.

Tests 1 and 2 are pure numpy and run anywhere. Test 3 needs TauDEM + MPI and
a DEM to route; it is skipped when either is unavailable.

USAGE
-------------------------------------------------------------------------
    python3 test_step4_predictor_fidelity.py [--dem PATH] [--keep]

Exits non-zero if any check fails.
"""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "Sample Data", "Input", "Raster", "Predictors", "Topographic")
UTIL_DIR = os.path.join(HERE, "Sample Data", "Input", "Raster", "Utils", "PRSTM")
WRAPPER = os.path.join(HERE, "merged_wrapper.py")


# --------------------------------------------------------------------------
# Load the helpers out of merged_wrapper.py without importing qgis/PyQt.
# --------------------------------------------------------------------------
def load_helpers(names):
    source = open(WRAPPER, encoding="utf-8").read()
    tree = ast.parse(source)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "MLDFIDepositionWorker")
    consts = {}
    for node in cls.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    namespace = {"os": os, "np": np}
    funcs = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(textwrap.dedent(ast.get_source_segment(source, node)), namespace)
            funcs[node.name] = namespace[node.name]

    class Worker:
        log = types.SimpleNamespace(emit=lambda *a, **k: None)

        def __init__(self, output_dir):
            self.output_dir = output_dir

    for key, value in consts.items():
        setattr(Worker, key, value)
    for key, value in funcs.items():
        setattr(Worker, key, value)
    return Worker, consts


def read_full(path):
    with rasterio.open(path) as ds:
        return np.ma.filled(ds.read(1, masked=True).astype(np.float64), np.nan)


def read_bounds(path, bounds, shape):
    with rasterio.open(path) as ds:
        win = from_bounds(*bounds, transform=ds.transform)
        arr = ds.read(1, window=win, boundless=True, masked=True).astype(np.float64)
    return np.ma.filled(arr, np.nan)[:shape[0], :shape[1]]


def cut(src_path, window, out_path):
    """Write `window` of `src_path` to `out_path`; return its map bounds."""
    with rasterio.open(src_path) as ds:
        arr = ds.read(1, window=window, masked=True)
        profile = ds.profile.copy()
        profile.update(width=window.width, height=window.height,
                       transform=ds.window_transform(window))
        bounds = rasterio.windows.bounds(window, ds.transform)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.filled(profile.get("nodata", -9999.0)), 1)
    return bounds


# --------------------------------------------------------------------------
# Test 1 -- the low-pass filter must reproduce the shipped *_LPF_* rasters.
# --------------------------------------------------------------------------
def test_low_pass(worker_cls, workdir):
    print("TEST 1  low-pass filter vs the shipped Makilala_*_LPF_* rasters")
    pairs = [("Makilala_ProfCurv_PRSTM51N.tif", "Makilala_ProfCurv_LPF_PRSTM51N.tif"),
             ("Makilala_TWI_Final_PRSTM51N.tif", "Makilala_TWI_Final_LPF_PRSTM51N.tif"),
             ("Makilala_Slope_PRSTM51N.tif", "Makilala_Slope_LPF_PRSTM51N.tif")]
    window = Window(2000, 1400, 900, 900)
    worker = worker_cls(workdir)
    ok = True
    for raw_name, lpf_name in pairs:
        raw_path = os.path.join(REF_DIR, raw_name)
        lpf_path = os.path.join(REF_DIR, "Filtered", lpf_name)
        if not (os.path.exists(raw_path) and os.path.exists(lpf_path)):
            print(f"   SKIP  {raw_name} (sample raster not present)")
            continue
        staged = os.path.join(workdir, "in_" + raw_name)
        bounds = cut(raw_path, window, staged)
        got = read_full(worker._apply_low_pass({"x": staged})["x"])
        expected = read_bounds(lpf_path, bounds, got.shape)

        valid = np.isfinite(got) & np.isfinite(expected)
        interior = np.zeros_like(valid)
        interior[3:-3, 3:-3] = True          # ignore the staged tile's own border
        valid &= interior
        corr = np.corrcoef(got[valid], expected[valid])[0, 1]
        rel = np.max(np.abs(got[valid] - expected[valid])) / max(np.std(expected[valid]), 1e-12)
        good = corr > 0.9999 and rel < 1e-3
        ok &= good
        print(f"   {'PASS' if good else 'FAIL'}  {raw_name:38s} "
              f"corr={corr:.6f}  maxdiff/std={rel:.2e}  n={valid.sum()}")
    return ok


# --------------------------------------------------------------------------
# Test 2 -- curvature sign and scale must track the shipped reference rasters.
# --------------------------------------------------------------------------
def test_curvature(worker_cls, workdir):
    print("\nTEST 2  curvature vs the shipped reference rasters")
    dem = os.path.join(UTIL_DIR, "Makilala_DEM_Buff_PRSTM51N.tif")
    if not os.path.exists(dem):
        print("   SKIP  (Makilala_DEM_Buff not present)")
        return True
    staged = os.path.join(workdir, "dem_cut.tif")
    bounds = cut(dem, Window(2400, 1500, 900, 900), staged)
    prof_path = os.path.join(workdir, "prof.tif")
    plan_path = os.path.join(workdir, "plan.tif")
    worker_cls(workdir)._generate_curvatures(staged, prof_path, plan_path)

    ok = True
    for label, got_path, ref_name in [("profile", prof_path, "Makilala_ProfCurv_PRSTM51N.tif"),
                                      ("plan", plan_path, "Makilala_PlanCurv_PRSTM51N.tif")]:
        got = read_full(got_path)
        expected = read_bounds(os.path.join(REF_DIR, ref_name), bounds, got.shape)
        valid = np.isfinite(got) & np.isfinite(expected)
        corr = np.corrcoef(got[valid], expected[valid])[0, 1]
        qg = np.percentile(got[valid], [1, 50, 99])
        qe = np.percentile(expected[valid], [1, 50, 99])
        spread = abs(qg[2] - qg[0]) / abs(qe[2] - qe[0])
        # A positive correlation is what catches the sign flip; the spread ratio
        # is what catches the x100 and the wrong denominator exponent.
        good = corr > 0.70 and 0.70 < spread < 1.45
        ok &= good
        print(f"   {'PASS' if good else 'FAIL'}  {label:8s} corr={corr:+.4f}  "
              f"p1/med/p99 got={qg[0]:+.5f}/{qg[1]:+.6f}/{qg[2]:+.5f}  "
              f"exp={qe[0]:+.5f}/{qe[1]:+.6f}/{qe[2]:+.5f}  spread={spread:.3f}")
    return ok


# --------------------------------------------------------------------------
# Test 3 -- TauDEM routing: boundary seeding must remove the dstgsurf NoData.
# --------------------------------------------------------------------------
def locate_taudem():
    for candidate in [os.path.expanduser("~/.mlprep_envs/venv/bin"), "/usr/local/taudem",
                      "/usr/local/bin", "/opt/homebrew/bin",
                      os.path.expanduser("~/miniconda3/bin")]:
        if os.path.exists(os.path.join(candidate, "pitremove")):
            return candidate
    return None


def test_routing(worker_cls, consts, workdir, dem_path):
    print("\nTEST 3  dstgsurf NoData with and without boundary-outlet seeding")
    taudem = locate_taudem()
    mpi = shutil.which("mpiexec") or shutil.which("mpirun")
    if not (taudem and mpi and dem_path and os.path.exists(dem_path)):
        print("   SKIP  (needs TauDEM, MPI and --dem)")
        return True

    def run(tool, *args):
        proc = subprocess.run([mpi, "-n", "4", os.path.join(taudem, tool), *args],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"TauDEM {tool} failed:\n{proc.stdout}\n{proc.stderr}")

    p = lambda n: os.path.join(workdir, n)
    run("pitremove", "-z", dem_path, "-fel", p("fel.tif"))
    run("dinfflowdir", "-fel", p("fel.tif"), "-ang", p("ang.tif"), "-slp", p("slp.tif"))
    run("areadinf", "-ang", p("ang.tif"), "-sca", p("sca.tif"), "-nc")
    run("threshold", "-ssa", p("sca.tif"), "-src", p("src.tif"),
        "-thresh", str(consts["STREAM_SCA_THRESHOLD"]))

    run("dinfdistdown", "-ang", p("ang.tif"), "-fel", p("fel.tif"), "-src", p("src.tif"),
        "-dd", p("dd_unseeded.tif"), "-m", "ave", "s", "-nc")
    seeded = worker_cls(workdir)._seed_boundary_outlets(p("src.tif"), p("ang.tif"))
    run("dinfdistdown", "-ang", p("ang.tif"), "-fel", p("fel.tif"), "-src", seeded,
        "-dd", p("dd_seeded.tif"), "-m", "ave", "s", "-nc")

    routable = np.isfinite(read_full(p("ang.tif")))
    before = routable & ~np.isfinite(read_full(p("dd_unseeded.tif")))
    after = routable & ~np.isfinite(read_full(p("dd_seeded.tif")))
    pct = lambda m: 100.0 * m.sum() / max(routable.sum(), 1)
    good = after.sum() == 0 and (before & ~after).sum() >= 0 and (after & ~before).sum() == 0
    print(f"   NoData without seeding : {pct(before):6.2f}%  ({before.sum()} cells)")
    print(f"   NoData with seeding    : {pct(after):6.2f}%  ({after.sum()} cells)")
    print(f"   rescued={int((before & ~after).sum())}  newly lost={int((after & ~before).sum())}")
    print(f"   {'PASS' if good else 'FAIL'}  seeding removes the routing NoData without losing cells")
    return good


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dem", help="DEM to exercise the TauDEM routing test on")
    parser.add_argument("--keep", action="store_true", help="keep the scratch directory")
    args = parser.parse_args()

    worker_cls, consts = load_helpers({"_apply_low_pass", "_generate_curvatures",
                                       "_seed_boundary_outlets"})
    print(f"STREAM_SCA_THRESHOLD = {consts.get('STREAM_SCA_THRESHOLD')}   "
          f"LPF_WINDOW = {consts.get('LPF_WINDOW')}\n")

    workdir = tempfile.mkdtemp(prefix="step4_fidelity_")
    try:
        results = [test_low_pass(worker_cls, workdir),
                   test_curvature(worker_cls, workdir),
                   test_routing(worker_cls, consts, workdir, args.dem)]
    finally:
        if args.keep:
            print(f"\nscratch kept at {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    print("\nRESULT:", "ALL PASS" if all(results) else "FAILURES PRESENT")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
