#!/usr/bin/env python3
"""Headless entry point for the MLPREP pipelines.

This is merged_wrapper.py's orchestration without Qt. It runs the same scripts,
with the same flags, in the same order, against the same virtual environments --
nothing in py_files/, ml-prep-scripts/, the TauDEM chain or the models is
touched. The dialog's job was to collect inputs, chain the stages and show
progress; here the inputs arrive as command-line arguments, the chaining is
straight-line code, and progress is written to stdout as NDJSON so a caller can
follow a run that takes hours.

One event per line:

    {"event": "progress", "stage": "go-engine", "percent": 20, "message": "..."}
    {"event": "log",      "message": "raw line from a subprocess"}
    {"event": "result",   "status": "success", "outputs": {...}}
    {"event": "result",   "status": "error",   "message": "..."}

The only stage that cannot be called as a subprocess is the seam cleanup, which
the dialog runs through QGIS's `processing` module. QGIS is not importable
outside the desktop application, so it is invoked here through qgis_process --
the same algorithm, the same parameters, from the command line.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Matching merged_wrapper.py: the two environments its SetupWorker builds.
LS_VENV_PYTHON = os.path.expanduser("~/.mlprep_envs/ls_detect_env/bin/python")
EIL_VENV_PYTHON = os.path.expanduser("~/.mlprep_envs/venv/bin/python")

QGIS_PROCESS_CANDIDATES = [
    "/Applications/QGIS.app/Contents/MacOS/bin/qgis_process",
    "/usr/local/bin/qgis_process",
    "/usr/bin/qgis_process",
]

# The dialog hardcodes these; they are repeated rather than re-derived so a run
# from the web produces the same geometry as a run from the toolbar.
SLIVER_AREA_MIN = 5000

GO_PARAMS = {
    "areaMin": 5000,
    "thresh": 5000,
    "cvMin": 0.15,
    "rf": 10,
    "tileCols": 3,
    "tileRows": 3,
    "tileOverlap": 500,
}


def emit(payload):
    """Writes one NDJSON event, unbuffered so a watcher sees it immediately."""
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def log(message):
    emit({"event": "log", "message": message})


def progress(stage, percent, message=""):
    emit({"event": "progress", "stage": stage, "percent": percent, "message": message})


def fail(message):
    emit({"event": "result", "status": "error", "message": message})
    sys.exit(1)


def subprocess_env():
    """The environment merged_wrapper.py's _get_subprocess_env builds.

    Homebrew's bin has to be on PATH for the Go toolchain and for mpiexec, and
    the Qt plugin path QGIS exports breaks a plain Python child that imports
    anything Qt-adjacent, so it is dropped.
    """
    env = os.environ.copy()
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + env.get("PATH", "")
    for variable in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "PYTHONHOME"):
        env.pop(variable, None)
    return env


def stream(command, cwd=None, on_line=None):
    """Runs a command, forwarding each stdout line as a log event.

    Returns the last line that parsed as JSON, which is how landslide_worker.py
    and pinn_inference.py report their result.
    """
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=subprocess_env(),
    )

    last_json = None

    for line in iter(process.stdout.readline, ""):
        clean = line.rstrip()
        if not clean:
            continue
        try:
            parsed = json.loads(clean)
            if isinstance(parsed, dict):
                last_json = parsed
                continue
        except json.JSONDecodeError:
            pass
        log(clean)
        if on_line:
            on_line(clean)

    process.wait()

    if process.returncode != 0:
        raise RuntimeError("`%s` exited %d" % (os.path.basename(command[0]), process.returncode))

    return last_json


def locate_qgis_process():
    for candidate in QGIS_PROCESS_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return shutil.which("qgis_process")


def eliminate_seams(input_gpkg, output_gpkg, label):
    """Merges tile-boundary slivers into their largest neighbour.

    The dialog calls processing.run("qgis:eliminateselectedpolygons", MODE 2) on
    a selection of `$area < 5000`. qgis_process cannot carry a selection, so the
    same features are named by an OGR expression instead; the algorithm, the
    mode and the threshold are unchanged. A failure here is not fatal in the
    dialog either -- it falls back to the raw geometry, and so does this.
    """
    qgis_process = locate_qgis_process()

    if not qgis_process:
        log("[Warning] %s: qgis_process not found, keeping raw geometry." % label)
        return input_gpkg

    try:
        stream([
            qgis_process, "run", "native:eliminateselectedpolygons",
            "--INPUT=%s|subset=SELECT * FROM \"%s\" WHERE ST_Area(geom) < %d"
            % (input_gpkg, os.path.splitext(os.path.basename(input_gpkg))[0], SLIVER_AREA_MIN),
            "--MODE=2",
            "--OUTPUT=%s" % output_gpkg,
        ])
    except (RuntimeError, OSError) as error:
        log("[Warning] %s seam cleanup failed, using raw geometry: %s" % (label, error))
        return input_gpkg

    return output_gpkg if os.path.exists(output_gpkg) else input_gpkg


def run_landslide(args):
    """Landslide detection: one call, exactly as the dialog makes it."""
    output_mask = os.path.join(args.output_dir, "landslide_result.tif")

    command = [
        args.python or LS_VENV_PYTHON,
        os.path.join(BASE_DIR, "landslide_worker.py"),
        "--method", args.method,
        "--dem", args.dem,
        "--after", args.after,
        "--output", output_mask,
        "--roi_path", args.roi_path,
        "--filter_col", args.filter_col,
        "--filter_val", args.filter_val,
        "--model_path", os.path.join(BASE_DIR, "model_lsi", "default", "best.keras"),
        "--stats_path", os.path.join(BASE_DIR, "model_lsi", "default", "norm_stats.json"),
    ]

    if args.lulc:
        command += ["--lulc", args.lulc]
    if args.before:
        command += ["--before", args.before]

    progress("detection", 10, "Running %s detection" % args.method.upper())

    result = stream(command)

    if not result or result.get("status") != "success":
        fail((result or {}).get("message", "landslide_worker.py reported no result."))

    mask = result.get("output_mask", output_mask)

    progress("detection", 100, "Detection complete")
    emit({"event": "result", "status": "success", "outputs": {"Landslide mask": mask}})


def run_eil(args):
    """The five-stage EIL chain the dialog drives through Qt signals."""
    rasters = dict(pair.split("=", 1) for pair in args.raster)

    for required in ("NDVI", "BUK", "Slope"):
        if required not in rasters:
            fail("%s raster is required." % required)

    # The Go engine reads a directory of {PARAM}.tif, which is how the dialog
    # hands it the selection.
    raster_dir = tempfile.mkdtemp(prefix="go_rasters_")
    for name, path in rasters.items():
        try:
            os.symlink(path, os.path.join(raster_dir, "%s.tif" % name))
        except OSError:
            shutil.copy2(path, os.path.join(raster_dir, "%s.tif" % name))

    python_exe = args.python or EIL_VENV_PYTHON

    try:
        progress("go-engine", 5, "Building slope units and zonal features")

        go_binary = "/opt/homebrew/bin/go" if os.path.exists("/opt/homebrew/bin/go") else "go"

        stream([
            go_binary, "run", ".",
            "-dem", args.dem,
            "-rasterFolder", raster_dir,
            "-outputDir", args.output_dir,
            "-pythonExe", python_exe,
            "-targetCRS", args.crs,
            "-areaMin", str(GO_PARAMS["areaMin"]),
            "-thresh", str(GO_PARAMS["thresh"]),
            "-cvMin", str(GO_PARAMS["cvMin"]),
            "-rf", str(GO_PARAMS["rf"]),
            "-maxIter", str(args.max_iterations),
            "-tileCols", str(GO_PARAMS["tileCols"]),
            "-tileRows", str(GO_PARAMS["tileRows"]),
            "-tileOverlap", str(GO_PARAMS["tileOverlap"]),
        ], cwd=os.path.join(BASE_DIR, "ml-prep-scripts"))

        outputs = {}

        progress("seam-cleanup", 55, "Erasing tile seams")

        slope_units = os.path.join(args.output_dir, "Final_SlopeUnits.gpkg")
        if os.path.exists(slope_units):
            outputs["Slope units"] = eliminate_seams(
                slope_units,
                os.path.join(args.output_dir, "Cleaned_SlopeUnits.gpkg"),
                "Slope Units",
            )

        merged = os.path.join(args.output_dir, "Merged_PINN_Features.gpkg")
        if not os.path.exists(merged):
            fail("The Go engine produced no Merged_PINN_Features.gpkg.")

        merged = eliminate_seams(
            merged,
            os.path.join(args.output_dir, "Cleaned_PINN_Features.gpkg"),
            "PINN Features",
        )

        progress("pinn", 65, "Running the PINN model")

        hazard_map = os.path.join(args.output_dir, "Final_Hazard_Map.gpkg")

        result = stream([
            python_exe,
            os.path.join(BASE_DIR, "pinn_inference.py"),
            merged,
            args.model or os.path.join(BASE_DIR, "model", "my_model.keras"),
            hazard_map,
            args.dem,
        ])

        if not result or result.get("status") != "success":
            fail((result or {}).get("message", "pinn_inference.py reported no result."))

        outputs["Hazard map"] = result.get("output", hazard_map)

        overall_tif = result.get("overall_tif")
        if overall_tif:
            outputs["Hazard raster"] = overall_tif

        progress("pinn", 100, "Hazard map complete")
        emit({"event": "result", "status": "success", "outputs": outputs})
    finally:
        shutil.rmtree(raster_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Headless MLPREP pipelines.")
    parser.add_argument("--python", help="Interpreter to run pipeline scripts with.")
    parser.add_argument("--output-dir", required=True, dest="output_dir")

    subparsers = parser.add_subparsers(dest="pipeline", required=True)

    landslide = subparsers.add_parser("landslide")
    landslide.add_argument("--method", choices=["otsu", "cnn", "ensemble"], required=True)
    landslide.add_argument("--dem", required=True)
    landslide.add_argument("--after", required=True)
    landslide.add_argument("--before")
    landslide.add_argument("--lulc")
    landslide.add_argument("--roi-path", dest="roi_path", required=True)
    landslide.add_argument("--filter-col", dest="filter_col", required=True)
    landslide.add_argument("--filter-val", dest="filter_val", required=True)

    eil = subparsers.add_parser("eil")
    eil.add_argument("--dem", required=True)
    eil.add_argument("--raster", action="append", default=[],
                     help="NAME=path, repeated. NDVI, BUK and Slope are required.")
    eil.add_argument("--crs", default="EPSG:4326")
    eil.add_argument("--max-iterations", dest="max_iterations", type=int, default=10)
    eil.add_argument("--model")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.pipeline == "landslide":
            run_landslide(args)
        else:
            run_eil(args)
    except Exception as error:  # noqa: BLE001 - the caller only ever sees the event
        fail(str(error))


if __name__ == "__main__":
    main()
