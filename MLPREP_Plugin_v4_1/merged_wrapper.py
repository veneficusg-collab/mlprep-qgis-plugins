#merged_wrapper.py

import os
import sys
import subprocess
import json
import platform
import shutil
import tempfile
import csv
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# QGIS / PyQt Imports
from qgis.PyQt import QtWidgets, QtCore, QtGui
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal, QVariant, QSize, QEvent
from qgis.PyQt.QtWidgets import (
    QMainWindow, QTabWidget, QStackedWidget, QMessageBox, 
    QProgressDialog, QFileDialog, QGraphicsScene, QTableWidgetItem
)
from qgis.PyQt.QtGui import QColor, QPixmap, QPen, QBrush, QImage

from qgis.core import (
    QgsVectorLayer, QgsRasterLayer, QgsProject, QgsMapSettings,
    QgsMapRendererParallelJob, QgsMapRendererSequentialJob, QgsCoordinateReferenceSystem,
    QgsFeatureRequest, QgsGeometry, QgsPointXY, QgsField,
    QgsGraduatedSymbolRenderer, QgsRendererRange, QgsSymbol, QgsSingleSymbolRenderer,
    QgsMapLayerType
)

# Import BOTH UI files
from .landslide_mainwindow import Ui_MainWindow as Ui_Landslide
from .eil_mainwindow import Ui_MainWindow as Ui_EIL

from pathlib import Path
import rasterio

class MLDFIDepositionWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)

    def __init__(self, dem_path, dfi_tif, output_dir, step4_wrapper_py, cores=4):
        super().__init__()
        self.dem_path = dem_path
        self.dfi_tif = dfi_tif  # AI's Hazard_Susceptibility raster
        self.output_dir = output_dir
        self.step4_wrapper_py = step4_wrapper_py
        self.cores = cores

        self.taudem_dir = self._locate_taudem()
        self.mpi_exe = self._locate_mpi()

    def _locate_taudem(self):
        candidates = [
            os.path.expanduser("~/.mlprep_envs/venv/bin"),
            "/usr/local/taudem", "/usr/local/bin", "/opt/homebrew/bin",
            os.path.expanduser("~/miniconda3/bin"),
        ]
        for c in candidates:
            if os.path.exists(os.path.join(c, "pitremove")):
                return c
        return None

    def _locate_mpi(self):
        for candidate in ["/opt/homebrew/bin/mpiexec", "/usr/local/bin/mpiexec", "/opt/homebrew/bin/mpirun"]:
            if os.path.exists(candidate):
                return candidate
        return "mpiexec"

    def _clean_dfi_raster(self, input_dfi, clean_dfi):
        """Scrub the AI output to fix floating-point anomalies (e.g. 1.000001) that crash C++."""
        import rasterio
        import numpy as np

        self.log.emit("   > Cleaning DFI Raster (Clamping probabilities to [0, 1])...")
        with rasterio.open(input_dfi) as src:
            dfi = src.read(1).astype(np.float32)
            
            # Valid mask: finite numbers that are not -9999 NoData
            valid_mask = np.isfinite(dfi) & (dfi > -1.0)
            
            # Clamp valid pixels strictly between 0.0 and 1.0
            dfi[valid_mask] = np.clip(dfi[valid_mask], 0.0, 1.0)
            
            # Force all invalid/nodata pixels to exactly -9999.0
            dfi[~valid_mask] = -9999.0
            
            meta = src.meta.copy()
            meta.update(dtype=rasterio.float32, nodata=-9999.0)
            with rasterio.open(clean_dfi, 'w', **meta) as dst:
                dst.write(dfi, 1)

    def _generate_alpha_raster(self, slope_tif, alpha_tif):
        """Replicates Step 0 logic to generate Alpha prior from TauDEM slope in degrees."""
        import rasterio
        import numpy as np

        self.log.emit("   > Generating Baseline Alpha Raster (Step 0 Logic)...")
        
        UPPER_BOUNDS = np.asarray([10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45], dtype=np.float64)
        ALPHA_VALUES = np.asarray([np.nan, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 22.0, 25.0, 28.0, 29.0, 30.0], dtype=np.float32)
        
        with rasterio.open(slope_tif) as src:
            slope_drop = src.read(1)
            
            # CRITICAL FIX: Convert TauDEM's drop/distance fraction back into Degrees!
            slope_degrees = np.degrees(np.arctan(slope_drop))
            
            valid_mask = (slope_drop >= 0) & (slope_drop != src.nodata)
            alpha = np.full(slope_drop.shape, -9999.0, dtype=np.float32)
            
            valid_classes = np.searchsorted(UPPER_BOUNDS, slope_degrees[valid_mask], side="right")
            positive_alpha = np.isfinite(ALPHA_VALUES[valid_classes])
            
            valid_rows, valid_cols = np.nonzero(valid_mask)
            alpha[valid_rows[positive_alpha], valid_cols[positive_alpha]] = ALPHA_VALUES[valid_classes[positive_alpha]]
            
            meta = src.meta.copy()
            meta.update(dtype=rasterio.float32, nodata=-9999.0)
            with rasterio.open(alpha_tif, 'w', **meta) as dst:
                dst.write(alpha, 1)

    def _generate_source_raster(self, clean_dfi_tif, source_tif, threshold=0.0):
        import rasterio
        import numpy as np

        self.log.emit(f"   > Generating Binary Source Raster (Target DFI > {threshold})...")
        with rasterio.open(clean_dfi_tif) as src:
            dfi = src.read(1)
            source = np.full(dfi.shape, -9999.0, dtype=np.float32)
            
            valid_dfi = dfi[(dfi != src.nodata) & np.isfinite(dfi)]
            if valid_dfi.size == 0:
                self.error.emit("DFI raster contains no valid probability cells.")
                return

            # Flag all cells with DFI > 0.0 as active initiation sources
            source_mask = (dfi > threshold) & (dfi != src.nodata) & np.isfinite(dfi)
            source_count = np.count_nonzero(source_mask)
            
            # Fallback only if no pixels are above 0.0
            if source_count == 0:
                max_val = float(np.max(valid_dfi))
                if max_val > 0.0:
                    source_mask = (dfi >= max_val) & (dfi != src.nodata)
                    source_count = np.count_nonzero(source_mask)
            
            self.log.emit(f"   > Source cells designated: {source_count}")
            source[source_mask] = 1.0
            
            meta = src.meta.copy()
            meta.update(dtype=rasterio.float32, nodata=-9999.0)
            with rasterio.open(source_tif, 'w', **meta) as dst:
                dst.write(source, 1)

    def run(self):
        import subprocess
        import sys
        import json
        
        try:
            if not self.taudem_dir:
                self.error.emit("TauDEM not found. Please install TauDEM.")
                return

            os.makedirs(self.output_dir, exist_ok=True)
            fel = os.path.join(self.output_dir, "fel.tif")
            ang = os.path.join(self.output_dir, "ang.tif")
            slp = os.path.join(self.output_dir, "slp.tif")
            alpha = os.path.join(self.output_dir, "alpha.tif")
            source = os.path.join(self.output_dir, "source.tif")
            dfi_clean = os.path.join(self.output_dir, "dfi_clean.tif")
            output_dep = os.path.join(self.output_dir, "step4_pure_depositional_mask.tif")
            
            mpi_base = [self.mpi_exe, "-n", str(self.cores)]
            
            clean_env = os.environ.copy()
            clean_env.pop("PYTHONHOME", None)
            clean_env.pop("PYTHONPATH", None)

            # 1. TauDEM Pre-processing
            self.log.emit("\n[TauDEM] Step 1/2: Pit-filling DEM...")
            subprocess.run(mpi_base + [os.path.join(self.taudem_dir, "pitremove"), "-z", self.dem_path, "-fel", fel], check=True, capture_output=True, env=clean_env)
            self.progress.emit(10)

            self.log.emit("\n[TauDEM] Step 2/2: Computing Flow Direction & Slope...")
            subprocess.run(mpi_base + [os.path.join(self.taudem_dir, "dinfflowdir"), "-fel", fel, "-ang", ang, "-slp", slp], check=True, capture_output=True, env=clean_env)
            self.progress.emit(25)

            # 2. Derive Mandatory Step 4 Inputs
            self._clean_dfi_raster(self.dfi_tif, dfi_clean)
            self.progress.emit(30)

            self._generate_alpha_raster(slp, alpha)
            self.progress.emit(35)
            
            self._generate_source_raster(dfi_clean, source, threshold=0.0)
            self.progress.emit(45)

            # 3. Locate the direct C++ engine (Bypassing Python Wrapper)
            step4_dir = os.path.dirname(self.step4_wrapper_py)
            cpp_exe_mac = os.path.join(step4_dir, "cpp_mpi_port", "build_manual", "step4_deposition_zone_mpi")
            cpp_exe_win = os.path.join(step4_dir, "cpp_mpi_port", "build_manual", "step4_deposition_zone_mpi.exe")
            cpp_exe = cpp_exe_win if sys.platform == "win32" else cpp_exe_mac

            if not os.path.exists(cpp_exe):
                self.error.emit(f"C++ routing executable not found at:\n{cpp_exe}\nPlease ensure it is compiled.")
                return

            # 4. Execute C++/MPI Routing Engine Directly
            self.log.emit("\n--- INITIATING C++/MPI DEPOSITIONAL ROUTING ---")
            cmd = [
                self.mpi_exe, "-n", str(self.cores),
                cpp_exe,
                "--fel", fel,
                "--ang", ang,
                "--source", source,
                "--dfi", dfi_clean,
                "--alpha", alpha,
                "--out-alpha", os.path.join(self.output_dir, "step4_dynamic_alpha.tif"),
                "--out-beta", os.path.join(self.output_dir, "step4_beta_angle.tif"),
                "--out-dfs", os.path.join(self.output_dir, "step4_dfs.tif"),
                "--out-mask", os.path.join(self.output_dir, "step4_runout_mask.tif"),
                "--out-deposition", output_dep,
                "--threshold", "0.2",
                "--dfi-mid", "0.69",
                "--alpha-gain-per-meter", "0.03333"
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=clean_env)
            for line in iter(process.stdout.readline, ''):
                clean = line.strip()
                if clean:
                    self.log.emit(f"   > MPI: {clean}")
            process.wait()

            if process.returncode != 0:
                self.error.emit(f"C++/MPI Step 4 failed with exit code {process.returncode}.")
                return

            self.progress.emit(100)
            self.log.emit(f"\n✅ Depositional Routing Complete!")
            self.finished.emit(output_dep, "success")

        except Exception as e:
            import traceback
            self.error.emit(f"Depositional Worker Error:\n{str(e)}\n{traceback.format_exc()}")

class GoParallelZonalWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str, str) 
    error = pyqtSignal(str)

    def __init__(self, go_project_dir, dem_path, input_raster_folder, output_folder, venv_python, target_crs, total_rasters, params):
        super().__init__()
        self.go_project_dir = go_project_dir 
        self.dem_path = dem_path
        self.input_raster_folder = input_raster_folder
        self.output_folder = output_folder
        self.venv_python = venv_python
        self.target_crs = target_crs
        self.total_rasters = total_rasters 
        self.params = params

    def run(self):
        try:
            self.log.emit("\n--- PASSING BATON TO GO ENGINE ---")
            
            # Cross-platform check to support Windows (go.exe) and Mac
            go_executable = "go"
            if sys.platform == "win32":
                go_executable = "go.exe"
            elif os.path.exists("/opt/homebrew/bin/go"):
                go_executable = "/opt/homebrew/bin/go"
            elif os.path.exists("/usr/local/bin/go"):
                go_executable = "/usr/local/bin/go"
            
            cmd = [
                go_executable, "run", ".", 
                "-dem", self.dem_path, 
                "-rasterFolder", self.input_raster_folder, 
                "-outputDir", self.output_folder,
                "-pythonExe", self.venv_python,  
                "-targetCRS", self.target_crs,
                "-areaMin", str(self.params.get('areaMin', 5000)),
                # "-areaMax", str(self.params.get('areaMax', 5000000)),
                "-thresh", str(self.params.get('thresh', 5000)),
                "-cvMin", str(self.params.get('cvMin', 0.15)),
                "-rf", str(self.params.get('rf', 10)),
                "-maxIter", str(self.params.get('maxIter', 10)),
                "-tileCols", str(self.params.get('tileCols', 3)),
                "-tileRows", str(self.params.get('tileRows', 3)),
                "-tileOverlap", str(self.params.get('tileOverlap', 50))
            ]

            self.log.emit(f"Executing: {' '.join(cmd)}")
            
            env = os.environ.copy()
            if sys.platform != "win32":
                env["PATH"] = f"/opt/homebrew/bin:/usr/local/bin:{env.get('PATH', '')}"
            
            process = subprocess.Popen(
                cmd,
                cwd=self.go_project_dir, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )

            # --- PROGRESS BAR MATH FOR BOTH PHASES ---
            current_progress = 0.0  
            # Phase 1 gets 50%, Phase 2 gets 45%, Merge gets 5%
            zonal_step = 45.0 / max(1, self.total_rasters) 

            for line in iter(process.stdout.readline, ''):
                clean_line = line.strip()
                if clean_line:
                    self.log.emit(f"   > GO: {clean_line}")
                    
                    # Track Phase 1 Progress
                    if "[Phase 1]" in clean_line and "Merging Tiles" in clean_line:
                        current_progress = 40.0
                        self.progress.emit(int(current_progress))
                    elif "[Phase 1]" in clean_line and "Polygonizing" in clean_line:
                        current_progress = 50.0
                        self.progress.emit(int(current_progress))
                    
                    # Track Phase 2 Progress
                    elif "[RASTER_DONE]" in clean_line:
                        current_progress += zonal_step
                        self.progress.emit(int(min(95, current_progress))) 
                    
                    elif "[MERGE_DONE]" in clean_line:
                        self.progress.emit(100)

            process.wait()

            if process.returncode == 0:
                self.log.emit("--- GO PARALLEL PIPELINE COMPLETE ---")
                self.progress.emit(100) 
                self.finished.emit(self.output_folder, "success")
            else:
                self.error.emit(f"Go engine crashed with exit code {process.returncode}")

        except Exception as e:
            import traceback
            self.error.emit(f"Go Subprocess Error:\n{str(e)}\n{traceback.format_exc()}")

class PINNPredictionWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(object, str)  

    def __init__(self, python_exe, script_path, input_gpkg, model_path, output_gpkg, dem_path):
        super().__init__()
        self.python_exe = python_exe
        self.script_path = script_path
        self.input_gpkg = input_gpkg
        self.model_path = model_path
        self.output_gpkg = output_gpkg
        self.dem_path = dem_path

    def run(self):
        try:
            self.log.emit("\n--- INITIATING PINN AI MODEL ---")
            self.log.emit("Loading TensorFlow and parsing features...")

            cmd = [
                self.python_exe, self.script_path,
                self.input_gpkg, self.model_path, self.output_gpkg, self.dem_path
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge stderr into stdout
                text=True
            )

            last_line = ""
            for line in iter(process.stdout.readline, ''):
                clean = line.strip()
                if not clean:
                    continue
                # Try to parse as final JSON status
                try:
                    response = json.loads(clean)
                    if "status" in response:
                        last_line = clean  # save for final handling
                        continue
                except json.JSONDecodeError:
                    pass
                # Everything else goes to the log panel
                self.log.emit(f"   > PINN: {clean}")

            process.wait()

            if last_line:
                response = json.loads(last_line)
                if response.get("status") == "success":
                    self.log.emit("✅ AI Predictions generated successfully!")
                    self.finished.emit(response, "success")  # ← pass the whole dict
                else:
                    self.log.emit(f"❌ AI Prediction Failed:\n{response.get('message')}")
                    self.log.emit(response.get('traceback', ''))
                    self.finished.emit({}, "error")  # ← empty dict on error
            else:
                self.log.emit("❌ AI Script produced no final status line.")
                self.finished.emit({}, "error")

        except Exception as e:
            self.log.emit(f"Worker Error: {str(e)}")
            self.finished.emit({}, "error")

# ==============================================================================
# EIL WORKER THREAD
# ==============================================================================
class PredictionWorker(QThread):
    finished = pyqtSignal(dict, dict)
    error = pyqtSignal(str)

    def __init__(self, cmd, env, output_csv, output_gpkg):
        super().__init__()
        self.cmd = cmd
        self.env = env
        self.output_csv = output_csv
        self.output_gpkg = output_gpkg

    def run(self):
        try:
            process = subprocess.run(self.cmd, capture_output=True, text=True, env=self.env)
            if process.returncode != 0:
                self.error.emit(f"Process Failed:\n{process.stderr}")
                return
            if not os.path.exists(self.output_csv):
                self.error.emit("Worker finished but output CSV not found.")
                return

            csv_data = []
            with open(self.output_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    csv_data.append(row)
            
            self.finished.emit({"rows": csv_data}, {})
        except Exception as e:
            self.error.emit(str(e))

# ==============================================================================
# AUTO-SETUP THREAD (First-time installation)
# ==============================================================================
class SetupWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, base_dir, ls_python, eil_python):
        super().__init__()
        self.base_dir = base_dir
        self.ls_python = ls_python
        self.eil_python = eil_python
        self.envs_dir = os.path.join(os.path.expanduser("~"), ".mlprep_envs")
        os.makedirs(self.envs_dir, exist_ok=True)

    def get_base_python(self):
        import sys, os, platform
        if platform.system() == 'Windows':
            py_path = os.path.join(sys.prefix, 'python.exe')
            if os.path.exists(py_path): return py_path
        elif platform.system() == 'Darwin':
            py_path = os.path.join(sys.prefix, 'bin', 'python3')
            if os.path.exists(py_path): return py_path
            
        if os.path.basename(sys.executable).lower().startswith('python'):
            return sys.executable
        return "python3"

    def run(self):
        import os, sys, subprocess
        base_python = self.get_base_python()

        setup_env = os.environ.copy()
        setup_env.pop("PYTHONHOME", None)
        setup_env.pop("PYTHONPATH", None)
        
        if sys.platform == 'win32':
            qgis_root = os.path.dirname(os.path.dirname(sys.prefix))
            qgis_bin = os.path.join(qgis_root, "bin")
            if os.path.exists(qgis_bin):
                setup_env["PATH"] = qgis_bin + os.pathsep + setup_env.get("PATH", "")

        try:
            ls_venv_dir = os.path.join(self.envs_dir, "ls_detect_env")
            if not os.path.exists(ls_venv_dir):
                self.progress.emit("Creating Landslide Environment...")
                subprocess.run([base_python, "-m", "venv", ls_venv_dir], check=True, env=setup_env)
                
                self.progress.emit("Installing Landslide Dependencies (This may take a few minutes)...")
                req_ls = os.path.join(self.base_dir, "requirements_ls.txt")
                res = subprocess.run([self.ls_python, "-m", "pip", "install", "-r", req_ls], capture_output=True, text=True, env=setup_env)
                if res.returncode != 0:
                    raise Exception(f"Landslide PIP Error:\n{res.stderr}")

            eil_venv_dir = os.path.join(self.envs_dir, "venv")
            if not os.path.exists(eil_venv_dir):
                self.progress.emit("Creating EIL Environment...")
                subprocess.run([base_python, "-m", "venv", eil_venv_dir], check=True, env=setup_env)
                
                self.progress.emit("Installing EIL Dependencies (This may take a few minutes)...")
                req_eil = os.path.join(self.base_dir, "requirements_eil.txt")
                res = subprocess.run([self.eil_python, "-m", "pip", "install", "-r", req_eil], capture_output=True, text=True, env=setup_env)
                if res.returncode != 0:
                    raise Exception(f"EIL PIP Error:\n{res.stderr}")

            self.finished.emit(True, "Setup Complete!")
        except Exception as e:
            self.finished.emit(False, str(e))

# ==============================================================================
# STEP 5: POST-DEPOSITONAL SPREAD (PDS) WORKER
# ==============================================================================
class Step5SpreadWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)

    def __init__(self, step5_script_path, config_dict, python_exe):
        super().__init__()
        self.step5_script_path = step5_script_path
        self.config_dict = config_dict
        self.python_exe = python_exe

    def run(self):
        try:
            self.log.emit("\n--- INITIATING STEP 5: POST-DEPOSITIONAL SPREAD (PDS) ---")
            
            temp_config_path = os.path.join(tempfile.gettempdir(), "step5_run_config.json")
            with open(temp_config_path, "w", encoding="utf-8") as f:
                json.dump(self.config_dict, f, indent=2)

            # Strictly use the virtual environment Python
            cmd = [self.python_exe, self.step5_script_path, "--config", temp_config_path]
            
            # Clean the environment so QGIS doesn't interfere with the venv
            clean_env = os.environ.copy()
            clean_env.pop("PYTHONHOME", None)
            clean_env.pop("PYTHONPATH", None)

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=clean_env)
            for line in iter(process.stdout.readline, ''):
                clean = line.strip()
                if clean:
                    self.log.emit(f"   > PDS: {clean}")
            process.wait()

            if process.returncode != 0:
                self.error.emit(f"Step 5 PDS failed with exit code {process.returncode}.")
                return

            out_combined = self.config_dict.get("output_combined_mask")
            self.progress.emit(100)
            self.finished.emit(out_combined, "success")

        except Exception as e:
            import traceback
            self.error.emit(f"Step 5 Worker Error:\n{str(e)}\n{traceback.format_exc()}")


# ==============================================================================
# STEP 6: LANDSLIDE DAMMING POTENTIAL (LDP) WORKER
# ==============================================================================
class Step6DammingWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)

    def __init__(self, step6_script_path, config_dict, python_exe):
        super().__init__()
        self.step6_script_path = step6_script_path
        self.config_dict = config_dict
        self.python_exe = python_exe

    def run(self):
        try:
            self.log.emit("\n--- INITIATING STEP 6: LANDSLIDE DAMMING POTENTIAL (LDP) ---")
            
            temp_config_path = os.path.join(tempfile.gettempdir(), "step6_run_config.json")
            with open(temp_config_path, "w", encoding="utf-8") as f:
                json.dump(self.config_dict, f, indent=2)

            # Strictly use the virtual environment Python
            cmd = [self.python_exe, self.step6_script_path, "--config", temp_config_path, "--overwrite"]
            
            # Clean the environment so QGIS doesn't interfere with the venv
            clean_env = os.environ.copy()
            clean_env.pop("PYTHONHOME", None)
            clean_env.pop("PYTHONPATH", None)

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=clean_env)
            for line in iter(process.stdout.readline, ''):
                clean = line.strip()
                if clean:
                    self.log.emit(f"   > LDP: {clean}")
            process.wait()

            if process.returncode != 0:
                self.error.emit(f"Step 6 LDP failed with exit code {process.returncode}.")
                return

            out_damming = self.config_dict.get("output_damming_potential_class")
            self.progress.emit(100)
            self.finished.emit(out_damming, "success")

        except Exception as e:
            import traceback
            self.error.emit(f"Step 6 Worker Error:\n{str(e)}\n{traceback.format_exc()}")



# ==============================================================================
# MAIN UNIFIED DIALOG
# ==============================================================================
class UnifiedPluginDialog(QMainWindow):
    def __init__(self, parent=None):
        super(UnifiedPluginDialog, self).__init__(parent)
        self.setWindowTitle("MLPREP Detection Suite")
        
        self.centralwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralwidget)
        
        self.main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.main_tabs = QTabWidget(self.centralwidget)
        self.main_layout.addWidget(self.main_tabs)

        self.ui_ls = Ui_Landslide()
        self.ui_eil = Ui_EIL()

        self.dummy_ls = QMainWindow()
        self.ui_ls.setupUi(self.dummy_ls)
        
        self.dummy_eil = QMainWindow()
        self.ui_eil.setupUi(self.dummy_eil)

        # Initialize the Graph Canvas in the Bottom Middle Box
        self.figure_hazard = Figure(figsize=(5, 4), dpi=100)
        self.figure_hazard.patch.set_facecolor('#ffffff') # White background
        self.canvas_hazard = FigureCanvas(self.figure_hazard)
        self.ui_eil.layout_Middle2.addWidget(self.canvas_hazard)

        self.tab_landslide = QtWidgets.QWidget()
        self.layout_ls = QtWidgets.QHBoxLayout(self.tab_landslide)
        self.layout_ls.setContentsMargins(0, 0, 0, 0)
        self.layout_ls.setSpacing(0)
        self.layout_ls.addWidget(self.ui_ls.Widget_Map_Area, stretch=3)
        self.layout_ls.addWidget(self.ui_ls.scrollArea, stretch=1)

        self.tab_eil = self.ui_eil.centralwidget

        self.main_tabs.addTab(self.tab_landslide, "Landslide Detection")
        self.main_tabs.addTab(self.tab_eil, "EIL Detection")

        self.ui_eil.plainTextEdit_Details.setReadOnly(True)

        self.feature_info = {
            "DEM": {
                "title": "Digital Elevation Model (DEM)", 
                "desc": "A 3D computer graphics representation of elevation data to represent terrain. It serves as the foundational layer for deriving slope, wetness, and other topographic metrics."
            },
            "BulkDensity": {
                "title": "Bulk Density", 
                "desc": "The weight of soil per unit volume. High bulk density indicates low pore space and high compaction, affecting water movement."
            },
            "Clay": {
                "title": "Clay Content", 
                "desc": "Percentage of clay particles in the soil. High clay content can increase cohesion but may significantly reduce drainage."
            },
            "Sand": {
                "title": "Sand Content", 
                "desc": "Percentage of sand particles. Sand increases drainage and reduces the overall stickiness and cohesion of the soil."
            },
            "Silt": {
                "title": "Silt Content", 
                "desc": "Percentage of silt particles. Silt-heavy soils can be highly susceptible to erosion and landslides when saturated."
            },
            "Eastness": {
                "title": "Eastness", 
                "desc": "A quantitative terrain parameter used to analyze land-surface orientation (aspect). It measures the degree to which a slope faces east."
            },
            "Northness": {
                "title": "Northness", 
                "desc": "Measures the degree to which a slope faces north. Combined with Eastness, it creates a continuous gradient of slope direction."
            },
            "Slope": {
                "title": "Slope Angle", 
                "desc": "The steepness of the terrain, measured in degrees. Steeper slopes generally have a higher risk of landslide occurrence."
            },
            "PGA": {
                "title": "Peak Ground Acceleration (PGA)", 
                "desc": "Measures the maximum ground shaking intensity at a specific location during an earthquake, representing the largest amplitude of acceleration recorded on an accelerogram."
            },
            "NDVI": {
                "title": "Normalized Difference Vegetation Index (NDVI)", 
                "desc": "A remote sensing indicator used to measure and analyze the health and density of green vegetation. Plant roots help stabilize soil."
            },
            "PRC": {
                "title": "Precipitation", 
                "desc": "The process where water vapor condenses in the atmosphere to form water droplets that fall to the Earth as rain. Intense rainfall is a primary trigger for landslides."
            },
            "HorizontalCurve": {
                "title": "Horizontal Curvature", 
                "desc": "Also known as Plan Curvature. It describes the curvature of contour lines and influences the convergence or divergence of surface water flow."
            },
            "VerticalCurve": {
                "title": "Vertical Curvature", 
                "desc": "Also known as Profile Curvature. It measures the rate of change of slope, influencing the acceleration and deceleration of surface flow."
            },
            "SoilThickness": {
                "title": "Soil Thickness", 
                "desc": "The depth of the soil layer above bedrock. Thicker soils can absorb more water but also provide more mass that can potentially fail and slide."
            },
            "LULC": {
                "title": "Land Use / Land Cover", 
                "desc": "Categorizes the physical material at the surface of the earth (e.g., forest, agriculture, urban). Different land covers affect water infiltration and root cohesion."
            },
            "DistanceToRiver": {
                "title": "Distance to River", 
                "desc": "Proximity to the nearest river or stream. Rivers can undercut slopes through lateral erosion, increasing landslide susceptibility."
            },
            "DistanceToRoad": {
                "title": "Distance to Road", 
                "desc": "Proximity to the nearest road network. Road construction often involves cutting into slopes, which can alter drainage and destabilize the terrain."
            },
            "SoilType": {
                "title": "Soil Type", 
                "desc": "Categorical classification of soils based on their physical and chemical properties, which heavily influences internal friction and cohesion."
            },
            "TWI": {
                "title": "Topographic Wetness Index", 
                "desc": "A steady-state wetness index. It identifies areas where water tends to accumulate based on local topography."
            },
            "DistanceToFault": {
                "title": "Distance to Fault", 
                "desc": "Proximity to tectonic fault lines. Areas closer to active faults experience more structural rock damage and higher earthquake intensities."
            },
            "ContributingFactor": {
                "title": "Contributing Factor", 
                "desc": "Also known as Catchment Area. Represents the total area uphill that drains water into this specific unit, drastically affecting soil saturation."
            }
        }

        self.apply_master_stylesheet()

        self._base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.ls_venv_python = self.get_venv_python_path("ls_detect_env")
        self.eil_venv_python = self.get_venv_python_path("venv")

        self.current_result_path_ls = None
        self.location_index = {}
        roi_dir = os.path.join(self._base_dir, "roi_data")
        self.admin_config = {
            "Region":       {"path": os.path.join(roi_dir, "phl_admbnda_adm1_psa_namria_20231106.shp"), "col": "ADM1_EN"},
            "Province":     {"path": os.path.join(roi_dir, "phl_admbnda_adm2_psa_namria_20231106.shp"), "col": "ADM2_EN"},
            "Municipality": {"path": os.path.join(roi_dir, "phl_admbnda_adm3_psa_namria_20231106.shp"), "col": "ADM3_EN"},
            "Barangay":     {"path": os.path.join(roi_dir, "phl_admbnda_adm4_psa_namria_20231106.shp"), "col": "ADM4_EN"}
        }

        self.MODEL_PATH = os.path.join(self._base_dir, "model", "my_model.keras")

        self.build_dynamic_landslide_ui()
        self.connect_landslide_signals()
        self.connect_eil_signals()

        QtCore.QTimer.singleShot(500, self.check_and_setup_environments)

    def browse_local_raster(self, combobox):
        """Opens a file dialog for the user to select a local .tif file."""
        
        filename, _ = QFileDialog.getOpenFileName(self, "Select Local Raster", "", "GeoTIFF (*.tif *.tiff)")
        if filename:
            # Add the file to the dropdown (Display name, hidden absolute path)
            combobox.addItem(os.path.basename(filename), filename)
            # Immediately select the item they just added
            combobox.setCurrentIndex(combobox.count() - 1)

    def check_and_setup_environments(self):
        envs_dir = os.path.join(os.path.expanduser("~"), ".mlprep_envs")
        ls_venv_dir = os.path.join(envs_dir, "ls_detect_env")
        eil_venv_dir = os.path.join(envs_dir, "venv")
        
        if os.path.exists(ls_venv_dir) and os.path.exists(eil_venv_dir):
            return

        reply = QMessageBox.question(
            self, 
            "First Time Setup Required", 
            "The MLPREP Detection Suite requires Python dependencies to run.\n\nWould you like to install them now?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.setup_dialog = QProgressDialog("Initializing Setup...", None, 0, 0, self)
            self.setup_dialog.setWindowTitle("Installing Dependencies")
            self.setup_dialog.setWindowModality(Qt.WindowModal)
            self.setup_dialog.setCancelButton(None)
            self.setup_dialog.show()

            self.setup_worker = SetupWorker(self._base_dir, self.ls_venv_python, self.eil_venv_python)
            self.setup_worker.progress.connect(self.setup_dialog.setLabelText)
            self.setup_worker.finished.connect(self.on_setup_finished)
            self.setup_worker.start()

    def on_setup_finished(self, success, message):
        self.setup_dialog.close()
        if success:
            QMessageBox.information(self, "Setup Complete", "All dependencies installed successfully!")
        else:
            QMessageBox.critical(self, "Setup Failed", f"An error occurred:\n{message}")

    def apply_master_stylesheet(self):
        original_css = """
            QMainWindow { background-color: #2b2b2b; }
            QTabWidget::pane { border: none; background-color: #D7D5D2; }
            QTabBar::tab { background-color: #b0b0b0; color: black; padding: 10px 20px; font-weight: bold; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; }
            QTabBar::tab:selected { background-color: #D7D5D2; }

            QWidget#Widget_Map_Area { background-color: #ffffff; }
            QScrollArea#scrollArea, QWidget#Widget_Inputs { background-color: #D7D5D2; border: none; }
            
            QWidget#Widget_Map_Area QGroupBox, QWidget#Widget_Inputs QGroupBox { font: 600 12px "Arial"; color: #E0E0E0; border: 2px solid #555; border-radius: 10px; margin-top: 16px; background-color: transparent; padding: 12px; }
            QWidget#Widget_Map_Area QGroupBox::title { background-color: #ffffff; color: black; subcontrol-origin: margin; subcontrol-position: top center; padding: 0 8px; position: absolute; margin-top: 10px;}
            QWidget#Widget_Inputs QGroupBox::title { background-color: #D7D5D2; color: black; subcontrol-origin: margin; subcontrol-position: top center; padding: 0 8px; position: absolute; margin-top: 10px;}

            QLabel#Lable_Landslide { color: black; font: 700 16pt "Arial"; }
            
            QPushButton#Button_Detect, QPushButton#Button_Save { font-family: montserrat; border: 1px solid #3a3a3a; border-radius: 6px; background-color: #ffffff; color: black; padding: 6px 14px; font-weight: bold; font-size: 10px; min-height: 10px; min-width: 55px; }
            QComboBox#ComboBox_ROI_Level, QComboBox#ComboBox_ROI_Name { border: 1px solid #3a3a3a; border-radius: 6px; padding: 4px 10px; background-color: #D5D5D5; color: black; font-size: 11px; min-height: 25px; }
            QLineEdit#LineEdit_Search { border: 1px solid #3a3a3a; border-radius: 6px; padding: 4px 10px; background-color: #ffffff; color: black; font-size: 11px; min-height: 25px; }

            QWidget#scrollAreaWidgetContentsLeft QLabel, QWidget#scrollAreaWidgetContentsLeft QCheckBox { color: black; }
        """
        self.setStyleSheet(original_css)

    def get_venv_python_path(self, venv_name):
        import platform
        envs_dir = os.path.join(os.path.expanduser("~"), ".mlprep_envs")
        if platform.system() == "Windows":
            return os.path.join(envs_dir, venv_name, "Scripts", "python.exe")
        else:
            return os.path.join(envs_dir, venv_name, "bin", "python")

    def _get_subprocess_env(self):
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        if sys.platform == 'win32':
            qgis_root = os.path.dirname(os.path.dirname(sys.prefix))
            qgis_bin = os.path.join(qgis_root, "bin")
            if os.path.exists(qgis_bin):
                env["PATH"] = qgis_bin + os.pathsep + env.get("PATH", "")
        return env

    def disable_cb_scroll(self, cb):
        cb.wheelEvent = lambda event: event.ignore()

    def eventFilter(self, source, event):
            # 1. When the mouse ENTERS the label's space
            if event.type() == QtCore.QEvent.Enter:
                
                # Loop through our dictionary to find which label triggered the hover
                for name, info in self.feature_info.items():
                    label_obj = getattr(self.ui_eil, f"label_{name}", None)
                    
                    if source == label_obj:
                        # Clear any existing text first
                        self.ui_eil.plainTextEdit_Details.clear()
                        
                        # Build the HTML string with inline CSS
                        html_format = f"""
                            <h1 style="color: #000000; margin-top: 20px; margin-bottom: 20px; margin-left: 20px;">{info['title']}</h1>
                            <p style="font-size: 24px; margin-top: 0px; margin-bottom: 20px; margin-left: 20px; margin-right: 20px; line-height: 1.4;">
                                {info['desc']}
                            </p>
                        """
                        
                        # Inject the HTML into the box
                        self.ui_eil.plainTextEdit_Details.appendHtml(html_format)
                        return True # Tell PyQt we successfully handled this event
                        
            # 2. When the mouse LEAVES the label's space
            elif event.type() == QtCore.QEvent.Leave:
                self.ui_eil.plainTextEdit_Details.clear()
                
                # Optional: Add a default placeholder text when nothing is hovered
                self.ui_eil.plainTextEdit_Details.appendHtml(
                    "<p style='color: #666666; font-style: italic;'>Hover over a parameter label to view its details.</p>"
                )
                return True

            # Pass all other events back to the normal system
            return super(UnifiedPluginDialog, self).eventFilter(source, event)
    
    def update_municipality_table(self, final_gpkg_path):
            from qgis.core import (
                QgsVectorLayer, QgsSpatialIndex, QgsCoordinateTransform, QgsProject
            )
            from qgis.PyQt import QtWidgets

            try:
                self.ui_eil.plainTextEdit_Logs.appendPlainText("\n--> Calculating Top 5 Hazardous Municipalities...")
                QtWidgets.QApplication.processEvents() # Keeps UI from freezing during math
                
                roi_config = self.admin_config.get("Municipality")
                if not roi_config or not os.path.exists(roi_config["path"]):
                    self.ui_eil.plainTextEdit_Logs.appendPlainText("[Warning] Municipality ROI shapefile not found.")
                    return

                # 1. Load layers natively in QGIS
                hazard_layer = QgsVectorLayer(final_gpkg_path, "hazard_temp", "ogr")
                roi_layer = QgsVectorLayer(roi_config["path"], "roi_temp", "ogr")
                roi_col = roi_config["col"]

                if not hazard_layer.isValid() or not roi_layer.isValid():
                    self.ui_eil.plainTextEdit_Logs.appendPlainText("[Error] Could not load layers for table generation.")
                    return

                # 2. Setup Coordinate Transformation (In case your GPKG and ROI have different EPSGs)
                coord_transform = None
                if hazard_layer.crs() != roi_layer.crs():
                    coord_transform = QgsCoordinateTransform(hazard_layer.crs(), roi_layer.crs(), QgsProject.instance().transformContext())

                # 3. Build a fast Spatial Index for the ROI Map
                roi_names = {}
                roi_index = QgsSpatialIndex()
                
                for feat in roi_layer.getFeatures():
                    roi_names[feat.id()] = str(feat[roi_col])
                    roi_index.addFeature(feat)

                muni_stats = {}
                idx_sus = hazard_layer.fields().indexOf("Hazard_Susceptibility")

                # 4. Loop through every generated slope unit
                for feat in hazard_layer.getFeatures():
                    geom = feat.geometry()
                    if geom.isNull(): continue
                    
                    # Get the center point of the slope unit
                    centroid = geom.centroid()
                    
                    # Transform point to match the ROI map
                    if coord_transform:
                        centroid.transform(coord_transform)
                    
                    # Ask the spatial engine which city this point landed in
                    intersecting_ids = roi_index.intersects(centroid.boundingBox())
                    
                    muni_name = "Unknown"
                    if intersecting_ids:
                        # Do a precise boundary check
                        for roi_id in intersecting_ids:
                            roi_feat = roi_layer.getFeature(roi_id)
                            if roi_feat.geometry().contains(centroid):
                                muni_name = roi_names[roi_id]
                                break
                        # Fallback if edge-case
                        if muni_name == "Unknown":
                            muni_name = roi_names[intersecting_ids[0]]
                    
                    if muni_name == "Unknown":
                        continue

                    if muni_name not in muni_stats:
                        muni_stats[muni_name] = {"High": 0, "Moderate": 0, "Low": 0}
                    
                    # Categorize the Hazard value
                    val = feat.attributes()[idx_sus]
                    try:
                        val_float = float(val)
                        if val_float > 0.66:
                            muni_stats[muni_name]["High"] += 1
                        elif val_float >= 0.33:
                            muni_stats[muni_name]["Moderate"] += 1
                        else:
                            muni_stats[muni_name]["Low"] += 1
                    except:
                        muni_stats[muni_name]["Low"] += 1

                # 5. Sort dictionary by High hazard counts (descending), take top 10
                sorted_stats = sorted(muni_stats.items(), key=lambda x: x[1]["High"], reverse=True)[:10]

                # 6. Inject the data into the QTableWidget
                table = self.ui_eil.tableWidget_Hazard
                table.setRowCount(len(sorted_stats))
                
                for row_idx, (muni_name, stats) in enumerate(sorted_stats):
                    table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(muni_name))
                    table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(str(stats['High'])))
                    table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(str(stats['Moderate'])))
                    table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(str(stats['Low'])))

                self.ui_eil.plainTextEdit_Logs.appendPlainText("--> Table successfully updated!")


                # --- NEW STEP: DRAW THE STACKED BAR CHART ---
                munis = [x[0] for x in sorted_stats]
                highs = [x[1]["High"] for x in sorted_stats]
                mods = [x[1]["Moderate"] for x in sorted_stats]
                lows = [x[1]["Low"] for x in sorted_stats]

                self.figure_hazard.clear()
                ax = self.figure_hazard.add_subplot(111)

                ind = np.arange(len(munis))
                width = 0.6

                # Stack the bars on top of each other (High -> Moderate -> Low)
                p1 = ax.bar(ind, highs, width, color='#ff0000') # Red
                p2 = ax.bar(ind, mods, width, bottom=highs, color='#7030a0') # Purple
                p3 = ax.bar(ind, lows, width, bottom=np.array(highs)+np.array(mods), color='#ffff00') # Yellow

                # Style the Chart
                ax.set_ylabel('Number of Landslides', fontsize=10)
                ax.set_title('Top 10 Hazard Distribution', fontsize=12, fontweight='bold')
                ax.set_xticks(ind)
                ax.set_xticklabels(munis, rotation=45, ha="right", fontsize=9)
                
                # Hide the top and right borders of the graph for a cleaner look
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Add Legend
                ax.legend((p1[0], p2[0], p3[0]), ('High Hazard', 'Moderate Hazard', 'Low Hazard'), 
                        loc='upper right', frameon=False, fontsize=9)

                # Draw it!
                self.figure_hazard.tight_layout()
                self.canvas_hazard.draw()
                
                self.ui_eil.plainTextEdit_Logs.appendPlainText("--> Dashboard successfully updated!")

            except Exception as e:
                import traceback
                self.ui_eil.plainTextEdit_Logs.appendPlainText(f"\n[Table Error] Failed to generate table:\n{str(e)}")

    # ==================================================================
    # LANDSLIDE LOGIC
    # ==================================================================
    def build_dynamic_landslide_ui(self):
        self.ui_ls.Label_ROI.setText("LULC Shapefile:")
        
        self.ui_ls.GroupBox_ROI_Selection = QtWidgets.QGroupBox(self.ui_ls.Widget_Inputs)
        self.ui_ls.GroupBox_ROI_Selection.setTitle("Region of Interest")
        
        self.ui_ls.layout_roi_select = QtWidgets.QGridLayout(self.ui_ls.GroupBox_ROI_Selection)
        self.ui_ls.layout_roi_select.setContentsMargins(10, 15, 10, 15)
        self.ui_ls.layout_roi_select.setVerticalSpacing(15)
        self.ui_ls.layout_roi_select.setHorizontalSpacing(10)
        self.ui_ls.layout_roi_select.setColumnStretch(1, 1)

        self.ui_ls.Label_Search = QtWidgets.QLabel("Search Location:", self.ui_ls.GroupBox_ROI_Selection)
        self.ui_ls.LineEdit_Search = QtWidgets.QLineEdit(self.ui_ls.GroupBox_ROI_Selection)
        self.ui_ls.LineEdit_Search.setPlaceholderText("Type to search (e.g. Alamada)...")
        
        self.ui_ls.Label_Level = QtWidgets.QLabel("Select Level:", self.ui_ls.GroupBox_ROI_Selection)
        self.ui_ls.ComboBox_ROI_Level = QtWidgets.QComboBox(self.ui_ls.GroupBox_ROI_Selection)
        self.ui_ls.ComboBox_ROI_Level.addItems(["Region", "Province", "Municipality", "Barangay"])
        self.ui_ls.ComboBox_ROI_Level.setCurrentText("Municipality") 
        
        self.ui_ls.Label_Name = QtWidgets.QLabel("Select Name:", self.ui_ls.GroupBox_ROI_Selection)
        self.ui_ls.ComboBox_ROI_Name = QtWidgets.QComboBox(self.ui_ls.GroupBox_ROI_Selection)
        self.ui_ls.ComboBox_ROI_Name.setEditable(False)

        self.disable_cb_scroll(self.ui_ls.ComboBox_ROI_Level)
        self.disable_cb_scroll(self.ui_ls.ComboBox_ROI_Name)

        self.ui_ls.layout_roi_select.addWidget(self.ui_ls.Label_Search, 0, 0)
        self.ui_ls.layout_roi_select.addWidget(self.ui_ls.LineEdit_Search, 0, 1)
        self.ui_ls.layout_roi_select.addWidget(self.ui_ls.Label_Level, 1, 0)
        self.ui_ls.layout_roi_select.addWidget(self.ui_ls.ComboBox_ROI_Level, 1, 1)
        self.ui_ls.layout_roi_select.addWidget(self.ui_ls.Label_Name, 2, 0)
        self.ui_ls.layout_roi_select.addWidget(self.ui_ls.ComboBox_ROI_Name, 2, 1)
        
        self.ui_ls.verticalLayout_4.insertWidget(2, self.ui_ls.GroupBox_ROI_Selection)

    def connect_landslide_signals(self):
        self.ui_ls.PushButton_ROI.clicked.connect(self.select_lulc_file_ls) 
        self.ui_ls.PushButton_DEM.clicked.connect(self.select_dem_file_ls)
        
        self.ui_ls.PushButton_Sentinel_2_Pre_Event.clicked.connect(self.select_pre_event_file_ls)
        self.ui_ls.PushButton_Sentinel_2_Post_Event.clicked.connect(self.select_post_event_file_ls)
        
        self.ui_ls.Button_Detect.clicked.connect(self.run_landslide_detection)
        self.ui_ls.Button_Save.clicked.connect(self.save_landslide_result)

        self.ui_ls.ComboBox_ROI_Level.currentTextChanged.connect(self.populate_location_names_ls)
        self.ui_ls.ComboBox_Method.currentTextChanged.connect(self.toggle_method_inputs_ls)
        
        self.populate_location_names_ls()
        self.toggle_method_inputs_ls(self.ui_ls.ComboBox_Method.currentText()) # Initialize UI states
        
        QtCore.QTimer.singleShot(100, self.build_search_index_ls)

    def toggle_method_inputs_ls(self, method_text):
        """Dynamically hides inputs, changes labels, and perfectly reflows the previews."""
        is_cnn_only = (method_text == "CNN Deep Learning Only")
        is_otsu_only = (method_text == "Otsu's Method Only")
        
        # 1. Update Labels Dynamically
        if is_otsu_only:
            self.ui_ls.Label_Sentinel_2_Pre_Event.setText("Pre-Event (S2):")
            self.ui_ls.Label_Sentinel_2_Post_Event.setText("Post-Event (S2):")
        elif is_cnn_only:
            self.ui_ls.Label_Sentinel_2_Post_Event.setText("Post-Event (S2 12-Band):")
        else: # Ensemble
            self.ui_ls.Label_Sentinel_2_Pre_Event.setText("Pre-Event (S2 12-Band):")
            self.ui_ls.Label_Sentinel_2_Post_Event.setText("Post-Event (S2 12-Band):")

        # 2. Toggle File Inputs Visibility
        self.ui_ls.Label_Sentinel_2_Pre_Event.setVisible(not is_cnn_only)
        self.ui_ls.Widget_Sentinel_2_Pre_Event.setVisible(not is_cnn_only)
        
        # 3. Safely Reflow Previews
        boxes = [
            self.ui_ls.GroupBox_Pre_Event, 
            self.ui_ls.GroupBox_Post_Event, 
            self.ui_ls.GroupBox_DEM_2, 
            self.ui_ls.GroupBox_Slope, 
            self.ui_ls.GroupBox_ROI
        ]
        
        # Detach all preview boxes from grid first
        for box in boxes:
            self.ui_ls.gridLayout_Preview.removeWidget(box)
            
        if is_cnn_only:
            self.ui_ls.GroupBox_Pre_Event.hide()
            # 2x2 Perfect Symmetry for CNN
            self.ui_ls.gridLayout_Preview.addWidget(self.ui_ls.GroupBox_Post_Event, 0, 0)
            self.ui_ls.gridLayout_Preview.addWidget(self.ui_ls.GroupBox_DEM_2, 0, 1)
            self.ui_ls.gridLayout_Preview.addWidget(self.ui_ls.GroupBox_Slope, 1, 0)
            self.ui_ls.gridLayout_Preview.addWidget(self.ui_ls.GroupBox_ROI, 1, 1)
        else:
            self.ui_ls.GroupBox_Pre_Event.show()
            # 5-Box Layout for Otsu/Ensemble (Bottom one spans across)
            self.ui_ls.gridLayout_Preview.addWidget(self.ui_ls.GroupBox_Pre_Event, 0, 0)
            self.ui_ls.gridLayout_Preview.addWidget(self.ui_ls.GroupBox_Post_Event, 0, 1)
            self.ui_ls.gridLayout_Preview.addWidget(self.ui_ls.GroupBox_DEM_2, 1, 0)
            self.ui_ls.gridLayout_Preview.addWidget(self.ui_ls.GroupBox_Slope, 1, 1)
            self.ui_ls.gridLayout_Preview.addWidget(self.ui_ls.GroupBox_ROI, 2, 0, 1, 2)

    def build_search_index_ls(self):
        self.location_index = {}
        all_names = []
        for level, config in self.admin_config.items():
            path = config["path"]
            col = config["col"]
            if os.path.exists(path):
                try:
                    layer = QgsVectorLayer(path, "temp", "ogr")
                    if layer.isValid():
                        idx = layer.fields().indexOf(col)
                        if idx != -1:
                            values = layer.uniqueValues(idx)
                            for val in values:
                                if val:
                                    name = str(val)
                                    if name not in self.location_index:
                                        self.location_index[name] = level
                                        all_names.append(name)
                except: pass
        completer = QtWidgets.QCompleter(all_names, self.ui_ls.LineEdit_Search)
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchContains)
        self.ui_ls.LineEdit_Search.setCompleter(completer)
        completer.activated.connect(self.on_search_selected_ls)

    def on_search_selected_ls(self, text):
        level = self.location_index.get(text)
        if level:
            self.ui_ls.ComboBox_ROI_Level.setCurrentText(level)
            index = self.ui_ls.ComboBox_ROI_Name.findText(text)
            if index != -1:
                self.ui_ls.ComboBox_ROI_Name.setCurrentIndex(index)
            else:
                self.ui_ls.ComboBox_ROI_Name.setCurrentText(text)

    def populate_location_names_ls(self):
        self.ui_ls.ComboBox_ROI_Name.clear()
        selected_level = self.ui_ls.ComboBox_ROI_Level.currentText()
        config = self.admin_config.get(selected_level)
        if not config: return

        shp_path = config["path"]
        col_name = config["col"]
        if not os.path.exists(shp_path):
            self.ui_ls.ComboBox_ROI_Name.addItem(f"Error: missing file")
            return

        layer = QgsVectorLayer(shp_path, "roi_temp", "ogr")
        if not layer.isValid(): return

        try:
            idx = layer.fields().indexOf(col_name)
            unique_values = layer.uniqueValues(idx)
            sorted_names = sorted([str(v) for v in unique_values if v])
            self.ui_ls.ComboBox_ROI_Name.addItems(sorted_names)
        except: pass

    def generate_and_preview_slope(self, dem_path):
        import processing
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            result = processing.run("native:slope", {
                'INPUT': dem_path,
                'Z_FACTOR': 1.0,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })
            slope_temp_path = result['OUTPUT']
            self.preview_layer(slope_temp_path, self.ui_ls.GraphicsView_Slope, is_raster=True)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Preview Error", f"Could not generate Slope preview:\n{str(e)}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def select_lulc_file_ls(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select LULC Shapefile", "", "Shapefiles (*.shp)")
        if filename: 
            self.ui_ls.PTE_ROI.setPlainText(filename)
            self.preview_layer(filename, self.ui_ls.GraphicsView_ROI, is_raster=False)

    def select_dem_file_ls(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select DEM", "", "Tiff (*.tif)")
        if filename: 
            self.ui_ls.PTE_DEM.setPlainText(filename)
            self.preview_layer(filename, self.ui_ls.GraphicsView_DEM_2, is_raster=True)
            self.generate_and_preview_slope(filename)

    def select_pre_event_file_ls(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Pre-Event", "", "Tiff (*.tif)")
        if filename: 
            self.ui_ls.PTE_Sentinel_2_Pre_Event.setPlainText(filename)
            self.preview_layer(filename, self.ui_ls.GraphicsView_Pre_Event, is_raster=True)

    def select_post_event_file_ls(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Post-Event", "", "Tiff (*.tif)")
        if filename: 
            self.ui_ls.PTE_Sentinel_2_Post_Event.setPlainText(filename)
            self.preview_layer(filename, self.ui_ls.GraphicsView_Post_Event, is_raster=True)

    def preview_layer(self, file_path, graphics_view, is_raster=False):
        if is_raster:
            layer = QgsRasterLayer(file_path, "preview")
        else:
            layer = QgsVectorLayer(file_path, "preview", "ogr")
        
        if not layer.isValid(): return

        settings = QgsMapSettings()
        settings.setLayers([layer])
        settings.setBackgroundColor(QColor(255, 255, 255, 0))
        settings.setExtent(layer.extent())
        
        view_size = graphics_view.size()
        if view_size.width() <= 0: view_size = QtCore.QSize(200, 200)
        settings.setOutputSize(view_size)

        render_job = QgsMapRendererParallelJob(settings)
        render_job.start()
        render_job.waitForFinished()
        
        image = render_job.renderedImage()
        scene = QtWidgets.QGraphicsScene()
        item = scene.addPixmap(QPixmap.fromImage(image))
        graphics_view.setScene(scene)
        graphics_view.fitInView(item, Qt.KeepAspectRatio)

    def run_landslide_detection(self):
        lulc = self.ui_ls.PTE_ROI.toPlainText() or "None"
        dem = self.ui_ls.PTE_DEM.toPlainText()
        
        before = self.ui_ls.PTE_Sentinel_2_Pre_Event.toPlainText()
        after = self.ui_ls.PTE_Sentinel_2_Post_Event.toPlainText()
        
        selected_level = self.ui_ls.ComboBox_ROI_Level.currentText()
        selected_name = self.ui_ls.ComboBox_ROI_Name.currentText()
        config = self.admin_config.get(selected_level)
        if not config: return
        
        method_str = self.ui_ls.ComboBox_Method.currentText()
        if "Ensemble" in method_str:
            method_flag = "ensemble"
        elif "Otsu" in method_str:
            method_flag = "otsu"
        else:
            method_flag = "cnn"

        # Safe Validation depending on what the user checked
        if not dem or not after:
            QMessageBox.warning(self, "Missing Inputs", "DEM and Post-Event files are required.")
            return
            
        if method_flag in ["otsu", "ensemble"] and not before:
            QMessageBox.warning(self, "Missing Inputs", "Otsu's and Ensemble methods require a Pre-Event file.")
            return

        roi_file_path = config["path"]
        filter_col = config["col"]

        import tempfile
        temp_output_path = os.path.join(tempfile.gettempdir(), "temp_landslide_result.tif")
        worker_script = os.path.join(self._base_dir, "landslide_worker.py")
        
        # Load models based on your actual path
        model_path = os.path.join(self._base_dir, "model_lsi", "default", "best.keras")
        stats_path = os.path.join(self._base_dir, "model_lsi", "default", "norm_stats.json")

        if not os.path.exists(self.ls_venv_python):
             QMessageBox.critical(self, "Config Error", f"Virtual Environment Python not found at:\n{self.ls_venv_python}")
             return

        command = [
            self.ls_venv_python, worker_script,
            "--method", method_flag,
            "--dem", dem,
            "--after", after,
            "--output", temp_output_path,
            "--roi_path", roi_file_path,
            "--filter_col", filter_col,
            "--filter_val", selected_name,
            "--model_path", model_path, 
            "--stats_path", stats_path
        ]
        
        if lulc and lulc != "None": command.extend(["--lulc", lulc])
        if method_flag in ["otsu", "ensemble"] and before: command.extend(["--before", before])

        my_env = self._get_subprocess_env()

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            process = subprocess.run(command, capture_output=True, text=True, env=my_env)
            if process.returncode == 0:
                lines = process.stdout.strip().split('\n')
                last_line = lines[-1] if lines else ""
                try:
                    result_json = json.loads(last_line)
                    if result_json.get("status") == "success":
                        QMessageBox.information(self, "Success", f"Detection Complete ({method_flag.upper()})! Click SAVE to export.")
                        
                        mask_path = result_json.get("output_mask")
                        self.current_result_path_ls = mask_path 
                        
                        if mask_path and os.path.exists(mask_path):
                             self.preview_layer(mask_path, self.ui_ls.GraphicsView_Result, is_raster=True)
                             rlayer = QgsRasterLayer(mask_path, f"Landslide Result ({method_flag.upper()})")
                             if rlayer.isValid(): QgsProject.instance().addMapLayer(rlayer)
                    else:
                        QMessageBox.critical(self, "Worker Error", f"Script failed: {result_json.get('message')}")
                except json.JSONDecodeError:
                     QMessageBox.warning(self, "Error", f"Could not parse script output.\n\nTerminal Output:\n{process.stdout}")
            else:
                QMessageBox.critical(self, "Error", f"Process failed.\n{process.stderr}")
        except Exception as e:
            QMessageBox.critical(self, "System Error", str(e))
        finally:
             QtWidgets.QApplication.restoreOverrideCursor()

    def save_landslide_result(self):
        if not self.current_result_path_ls or not os.path.exists(self.current_result_path_ls):
            QMessageBox.warning(self, "Save", "No detection result exists yet.\nPlease run 'DETECT' first.")
            return

        roi_name = self.ui_ls.ComboBox_ROI_Name.currentText()
        safe_name = (roi_name or "Landslide_Result").replace(" ", "_")
        method = self.ui_ls.ComboBox_Method.currentText().split(" ")[0]
        default_path = os.path.join(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DocumentsLocation), f"{safe_name}_{method}_Landslide.tif")

        filename, _ = QFileDialog.getSaveFileName(self, "Save Landslide Mask", default_path, "Tiff Files (*.tif)")
        if filename:
            try:
                shutil.copy2(self.current_result_path_ls, filename)
                QMessageBox.information(self, "Saved", f"File successfully saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Could not save file:\n{str(e)}")

    # ==================================================================
    # EIL LOGIC
    # ==================================================================
    def connect_eil_signals(self):
        self.dem_icon_path = os.path.join(self._base_dir, "DEM_icon.svg")
        self.icon_dem = QtGui.QIcon(self.dem_icon_path) if os.path.exists(self.dem_icon_path) else QtGui.QIcon()

        # Connect hover descriptions
        for name in self.feature_info.keys():
            label_obj = getattr(self.ui_eil, f"label_{name}", None)
            if label_obj:
                label_obj.setMouseTracking(True)
                label_obj.installEventFilter(self)

        self.pinn_comboboxes = [
            self.ui_eil.comboBox_BulkDensity, self.ui_eil.comboBox_Clay,
            self.ui_eil.comboBox_Sand, self.ui_eil.comboBox_Silt,
            self.ui_eil.comboBox_Slope, self.ui_eil.comboBox_PGA,
            self.ui_eil.comboBox_PRC, self.ui_eil.comboBox_SoilThickness, 
            self.ui_eil.comboBox_SoilType, self.ui_eil.comboBox_ContributingFactor,
        ]

        self.pinn_buttons = {
            self.ui_eil.pushButton_DEM: self.ui_eil.comboBox_DEM,
            self.ui_eil.pushButton_BulkDensity: self.ui_eil.comboBox_BulkDensity,
            self.ui_eil.pushButton_Clay: self.ui_eil.comboBox_Clay,
            self.ui_eil.pushButton_Sand: self.ui_eil.comboBox_Sand,
            self.ui_eil.pushButton_Silt: self.ui_eil.comboBox_Silt,
            self.ui_eil.pushButton_Slope: self.ui_eil.comboBox_Slope,
            self.ui_eil.pushButton_PGA: self.ui_eil.comboBox_PGA,
            self.ui_eil.pushButton_PRC: self.ui_eil.comboBox_PRC,
            self.ui_eil.pushButton_SoilThickness: self.ui_eil.comboBox_SoilThickness,
            self.ui_eil.pushButton_SoilType: self.ui_eil.comboBox_SoilType,
            self.ui_eil.pushButton_ContributingFactor: self.ui_eil.comboBox_ContributingFactor,
        }

        for btn, box in self.pinn_buttons.items():
            btn.clicked.connect(lambda checked, b=box: self.browse_local_raster(b))

        # Wire up the new single Output Folder selector
        self.ui_eil.pushButton_OutputFolder.clicked.connect(self.select_output_folder)

        self.ui_eil.plainTextEdit_Logs.clear()
        self.ui_eil.plainTextEdit_Details.clear()
        self.ui_eil.progressBar.setValue(0)

        # "Dummy Text" Hack for the single Max Iteration box
        self.ui_eil.plainTextEdit_MaxIteration.setStyleSheet("QPlainTextEdit { color: black; }")
        self.ui_eil.plainTextEdit_MaxIteration.setPlaceholderText("[Default: 10]")
        self.ui_eil.plainTextEdit_MaxIteration.setPlainText("10")
        
        self.disable_cb_scroll(self.ui_eil.comboBox_DEM)
        for box in self.pinn_comboboxes:
            self.disable_cb_scroll(box)

        self.ui_eil.comboBox_DEM.currentTextChanged.connect(lambda t: self.log_layer_selection(t, "DEM"))
        for box in self.pinn_comboboxes:
            box.currentTextChanged.connect(
                lambda text, b=box: self.log_layer_selection(text, b.objectName().replace("comboBox_", ""))
            )

        self.populate_loaded_layers_eil()
        QgsProject.instance().layersAdded.connect(self.populate_loaded_layers_eil)
        QgsProject.instance().layersRemoved.connect(self.populate_loaded_layers_eil)
        
        self.ui_eil.pushButton_DEM.clicked.connect(self.select_dem_file_eil)
        self.populate_pinn_comboboxes()
        
        self.ui_eil.pushButton_Run.clicked.connect(self.run_new_eil_algorithm)
        self.ui_eil.pushButton_Cancel.clicked.connect(self.cancel_eil_algorithm)

    def select_output_folder(self):
        """Opens a file dialog to select the master output directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.ui_eil.plainTextEdit_OutputFolder.setPlainText(folder)
            self.ui_eil.plainTextEdit_Logs.appendPlainText(f"[Output] Results will be saved to: {folder}")

    def populate_pinn_comboboxes(self):
        # Clear them all first
        for box in self.pinn_comboboxes:
            box.blockSignals(True)
            box.clear()
            box.addItem("Select a layer...", None)
            
        # Get all raster layers currently loaded in QGIS
        layers = QgsProject.instance().mapLayers().values()
        raster_layers = [layer for layer in layers if isinstance(layer, QgsRasterLayer)]
        
        # Populate every box with the available rasters
        for box in self.pinn_comboboxes:
            for layer in raster_layers:
                # Add the layer name, and store the layer ID as the hidden data
                box.addItem(layer.name(), layer.id())
            box.blockSignals(False)

    def log_layer_selection(self, text, layer_type):
        if text and "Select a" not in text:
            self.ui_eil.plainTextEdit_Logs.appendPlainText(f"[{layer_type} Selected]: {text}")

    def handle_output_option(self, selection_text, combobox, text_box, log_name):
        if selection_text == "Save to File...":
            filename, _ = QFileDialog.getSaveFileName(self, f"Save {log_name}", "", "GeoPackage (*.gpkg);;Shapefile (*.shp);;Tiff (*.tif)")
            if filename:
                # If they select a file, physically write the path into the box
                text_box.setPlainText(filename)
                self.ui_eil.plainTextEdit_Logs.appendPlainText(f"[Output] {log_name} will be saved to: {os.path.basename(filename)}")
            else:
                # If they cancel the browse window, revert to temporary
                combobox.blockSignals(True)
                combobox.setCurrentIndex(1) 
                combobox.blockSignals(False)
                text_box.clear() # Clear any physical text
                text_box.setPlaceholderText("[Save to temporary file (optional)]")
        elif selection_text == "Skip Output":
            text_box.clear() # Clear physical text
            text_box.setPlaceholderText("[Output will be skipped]")
            self.ui_eil.plainTextEdit_Logs.appendPlainText(f"[Output] {log_name} generation skipped.")
        elif selection_text == "Save to a Temporary File":
            text_box.clear() # Clear physical text
            text_box.setPlaceholderText("[Save to temporary file (optional)]")
            self.ui_eil.plainTextEdit_Logs.appendPlainText(f"[Output] {log_name} will be temporary.")
    # ------------------------------------------------------------------
    # DYNAMIC UI GENERATOR (WITH DELETE BUTTON)
    # ------------------------------------------------------------------
    def add_feature_row(self):
        grid = self.ui_eil.gridLayout
        row = self.feature_row_count
        
        grid.removeWidget(self.ui_eil.pushButton_AddFeature)
        
        new_feat_cb = QtWidgets.QComboBox()
        new_stats_cb = QtWidgets.QComboBox()
        
        # Create a mini container to hold both the Browse and Delete buttons
        btn_container = QtWidgets.QWidget()
        btn_layout = QtWidgets.QHBoxLayout(btn_container)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(4)
        
        new_btn = QtWidgets.QPushButton("...")
        new_del_btn = QtWidgets.QPushButton("X")
        new_del_btn.setStyleSheet("color: #cc0000; font-weight: bold;") # Red X
        new_del_btn.setFixedWidth(25)
        
        btn_layout.addWidget(new_btn)
        btn_layout.addWidget(new_del_btn)
        
        self.disable_cb_scroll(new_feat_cb)
        self.disable_cb_scroll(new_stats_cb)
        
        for stat in self.zonal_stats_options:
            new_stats_cb.addItem(self.icon_stats, stat)
            
        new_btn.clicked.connect(lambda _, cb=new_feat_cb: self.select_dynamic_feature_file(cb))
        new_feat_cb.currentTextChanged.connect(lambda t: self.log_layer_selection(t, f"Feature {len(self.dynamic_feature_rows) + 2}"))
        
        grid.addWidget(new_feat_cb, row, 0)
        grid.addWidget(new_stats_cb, row, 1)
        grid.addWidget(btn_container, row, 2)
        grid.addWidget(self.ui_eil.pushButton_AddFeature, row + 1, 0, 1, 3)
        
        # Save the container so we can destroy it entirely later
        row_data = {'feature_cb': new_feat_cb, 'stats_cb': new_stats_cb, 'btn_container': btn_container}
        self.dynamic_feature_rows.append(row_data)
        self.feature_row_count += 1
        
        # Connect the Delete Button
        new_del_btn.clicked.connect(lambda _, r=row_data: self.remove_feature_row(r))
        
        self.populate_loaded_layers_eil()
        self.ui_eil.plainTextEdit_Logs.appendPlainText(f"--> Added Dynamic Feature Row")

    def remove_feature_row(self, row_data):
        """Destroys the UI elements and removes them from the model's memory."""
        # 1. Destroy the physical UI widgets
        row_data['feature_cb'].deleteLater()
        row_data['stats_cb'].deleteLater()
        row_data['btn_container'].deleteLater()
        
        # 2. Remove them from the tracking list so the algorithm ignores them
        if row_data in self.dynamic_feature_rows:
            self.dynamic_feature_rows.remove(row_data)
            
        self.ui_eil.plainTextEdit_Logs.appendPlainText("--> Removed Feature Row")

    def select_dynamic_feature_file(self, target_combobox):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Feature Raster", "", "Tiff Files (*.tif *.tiff);;All Files (*.*)")
        if filename:
            display_name = os.path.basename(filename)
            target_combobox.addItem(self.icon_feat, display_name, filename)
            target_combobox.setCurrentIndex(target_combobox.count() - 1)

    # ------------------------------------------------------------------
    # QGIS LAYER POPULATION
    # ------------------------------------------------------------------
    def populate_loaded_layers_eil(self):
        # 1. Save the current selections so they don't reset when a new layer is added to QGIS
        dem_state = self.ui_eil.comboBox_DEM.currentData()
        pinn_states = [box.currentData() for box in self.pinn_comboboxes]

        # 2. Grab all valid raster layers currently in the QGIS project
        layers = QgsProject.instance().mapLayers().values()
        raster_layers = [layer for layer in layers if isinstance(layer, QgsRasterLayer)]

        # 3. Update the DEM dropdown
        self.ui_eil.comboBox_DEM.blockSignals(True)
        self.ui_eil.comboBox_DEM.clear()
        self.ui_eil.comboBox_DEM.addItem("Select a DEM layer...", None)
        for layer in raster_layers:
            self.ui_eil.comboBox_DEM.addItem(layer.name(), layer.id())
        
        # Restore the user's previous DEM selection
        idx = self.ui_eil.comboBox_DEM.findData(dem_state)
        if idx != -1:
            self.ui_eil.comboBox_DEM.setCurrentIndex(idx)
        self.ui_eil.comboBox_DEM.blockSignals(False)

        # 4. Update all PINN parameter dropdowns
        for i, box in enumerate(self.pinn_comboboxes):
            box.blockSignals(True)
            box.clear()
            box.addItem("Select a layer...", None)
            
            for layer in raster_layers:
                box.addItem(layer.name(), layer.id())
                
            # Restore the user's previous selection for this specific box
            idx = box.findData(pinn_states[i])
            if idx != -1:
                box.setCurrentIndex(idx)
            box.blockSignals(False)

    def select_dem_file_eil(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select DEM Raster", "", "Tiff Files (*.tif *.tiff);;All Files (*.*)")
        if filename:
            display_name = os.path.basename(filename)
            self.ui_eil.comboBox_DEM.addItem(self.icon_dem, display_name, filename)
            self.ui_eil.comboBox_DEM.setCurrentIndex(self.ui_eil.comboBox_DEM.count() - 1)

    def cancel_eil_algorithm(self):
        self.ui_eil.plainTextEdit_Logs.appendPlainText("\n--> Algorithm Cancelled by User.")
        self.ui_eil.progressBar.setValue(0)

    # ------------------------------------------------------------------
    # RUN SCRIPT (UPDATED WITH GRASS MAC WORKER)
    # ------------------------------------------------------------------
    def run_new_eil_algorithm(self):
        self.ui_eil.plainTextEdit_Logs.appendPlainText("\n--- STARTING FULL EIL PIPELINE ---")
        self.ui_eil.progressBar.setValue(0)
        
        # 1. Grab and validate the DEM
        dem_id = self.ui_eil.comboBox_DEM.currentData()
        if not dem_id:
            QMessageBox.warning(self, "Error", "Please select a DEM.")
            self.ui_eil.plainTextEdit_Logs.appendPlainText("[ERROR] No DEM selected.")
            return
            
        dem_path = ""
        self.dem_crs_authid = "EPSG:4326" 
        
        if os.path.isabs(str(dem_id)):
            dem_path = str(dem_id)
            rlayer = QgsRasterLayer(dem_path, "temp_dem")
            if rlayer.isValid():
                self.dem_crs_authid = rlayer.crs().authid()
        else:
            layer = QgsProject.instance().mapLayer(dem_id)
            if layer:
                dem_path = layer.source().split("|")[0] 
                self.dem_crs_authid = layer.crs().authid()
                
        if not dem_path or not os.path.exists(dem_path):
            self.ui_eil.plainTextEdit_Logs.appendPlainText(f"[ERROR] Could not locate DEM file:\n{dem_path}")
            return
        
        self.dem_path_for_taudem = dem_path
            
        # 2. Gather rasters from the UI
        selected_rasters = self.get_selected_rasters_from_ui() 
        if not selected_rasters:
            self.ui_eil.plainTextEdit_Logs.appendPlainText("No parameters selected. Pipeline finished early.")
            return

        import tempfile, shutil
        self.temp_raster_dir = tempfile.mkdtemp(prefix="go_rasters_")
        
        for param_name, tif_path in selected_rasters.items():
            symlink_path = os.path.join(self.temp_raster_dir, f"{param_name}.tif")
            try:
                os.symlink(tif_path, symlink_path)
            except OSError:
                shutil.copy2(tif_path, symlink_path)

        # 3. Handle Unified Output Folder
        user_out = self.ui_eil.plainTextEdit_OutputFolder.toPlainText().strip()
        if user_out and os.path.isdir(user_out):
            final_output_folder = user_out
        else:
            # Fallback to DEM folder if user left it blank
            final_output_folder = os.path.join(os.path.dirname(dem_path), "EIL_Results")
            self.ui_eil.plainTextEdit_OutputFolder.setPlainText(final_output_folder)
        os.makedirs(final_output_folder, exist_ok=True)

        # 4. Enforce Hardcoded Default Parameters
        def safe_int(text, default):
            try: return int(text)
            except: return default

        params = {
            'areaMin': 5000,
            'thresh': 5000,
            'cvMin': 0.15,
            'rf': 10,
            'maxIter': safe_int(self.ui_eil.plainTextEdit_MaxIteration.toPlainText().strip(), 10),
            'tileCols': 3, 
            'tileRows': 3, 
            'tileOverlap': 500
        }
        
        self.ui_eil.plainTextEdit_Logs.appendPlainText(f"[Input DEM]: {os.path.basename(dem_path)}")
        self.ui_eil.plainTextEdit_Logs.appendPlainText(f"[Slope Params]: {params}")
        self.ui_eil.pushButton_Run.setEnabled(False)
        
        # 5. Start the Go Worker
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        go_repo_dir = os.path.join(plugin_dir, "ml-prep-scripts")
        
        self.go_worker = GoParallelZonalWorker(
            go_repo_dir, dem_path, self.temp_raster_dir, final_output_folder,
            self.eil_venv_python, self.dem_crs_authid, len(selected_rasters), params
        )
        
        self.go_worker.progress.connect(self.ui_eil.progressBar.setValue)
        self.go_worker.log.connect(self.ui_eil.plainTextEdit_Logs.appendPlainText)
        self.go_worker.error.connect(self.on_slope_units_error) 
        self.go_worker.finished.connect(self.on_pipeline_complete)
        self.go_worker.start()


    def get_selected_rasters_from_ui(self):
        selected_rasters = {} 
        
        def extract_path(combobox):
            layer_id = combobox.currentData()
            if not layer_id:
                return None
            if os.path.isabs(str(layer_id)):
                return str(layer_id)
            else:
                from qgis.core import QgsProject
                layer = QgsProject.instance().mapLayer(layer_id)
                if layer and layer.source():
                    return layer.source().split("|")[0]
            return None

        # 1. Manually add the DEM using the exact prefix the Keras model expects
        dem_path = extract_path(self.ui_eil.comboBox_DEM)
        if dem_path and os.path.exists(dem_path):
            selected_rasters["Elev"] = dem_path # Go will automatically output 'Elev_mean'

        # 2. Map the UI comboboxes directly to the Keras model's expected base names
        strict_mapping = {
            "comboBox_Slope": "Slope",                           # -> Slope_mean
            "comboBox_BulkDensity": "BUK",                       # -> BUK_mean
            "comboBox_Clay": "Clay",                             # -> Clay_mean
            "comboBox_Sand": "Sand",                             # -> Sand_mean
            "comboBox_Silt": "Silt",                             # -> Silt_mean
            "comboBox_PRC": "Prc",                               # -> Prc_mean
            "comboBox_SoilThickness": "SoilThc",                 # -> SoilThc_mean
            "comboBox_ContributingFactor": "ContributingFactor", # -> ContributingFactor_mean
            "comboBox_PGA": "PGA2",                              # -> PGA2_mean 
            "comboBox_SoilType": "type"                          # -> type
        }

        # Loop through your UI elements and enforce the naming convention
        for box in self.pinn_comboboxes:
            path = extract_path(box)
            if path and os.path.exists(path) and path.lower().endswith(('.tif', '.tiff')):
                box_name = box.objectName()
                if box_name in strict_mapping:
                    # Force the alias to be the strict standard name
                    standard_name = strict_mapping[box_name]
                    selected_rasters[standard_name] = path
                    
        return selected_rasters
    
    def _eliminate_seams(self, input_gpkg, output_gpkg, area_min, log_label="Layer"):
        """Merges tile-boundary sliver polygons into their largest neighbor,
        erasing seams left by the parallel Go tiling."""
        import processing
        from qgis.core import QgsVectorLayer

        try:
            src_layer = QgsVectorLayer(input_gpkg, "eliminate_src", "ogr")
            if not src_layer.isValid():
                self.ui_eil.plainTextEdit_Logs.appendPlainText(
                    f"[Warning] {log_label}: could not open {input_gpkg} for seam cleanup."
                )
                return input_gpkg

            src_layer.selectByExpression(f'$area < {area_min}')
            n_selected = src_layer.selectedFeatureCount()

            if n_selected == 0:
                self.ui_eil.plainTextEdit_Logs.appendPlainText(
                    f"--> {log_label}: no slivers under {area_min} found, skipping cleanup."
                )
                return input_gpkg

            processing.run("qgis:eliminateselectedpolygons", {
                'INPUT': src_layer,   # pass the layer object itself — selection lives on it
                'MODE': 2,            # merge with neighbor sharing the largest boundary
                'OUTPUT': output_gpkg
            })

            if os.path.exists(output_gpkg):
                self.ui_eil.plainTextEdit_Logs.appendPlainText(
                    f"--> {log_label}: {n_selected} slivers merged, seams erased."
                )
                return output_gpkg

        except Exception as e:
            self.ui_eil.plainTextEdit_Logs.appendPlainText(
                f"[Warning] {log_label} seam cleanup failed, using raw geometry: {e}"
            )
        return input_gpkg

    def on_pipeline_complete(self, final_output_folder, status):
        if status == "success":
            self.ui_eil.plainTextEdit_Logs.appendPlainText(f"\n[Data Prep Complete] All features merged!")
            self.ui_eil.progressBar.setValue(90)

            area_min = 5000 # Hardcoded to match params

            # --- Clean Slope Units ---
            raw_su_file = os.path.join(final_output_folder, "Final_SlopeUnits.gpkg")
            final_su_file = raw_su_file
            if os.path.exists(raw_su_file):
                self.ui_eil.plainTextEdit_Logs.appendPlainText("--> Erasing tile seams on Slope Units...")
                cleaned_su_file = os.path.join(final_output_folder, "Cleaned_SlopeUnits.gpkg")
                final_su_file = self._eliminate_seams(raw_su_file, cleaned_su_file, area_min, "Slope Units")

            if os.path.exists(final_su_file) and self.ui_eil.checkBox_LoadResults.isChecked():
                self.finish_and_load_layer(final_su_file, "EIL Slope Units")

            # --- Clean Merged PINN Features ---
            merged_file = os.path.join(final_output_folder, "Merged_PINN_Features.gpkg")
            if os.path.exists(merged_file):
                cleaned_file = os.path.join(final_output_folder, "Cleaned_PINN_Features.gpkg")
                merged_file = self._eliminate_seams(merged_file, cleaned_file, area_min, "PINN Features")

            if os.path.exists(merged_file):
                plugin_dir = os.path.dirname(os.path.abspath(__file__))
                pinn_script = os.path.join(plugin_dir, "pinn_inference.py")
                
                if not os.path.exists(self.MODEL_PATH):
                    self.ui_eil.plainTextEdit_Logs.appendPlainText(f"\n[WARNING] AI Model not found at: {self.MODEL_PATH}")
                    if self.ui_eil.checkBox_LoadResults.isChecked():
                        self.finish_and_load_layer(merged_file, "Merged PINN Features")
                    return

                # Force final map to save directly into the chosen output folder
                final_ai_map = os.path.join(final_output_folder, "Final_Hazard_Map.gpkg")
                
                self.pinn_worker = PINNPredictionWorker(
                    self.eil_venv_python, pinn_script, merged_file, self.MODEL_PATH, final_ai_map, self.dem_path_for_taudem
                )
                self.pinn_worker.log.connect(self.ui_eil.plainTextEdit_Logs.appendPlainText)
                self.pinn_worker.finished.connect(self.on_pinn_complete)
                self.pinn_worker.start()
            else:
                self.ui_eil.pushButton_Run.setEnabled(True)
                
        if hasattr(self, 'temp_raster_dir') and os.path.exists(self.temp_raster_dir):
            import shutil
            shutil.rmtree(self.temp_raster_dir)

    def on_pinn_complete(self, result_json, status):
        self.ui_eil.pushButton_Run.setEnabled(True)
        self.ui_eil.progressBar.setValue(100)

        if status == "success":
            self.ui_eil.plainTextEdit_Logs.appendPlainText(f"\n✅ AI Predictions generated successfully!")

            if self.ui_eil.checkBox_LoadResults.isChecked():
                self.finish_and_load_layer(result_json.get("output"), "EIL AI Predictions")

            self.ui_eil.plainTextEdit_Logs.appendPlainText(f"\n🎉 ENTIRE PIPELINE COMPLETE!")
            self.update_municipality_table(result_json.get("output"))

            overall_tif = result_json.get("overall_tif")

            # ── ML-DFI Step 4 Routing ──────────────────────────────────────────────────
            if overall_tif and os.path.exists(overall_tif):
                step4_out = os.path.join(self.ui_eil.plainTextEdit_OutputFolder.toPlainText().strip(), "deposition_zone_output")
                
                step4_wrapper = os.path.join(
                    self._base_dir, "ml_dfi_handoff", "steps", 
                    "step4_deposition_zone", "step4_deposition_zone_mpi_wrapper.py"
                )

                if not os.path.exists(step4_wrapper):
                    self.ui_eil.plainTextEdit_Logs.appendPlainText(f"[Warning] Step 4 Wrapper not found. Skipping deposition routing.")
                    return

                self.ui_eil.plainTextEdit_Logs.appendPlainText("\n--- INITIATING DYNAMIC-ALPHA DEPOSITION ROUTING ---")
                
                self.deposition_worker = MLDFIDepositionWorker(
                    dem_path=self.dem_path_for_taudem, 
                    dfi_tif=overall_tif,
                    output_dir=step4_out,
                    step4_wrapper_py=step4_wrapper,
                    cores=4
                )
                self.deposition_worker.log.connect(self.ui_eil.plainTextEdit_Logs.appendPlainText)
                self.deposition_worker.progress.connect(self.ui_eil.progressBar.setValue)
                self.deposition_worker.finished.connect(self.on_deposition_complete)
                self.deposition_worker.error.connect(self.on_slope_units_error)
                self.deposition_worker.start()

    def on_deposition_complete(self, depositional_mask_path, status):
        if status == "success" and os.path.exists(depositional_mask_path):
            self.ui_eil.plainTextEdit_Logs.appendPlainText("✅ Step 4 Depositional Routing Complete!")

            out_dir = self.ui_eil.plainTextEdit_OutputFolder.toPlainText().strip()
            step4_out = os.path.join(out_dir, "deposition_zone_output")
            
            # ── COPY FINAL RASTERS DIRECTLY TO ROOT OUTPUT FOLDER ──────────────
            main_dep_tif = os.path.join(out_dir, "ML_DFI_Depositional_Zone.tif")
            main_runout_tif = os.path.join(out_dir, "ML_DFI_Full_Runout_Track.tif")
            
            import shutil
            shutil.copy2(depositional_mask_path, main_dep_tif)
            
            step4_runout_src = os.path.join(step4_out, "step4_runout_mask.tif")
            if os.path.exists(step4_runout_src):
                shutil.copy2(step4_runout_src, main_runout_tif)

            if self.ui_eil.checkBox_LoadResults.isChecked():
                rlayer = QgsRasterLayer(main_dep_tif, "ML-DFI Depositional Zone")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)

            # ── GENERATE SAFE HYDROLOGY RASTERS FOR STEPS 5 & 6 ─────────────────────
            out_dir = self.ui_eil.plainTextEdit_OutputFolder.toPlainText().strip()
            step4_out = os.path.join(out_dir, "deposition_zone_output")
            
            # Grab the permanent outputs generated by Step 4
            dem_fel = os.path.join(step4_out, "fel.tif")
            source_tif = os.path.join(step4_out, "source.tif")
            dfi_tif = os.path.join(step4_out, "dfi_clean.tif")
            
            # Step 5 needs the full runout mask. If step4_runout_mask.tif exists, use it. Otherwise use the deposit mask.
            runout_mask = os.path.join(step4_out, "step4_runout_mask.tif")
            if not os.path.exists(runout_mask):
                runout_mask = depositional_mask_path

            stream_tif = os.path.join(step4_out, "stream_mask.tif")
            swca_tif = os.path.join(step4_out, "swca.tif")

            try:
                import rasterio
                import numpy as np
                with rasterio.open(source_tif) as src:
                    meta = src.meta.copy()
                    shape = src.shape
                    
                    # Generate safe Stream Mask (All zeros = safe fallback)
                    meta.update(dtype=rasterio.uint8, nodata=0)
                    with rasterio.open(stream_tif, 'w', **meta) as dst:
                        dst.write(np.zeros(shape, dtype=np.uint8), 1)
                        
                    # Generate safe Catchment Area (All 4400s = safe max spread radius fallback)
                    meta.update(dtype=rasterio.float32, nodata=-9999.0)
                    with rasterio.open(swca_tif, 'w', **meta) as dst:
                        dst.write(np.full(shape, 4400.0, dtype=np.float32), 1)
            except Exception as e:
                self.ui_eil.plainTextEdit_Logs.appendPlainText(f"[Warning] Failed to generate fallback hydrologic rasters: {e}")
                return

            # ── TRIGGER STEP 5: POST-DEPOSITIONAL SPREAD (PDS) ──────────────────
            step5_script = os.path.join(self._base_dir, "ml_dfi_handoff", "steps", "step5_post_depositional_spread_PDS", "step5_post_depositional_spread_PDS.py")
            
            if not os.path.exists(step5_script):
                self.ui_eil.plainTextEdit_Logs.appendPlainText(f"[Warning] Step 5 Script not found at {step5_script}. Pipeline complete.")
                self.ui_eil.pushButton_Run.setEnabled(True)
                return

            step5_out_dir = os.path.join(out_dir, "step5_pds_output")
            os.makedirs(step5_out_dir, exist_ok=True)

            step5_config = {
                "runout_mask_raster": runout_mask,
                "dfi_raster": dfi_tif,
                "dem_raster": dem_fel,
                "source_mask_raster": source_tif,
                "stream_mask_raster": stream_tif,
                "source_contributing_area_raster": swca_tif,
                "output_spread_mask": os.path.join(step5_out_dir, "post_depositional_spread_mask.tif"),
                "output_combined_mask": os.path.join(step5_out_dir, "post_depositional_combined_mask.tif"),
                "output_seed_mask": os.path.join(step5_out_dir, "post_depositional_seed_mask.tif"),
                "output_cleaned_runout_mask": os.path.join(step5_out_dir, "post_depositional_cleaned_runout_mask.tif"),
                "output_removed_runout_mask": os.path.join(step5_out_dir, "post_depositional_removed_runout_mask.tif"),
                "output_summary_json": os.path.join(step5_out_dir, "post_depositional_spread_summary.json"),
                "dfi_threshold": 0.69,
                "max_relative_rise_m": 1.0,
                "min_runout_source_area_m2": 200.0,
                "min_spread_radius_m": 5.0,
                "max_spread_radius_m": 30.0,
                "source_contributing_area_reference": 4400.0,
                "write_debug_rasters": False
            }

            self.step5_worker = Step5SpreadWorker(step5_script, step5_config, self.eil_venv_python)
            self.step5_worker.log.connect(self.ui_eil.plainTextEdit_Logs.appendPlainText)
            self.step5_worker.progress.connect(self.ui_eil.progressBar.setValue)
            self.step5_worker.finished.connect(self.on_step5_complete)
            self.step5_worker.error.connect(self.on_slope_units_error)
            self.step5_worker.start()

    def on_step5_complete(self, combined_mask_path, status):
        if status == "success" and os.path.exists(combined_mask_path):
            self.ui_eil.plainTextEdit_Logs.appendPlainText("✅ Step 5 Post-Depositional Spread Complete!")
            if self.ui_eil.checkBox_LoadResults.isChecked():
                rlayer = QgsRasterLayer(combined_mask_path, "PDS Combined Runout Footprint")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)

            # ── TRIGGER STEP 6: LANDSLIDE DAMMING POTENTIAL (LDP) ────────────────
            step6_script = os.path.join(self._base_dir, "ml_dfi_handoff", "steps", "step6_landslide_damming_potential_LDP", "step6_landslide_damming_potential_PDS.py")
            
            if not os.path.exists(step6_script):
                self.ui_eil.plainTextEdit_Logs.appendPlainText(f"[Warning] Step 6 Script not found at {step6_script}. Pipeline complete.")
                self.ui_eil.pushButton_Run.setEnabled(True)
                return

            out_dir = self.ui_eil.plainTextEdit_OutputFolder.toPlainText().strip()
            step4_out = os.path.join(out_dir, "deposition_zone_output")
            step6_out_dir = os.path.join(out_dir, "step6_ldp_output")
            os.makedirs(step6_out_dir, exist_ok=True)

            step6_config = {
                "dem_fel_raster": os.path.join(step4_out, "fel.tif"),
                "dinf_flow_raster": os.path.join(step4_out, "ang.tif"),
                "runout_mask_raster": combined_mask_path,
                "stream_raster": os.path.join(step4_out, "stream_mask.tif"),
                "slope_raster": os.path.join(step4_out, "slp.tif"),
                "output_stream_intersection_mask": os.path.join(step6_out_dir, "step6_runout_stream_intersection_mask.tif"),
                "output_inflow_angle": os.path.join(step6_out_dir, "step6_runout_inflow_angle_deg.tif"),
                "output_stream_angle": os.path.join(step6_out_dir, "step6_runout_stream_angle_deg.tif"),
                "output_runout_approach_angle": os.path.join(step6_out_dir, "step6_mean_runout_approach_angle_25m_deg.tif"),
                "output_channel_slope": os.path.join(step6_out_dir, "step6_mean_channel_slope_25m_deg.tif"),
                "output_candidate_damming_potential_class": os.path.join(step6_out_dir, "step6_candidate_damming_potential_class.tif"),
                "output_damming_potential_class": os.path.join(step6_out_dir, "step6_buffered_damming_potential_class.tif"),
                "output_damming_potential_score": os.path.join(step6_out_dir, "step6_runout_damming_potential_score.tif"),
                "output_summary_json": os.path.join(step6_out_dir, "step6_summary.json")
            }

            self.step6_worker = Step6DammingWorker(step6_script, step6_config, self.eil_venv_python)
            self.step6_worker.log.connect(self.ui_eil.plainTextEdit_Logs.appendPlainText)
            self.step6_worker.progress.connect(self.ui_eil.progressBar.setValue)
            self.step6_worker.finished.connect(self.on_step6_complete)
            self.step6_worker.error.connect(self.on_slope_units_error)
            self.step6_worker.start()

    def on_step6_complete(self, damming_class_path, status):
        self.ui_eil.pushButton_Run.setEnabled(True)
        if status == "success" and os.path.exists(damming_class_path):
            self.ui_eil.plainTextEdit_Logs.appendPlainText("\n🎉 ENTIRE PIPELINE (STEPS 1 THROUGH 6) FULLY COMPLETE!")
            if self.ui_eil.checkBox_LoadResults.isChecked():
                rlayer = QgsRasterLayer(damming_class_path, "LDP Landslide Damming Potential")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                    self.ui_eil.plainTextEdit_Logs.appendPlainText("--> Loaded into map: LDP Landslide Damming Potential")


    def finish_and_load_layer(self, layer_path, layer_name):
        if not layer_path or not os.path.exists(layer_path):
            self.ui_eil.plainTextEdit_Logs.appendPlainText(f"[Warning] {layer_name} path not found: {layer_path}")
            return
        vlayer = QgsVectorLayer(layer_path, layer_name, "ogr")
        if vlayer.isValid():
            QgsProject.instance().addMapLayer(vlayer)
            self.ui_eil.plainTextEdit_Logs.appendPlainText(f"--> Loaded into map: {layer_name}")

    def on_slope_units_error(self, err_msg):
        self.ui_eil.pushButton_Run.setEnabled(True)
        self.ui_eil.plainTextEdit_Logs.appendPlainText(f"\n[CRITICAL ERROR] {err_msg}")
        self.ui_eil.progressBar.setValue(0)