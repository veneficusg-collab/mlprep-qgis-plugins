import os
import subprocess
import json
import platform
import shutil
import tempfile 

from qgis.core import (
    QgsVectorLayer, 
    QgsRasterLayer, 
    QgsProject,
    QgsMapSettings,
    QgsMapRendererParallelJob
)
from qgis.PyQt import QtWidgets, QtCore
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QPixmap

# Import the new Unified UI class
from .landslide_mainwindow import Ui_MainWindow

class PluginDialog(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(PluginDialog, self).__init__(parent)
        self.setupUi(self)
        
        # --- CONFIGURATION ---
        self.venv_python = self.get_venv_python_path()
        self.current_result_path = None 
        
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        roi_dir = os.path.join(plugin_dir, "roi_data")
        
        self.admin_config = {
            "Region": {"path": os.path.join(roi_dir, "phl_admbnda_adm1_psa_namria_20231106.shp"), "col": "ADM1_EN"},
            "Province": {"path": os.path.join(roi_dir, "phl_admbnda_adm2_psa_namria_20231106.shp"), "col": "ADM2_EN"},
            "Municipality": {"path": os.path.join(roi_dir, "phl_admbnda_adm3_psa_namria_20231106.shp"), "col": "ADM3_EN"},
            "Barangay": {"path": os.path.join(roi_dir, "phl_admbnda_adm4_psa_namria_20231106.shp"), "col": "ADM4_EN"}
        }

        self.location_index = {} 
        self.build_dynamic_ui()
        self.connect_signals()

    def build_dynamic_ui(self):
        """Sets up the dynamic UI elements like search bars and specific form behavior."""
        # Setup ROI Selection Group layout injected dynamically from wrapper
        self.GroupBox_ROI_Selection = QtWidgets.QGroupBox(self.Widget_Inputs)
        self.GroupBox_ROI_Selection.setTitle("Region of Interest")
        self.layout_roi_select = QtWidgets.QGridLayout(self.GroupBox_ROI_Selection)
        self.layout_roi_select.setContentsMargins(10, 15, 10, 15)
        self.layout_roi_select.setVerticalSpacing(15)
        
        self.Label_Search = QtWidgets.QLabel("Search Location:")
        self.LineEdit_Search = QtWidgets.QLineEdit()
        self.LineEdit_Search.setPlaceholderText("Type to search (e.g. Alamada)...")
        self.LineEdit_Search.setStyleSheet("border: 1px solid #3a3a3a; border-radius: 6px; padding: 4px 10px; background: #ffffff; color: black; min-height: 25px;")
        
        self.Label_Level = QtWidgets.QLabel("Select Level:")
        self.ComboBox_ROI_Level = QtWidgets.QComboBox()
        self.ComboBox_ROI_Level.addItems(["Region", "Province", "Municipality", "Barangay"])
        self.ComboBox_ROI_Level.setCurrentText("Municipality") 
        self.ComboBox_ROI_Level.setStyleSheet("border: 1px solid #3a3a3a; border-radius: 6px; padding: 4px 10px; background: #D5D5D5; color: black; min-height: 25px;")
        
        self.Label_Name = QtWidgets.QLabel("Select Name:")
        self.ComboBox_ROI_Name = QtWidgets.QComboBox()
        self.ComboBox_ROI_Name.setStyleSheet("border: 1px solid #3a3a3a; border-radius: 6px; padding: 4px 10px; background: #D5D5D5; color: black; min-height: 25px;")
        
        self.layout_roi_select.addWidget(self.Label_Search, 0, 0)
        self.layout_roi_select.addWidget(self.LineEdit_Search, 0, 1)
        self.layout_roi_select.addWidget(self.Label_Level, 1, 0)
        self.layout_roi_select.addWidget(self.ComboBox_ROI_Level, 1, 1)
        self.layout_roi_select.addWidget(self.Label_Name, 2, 0)
        self.layout_roi_select.addWidget(self.ComboBox_ROI_Name, 2, 1)
        
        # Insert it right below the Method Selection
        self.verticalLayout_4.insertWidget(2, self.GroupBox_ROI_Selection)
        QtCore.QTimer.singleShot(100, self.build_search_index)

    def connect_signals(self):
        self.PushButton_ROI.clicked.connect(self.select_lulc_file) 
        self.PushButton_DEM.clicked.connect(self.select_dem_file)
        self.PushButton_Sentinel_2_Pre_Event.clicked.connect(self.select_pre_event_file)
        self.PushButton_Sentinel_2_Post_Event.clicked.connect(self.select_post_event_file)
        self.Button_Detect.clicked.connect(self.run_detection)
        self.Button_Save.clicked.connect(self.save_result_to_disk)

        self.ComboBox_ROI_Level.currentTextChanged.connect(self.populate_location_names)
        self.ComboBox_Method.currentTextChanged.connect(self.toggle_method_inputs)
        
        self.populate_location_names()
        self.toggle_method_inputs(self.ComboBox_Method.currentText())

    def toggle_method_inputs(self, method_text):
        """Disables the Pre-Event field if CNN is exclusively selected, as it doesn't use it."""
        is_cnn_only = "CNN" in method_text and "Otsu" not in method_text
        
        self.Widget_Sentinel_2_Pre_Event.setEnabled(not is_cnn_only)
        self.Label_Sentinel_2_Pre_Event.setEnabled(not is_cnn_only)
        
        if is_cnn_only:
            self.PTE_Sentinel_2_Pre_Event.setPlaceholderText("[Not required for CNN Deep Learning]")
        else:
            self.PTE_Sentinel_2_Pre_Event.setPlaceholderText("")

    def get_venv_python_path(self):
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        venv_name = "ls_detect_env"
        if platform.system() == "Windows":
            return os.path.join(plugin_dir, venv_name, "Scripts", "python.exe")
        return os.path.join(plugin_dir, venv_name, "bin", "python")

    def build_search_index(self):
        self.location_index = {}
        all_names = []
        for level, config in self.admin_config.items():
            if os.path.exists(config["path"]):
                try:
                    layer = QgsVectorLayer(config["path"], "temp", "ogr")
                    if layer.isValid():
                        idx = layer.fields().indexOf(config["col"])
                        if idx != -1:
                            for val in layer.uniqueValues(idx):
                                if val:
                                    name = str(val)
                                    self.location_index[name] = level
                                    all_names.append(name)
                except:
                    pass
        
        completer = QtWidgets.QCompleter(all_names, self.LineEdit_Search)
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchContains)
        self.LineEdit_Search.setCompleter(completer)
        completer.activated.connect(self.on_search_selected)

    def on_search_selected(self, text):
        level = self.location_index.get(text)
        if level:
            self.ComboBox_ROI_Level.setCurrentText(level)
            idx = self.ComboBox_ROI_Name.findText(text)
            self.ComboBox_ROI_Name.setCurrentIndex(idx) if idx != -1 else self.ComboBox_ROI_Name.setCurrentText(text)

    def populate_location_names(self):
        self.ComboBox_ROI_Name.clear()
        config = self.admin_config.get(self.ComboBox_ROI_Level.currentText())
        if not config or not os.path.exists(config["path"]): return
        
        layer = QgsVectorLayer(config["path"], "temp", "ogr")
        if layer.isValid():
            try:
                idx = layer.fields().indexOf(config["col"])
                self.ComboBox_ROI_Name.addItems(sorted([str(v) for v in layer.uniqueValues(idx) if v]))
            except: pass

    # --- FILE SELECTION ---
    def select_lulc_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select LULC", "", "Shapefiles (*.shp)")
        if filename: 
            self.PTE_ROI.setPlainText(filename)
            self.preview_layer(filename, self.GraphicsView_ROI, is_raster=False)

    def select_dem_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select DEM", "", "Tiff (*.tif)")
        if filename: 
            self.PTE_DEM.setPlainText(filename)
            self.preview_layer(filename, self.GraphicsView_DEM_2, is_raster=True)
            self.generate_and_preview_slope(filename)

    def select_pre_event_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Pre-Event", "", "Tiff (*.tif)")
        if filename: 
            self.PTE_Sentinel_2_Pre_Event.setPlainText(filename)
            self.preview_layer(filename, self.GraphicsView_Pre_Event, is_raster=True)

    def select_post_event_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Post-Event", "", "Tiff (*.tif)")
        if filename: 
            self.PTE_Sentinel_2_Post_Event.setPlainText(filename)
            self.preview_layer(filename, self.GraphicsView_Post_Event, is_raster=True)

    def generate_and_preview_slope(self, dem_path):
        import processing
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            result = processing.run("native:slope", {'INPUT': dem_path, 'Z_FACTOR': 1.0, 'OUTPUT': 'TEMPORARY_OUTPUT'})
            self.preview_layer(result['OUTPUT'], self.GraphicsView_Slope, is_raster=True)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Preview Error", f"Could not generate Slope preview:\n{str(e)}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def preview_layer(self, file_path, graphics_view, is_raster=False):
        layer = QgsRasterLayer(file_path, "preview") if is_raster else QgsVectorLayer(file_path, "preview", "ogr")
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
        
        scene = QtWidgets.QGraphicsScene()
        item = scene.addPixmap(QPixmap.fromImage(render_job.renderedImage()))
        graphics_view.setScene(scene)
        graphics_view.fitInView(item, Qt.KeepAspectRatio)

    # --- DETECTION LOGIC ---
    def run_detection(self):
        lulc = self.PTE_ROI.toPlainText() or "None"
        dem = self.PTE_DEM.toPlainText()
        before = self.PTE_Sentinel_2_Pre_Event.toPlainText()
        after = self.PTE_Sentinel_2_Post_Event.toPlainText()
        
        config = self.admin_config.get(self.ComboBox_ROI_Level.currentText())
        if not config: return
        
        method_str = self.ComboBox_Method.currentText()
        if "Ensemble" in method_str:
            method_flag = "ensemble"
        elif "Otsu" in method_str:
            method_flag = "otsu"
        else:
            method_flag = "cnn"

        # Validation
        if method_flag in ["otsu", "ensemble"] and not before:
            QtWidgets.QMessageBox.warning(self, "Missing Inputs", "Otsu/Ensemble methods require a Pre-Event file.")
            return
        if not dem or not after:
            QtWidgets.QMessageBox.warning(self, "Missing Inputs", "DEM and Post-Event files are required.")
            return

        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        temp_output_path = os.path.join(tempfile.gettempdir(), "temp_landslide_result.tif")
        worker_script = os.path.join(plugin_dir, "landslide_worker.py")
        model_path = os.path.join(plugin_dir, "model_lsi", "default", "best.keras")
        stats_path = os.path.join(plugin_dir, "model_lsi", "default", "norm_stats.json")

        command = [
            self.venv_python, worker_script,
            "--method", method_flag,
            "--dem", dem,
            "--after", after,
            "--output", temp_output_path,
            "--roi_path", config["path"],
            "--filter_col", config["col"],
            "--filter_val", self.ComboBox_ROI_Name.currentText()
        ]
        
        if before: command.extend(["--before", before])
        if lulc and lulc != "None": command.extend(["--lulc", lulc])
        if method_flag in ["cnn", "ensemble"]:
            command.extend(["--model_path", model_path, "--stats_path", stats_path])

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            # Drop QGIS-specific Python paths so external TF/Scikit environments don't conflict
            my_env = os.environ.copy()
            my_env.pop("PYTHONHOME", None)
            my_env.pop("PYTHONPATH", None)

            process = subprocess.run(command, capture_output=True, text=True, env=my_env)
            
            if process.returncode == 0:
                lines = process.stdout.strip().split('\n')
                try:
                    result_json = json.loads(lines[-1])
                    if result_json.get("status") == "success":
                        QtWidgets.QMessageBox.information(self, "Success", f"Detection Complete ({method_flag.upper()})! Click SAVE to export.")
                        self.current_result_path = result_json.get("output_mask")
                        
                        if os.path.exists(self.current_result_path):
                             self.preview_layer(self.current_result_path, self.GraphicsView_Result, is_raster=True)
                             rlayer = QgsRasterLayer(self.current_result_path, f"Landslide Result ({method_flag.upper()})")
                             if rlayer.isValid(): QgsProject.instance().addMapLayer(rlayer)
                    else:
                        QtWidgets.QMessageBox.critical(self, "Worker Error", f"Script failed:\n{result_json.get('message')}")
                except json.JSONDecodeError:
                     QtWidgets.QMessageBox.warning(self, "Error", f"Could not parse script output.\nTerminal:\n{process.stdout}")
            else:
                QtWidgets.QMessageBox.critical(self, "Error", f"Process failed.\n{process.stderr}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "System Error", str(e))
        finally:
             QtWidgets.QApplication.restoreOverrideCursor()

    # --- SAVE BUTTON LOGIC ---
    def save_result_to_disk(self):
        if not self.current_result_path or not os.path.exists(self.current_result_path):
            QtWidgets.QMessageBox.warning(self, "Save", "No detection result exists yet.\nPlease run 'DETECT' first.")
            return

        safe_name = (self.ComboBox_ROI_Name.currentText() or "Result").replace(" ", "_")
        method = self.ComboBox_Method.currentText().split(" ")[0]
        default_path = os.path.join(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DocumentsLocation), f"{safe_name}_{method}_Landslide.tif")

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Landslide Mask", default_path, "Tiff Files (*.tif)")
        if filename:
            try:
                shutil.copy2(self.current_result_path, filename)
                QtWidgets.QMessageBox.information(self, "Saved", f"File successfully saved to:\n{filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Save Error", f"Could not save file:\n{str(e)}")