# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(990, 950)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("QGroupBox::title {\n"
"    background-color: transparent;\n"
"    padding-left: 5px;\n"
"    padding-right: 5px;\n"
"}\n"
"QPlainTextEdit { border-radius: 4px; border: 1px solid #b3b3b3; }\n"
"QLabel, QCheckBox { background-color: transparent; }\n"
"QComboBox {\n"
"    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e6e6e6);\n"
"    border: 1px solid #b3b3b3; border-radius: 4px; padding: 3px 10px; color: black; min-height: 22px;\n"
"}\n"
"QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 25px; border-left: 1px solid #cccccc; border-top-right-radius: 4px; border-bottom-right-radius: 4px; }\n"
"QComboBox::down-arrow { image: url(:/down-arrow.png); width: 12px; height: 12px; }\n"
"QComboBox::down-arrow:on { top: 1px; left: 1px; }\n"
"QPushButton {\n"
"    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e6e6e6);\n"
"    border: 1px solid #b3b3b3; border-radius: 4px; padding: 3px 10px; color: black; min-height: 22px;\n"
"}\n"
"QPushButton:pressed { background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e6e6e6, stop:1 #d4d4d4); }\n"
"QProgressBar { border: 1px solid #b3b3b3; border-radius: 4px; background-color: #f0f0f0; text-align: center; color: black; min-height: 18px; max-height: 18px; }\n"
"QProgressBar::chunk { background-color: #007aff; border-radius: 3px; }")
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        self.scrollAreaLeft = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollAreaLeft.setStyleSheet("background-color:\"white\";\ncolor:\"black\";")
        self.scrollAreaLeft.setWidgetResizable(True)
        self.scrollAreaLeft.setObjectName("scrollAreaLeft")
        self.scrollAreaWidgetContentsLeft = QtWidgets.QWidget()
        self.scrollAreaWidgetContentsLeft.setGeometry(QtCore.QRect(0, 0, 705, 1100))
        self.scrollAreaWidgetContentsLeft.setObjectName("scrollAreaWidgetContentsLeft")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContentsLeft)
        self.verticalLayout.setObjectName("verticalLayout")
        
        # ==========================================
        # 🟢 PARAMETER GRID
        # ==========================================
        self.groupBox_Parameters = QtWidgets.QGroupBox(self.scrollAreaWidgetContentsLeft)
        self.groupBox_Parameters.setObjectName("groupBox_Parameters")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_Parameters)
        self.gridLayout.setObjectName("gridLayout")
        
        sizePolicyCombo = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicyCombo.setHorizontalStretch(0)
        sizePolicyCombo.setVerticalStretch(0)

        # DEM ROW
        self.label_DEM = QtWidgets.QLabel(self.groupBox_Parameters)
        self.label_DEM.setObjectName("label_DEM")
        self.gridLayout.addWidget(self.label_DEM, 0, 0, 1, 1)

        self.comboBox_DEM = QtWidgets.QComboBox(self.groupBox_Parameters)
        self.comboBox_DEM.setSizePolicy(sizePolicyCombo)
        self.comboBox_DEM.setObjectName("comboBox_DEM")
        self.gridLayout.addWidget(self.comboBox_DEM, 0, 1, 1, 1)

        self.pushButton_DEM = QtWidgets.QPushButton(self.groupBox_Parameters)
        self.pushButton_DEM.setMaximumWidth(40)
        self.pushButton_DEM.setObjectName("pushButton_DEM")
        self.gridLayout.addWidget(self.pushButton_DEM, 0, 2, 1, 1)

        parameters = [
            ("BulkDensity", "Bulk Density"), ("Clay", "Clay"), ("Sand", "Sand"), ("Silt", "Silt"),
            ("Slope", "Slope"), 
            ("PGA", "Peak Ground Acceleration (PGA)"),
            ("PRC", "Precipitation"), 
            ("SoilThickness", "Soil Thickness"), 
            ("SoilType", "Soil Type / Lithology"),
            ("ContributingFactor", "Contributing Factor"),
        ]

        for idx, (name, title) in enumerate(parameters):
            row = idx + 1
            label = QtWidgets.QLabel(self.groupBox_Parameters)
            label.setObjectName(f"label_{name}")
            self.gridLayout.addWidget(label, row, 0, 1, 1)
            setattr(self, f"label_{name}", label)
            
            combo = QtWidgets.QComboBox(self.groupBox_Parameters)
            combo.setSizePolicy(sizePolicyCombo)
            combo.setObjectName(f"comboBox_{name}")
            self.gridLayout.addWidget(combo, row, 1, 1, 1)
            setattr(self, f"comboBox_{name}", combo)
            
            btn = QtWidgets.QPushButton(self.groupBox_Parameters)
            btn.setMaximumWidth(40)
            btn.setObjectName(f"pushButton_{name}")
            self.gridLayout.addWidget(btn, row, 2, 1, 1)
            setattr(self, f"pushButton_{name}", btn)

        self.gridLayout.setColumnStretch(1, 5)
        self.verticalLayout.addWidget(self.groupBox_Parameters)

        # ==========================================
        # 🟢 UPDATED: SLOPE UNIT (Max Iteration Only)
        # ==========================================
        self.groupBox_SlopeUnit = QtWidgets.QGroupBox(self.scrollAreaWidgetContentsLeft)
        self.groupBox_SlopeUnit.setStyleSheet("QLabel { background-color: transparent; }")
        self.groupBox_SlopeUnit.setObjectName("groupBox_SlopeUnit")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_SlopeUnit)
        self.gridLayout_2.setContentsMargins(-1, 10, -1, 12)
        
        self.label_MaxIteration = QtWidgets.QLabel(self.groupBox_SlopeUnit)
        self.label_MaxIteration.setObjectName("label_MaxIteration")
        self.gridLayout_2.addWidget(self.label_MaxIteration, 0, 0, 1, 1)
        
        self.plainTextEdit_MaxIteration = QtWidgets.QPlainTextEdit(self.groupBox_SlopeUnit)
        self.plainTextEdit_MaxIteration.setMaximumSize(QtCore.QSize(16777215, 30))
        self.plainTextEdit_MaxIteration.setObjectName("plainTextEdit_MaxIteration")
        self.gridLayout_2.addWidget(self.plainTextEdit_MaxIteration, 0, 1, 1, 1)

        self.verticalLayout.addWidget(self.groupBox_SlopeUnit)

        # ==========================================
        # 🟢 UPDATED: SINGLE OUTPUT OPTIONS
        # ==========================================
        self.groupBox_Option = QtWidgets.QGroupBox(self.scrollAreaWidgetContentsLeft)
        self.groupBox_Option.setStyleSheet("""
            QLabel { background-color: transparent; }
            QCheckBox { background-color: transparent; }
            QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #b3b3b3; border-radius: 3px; background-color: white; }
            QCheckBox::indicator:hover { border: 1px solid #007aff; }
            QCheckBox::indicator:checked { background-color: #007aff; border: 1px solid #007aff; }
        """)
        self.groupBox_Option.setObjectName("groupBox_Option")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_Option)
        self.gridLayout_5.setVerticalSpacing(12)

        self.label_OutputFolder = QtWidgets.QLabel(self.groupBox_Option)
        self.label_OutputFolder.setObjectName("label_OutputFolder")
        self.gridLayout_5.addWidget(self.label_OutputFolder, 0, 0, 1, 2)

        self.plainTextEdit_OutputFolder = QtWidgets.QPlainTextEdit(self.groupBox_Option)
        self.plainTextEdit_OutputFolder.setMaximumSize(QtCore.QSize(16777215, 30))
        self.plainTextEdit_OutputFolder.setObjectName("plainTextEdit_OutputFolder")
        self.gridLayout_5.addWidget(self.plainTextEdit_OutputFolder, 1, 0, 1, 1)

        self.pushButton_OutputFolder = QtWidgets.QPushButton(self.groupBox_Option)
        self.pushButton_OutputFolder.setMaximumWidth(40)
        self.pushButton_OutputFolder.setObjectName("pushButton_OutputFolder")
        self.gridLayout_5.addWidget(self.pushButton_OutputFolder, 1, 1, 1, 1)

        self.checkBox_LoadResults = QtWidgets.QCheckBox(self.groupBox_Option)
        self.checkBox_LoadResults.setObjectName("checkBox_LoadResults")
        self.checkBox_LoadResults.setChecked(True) # Checked by default
        self.gridLayout_5.addWidget(self.checkBox_LoadResults, 2, 0, 1, 2)

        self.verticalLayout.addWidget(self.groupBox_Option)
        
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.scrollAreaLeft.setWidget(self.scrollAreaWidgetContentsLeft)
        self.horizontalLayout.addWidget(self.scrollAreaLeft)
        
        # ==========================================
        # MIDDLE COLUMN (Table & Chart)
        # ==========================================
        self.scrollAreaMiddle = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollAreaMiddle.setStyleSheet("background-color: white; color: black;")
        self.scrollAreaMiddle.setWidgetResizable(True)
        self.scrollAreaMiddle.setObjectName("scrollAreaMiddle")
        
        self.scrollAreaWidgetContentsMiddle = QtWidgets.QWidget()
        self.scrollAreaWidgetContentsMiddle.setObjectName("scrollAreaWidgetContentsMiddle")
        
        self.verticalLayout_Middle = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContentsMiddle)
        self.verticalLayout_Middle.setObjectName("verticalLayout_Middle")
        
        box_style = "QFrame { background-color: white; border: 1px solid #b3b3b3; border-radius: 4px; }"
        
        self.frame_Middle1 = QtWidgets.QFrame(self.scrollAreaWidgetContentsMiddle)
        self.frame_Middle1.setObjectName("frame_Middle1")
        self.frame_Middle1.setStyleSheet(box_style)
        self.layout_Middle1 = QtWidgets.QVBoxLayout(self.frame_Middle1)
        
        self.label_TableTitle = QtWidgets.QLabel("Top 10 Hazardous Municipalities", self.frame_Middle1)
        self.label_TableTitle.setStyleSheet("font-weight: bold; font-size: 14px; color: black; margin-bottom: 5px;")
        self.layout_Middle1.addWidget(self.label_TableTitle)
        
        self.tableWidget_Hazard = QtWidgets.QTableWidget(self.frame_Middle1)
        self.tableWidget_Hazard.setColumnCount(4)
        self.tableWidget_Hazard.verticalHeader().setVisible(False)
        
        headers = ["Municipality", "High", "Moderate", "Low"]
        bg_colors = ["#000000", "#000000", "#000000", "#000000"]
        text_colors = ["#ffffff", "#ffffff", "#ffffff", "#ffffff"]

        for i in range(4):
            item = QtWidgets.QTableWidgetItem(headers[i])
            item.setBackground(QtGui.QColor(bg_colors[i]))
            item.setForeground(QtGui.QColor(text_colors[i]))
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            self.tableWidget_Hazard.setHorizontalHeaderItem(i, item)

        self.tableWidget_Hazard.setStyleSheet("""
            QTableWidget { background-color: white; color: black; gridline-color: #cccccc; }
            QHeaderView::section { border: 1px solid #cccccc; padding: 4px; }
        """)
        self.tableWidget_Hazard.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tableWidget_Hazard.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        self.layout_Middle1.addWidget(self.tableWidget_Hazard)
        self.verticalLayout_Middle.addWidget(self.frame_Middle1)
        
        self.frame_Middle2 = QtWidgets.QFrame(self.scrollAreaWidgetContentsMiddle)
        self.frame_Middle2.setObjectName("frame_Middle2")
        self.frame_Middle2.setStyleSheet(box_style)
        self.layout_Middle2 = QtWidgets.QVBoxLayout(self.frame_Middle2)
        self.verticalLayout_Middle.addWidget(self.frame_Middle2)
        
        self.verticalLayout_Middle.setStretch(0, 1)
        self.verticalLayout_Middle.setStretch(1, 1)
        self.scrollAreaMiddle.setWidget(self.scrollAreaWidgetContentsMiddle)
        self.horizontalLayout.addWidget(self.scrollAreaMiddle)
        
        # ==========================================
        # RIGHT COLUMN (Logs & Details)
        # ==========================================
        self.scrollAreaRight = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollAreaRight.setStyleSheet("background-color:\"white\";\ncolor:\"black\";")
        self.scrollAreaRight.setWidgetResizable(True)
        self.scrollAreaRight.setObjectName("scrollAreaRight")
        
        self.scrollAreaWidgetContentsRight = QtWidgets.QWidget()
        self.scrollAreaWidgetContentsRight.setGeometry(QtCore.QRect(0, 0, 239, 621))
        self.scrollAreaWidgetContentsRight.setObjectName("scrollAreaWidgetContentsRight")
        
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContentsRight)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        
        self.plainTextEdit_Details = QtWidgets.QPlainTextEdit(self.scrollAreaWidgetContentsRight)
        self.plainTextEdit_Details.setObjectName("plainTextEdit_Details")
        self.verticalLayout_2.addWidget(self.plainTextEdit_Details)
        
        self.plainTextEdit_Logs = QtWidgets.QPlainTextEdit(self.scrollAreaWidgetContentsRight)
        self.plainTextEdit_Logs.setObjectName("plainTextEdit_Logs")
        self.verticalLayout_2.addWidget(self.plainTextEdit_Logs)
        
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 1)
        self.scrollAreaRight.setWidget(self.scrollAreaWidgetContentsRight)
        
        self.horizontalLayout.addWidget(self.scrollAreaRight)
        
        self.horizontalLayout.setStretch(0, 3) 
        self.horizontalLayout.setStretch(1, 3) 
        self.horizontalLayout.setStretch(2, 2) 
        
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        
        self.widgetBottom = QtWidgets.QWidget(self.centralwidget)
        self.widgetBottom.setStyleSheet("QWidget { background-color: white; }")
        self.widgetBottom.setObjectName("widgetBottom")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.widgetBottom)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.pushButton_Cancel = QtWidgets.QPushButton(self.widgetBottom)
        self.pushButton_Cancel.setObjectName("pushButton_Cancel")
        self.gridLayout_3.addWidget(self.pushButton_Cancel, 0, 3, 1, 1)
        self.pushButton_Run = QtWidgets.QPushButton(self.widgetBottom)
        self.pushButton_Run.setObjectName("pushButton_Run")
        self.gridLayout_3.addWidget(self.pushButton_Run, 0, 2, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.widgetBottom)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout_3.addWidget(self.progressBar, 0, 0, 1, 2)
        self.gridLayout_3.setColumnStretch(0, 8)
        self.verticalLayout_3.addWidget(self.widgetBottom)
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 990, 33))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MLPREP Detection Suite"))
        self.groupBox_Parameters.setTitle(_translate("MainWindow", "Parameters"))
        
        self.label_DEM.setText(_translate("MainWindow", "DEM"))
        self.pushButton_DEM.setText(_translate("MainWindow", "..."))
        
        parameters = [
            ("BulkDensity", "Bulk Density"), ("Clay", "Clay"), ("Sand", "Sand"), ("Silt", "Silt"),
            ("Slope", "Slope"), 
            ("PGA", "Peak Ground Acceleration (PGA)"),
            ("PRC", "Precipitation"), 
            ("SoilThickness", "Soil Thickness"), 
            ("SoilType", "Soil Type / Lithology"),
            ("ContributingFactor", "Contributing Factor"),
        ]
        
        for name, title in parameters:
            getattr(self, f"label_{name}").setText(_translate("MainWindow", title))
            getattr(self, f"pushButton_{name}").setText(_translate("MainWindow", "..."))

        self.groupBox_SlopeUnit.setTitle(_translate("MainWindow", "Slope Unit Tuning"))
        self.label_MaxIteration.setText(_translate("MainWindow", "Max Iteration"))
        
        self.groupBox_Option.setTitle(_translate("MainWindow", "Output Setup"))
        self.label_OutputFolder.setText(_translate("MainWindow", "Save Output Files To:"))
        self.plainTextEdit_OutputFolder.setPlaceholderText(_translate("MainWindow", "[Defaults to DEM folder if left blank]"))
        self.pushButton_OutputFolder.setText(_translate("MainWindow", "..."))
        self.checkBox_LoadResults.setText(_translate("MainWindow", "Automatically load results into map"))
        
        self.pushButton_Cancel.setText(_translate("MainWindow", "Cancel"))
        self.pushButton_Run.setText(_translate("MainWindow", "Run Pipeline"))