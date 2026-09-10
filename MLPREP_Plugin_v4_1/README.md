# MLPREP Detection Suite V4.1

A unified QGIS plugin for **Landslide Detection** and **Earthquake-Induced Landslide (EIL) Hazard Mapping** using machine learning and deep learning techniques.

## Overview

The MLPREP Detection Suite merges two powerful detection workflows into a single, integrated toolbar interface within QGIS. This plugin leverages advanced machine learning models to:

- **Detect landslides** from geospatial data
- **Map Earthquake-Induced Landslide (EIL) hazards** using Physics-Informed Neural Networks (PINNs)

Both capabilities are accessible through a unified graphical interface, enabling seamless hazard assessment for disaster risk reduction.

## Features

### 1. Landslide Detection
- Automated landslide identification from raster and vector data
- Integration with geospatial processing workflows
- Support for multiple input data formats

### 2. Earthquake-Induced Landslide (EIL) Detection
- PINN-based inference for earthquake-triggered landslide prediction
- Hazard mapping capabilities
- Physics-informed predictions for improved accuracy

### 3. Unified Interface
- Single toolbar for both detection methods
- Streamlined workflow management
- Integrated result visualization

## System Requirements

### QGIS Version
- **Minimum QGIS version**: 3.0

### Python Dependencies

The plugin requires two sets of dependencies depending on your use case:

#### For EIL Detection (`requirements_eil.txt`):
- tensorflow
- numpy
- pandas
- geopandas
- scikit-learn
- keras-tuner
- imbalanced-learn
- matplotlib
- seaborn
- contextily
- scikeras

#### For Landslide Detection (`requirements_ls.txt`):
- numpy
- rasterio
- geopandas
- shapely
- tensorflow

## Installation

### 1. Extract Plugin
Extract the plugin folder to your QGIS plugins directory:
- **Linux/Mac**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
- **Windows**: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`

### 2. Install Dependencies
Install the required Python packages using pip:

```bash
# For EIL detection
pip install -r requirements_eil.txt

# For Landslide detection
pip install -r requirements_ls.txt
```

### 3. Enable Plugin
1. Open QGIS
2. Go to **Plugins → Manage and Install Plugins**
3. Search for "MLPREP Detection V4.1"
4. Click **Install Plugin**

## Usage

### Accessing the Plugin

1. **Open QGIS** with a project containing your geospatial data
2. Click the **MLPREP Detection Suite** icon in the toolbar (or access via **Plugins → MLPREP Suite**)
3. The unified detection dialog will open in a maximized window

### Workflow

#### Landslide Detection
1. Load or prepare your raster/vector data in QGIS
2. Access the **Landslide Detection** tab in the plugin
3. Configure detection parameters
4. Run detection
5. Review results on the map

#### EIL Hazard Mapping
1. Prepare earthquake event data and terrain data
2. Access the **EIL Detection** tab
3. Configure PINN model parameters
4. Run inference
5. Visualize hazard predictions

## Plugin Architecture

### Core Files

| File | Purpose |
|------|---------|
| `mainPlugin.py` | Main plugin entry point and QGIS integration |
| `merged_wrapper.py` | Unified dialog interface combining both detection modules |
| `landslide_mainwindow.py` | Landslide detection UI |
| `landslide_wrapper.py` | Landslide detection processing logic |
| `landslide_worker.py` | Background worker for landslide processing |
| `eil_mainwindow.py` | EIL detection UI |
| `eil_wrapper.py` | EIL detection processing logic |
| `pinn_inference.py` | Physics-Informed Neural Network inference engine |
| `headless_model.py` | Headless execution support for batch processing |
| `metadata.txt` | QGIS plugin metadata |
| `__init__.py` | Package initialization and library path management |

### Supporting Directories

- **`model/`** - Pre-trained landslide detection models
- **`model_lsi/`** - Pre-trained EIL detection models
- **`ml-prep-scripts/`** - Utility scripts for data preparation
- **`ml_dfi_handoff/`** - DFI integration handoff files
- **`py_files/`** - Additional Python utilities

## Configuration

### Metadata Configuration (`metadata.txt`)

```ini
[general]
name=MLPREP Detection V4.1
qgisMinimumVersion=3.0
description=A unified suite for Landslide Detection and Earthquake-Induced Landslide (EIL) Hazard mapping.
version=1.0
category=Analysis
```

## Known Limitations

- TensorFlow and deep learning models require significant computational resources
- Large-scale processing may require adequate system memory
- Some features may require GPU acceleration for optimal performance

## Development

### Environment Setup

1. Clone the repository
2. Install QGIS development tools
3. Install Python dependencies for both detection methods
4. Test the plugin in a QGIS development environment

### Testing

Run the plugin in QGIS's Python console to debug and test individual components:

```python
from MLPREP_Plugin_v4_1.mainPlugin import MainPlugin
```

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- Dependencies are documented
- New features include relevant documentation updates

## License

[Specify your license here]

## Authors

- **Plugin Developer**: Your Name
- **Email**: your.email@example.com

## Support & Issues

For issues, feature requests, or questions:
1. Check existing documentation
2. Review the plugin's issue tracker
3. Contact the development team

## Changelog

### Version 1.0 (V4.1)
- Initial release
- Unified Landslide Detection and EIL Hazard Mapping
- Physics-Informed Neural Network integration
- Combined toolbar interface
- Support for QGIS 3.0+

## References

- [QGIS Plugin Development](https://docs.qgis.org/latest/en/docs/pyqgis_developer_guide/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Rasterio Documentation](https://rasterio.readthedocs.io/)
- [GeoPandas Documentation](https://geopandas.org/)

---

**Last Updated**: 2026

**Current Version**: 1.0 (V4.1)
