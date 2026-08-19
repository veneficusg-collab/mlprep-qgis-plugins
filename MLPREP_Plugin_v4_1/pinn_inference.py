import sys
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import tensorflow as tf

import rasterio
from rasterio.features import rasterize as rio_rasterize

# Ensure the current directory is in sys.path to access py_files
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from py_files.helpers import add_soil_texture_index
    from py_files.data import (
        preprocessing_v2,
        apply_log_transform,
        apply_clip_thresholds,
        dataframe_to_dataset,
    )
    from py_files.GallenModel import CriticalAcceleration, DisplacementIntermediate, FosLayer
    from py_files.GallenModel_v1 import (
        NewmarkActivation, DisplacementLayerRainFall, WetnessLayer,
        ClipLayer, CohesionLayer, InternalFrictionLayer, LogitLayer
    )
    from py_files.GallenModel_v3 import HydraulicConductivityLayerV3
    from py_files.Landslidev2_Old import DiceCrossEntropyLoss
except ImportError as e:
    print(json.dumps({"status": "error", "message": f"Import Error: {e}"}))
    sys.exit(1)


# ---------------------------------------------------------
# LogitLayer definition to fix the deserialization error
# ---------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="Custom")
class LogitLayer(tf.keras.layers.Layer):
    """Maps probabilities [0, 1] to logits [-inf, inf] using log(p / (1 - p))."""
    def __init__(self, eps=1e-6, **kwargs):
        super(LogitLayer, self).__init__(**kwargs)
        self.eps = eps

    def call(self, inputs):
        # Clip to avoid log(0) or division by zero
        p = tf.clip_by_value(inputs, self.eps, 1.0 - self.eps)
        return tf.math.log(p / (1.0 - p))

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Columns to drop during preprocessing if they exist
COLUMNS_DROP = [
    "Landslide1", "descriptio", "sus_pinn_ground truth", "ds",
    "cohesion", "internal_friction", "sus_pinn_landslide",
    "confusion", "landslide_preds", "landslide_probability",
    "Lithology", "LITHO", "Geomorphology", "LITHODESC",
    "LITHO_2", "LITHODESC_2", "value",
]

def run_prediction():
    try:
        # 1. Parse Arguments from the QGIS Wrapper
        raw_gpkg_path = sys.argv[1]
        clean_gpkg_path = raw_gpkg_path.split('|')[0]
        model_path = sys.argv[2]
        output_gpkg = sys.argv[3]
        dem_template_path = sys.argv[4]
        
        # Locate the manifest json
        manifest_path = os.path.join(current_dir, "feature_manifests", "v1_cotabato_transforms_production.json")
        if not os.path.exists(manifest_path):
            manifest_path = os.path.join(os.path.dirname(model_path), "v1_cotabato_transforms_production.json")
            
        base_dir = os.path.dirname(clean_gpkg_path)
        
        # Determine the Zonal stats GPKG path (from Go pipeline)
        if base_dir.endswith("Zonal_Results"):
            zonal_path = os.path.join(base_dir, "Merged_PINN_Features.gpkg")
        else:
            zonal_path = os.path.join(base_dir, "Zonal_Results", "Merged_PINN_Features.gpkg")
        if not os.path.exists(zonal_path):
            zonal_path = os.path.join(base_dir, "Merged_PINN_Features.gpkg")

        # 2. Setup Custom Objects and Load Model
        custom_objects = {
            "NewmarkActivation": NewmarkActivation,
            "DisplacementLayerRainFall": DisplacementLayerRainFall,
            "WetnessLayer": WetnessLayer,
            "ClipLayer": ClipLayer,
            "CohesionLayer": CohesionLayer,
            "InternalFrictionLayer": InternalFrictionLayer,
            "CriticalAcceleration": CriticalAcceleration,
            "DisplacementIntermediate": DisplacementIntermediate,
            "FosLayer": FosLayer,
            "HydraulicConductivityLayerV3": HydraulicConductivityLayerV3,
            "DiceCrossEntropyLoss": DiceCrossEntropyLoss,
            "Custom>LogitLayer": LogitLayer,
        }
        
        for name, cls in custom_objects.items():
            tf.keras.utils.get_custom_objects()[name] = cls
            tf.keras.utils.get_custom_objects()[f"Custom>{name}"] = cls

        model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        input_cols = [inp.name.split(':')[0] for inp in model.inputs]
        
        # 3. Load GPKG
        gdf_raw = gpd.read_file(zonal_path)
        
        # Safely rename Go Engine outputs to Keras expected inputs
        rename_dict = {}
        for col in gdf_raw.columns:
            cl = col.lower()
            if 'slope' in cl: rename_dict[col] = 'Slope_mean'
            elif 'clay' in cl: rename_dict[col] = 'Clay_mean'
            elif 'sand' in cl: rename_dict[col] = 'Sand_mean'
            elif 'silt' in cl: rename_dict[col] = 'Silt_mean'
            elif 'elev' in cl: rename_dict[col] = 'Elev_mean'
            elif 'soilth' in cl: rename_dict[col] = 'SoilThc_mean'
            elif 'prc' in cl: rename_dict[col] = 'Prc_mean'
            elif 'pga' in cl: rename_dict[col] = 'PGA2_max'
            elif 'contributing' in cl: rename_dict[col] = 'ContributingFactor_mean'
            elif 'buk' in cl or 'bulk' in cl: rename_dict[col] = 'BUK_mean'
            elif 'type' in cl: rename_dict[col] = 'type'
            
        gdf_raw.rename(columns=rename_dict, inplace=True)

        # ---------------------------------------------------------
        # FIX 1: Bulk Density Unit Conversion
        # If values are > 50, they are likely in cg/cm^3. 
        # Multiply by 0.0981 to convert to kN/m^3 for the physics engine.
        # ---------------------------------------------------------
        if 'BUK_mean' in gdf_raw.columns:
            gdf_raw['BUK_mean'] = pd.to_numeric(gdf_raw['BUK_mean'], errors='coerce')
            gdf_raw.loc[gdf_raw['BUK_mean'] > 50, 'BUK_mean'] = gdf_raw['BUK_mean'] * 0.0981

        # ---------------------------------------------------------
        # FIX 2: Soil Type / Lithology Catcher
        # If the raster outputs numerical codes (like 45.0) instead of strings,
        # override it with "Unknown" so the network relies on the Sand/Silt/Clay percentages instead.
        # ---------------------------------------------------------
        if 'type' in gdf_raw.columns:
            gdf_raw['type'] = gdf_raw['type'].astype(str)
            # If the string is just a number (e.g., "45.0"), mark it unknown
            gdf_raw.loc[gdf_raw['type'].str.replace('.', '', 1).str.isnumeric(), 'type'] = "Unknown"
        
        # Absolute safety net for the preprocessor
        if 'Slope_mean' not in gdf_raw.columns:
            gdf_raw['Slope_mean'] = 15.0
        if 'type' not in gdf_raw.columns:
            gdf_raw['type'] = "Unknown"
            
        # Safely run preprocessing_v2
        valid_drop = [c for c in COLUMNS_DROP if c in gdf_raw.columns]
        df, columns, numeric_cols, _, _ = preprocessing_v2(
            gdf_raw, columns_drop=valid_drop, track_imputation=True
        )
        
        # Preserve Geometry for the final GPKG output
        geometry = df.geometry.copy()
        crs = df.crs
        
        # Add indexing and metadata
        try:
            df = add_soil_texture_index(df[columns].copy())
        except Exception as e:
            print(f"[WARN] add_soil_texture_index failed: {e}", flush=True)

        # 4. Replay Production Manifest Transforms (Safe Fallback)
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                transform_meta = json.load(f)
            
            log_cols = transform_meta.get("log_transformed_cols", [])
            clip_thresh = transform_meta.get("clip_thresholds", {})
            
            if log_cols:
                df = apply_log_transform(df, log_cols)
            if clip_thresh:
                df = apply_clip_thresholds(df, clip_thresh)
        else:
            print(f"[WARN] Manifest not found at {manifest_path}. Skipping data scaling.", flush=True)

        # ---------------------------------------------------------
        # NEW: Explicit Type-Casting to Prevent TensorFlow Crashes
        # ---------------------------------------------------------
        for exp_col in input_cols:
            if exp_col not in df.columns:
                print(f"[WARN] Imputing missing required input: {exp_col}", flush=True)
                if 'type' in exp_col.lower() or exp_col == 'LITHO':
                    df[exp_col] = "Unknown"
                elif 'idx' in exp_col.lower():
                    df[exp_col] = 4 
                else:
                    df[exp_col] = 0.001
                    
            # Force Data Types
            if 'type' in exp_col.lower() or exp_col == 'LITHO':
                df[exp_col] = df[exp_col].astype(str)
            elif 'idx' in exp_col.lower():
                df[exp_col] = pd.to_numeric(df[exp_col], errors='coerce').fillna(4).astype('int32')
            else:
                df[exp_col] = pd.to_numeric(df[exp_col], errors='coerce').fillna(0.001).astype('float32')
            
        # 5. Predict
        # Inject dummy target column for compatibility with dataframe_to_dataset
        df["landslide"] = 0 
        ds = dataframe_to_dataset(
            df[input_cols + ["landslide"]].copy(), shuffle=False, batch_size=128
        )
        
        out = model.predict(ds, verbose=0)
        susceptibility = out["final_head"].flatten() if isinstance(out, dict) else out.flatten()

        # 6. Extract Intermediate Physics
        physics_extractor = tf.keras.Model(
            inputs=model.inputs,
            outputs={
                "fos": model.get_layer("fos_layer").output,
                "displacement": model.get_layer("displacement_layer").output,
                "cohesion": model.get_layer("cohesion_layer").output,
                "internal_friction": model.get_layer("internal_friction").output,
            },
        )
        phys = physics_extractor.predict(ds, verbose=0)

        # 7. Assemble Output GeoDataFrame
        out_gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs)
        
        # The QGIS plugin expects exactly 'Hazard_Susceptibility' instead of 'susceptibility'
        out_gdf["Hazard_Susceptibility"] = susceptibility
        
        # Reattach parameters
        out_gdf["FactorOfSafety"] = np.asarray(phys["fos"]).reshape(len(out_gdf), -1)[:, 0]
        out_gdf["Displacement"] = np.asarray(phys["displacement"]).reshape(len(out_gdf), -1)[:, 0]
        out_gdf["Cohesion"] = np.asarray(phys["cohesion"]).reshape(len(out_gdf), -1)[:, 0]
        out_gdf["Internal_Friction"] = np.asarray(phys["internal_friction"]).reshape(len(out_gdf), -1)[:, 0]

        # Save to Output GPKG
        main_output_path = output_gpkg
        if os.path.exists(main_output_path): 
            os.remove(main_output_path)
        
        out_gdf.to_file(main_output_path, driver="GPKG")

        # 8. OVERALL OUTPUT: Rasterize Susceptibility for TauDEM integration
        overall_dir = os.path.join(base_dir, "overall_output")
        os.makedirs(overall_dir, exist_ok=True)
        overall_tif = os.path.join(overall_dir, "Overall_Output.tif")

        with rasterio.open(dem_template_path) as src:
            dem_data = src.read(1)
            dem_nodata = src.nodata
            transform = src.transform
            cols = src.width
            rows = src.height

        # Identify valid DEM terrain outside of the bounding box nodata
        valid_dem_mask = np.isfinite(dem_data)
        if dem_nodata is not None:
            valid_dem_mask &= ~np.isclose(dem_data, dem_nodata)

        shapes = (
            (geom, value)
            for geom, value in zip(out_gdf.geometry, out_gdf["Hazard_Susceptibility"])
            if geom is not None and not geom.is_empty
        )

        # Burn slope units to raster. Background fill is 0.0 to allow routing!
        burned = rio_rasterize(
            shapes,
            out_shape=(rows, cols),
            transform=transform,
            fill=0.0,
            dtype="float32"
        )
        
        # Restore actual bounding box NoData
        burned[~valid_dem_mask] = -9999.0

        crs_wkt = out_gdf.crs.to_wkt() if out_gdf.crs else None

        with rasterio.open(
            overall_tif, "w",
            driver="GTiff",
            height=rows, width=cols,
            count=1,
            dtype="float32",
            crs=crs_wkt,
            transform=transform,
            nodata=-9999.0
        ) as dst:
            dst.write(burned, 1)

        # 9. Return paths to UI Wrapper
        output_json = {
            "status": "success",
            "output": main_output_path,
            "overall_tif": overall_tif
        }
        print(json.dumps(output_json))

    except Exception as e:
        import traceback
        print(json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()}))

if __name__ == "__main__":
    run_prediction()