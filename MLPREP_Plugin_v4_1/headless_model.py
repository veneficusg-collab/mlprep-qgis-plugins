#headless_model.py
import sys
import os
import argparse
import numpy as np
import pandas as pd
import sqlite3
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable

import importlib, MLPREP_Plugin_v3_Final.py_files.GallenModel_v1 as GallenModel_v1, MLPREP_Plugin_v3_Final.py_files.Landslidev2_Old as Landslidev2_Old
from typing import Optional
importlib.reload(GallenModel_v1)
importlib.reload(Landslidev2_Old)
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn import pipeline, under_sampling
import keras_tuner as kt
import sklearn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import geopandas as gpd
import seaborn as sns
import pandas as pd
import contextily as cx
from MLPREP_Plugin_v3_Final.py_files.GallenModel_v1 import  NewmarkActivation, DisplacementLayer, LandslideActivationLayer, CohesionLayer, InternalFrictionLayer
from MLPREP_Plugin_v3_Final.py_files.Landslidev2_Old import DiceCrossEntropyLoss



def run_prediction(gpkg_path, model_path, output_csv):
    df = gpd.read_file(gpkg_path)
    df.drop(columns=['landslide_probability', 'landslide_preds', 'confusion', 'sus_pinn_landslide', 'sus_pinn_ground truth', 'ds', 'cohesion', 'internal_friction'], inplace=True, errors='ignore')

    df = df[df['Slope_mean'] >= 10]
    df['fid'] = df.index
    columns = list(df.columns)

    

    df.dropna(subset=list(columns), inplace=True) #cleans the dataframe by removing null rows for all columns

    cols_remove = ['DN', 'BD_mean', 'geometry', 'PGA2_max', 'Soil Type', 'description', 'descriptio',]
    columns = [col for col in list(columns) if col not in cols_remove]
    sampling_columns = [col for col in columns if col != 'landslide' and col != 'type']

    model = load_model(model_path)

    def dataframe_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe)))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    # 1. Handle the 'type' column (MUST BE STRING)
    if 'type' not in df.columns:
        if 'Lithology' in df.columns:
            print("Renaming 'Lithology' to 'type'...")
            df['type'] = df['Lithology'].astype(str)
        elif 'LITHO' in df.columns:
            print("Renaming 'LITHO' to 'type'...")
            df['type'] = df['LITHO'].astype(str)
        elif 'LITHODESC' in df.columns:
            print("Renaming 'LITHODESC' to 'type'...")
            df['type'] = df['LITHODESC'].astype(str)
        else:
            print("Warning: Creating dummy 'type' column ('0').")
            df['type'] = "0"
    else:
        df['type'] = df['type'].astype(str)

    # 2. Add 'type' to columns list if missing
    if 'type' not in columns:
        columns.append('type')

    for col in columns:
        if col not in ['type', 'fid', 'geometry']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop features where critical physics inputs are missing
    critical_cols = ['Slope_mean', 'PGA2_max', 'SoilThc_mean', 'BUK_mean']
    available_criticals = [c for c in critical_cols if c in df.columns]
    df.dropna(subset=available_criticals, inplace=True)

    # For non-criticals, use the mean instead of 0
    df.fillna(df.mean(numeric_only=True), inplace=True)
    

    validation_ds = dataframe_to_dataset(df[columns], shuffle=False)
    
    susceptibility_prediction = model.predict(validation_ds)

    df_wm = df.to_crs(epsg=3857)
    cohesion_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer("cohesion_layer").output)
    predicted_cohesion = cohesion_model.predict(validation_ds)
    df_wm['estimated_cohesion'] = predicted_cohesion

    friction_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("internal_friction").output)
    geotech_preds = friction_model.predict(validation_ds)
    df_wm["ifi"] = geotech_preds

    df_result = pd.DataFrame()
    df_result['fid'] = df['fid']

    df_result['sus_pinn_landslide'] = np.array(susceptibility_prediction).flatten()
    df_result['cohesion'] = np.array(predicted_cohesion).flatten()
    df_result['internal_friction'] = np.array(geotech_preds).flatten()

    df_result.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpkg", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True) 
    args = parser.parse_args()
    
    run_prediction(args.gpkg, args.model, args.output)
