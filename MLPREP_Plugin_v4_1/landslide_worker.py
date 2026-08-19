import sys
import argparse
import os
import json
import numpy as np
import shapely.geometry
import rasterio
import rasterio.mask
import geopandas as gpd
from rasterio import features
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds
from skimage.filters import threshold_otsu

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

TARGET_CRS = "EPSG:4326"
METRIC_CRS = "EPSG:32651" 
BUFFER_METERS = 10

def generate_otsu_mask(dem_path, before_path, after_path, lulc_path, roi_geom, roi_crs):
    """Executes the Otsu Thresholding algorithm."""
    shapes = [roi_geom] if roi_geom else None
    
    with rasterio.open(after_path) as src_after:
        after_img, after_transform = rasterio.mask.mask(src_after, shapes, crop=True, nodata=np.nan, filled=True) if shapes else (src_after.read(), src_after.transform)
        after_crs = src_after.crs
        
    with rasterio.open(before_path) as src_before:
        before_img, _ = rasterio.mask.mask(src_before, shapes, crop=True, nodata=np.nan, filled=True) if shapes else (src_before.read(), src_before.transform)

    if after_img.shape != before_img.shape:
        raise ValueError(f"Shape mismatch: Post {after_img.shape} vs Pre {before_img.shape}")

    def calculate_ndvi(arr):
        # Automatically extract the correct bands from a 12-band Sentinel-2 stack
        if arr.shape[0] == 12:
            nir = arr[7].astype('float64') # Band 8 (NIR) is index 7
            red = arr[3].astype('float64') # Band 4 (Red) is index 3
        # Fallback to standard false color composition logic
        elif arr.shape[0] >= 3:
            nir = arr[0].astype('float64')
            red = arr[1].astype('float64')
        else:
            raise ValueError(f"Image has {arr.shape[0]} bands. Need at least 3 for NDVI.")
            
        denominator = nir + red
        np.seterr(divide='ignore', invalid='ignore')
        return np.where(denominator != 0, (nir - red) / denominator, np.nan)

    ndvi_diff = calculate_ndvi(after_img) - calculate_ndvi(before_img)

    if dem_path and os.path.exists(dem_path):
        with rasterio.open(dem_path) as src_dem:
            dem_img, dem_transform = rasterio.mask.mask(src_dem, [gpd.GeoSeries([roi_geom], crs=roi_crs).to_crs(src_dem.crs).iloc[0]], crop=True) if shapes else (src_dem.read(), src_dem.transform)
            elevation = np.where(dem_img[0] == src_dem.nodata, np.nan, dem_img[0])
            
            res_x, res_y = src_dem.res
            scale_factor = 111320 if src_dem.crs.is_geographic else 1.0
            dy, dx = np.gradient(elevation, res_y * scale_factor, res_x * scale_factor)
            slope_percent = np.sqrt(dx**2 + dy**2) * 100
            
            conditions = [
                (elevation < 5) & (slope_percent < 8),
                (elevation >= 5) & (elevation <= 50) & (slope_percent < 8),
                (elevation > 50) & (elevation <= 150) & (slope_percent < 8),
                (elevation > 150) & (elevation <= 500) & (slope_percent < 8),
                (elevation >= 500) & (slope_percent < 20),
                (elevation > 50) & (elevation <= 500) & (slope_percent >= 8) & (slope_percent < 20),
                (elevation < 50) & (slope_percent >= 8),
                (elevation >= 50) & (elevation <= 500) & (slope_percent >= 20),
                (elevation >= 500) & (slope_percent >= 20)
            ]
            geomorphology = np.select(conditions, [1, 2, 3, 4, 5, 6, 7, 8, 9], default=0).astype(np.uint8)
            raw_hazard_mask = (geomorphology >= 8).astype(np.uint8)

            aligned_hazard_mask = np.zeros(ndvi_diff.shape, dtype='uint8')
            reproject(source=raw_hazard_mask, destination=aligned_hazard_mask, src_transform=dem_transform, src_crs=src_dem.crs, dst_transform=after_transform, dst_crs=after_crs, resampling=Resampling.nearest)
            ndvi_diff[aligned_hazard_mask == 0] = np.nan

    valid_pixels = ndvi_diff[~np.isnan(ndvi_diff)]
    if valid_pixels.size > 0:
        calculated_otsu = threshold_otsu(valid_pixels)
        mask = (ndvi_diff < calculated_otsu).astype('uint8')
    else:
        mask = np.zeros_like(ndvi_diff, dtype='uint8')

    mask[np.isnan(ndvi_diff)] = 0
    return mask, after_transform, after_crs, (after_img.shape[1], after_img.shape[2])


def generate_cnn_mask(dem_path, after_path, roi_geom, roi_crs, model_path, stats_path):
    """Executes the Deep Learning CNN inference."""
    PATCH_SIZE = 128
    OVERLAP = 0.5
    
    with rasterio.open(after_path) as src_after:
        s2_crs = src_after.crs
        if roi_geom is not None:
            geom_s2 = gpd.GeoSeries([roi_geom], crs=roi_crs).to_crs(s2_crs).iloc[0]
            s2_img_raw, s2_transform = rasterio.mask.mask(src_after, [geom_s2], crop=True, nodata=np.nan, filled=True)
        else:
            s2_img_raw = src_after.read(boundless=True, fill_value=0)
            s2_transform = src_after.transform
        after_crs = src_after.crs

    with rasterio.open(dem_path) as src_dem:
        dem_img = np.zeros((s2_img_raw.shape[1], s2_img_raw.shape[2]), dtype=np.float32)
        reproject(source=src_dem.read(1), destination=dem_img, src_transform=src_dem.transform, src_crs=src_dem.crs, dst_transform=s2_transform, dst_crs=after_crs, resampling=Resampling.bilinear)
        res_x, res_y = abs(s2_transform[0]), abs(s2_transform[4])
        scale_factor = 111320 if after_crs.to_epsg() == 4326 else 1
        dy, dx = np.gradient(dem_img, res_y * scale_factor, res_x * scale_factor)
        
        slope_scaled = (np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))) / 10.0
        dem_scaled = dem_img / 1000.0

        s2_img_scaled = np.zeros_like(s2_img_raw[:12], dtype=np.float32)
        for i in range(12):
            band = s2_img_raw[i]
            valid = band[np.isfinite(band) & (band > 0)]
            s2_img_scaled[i] = np.nan_to_num(band) / (valid.mean() if valid.size > 0 else 1.0)

    input_array = np.nan_to_num(np.transpose(np.vstack([s2_img_scaled, slope_scaled[np.newaxis, ...], dem_scaled[np.newaxis, ...]]), (1, 2, 0)), nan=0.0)

    with open(stats_path, 'r') as f:
        stats = json.load(f)
    input_array = (input_array - np.array(stats["mean"], dtype=np.float32)) / np.array(stats["std"], dtype=np.float32)

    model = tf.keras.models.load_model(model_path, compile=False)
    H, W = input_array.shape[:2]
    win_1d = np.hanning(PATCH_SIZE + 2)[1:-1]
    win_2d = np.outer(win_1d, win_1d).astype(np.float32)
    prob_acc = np.zeros((H, W), np.float32)
    weight_acc = np.zeros((H, W), np.float32)

    stride = max(1, int(PATCH_SIZE * (1.0 - OVERLAP)))
    rows = sorted(set(list(range(0, max(1, H - PATCH_SIZE + 1), stride)) + [max(0, H - PATCH_SIZE)]))
    cols = sorted(set(list(range(0, max(1, W - PATCH_SIZE + 1), stride)) + [max(0, W - PATCH_SIZE)]))

    for r in rows:
        for c in cols:
            h, w = min(PATCH_SIZE, H - r), min(PATCH_SIZE, W - c)
            tile = input_array[r:r+h, c:c+w, :]
            if (h, w) != (PATCH_SIZE, PATCH_SIZE):
                tile = np.pad(tile, ((0, PATCH_SIZE - h), (0, PATCH_SIZE - w), (0, 0)), mode="reflect")
            
            preds = model.predict(tile[np.newaxis, ...].astype(np.float32), verbose=0)[0]
            prob = tf.nn.softmax(preds, axis=-1)[..., 1].numpy() if preds.shape[-1] > 1 else preds[..., 0]
            prob_acc[r:r+h, c:c+w] += (prob * win_2d)[:h, :w]
            weight_acc[r:r+h, c:c+w] += win_2d[:h, :w]

    final_prediction = ((prob_acc / np.maximum(weight_acc, 1e-6)) > 0.5).astype(np.uint8)
    return final_prediction, s2_transform, after_crs, (H, W)


def apply_lulc_mask(mask, lulc_path, transform, crs, shape):
    """Masks out built-up, water, and barren areas using LULC."""
    if lulc_path and str(lulc_path).lower() != 'none' and os.path.exists(lulc_path):
        landcover_gdf = gpd.read_file(lulc_path).to_crs(crs)
        exclude_classes = ["Built-up", "Inland Water", "Marshland/Swamp", "Open/Barren", "Annual Crop"]
        col_name = next((c for c in ['Class_Name', 'CLASS_NAME', 'type', 'TYPE', 'class_name'] if c in landcover_gdf.columns), None)
        
        if col_name:
            mask_out_gdf = landcover_gdf[landcover_gdf[col_name].isin(exclude_classes)]
            if not mask_out_gdf.empty:
                mask_out_gdf = mask_out_gdf.to_crs(METRIC_CRS)
                mask_out_gdf['geometry'] = mask_out_gdf.geometry.buffer(BUFFER_METERS)
                mask_out_gdf = mask_out_gdf.to_crs(crs)
                if not mask_out_gdf.empty:
                    lc_mask = features.geometry_mask(mask_out_gdf.geometry, out_shape=shape, transform=transform, invert=True)
                    mask[lc_mask] = 0
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["otsu", "cnn", "ensemble"])
    parser.add_argument("--dem", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--before", required=False, default=None)
    parser.add_argument("--lulc", required=False, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--roi_path", required=False, default=None)
    parser.add_argument("--filter_col", required=False, default=None)
    parser.add_argument("--filter_val", required=False, default=None)
    parser.add_argument("--model_path", required=False, default=None)
    parser.add_argument("--stats_path", required=False, default=None)
    args = parser.parse_args()

    try:
        # Load ROI Geometry once
        roi_geom, roi_crs = None, None
        if args.roi_path and str(args.roi_path).lower() != 'none' and os.path.exists(args.roi_path):
            gdf = gpd.read_file(args.roi_path)
            if args.filter_col and args.filter_val:
                gdf = gdf[gdf[args.filter_col] == args.filter_val]
            if not gdf.empty:
                roi_geom, roi_crs = gdf.geometry.unary_union, gdf.crs

        # Execute appropriate models
        final_mask, transform, crs, shape = None, None, None, None
        
        if args.method in ["otsu", "ensemble"]:
            otsu_mask, transform, crs, shape = generate_otsu_mask(args.dem, args.before, args.after, args.lulc, roi_geom, roi_crs)
            final_mask = otsu_mask
            
        if args.method in ["cnn", "ensemble"]:
            cnn_mask, transform_cnn, crs_cnn, shape_cnn = generate_cnn_mask(args.dem, args.after, roi_geom, roi_crs, args.model_path, args.stats_path)
            if args.method == "ensemble":
                # Ensemble Intersection: Only keep pixels where BOTH models detected a landslide
                final_mask = np.bitwise_and(final_mask, cnn_mask)
            else:
                final_mask, transform, crs, shape = cnn_mask, transform_cnn, crs_cnn, shape_cnn

        # Apply common LULC filtering
        final_mask = apply_lulc_mask(final_mask, args.lulc, transform, crs, shape)

        # Write Output
        with rasterio.open(args.after) as src:
            meta = src.meta.copy()
            meta.update({"driver": "GTiff", "height": shape[0], "width": shape[1], "transform": transform, "count": 1, "dtype": 'uint8', "nodata": 0})
        
        with rasterio.open(args.output, "w", **meta) as dst:
            dst.write(final_mask, 1)

        print(json.dumps({"status": "success", "output_mask": args.output, "method_used": args.method}))

    except Exception as e:
        import traceback
        print(json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()}))

if __name__ == "__main__":
    main()