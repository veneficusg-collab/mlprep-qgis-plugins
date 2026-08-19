// ml-prep-scripts/main.go
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

const noDataValue = "-9999"

var pythonPath string

// ── Python Scripts ──────────────────────────────────────────────────────────

var tileDEMScript = `
import sys, json, os
import rasterio
from rasterio.windows import Window
import rasterio.windows

dem_path   = sys.argv[1]
tiles_dir  = sys.argv[2]
n_cols     = int(sys.argv[3])
n_rows     = int(sys.argv[4])
overlap_px = int(sys.argv[5])

os.makedirs(tiles_dir, exist_ok=True)

with rasterio.open(dem_path) as src:
    W, H      = src.width, src.height
    transform = src.transform
    crs       = src.crs
    nodata    = src.nodata
    dtype     = src.dtypes[0]

    tile_w = W // n_cols
    tile_h = H // n_rows

    tiles = []
    for row in range(n_rows):
        for col in range(n_cols):
            x0 = max(0, col * tile_w - overlap_px)
            y0 = max(0, row * tile_h - overlap_px)
            x1 = min(W, (col + 1) * tile_w + overlap_px) if col < n_cols - 1 else W
            y1 = min(H, (row + 1) * tile_h + overlap_px) if row < n_rows - 1 else H

            cx0 = col * tile_w
            cy0 = row * tile_h
            cx1 = min(W, (col + 1) * tile_w)
            cy1 = min(H, (row + 1) * tile_h)

            window = Window(x0, y0, x1 - x0, y1 - y0)
            tile_transform = rasterio.windows.transform(window, transform)
            tile_path = os.path.join(tiles_dir, f"tile_{row}_{col}.tif")
            data = src.read(1, window=window)

            with rasterio.open(tile_path, "w", driver="GTiff", height=y1 - y0, width=x1 - x0,
                count=1, dtype=dtype, crs=crs, transform=tile_transform, nodata=nodata) as dst:
                dst.write(data, 1)

            # We pass the local core boundaries so Python knows exactly who owns what
            tiles.append({
                "path"     : tile_path,
                "global_x0": cx0,
                "global_y0": cy0,
                "global_x1": cx1,
                "global_y1": cy1,
                "local_x0" : cx0 - x0,
                "local_y0" : cy0 - y0,
                "local_x1" : cx1 - x0,
                "local_y1" : cy1 - y0,
            })

print(json.dumps({"tiles": tiles}))
`

var mergeSlopeUnitsScript = `
import sys, json, os
import numpy as np
import rasterio
from rasterio.merge import merge
import scipy.ndimage as ndi

dem_path    = sys.argv[1]
tiles_json  = sys.argv[2]
output_path = sys.argv[3]

tiles = json.loads(tiles_json)

id_offset = 0
tmp_files = []

for tile in tiles:
    slu_path = tile.get("slu_path")
    if not slu_path or not os.path.exists(slu_path): continue
    
    lx0, ly0 = tile["local_x0"], tile["local_y0"]
    lx1, ly1 = tile["local_x1"], tile["local_y1"]
    
    with rasterio.open(slu_path) as src:
        data = src.read(1)
        meta = src.meta.copy()
        nodata_val = src.nodata if src.nodata is not None else -9999
        
    valid_mask = data > 0
    if not valid_mask.any():
        continue
        
    # --- THE CENTROID OWNERSHIP ALGORITHM ---
    # 1. Get all unique polygon IDs in this tile
    unique_ids = np.unique(data[valid_mask])
    
    # 2. Calculate the exact center of mass for every polygon
    centers = ndi.center_of_mass(np.ones_like(data), data, unique_ids)
    
    # 3. Keep ONLY the polygons whose centroid falls inside this tile's CORE territory.
    keep_ids = set()
    for uid, (cy, cx) in zip(unique_ids, centers):
        if np.isnan(cy) or np.isnan(cx): continue
        if ly0 <= cy < ly1 and lx0 <= cx < lx1:
            keep_ids.add(uid)
            
    # 4. Erase all polygons that belong to a neighboring tile
    mask = ~np.isin(data, list(keep_ids))
    data[mask] = nodata_val
    
    # 5. Renumber the kept polygons so there are no duplicate IDs globally
    final_valid = data > 0
    if final_valid.any():
        final_ids = np.unique(data[final_valid])
        mapping = {old_id: new_id + id_offset for new_id, old_id in enumerate(final_ids, start=1)}
        
        palette = np.zeros(int(data.max()) + 1, dtype=np.int32)
        for old_id, new_id in mapping.items():
            palette[int(old_id)] = new_id
            
        data[final_valid] = palette[data[final_valid]]
        id_offset += len(final_ids)

    tmp_path = slu_path.replace(".tif", "_centroid_jagged.tif")
    with rasterio.open(tmp_path, "w", **meta) as dst:
        dst.write(data, 1)
    tmp_files.append(tmp_path)

if not tmp_files:
    print(json.dumps({"status": "error", "message": "No tiles generated valid data"}))
    sys.exit(1)

# Because ownership is strictly assigned by centroid, the jagged tiles will interlock 
# perfectly without leaving any straight lines or any gaps!
datasets = [rasterio.open(p) for p in tmp_files]
mosaic, out_tf = merge(datasets, method="first", nodata=-9999)

with rasterio.open(dem_path) as ref:
    out_meta = ref.meta.copy()
    
out_meta.update({
    "height": mosaic.shape[1], 
    "width": mosaic.shape[2], 
    "transform": out_tf, 
    "dtype": "int32", 
    "nodata": -9999
})

with rasterio.open(output_path, "w", **out_meta) as dst:
    dst.write(mosaic)

for ds in datasets: ds.close()
for p in tmp_files:
    try: os.remove(p)
    except: pass

print(json.dumps({"status": "ok", "output": output_path}))
`

var polygonizeScript = `
import sys, json, traceback
import rasterio
from rasterio.features import shapes, sieve
import fiona

input_tif = sys.argv[1]
output_gpkg = sys.argv[2]
area_min = float(sys.argv[3])

try:
    with rasterio.open(input_tif) as src:
        image = src.read(1)
        mask = image > 0
        
        pixel_area = abs(src.transform.a * src.transform.e)
        sieve_size = max(1, int(area_min / pixel_area))
        
        # Melt away tiny artifact splinters
        sieved_image = sieve(image, size=sieve_size, connectivity=8, mask=mask)
        
        schema = {
            'geometry': 'Polygon',
            'properties': {'su_id': 'int'}
        }
        
        with fiona.open(output_gpkg, 'w', driver='GPKG', crs=src.crs, schema=schema) as dest:
            for s, v in shapes(sieved_image, mask=(sieved_image > 0), transform=src.transform):
                dest.write({
                    'geometry': s,
                    'properties': {'su_id': int(v)}
                })

    print(json.dumps({"status": "ok"}))
except Exception as e:
    sys.stderr.write(f"Polygonize Python Error:\n{traceback.format_exc()}\n")
    sys.exit(1)
`

var zonalStatScript = `
import sys, json, os
import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats
import rasterio
from rasterio.warp import reproject, Resampling
from shapely.geometry import box

zone_input      = sys.argv[1]
value_raster    = sys.argv[2]
output_gpkg     = sys.argv[3]
nodata_val      = float(sys.argv[4])
stat_rule       = sys.argv[5] 
resample_method = sys.argv[6]

raster_name = os.path.basename(value_raster).replace(".tif", "").replace(" ", "_")

if zone_input.lower().endswith(('.gpkg', '.shp')):
    gdf = gpd.read_file(zone_input)
    stats = zonal_stats(gdf, value_raster, stats=stat_rule, geojson_out=False, nodata=nodata_val)
    gdf[raster_name] = [s.get(stat_rule, None) for s in stats]
    gdf.to_file(output_gpkg, driver="GPKG")
    rows_written = len(gdf)
else:
    CHUNK_SIZE = 500_000
    resampling = Resampling.bilinear if resample_method == "bilinear" else Resampling.nearest

    with rasterio.open(zone_input) as z:
        zone_data = z.read(1)
        zone_transform = z.transform
        zone_crs = z.crs
        zone_shape = z.shape

    with rasterio.open(value_raster) as v:
        value_nodata = v.nodata if v.nodata is not None else nodata_val
        value_data = np.empty(zone_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(v, 1), destination=value_data,
            src_transform=v.transform, src_crs=v.crs,
            dst_transform=zone_transform, dst_crs=zone_crs, dst_shape=zone_shape,
            resampling=resampling,
        )

    rows_idx, cols_idx = np.where(zone_data > 0)
    values = value_data[rows_idx, cols_idx].astype(float)
    nodata_mask = (values == value_nodata) | (values == nodata_val) | np.isnan(values)
    values[nodata_mask] = np.nan

    half = abs(zone_transform.a) / 2.0
    first_chunk, total_written = True, 0

    for start in range(0, len(rows_idx), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(rows_idx))
        c_rows, c_cols, c_vals = rows_idx[start:end], cols_idx[start:end], values[start:end]
        if np.all(np.isnan(c_vals)): continue

        c_east = zone_transform.c + (c_cols + 0.5) * zone_transform.a
        c_north = zone_transform.f + (c_rows + 0.5) * zone_transform.e
        c_geoms = [box(c_east[i]-half, c_north[i]-half, c_east[i]+half, c_north[i]+half) for i in range(len(c_rows))]

        chunk_gdf = gpd.GeoDataFrame({raster_name: c_vals}, geometry=c_geoms, crs=zone_crs).dropna(subset=[raster_name])
        if len(chunk_gdf) == 0: continue

        chunk_gdf.to_file(output_gpkg, layer=raster_name, driver="GPKG", mode="w" if first_chunk else "a")
        first_chunk = False
        total_written += len(chunk_gdf)
    rows_written = total_written

print(json.dumps({"status": "ok", "rows": rows_written, "layer": raster_name}))
`

var mergeScript = `
import sys, os, glob
import geopandas as gpd

zone_gpkg = sys.argv[1]
output_dir = sys.argv[2]
final_out = sys.argv[3]

master_gdf = gpd.read_file(zone_gpkg)
gpkg_files = glob.glob(os.path.join(output_dir, "*_zonal.gpkg"))

for f in gpkg_files:
    temp_gdf = gpd.read_file(f)
    new_cols = [c for c in temp_gdf.columns if c not in master_gdf.columns and c != 'geometry']
    for c in new_cols:
        master_gdf[c] = temp_gdf[c]

master_gdf.to_file(final_out, driver="GPKG")

for f in gpkg_files:
    try: os.remove(f)
    except: pass
`

// ── Core Utilities ──────────────────────────────────────────────────────────

func runPythonJSON(script, desc string, args ...string) ([]byte, error) {
	tmp, err := os.CreateTemp("", "*.py")
	if err != nil {
		return nil, err
	}
	defer os.Remove(tmp.Name())
	tmp.WriteString(script)
	tmp.Close()

	cmdArgs := append([]string{tmp.Name()}, args...)
	cmd := exec.Command(pythonPath, cmdArgs...)
	var stderrBuf strings.Builder
	cmd.Stderr = io.MultiWriter(os.Stderr, &stderrBuf)

	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("%s failed: %w\nStdout: %s\nStderr: %s", desc, err, string(out), stderrBuf.String())
	}
	return out, nil
}

// ── Helper: Find GRASS Executable ───────────────────────────────────────────
func getGrassExe() string {
	if runtime.GOOS == "darwin" {
		paths := []string{
			"/Applications/GRASS-8.4.app/Contents/Resources/bin/grass",
			"/Applications/GRASS-8.4.app/Contents/Resources/bin/grass84",
			"/Applications/GRASS-8.3.app/Contents/Resources/bin/grass",
			"/Applications/GRASS-8.3.app/Contents/Resources/bin/grass83",
		}
		for _, p := range paths {
			if _, err := os.Stat(p); err == nil {
				return p 
			}
		}
	}
	return "grass"
}

// ── Phase 1: Parallel Slope Unit Generation ─────────────────────────────────

type SlopeUnitParams struct {
	AreaMin      int
	Thresh       int
	CvMin        float64
	Rf           int
	MaxIteration int
}

type TileInfo struct {
	Path     string `json:"path"`
	SluPath  string `json:"slu_path"`
	GlobalX0 int    `json:"global_x0"`
	GlobalY0 int    `json:"global_y0"`
	GlobalX1 int    `json:"global_x1"`
	GlobalY1 int    `json:"global_y1"`
	LocalX0  int    `json:"local_x0"`
	LocalY0  int    `json:"local_y0"`
	LocalX1  int    `json:"local_x1"`
	LocalY1  int    `json:"local_y1"`
}

func tileDEM(demPath, tilesDir string, nCols, nRows, overlapPx int) ([]TileInfo, error) {
	out, err := runPythonJSON(tileDEMScript, "Tile DEM", demPath, tilesDir, fmt.Sprint(nCols), fmt.Sprint(nRows), fmt.Sprint(overlapPx))
	if err != nil {
		return nil, err
	}
	var res struct{ Tiles []TileInfo `json:"tiles"` }
	if err := json.Unmarshal(out, &res); err != nil {
		return nil, err
	}
	return res.Tiles, nil
}

func generateGrassSlopeUnits(demPath, outputDir, crs string, p SlopeUnitParams) (string, error) {
	baseName := strings.TrimSuffix(filepath.Base(demPath), ".tif")
	gisdbase := filepath.Join(os.TempDir(), "grass_db_" + baseName)
	grassMapset := filepath.Join(gisdbase, "loc", "PERMANENT")
	outputPath := filepath.Join(outputDir, baseName+"_slopeunit.tif")

	os.MkdirAll(filepath.Dir(outputPath), 0755)
	os.RemoveAll(gisdbase)
	os.MkdirAll(gisdbase, 0755) 
	defer os.RemoveAll(gisdbase)

	script := fmt.Sprintf(`#!/bin/bash
set -e
r.in.gdal -o input="%s" output=dem band=1 --overwrite
g.region raster=dem
r.slopeunits.create demmap=dem slumap=slu areamin=%d thresh=%d cvmin=%f rf=%d maxiteration=%d --overwrite
r.out.gdal -c input=slu output="%s" format=GTiff type=Int32 createopt="COMPRESS=LZW,TILED=YES" --overwrite
`, demPath, p.AreaMin, p.Thresh, p.CvMin, p.Rf, p.MaxIteration, outputPath)

	tmp, _ := os.CreateTemp("", "slu_*.sh")
	tmp.WriteString(script)
	tmp.Close()
	os.Chmod(tmp.Name(), 0755)
	defer os.Remove(tmp.Name())

	addonPath := filepath.Join(os.Getenv("HOME"), ".grass8", "addons")
	grassExe := getGrassExe()
	
	cmd := exec.Command(grassExe, "-c", demPath, grassMapset, "--exec", "bash", tmp.Name())
	cmd.Env = append(os.Environ(), 
		"GRASS_ADDON_PATH="+addonPath,
		"GISRC="+filepath.Join(gisdbase, ".grassrc"), 
		"GRASS_SKIP_MAPSET_OWNER_CHECK=1",            
	)
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("grass execution failed: %w", err)
	}
	return outputPath, nil
}

func generateSlopeUnitsParallel(demPath, outputDir, crs string, p SlopeUnitParams, cols, rows, overlap int) (string, error) {
	fmt.Println("[Phase 1] Tiling DEM for Parallel Slope Unit Generation...")
	tilesDir := filepath.Join(outputDir, "tiles_temp")
	os.MkdirAll(tilesDir, 0755)
	defer os.RemoveAll(tilesDir)

	tiles, err := tileDEM(demPath, tilesDir, cols, rows, overlap)
	if err != nil {
		return "", err
	}

	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())
	errs := make([]error, len(tiles))

	for i := range tiles {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int) {
			defer wg.Done()
			defer func() { <-sem }()
			sluDir := filepath.Join(tilesDir, fmt.Sprintf("slu_%d", idx))
			os.MkdirAll(sluDir, 0755)
			
			sluPath, e := generateGrassSlopeUnits(tiles[idx].Path, sluDir, crs, p)
			if e == nil {
				tiles[idx].SluPath = sluPath
				fmt.Printf("   > Tile %d generated.\n", idx)
			} else {
				errs[idx] = e
			}
		}(i)
	}
	wg.Wait()

	for _, e := range errs {
		if e != nil {
			return "", fmt.Errorf("a tile failed to process: %v", e)
		}
	}

	fmt.Println("[Phase 1] Merging Interlocking Tiles...")
	mergedTif := filepath.Join(outputDir, "Merged_SlopeUnits.tif")
	tilesJSON, _ := json.Marshal(tiles)
	
	_, err = runPythonJSON(mergeSlopeUnitsScript, "Merge Tiles", demPath, string(tilesJSON), mergedTif)
	if err != nil {
		return "", err
	}

	fmt.Println("[Phase 1] Sieve-Cleaning Seams and Polygonizing to GPKG...")
	finalGpkg := filepath.Join(outputDir, "Final_SlopeUnits.gpkg")
	_, err = runPythonJSON(polygonizeScript, "Polygonize", mergedTif, finalGpkg, fmt.Sprint(p.AreaMin))
	
	os.Remove(mergedTif) 
	return finalGpkg, err
}

// ── Phase 2: Zonal Statistics ───────────────────────────────────────────────

func reprojectRaster(src, dstFolder, crs string) (string, error) {
	dst := filepath.Join(dstFolder, filepath.Base(src))
	if _, err := os.Stat(dst); err == nil {
		return dst, nil
	}
	cmd := exec.Command("gdalwarp", "-t_srs", crs, "-r", "near", "-dstnodata", noDataValue, "-overwrite", src, dst)
	if err := cmd.Run(); err != nil {
		return "", err
	}
	return dst, nil
}

func processRaster(rasterPath, zoneRaster, outputDir, crs string, wg *sync.WaitGroup, sem chan struct{}) {
	defer wg.Done()
	defer func() { <-sem }()

	baseName := strings.TrimSuffix(filepath.Base(rasterPath), ".tif")
	statRule := "mean"
	resample := "bilinear"

	switch baseName {
	case "SoilType", "LULC":
		statRule = "majority"
		resample = "nearest"
	case "DistanceToFault", "DistanceToRoad", "DistanceToRiver":
		statRule = "min"
	}

	reprojPath, err := reprojectRaster(rasterPath, outputDir, crs)
	if err != nil {
		fmt.Printf("❌ Failed to reproject %s: %v\n", baseName, err)
		return
	}

	outGpkg := filepath.Join(outputDir, fmt.Sprintf("%s_zonal.gpkg", baseName))
	_, err = runPythonJSON(zonalStatScript, "Zonal Stats", zoneRaster, reprojPath, outGpkg, noDataValue, statRule, resample)
	if err != nil {
		fmt.Printf("❌ Zonal Error on %s: %v\n", baseName, err)
		return
	}

	fmt.Printf("[RASTER_DONE] %s\n", baseName)
}

func main() {
	var (
		demPath       string
		slopeUnitPath string
		rasterFolder  string
		outputDir     string
		targetCRS     string
		areaMin       int
		thresh        int
		cvMin         float64
		rf            int
		maxIter       int
		tileCols      int
		tileRows      int
		tileOverlap   int
	)

	flag.StringVar(&demPath, "dem", "", "Path to DEM for slope unit generation")
	flag.StringVar(&slopeUnitPath, "slopeUnit", "", "Path to existing slope unit GPKG (skips generation)")
	flag.StringVar(&rasterFolder, "rasterFolder", "", "Directory with value rasters")
	flag.StringVar(&outputDir, "outputDir", "output", "Output directory")
	flag.StringVar(&pythonPath, "pythonExe", "python3", "Path to Python executable")
	flag.StringVar(&targetCRS, "targetCRS", "EPSG:4326", "Target CRS")
	
	flag.IntVar(&areaMin, "areaMin", 5000, "r.slopeunits: area min")
	flag.IntVar(&thresh, "thresh", 1000, "r.slopeunits: threshold")
	flag.Float64Var(&cvMin, "cvMin", 0.15, "r.slopeunits: cv min")
	flag.IntVar(&rf, "rf", 10, "r.slopeunits: reduction factor")
	flag.IntVar(&maxIter, "maxIter", 10, "r.slopeunits: max iterations")
	flag.IntVar(&tileCols, "tileCols", 1, "Number of tile columns")
	flag.IntVar(&tileRows, "tileRows", 1, "Number of tile rows")
	flag.IntVar(&tileOverlap, "tileOverlap", 50, "Tile overlap in pixels")
	flag.Parse()

	os.MkdirAll(outputDir, 0755)

	zoneRaster := slopeUnitPath
	if zoneRaster == "" && demPath != "" {
		p := SlopeUnitParams{AreaMin: areaMin, Thresh: thresh, CvMin: cvMin, Rf: rf, MaxIteration: maxIter}
		out, err := generateSlopeUnitsParallel(demPath, outputDir, targetCRS, p, tileCols, tileRows, tileOverlap)
		if err != nil {
			log.Fatalf("Slope Unit Generation Failed: %v", err)
		}
		zoneRaster = out
	}

	if zoneRaster == "" {
		log.Fatalf("Must provide either -dem or -slopeUnit")
	}

	entries, _ := os.ReadDir(rasterFolder)
	var rasters []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(strings.ToLower(e.Name()), ".tif") {
			rasters = append(rasters, filepath.Join(rasterFolder, e.Name()))
		}
	}

	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())

	for _, raster := range rasters {
		wg.Add(1)
		sem <- struct{}{}
		go processRaster(raster, zoneRaster, outputDir, targetCRS, &wg, sem)
	}
	wg.Wait()

	fmt.Println("[Merging] Combining all features...")
	finalGPKG := filepath.Join(outputDir, "Merged_PINN_Features.gpkg")
	_, err := runPythonJSON(mergeScript, "Merge Script", zoneRaster, outputDir, finalGPKG)
	if err != nil {
		log.Fatalf("Merge Failed: %v", err)
	}

	fmt.Println("[MERGE_DONE] " + finalGPKG)
}