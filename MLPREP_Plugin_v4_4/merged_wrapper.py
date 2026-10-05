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

    # Stream Definition by Threshold, applied to the D-infinity specific
    # catchment area. Calibrated so the stream-cell density matches the ~1.8%
    # of the Strahler network the shipped ML-DFI model was trained against;
    # the project's own Step 6 stream-mask comparison used SCA 500 and 5000.
    STREAM_SCA_THRESHOLD = 1500

    # The trained model was fitted on low-pass-filtered predictors
    # (Makilala_*_LPF_*.tif). Verified against the shipped rasters: the filter is
    # exactly a 3x3 uniform mean (corr 1.0000, max abs diff 3.3e-9). Applied to
    # every continuous predictor; geomorphology is categorical and stays raw,
    # as it does in Step2/run_config.json.
    LPF_WINDOW = 3

    def __init__(self, dem_path, pinn_tif, ndvi_path, fos_inputs, plugin_dir, output_dir, step4_wrapper_py, python_exe, cores=4):
        super().__init__()
        self.dem_path = dem_path
        self.pinn_tif = pinn_tif  # PINN overall raster (kept for reference; no longer selects sources)
        self.ndvi_path = ndvi_path
        # Objective 2 (PINN) rasters + UI predictors for fos_source_refinement.py:
        # keys: susceptibility, cohesion, friction_angle, wetness, friction_angle_unit,
        #       bulk_density, slope
        self.fos_inputs = fos_inputs or {}
        self.plugin_dir = plugin_dir
        self.output_dir = output_dir
        self.step4_wrapper_py = step4_wrapper_py
        self.python_exe = python_exe
        self.cores = cores

        self.taudem_dir = self._locate_taudem()
        self.mpi_exe = self._locate_mpi()

    def _build_cpp_inputs(self, ang, fel, alpha_tif, dfi_clean, source_mask_tif):
            """Build source_final.tif, dfi_final.tif and alpha_final.tif so they
            satisfy every validation rule in step4_deposition_zone_mpi.cpp.
    
            Must run AFTER Step 0 has written alpha.tif. Returns
            (source_path, dfi_path, alpha_path), or (None, None, None) after
            emitting a specific error if no legal source cell exists."""
            import os
            import numpy as np
            import rasterio
            from rasterio.warp import reproject, Resampling
    
            # Engine constants, copied from the C++ so they cannot drift apart.
            ALPHA_VALID_MIN, ALPHA_VALID_MAX = 1.0, 90.0    # kAlphaAngleValidMin/Max
            ALPHA_FIXED_MIN, ALPHA_FIXED_MAX = 6.0, 72.0    # kFixedAlphaMin/Max
            DFI_MID = 0.69                                  # matches --dfi-mid below
            OUT_NODATA = -9999.0
    
            self.log.emit("   > Building Step 4 inputs against the engine's validation contract...")
    
            with rasterio.open(ang) as template:
                t_meta = template.meta.copy()
                t_transform = template.transform
                t_crs = template.crs
                t_shape = template.shape
                ang_nodata = template.nodata
                ang_data = template.read(1).astype(np.float32)
    
            with rasterio.open(fel) as f:
                fel_data = f.read(1).astype(np.float32)
                fel_nodata = f.nodata
    
            # Terrain, defined exactly as the C++ defines it.
            valid_terrain = np.ones(t_shape, dtype=bool)
            if ang_nodata is not None:
                valid_terrain &= (ang_data != np.float32(ang_nodata))
            if fel_nodata is not None:
                valid_terrain &= (fel_data != np.float32(fel_nodata))
            valid_terrain &= np.isfinite(ang_data) & np.isfinite(fel_data)
            del fel_data
            self.log.emit(f"   > Terrain cells: {int(valid_terrain.sum())}")
    
            # ---- alpha: neutralise values the engine would reject outright ----
            with rasterio.open(alpha_tif) as a:
                alpha_data = a.read(1).astype(np.float32)
                alpha_nodata = a.nodata
            alpha_present = np.ones(t_shape, dtype=bool)
            if alpha_nodata is not None:
                alpha_present &= (alpha_data != np.float32(alpha_nodata))
            alpha_present &= valid_terrain
    
            alpha_bad = alpha_present & (~np.isfinite(alpha_data) |
                                         (alpha_data < ALPHA_VALID_MIN) |
                                         (alpha_data > ALPHA_VALID_MAX))
            n_bad = int(alpha_bad.sum())
            if n_bad:
                self.log.emit(f"   > Neutralising {n_bad} alpha cells outside "
                              f"[{ALPHA_VALID_MIN}, {ALPHA_VALID_MAX}] to NoData "
                              f"(the engine aborts on any of these).")
                alpha_data[alpha_bad] = np.float32(OUT_NODATA)
                alpha_present &= ~alpha_bad
    
            final_alpha = os.path.join(self.output_dir, "alpha_final.tif")
            a_meta = t_meta.copy()
            a_meta.update(dtype=rasterio.float32, nodata=OUT_NODATA, count=1)
            alpha_out = np.full(t_shape, OUT_NODATA, dtype=np.float32)
            alpha_out[alpha_present] = alpha_data[alpha_present]
            with rasterio.open(final_alpha, "w", **a_meta) as dst:
                dst.write(alpha_out, 1)
            del alpha_out
    
            alpha_usable = alpha_present & (alpha_data >= ALPHA_FIXED_MIN) & \
                                            (alpha_data <= ALPHA_FIXED_MAX)
            n_present = int(alpha_present.sum())
            n_usable = int(alpha_usable.sum())
            self.log.emit(f"   > Alpha valid on terrain: {n_present} | "
                          f"inside [{ALPHA_FIXED_MIN}, {ALPHA_FIXED_MAX}]: {n_usable}")
            del alpha_data
    
            # ---- DFI: must be non-NoData across the terrain for propagation ----
            with rasterio.open(dfi_clean) as d:
                dfi_warp = np.full(t_shape, OUT_NODATA, dtype=np.float32)
                reproject(source=rasterio.band(d, 1), destination=dfi_warp,
                          src_transform=d.transform, src_crs=d.crs,
                          dst_transform=t_transform, dst_crs=t_crs,
                          src_nodata=d.nodata, dst_nodata=OUT_NODATA,
                          resampling=Resampling.nearest)
            dfi_ok = valid_terrain & (dfi_warp != np.float32(OUT_NODATA)) & \
                     np.isfinite(dfi_warp) & (dfi_warp >= 0.0) & (dfi_warp <= 1.0)
    
            dfi_out = np.full(t_shape, OUT_NODATA, dtype=np.float32)
            # Neutral fill: DFI_MID sits inside the engine's deadband, so cells with
            # no model prediction neither speed up nor slow down the flow.
            dfi_out[valid_terrain] = np.float32(DFI_MID)
            dfi_out[dfi_ok] = np.clip(dfi_warp[dfi_ok], 0.0, 1.0)
            self.log.emit(f"   > DFI predicted on {int(dfi_ok.sum())} cells; "
                          f"{int(valid_terrain.sum() - dfi_ok.sum())} filled with "
                          f"neutral {DFI_MID}")
            final_dfi = os.path.join(self.output_dir, "dfi_final.tif")
            with rasterio.open(final_dfi, "w", **a_meta) as dst:
                dst.write(dfi_out, 1)
            del dfi_warp, dfi_out
    
            # ---- source: strictly binary, only where alpha is usable ----
            # Source cells come from the FoS-refined mask (fos_source_refinement.py):
            # a cell qualifies only if it lies in a PINN-susceptible slope unit AND its
            # cell-scale Factor of Safety is <= 1.5. Mask values: 1 = source,
            # 0 = non-source, 255 = NoData.
            with rasterio.open(source_mask_tif) as rsm:
                rsm_nodata = int(rsm.nodata) if rsm.nodata is not None else 255
                rsm_data = np.full(t_shape, rsm_nodata, dtype=np.uint8)
                reproject(source=rasterio.band(rsm, 1), destination=rsm_data,
                          src_transform=rsm.transform, src_crs=rsm.crs,
                          dst_transform=t_transform, dst_crs=t_crs,
                          src_nodata=rsm_nodata, dst_nodata=rsm_nodata,
                          resampling=Resampling.nearest)

            is_source = valid_terrain & (rsm_data == 1)
            n_raw = int(is_source.sum())
            legal = is_source & alpha_usable
            n_legal = int(legal.sum())
            del rsm_data

            self.log.emit(f"   > FoS-refined source candidates: {n_raw} | "
                          f"accepted by the engine's alpha rule: {n_legal} | "
                          f"discarded for missing/out-of-range alpha: {n_raw - n_legal}")

            if n_legal == 0:
                self.error.emit(
                    "No legal Step 4 source cell exists.\n\n"
                    f"The FoS-refined mask marked {n_raw} candidate cells, but BaselineAlpha.tif has no "
                    f"usable value at any of them (alpha valid on terrain: {n_present}, "
                    f"inside [{ALPHA_FIXED_MIN}, {ALPHA_FIXED_MAX}]: {n_usable}).\n\n"
                    "The C++ engine silently discards these into a counter it never prints, "
                    "which is why it only ever reported 'Initiating source cells: 0'.\n"
                    "Fix Step 0's alpha generation (coverage and/or value range) -- the "
                    "source raster is not the problem."
                )
                return None, None, None
    
            source_out = np.full(t_shape, OUT_NODATA, dtype=np.float32)
            source_out[valid_terrain] = 0.0     # exact 0.0 background: binary-legal
            source_out[legal] = 1.0             # exact 1.0: binary-legal
            final_source = os.path.join(self.output_dir, "source_final.tif")
            with rasterio.open(final_source, "w", **a_meta) as dst:
                dst.write(source_out, 1)
    
            uniq = np.unique(source_out[valid_terrain])
            self.log.emit(f"   > source_final.tif distinct values on terrain: {uniq.tolist()}")
            del source_out, valid_terrain, alpha_usable, alpha_present, is_source, legal
    
            return final_source, final_dfi, final_alpha

    def _convert_slope_tangent_to_degrees(self, tangent_path, degrees_path):
        """TauDEM's dinfflowdir -slp emits slope as a tangent (rise/run), not
        degrees. step0_baseline_alpha_angle.py expects degrees and does no
        conversion of its own -- feeding it the raw tangent puts nearly all
        valid cells into its 'slope < 10 degrees' policy class, which that
        script deliberately writes as alpha NoData."""
        import rasterio
        import numpy as np
        with rasterio.open(tangent_path) as src:
            tangent = src.read(1)
            nodata = src.nodata
            meta = src.profile.copy()
        valid = np.isfinite(tangent)
        if nodata is not None:
            valid &= (tangent != np.float32(nodata))
        degrees = np.full(tangent.shape, -9999.0, dtype=np.float32)
        degrees[valid] = np.degrees(np.arctan(tangent[valid])).astype(np.float32)
        meta.update(dtype=rasterio.float32, nodata=-9999.0, compress="LZW")
        with rasterio.open(degrees_path, "w", **meta) as dst:
            dst.write(degrees, 1)



    
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
        """Scrub the ML-DFI output to fix floating-point anomalies (e.g. 1.000001) that crash C++."""
        import rasterio
        import numpy as np

        self.log.emit("   > Cleaning DFI Raster (Clamping probabilities to [0, 1])...")
        with rasterio.open(input_dfi) as src:
            dfi = src.read(1).astype(np.float32)
            
            valid_mask = np.isfinite(dfi) & (dfi > -1.0)
            dfi[valid_mask] = np.clip(dfi[valid_mask], 0.0, 1.0)
            dfi[~valid_mask] = -9999.0
            
            meta = src.meta.copy()
            meta.update(dtype=rasterio.float32, nodata=-9999.0)
            with rasterio.open(clean_dfi, 'w', **meta) as dst:
                dst.write(dfi, 1)

    def _align_raster(self, input_raster, reference_raster, output_raster, is_mask=False):
        """Uses rasterio to strictly warp the input raster to match the reference raster."""
        import os
        self.log.emit(f"   > Aligning {os.path.basename(input_raster)} to exact DEM grid...")
        import rasterio
        from rasterio.warp import reproject, Resampling
        import numpy as np

        with rasterio.open(reference_raster) as ref:
            ref_transform = ref.transform
            ref_crs = ref.crs
            ref_width = ref.width
            ref_height = ref.height
            ref_meta = ref.meta.copy()

        with rasterio.open(input_raster) as src:
            nodata_val = src.nodata if src.nodata is not None else -9999.0
            
            ref_meta.update({
                "dtype": src.dtypes[0],
                "nodata": nodata_val
            })

            with rasterio.open(output_raster, 'w', **ref_meta) as dst:
                destination = np.full((ref_height, ref_width), nodata_val, dtype=src.dtypes[0])
                resampling_method = Resampling.nearest if is_mask else Resampling.bilinear
                
                # Explicity passing src_nodata and dst_nodata guarantees safe reprojection
                reproject(
                    source=rasterio.band(src, 1),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    src_nodata=nodata_val,
                    dst_nodata=nodata_val,
                    resampling=resampling_method
                )
                dst.write(destination, 1)

    def _generate_curvatures(self, dem_path, prof_path, plan_path):
        """Profile and plan curvature (Zevenbergen & Thorne), in 1/m, matching
        the sign and scale of the rasters the shipped ML-DFI model was trained on.

        NoData neighbours are replaced by the centre cell's elevation before the
        3x3 terms are computed. Without this, cells on the study-area edge
        pull the DEM's NoData value (e.g. -9999) into the formula and produce
        huge artificial curvature values along the boundary."""
        self.log.emit("   > Generating Plan & Profile Curvatures...")
        import rasterio
        import numpy as np
        with rasterio.open(dem_path) as src:
            Z = src.read(1).astype(np.float64)
            nodata = src.nodata
            cell_size = abs(src.transform.a)
            meta = src.meta.copy()

        invalid = ~np.isfinite(Z)
        if nodata is not None:
            invalid |= (Z == nodata)
        Z[invalid] = np.nan

        Zp = np.pad(Z, 1, mode="constant", constant_values=np.nan)

        def nb(dr, dc):
            """Neighbour grid shifted by (dr, dc); NoData replaced by the centre value."""
            a = Zp[1 + dr: Zp.shape[0] - 1 + dr, 1 + dc: Zp.shape[1] - 1 + dc]
            return np.where(np.isnan(a), Z, a)

        z1, z2, z3 = nb(-1, -1), nb(-1, 0), nb(-1, 1)
        z4, z6 = nb(0, -1), nb(0, 1)
        z7, z8, z9 = nb(1, -1), nb(1, 0), nb(1, 1)
        L = cell_size

        D = ((z4 + z6) / 2.0 - Z) / (L ** 2)
        E = ((z2 + z8) / 2.0 - Z) / (L ** 2)
        F = (-z1 + z3 + z7 - z9) / (4.0 * L ** 2)
        G = (-z4 + z6) / (2.0 * L)
        H = (z2 - z8) / (2.0 * L)
        del z1, z2, z3, z4, z6, z7, z8, z9, Zp

        denom = G ** 2 + H ** 2
        ok = (~invalid) & (denom > 0)

        plan = np.zeros(Z.shape, dtype=np.float32)
        prof = np.zeros(Z.shape, dtype=np.float32)
        # Scale and sign are set by the rasters the shipped model was trained on
        # (Makilala_ProfCurv/PlanCurv_*.tif), not by ArcGIS's display convention:
        # those are plain 1/m, so there is no x100 here. Regressing each candidate
        # form against them on the reference DEM gives
        #   profile: -2(DG^2+EH^2+FGH)/p       corr +0.93  (the +2 form is -0.95,
        #                                                   i.e. sign-inverted)
        #   plan:    -2(DH^2+EG^2-FGH)/p^1.5   corr +0.86  (the /p form is only +0.48)
        # and reproduces their p1/median/p99 to within a few percent.
        plan[ok] = (-2.0 * (D * H ** 2 + E * G ** 2 - F * G * H)[ok] / denom[ok] ** 1.5).astype(np.float32)
        prof[ok] = (-2.0 * (D * G ** 2 + E * H ** 2 + F * G * H)[ok] / denom[ok]).astype(np.float32)
        plan[invalid] = -9999.0
        prof[invalid] = -9999.0

        for name, arr in (("profile", prof), ("plan", plan)):
            v = arr[~invalid]
            if v.size:
                p1, p50, p99 = np.percentile(v, [1, 50, 99])
                self.log.emit(f"   > {name} curvature p1/p50/p99: {p1:.4f} / {p50:.4f} / {p99:.4f}")

        meta.update(dtype=rasterio.float32, nodata=-9999.0, count=1)
        with rasterio.open(plan_path, 'w', **meta) as dst:
            dst.write(plan, 1)
        with rasterio.open(prof_path, 'w', **meta) as dst:
            dst.write(prof, 1)

    def _seed_boundary_outlets(self, src_path, ang_path, block_rows=2048):
        """Write a routing copy of the stream raster with the analysis-domain
        boundary added as outlet cells, and return its path.

        TauDEM's dinfdistdown sets a cell to NoData whenever none of its
        downslope neighbours carries a result - the `sump<=0` branch of
        DinfDistDown.cpp, which -nc does NOT disable. On a DEM clipped to an
        irregular polygon, every interior NoData edge is such a sink, and the
        resulting NoData then propagates upstream along the flow paths, which is
        what produced the blank streaks through dstgsurf and the DFI. The
        reference workflow avoided this by running the whole hydrology chain on
        a buffered DEM (Makilala_*_Buff_*) and clipping only at the end; for an
        already-clipped DEM, treating the domain boundary as a stream outlet is
        the equivalent - every flow path terminates at either a real stream cell
        or the boundary, so sump is never 0. Marking a cell as a stream also
        short-circuits that branch entirely: dinfdistdown assigns it distance 0
        before it ever inspects its neighbours.

        The ring is taken from the D-infinity flow-direction grid, not the DEM.
        dinfflowdir needs all eight neighbours, so it already NoDatas the DEM's
        outermost ring; seeding that ring would place outlets on cells
        dinfdistdown never routes through and change nothing."""
        import rasterio
        import numpy as np

        routing_path = os.path.join(self.output_dir, "src_routing.tif")
        with rasterio.open(ang_path) as fel_src, rasterio.open(src_path) as src_ds:
            if (fel_src.width, fel_src.height) != (src_ds.width, src_ds.height):
                raise RuntimeError("ang.tif and src.tif are not on the same grid; "
                                   "cannot seed boundary outlets.")
            height, width = src_ds.height, src_ds.width
            meta = src_ds.meta.copy()
            stream_value = 1
            seeded = 0

            with rasterio.open(routing_path, "w", **meta) as dst:
                for row0 in range(0, height, block_rows):
                    row1 = min(row0 + block_rows, height)
                    # One row of halo on each side so neighbours are available
                    # for the cells on the block's own edges.
                    pad0, pad1 = max(row0 - 1, 0), min(row1 + 1, height)
                    halo = rasterio.windows.Window(0, pad0, width, pad1 - pad0)
                    valid = ~np.ma.getmaskarray(fel_src.read(1, window=halo, masked=True))

                    # A cell is interior only when all eight neighbours are inside
                    # the domain; pad with False so the raster border counts as
                    # boundary too.
                    padded = np.zeros((valid.shape[0] + 2, valid.shape[1] + 2), dtype=bool)
                    padded[1:-1, 1:-1] = valid
                    interior = np.ones_like(valid)
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            interior &= padded[1 + dr:1 + dr + valid.shape[0],
                                               1 + dc:1 + dc + valid.shape[1]]
                    ring = valid & ~interior

                    # Trim the halo back to the block's own rows.
                    top = row0 - pad0
                    ring = ring[top:top + (row1 - row0)]

                    block = rasterio.windows.Window(0, row0, width, row1 - row0)
                    data = src_ds.read(1, window=block)
                    data[ring] = stream_value
                    dst.write(data, 1, window=block)
                    seeded += int(ring.sum())

        self.log.emit(f"   > Seeded {seeded} domain-boundary outlet cells into "
                      f"src_routing.tif (prevents dinfdistdown NoData streaks)")
        return routing_path

    def _apply_low_pass(self, raster_paths, block_rows=512):
        """Low-pass filter the given rasters in place-by-copy, returning a dict of
        {key: filtered_path}.

        The shipped ML-DFI model was trained and applied on the Makilala_*_LPF_*
        predictors. Comparing the shipped filtered and unfiltered pairs shows the
        filter is exactly a 3x3 uniform mean over valid cells (correlation 1.0000,
        max abs difference 3.3e-9), so that is what is reproduced here. NoData is
        excluded from each window rather than treated as a value, and cells that
        are NoData in the source stay NoData."""
        import rasterio
        import numpy as np

        k = self.LPF_WINDOW
        pad = k // 2
        filtered = {}
        for key, path in raster_paths.items():
            out_path = os.path.join(self.output_dir,
                                    f"{os.path.splitext(os.path.basename(path))[0]}_lpf.tif")
            with rasterio.open(path) as src:
                meta = src.meta.copy()
                nodata = src.nodata if src.nodata is not None else -9999.0
                meta.update(dtype="float32", nodata=nodata, count=1)
                height, width = src.height, src.width
                with rasterio.open(out_path, "w", **meta) as dst:
                    for row0 in range(0, height, block_rows):
                        row1 = min(row0 + block_rows, height)
                        pad0, pad1 = max(row0 - pad, 0), min(row1 + pad, height)
                        halo = rasterio.windows.Window(0, pad0, width, pad1 - pad0)
                        arr = src.read(1, window=halo, masked=True).astype(np.float64)
                        valid = ~np.ma.getmaskarray(arr)
                        values = np.where(valid, np.ma.getdata(arr), 0.0)

                        # Sum of values and count of valid cells over each 3x3
                        # window, via zero-padded shifts (NoData contributes
                        # nothing to either), so edge and coastline cells are
                        # averaged over the neighbours they actually have.
                        vs = np.zeros((values.shape[0] + 2 * pad,
                                       values.shape[1] + 2 * pad), dtype=np.float64)
                        cs = np.zeros_like(vs)
                        vs[pad:pad + values.shape[0], pad:pad + values.shape[1]] = values
                        cs[pad:pad + values.shape[0], pad:pad + values.shape[1]] = valid
                        total = np.zeros_like(values)
                        count = np.zeros_like(values)
                        for dr in range(-pad, pad + 1):
                            for dc in range(-pad, pad + 1):
                                total += vs[pad + dr:pad + dr + values.shape[0],
                                            pad + dc:pad + dc + values.shape[1]]
                                count += cs[pad + dr:pad + dr + values.shape[0],
                                            pad + dc:pad + dc + values.shape[1]]

                        out = np.full(values.shape, nodata, dtype=np.float32)
                        ok = valid & (count > 0)
                        out[ok] = (total[ok] / count[ok]).astype(np.float32)

                        top = row0 - pad0
                        block = rasterio.windows.Window(0, row0, width, row1 - row0)
                        dst.write(out[top:top + (row1 - row0)], 1, window=block)
            filtered[key] = out_path
        self.log.emit(f"   > Low-pass filtered ({k}x{k} mean) "
                      f"{len(filtered)} continuous predictors to match the training rasters")
        return filtered

    def _generate_twi(self, sca_path, slp_tangent_path, twi_path):
        """Topographic Wetness Index = ln(SCA / tan(slope)), using TauDEM's
        D-infinity specific catchment area and D-infinity slope grid (slp.tif,
        already a tangent), as in the DFI predictor prep guide."""
        self.log.emit("   > Computing Topographic Wetness Index (TWI) from TauDEM SCA and slope...")
        import rasterio
        import numpy as np
        with rasterio.open(sca_path) as sca_src, rasterio.open(slp_tangent_path) as slp_src:
            sca = sca_src.read(1).astype(np.float32)
            tan_slp = slp_src.read(1).astype(np.float32)
            sca_nd, slp_nd = sca_src.nodata, slp_src.nodata
            meta = sca_src.meta.copy()

        valid = np.isfinite(sca) & np.isfinite(tan_slp) & (sca > 0) & (tan_slp >= 0)
        if sca_nd is not None:
            valid &= sca != np.float32(sca_nd)
        if slp_nd is not None:
            valid &= tan_slp != np.float32(slp_nd)

        tan_safe = np.maximum(tan_slp, 0.001)   # avoid division by zero on flats
        twi = np.full(sca.shape, -9999.0, dtype=np.float32)
        twi[valid] = np.log(sca[valid] / tan_safe[valid])

        meta.update(dtype=rasterio.float32, nodata=-9999.0)
        with rasterio.open(twi_path, 'w', **meta) as dst:
            dst.write(twi, 1)

    def _sanitise_bulk_density(self, bulk_density_path, fos_dir, block_rows=1024):
        """Copy the bulk-density raster with non-positive cells declared NoData,
        and return the copy's path (or the original when nothing needed masking).

        A unit weight of 0 makes the pore-pressure term of the FoS equation
        divide by zero, which is how a landscape-wide mean FoS of 0.24 and a
        minimum of -162 arise. Such rasters usually declare some other NoData
        value (32767 in the Cotabato run) while actually using 0 as background."""
        import rasterio
        import numpy as np

        out_path = os.path.join(fos_dir, "bulk_density_masked.tif")
        with rasterio.open(bulk_density_path) as src:
            meta = src.meta.copy()
            nodata = -9999.0
            meta.update(dtype="float32", nodata=nodata, count=1)
            masked_cells = 0
            with rasterio.open(out_path, "w", **meta) as dst:
                for row0 in range(0, src.height, block_rows):
                    row1 = min(row0 + block_rows, src.height)
                    win = rasterio.windows.Window(0, row0, src.width, row1 - row0)
                    arr = src.read(1, window=win, masked=True).astype(np.float64)
                    data = np.ma.filled(arr, nodata).astype(np.float32)
                    bad = ~np.isfinite(data) | (data <= 0)
                    bad &= data != np.float32(nodata)
                    masked_cells += int(bad.sum())
                    data[~np.isfinite(data) | (data <= 0)] = nodata
                    dst.write(data, 1, window=win)

        if masked_cells:
            self.log.emit(f"   > Bulk density: masked {masked_cells} non-positive cells "
                          f"as NoData before the FoS solve")
            return out_path
        os.remove(out_path)
        return bulk_density_path

    def _run_fos_source_refinement(self, clean_env):
        """Run fos_source_refinement.py on the PINN (Objective 2) outputs.

        Returns the refined source-mask path, or None after emitting an error."""
        import rasterio
        import numpy as np

        fi = self.fos_inputs
        script = os.path.join(self.plugin_dir, "ml_dfi_handoff", "steps",
                              "fos_source_refinement", "fos_source_refinement.py")
        if not os.path.exists(script):
            self.error.emit(f"FoS source refinement script not found at:\n{script}")
            return None

        required = {
            "susceptibility": "PINN susceptible-unit raster",
            "cohesion": "PINN cohesion raster",
            "friction_angle": "PINN internal friction angle raster",
            "wetness": "PINN wetness (saturation ratio) raster",
            "bulk_density": "Bulk Density raster (UI input)",
            "slope": "Slope raster (UI input)",
        }
        for key, label in required.items():
            path = fi.get(key)
            if not path or not os.path.exists(path):
                self.error.emit(f"FoS source refinement is missing its {label}:\n{path}")
                return None

        # Unit-weight scale: detect the bulk-density encoding from a decimated sample.
        with rasterio.open(fi["bulk_density"]) as bd:
            sample = bd.read(1, out_shape=(min(1000, bd.height), min(1000, bd.width)), masked=True)
        vals = np.asarray(sample.compressed(), dtype=np.float64)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        median_bd = float(np.median(vals)) if vals.size else float("nan")
        if median_bd > 50:
            bd_scale, bd_note = 0.0981, "cg/cm3 (g/cm3 x 100)"
        elif 0.5 <= median_bd <= 3.0:
            bd_scale, bd_note = 9.81, "g/cm3"
        else:
            bd_scale, bd_note = 1.0, "already kN/m3"
        self.log.emit(f"   > Bulk density median {median_bd:.3f} read as {bd_note}; "
                      f"scale to kN/m3 = {bd_scale}")

        fos_dir = os.path.join(self.output_dir, "fos_source_refinement")
        os.makedirs(fos_dir, exist_ok=True)
        refined_mask = os.path.join(fos_dir, "refined_source_mask.tif")
        summary_path = os.path.join(fos_dir, "source_refinement_summary.json")

        # The unit weight divides the pore-pressure term of the FoS equation, so a
        # single zero-valued cell sends FoS to -inf. Bulk-density rasters commonly
        # carry a 0 background with a different value declared as NoData, and
        # fos_source_refinement.py only sets NoData on the warp destination - the
        # source zeros survive and get blended into real cells. Mask them first.
        bulk_density_path = self._sanitise_bulk_density(fi["bulk_density"], fos_dir)

        # Prefer the PINN's own slope-unit Factor of Safety over re-deriving one
        # from the parameter rasters. The two disagree by roughly six times: the
        # PINN's has a median near 1.4 with about a third of the area above the
        # 1.5 threshold, while the re-derived cell-scale FoS averages 0.24 and
        # puts over 99% below it, which makes the refinement a no-op. The
        # parameter rasters are still passed through for their diagnostics.
        pinn_fos = fi.get("factor_of_safety")
        if pinn_fos and os.path.exists(pinn_fos):
            self.log.emit("   > Refining the source mask with the PINN's own "
                          "FactorOfSafety raster")
        else:
            pinn_fos = None
            self.log.emit("   > No PINN FactorOfSafety raster available; deriving a "
                          "cell-scale FoS from the parameter rasters instead")

        config = {
            "susceptibility_raster": fi["susceptibility"],
            "cohesion_raster": fi["cohesion"],
            "friction_angle_raster": fi["friction_angle"],
            "saturation_ratio_raster": fi["wetness"],
            "total_unit_weight_raster": bulk_density_path,
            "slope_angle_raster": fi["slope"],
            "friction_angle_unit": fi.get("friction_angle_unit", "degrees"),
            "total_unit_weight_band": 1,
            "total_unit_weight_scale_to_kn_m3": bd_scale,
            # Nearest, not bilinear: this grid is typically far coarser than the
            # slope grid it is warped onto (250 m -> 5 m in the Cotabato run), and
            # interpolating across its unit boundaries invents unit weights that
            # were never measured.
            "total_unit_weight_resampling": "nearest",
            "fos_threshold": 1.5,
            "output_fos_raster": os.path.join(fos_dir, "cell_scale_factor_of_safety.tif"),
            "output_source_mask_raster": refined_mask,
            "output_summary_json": summary_path,
            "factor_of_safety_raster": pinn_fos,
            "float_nodata": -9999.0,
            "mask_nodata": 255,
            "block_rows": 256,
            "overwrite": True,
        }
        config_path = os.path.join(fos_dir, "fos_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        proc = subprocess.run([self.python_exe, script, "--config", config_path, "--overwrite"],
                              capture_output=True, text=True, env=clean_env)
        for line in (proc.stdout or "").splitlines():
            if line.strip():
                self.log.emit(f"   > FoS: {line.strip()}")
        if proc.returncode != 0 or not os.path.exists(refined_mask):
            self.error.emit(f"FoS source refinement failed (exit {proc.returncode}):\n"
                            f"{proc.stdout}\n{proc.stderr}")
            return None

        try:
            with open(summary_path, encoding="utf-8") as f:
                counts = json.load(f).get("counts", {})
            self.log.emit(f"   > FoS: {counts.get('susceptible_cells', '?')} susceptible cells | "
                          f"{counts.get('retained_source_cells', '?')} retained (FoS <= 1.5) | "
                          f"{counts.get('stable_cells_excluded_from_susceptible_units', '?')} excluded as stable | "
                          f"{counts.get('unresolved_cells_in_susceptible_units', '?')} unresolved")
        except Exception:
            pass
        return refined_mask

    def run(self):
        import subprocess
        import sys
        import json
        import os
        
        try:
            if not self.taudem_dir:
                self.error.emit("TauDEM not found. Please install TauDEM.")
                return

            os.makedirs(self.output_dir, exist_ok=True)
            fel = os.path.join(self.output_dir, "fel.tif")
            ang = os.path.join(self.output_dir, "ang.tif")
            slp = os.path.join(self.output_dir, "slp.tif")
            alpha = os.path.join(self.output_dir, "BaselineAlpha.tif")
            raw_dfi = os.path.join(self.output_dir, "ml_dfi_probability.tif")
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

            slp_degrees = os.path.join(self.output_dir, "slp_degrees.tif")
            self._convert_slope_tangent_to_degrees(slp, slp_degrees)
            self.progress.emit(20)

            

            # 2. On-The-Fly Predictor Generation
            self.log.emit("\n[Predictors] Dynamically Generating ML-DFI Predictors...")
            
            aligned_ndvi = os.path.join(self.output_dir, "aligned_ndvi.tif")
            self._align_raster(self.ndvi_path, self.dem_path, aligned_ndvi)

            prof_curv = os.path.join(self.output_dir, "profile_curvature.tif")
            plan_curv = os.path.join(self.output_dir, "plan_curvature.tif")
            self._generate_curvatures(self.dem_path, prof_curv, plan_curv)

            weiss_script = os.path.join(self.plugin_dir, "ml-prep-scripts", "weiss_tpi_landform.py")
            if not os.path.exists(weiss_script):
                self.error.emit(f"Weiss TPI script not found at:\n{weiss_script}")
                return
            self.log.emit("   > Computing Weiss slope position (6-class) & Horn slope...")
            weiss_proc = subprocess.run(
                [self.python_exe, weiss_script, self.dem_path, "--out-dir", self.output_dir,
                 "--landform-classification", "6"],
                capture_output=True, text=True, env=clean_env)
            if weiss_proc.returncode != 0:
                self.error.emit(f"Weiss TPI script failed (exit {weiss_proc.returncode}):\n"
                                f"{weiss_proc.stdout}\n{weiss_proc.stderr}")
                return
            for line in weiss_proc.stdout.splitlines():
                if "histogram" in line.lower() or line.strip().startswith(("0 (", "1 (", "2 (", "3 (", "4 (", "5 (", "6 (")):
                    self.log.emit(f"   > Weiss: {line.strip()}")

            slope_deg = os.path.join(self.output_dir, "slope_deg.tif")
            geomorphology = os.path.join(self.output_dir, "Geomorphology.tif")
            for label, path in (("slope_deg.tif", slope_deg), ("Geomorphology.tif", geomorphology)):
                if not os.path.exists(path):
                    self.error.emit(f"Weiss TPI script did not produce {label}.")
                    return

            self.log.emit("   > Computing Catchment Areas & Distance to Stream/Ridge...")
            sca = os.path.join(self.output_dir, "sca.tif")
            src_mask = os.path.join(self.output_dir, "src.tif")
            dstgsurf = os.path.join(self.output_dir, "dstgsurf.tif")
            dugsurf = os.path.join(self.output_dir, "dugsurf.tif")

            def taudem(tool, *args):
                proc = subprocess.run(mpi_base + [os.path.join(self.taudem_dir, tool), *args],
                                      capture_output=True, text=True, env=clean_env)
                if proc.returncode != 0:
                    raise RuntimeError(f"TauDEM {tool} failed (exit {proc.returncode}):\n"
                                       f"{proc.stdout}\n{proc.stderr}")

            # D-infinity specific catchment area (used by TWI).
            # -nc matches the reference lineage (AreaDinf ... false 8): without it
            # TauDEM NoDatas every cell whose upslope area touches the DEM edge.
            taudem("areadinf", "-ang", ang, "-sca", sca, "-nc")

            # Stream Network Analysis -> Stream Definition by Threshold on the
            # D-infinity specific catchment area. The reference predictors were
            # cut against a Strahler stream-order network whose stream-cell
            # density is ~1.8% of the valid domain; a 1500 threshold on SCA gives
            # ~3.7% (twice as dense), which halves dstgsurf. See STREAM_SCA_THRESHOLD.
            taudem("threshold", "-ssa", sca, "-src", src_mask,
                   "-thresh", str(self.STREAM_SCA_THRESHOLD))

            # dinfdistdown NoDatas any cell whose D-infinity flow leaves the
            # analysis domain before reaching a stream cell (the `sump<=0` branch
            # in DinfDistDown.cpp, which -nc does NOT disable), and that NoData
            # then propagates upstream. The reference workflow avoided it by
            # running the whole chain on a buffered DEM and clipping afterwards.
            # Seeding the domain-boundary ring as an outlet is the equivalent for
            # an already-clipped DEM: every flow path now terminates somewhere.
            routing_src = self._seed_boundary_outlets(src_mask, ang)

            # D-Infinity Distance Down to stream, surface flow-path distance (dstgsurf)
            taudem("dinfdistdown", "-ang", ang, "-fel", fel, "-src", routing_src,
                   "-dd", dstgsurf, "-m", "ave", "s", "-nc")
            # D-Infinity Distance Up to ridge, surface flow-path distance (dugsurf).
            # Proportion threshold 0 per the reference lineage
            # (DInfDistanceUp ... 0 Average Surface false); TauDEM's default of 0.5
            # truncates most flow paths immediately and collapses the distances.
            taudem("dinfdistup", "-ang", ang, "-fel", fel, "-slp", slp,
                   "-du", dugsurf, "-m", "ave", "s", "-thresh", "0", "-nc")

            twi = os.path.join(self.output_dir, "twi.tif")
            self._generate_twi(sca, slp, twi)
            self.progress.emit(30)

            # 3. ML-DFI Model Inference
            self.log.emit("\n[ML-DFI] Step 3: Running ML Inference Engine...")
            config_json_path = os.path.join(self.output_dir, "predictor_config.json")
            # Every continuous predictor the model was fitted on is a low-pass
            # filtered raster (see Step2/run_config.json); geomorphology is
            # categorical and is the one predictor used unfiltered.
            lpf = self._apply_low_pass({
                "twi": twi,
                "slope": slope_deg,
                "profile_curvature": prof_curv,
                "plan_curvature": plan_curv,
                "ndvi": aligned_ndvi,
                "dugsurf": dugsurf,
                "dstgsurf": dstgsurf,
            })
            config_data = {
                "predictor_rasters": {
                    "twi": lpf["twi"],
                    "slope": lpf["slope"],
                    "profile_curvature": lpf["profile_curvature"],
                    "plan_curvature": lpf["plan_curvature"],
                    "ndvi": lpf["ndvi"],
                    "geomorphology": geomorphology,
                    "dugsurf": lpf["dugsurf"],
                    "dstgsurf": lpf["dstgsurf"]
                }
            }
            with open(config_json_path, 'w') as f:
                json.dump(config_data, f, indent=4)

            apply_script = os.path.join(self.plugin_dir, "ml_dfi_handoff", "steps", "step3_train_ml_dfi_model", "apply_ml_dfi_model_to_rasters.py")
            model_pkl = os.path.join(self.plugin_dir, "model_dfi", "trained_ml_dfi_model.pkl")
            model_meta = os.path.join(self.plugin_dir, "model_dfi", "model_metadata.json")

            apply_cmd = [
                self.python_exe, apply_script,
                "--model-path", model_pkl,
                "--metadata-path", model_meta,
                "--predictor-config-json", config_json_path,
                "--reference-raster-path", slope_deg,
                "--output-dir", self.output_dir,
                "--output-probability-raster", "ml_dfi_probability.tif",
                "--disable-full-domain-shap"
            ]

            process_apply = subprocess.run(apply_cmd, capture_output=True, text=True, env=clean_env)
            if process_apply.returncode != 0:
                self.error.emit(f"ML-DFI Inference Failed:\n{process_apply.stderr}")
                return

            for line in process_apply.stdout.split('\n'):
                if line.strip():
                    self.log.emit(f"   > ML-DFI: {line.strip()}")
            self.progress.emit(35)

            self._clean_dfi_raster(raw_dfi, dfi_clean)
            self.progress.emit(40)

            # ---------------------------------------------------------
            # INTEGRATION: Call official Step 0 script for Alpha Prior
            # ---------------------------------------------------------
            self.log.emit("\n[Handoff] Step 0: Generating Baseline Alpha Raster...")
            
            step4_dir = os.path.dirname(self.step4_wrapper_py)
            handoff_dir = os.path.dirname(os.path.dirname(step4_dir))
            step0_script = os.path.join(handoff_dir, "steps", "step0_baseline_alpha_angle", "step0_baseline_alpha_angle.py")
            
            if not os.path.exists(step0_script):
                self.error.emit(f"Step 0 script not found at:\n{step0_script}")
                return
                
            step0_cmd = [
                self.python_exe, step0_script,
                "--slope", slp_degrees,
                "--output-alpha", alpha,
                "--nodata", "-9999.0",
                "--overwrite"
            ]
            
            process_step0 = subprocess.run(step0_cmd, capture_output=True, text=True, env=clean_env)
            if process_step0.returncode != 0:
                self.error.emit(f"Step 0 Alpha Generation Failed:\n{process_step0.stderr}")
                return
                
            for line in process_step0.stdout.split('\n'):
                if line.strip():
                    self.log.emit(f"   > Step 0: {line.strip()}")
            self.progress.emit(45)

            # 4b. Objective 2 FoS source refinement (cell-scale source mask)
            self.log.emit("\n[FoS] Refining PINN susceptibility into a cell-scale source mask...")
            refined_source_mask = self._run_fos_source_refinement(clean_env)
            if refined_source_mask is None:
                return
            self.progress.emit(50)

            final_source, final_dfi, final_alpha = self._build_cpp_inputs(
                ang=ang, fel=fel, alpha_tif=alpha, dfi_clean=dfi_clean,
                source_mask_tif=refined_source_mask
            )
            if final_source is None:
                    self.error.emit("Failed to build C++ compliant source and DFI rasters.")
                    return

            # 5. Locate the direct C++ engine
            cpp_exe_mac = os.path.join(step4_dir, "cpp_mpi_port", "build_manual", "step4_deposition_zone_mpi")
            cpp_exe_win = os.path.join(step4_dir, "cpp_mpi_port", "build_manual", "step4_deposition_zone_mpi.exe")
            cpp_exe = cpp_exe_win if sys.platform == "win32" else cpp_exe_mac

            if not os.path.exists(cpp_exe):
                self.error.emit(f"C++ routing executable not found at:\n{cpp_exe}\nPlease ensure it is compiled.")
                return

            # 6. Execute C++/MPI Routing Engine Directly
            self.log.emit("\n--- INITIATING C++/MPI DEPOSITIONAL ROUTING ---")
            cmd = [
                self.mpi_exe, "-n", str(self.cores),
                cpp_exe,
                "--fel", fel,
                "--ang", ang,
                "--source", final_source,  # <--- USING THE SNAPPED SOURCE
                "--dfi", final_dfi,        # <--- USING THE SNAPPED DFI
                "--alpha", final_alpha,          # Alpha was built from 'slp', so it already matches!
                "--out-alpha", os.path.join(self.output_dir, "step4_dynamic_alpha.tif"),
                "--out-beta", os.path.join(self.output_dir, "step4_beta_angle.tif"),
                "--out-dfs", os.path.join(self.output_dir, "step4_dfs.tif"),
                "--out-mask", os.path.join(self.output_dir, "Initial_DepZone.tif"),
                "--out-deposition", output_dep,
                "--threshold", "0.01",
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

            self.progress.emit(85)
            self.log.emit(f"\n✅ Depositional Routing Complete!")

            # 7. Source-weighted D-Infinity contributing area for Step 5 cleanup:
            #    binary source mask as weight grid (same grid as ang.tif),
            #    no edge-contamination check (-nc).
            sca_out = os.path.join(self.output_dir, "source_contributing_area.tif")
            self.log.emit("   > Computing source-weighted D-Infinity contributing area for Step 5...")
            sca_proc = subprocess.run(
                mpi_base + [os.path.join(self.taudem_dir, "areadinf"),
                            "-ang", ang, "-sca", sca_out, "-wg", final_source, "-nc"],
                capture_output=True, text=True, env=clean_env
            )
            if sca_proc.returncode != 0 or not os.path.exists(sca_out):
                self.error.emit(f"Source contributing-area (areadinf) failed:\n{sca_proc.stdout}\n{sca_proc.stderr}")
                return

            self.progress.emit(100)
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
# STEP 5: RUNOUT CLEANUP ONLY WORKER
# ==============================================================================
class Step5CleanupWorker(QThread):
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
            self.log.emit("\n--- INITIATING STEP 5: RUNOUT CLEANUP ---")

            temp_config_path = os.path.join(tempfile.gettempdir(), "step5_cleanup_run_config.json")
            with open(temp_config_path, "w", encoding="utf-8") as f:
                json.dump(self.config_dict, f, indent=2)

            cmd = [self.python_exe, self.step5_script_path, "--config", temp_config_path, "--overwrite"]

            clean_env = os.environ.copy()
            clean_env.pop("PYTHONHOME", None)
            clean_env.pop("PYTHONPATH", None)

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=clean_env)
            for line in iter(process.stdout.readline, ''):
                clean = line.strip()
                if clean:
                    self.log.emit(f"   > Step5 Cleanup: {clean}")
            process.wait()

            if process.returncode != 0:
                self.error.emit(f"Step 5 Cleanup failed with exit code {process.returncode}.")
                return

            output_dir = self.config_dict["runout_cleanup_only"]["output_dir"]
            out_combined = os.path.join(output_dir, "post_depositional_combined_mask.tif")
            self.progress.emit(100)
            self.finished.emit(out_combined, "success")

        except Exception as e:
            import traceback
            self.error.emit(f"Step 5 Cleanup Worker Error:\n{str(e)}\n{traceback.format_exc()}")


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
            self.ui_eil.comboBox_NDVI
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
            self.ui_eil.pushButton_NDVI: self.ui_eil.comboBox_NDVI,
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

        # Explicitly check for NDVI for the ML-DFI step
        if "NDVI" not in selected_rasters:
            self.ui_eil.plainTextEdit_Logs.appendPlainText("[ERROR] NDVI Raster is required for ML-DFI. Please select it in the Parameters.")
            QMessageBox.warning(self, "Missing Input", "The NDVI Raster is required for Depositional Zone modeling.")
            return

        # 🟢 ADD THIS LINE: Safely store the path before QGIS map refreshes wipe the UI
        self.saved_ndvi_path = selected_rasters["NDVI"]

        # Bulk Density and gridded Slope are also needed later by the FoS source refinement.
        missing_fos = [name for key, name in (("BUK", "Bulk Density"), ("Slope", "Slope"))
                       if key not in selected_rasters]
        if missing_fos:
            msg = f"{' and '.join(missing_fos)} raster(s) required for the FoS source refinement."
            self.ui_eil.plainTextEdit_Logs.appendPlainText(f"[ERROR] {msg}")
            QMessageBox.warning(self, "Missing Input", msg)
            return
        self.saved_bulk_density_path = selected_rasters["BUK"]
        self.saved_slope_path = selected_rasters["Slope"]

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
            "comboBox_Slope": "Slope",                             # -> Slope_mean
            "comboBox_BulkDensity": "BUK",                         # -> BUK_mean
            "comboBox_Clay": "Clay",                               # -> Clay_mean
            "comboBox_Sand": "Sand",                               # -> Sand_mean
            "comboBox_Silt": "Silt",                               # -> Silt_mean
            "comboBox_PRC": "Prc",                                 # -> Prc_mean
            "comboBox_SoilThickness": "SoilThc",                   # -> SoilThc_mean
            "comboBox_ContributingFactor": "ContributingFactor",   # -> ContributingFactor_mean
            "comboBox_PGA": "PGA2",                                # -> PGA2_mean 
            "comboBox_SoilType": "type",                           # -> type
            "comboBox_NDVI": "NDVI"                                # -> NDVI_mean (And for ML-DFI)
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
        self.ui_eil.progressBar.setValue(100)

        if status == "success":
            self.ui_eil.plainTextEdit_Logs.appendPlainText(f"\n✅ AI Predictions generated successfully!")

            if self.ui_eil.checkBox_LoadResults.isChecked():
                self.finish_and_load_layer(result_json.get("output"), "EIL AI Predictions")

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
                    self.ui_eil.pushButton_Run.setEnabled(True)
                    return

                # 🟢 REMOVE the two lines that query get_selected_rasters_from_ui() 
                # and REPLACE them with this single line:
                ndvi_path = self.saved_ndvi_path

                self.ui_eil.plainTextEdit_Logs.appendPlainText("\n--- INITIATING DYNAMIC-ALPHA DEPOSITION ROUTING ---")
                
                fos_inputs = {
                    "susceptibility": result_json.get("susceptibility_tif"),
                    # The PINN's own fos_layer. When present, the source mask is
                    # refined with it rather than with a second FoS re-derived
                    # from the parameter rasters.
                    "factor_of_safety": result_json.get("factor_of_safety_tif"),
                    "cohesion": result_json.get("cohesion_tif"),
                    "friction_angle": result_json.get("friction_angle_tif"),
                    "wetness": result_json.get("wetness_tif"),
                    "friction_angle_unit": result_json.get("friction_angle_unit", "degrees"),
                    "bulk_density": self.saved_bulk_density_path,
                    "slope": self.saved_slope_path,
                }

                self.deposition_worker = MLDFIDepositionWorker(
                    dem_path=self.dem_path_for_taudem, 
                    pinn_tif=overall_tif,
                    ndvi_path=ndvi_path,
                    fos_inputs=fos_inputs,
                    plugin_dir=self._base_dir,
                    output_dir=step4_out,
                    step4_wrapper_py=step4_wrapper,
                    python_exe=self.eil_venv_python,
                    cores=4
                )
                self.deposition_worker.log.connect(self.ui_eil.plainTextEdit_Logs.appendPlainText)
                self.deposition_worker.progress.connect(self.ui_eil.progressBar.setValue)
                self.deposition_worker.finished.connect(self.on_deposition_complete)
                self.deposition_worker.error.connect(self.on_slope_units_error)
                self.deposition_worker.start()
            else:
                self.ui_eil.plainTextEdit_Logs.appendPlainText(f"\n🎉 ENTIRE PIPELINE COMPLETE!")
                self.ui_eil.pushButton_Run.setEnabled(True)
        else:
            self.ui_eil.pushButton_Run.setEnabled(True)

    def on_deposition_complete(self, depositional_mask_path, status):
        if status == "success" and os.path.exists(depositional_mask_path):
            self.ui_eil.plainTextEdit_Logs.appendPlainText("✅ Step 4 Depositional Routing Complete!")

            out_dir = self.ui_eil.plainTextEdit_OutputFolder.toPlainText().strip()
            step4_out = os.path.join(out_dir, "deposition_zone_output")

            # ── COPY FINAL RASTERS DIRECTLY TO ROOT OUTPUT FOLDER ──────────────
            import shutil
            main_dep_tif = os.path.join(out_dir, "ML_DFI_Depositional_Zone.tif")
            shutil.copy2(depositional_mask_path, main_dep_tif)

            initial_depzone = os.path.join(step4_out, "Initial_DepZone.tif")
            if os.path.exists(initial_depzone):
                shutil.copy2(initial_depzone, os.path.join(out_dir, "Initial_DepZone.tif"))

            if self.ui_eil.checkBox_LoadResults.isChecked():
                rlayer = QgsRasterLayer(main_dep_tif, "ML-DFI Depositional Zone")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)

            # ── TRIGGER STEP 5: RUNOUT CLEANUP ──────────────────────────────────
            step5_script = os.path.join(
                self._base_dir, "ml_dfi_handoff", "steps",
                "step5_cleanup_only", "step5_cleanup_only.py"
            )
            if not os.path.exists(step5_script):
                self.ui_eil.plainTextEdit_Logs.appendPlainText(f"[Warning] Step 5 script not found at {step5_script}. Pipeline complete through Step 4.")
                self.ui_eil.pushButton_Run.setEnabled(True)
                return

            source_final = os.path.join(step4_out, "source_final.tif")
            sca_path = os.path.join(step4_out, "source_contributing_area.tif")
            ang_path = os.path.join(step4_out, "ang.tif")
            for label, path in (("Initial_DepZone.tif", initial_depzone),
                                ("source_final.tif", source_final),
                                ("source_contributing_area.tif", sca_path),
                                ("ang.tif", ang_path)):
                if not os.path.exists(path):
                    self.ui_eil.plainTextEdit_Logs.appendPlainText(
                        f"[Warning] Missing Step 5 input {label} at {path}. Pipeline complete through Step 4.")
                    self.ui_eil.pushButton_Run.setEnabled(True)
                    return

            step5_out_dir = os.path.join(out_dir, "step5_cleanup_output")
            os.makedirs(step5_out_dir, exist_ok=True)

            step5_config = {
                "runout_cleanup_only": {
                    "runout_mask_raster": initial_depzone,
                    "source_mask_raster": source_final,
                    "source_contributing_area_raster": sca_path,
                    "dinf_flow_raster": ang_path,
                    "output_dir": step5_out_dir,
                    "min_runout_source_area_m2": 200.0,
                    "min_dinf_flow_proportion": 0.01,
                    "edge_seed_buffer_m": 5.0,
                    "overwrite": True
                }
            }

            self.step5_worker = Step5CleanupWorker(step5_script, step5_config, self.eil_venv_python)
            self.step5_worker.log.connect(self.ui_eil.plainTextEdit_Logs.appendPlainText)
            self.step5_worker.progress.connect(self.ui_eil.progressBar.setValue)
            self.step5_worker.finished.connect(self.on_step5_cleanup_complete)
            self.step5_worker.error.connect(self.on_slope_units_error)
            self.step5_worker.start()

    def on_step5_cleanup_complete(self, combined_mask_path, status):
        self.ui_eil.pushButton_Run.setEnabled(True)
        if status == "success" and os.path.exists(combined_mask_path):
            import shutil
            out_dir = self.ui_eil.plainTextEdit_OutputFolder.toPlainText().strip()
            postdep_tif = os.path.join(out_dir, "PostDep_DepZone.tif")
            shutil.copy2(combined_mask_path, postdep_tif)

            self.ui_eil.plainTextEdit_Logs.appendPlainText("\n🎉 PIPELINE (STEPS 1 THROUGH 5) COMPLETE!")
            if self.ui_eil.checkBox_LoadResults.isChecked():
                rlayer = QgsRasterLayer(postdep_tif, "PostDep_DepZone")
                if rlayer.isValid():
                    QgsProject.instance().addMapLayer(rlayer)
                    self.ui_eil.plainTextEdit_Logs.appendPlainText("--> Loaded into map: PostDep_DepZone")

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