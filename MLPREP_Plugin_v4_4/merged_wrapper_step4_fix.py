# =============================================================================
# FIX FOR merged_wrapper.py -- Step 4 C++/MPI input construction
# =============================================================================
#
# WHAT THIS CHANGES AND WHY
# -------------------------
# step4_deposition_zone_mpi.cpp counts a cell as an "initiating source cell"
# ONLY when ALL of these hold at that same pixel:
#
#   1. fel.tif and ang.tif are both non-NoData there       (terrain)
#   2. source raster is non-NoData and > 0                 (marked)
#   3. source value is EXACTLY 0.0 or 1.0 (tol 1e-7)       (source_is_binary)
#   4. alpha.tif is non-NoData there                       (alpha_is_valid)
#   5. alpha is finite and within [1, 90]                  (alpha_is_valid)
#   6. alpha is within [6, 72]                             (kFixedAlphaMin/Max)
#
# Requirement 4 is the one nothing in the pipeline was enforcing, and when it
# fails the engine increments local_source_without_valid_alpha -- a counter it
# computes, reduces across ranks, writes into its stats JSON, and then never
# prints in the failure block. That is why "Initiating source cells: 0" came
# back with every other printed counter reading 0, run after run.
#
# The engine also aborts if ANY terrain cell anywhere has a non-NoData alpha
# outside [1, 90], so this code neutralises those by rewriting them as NoData
# (NoData cells are skipped wholesale by the engine; it is the only correction
# that does not invent a physical value).
#
# ORDERING MATTERS: source cells can only be placed where alpha is valid, so
# the source raster must now be built AFTER Step 0 writes alpha.tif -- not
# before it, as the current code does.
#
#
# HOW TO APPLY (three edits to merged_wrapper.py)
# ----------------------------------------------
# EDIT 1 -- In MLDFIDepositionWorker.run(), find the "# 4. Clean DFI and
#           Generate Source" block (everything from that comment down to and
#           including "self.progress.emit(40)") and replace the WHOLE block
#           with just these three lines:
#
#                 # 4. Clean the DFI raster only. The C++ inputs are built
#                 #    after Step 0, because they depend on alpha.tif.
#                 self._clean_dfi_raster(raw_dfi, dfi_clean)
#                 self.progress.emit(40)
#
# EDIT 2 -- Immediately after the Step 0 block's "self.progress.emit(45)"
#           line, insert:
#
#                 final_source, final_dfi, final_alpha = self._build_cpp_inputs(
#                     ang=ang, fel=fel, alpha_tif=alpha, dfi_clean=dfi_clean
#                 )
#                 if final_source is None:
#                     return
#
# EDIT 3 -- In the "cmd = [" list that launches the C++ engine, change the
#           alpha argument from:
#                 "--alpha", alpha,
#           to:
#                 "--alpha", final_alpha,
#           (leave --source final_source and --dfi final_dfi as they are)
#
# Then paste the _build_cpp_inputs method below into the MLDFIDepositionWorker
# class, at the same indentation as its other methods.
# =============================================================================


    def _build_cpp_inputs(self, ang, fel, alpha_tif, dfi_clean):
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
        with rasterio.open(self.pinn_tif) as pinn:
            pinn_data = np.full(t_shape, OUT_NODATA, dtype=np.float32)
            reproject(source=rasterio.band(pinn, 1), destination=pinn_data,
                      src_transform=pinn.transform, src_crs=pinn.crs,
                      dst_transform=t_transform, dst_crs=t_crs,
                      src_nodata=pinn.nodata, dst_nodata=OUT_NODATA,
                      resampling=Resampling.nearest)

        is_source = valid_terrain & (pinn_data > 0.01) & \
                    (pinn_data != np.float32(OUT_NODATA)) & np.isfinite(pinn_data)
        n_raw = int(is_source.sum())
        legal = is_source & alpha_usable
        n_legal = int(legal.sum())
        del pinn_data

        self.log.emit(f"   > PINN source candidates: {n_raw} | "
                      f"accepted by the engine's alpha rule: {n_legal} | "
                      f"discarded for missing/out-of-range alpha: {n_raw - n_legal}")

        if n_legal == 0:
            self.error.emit(
                "No legal Step 4 source cell exists.\n\n"
                f"The PINN model marked {n_raw} candidate cells, but alpha.tif has no "
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
