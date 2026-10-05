"""Standalone Step 5 cleanup with a minimal fixed edge-seed buffer."""

from __future__ import annotations

import argparse
from collections import deque
import json
import logging
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

for variable in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(variable, None)

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from scipy import ndimage  # noqa: E402

LOGGER = logging.getLogger("step5_cleanup_only")
SECTION = "runout_cleanup_only"
OUTPUT_CLEANED = "post_depositional_cleaned_runout_mask.tif"
OUTPUT_REMOVED = "post_depositional_removed_runout_mask.tif"
OUTPUT_DIRECTIONAL_REMOVED = "post_depositional_directionally_removed_runout_mask.tif"
OUTPUT_EDGE_SEEDS = "post_depositional_edge_seed_mask.tif"
OUTPUT_SPREAD = "post_depositional_spread_mask.tif"
OUTPUT_COMBINED = "post_depositional_combined_mask.tif"
OUTPUT_SUMMARY = "post_depositional_cleanup_summary.json"

# TauDEM D-Infinity neighbor order: E, NE, N, NW, W, SW, S, SE.
DINF_ROW_OFFSETS: tuple[int, ...] = (0, 0, -1, -1, -1, 0, 1, 1, 1)
DINF_COL_OFFSETS: tuple[int, ...] = (0, 1, 1, 0, -1, -1, -1, 0, 1)
DINF_INDEX_BY_OFFSET = {
    (DINF_ROW_OFFSETS[index], DINF_COL_OFFSETS[index]): index
    for index in range(1, 9)
}


@dataclass(frozen=True)
class CleanupParams:
    runout_mask_raster: Path
    source_mask_raster: Path
    source_contributing_area_raster: Path
    dinf_flow_raster: Path
    output_dir: Path
    min_runout_source_area_m2: float = 200.0
    min_dinf_flow_proportion: float = 0.2
    edge_seed_buffer_m: float = 5.0
    overwrite: bool = False


@dataclass(frozen=True)
class CleanupResult:
    cleaned_runout_mask: np.ndarray
    removed_runout_mask: np.ndarray
    area_removed_runout_mask: np.ndarray
    disconnected_runout_mask: np.ndarray
    directionally_removed_runout_mask: np.ndarray
    edge_seed_mask: np.ndarray
    spread_mask: np.ndarray
    combined_depositional_zone_mask: np.ndarray
    original_runout_cells: int
    area_removed_runout_cells: int
    source_disconnected_runout_cells: int
    weak_diagonal_component_count: int
    weak_diagonal_component_cells: int
    directional_entry_accepted_components: int
    directional_entry_rejected_components: int
    directional_fully_retained_components: int
    directional_partially_retained_components: int
    directionally_retained_runout_cells: int
    directionally_removed_runout_cells: int
    disconnected_runout_cells: int
    removed_runout_cells: int
    cleaned_runout_cells: int
    edge_seed_cells: int
    spread_cells_added: int
    spread_cells_overlapping_removed_runout: int
    combined_depositional_zone_cells: int
    cell_area_m2: float
    raw_source_count_threshold: float
    buffer_offsets: tuple[tuple[int, int, float], ...]


def _valid_mask(array: np.ndarray, nodata: float | int | None) -> np.ndarray:
    valid = np.isfinite(array)
    if nodata is None:
        return valid
    if isinstance(nodata, float) and math.isnan(nodata):
        return valid & ~np.isnan(array)
    return valid & (array != nodata)


def _validate_binary(array: np.ndarray, valid: np.ndarray, label: str) -> None:
    values = np.unique(array[valid])
    invalid = values[(values != 0) & (values != 1)]
    if invalid.size:
        raise ValueError(
            f"{label} must contain only 0, 1, or NoData; found {invalid[0]!r}."
        )


def _source_contact_masks(
    cleaned_runout: np.ndarray,
    source_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return orthogonal/overlap anchors and exclusively diagonal anchors."""
    cleaned = np.asarray(cleaned_runout, dtype=bool)
    source = np.asarray(source_mask, dtype=bool)
    orthogonal_structure = np.asarray(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
    )
    all_neighbor_structure = np.ones((3, 3), dtype=np.uint8)
    near_orthogonal = ndimage.binary_dilation(
        source, structure=orthogonal_structure
    )
    near_any = ndimage.binary_dilation(source, structure=all_neighbor_structure)
    orthogonal_or_overlap = cleaned & near_orthogonal
    diagonal_only = cleaned & near_any & ~near_orthogonal
    return orthogonal_or_overlap, diagonal_only


def _build_taudem_aref(dx: float, dy: float) -> tuple[float, ...]:
    """Build TauDEM's D-Infinity direction-angle reference table."""
    aref0 = -math.atan2(float(dy), float(dx))
    aref = [0.0] * 10
    aref[0] = aref0
    aref[1] = 0.0
    aref[2] = -aref0
    aref[3] = 0.5 * math.pi
    aref[4] = math.pi - aref[2]
    aref[5] = math.pi
    aref[6] = math.pi + aref[2]
    aref[7] = 1.5 * math.pi
    aref[8] = 2.0 * math.pi - aref[2]
    aref[9] = 2.0 * math.pi
    return tuple(aref)


def _taudem_flow_proportion(
    angle: float,
    neighbor_index: int,
    aref: tuple[float, ...],
) -> float:
    """Return the TauDEM D-Infinity proportion routed to one neighbor."""
    index = int(neighbor_index)
    adjusted_angle = float(angle)
    if index == 1 and adjusted_angle > math.pi:
        adjusted_angle -= 2.0 * math.pi
    proportion = 0.0
    if aref[index - 1] < adjusted_angle < aref[index + 1]:
        if adjusted_angle > aref[index]:
            proportion = (aref[index + 1] - adjusted_angle) / (
                aref[index + 1] - aref[index]
            )
        else:
            proportion = (adjusted_angle - aref[index - 1]) / (
                aref[index] - aref[index - 1]
            )
    return float(proportion) if proportion >= 1e-5 else 0.0


def _source_flows_to_anchor(
    *,
    anchor_row: int,
    anchor_col: int,
    source_mask: np.ndarray,
    dinf_flow: np.ndarray,
    dinf_valid: np.ndarray,
    aref: tuple[float, ...],
    min_flow_proportion: float,
) -> bool:
    """Check whether any diagonally adjacent source routes into the anchor."""
    rows, cols = source_mask.shape
    for source_row_offset in (-1, 1):
        for source_col_offset in (-1, 1):
            source_row = anchor_row + source_row_offset
            source_col = anchor_col + source_col_offset
            if not (0 <= source_row < rows and 0 <= source_col < cols):
                continue
            if not source_mask[source_row, source_col] or not dinf_valid[
                source_row, source_col
            ]:
                continue
            target_offset = (-source_row_offset, -source_col_offset)
            neighbor_index = DINF_INDEX_BY_OFFSET[target_offset]
            proportion = _taudem_flow_proportion(
                float(dinf_flow[source_row, source_col]), neighbor_index, aref
            )
            if proportion >= min_flow_proportion:
                return True
    return False


def _trace_component_downstream(
    *,
    component_label: int,
    anchor_row: int,
    anchor_col: int,
    labels: np.ndarray,
    dinf_flow: np.ndarray,
    dinf_valid: np.ndarray,
    aref: tuple[float, ...],
    min_flow_proportion: float,
) -> set[tuple[int, int]]:
    """Trace D-Infinity recipients from one validated anchor within its component."""
    rows, cols = labels.shape
    connected: set[tuple[int, int]] = {(anchor_row, anchor_col)}
    pending: deque[tuple[int, int]] = deque([(anchor_row, anchor_col)])
    while pending:
        row, col = pending.popleft()
        if not dinf_valid[row, col]:
            continue
        angle = float(dinf_flow[row, col])
        for neighbor_index in range(1, 9):
            proportion = _taudem_flow_proportion(angle, neighbor_index, aref)
            if proportion < min_flow_proportion:
                continue
            next_row = row + DINF_ROW_OFFSETS[neighbor_index]
            next_col = col + DINF_COL_OFFSETS[neighbor_index]
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                continue
            next_cell = (next_row, next_col)
            if (
                labels[next_row, next_col] == component_label
                and next_cell not in connected
            ):
                connected.add(next_cell)
                pending.append(next_cell)
    return connected


def _build_edge_seed_mask(
    *,
    cleaned_runout: np.ndarray,
    source_mask: np.ndarray,
    valid_domain: np.ndarray,
) -> np.ndarray:
    """Select non-source cleaned-runout cells touching valid outside cells."""
    cleaned = np.asarray(cleaned_runout, dtype=bool)
    source = np.asarray(source_mask, dtype=bool)
    valid = np.asarray(valid_domain, dtype=bool)
    candidate_seeds = cleaned & ~source
    seeds = np.zeros_like(cleaned)
    rows, cols = cleaned.shape
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if row_offset == 0 and col_offset == 0:
                continue
            seed_rows = slice(max(0, -row_offset), min(rows, rows - row_offset))
            seed_cols = slice(max(0, -col_offset), min(cols, cols - col_offset))
            neighbor_rows = slice(max(0, row_offset), min(rows, rows + row_offset))
            neighbor_cols = slice(max(0, col_offset), min(cols, cols + col_offset))
            outside_runout = (
                valid[neighbor_rows, neighbor_cols]
                & ~cleaned[neighbor_rows, neighbor_cols]
            )
            seeds[seed_rows, seed_cols] |= (
                candidate_seeds[seed_rows, seed_cols] & outside_runout
            )
    return seeds


def _compute_buffer_offsets(
    *,
    cell_size_x_m: float,
    cell_size_y_m: float,
    buffer_distance_m: float,
) -> tuple[tuple[int, int, float], ...]:
    """Return nonzero cell-center offsets within the Euclidean buffer radius."""
    if buffer_distance_m <= 0:
        return ()
    max_row_offset = int(math.floor(buffer_distance_m / cell_size_y_m))
    max_col_offset = int(math.floor(buffer_distance_m / cell_size_x_m))
    offsets: list[tuple[int, int, float]] = []
    tolerance = 1e-9
    for row_offset in range(-max_row_offset, max_row_offset + 1):
        for col_offset in range(-max_col_offset, max_col_offset + 1):
            if row_offset == 0 and col_offset == 0:
                continue
            distance = math.hypot(
                row_offset * cell_size_y_m,
                col_offset * cell_size_x_m,
            )
            if distance <= buffer_distance_m + tolerance:
                offsets.append((row_offset, col_offset, float(distance)))
    return tuple(sorted(offsets, key=lambda item: (item[2], item[0], item[1])))


def _buffer_edge_seeds(
    *,
    edge_seed_mask: np.ndarray,
    valid_domain: np.ndarray,
    buffer_offsets: Sequence[tuple[int, int, float]],
) -> np.ndarray:
    """Expand edge seeds to valid cells at the configured center distance."""
    seeds = np.asarray(edge_seed_mask, dtype=bool)
    valid = np.asarray(valid_domain, dtype=bool)
    buffered = np.zeros_like(seeds)
    rows, cols = seeds.shape
    for row_offset, col_offset, _distance in buffer_offsets:
        source_rows = slice(max(0, -row_offset), min(rows, rows - row_offset))
        source_cols = slice(max(0, -col_offset), min(cols, cols - col_offset))
        target_rows = slice(max(0, row_offset), min(rows, rows + row_offset))
        target_cols = slice(max(0, col_offset), min(cols, cols + col_offset))
        buffered[target_rows, target_cols] |= (
            seeds[source_rows, source_cols] & valid[target_rows, target_cols]
        )
    return buffered


def cleanup_runout_arrays(
    *,
    runout_array: np.ndarray,
    runout_nodata: float | int | None,
    source_mask_array: np.ndarray,
    source_mask_nodata: float | int | None,
    source_contributing_area_array: np.ndarray,
    source_contributing_area_nodata: float | int | None,
    dinf_flow_array: np.ndarray,
    dinf_flow_nodata: float | int | None,
    cell_area_m2: float,
    cell_size_x_m: float,
    cell_size_y_m: float,
    min_runout_source_area_m2: float,
    min_dinf_flow_proportion: float,
    edge_seed_buffer_m: float,
) -> CleanupResult:
    """Apply cleanup rules and a fixed buffer around simplified edge seeds."""
    runout = np.asarray(runout_array)
    source = np.asarray(source_mask_array)
    contributing = np.asarray(source_contributing_area_array)
    dinf_flow = np.asarray(dinf_flow_array)
    if any(
        array.shape != runout.shape
        for array in (source, contributing, dinf_flow)
    ):
        raise ValueError(
            "Runout, source, source-contributing-area, and D-Infinity arrays "
            "must align."
        )
    if not math.isfinite(cell_area_m2) or cell_area_m2 <= 0:
        raise ValueError("cell_area_m2 must be finite and greater than zero.")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (cell_size_x_m, cell_size_y_m)
    ):
        raise ValueError("Cell dimensions must be finite and greater than zero.")
    if (
        not math.isfinite(min_runout_source_area_m2)
        or min_runout_source_area_m2 < 0
    ):
        raise ValueError(
            "min_runout_source_area_m2 must be finite and greater than or equal to zero."
        )
    if (
        not math.isfinite(min_dinf_flow_proportion)
        or min_dinf_flow_proportion <= 0
        or min_dinf_flow_proportion > 1
    ):
        raise ValueError("min_dinf_flow_proportion must be within (0, 1].")
    if not math.isfinite(edge_seed_buffer_m) or edge_seed_buffer_m < 0:
        raise ValueError("edge_seed_buffer_m must be finite and non-negative.")

    runout_valid = _valid_mask(runout, runout_nodata)
    source_valid = _valid_mask(source, source_mask_nodata)
    contribution_valid = _valid_mask(
        contributing, source_contributing_area_nodata
    )
    dinf_valid = _valid_mask(dinf_flow, dinf_flow_nodata)
    _validate_binary(runout, runout_valid, "runout_mask_raster")
    _validate_binary(source, source_valid, "source_mask_raster")
    valid_contributions = contributing[contribution_valid]
    if np.any(valid_contributions < 0):
        raise ValueError(
            "source_contributing_area_raster must be non-negative or NoData."
        )
    valid_angles = dinf_flow[dinf_valid]
    if valid_angles.size == 0:
        raise ValueError("dinf_flow_raster contains no valid cells.")
    angle_min = float(np.min(valid_angles))
    angle_max = float(np.max(valid_angles))
    if angle_min < -1e-6 or angle_max > 2.0 * math.pi + 1e-3:
        raise ValueError(
            "dinf_flow_raster must contain TauDEM directions in radians within "
            f"[0, 2*pi]; observed [{angle_min:.6f}, {angle_max:.6f}]."
        )

    original = runout_valid & (runout > 0)
    missing_at_runout = original & ~contribution_valid
    missing_count = int(np.count_nonzero(missing_at_runout))
    if missing_count:
        raise ValueError(
            "source_contributing_area_raster has NoData or non-finite values at "
            f"{missing_count} runout cells."
        )

    raw_threshold = float(min_runout_source_area_m2) / float(cell_area_m2)
    area_removed = original & (contributing < raw_threshold)
    area_cleaned = original & ~area_removed

    source_bool = source_valid & (source > 0)
    orthogonal_anchors, diagonal_only_anchors = _source_contact_masks(
        area_cleaned, source_bool
    )

    source_disconnected = np.zeros_like(area_cleaned)
    directionally_removed = np.zeros_like(area_cleaned)
    connected = np.zeros_like(area_cleaned)
    weak_component_count = 0
    weak_component_cells = 0
    entry_accepted = 0
    entry_rejected = 0
    fully_retained = 0
    partially_retained = 0
    directionally_retained_cells = 0

    if np.any(area_cleaned):
        labels, component_count = ndimage.label(
            np.ascontiguousarray(area_cleaned, dtype=np.uint8),
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        component_sizes = np.bincount(labels.ravel(), minlength=component_count + 1)
        orthogonal_counts = np.bincount(
            labels[orthogonal_anchors], minlength=component_count + 1
        )
        diagonal_counts = np.bincount(
            labels[diagonal_only_anchors], minlength=component_count + 1
        )
        strong_flags = (orthogonal_counts > 0) | (diagonal_counts >= 2)
        weak_flags = (orthogonal_counts == 0) & (diagonal_counts == 1)
        no_anchor_flags = (orthogonal_counts == 0) & (diagonal_counts == 0)
        strong_flags[0] = False
        weak_flags[0] = False
        no_anchor_flags[0] = False

        connected = strong_flags[labels]
        source_disconnected = area_cleaned & no_anchor_flags[labels]
        weak_labels = np.flatnonzero(weak_flags)
        weak_component_count = int(weak_labels.size)
        weak_component_cells = int(component_sizes[weak_labels].sum())
        aref = _build_taudem_aref(cell_size_x_m, cell_size_y_m)

        for anchor_row, anchor_col in np.argwhere(diagonal_only_anchors):
            row = int(anchor_row)
            col = int(anchor_col)
            component_label = int(labels[row, col])
            if component_label == 0 or not weak_flags[component_label]:
                continue
            entry_is_valid = _source_flows_to_anchor(
                anchor_row=row,
                anchor_col=col,
                source_mask=source_bool,
                dinf_flow=dinf_flow,
                dinf_valid=dinf_valid,
                aref=aref,
                min_flow_proportion=min_dinf_flow_proportion,
            )
            if not entry_is_valid:
                entry_rejected += 1
                continue

            entry_accepted += 1
            routed_cells = _trace_component_downstream(
                component_label=component_label,
                anchor_row=row,
                anchor_col=col,
                labels=labels,
                dinf_flow=dinf_flow,
                dinf_valid=dinf_valid,
                aref=aref,
                min_flow_proportion=min_dinf_flow_proportion,
            )
            routed_rows, routed_cols = zip(*routed_cells)
            connected[np.asarray(routed_rows), np.asarray(routed_cols)] = True
            retained_count = len(routed_cells)
            directionally_retained_cells += retained_count
            if retained_count == int(component_sizes[component_label]):
                fully_retained += 1
            else:
                partially_retained += 1

        directionally_removed = area_cleaned & weak_flags[labels] & ~connected

    disconnected = source_disconnected | directionally_removed
    removed = area_removed | disconnected
    edge_seeds = _build_edge_seed_mask(
        cleaned_runout=connected,
        source_mask=source_bool,
        valid_domain=runout_valid,
    )
    buffer_offsets = _compute_buffer_offsets(
        cell_size_x_m=cell_size_x_m,
        cell_size_y_m=cell_size_y_m,
        buffer_distance_m=edge_seed_buffer_m,
    )
    buffered_cells = _buffer_edge_seeds(
        edge_seed_mask=edge_seeds,
        valid_domain=runout_valid,
        buffer_offsets=buffer_offsets,
    )
    # The conservative envelope may cover rejected runout, but never source cells.
    spread = buffered_cells & ~connected & ~source_bool
    combined = connected | spread
    return CleanupResult(
        cleaned_runout_mask=np.asarray(connected, dtype=np.uint8),
        removed_runout_mask=np.asarray(removed, dtype=np.uint8),
        area_removed_runout_mask=np.asarray(area_removed, dtype=np.uint8),
        disconnected_runout_mask=np.asarray(disconnected, dtype=np.uint8),
        directionally_removed_runout_mask=np.asarray(
            directionally_removed, dtype=np.uint8
        ),
        edge_seed_mask=np.asarray(edge_seeds, dtype=np.uint8),
        spread_mask=np.asarray(spread, dtype=np.uint8),
        combined_depositional_zone_mask=np.asarray(combined, dtype=np.uint8),
        original_runout_cells=int(np.count_nonzero(original)),
        area_removed_runout_cells=int(np.count_nonzero(area_removed)),
        source_disconnected_runout_cells=int(np.count_nonzero(source_disconnected)),
        weak_diagonal_component_count=weak_component_count,
        weak_diagonal_component_cells=weak_component_cells,
        directional_entry_accepted_components=entry_accepted,
        directional_entry_rejected_components=entry_rejected,
        directional_fully_retained_components=fully_retained,
        directional_partially_retained_components=partially_retained,
        directionally_retained_runout_cells=directionally_retained_cells,
        directionally_removed_runout_cells=int(
            np.count_nonzero(directionally_removed)
        ),
        disconnected_runout_cells=int(np.count_nonzero(disconnected)),
        removed_runout_cells=int(np.count_nonzero(removed)),
        cleaned_runout_cells=int(np.count_nonzero(connected)),
        edge_seed_cells=int(np.count_nonzero(edge_seeds)),
        spread_cells_added=int(np.count_nonzero(spread)),
        spread_cells_overlapping_removed_runout=int(
            np.count_nonzero(spread & removed)
        ),
        combined_depositional_zone_cells=int(np.count_nonzero(combined)),
        cell_area_m2=float(cell_area_m2),
        raw_source_count_threshold=float(raw_threshold),
        buffer_offsets=buffer_offsets,
    )


def _resolve(base: Path, value: object, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Config field '{field}' must be a non-empty path.")
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(config_path: Path) -> CleanupParams:
    path = config_path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    config = payload.get(SECTION, payload)
    if not isinstance(config, dict):
        raise ValueError(f"Config section '{SECTION}' must be an object.")
    allowed = {
        "runout_mask_raster",
        "source_mask_raster",
        "source_contributing_area_raster",
        "dinf_flow_raster",
        "output_dir",
        "min_runout_source_area_m2",
        "min_dinf_flow_proportion",
        "edge_seed_buffer_m",
        "overwrite",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    return CleanupParams(
        runout_mask_raster=_resolve(
            path.parent, config.get("runout_mask_raster"), "runout_mask_raster"
        ),
        source_mask_raster=_resolve(
            path.parent, config.get("source_mask_raster"), "source_mask_raster"
        ),
        source_contributing_area_raster=_resolve(
            path.parent,
            config.get("source_contributing_area_raster"),
            "source_contributing_area_raster",
        ),
        dinf_flow_raster=_resolve(
            path.parent, config.get("dinf_flow_raster"), "dinf_flow_raster"
        ),
        output_dir=_resolve(path.parent, config.get("output_dir"), "output_dir"),
        min_runout_source_area_m2=float(
            config.get("min_runout_source_area_m2", 200.0)
        ),
        min_dinf_flow_proportion=float(
            config.get("min_dinf_flow_proportion", 0.2)
        ),
        edge_seed_buffer_m=float(config.get("edge_seed_buffer_m", 5.0)),
        overwrite=bool(config.get("overwrite", False)),
    )


def _output_paths(params: CleanupParams) -> dict[str, Path]:
    return {
        "cleaned_runout_mask": params.output_dir / OUTPUT_CLEANED,
        "removed_runout_mask": params.output_dir / OUTPUT_REMOVED,
        "directionally_removed_runout_mask": (
            params.output_dir / OUTPUT_DIRECTIONAL_REMOVED
        ),
        "edge_seed_mask": params.output_dir / OUTPUT_EDGE_SEEDS,
        "spread_mask": params.output_dir / OUTPUT_SPREAD,
        "combined_depositional_zone_mask": params.output_dir / OUTPUT_COMBINED,
        "summary_json": params.output_dir / OUTPUT_SUMMARY,
    }


def _same_crs(first: rasterio.crs.CRS, second: rasterio.crs.CRS) -> bool:
    if first == second:
        return True
    return (
        first.to_authority() is not None
        and first.to_authority() == second.to_authority()
    )


def _validate_grid(
    reference: rasterio.io.DatasetReader,
    other: rasterio.io.DatasetReader,
    label: str,
) -> None:
    if (
        reference.shape != other.shape
        or reference.transform != other.transform
        or reference.crs is None
        or other.crs is None
        or not _same_crs(reference.crs, other.crs)
    ):
        raise ValueError(f"{label} is not exactly aligned with the runout raster.")


def _write_mask(path: Path, reference: rasterio.io.DatasetReader, mask: np.ndarray) -> None:
    profile = reference.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="LZW",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    temporary = path.with_name(
        f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}"
    )
    try:
        with rasterio.open(temporary, "w", **profile) as destination:
            destination.write(mask, 1)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_cleanup(params: CleanupParams) -> dict[str, Any]:
    started = time.perf_counter()
    for path in (
        params.runout_mask_raster,
        params.source_mask_raster,
        params.source_contributing_area_raster,
        params.dinf_flow_raster,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    outputs = _output_paths(params)
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not params.overwrite:
        raise FileExistsError(
            "Cleanup outputs already exist; pass --overwrite to replace them: "
            + ", ".join(map(str, existing))
        )
    if len({path.resolve() for path in outputs.values()}) != len(outputs):
        raise ValueError("Cleanup output paths must be distinct.")
    params.output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(params.runout_mask_raster) as runout_dataset, rasterio.open(
        params.source_mask_raster
    ) as source_dataset, rasterio.open(
        params.source_contributing_area_raster
    ) as contribution_dataset, rasterio.open(
        params.dinf_flow_raster
    ) as dinf_dataset:
        _validate_grid(runout_dataset, source_dataset, "source_mask_raster")
        _validate_grid(
            runout_dataset,
            contribution_dataset,
            "source_contributing_area_raster",
        )
        _validate_grid(runout_dataset, dinf_dataset, "dinf_flow_raster")
        if runout_dataset.count != 1:
            raise ValueError("runout_mask_raster must be single-band.")
        if any(
            dataset.count != 1
            for dataset in (source_dataset, contribution_dataset, dinf_dataset)
        ):
            raise ValueError("Source and D-Infinity inputs must be single-band.")
        if runout_dataset.crs is None or not runout_dataset.crs.is_projected:
            raise ValueError("Runout raster must use a projected CRS.")
        transform = runout_dataset.transform
        if transform.b != 0 or transform.d != 0:
            raise ValueError("Rotated raster grids are not supported.")
        cell_area = abs(float(transform.a * transform.e))
        cell_size_x = abs(float(transform.a))
        cell_size_y = abs(float(transform.e))

        result = cleanup_runout_arrays(
            runout_array=runout_dataset.read(1),
            runout_nodata=runout_dataset.nodata,
            source_mask_array=source_dataset.read(1),
            source_mask_nodata=source_dataset.nodata,
            source_contributing_area_array=contribution_dataset.read(1),
            source_contributing_area_nodata=contribution_dataset.nodata,
            dinf_flow_array=dinf_dataset.read(1),
            dinf_flow_nodata=dinf_dataset.nodata,
            cell_area_m2=cell_area,
            cell_size_x_m=cell_size_x,
            cell_size_y_m=cell_size_y,
            min_runout_source_area_m2=params.min_runout_source_area_m2,
            min_dinf_flow_proportion=params.min_dinf_flow_proportion,
            edge_seed_buffer_m=params.edge_seed_buffer_m,
        )
        _write_mask(
            outputs["cleaned_runout_mask"],
            runout_dataset,
            result.cleaned_runout_mask,
        )
        _write_mask(
            outputs["removed_runout_mask"],
            runout_dataset,
            result.removed_runout_mask,
        )
        _write_mask(
            outputs["directionally_removed_runout_mask"],
            runout_dataset,
            result.directionally_removed_runout_mask,
        )
        _write_mask(
            outputs["edge_seed_mask"],
            runout_dataset,
            result.edge_seed_mask,
        )
        _write_mask(
            outputs["spread_mask"],
            runout_dataset,
            result.spread_mask,
        )
        _write_mask(
            outputs["combined_depositional_zone_mask"],
            runout_dataset,
            result.combined_depositional_zone_mask,
        )
        raster = {
            "shape": list(runout_dataset.shape),
            "transform": list(runout_dataset.transform)[:6],
            "resolution": list(runout_dataset.res),
            "crs": str(runout_dataset.crs),
            "crs_authority": runout_dataset.crs.to_authority(),
        }

    removed_percent = (
        100.0 * result.removed_runout_cells / result.original_runout_cells
        if result.original_runout_cells
        else None
    )
    summary: dict[str, Any] = {
        "tool": "step5_cleanup_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "runout_mask_raster": str(params.runout_mask_raster),
            "source_mask_raster": str(params.source_mask_raster),
            "source_contributing_area_raster": str(
                params.source_contributing_area_raster
            ),
            "dinf_flow_raster": str(params.dinf_flow_raster),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
        "parameters": {
            "min_runout_source_area_m2": params.min_runout_source_area_m2,
            "min_dinf_flow_proportion": params.min_dinf_flow_proportion,
            "edge_seed_buffer_m": params.edge_seed_buffer_m,
            "connectivity": 8,
            "source_adjacency": 8,
            "edge_seed_adjacency": 8,
            "buffer_distance_method": "cell-center Euclidean distance",
            "directional_validation_scope": (
                "components with exactly one diagonal-only source anchor"
            ),
        },
        "raster": raster,
        "statistics": {
            "cell_area_m2": result.cell_area_m2,
            "raw_source_count_threshold": result.raw_source_count_threshold,
            "original_runout_cells": result.original_runout_cells,
            "area_removed_runout_cells": result.area_removed_runout_cells,
            "source_disconnected_runout_cells": (
                result.source_disconnected_runout_cells
            ),
            "weak_diagonal_component_count": (
                result.weak_diagonal_component_count
            ),
            "weak_diagonal_component_cells": result.weak_diagonal_component_cells,
            "directional_entry_accepted_components": (
                result.directional_entry_accepted_components
            ),
            "directional_entry_rejected_components": (
                result.directional_entry_rejected_components
            ),
            "directional_fully_retained_components": (
                result.directional_fully_retained_components
            ),
            "directional_partially_retained_components": (
                result.directional_partially_retained_components
            ),
            "directionally_retained_runout_cells": (
                result.directionally_retained_runout_cells
            ),
            "directionally_removed_runout_cells": (
                result.directionally_removed_runout_cells
            ),
            "disconnected_runout_cells": result.disconnected_runout_cells,
            "removed_runout_cells": result.removed_runout_cells,
            "removed_runout_percent": removed_percent,
            "cleaned_runout_cells": result.cleaned_runout_cells,
            "edge_seed_cells": result.edge_seed_cells,
            "buffer_neighbor_offset_count": len(result.buffer_offsets),
            "buffer_neighbor_offsets": [
                [row_offset, col_offset, distance]
                for row_offset, col_offset, distance in result.buffer_offsets
            ],
            "spread_cells_added": result.spread_cells_added,
            "spread_cells_overlapping_removed_runout": (
                result.spread_cells_overlapping_removed_runout
            ),
            "spread_area_added_m2": (
                result.spread_cells_added * result.cell_area_m2
            ),
            "combined_depositional_zone_cells": (
                result.combined_depositional_zone_cells
            ),
            "combined_depositional_zone_area_m2": (
                result.combined_depositional_zone_cells * result.cell_area_m2
            ),
        },
        "rules": [
            "Remove runout cells with converted contributing source area below the configured threshold.",
            "Remove retained 8-connected components that have no source overlap or 8-neighbor source contact.",
            "Keep components with source overlap, orthogonal source contact, or at least two diagonal-only source anchors.",
            "For a component with exactly one diagonal-only source anchor, require a qualifying D-Infinity branch from that source into the anchor and retain only component cells reachable downstream from the anchor.",
            "Select cleaned-runout edge cells as seeds when they are not source cells and touch at least one valid outside-runout cell in an 8-neighbor window.",
            "Apply a fixed cell-center Euclidean buffer around every edge seed and add valid non-source cells to the final depositional-zone mask; rejected runout may be covered by this separate conservative envelope but remains absent from the cleaned-runout mask.",
        ],
        "not_applied": [
            "No variable-radius or terrain-conditioned post-depositional spreading.",
            "No DFI, DEM, stream, alpha-angle, beta-angle, or additional runout-length filtering.",
        ],
        "runtime_seconds": time.perf_counter() - started,
    }
    temporary_summary = outputs["summary_json"].with_name(
        f".{outputs['summary_json'].stem}.{uuid.uuid4().hex}.tmp.json"
    )
    try:
        temporary_summary.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        os.replace(temporary_summary, outputs["summary_json"])
    finally:
        temporary_summary.unlink(missing_ok=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Clean a Step 4 runout mask using contributing-source-area and "
            "targeted D-Infinity source-connectivity rules, then apply a "
            "minimal fixed buffer around non-source edge seeds."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.default.json"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing cleanup-only outputs.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        params = load_config(args.config)
        if args.overwrite:
            params = replace(params, overwrite=True)
        summary = run_cleanup(params)
    except Exception as exc:
        LOGGER.exception("Cleanup failed: %s", exc)
        return 1
    statistics = summary["statistics"]
    LOGGER.info(
        "Cleanup complete: %d original, %d removed, %d retained cells",
        statistics["original_runout_cells"],
        statistics["removed_runout_cells"],
        statistics["cleaned_runout_cells"],
    )
    LOGGER.info(
        "Fixed buffer: %d edge seeds, %d added cells, %d combined cells",
        statistics["edge_seed_cells"],
        statistics["spread_cells_added"],
        statistics["combined_depositional_zone_cells"],
    )
    LOGGER.info("Outputs: %s", Path(summary["outputs"]["summary_json"]).parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
