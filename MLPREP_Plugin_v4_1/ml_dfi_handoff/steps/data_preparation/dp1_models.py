"""Typed containers for the pure Step 1 observed-alpha core.

Plain-English map of this file:
1. Define small geometry/data containers used by the Step 1 path engine.
2. Store the user-facing Step 1 calculation settings.
3. Store one polygon's inputs and outputs in predictable shapes.

These classes do not run GIS tools. They keep the crown/toe/path data organized
while the Step 1 engine calculates observed alpha.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


XY = tuple[float, float]


# ---------------------------------------------------------------------------
# Block 1: Contour and configuration inputs
# ---------------------------------------------------------------------------
# A ContourNode is one simplified contour/path candidate. Step1Config holds the
# scientific and QC settings used by the pure path engine.
@dataclass(frozen=True)
class ContourNode:
    xy: XY
    elevation_m: float
    source_length_2d_m: float = 0.0
    segment_xy: tuple[XY, ...] = field(default_factory=lambda: ())


@dataclass(frozen=True)
class Step1Config:
    contour_interval_m: float = 2.0
    buffer_min_m: float = 10.0
    buffer_pixel_multiple: float = 2.0
    boundary_sample_step_m: float = 10.0
    crown_percentile_high: float = 90.0
    qc_percentile_high: float = 90.0
    complexity_threshold: float = 1.0
    enable_complexity_filter: bool = True
    endpoint_snap_search_radius_px: int = 4


# ---------------------------------------------------------------------------
# Block 2: One polygon's core calculation input
# ---------------------------------------------------------------------------
# This is the bundle of prepared data the core engine needs for one landslide:
# crown/toe coordinates, elevations, DEM arrays, masks, and optional metadata.
@dataclass(frozen=True)
class Step1CoreInput:
    config: Step1Config
    contour_nodes: tuple[ContourNode, ...]
    crown_xy: XY
    toe_xy: XY
    crown_elev: Optional[float]
    toe_elev: Optional[float]
    pixel_size_m: float
    dem_array: Any = None
    valid_mask: Any = None
    landslide_mask: Any = None
    origin_x: Optional[float] = None
    origin_y: Optional[float] = None
    pixel_width: Optional[float] = None
    pixel_height: Optional[float] = None
    crown_cell: Optional[tuple[int, int]] = None
    toe_cell: Optional[tuple[int, int]] = None
    landslide_polygon: Any = None
    metadata: dict[str, Any] = field(default_factory=lambda: {})


# ---------------------------------------------------------------------------
# Block 3: Path-building result
# ---------------------------------------------------------------------------
# This records whether the crown-to-toe path worked and stores diagnostics such
# as path length, outside-mask travel, and tortuosity.
@dataclass
class Step1PathResult:
    success: bool
    method_used: str = ""
    fallback_triggered: bool = False
    fallback_reason: str = ""
    diagnostic_flags: list[str] = field(default_factory=lambda: [])
    diagnostic_reason: str = ""
    path_xy: list[XY] = field(default_factory=lambda: [])
    l_2d: Optional[float] = None
    l_3d: Optional[float] = None
    outside_mask_steps: int = 0
    outside_mask_len_2d: float = 0.0
    outside_mask_len_3d: float = 0.0
    outside_mask_fraction_2d: float = 0.0
    outside_mask_fraction_3d: float = 0.0
    max_consecutive_outside_steps: int = 0
    tortuosity_2d: Optional[float] = None
    path_steps: int = 0


# ---------------------------------------------------------------------------
# Block 4: Per-polygon and batch outputs
# ---------------------------------------------------------------------------
# Step1PolygonResult is the geologist-facing measurement for one landslide.
# Step1CoreResult wraps one or more polygon results plus a summary.
@dataclass
class Step1PolygonResult:
    polygon_id: str
    crown_xy: Optional[XY]
    toe_xy: Optional[XY]
    h_m: Optional[float]
    l_m: Optional[float]
    l_2d_m: Optional[float]
    l_3d_m: Optional[float]
    a_obs_deg: Optional[float]
    path_method: str
    path_success: bool
    fallback_triggered: bool
    diagnostic_reason: str
    qc_reasons: list[str]


@dataclass
class Step1CoreResult:
    polygon_results: list[Step1PolygonResult] = field(default_factory=lambda: [])
    path_result: Optional[Step1PathResult] = None
    summary: dict[str, Any] = field(default_factory=lambda: {})
