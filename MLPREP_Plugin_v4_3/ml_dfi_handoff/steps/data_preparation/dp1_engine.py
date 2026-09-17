"""Core Step 1 path engine.

Plain-English map of this file:
1. Basic checks and alpha-angle math.
2. Crown/toe detection from landslide boundary elevations.
3. Geometry helper functions used by the path builder.
4. Main contour-midpoint path routing from crown to toe.
5. Optional path smoothing when the route is too winding.
6. Final metric packaging for the Step 1 workflow.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, cast

import numpy as np
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from dp1_models import (
    ContourNode,
    Step1CoreInput,
    Step1CoreResult,
    Step1PathResult,
    Step1PolygonResult,
    XY,
)


# ---------------------------------------------------------------------------
# Block 1: Internal tuning values
# ---------------------------------------------------------------------------
# These constants are guardrails for path building. They are not user-facing
# workflow parameters; they control how many contour samples are checked, how
# much same-level contour stepping is tolerated, and when a route is considered
# too winding.
_MAX_SEGMENT_CONNECTION_SAMPLES = 129
_SAME_OR_NEAR_LEVEL_TOLERANCE_MULTIPLIER = 0.25
_MAX_GREEDY_BACKTRACKS = 64
_BACKTRACK_CANDIDATE_LIMIT = 4
_TORTUOSITY_TARGET_2D = 1.5
_MAX_TORTUOSITY_SMOOTHING_PASSES = 6
_TORTUOSITY_IMPROVEMENT_EPS = 1e-6
_PATH_CONTAINMENT_ABSOLUTE_TOLERANCE_M = 1e-7


# ---------------------------------------------------------------------------
# Block 2: Small internal containers
# ---------------------------------------------------------------------------
# These small dataclasses keep temporary geometry/path values organized while
# the engine builds a crown-to-toe route. They are intentionally private to this
# file because other modules should use the public models in models.py.
@dataclass(frozen=True)
class _PointXY:
    x: float
    y: float


@dataclass(frozen=True)
class _PathVertex:
    xy: XY
    elevation_m: float
    kind: str
    node_index: int = -1


@dataclass
class _BacktrackBudget:
    attempts: int = 0


@dataclass(frozen=True)
class _ContainmentGeometry:
    geometry: BaseGeometry
    prepared: Any



# ---------------------------------------------------------------------------
# Block 3: Simple quality-control and alpha-angle math
# ---------------------------------------------------------------------------
# These functions answer basic physical questions:
# - Is the polygon shape too complex?
# - Is the height drop positive?
# - Is the path length valid?
# - What is the observed alpha angle?
def complexity_ratio_from_perimeters(
    poly_perimeter: float,
    ombb_perimeter: float,
) -> Optional[float]:
    try:
        p_poly = float(poly_perimeter)
        p_ombb = float(ombb_perimeter)
    except Exception:
        return None
    if not np.isfinite(p_poly) or not np.isfinite(p_ombb) or p_ombb <= 0.0:
        return None
    return float(p_poly / p_ombb)


def should_exclude_polygon_early(
    area_m2: float,
    complexity_ratio: Optional[float],
    *,
    enabled: bool,
    complexity_threshold: float,
    area_threshold_m2: float,
) -> bool:
    if not enabled:
        return False
    is_small = float(area_m2) < float(area_threshold_m2)
    is_complex = bool(
        complexity_ratio is not None
        and float(complexity_ratio) > float(complexity_threshold)
    )
    return bool(is_small or is_complex)


def evaluate_physical_checks_and_alpha(
    h_c: Optional[float],
    h_f: Optional[float],
    l_m: Optional[float],
    pixel_size: float,
) -> tuple[Optional[float], Optional[float], set[str]]:
    reasons: set[str] = set()

    h_m: Optional[float]
    if h_c is not None and h_f is not None:
        h_m = float(h_c) - float(h_f)
        if h_m <= 0.0:
            reasons.add("H<=0")
    else:
        h_m = None
        reasons.add("H<=0")

    l_val: Optional[float]
    if l_m is None:
        l_val = None
    else:
        l_val = float(l_m)

    if l_val is None or l_val <= 0.0:
        reasons.add("L<=0")
    elif pixel_size > 0.0 and l_val < float(pixel_size):
        reasons.add("L<DEM_pixel_size")

    a_obs_deg: Optional[float] = None
    if h_m is not None and l_val is not None and l_val > 0.0:
        a_val = float(math.degrees(math.atan(float(h_m) / float(l_val))))
        if np.isfinite(a_val):
            a_obs_deg = a_val
        else:
            reasons.add("nan_inf")
    else:
        reasons.add("nan_inf")

    return h_m, a_obs_deg, reasons


def upper_percentile_threshold(values: Sequence[float], percentile_high: float) -> Optional[float]:
    finite: list[float] = []
    for value in values:
        try:
            fval = float(value)
        except Exception:
            continue
        if np.isfinite(fval):
            finite.append(fval)
    if not finite:
        return None
    return float(np.percentile(np.asarray(finite, dtype=np.float64), float(percentile_high)))


# ---------------------------------------------------------------------------
# Block 4: Crown and toe detection from boundary samples
# ---------------------------------------------------------------------------
# Step 1 receives boundary points with DEM elevations. The toe is the lowest
# valid boundary point. The crown is taken from the high-elevation boundary run,
# then represented by the midpoint of that high run.
def detect_toe(
    boundary_samples: Sequence[tuple[Any, Optional[float]]],
) -> tuple[Optional[Any], Optional[float], str]:
    valid = [(pt, z) for pt, z in boundary_samples if z is not None]
    if not valid:
        return None, None, "boundary_sampling_failed"
    pt, z = min(valid, key=lambda item: float(item[1]))
    return pt, float(z), "ok"


def detect_crown(
    boundary_samples: Sequence[tuple[Any, Optional[float]]],
    percentile_high: float = 90.0,
) -> tuple[Optional[Any], Optional[float], str]:
    ordered = list(boundary_samples)
    if not ordered:
        return None, None, "boundary_sampling_failed"
    if len(ordered) >= 2 and _point_distance_2d(ordered[0][0], ordered[-1][0]) <= 1e-9:
        ordered = ordered[:-1]
    valid = [(pt, z) for pt, z in ordered if z is not None]
    if not valid:
        return None, None, "boundary_sampling_failed"
    arr = np.asarray([float(z) for _, z in valid], dtype=np.float64)
    if arr.size == 0:
        return None, None, "boundary_sampling_failed"
    z_high = float(np.percentile(arr, float(percentile_high)))
    runs = _high_boundary_runs(ordered, z_high)
    if not runs:
        return None, None, "crown_not_found"
    crown_points, crown_zs = max(
        runs,
        key=lambda run: (
            float(_polyline_length_2d(run[0])),
            float(np.mean(np.asarray(run[1], dtype=np.float64))),
            float(np.max(np.asarray(run[1], dtype=np.float64))),
        ),
    )
    crown_pt, crown_z = _polyline_midpoint(crown_points, crown_zs)
    return crown_pt, crown_z, "ok"


# ---------------------------------------------------------------------------
# Block 5: Boundary polyline helper functions
# ---------------------------------------------------------------------------
# These are small geometry tools used by crown detection. They measure
# distances along the polygon boundary, group high-elevation boundary runs, and
# find the midpoint of a selected crown run.
def _point_distance_2d(a: Any, b: Any) -> float:
    return float(math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y)))


def _make_point_like(template: Any, x: float, y: float) -> Any:
    cls = template.__class__
    try:
        return cls(float(x), float(y))
    except Exception:
        return _PointXY(float(x), float(y))


def _polyline_length_2d(points: Sequence[Any]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        total += _point_distance_2d(a, b)
    return float(total)


def _high_boundary_runs(
    ordered_samples: Sequence[tuple[Any, Optional[float]]],
    z_high: float,
) -> list[tuple[list[Any], list[float]]]:
    if not ordered_samples:
        return []
    flags = [z is not None and float(z) >= float(z_high) for _, z in ordered_samples]
    runs: list[tuple[int, int]] = []
    start: Optional[int] = None
    for idx, is_high in enumerate(flags):
        if is_high:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    if not runs:
        return []
    if len(runs) > 1 and flags[0] and flags[-1]:
        _first_start, first_end = runs[0]
        last_start, _last_end = runs[-1]
        runs = [(last_start, first_end + len(flags)), *runs[1:-1]]
    out: list[tuple[list[Any], list[float]]] = []
    n = len(ordered_samples)
    for start_idx, end_idx in runs:
        pts: list[Any] = []
        zs: list[float] = []
        for idx in range(start_idx, end_idx + 1):
            pt, z = ordered_samples[idx % n]
            if z is None:
                break
            pts.append(pt)
            zs.append(float(z))
        if pts:
            out.append((pts, zs))
    return out


def _polyline_midpoint(
    points: Sequence[Any],
    zs: Sequence[float],
) -> tuple[Any, float]:
    if not points or not zs or len(points) != len(zs):
        raise ValueError("Polyline midpoint requires matched point and elevation sequences.")
    if len(points) == 1:
        return points[0], float(zs[0])
    total_len = _polyline_length_2d(points)
    if total_len <= 0.0:
        mid_idx = len(points) // 2
        return points[mid_idx], float(zs[mid_idx])
    target = total_len / 2.0
    walked = 0.0
    for idx, (pt_a, pt_b) in enumerate(zip(points[:-1], points[1:])):
        seg_len = _point_distance_2d(pt_a, pt_b)
        if seg_len <= 0.0:
            continue
        if walked + seg_len >= target:
            t = float((target - walked) / seg_len)
            x = float(pt_a.x) + t * (float(pt_b.x) - float(pt_a.x))
            y = float(pt_a.y) + t * (float(pt_b.y) - float(pt_a.y))
            z = float(zs[idx]) + t * (float(zs[idx + 1]) - float(zs[idx]))
            return _make_point_like(pt_a, x, y), float(z)
        walked += seg_len
    return points[-1], float(zs[-1])


# ---------------------------------------------------------------------------
# Block 6: Input validation before path building
# ---------------------------------------------------------------------------
# This block rejects incomplete or impossible inputs before routing starts. It
# checks crown/toe coordinates, contour nodes, and pixel size.
def _validate_core_input(core_input: Step1CoreInput) -> None:
    if float(core_input.config.contour_interval_m) <= 0.0:
        raise ValueError("contour_interval_m must be > 0.")
    if float(core_input.pixel_size_m) <= 0.0:
        raise ValueError("pixel_size_m must be > 0.")
    if core_input.crown_elev is None or not np.isfinite(float(core_input.crown_elev)):
        raise ValueError("Step 1 contour core requires a finite crown_elev.")
    if core_input.toe_elev is None or not np.isfinite(float(core_input.toe_elev)):
        raise ValueError("Step 1 contour core requires a finite toe_elev.")
    for label, xy in (("crown_xy", core_input.crown_xy), ("toe_xy", core_input.toe_xy)):
        if len(xy) != 2 or not np.isfinite(float(xy[0])) or not np.isfinite(float(xy[1])):
            raise ValueError(f"{label} must contain finite XY coordinates.")
    for node in core_input.contour_nodes:
        if not np.isfinite(float(node.xy[0])) or not np.isfinite(float(node.xy[1])) or not np.isfinite(float(node.elevation_m)):
            raise ValueError("Contour nodes must have finite coordinates and elevation.")
        for seg_xy in tuple(node.segment_xy):
            if len(seg_xy) != 2 or not np.isfinite(float(seg_xy[0])) or not np.isfinite(float(seg_xy[1])):
                raise ValueError("Contour node segment_xy coordinates must be finite XY pairs.")


# ---------------------------------------------------------------------------
# Block 7: Distance, raster, and polygon helper functions
# ---------------------------------------------------------------------------
# These helpers keep the main routing code readable. They measure path lengths,
# check whether a segment stays inside the landslide polygon, and sort contour
# nodes from high to low elevation.
def _xy_distance_2d(a: XY, b: XY) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _segment_lengths(a: _PathVertex, b: _PathVertex) -> tuple[float, float]:
    dx = float(b.xy[0]) - float(a.xy[0])
    dy = float(b.xy[1]) - float(a.xy[1])
    dz = float(b.elevation_m) - float(a.elevation_m)
    seg_2d = float(math.hypot(dx, dy))
    seg_3d = float(math.sqrt(dx * dx + dy * dy + dz * dz))
    return seg_2d, seg_3d


def _polyline_tortuosity_2d(path_vertices: Sequence[_PathVertex]) -> Optional[float]:
    if len(path_vertices) < 2:
        return None
    l_2d = 0.0
    for a, b in zip(path_vertices[:-1], path_vertices[1:]):
        seg_2d, _seg_3d = _segment_lengths(a, b)
        l_2d += float(seg_2d)
    chord = _xy_distance_2d(path_vertices[0].xy, path_vertices[-1].xy)
    return float(l_2d / chord) if chord > 0.0 else None


def _as_polygon(geom: Any) -> Optional[BaseGeometry]:
    if geom is None:
        return None
    if hasattr(geom, "covers") and hasattr(geom, "buffer"):
        return geom
    return None


def _prepare_containment_geometry(poly: Optional[BaseGeometry]) -> Optional[_ContainmentGeometry]:
    if poly is None:
        return None
    try:
        buffered_poly = cast(BaseGeometry, poly.buffer(1e-9))
        return _ContainmentGeometry(geometry=buffered_poly, prepared=prep(buffered_poly))
    except Exception:
        return None


def _segment_within_polygon(poly: Optional[Any], a: XY, b: XY) -> bool:
    if poly is None:
        return True
    line = LineString([(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))])
    if line.is_empty:
        return False
    try:
        containment = poly if isinstance(poly, _ContainmentGeometry) else _prepare_containment_geometry(poly)
        return bool(containment is not None and containment.prepared.covers(line))
    except Exception:
        return False


def _path_containment_tolerance_m(pixel_size_m: float) -> float:
    return float(max(_PATH_CONTAINMENT_ABSOLUTE_TOLERANCE_M, abs(float(pixel_size_m)) * 1e-8))


def _sorted_contour_nodes(core_input: Step1CoreInput) -> list[ContourNode]:
    return sorted(
        list(core_input.contour_nodes),
        key=lambda item: (-float(item.elevation_m), float(item.xy[0]), float(item.xy[1])),
    )


def _segment_line(node: ContourNode) -> Optional[LineString]:
    coords = tuple((float(x), float(y)) for x, y in tuple(node.segment_xy))
    if len(coords) < 2:
        return None
    line = LineString(coords)
    if line.is_empty or float(line.length) <= 0.0:
        return None
    return line


def _iter_segment_sample_points(
    line: LineString,
    pixel_size_m: float,
) -> list[tuple[float, XY]]:
    length = float(line.length)
    if length <= 0.0:
        return []
    sample_spacing = max(float(pixel_size_m) * 0.5, 0.25)
    sample_count = int(min(_MAX_SEGMENT_CONNECTION_SAMPLES, max(33, math.ceil(length / sample_spacing) + 1)))
    distances = np.linspace(0.0, length, sample_count, dtype=np.float64)
    extra_distances = [float(line.project(Point(float(x), float(y)))) for x, y in list(line.coords)]
    seen: set[tuple[int, int]] = set()
    out: list[tuple[float, XY]] = []
    for distance in [*distances.tolist(), *extra_distances]:
        pt = line.interpolate(float(distance))
        xy = (float(pt.x), float(pt.y))
        key = (int(round(xy[0] * 1000.0)), int(round(xy[1] * 1000.0)))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(distance), xy))
    return out


# ---------------------------------------------------------------------------
# Block 8: Choosing connection points on contour segments
# ---------------------------------------------------------------------------
# A contour node has a representative midpoint, but the straight line to that
# midpoint may leave the polygon. This block samples points along the contour
# segment and chooses a nearby point that keeps the path inside the landslide.
def _resolve_segment_connection_xy(
    current_xy: XY,
    node: ContourNode,
    polygon: Optional[Any],
    pixel_size_m: float,
) -> Optional[XY]:
    midpoint_xy = (float(node.xy[0]), float(node.xy[1]))
    if _segment_within_polygon(polygon, current_xy, midpoint_xy):
        return midpoint_xy

    line = _segment_line(node)
    if line is None:
        return None

    midpoint_distance = float(line.project(Point(float(midpoint_xy[0]), float(midpoint_xy[1]))))
    candidates: list[tuple[float, float, float, float, XY]] = []
    for along_distance, sample_xy in _iter_segment_sample_points(line, pixel_size_m=float(pixel_size_m)):
        seg_dist = _xy_distance_2d(current_xy, sample_xy)
        if seg_dist <= 1e-9:
            continue
        if not _segment_within_polygon(polygon, current_xy, sample_xy):
            continue
        candidates.append(
            (
                float(seg_dist),
                float(abs(along_distance - midpoint_distance)),
                float(along_distance),
                float(sample_xy[0]),
                sample_xy,
            )
        )
    if not candidates:
        return None
    return min(candidates)[-1]


def _lower_node_sort_key(current_xy: XY, node: ContourNode, idx: int) -> tuple[float, float, float, int]:
    return (
        float(_xy_distance_2d(current_xy, (float(node.xy[0]), float(node.xy[1])))),
        float(node.xy[0]),
        float(node.xy[1]),
        int(idx),
    )


def _contour_step_tolerance_m(contour_interval_m: float) -> float:
    return float(max(float(contour_interval_m) * _SAME_OR_NEAR_LEVEL_TOLERANCE_MULTIPLIER, 1e-6))


# ---------------------------------------------------------------------------
# Block 9: Main contour-midpoint routing
# ---------------------------------------------------------------------------
# This is the primary Step 1 path builder. Starting at the crown, it follows the
# old nearest-lower contour choice first. If that route gets trapped, it can
# backtrack a bounded number of times and try the next few valid lower-contour
# alternatives from the most recent decision point.
def _rank_path_candidates(
    current_vertex: _PathVertex,
    used_indices: set[int],
    nodes: Sequence[ContourNode],
    toe_vertex: _PathVertex,
    step_tolerance_m: float,
) -> list[tuple[float, int, int, float, float, int]]:
    ranked: list[tuple[float, int, int, float, float, int]] = []
    toe_dist = _xy_distance_2d(current_vertex.xy, toe_vertex.xy)
    if toe_dist > 1e-9 and float(toe_vertex.elevation_m) <= float(current_vertex.elevation_m) + float(step_tolerance_m):
        ranked.append(
            (
                float(toe_dist),
                0,
                1,
                float(toe_vertex.xy[0]),
                float(toe_vertex.xy[1]),
                -1,
            )
        )
    for idx, node in enumerate(nodes):
        if idx in used_indices:
            continue
        if float(node.elevation_m) > float(current_vertex.elevation_m) + float(step_tolerance_m):
            continue
        dist_xy, x_key, y_key, idx_key = _lower_node_sort_key(current_vertex.xy, node, idx)
        level_penalty = 0 if float(node.elevation_m) < float(current_vertex.elevation_m) - 1e-9 else 1
        ranked.append((float(dist_xy), int(level_penalty), 0, float(x_key), float(y_key), int(idx_key)))
    ranked.sort()
    return ranked


def _search_contour_path_with_backtracking(
    path_vertices: list[_PathVertex],
    used_indices: set[int],
    nodes: Sequence[ContourNode],
    toe_vertex: _PathVertex,
    polygon: Optional[Any],
    pixel_size_m: float,
    step_tolerance_m: float,
    budget: _BacktrackBudget,
) -> Optional[list[_PathVertex]]:
    current_vertex = path_vertices[-1]
    ranked = _rank_path_candidates(
        current_vertex=current_vertex,
        used_indices=used_indices,
        nodes=nodes,
        toe_vertex=toe_vertex,
        step_tolerance_m=float(step_tolerance_m),
    )

    valid_contour_choices = 0
    for *_sort_key, idx in ranked:
        if int(idx) == -1:
            if _segment_within_polygon(polygon, current_vertex.xy, toe_vertex.xy):
                return [*path_vertices, toe_vertex]
            continue

        if valid_contour_choices >= _BACKTRACK_CANDIDATE_LIMIT:
            break

        node = nodes[int(idx)]
        actual_xy = _resolve_segment_connection_xy(
            current_xy=current_vertex.xy,
            node=node,
            polygon=polygon,
            pixel_size_m=float(pixel_size_m),
        )
        if actual_xy is None:
            continue

        valid_contour_choices += 1
        next_vertex = _PathVertex(
            xy=(float(actual_xy[0]), float(actual_xy[1])),
            elevation_m=float(node.elevation_m),
            kind="contour",
            node_index=int(idx),
        )
        next_path = [*path_vertices, next_vertex]
        next_used = {*used_indices, int(idx)}
        out = _search_contour_path_with_backtracking(
            path_vertices=next_path,
            used_indices=next_used,
            nodes=nodes,
            toe_vertex=toe_vertex,
            polygon=polygon,
            pixel_size_m=float(pixel_size_m),
            step_tolerance_m=float(step_tolerance_m),
            budget=budget,
        )
        if out is not None:
            return out

        budget.attempts += 1
        if int(budget.attempts) >= _MAX_GREEDY_BACKTRACKS:
            return None

    return None


def _run_contour_midpoint_path(core_input: Step1CoreInput) -> tuple[Optional[list[_PathVertex]], str]:
    if not core_input.contour_nodes:
        return None, "no_contour_nodes"

    polygon = _prepare_containment_geometry(_as_polygon(core_input.landslide_polygon))
    nodes = _sorted_contour_nodes(core_input)
    crown_elev = core_input.crown_elev
    toe_elev = core_input.toe_elev
    if crown_elev is None or toe_elev is None:
        raise ValueError("Contour routing requires finite crown and toe elevations.")
    toe_vertex = _PathVertex(
        xy=(float(core_input.toe_xy[0]), float(core_input.toe_xy[1])),
        elevation_m=float(toe_elev),
        kind="toe",
        node_index=-1,
    )
    crown_vertex = _PathVertex(
        xy=(float(core_input.crown_xy[0]), float(core_input.crown_xy[1])),
        elevation_m=float(crown_elev),
        kind="crown",
        node_index=-1,
    )
    path_vertices = _search_contour_path_with_backtracking(
        path_vertices=[crown_vertex],
        used_indices=set(),
        nodes=nodes,
        toe_vertex=toe_vertex,
        polygon=polygon,
        pixel_size_m=float(core_input.pixel_size_m),
        step_tolerance_m=_contour_step_tolerance_m(float(core_input.config.contour_interval_m)),
        budget=_BacktrackBudget(),
    )
    if path_vertices is None:
        return None, "no_contour_chain"
    return path_vertices, "ok"


# ---------------------------------------------------------------------------
# Block 10: Path smoothing for overly winding routes
# ---------------------------------------------------------------------------
# Tortuosity means "how winding the path is" compared with a straight line. If
# the contour route is too winding, these functions find the worst intermediate
# contour vertex, then move that vertex along its same contour segment toward a
# balanced position between its upper and lower neighboring vertices.
def _local_sinuosity_score(path_vertices: Sequence[_PathVertex], idx: int) -> Optional[float]:
    if idx <= 0 or idx >= len(path_vertices) - 1:
        return None
    prev_vertex = path_vertices[idx - 1]
    vertex = path_vertices[idx]
    next_vertex = path_vertices[idx + 1]
    local_chord = _xy_distance_2d(prev_vertex.xy, next_vertex.xy)
    if local_chord <= _TORTUOSITY_IMPROVEMENT_EPS:
        return None
    local_path = _xy_distance_2d(prev_vertex.xy, vertex.xy) + _xy_distance_2d(vertex.xy, next_vertex.xy)
    return float(local_path / local_chord)


def _rank_tortuosity_outlier_indices(
    path_vertices: Sequence[_PathVertex],
    nodes: Sequence[ContourNode],
) -> list[int]:
    ranked: list[tuple[float, float, int]] = []
    for idx in range(1, len(path_vertices) - 1):
        vertex = path_vertices[idx]
        if vertex.kind != "contour" or int(vertex.node_index) < 0 or int(vertex.node_index) >= len(nodes):
            continue
        score = _local_sinuosity_score(path_vertices, idx)
        if score is None:
            continue
        local_path = _xy_distance_2d(path_vertices[idx - 1].xy, vertex.xy) + _xy_distance_2d(vertex.xy, path_vertices[idx + 1].xy)
        ranked.append((-float(score), -float(local_path), int(idx)))
    ranked.sort()
    return [idx for _neg_score, _neg_path, idx in ranked]


def _balanced_vertex_xy_for_tortuosity(
    prev_vertex: _PathVertex,
    vertex: _PathVertex,
    next_vertex: _PathVertex,
    node: ContourNode,
    polygon: Optional[Any],
    pixel_size_m: float,
) -> XY:
    line = _segment_line(node)
    if line is None:
        return vertex.xy

    current_xy = (float(vertex.xy[0]), float(vertex.xy[1]))
    current_dist_upper = _xy_distance_2d(prev_vertex.xy, current_xy)
    current_dist_lower = _xy_distance_2d(current_xy, next_vertex.xy)
    current_score = (
        float(abs(current_dist_upper - current_dist_lower)),
        float(current_dist_upper + current_dist_lower),
        0.0,
        float(current_xy[0]),
        float(current_xy[1]),
    )
    best_score = current_score
    best_xy = current_xy

    for _distance_along, sample_xy in _iter_segment_sample_points(line, pixel_size_m=float(pixel_size_m)):
        if not _segment_within_polygon(polygon, prev_vertex.xy, sample_xy):
            continue
        if not _segment_within_polygon(polygon, sample_xy, next_vertex.xy):
            continue
        dist_upper = _xy_distance_2d(prev_vertex.xy, sample_xy)
        dist_lower = _xy_distance_2d(sample_xy, next_vertex.xy)
        candidate_score = (
            float(abs(dist_upper - dist_lower)),
            float(dist_upper + dist_lower),
            float(_xy_distance_2d(sample_xy, current_xy)),
            float(sample_xy[0]),
            float(sample_xy[1]),
        )
        if candidate_score < best_score:
            best_score = candidate_score
            best_xy = sample_xy
    return (float(best_xy[0]), float(best_xy[1]))


def _smooth_path_vertices_for_tortuosity(
    core_input: Step1CoreInput,
    path_vertices: Sequence[_PathVertex],
) -> tuple[list[_PathVertex], bool]:
    nodes = _sorted_contour_nodes(core_input)
    polygon = _prepare_containment_geometry(_as_polygon(core_input.landslide_polygon))
    current = list(path_vertices)
    initial_tortuosity = _polyline_tortuosity_2d(current)
    if initial_tortuosity is None or float(initial_tortuosity) <= float(_TORTUOSITY_TARGET_2D) or len(current) < 4:
        return list(current), False

    changed_any = False
    previous_tortuosity = float(initial_tortuosity)
    for _pass_idx in range(_MAX_TORTUOSITY_SMOOTHING_PASSES):
        accepted_update: Optional[list[_PathVertex]] = None
        accepted_tortuosity: Optional[float] = None
        for idx in _rank_tortuosity_outlier_indices(current, nodes):
            updated = list(current)
            vertex = updated[idx]
            node = nodes[int(vertex.node_index)]
            prev_vertex = updated[idx - 1]
            next_vertex = updated[idx + 1]
            best_xy = _balanced_vertex_xy_for_tortuosity(
                prev_vertex=prev_vertex,
                vertex=vertex,
                next_vertex=next_vertex,
                node=node,
                polygon=polygon,
                pixel_size_m=float(core_input.pixel_size_m),
            )
            if _xy_distance_2d(best_xy, vertex.xy) <= _TORTUOSITY_IMPROVEMENT_EPS:
                continue
            updated[idx] = _PathVertex(
                xy=(float(best_xy[0]), float(best_xy[1])),
                elevation_m=float(vertex.elevation_m),
                kind=str(vertex.kind),
                node_index=int(vertex.node_index),
            )

            new_tortuosity = _polyline_tortuosity_2d(updated)
            if new_tortuosity is None or float(new_tortuosity) >= float(previous_tortuosity) - _TORTUOSITY_IMPROVEMENT_EPS:
                continue
            accepted_update = updated
            accepted_tortuosity = float(new_tortuosity)
            break

        if accepted_update is None or accepted_tortuosity is None:
            break
        current = accepted_update
        previous_tortuosity = float(accepted_tortuosity)
        changed_any = True
        if float(accepted_tortuosity) <= float(_TORTUOSITY_TARGET_2D) + _TORTUOSITY_IMPROVEMENT_EPS:
            break

    return current, changed_any


# ---------------------------------------------------------------------------
# Block 11: Turning path vertices into Step 1 metrics
# ---------------------------------------------------------------------------
# Once a path exists, this block computes 2D length, 3D length, tortuosity, and
# the final Step1PathResult object used by the outer QGIS workflow.
def _compute_path_metrics(
    core_input: Step1CoreInput,
    path_vertices: Sequence[_PathVertex],
) -> Step1PathResult:
    path_xy = [(float(vertex.xy[0]), float(vertex.xy[1])) for vertex in path_vertices]
    l_2d = 0.0
    l_3d = 0.0
    outside_steps = 0
    outside_len_2d = 0.0
    outside_len_3d = 0.0
    consecutive_outside_steps = 0
    max_consecutive_outside_steps = 0
    polygon = _as_polygon(core_input.landslide_polygon)
    containment_tolerance_m = _path_containment_tolerance_m(float(core_input.pixel_size_m))
    containment_polygon = None
    if polygon is not None:
        containment_polygon = cast(Any, polygon.buffer(float(containment_tolerance_m)))
    for a, b in zip(path_vertices[:-1], path_vertices[1:]):
        seg_2d, seg_3d = _segment_lengths(a, b)
        l_2d += float(seg_2d)
        l_3d += float(seg_3d)
        segment_outside_2d = 0.0
        if containment_polygon is not None and float(seg_2d) > 0.0:
            segment = LineString(
                [
                    (float(a.xy[0]), float(a.xy[1])),
                    (float(b.xy[0]), float(b.xy[1])),
                ]
            )
            segment_outside_2d = float(segment.difference(containment_polygon).length)
        if float(segment_outside_2d) > float(containment_tolerance_m):
            outside_steps += 1
            consecutive_outside_steps += 1
            max_consecutive_outside_steps = max(max_consecutive_outside_steps, consecutive_outside_steps)
            outside_len_2d += float(segment_outside_2d)
            if float(seg_2d) > 0.0:
                outside_len_3d += float(seg_3d) * float(segment_outside_2d / seg_2d)
        else:
            consecutive_outside_steps = 0

    chord = _xy_distance_2d(path_vertices[0].xy, path_vertices[-1].xy)
    tortuosity_2d = float(l_2d / chord) if chord > 0.0 else None
    return Step1PathResult(
        success=True,
        method_used="contour_midpoint",
        fallback_triggered=False,
        fallback_reason="",
        diagnostic_flags=[],
        diagnostic_reason="OK",
        path_xy=path_xy,
        l_2d=float(l_2d),
        l_3d=float(l_3d),
        outside_mask_steps=int(outside_steps),
        outside_mask_len_2d=float(outside_len_2d),
        outside_mask_len_3d=float(outside_len_3d),
        outside_mask_fraction_2d=float(outside_len_2d / l_2d) if l_2d > 0.0 else 0.0,
        outside_mask_fraction_3d=float(outside_len_3d / l_3d) if l_3d > 0.0 else 0.0,
        max_consecutive_outside_steps=int(max_consecutive_outside_steps),
        tortuosity_2d=tortuosity_2d,
        path_steps=max(0, int(len(path_xy) - 1)),
    )


def format_qc_reason(reasons: set[str]) -> str:
    if not reasons:
        return "OK"
    return ";".join(sorted(reasons))


# ---------------------------------------------------------------------------
# Block 12: Main public engine entry point
# ---------------------------------------------------------------------------
# This is the function the surrounding Step 1 workflow calls. It validates the
# input, builds the path, smooths it if needed, computes H/L/alpha, and returns
# one Step1CoreResult.
def run_step1_core(core_input: Step1CoreInput) -> Step1CoreResult:
    _validate_core_input(core_input)
    crown_elev = core_input.crown_elev
    toe_elev = core_input.toe_elev
    if crown_elev is None or toe_elev is None:
        raise ValueError("Step 1 contour core requires finite crown and toe elevations.")
    summary: dict[str, Any] = {
        "status": "ok",
        "metadata": dict(core_input.metadata),
        "path_algorithm": "contour_midpoint",
        "contour_interval_m": float(core_input.config.contour_interval_m),
    }

    t0_path = time.perf_counter()
    path_vertices, status = _run_contour_midpoint_path(core_input)
    summary["timings"] = {"contour_midpoint_s": float(time.perf_counter() - t0_path)}

    if path_vertices is None:
        h_m, _a_obs_deg, qc_reasons = evaluate_physical_checks_and_alpha(
            h_c=float(crown_elev),
            h_f=float(toe_elev),
            l_m=None,
            pixel_size=float(core_input.pixel_size_m),
        )
        qc_reasons.add(str(status))
        path_result = Step1PathResult(
            success=False,
            method_used="",
            fallback_triggered=False,
            fallback_reason="",
            diagnostic_flags=[str(status)],
            diagnostic_reason=format_qc_reason({str(status)}),
        )
        polygon_result = Step1PolygonResult(
            polygon_id=str(core_input.metadata.get("polygon_id", "")),
            crown_xy=core_input.crown_xy,
            toe_xy=core_input.toe_xy,
            h_m=h_m,
            l_m=None,
            l_2d_m=None,
            l_3d_m=None,
            a_obs_deg=None,
            path_method="",
            path_success=False,
            fallback_triggered=False,
            diagnostic_reason=path_result.diagnostic_reason,
            qc_reasons=sorted(qc_reasons),
        )
        summary["status"] = "failed"
        summary["method_used"] = ""
        summary["path_success"] = False
        summary["diagnostic_reason"] = path_result.diagnostic_reason
        return Step1CoreResult(polygon_results=[polygon_result], path_result=path_result, summary=summary)

    t0_smooth = time.perf_counter()
    smoothed_vertices, tortuosity_adjusted = _smooth_path_vertices_for_tortuosity(core_input, path_vertices)
    summary["timings"]["tortuosity_adjustment_s"] = float(time.perf_counter() - t0_smooth)
    summary["tortuosity_adjusted"] = bool(tortuosity_adjusted)

    path_result = _compute_path_metrics(core_input, smoothed_vertices)
    containment_tolerance_m = _path_containment_tolerance_m(float(core_input.pixel_size_m))
    if float(path_result.outside_mask_len_2d) > float(containment_tolerance_m):
        path_result.diagnostic_flags.append("path_outside_landslide")
    if tortuosity_adjusted:
        path_result.diagnostic_flags.append("tortuosity_adjusted")
    if path_result.tortuosity_2d is not None and float(path_result.tortuosity_2d) > float(_TORTUOSITY_TARGET_2D) + _TORTUOSITY_IMPROVEMENT_EPS:
        path_result.diagnostic_flags.append("tortuosity_excess_contour")
    if path_result.diagnostic_flags:
        path_result.diagnostic_reason = format_qc_reason(set(path_result.diagnostic_flags))
    h_m, a_obs_deg, qc_reasons = evaluate_physical_checks_and_alpha(
        h_c=float(crown_elev),
        h_f=float(toe_elev),
        l_m=path_result.l_3d,
        pixel_size=float(core_input.pixel_size_m),
    )
    if float(path_result.outside_mask_len_2d) > float(containment_tolerance_m):
        qc_reasons.add("path_outside_landslide")
    polygon_result = Step1PolygonResult(
        polygon_id=str(core_input.metadata.get("polygon_id", "")),
        crown_xy=core_input.crown_xy,
        toe_xy=core_input.toe_xy,
        h_m=h_m,
        l_m=path_result.l_3d,
        l_2d_m=path_result.l_2d,
        l_3d_m=path_result.l_3d,
        a_obs_deg=a_obs_deg,
        path_method=str(path_result.method_used),
        path_success=True,
        fallback_triggered=bool(path_result.fallback_triggered),
        diagnostic_reason=str(path_result.diagnostic_reason),
        qc_reasons=sorted(qc_reasons),
    )
    summary["method_used"] = str(path_result.method_used)
    summary["path_success"] = True
    summary["diagnostic_reason"] = str(path_result.diagnostic_reason)
    return Step1CoreResult(
        polygon_results=[polygon_result],
        path_result=path_result,
        summary=summary,
    )
