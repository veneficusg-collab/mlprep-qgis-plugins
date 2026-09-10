from __future__ import annotations

import sys
from pathlib import Path

from shapely.geometry import Polygon


DATA_PREPARATION = (
    Path(__file__).resolve().parents[1] / "steps" / "data_preparation"
)
sys.path.insert(0, str(DATA_PREPARATION))

import dp1_engine  # noqa: E402


def test_early_filter_excludes_small_or_complex_polygon() -> None:
    kwargs = {
        "enabled": True,
        "complexity_threshold": 1.0,
        "area_threshold_m2": 250.0,
    }
    assert dp1_engine.should_exclude_polygon_early(249.9, 1.01, **kwargs)
    assert dp1_engine.should_exclude_polygon_early(250.0, 1.01, **kwargs)
    assert dp1_engine.should_exclude_polygon_early(249.9, 1.0, **kwargs)
    assert dp1_engine.should_exclude_polygon_early(249.9, None, **kwargs)
    assert not dp1_engine.should_exclude_polygon_early(250.0, 1.0, **kwargs)


def test_early_filter_can_be_disabled() -> None:
    assert not dp1_engine.should_exclude_polygon_early(
        100.0,
        2.0,
        enabled=False,
        complexity_threshold=1.0,
        area_threshold_m2=250.0,
    )


def test_prepared_containment_matches_polygon_covers_rule() -> None:
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (6, 10), (6, 4), (4, 4), (4, 10), (0, 10)])
    prepared = dp1_engine._prepare_containment_geometry(polygon)

    assert dp1_engine._segment_within_polygon(prepared, (1.0, 1.0), (9.0, 1.0))
    assert not dp1_engine._segment_within_polygon(prepared, (2.0, 8.0), (8.0, 8.0))
