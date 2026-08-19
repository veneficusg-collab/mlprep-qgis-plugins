import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "steps" / "data_preparation"))

from dp1_engine import (  # noqa: E402
    _PathVertex,
    _balanced_vertex_xy_for_tortuosity,
    _compute_path_metrics,
    _run_contour_midpoint_path,
    _segment_within_polygon,
    run_step1_core,
)
from dp1_models import ContourNode, Step1Config, Step1CoreInput  # noqa: E402


def _l_shaped_polygon() -> Polygon:
    return Polygon([(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)])


def _core_input(*, contour_nodes=(), crown_xy=(0.5, 2.0), toe_xy=(2.0, 0.5)) -> Step1CoreInput:
    return Step1CoreInput(
        config=Step1Config(),
        contour_nodes=tuple(contour_nodes),
        crown_xy=crown_xy,
        toe_xy=toe_xy,
        crown_elev=10.0,
        toe_elev=0.0,
        pixel_size_m=5.0,
        landslide_polygon=_l_shaped_polygon(),
    )


class DP1PathContainmentTests(unittest.TestCase):
    def test_invalid_direct_toe_connection_uses_contained_contour_detour(self) -> None:
        contour = ContourNode(
            xy=(3.5, 0.5),
            elevation_m=5.0,
            segment_xy=((0.5, 0.5), (3.5, 0.5)),
        )
        core_input = _core_input(contour_nodes=(contour,))

        path, status = _run_contour_midpoint_path(core_input)

        self.assertEqual(status, "ok")
        self.assertIsNotNone(path)
        assert path is not None
        self.assertGreaterEqual(len(path), 3)
        for start, end in zip(path[:-1], path[1:]):
            self.assertTrue(
                _segment_within_polygon(core_input.landslide_polygon, start.xy, end.xy)
            )

    def test_tortuosity_adjustment_keeps_segment_to_toe_inside(self) -> None:
        polygon = _l_shaped_polygon()
        previous = _PathVertex((0.5, 3.8), 10.0, "crown")
        current = _PathVertex((0.5, 0.5), 5.0, "contour", 0)
        toe = _PathVertex((2.0, 0.5), 0.0, "toe")
        contour = ContourNode(
            xy=(0.5, 2.0),
            elevation_m=5.0,
            segment_xy=((0.5, 0.5), (0.5, 3.5)),
        )

        adjusted = _balanced_vertex_xy_for_tortuosity(
            previous,
            current,
            toe,
            contour,
            polygon,
            5.0,
        )

        self.assertTrue(_segment_within_polygon(polygon, previous.xy, adjusted))
        self.assertTrue(_segment_within_polygon(polygon, adjusted, toe.xy))

    def test_final_metrics_measure_actual_outside_path(self) -> None:
        core_input = _core_input()
        vertices = [
            _PathVertex(core_input.crown_xy, 10.0, "crown"),
            _PathVertex(core_input.toe_xy, 0.0, "toe"),
        ]

        result = _compute_path_metrics(core_input, vertices)

        self.assertEqual(result.outside_mask_steps, 1)
        self.assertEqual(result.max_consecutive_outside_steps, 1)
        self.assertGreater(result.outside_mask_len_2d, 0.0)
        self.assertGreater(result.outside_mask_len_3d, 0.0)
        self.assertGreater(result.outside_mask_fraction_2d, 0.0)
        self.assertGreater(result.outside_mask_fraction_3d, 0.0)

    def test_final_audit_adds_outside_path_qc_reason(self) -> None:
        core_input = _core_input()
        outside_vertices = [
            _PathVertex(core_input.crown_xy, 10.0, "crown"),
            _PathVertex(core_input.toe_xy, 0.0, "toe"),
        ]

        with patch("dp1_engine._run_contour_midpoint_path", return_value=(outside_vertices, "ok")):
            result = run_step1_core(core_input)

        self.assertIsNotNone(result.path_result)
        assert result.path_result is not None
        self.assertIn("path_outside_landslide", result.path_result.diagnostic_flags)
        self.assertIn("path_outside_landslide", result.polygon_results[0].qc_reasons)


if __name__ == "__main__":
    unittest.main()
