import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import rasterio

from steps.step6_landslide_damming_potential_LDP import (
    step6_landslide_damming_potential_LDP as step6,
)


def _minimal_config() -> dict[str, object]:
    return {
        "dem_fel_raster": "dem.tif",
        "dinf_flow_raster": "dinf.tif",
        "runout_mask_raster": "runout.tif",
        "stream_raster": "stream.tif",
        "slope_raster": "slope.tif",
        "valley_width_enabled": False,
        "reference_volume_enabled": False,
        "output_stream_intersection_mask": "intersection.tif",
        "output_inflow_angle": "inflow.tif",
        "output_stream_angle": "stream_angle.tif",
        "output_runout_approach_angle": "runout_angle.tif",
        "output_channel_slope": "channel_slope.tif",
        "output_candidate_damming_potential_class": "candidate_class.tif",
        "output_damming_potential_class": "class.tif",
        "output_damming_potential_score": "score.tif",
        "output_summary_json": "summary.json",
    }


def _load_config(root: Path, payload: dict[str, object]) -> step6.Step6Params:
    config_path = root / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return step6.load_params(str(config_path))


def _output_params(root: Path) -> step6.Step6Params:
    payload = _minimal_config()
    return _load_config(root, payload)


class Step6ConfigSafetyTests(unittest.TestCase):
    def test_unknown_config_field_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = _minimal_config()
            payload["runout_approach_window_metres"] = 25.0
            with self.assertRaisesRegex(ValueError, "Unknown Step 6 config field"):
                _load_config(Path(temp_dir), payload)

    def test_removed_candidate_fallback_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = _minimal_config()
            payload["candidate_search_radius_cells"] = 1
            with self.assertRaisesRegex(ValueError, "direct runout-stream intersections only"):
                _load_config(Path(temp_dir), payload)

    def test_non_finite_numeric_value_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = _minimal_config()
            payload["runout_approach_window_m"] = float("nan")
            with self.assertRaisesRegex(ValueError, "must be finite"):
                _load_config(Path(temp_dir), payload)

    def test_output_cannot_alias_any_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = _minimal_config()
            payload["output_summary_json"] = "dem.tif"
            with self.assertRaisesRegex(ValueError, "must not overwrite input"):
                _load_config(Path(temp_dir), payload)

    def test_summary_and_candidate_outputs_participate_in_uniqueness_check(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = _minimal_config()
            payload["output_candidate_points"] = "summary.json"
            with self.assertRaisesRegex(ValueError, "duplicates output path"):
                _load_config(Path(temp_dir), payload)


class Step6RasterContractTests(unittest.TestCase):
    @staticmethod
    def _grid(values: np.ndarray, nodata: float | None = None) -> step6.RasterGrid:
        return step6.RasterGrid(
            path="memory",
            array=values,
            nodata=nodata,
            transform=rasterio.transform.from_origin(0.0, 2.0, 1.0, 1.0),
            crs=None,
            width=values.shape[1],
            height=values.shape[0],
            profile={},
        )

    def test_binary_mask_rejects_nonbinary_values(self) -> None:
        grid = self._grid(np.asarray([[0, 1], [2, 0]], dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "must be binary"):
            step6.validate_binary_mask_raster(grid, "runout_mask_raster")

    def test_negative_hans_is_clamped_and_counted(self) -> None:
        dem = np.asarray([[9.0, 10.0]], dtype=np.float32)
        valid = np.ones_like(dem, dtype=bool)
        stream = np.asarray([[False, True]])
        receiver_row = np.asarray([[0, -1]], dtype=np.int32)
        receiver_col = np.asarray([[1, -1]], dtype=np.int32)
        height, _stream_elevation, clamped = step6.trace_to_stream_and_compute_height_above_stream(
            dem,
            valid,
            stream,
            receiver_row,
            receiver_col,
        )
        self.assertEqual(clamped, 1)
        self.assertEqual(float(height[0, 0]), 0.0)

    def test_candidates_require_direct_runout_stream_overlap(self) -> None:
        params = step6._smoke_params()
        params.valley_width_enabled = False
        params.reference_volume_enabled = False
        dem = np.ones((3, 3), dtype=np.float32)
        runout = np.zeros((3, 3), dtype=np.uint8)
        stream = np.zeros((3, 3), dtype=np.uint8)
        runout[1, 0] = 1
        stream[1, 1] = 1
        result = step6.evaluate_damming_potential(
            step6.CoreArrays(
                dem=dem,
                dinf=np.zeros((3, 3), dtype=np.float32),
                runout_mask=runout,
                stream_mask=stream,
                slope=np.ones((3, 3), dtype=np.float32),
            ),
            params,
            rasterio.transform.from_origin(0.0, 3.0, 1.0, 1.0),
            1.0,
            1.0,
        )
        self.assertEqual(result.summary["counts"]["candidate_cells"], 0)

    def test_parallel_candidate_kernel_matches_single_thread(self) -> None:
        if step6._evaluate_candidate_cells_numba is None:
            self.skipTest("Numba is unavailable in this Python environment.")
        params = step6._smoke_params()
        params.valley_width_enabled = False
        params.reference_volume_enabled = False
        transform = rasterio.transform.from_origin(0.0, 9.0, 1.0, 1.0)
        dem = np.repeat(np.arange(9, 0, -1, dtype=np.float32)[:, None], 9, axis=1)
        stream = np.zeros((9, 9), dtype=np.uint8)
        stream[4, :] = 1
        runout = np.zeros((9, 9), dtype=np.uint8)
        runout[1:5, 1:8] = 1
        arrays = step6.CoreArrays(
            dem=dem,
            dinf=np.full((9, 9), np.float32(3.0 * np.pi / 2.0)),
            runout_mask=runout,
            stream_mask=stream,
            slope=np.full((9, 9), 5.0, dtype=np.float32),
        )
        original_threads = step6.get_num_threads()
        try:
            params.numba_threads = 1
            single = step6.evaluate_damming_potential(arrays, params, transform, 1.0, 1.0)
            params.numba_threads = min(2, original_threads)
            parallel = step6.evaluate_damming_potential(arrays, params, transform, 1.0, 1.0)
        finally:
            step6.set_num_threads(original_threads)
        np.testing.assert_array_equal(single.damming_class, parallel.damming_class)
        np.testing.assert_array_equal(single.damming_score, parallel.damming_score)


class Step6PublicationTests(unittest.TestCase):
    def test_complete_staged_bundle_replaces_official_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = step6.prepare_staged_output_bundle(_output_params(root))
            try:
                for index, (staged, official) in enumerate(bundle.staged_to_official.items()):
                    Path(official).write_text(f"old-{index}", encoding="utf-8")
                    Path(staged).write_text(f"new-{index}", encoding="utf-8")
                step6.publish_staged_output_bundle(bundle)
                for index, official in enumerate(bundle.staged_to_official.values()):
                    self.assertEqual(Path(official).read_text(encoding="utf-8"), f"new-{index}")
            finally:
                step6.cleanup_staged_output_bundle(bundle)

    def test_publication_failure_restores_previous_complete_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = step6.prepare_staged_output_bundle(_output_params(root))
            pairs = list(bundle.staged_to_official.items())
            try:
                for index, (staged, official) in enumerate(pairs):
                    Path(official).write_text(f"old-{index}", encoding="utf-8")
                    Path(staged).write_text(f"new-{index}", encoding="utf-8")
                real_replace = os.replace
                failing_staged = Path(pairs[1][0])
                failing_official = Path(pairs[1][1])

                def flaky_replace(source, destination):
                    if Path(source) == failing_staged and Path(destination) == failing_official:
                        raise OSError("simulated locked output")
                    return real_replace(source, destination)

                with mock.patch.object(step6.os, "replace", side_effect=flaky_replace):
                    with self.assertRaisesRegex(OSError, "simulated locked output"):
                        step6.publish_staged_output_bundle(bundle)
                for index, (_staged, official) in enumerate(pairs):
                    self.assertEqual(Path(official).read_text(encoding="utf-8"), f"old-{index}")
            finally:
                step6.cleanup_staged_output_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
