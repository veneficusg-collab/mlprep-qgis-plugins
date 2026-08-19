import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import steps.step5_post_depositional_spread_PDS.step5_post_depositional_spread_PDS as step5_module
from steps.step5_post_depositional_spread_PDS.step5_post_depositional_spread_PDS import (
    PostDepositionalSpreadParams,
    _build_edge_seed_mask,
    _collect_connected_local_candidates,
    _compute_source_contribution_from_area,
    _source_connected_runout_mask,
    _source_connected_runout_mask_deque,
    _source_connected_runout_mask_numpy,
    _validate_dfi_values,
    _validate_source_mask_values,
    _validate_stream_mask_values,
    _validate_source_contributing_area_values,
    cleanup_staged_output_bundle,
    load_post_depositional_spread_params,
    prepare_staged_output_bundle,
    publish_staged_output_bundle,
    run_post_depositional_spread_core,
    step5_output_paths,
)


def _config_payload(root: Path) -> dict[str, object]:
    return {
        "runout_mask_raster": str(root / "runout.tif"),
        "dfi_raster": str(root / "dfi.tif"),
        "dem_raster": str(root / "dem.tif"),
        "source_mask_raster": str(root / "source.tif"),
        "stream_mask_raster": str(root / "stream.tif"),
        "source_contributing_area_raster": str(root / "source_area.tif"),
        "output_spread_mask": str(root / "outputs" / "spread.tif"),
        "output_combined_mask": str(root / "outputs" / "combined.tif"),
        "output_seed_mask": str(root / "outputs" / "seed.tif"),
        "output_cleaned_runout_mask": str(root / "outputs" / "cleaned.tif"),
        "output_removed_runout_mask": str(root / "outputs" / "removed.tif"),
        "output_summary_json": str(root / "outputs" / "summary.json"),
    }


def _params_for_outputs(root: Path) -> PostDepositionalSpreadParams:
    payload = _config_payload(root)
    return PostDepositionalSpreadParams(
        **{key: str(value) for key, value in payload.items()}
    )


class Step5SourceContributionTests(unittest.TestCase):
    def test_config_uses_current_defaults_when_optional_values_are_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(_config_payload(root)), encoding="utf-8")

            params = load_post_depositional_spread_params(str(config_path))

            self.assertEqual(params.dfi_threshold, 0.69)
            self.assertEqual(params.source_contributing_area_reference, 4400.0)

    def test_config_rejects_unknown_fields_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = _config_payload(root)
            payload["stale_option"] = True
            config_path = root / "unknown.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown Step 5 config"):
                load_post_depositional_spread_params(str(config_path))

            payload = _config_payload(root)
            payload["dfi_threshold"] = float("nan")
            config_path = root / "nan.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be finite"):
                load_post_depositional_spread_params(str(config_path))

    def test_config_rejects_input_output_alias_and_duplicate_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = _config_payload(root)
            payload["output_spread_mask"] = payload["runout_mask_raster"]
            config_path = root / "input_alias.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not overwrite input"):
                load_post_depositional_spread_params(str(config_path))

            payload = _config_payload(root)
            payload["output_combined_mask"] = payload["output_spread_mask"]
            config_path = root / "output_alias.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "resolve to the same path"):
                load_post_depositional_spread_params(str(config_path))

    def test_dfi_values_must_stay_within_probability_range(self) -> None:
        values = np.asarray([[0.0, 0.5, 1.01]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, r"within \[0, 1\]"):
            _validate_dfi_values(values, np.ones_like(values, dtype=bool))

    def test_negative_source_contributing_area_fails(self) -> None:
        values = np.asarray([[0.0, 10.0, -1.0]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _validate_source_contributing_area_values(values, np.ones_like(values, dtype=bool))

    def test_nonbinary_source_mask_fails(self) -> None:
        values = np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            _validate_source_mask_values(values, np.ones_like(values, dtype=bool))

    def test_negative_stream_mask_fails(self) -> None:
        values = np.asarray([[0.0, 1.0, -1.0]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _validate_stream_mask_values(values, np.ones_like(values, dtype=bool))

    def test_source_contributing_area_shape_must_align_with_runout_grid(self) -> None:
        shape = (3, 3)
        with self.assertRaisesRegex(ValueError, "source contributing area array must align"):
            run_post_depositional_spread_core(
                runout_array=np.ones(shape),
                runout_nodata=0.0,
                dfi_array=np.ones(shape),
                dfi_nodata=None,
                dem_array=np.ones(shape),
                dem_nodata=None,
                source_mask_array=np.zeros(shape),
                source_mask_nodata=None,
                stream_mask_array=np.zeros(shape),
                stream_mask_nodata=None,
                source_contributing_area_array=np.zeros((2, 3)),
                source_contributing_area_nodata=None,
                dx=10.0,
                dy=10.0,
                dfi_threshold=0.5,
                max_relative_rise_m=1.0,
                min_runout_source_area_m2=200.0,
                min_spread_radius_m=5.0,
                max_spread_radius_m=50.0,
                source_contributing_area_reference=5000.0,
            )

    def test_zero_source_contributing_area_assigns_min_radius(self) -> None:
        contribution = _compute_source_contribution_from_area(
            source_contributing_area_value=0.0,
            cell_area_m2=25.0,
            min_spread_radius_m=5.0,
            max_spread_radius_m=50.0,
            source_contributing_area_reference=5000.0,
        )
        self.assertTrue(contribution.has_zero_source_area)
        self.assertEqual(contribution.source_contributing_area, 0.0)
        self.assertEqual(contribution.computed_spread_radius_m, 5.0)

    def test_intermediate_source_contributing_area_produces_intermediate_radius(self) -> None:
        contribution = _compute_source_contribution_from_area(
            source_contributing_area_value=100.0,
            cell_area_m2=25.0,
            min_spread_radius_m=5.0,
            max_spread_radius_m=50.0,
            source_contributing_area_reference=5000.0,
        )
        self.assertAlmostEqual(contribution.source_contributing_area, 2500.0)
        self.assertAlmostEqual(contribution.source_contribution_index, 0.5)
        self.assertGreater(contribution.computed_spread_radius_m, 5.0)
        self.assertLess(contribution.computed_spread_radius_m, 50.0)

    def test_large_source_contributing_area_reaches_max_radius(self) -> None:
        contribution = _compute_source_contribution_from_area(
            source_contributing_area_value=400.0,
            cell_area_m2=25.0,
            min_spread_radius_m=5.0,
            max_spread_radius_m=50.0,
            source_contributing_area_reference=5000.0,
        )
        self.assertAlmostEqual(contribution.source_contribution_index, 1.0)
        self.assertAlmostEqual(contribution.computed_spread_radius_m, 50.0)

    def test_connected_spread_keeps_all_connected_seed_relative_candidates(self) -> None:
        eligible = {
            (1, 2),
            (1, 3),
            (2, 1),
        }
        connected = _collect_connected_local_candidates(
            source_row=1,
            source_col=1,
            eligible_cells=eligible,
        )
        self.assertIn((1, 2), connected)
        self.assertIn((1, 3), connected)
        self.assertIn((2, 1), connected)

    def test_edge_seed_requires_valid_nonrunout_neighbor_and_excludes_source(self) -> None:
        runout = np.zeros((3, 3), dtype=bool)
        runout[1, 1] = True
        eligible = runout.copy()
        dem = np.full((3, 3), 100.0, dtype=np.float64)
        dem[1, 2] = 500.0
        source_mask = np.zeros((3, 3), dtype=bool)
        stream_mask = np.zeros((3, 3), dtype=bool)

        seed_mask = _build_edge_seed_mask(
            eligible_seed_mask=eligible,
            original_runout_mask=runout,
            source_mask=source_mask,
            stream_mask=stream_mask,
            dem_valid=np.ones((3, 3), dtype=bool),
        )
        self.assertTrue(bool(seed_mask[1, 1]))

        source_mask[1, 1] = True
        source_seed_mask = _build_edge_seed_mask(
            eligible_seed_mask=eligible,
            original_runout_mask=runout,
            source_mask=source_mask,
            stream_mask=stream_mask,
            dem_valid=np.ones((3, 3), dtype=bool),
        )
        self.assertFalse(bool(source_seed_mask[1, 1]))

        source_mask[1, 1] = False
        stream_mask[1, 1] = True
        stream_seed_mask = _build_edge_seed_mask(
            eligible_seed_mask=eligible,
            original_runout_mask=runout,
            source_mask=source_mask,
            stream_mask=stream_mask,
            dem_valid=np.ones((3, 3), dtype=bool),
        )
        self.assertFalse(bool(stream_seed_mask[1, 1]))

    def test_interior_runout_cell_does_not_become_seed_without_edge_contact(self) -> None:
        runout = np.zeros((5, 5), dtype=bool)
        runout[1:4, 1:4] = True
        eligible = runout.copy()
        dem = np.full((5, 5), 100.0, dtype=np.float64)
        source_mask = np.zeros((5, 5), dtype=bool)
        stream_mask = np.zeros((5, 5), dtype=bool)

        seed_mask = _build_edge_seed_mask(
            eligible_seed_mask=eligible,
            original_runout_mask=runout,
            source_mask=source_mask,
            stream_mask=stream_mask,
            dem_valid=np.ones((5, 5), dtype=bool),
        )

        self.assertFalse(bool(seed_mask[2, 2]))

    def test_seed_relative_relief_blocks_candidate_above_seed_limit(self) -> None:
        shape = (5, 5)
        runout = np.zeros(shape, dtype=np.float64)
        runout[2, 2] = 1.0
        dem = np.full(shape, 100.0, dtype=np.float64)
        dem[2, 3] = 101.2
        source_contributing_area = np.zeros(shape, dtype=np.float64)
        source_contributing_area[2, 2] = 1.0
        source_mask = np.zeros(shape, dtype=np.float64)
        source_mask[1, 1] = 1.0

        result = run_post_depositional_spread_core(
            runout_array=runout,
            runout_nodata=0.0,
            dfi_array=np.ones(shape, dtype=np.float64),
            dfi_nodata=None,
            dem_array=dem,
            dem_nodata=None,
            source_mask_array=source_mask,
            source_mask_nodata=None,
            stream_mask_array=np.zeros(shape, dtype=np.float64),
            stream_mask_nodata=None,
            source_contributing_area_array=source_contributing_area,
            source_contributing_area_nodata=None,
            dx=1.0,
            dy=1.0,
            dfi_threshold=0.5,
            max_relative_rise_m=1.0,
            min_runout_source_area_m2=0.0,
            min_spread_radius_m=2.0,
            max_spread_radius_m=2.0,
            source_contributing_area_reference=1.0,
        )

        self.assertEqual(int(result.spread_mask[2, 3]), 0)
        self.assertEqual(int(result.spread_mask[2, 1]), 1)
        self.assertGreater(result.summary.rejection_counts["too_high_above_seed"], 0)

    def test_runout_cleanup_removes_cells_below_min_source_area(self) -> None:
        shape = (3, 3)
        runout = np.zeros(shape, dtype=np.float64)
        runout[1, 1] = 1.0
        source_contributing_area = np.zeros(shape, dtype=np.float64)
        source_contributing_area[1, 1] = 1.0

        result = run_post_depositional_spread_core(
            runout_array=runout,
            runout_nodata=0.0,
            dfi_array=np.ones(shape, dtype=np.float64),
            dfi_nodata=None,
            dem_array=np.ones(shape, dtype=np.float64),
            dem_nodata=None,
            source_mask_array=np.ones(shape, dtype=np.float64),
            source_mask_nodata=None,
            stream_mask_array=np.zeros(shape, dtype=np.float64),
            stream_mask_nodata=None,
            source_contributing_area_array=source_contributing_area,
            source_contributing_area_nodata=None,
            dx=10.0,
            dy=10.0,
            dfi_threshold=0.5,
            max_relative_rise_m=1.0,
            min_runout_source_area_m2=200.0,
            min_spread_radius_m=5.0,
            max_spread_radius_m=5.0,
            source_contributing_area_reference=5000.0,
        )

        self.assertEqual(int(result.removed_runout_mask[1, 1]), 1)
        self.assertEqual(int(result.cleaned_runout_mask[1, 1]), 0)
        self.assertEqual(int(result.combined_mask[1, 1]), 0)
        self.assertEqual(result.summary.original_runout_cells, 1)
        self.assertEqual(result.summary.removed_runout_cells, 1)
        self.assertEqual(result.summary.cleaned_runout_cells, 0)

    def test_runout_cleanup_keeps_cells_at_min_source_area(self) -> None:
        shape = (3, 3)
        runout = np.zeros(shape, dtype=np.float64)
        runout[1, 1] = 1.0
        source_contributing_area = np.zeros(shape, dtype=np.float64)
        source_contributing_area[1, 1] = 2.0

        result = run_post_depositional_spread_core(
            runout_array=runout,
            runout_nodata=0.0,
            dfi_array=np.ones(shape, dtype=np.float64),
            dfi_nodata=None,
            dem_array=np.ones(shape, dtype=np.float64),
            dem_nodata=None,
            source_mask_array=np.ones(shape, dtype=np.float64),
            source_mask_nodata=None,
            stream_mask_array=np.zeros(shape, dtype=np.float64),
            stream_mask_nodata=None,
            source_contributing_area_array=source_contributing_area,
            source_contributing_area_nodata=None,
            dx=10.0,
            dy=10.0,
            dfi_threshold=0.5,
            max_relative_rise_m=1.0,
            min_runout_source_area_m2=200.0,
            min_spread_radius_m=5.0,
            max_spread_radius_m=5.0,
            source_contributing_area_reference=5000.0,
        )

        self.assertEqual(int(result.removed_runout_mask[1, 1]), 0)
        self.assertEqual(int(result.cleaned_runout_mask[1, 1]), 1)
        self.assertEqual(result.summary.removed_runout_cells, 0)
        self.assertEqual(result.summary.cleaned_runout_cells, 1)

    def test_runout_cell_with_missing_source_area_fails_instead_of_being_removed(self) -> None:
        shape = (3, 3)
        runout = np.zeros(shape, dtype=np.float64)
        runout[1, 1] = 1.0
        source_area = np.zeros(shape, dtype=np.float64)
        source_area[1, 1] = -9999.0

        with self.assertRaisesRegex(ValueError, "nodata or non-finite values"):
            run_post_depositional_spread_core(
                runout_array=runout,
                runout_nodata=0.0,
                dfi_array=np.ones(shape, dtype=np.float64),
                dfi_nodata=None,
                dem_array=np.ones(shape, dtype=np.float64),
                dem_nodata=None,
                source_mask_array=np.ones(shape, dtype=np.float64),
                source_mask_nodata=None,
                stream_mask_array=np.zeros(shape, dtype=np.float64),
                stream_mask_nodata=None,
                source_contributing_area_array=source_area,
                source_contributing_area_nodata=-9999.0,
                dx=10.0,
                dy=10.0,
                dfi_threshold=0.5,
                max_relative_rise_m=1.0,
                min_runout_source_area_m2=200.0,
                min_spread_radius_m=5.0,
                max_spread_radius_m=5.0,
                source_contributing_area_reference=5000.0,
            )

    def test_source_connectivity_cleanup_removes_disconnected_runout_island(self) -> None:
        cleaned_runout = np.zeros((5, 5), dtype=bool)
        cleaned_runout[1, 1] = True
        cleaned_runout[1, 2] = True
        cleaned_runout[4, 4] = True
        source_mask = np.zeros((5, 5), dtype=bool)
        source_mask[0, 0] = True

        connected = _source_connected_runout_mask(cleaned_runout, source_mask)

        self.assertTrue(bool(connected[1, 1]))
        self.assertTrue(bool(connected[1, 2]))
        self.assertFalse(bool(connected[4, 4]))

    def test_numpy_source_connectivity_matches_reference_flood_fill(self) -> None:
        cleaned_runout = np.zeros((7, 7), dtype=bool)
        cleaned_runout[1, 1:5] = True
        cleaned_runout[2, 4] = True
        cleaned_runout[3, 4] = True
        cleaned_runout[5, 5] = True

        anchors = np.zeros((7, 7), dtype=bool)
        anchors[1, 1] = True

        vectorized = _source_connected_runout_mask_numpy(cleaned_runout, anchors)
        reference = _source_connected_runout_mask_deque(cleaned_runout, anchors)

        np.testing.assert_array_equal(vectorized, reference)
        self.assertTrue(bool(vectorized[3, 4]))
        self.assertFalse(bool(vectorized[5, 5]))

    def test_staged_publication_replaces_complete_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            params = _params_for_outputs(root)
            bundle = prepare_staged_output_bundle(params)
            try:
                for staged_path in bundle.staged_to_official:
                    path = Path(staged_path)
                    path.write_text(
                        "{}" if path.suffix.lower() == ".json" else "new",
                        encoding="utf-8",
                    )

                publish_staged_output_bundle(bundle)

                for official_path in step5_output_paths(params).values():
                    self.assertTrue(Path(official_path).is_file())
            finally:
                cleanup_staged_output_bundle(bundle)

    def test_staged_publication_restores_previous_bundle_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            params = _params_for_outputs(root)
            for official_path in step5_output_paths(params).values():
                path = Path(official_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("old", encoding="utf-8")

            bundle = prepare_staged_output_bundle(params)
            try:
                for staged_path in bundle.staged_to_official:
                    Path(staged_path).write_text("new", encoding="utf-8")

                real_replace = os.replace

                def fail_during_second_publication(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
                    source_path = Path(source)
                    destination_path = Path(destination)
                    if (
                        source_path.parent.name.startswith(".step5-stage-")
                        and source_path.name == Path(params.output_combined_mask).name
                        and destination_path == Path(params.output_combined_mask)
                    ):
                        raise OSError("simulated publication failure")
                    real_replace(source, destination)

                with patch.object(step5_module.os, "replace", side_effect=fail_during_second_publication):
                    with self.assertRaisesRegex(OSError, "simulated publication failure"):
                        publish_staged_output_bundle(bundle)

                for official_path in step5_output_paths(params).values():
                    self.assertEqual(Path(official_path).read_text(encoding="utf-8"), "old")
            finally:
                cleanup_staged_output_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
