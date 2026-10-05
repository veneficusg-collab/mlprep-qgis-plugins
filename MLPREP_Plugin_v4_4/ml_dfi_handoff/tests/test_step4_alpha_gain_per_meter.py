import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from steps.step4_deposition_zone.step4_common import Step4Params, load_step4_params
from steps.step4_deposition_zone.step4_deposition_zone_mpi_wrapper import (
    cleanup_staged_output_bundle,
    prepare_staged_output_bundle,
    publish_staged_output_bundle,
)


def _minimal_config() -> dict[str, object]:
    return {
        "dem_fel_raster": "dem.tif",
        "dinf_flow_raster": "flow.tif",
        "source_raster": "source.tif",
        "dfi_raster": "dfi.tif",
        "alpha_raster": "alpha.tif",
        "output_dynamic_alpha": "dynamic_alpha.tif",
        "output_beta_angle": "beta.tif",
        "output_dfs": "dfs.tif",
        "output_mask": "mask.tif",
        "output_depositional_mask": "deposition.tif",
        "output_summary_json": "summary.json",
    }


class Step4AlphaGainPerMeterTests(unittest.TestCase):
    def _load(self, payload: dict[str, object]):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            return load_step4_params(str(config_path))

    def test_default_alpha_gain_per_meter_is_0p03333(self) -> None:
        params = self._load(_minimal_config())
        self.assertAlmostEqual(params.alpha_gain_per_meter, 0.03333)

    def test_explicit_alpha_gain_per_meter_is_loaded(self) -> None:
        config = _minimal_config()
        config["alpha_gain_per_meter"] = 0.2
        params = self._load(config)
        self.assertAlmostEqual(params.alpha_gain_per_meter, 0.2)

    def test_negative_alpha_gain_per_meter_fails(self) -> None:
        config = _minimal_config()
        config["alpha_gain_per_meter"] = -0.01
        with self.assertRaisesRegex(ValueError, "alpha_gain_per_meter must be >= 0"):
            self._load(config)

    def test_legacy_gain_and_reference_distance_fail_clearly(self) -> None:
        for legacy_key in ("alpha_step_gain", "alpha_gain_reference_distance_m"):
            with self.subTest(legacy_key=legacy_key):
                config = _minimal_config()
                config[legacy_key] = 1.0
                with self.assertRaisesRegex(ValueError, "alpha_gain_per_meter"):
                    self._load(config)

    def test_unknown_config_field_fails(self) -> None:
        config = _minimal_config()
        config["alpha_gain_per_metre"] = 0.1
        with self.assertRaisesRegex(ValueError, "Unknown Step 4 config field"):
            self._load(config)

    def test_non_finite_numeric_values_fail(self) -> None:
        for key in ("proportion_threshold", "dfi_mid", "alpha_gain_per_meter"):
            with self.subTest(key=key):
                config = _minimal_config()
                config[key] = float("nan")
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    self._load(config)

    def test_dfi_mid_must_be_in_unit_interval(self) -> None:
        config = _minimal_config()
        config["dfi_mid"] = 1.1
        with self.assertRaisesRegex(ValueError, "dfi_mid must be within"):
            self._load(config)

    def test_output_cannot_overwrite_input(self) -> None:
        config = _minimal_config()
        config["output_mask"] = "dem.tif"
        with self.assertRaisesRegex(ValueError, "must not overwrite input"):
            self._load(config)

    def test_output_paths_must_be_unique(self) -> None:
        config = _minimal_config()
        config["output_mask"] = "deposition.tif"
        with self.assertRaisesRegex(ValueError, "duplicates output"):
            self._load(config)

    def test_removed_output_mode_fails_clearly(self) -> None:
        config = _minimal_config()
        config["output_mode"] = "full"
        with self.assertRaisesRegex(ValueError, "output_mode has been removed"):
            self._load(config)


class Step4PublicationTests(unittest.TestCase):
    def _params(self, root: Path) -> Step4Params:
        return Step4Params(
            dem_fel_raster=str(root / "dem.tif"),
            dinf_flow_raster=str(root / "flow.tif"),
            source_raster=str(root / "source.tif"),
            dfi_raster=str(root / "dfi.tif"),
            alpha_raster=str(root / "alpha.tif"),
            output_dynamic_alpha=str(root / "dynamic_alpha.tif"),
            output_beta_angle=str(root / "beta.tif"),
            output_dfs=str(root / "dfs.tif"),
            output_mask=str(root / "mask.tif"),
            output_depositional_mask=str(root / "deposition.tif"),
            output_summary_json=str(root / "summary.json"),
        )

    def test_complete_staged_bundle_replaces_official_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = prepare_staged_output_bundle(self._params(root))
            try:
                for index, (staged, official) in enumerate(bundle.staged_to_official.items()):
                    Path(official).write_text(f"old-{index}", encoding="utf-8")
                    Path(staged).write_text(f"new-{index}", encoding="utf-8")
                publish_staged_output_bundle(bundle)
                for index, official in enumerate(bundle.staged_to_official.values()):
                    self.assertEqual(Path(official).read_text(encoding="utf-8"), f"new-{index}")
            finally:
                cleanup_staged_output_bundle(bundle)

    def test_incomplete_staged_bundle_preserves_official_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = prepare_staged_output_bundle(self._params(root))
            try:
                for index, official in enumerate(bundle.staged_to_official.values()):
                    Path(official).write_text(f"old-{index}", encoding="utf-8")
                first_staged = next(iter(bundle.staged_to_official))
                Path(first_staged).write_text("partial-new", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "staged output bundle is incomplete"):
                    publish_staged_output_bundle(bundle)
                for index, official in enumerate(bundle.staged_to_official.values()):
                    self.assertEqual(Path(official).read_text(encoding="utf-8"), f"old-{index}")
            finally:
                cleanup_staged_output_bundle(bundle)

    def test_publication_failure_rolls_back_every_replaced_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = prepare_staged_output_bundle(self._params(root))
            pairs = list(bundle.staged_to_official.items())
            try:
                for index, (staged, official) in enumerate(pairs):
                    Path(official).write_text(f"old-{index}", encoding="utf-8")
                    Path(staged).write_text(f"new-{index}", encoding="utf-8")

                real_replace = __import__("os").replace
                failing_staged = Path(pairs[1][0])
                failing_official = Path(pairs[1][1])

                def flaky_replace(source, destination):
                    if Path(source) == failing_staged and Path(destination) == failing_official:
                        raise OSError("simulated locked output")
                    return real_replace(source, destination)

                with mock.patch(
                    "steps.step4_deposition_zone."
                    "step4_deposition_zone_mpi_wrapper.os.replace",
                    side_effect=flaky_replace,
                ):
                    with self.assertRaisesRegex(OSError, "simulated locked output"):
                        publish_staged_output_bundle(bundle)

                for index, official in enumerate(bundle.staged_to_official.values()):
                    self.assertEqual(Path(official).read_text(encoding="utf-8"), f"old-{index}")
            finally:
                cleanup_staged_output_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
