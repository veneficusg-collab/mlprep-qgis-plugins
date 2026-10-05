"""Read-only diagnostics for the ML-DFI Objective 4 workspace."""

from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CORE_SCRIPTS = [
    ROOT / "steps" / "step0_baseline_alpha_angle" / "step0_baseline_alpha_angle.py",
    ROOT / "steps" / "step1_prepare_inventory_labels" / "step1_prepare_inventory_labels.py",
    ROOT / "steps" / "step2_extract_pixel_samples" / "step2_extract_pixel_samples.py",
    ROOT / "steps" / "step3_train_ml_dfi_model" / "step3_train_ml_dfi_model.py",
    ROOT / "steps" / "step3_train_ml_dfi_model" / "apply_ml_dfi_model_to_rasters.py",
    ROOT / "steps" / "step4_deposition_zone" / "step4_common.py",
    ROOT / "steps" / "step4_deposition_zone" / "step4_deposition_zone_mpi_wrapper.py",
    ROOT / "steps" / "step5_post_depositional_spread_PDS" / "step5_post_depositional_spread_PDS.py",
    ROOT / "steps" / "step6_landslide_damming_potential_LDP" / "step6_landslide_damming_potential_LDP.py",
]
REQUIRED_IMPORTS = [
    "numpy",
    "pandas",
    "rasterio",
    "geopandas",
    "shapely",
    "pyproj",
    "pyogrio",
    "joblib",
    "sklearn",
    "matplotlib",
    "xgboost",
]
OPTIONAL_IMPORTS = [
    "pyarrow",
    "fastparquet",
    "fiona",
]
LATEST_RUNS = ROOT / "Sample Data" / "Output" / "Latest Runs"
ACTIVE_STEP3_RUN = LATEST_RUNS / "Step3" / "model_training_validation"
ACTIVE_STEP3_DOMAIN_RUN = LATEST_RUNS / "Step3" / "domain_application_dfi"


def print_check(ok: bool, label: str, detail: str = "") -> None:
    status = "OK" if ok else "WARN"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")


def resolve_config_path(config_path: Path, raw_value: object) -> Path:
    raw = str(raw_value or "").strip()
    if not raw:
        return Path("")
    path = Path(raw)
    if path.is_absolute():
        return path
    config_relative = (config_path.parent / path).resolve()
    if config_relative.exists():
        return config_relative
    root_relative = (ROOT / path).resolve()
    if root_relative.exists():
        return root_relative
    return config_relative


def crs_authority(value: object) -> tuple[str, str] | None:
    try:
        authority = value.to_authority()  # type: ignore[attr-defined]
    except Exception:
        authority = None
    if authority is not None:
        return authority
    try:
        from pyproj import CRS as PyprojCRS

        return PyprojCRS.from_wkt(value.to_wkt()).to_authority()  # type: ignore[attr-defined]
    except Exception:
        return None


def crs_values_match(left: object, right: object) -> bool:
    if left == right:
        return True
    if left is None or right is None:
        return False
    left_authority = crs_authority(left)
    right_authority = crs_authority(right)
    return left_authority is not None and left_authority == right_authority


def check_imports() -> bool:
    all_ok = True
    for module_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            print_check(False, f"required import {module_name}", str(exc))
            all_ok = False
            continue
        version = getattr(module, "__version__", "unknown")
        print_check(True, f"required import {module_name}", str(version))

    for module_name in OPTIONAL_IMPORTS:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            print_check(False, f"optional import {module_name}", "not installed")
            continue
        version = getattr(module, "__version__", "unknown")
        print_check(True, f"optional import {module_name}", str(version))

    return all_ok


def check_syntax() -> bool:
    all_ok = True
    for script_path in CORE_SCRIPTS:
        try:
            ast.parse(script_path.read_text(encoding="utf-8-sig"), filename=str(script_path))
        except Exception as exc:
            print_check(False, f"syntax {script_path.relative_to(ROOT)}", str(exc))
            all_ok = False
            continue
        print_check(True, f"syntax {script_path.relative_to(ROOT)}")
    return all_ok


def check_table(path: Path, label: str, target_column: str = "target") -> bool:
    import pandas as pd

    if not path.exists():
        print_check(False, label, "missing")
        return False
    df = pd.read_csv(path)
    if target_column not in df.columns:
        print_check(False, label, f"missing {target_column} column")
        return False
    target_counts = df[target_column].value_counts().sort_index().to_dict()
    has_both_classes = set(target_counts) == {0, 1}
    print_check(has_both_classes, label, f"rows={len(df)}, target_counts={target_counts}")
    return bool(has_both_classes)


def find_existing_file(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def check_raster_alignment(reference_path: Path, raster_paths: list[tuple[str, Path]]) -> bool:
    import rasterio

    if not reference_path.exists():
        print_check(False, f"reference raster {reference_path.relative_to(ROOT)}", "missing")
        return False

    all_ok = True
    with rasterio.open(reference_path) as reference:
        ref_grid = (reference.crs, reference.transform, reference.width, reference.height)
        print_check(True, f"reference raster {reference_path.relative_to(ROOT)}", f"{reference.width}x{reference.height}")
        for label, path in raster_paths:
            if not path.exists():
                print_check(False, label, f"missing: {path.relative_to(ROOT)}")
                all_ok = False
                continue
            with rasterio.open(path) as raster:
                same_crs = crs_values_match(raster.crs, ref_grid[0])
                same_transform = raster.transform == ref_grid[1]
                same_size = raster.width == ref_grid[2] and raster.height == ref_grid[3]
                aligned = same_crs and same_transform and same_size
                print_check(
                    aligned,
                    label,
                    f"crs={same_crs}, transform={same_transform}, size={same_size}, dtype={raster.dtypes[0]}",
                )
                all_ok = all_ok and aligned
    return all_ok


def check_sample_outputs() -> bool:
    try:
        import pandas as pd  # noqa: F401
        import rasterio  # noqa: F401
    except Exception as exc:
        print_check(False, "sample output checks", f"dependencies unavailable: {exc}")
        return False

    all_ok = True
    step2_config_path = LATEST_RUNS / "Step2" / "config.step2.latest_run.json"
    sample_table_path = LATEST_RUNS / "Step2" / "ml_dfi_pixel_samples.csv"

    if step2_config_path.exists():
        step2_config = load_config(step2_config_path).get("extract_pixel_samples", {})
        if isinstance(step2_config, dict):
            label_path = resolve_config_path(step2_config_path, step2_config.get("label_raster_path"))
            predictor_values = step2_config.get("predictor_rasters", {})
            if isinstance(predictor_values, dict):
                predictor_paths = [
                    (f"predictor {name}", resolve_config_path(step2_config_path, raw_path))
                    for name, raw_path in predictor_values.items()
                ]
            else:
                predictor_paths = []
        else:
            label_path = Path("")
            predictor_paths = []
    else:
        print_check(False, "latest Step 2 config", "missing")
        label_path = Path("")
        predictor_paths = []
        all_ok = False

    all_ok = check_raster_alignment(
        label_path,
        predictor_paths,
    ) and all_ok
    if not predictor_paths:
        print_check(False, "Step 2 predictor rasters", "none configured")
        all_ok = False

    all_ok = check_table(sample_table_path, "Step 2 sample table") and all_ok

    step3_required_outputs = [
        "trained_ml_dfi_model.pkl",
        "model_metadata.json",
        "validation_predictions.csv",
        "training_predictions.csv",
        "model_metrics.csv",
        "threshold_analysis.csv",
        "feature_importance.csv",
        "processing_log.txt",
    ]
    for name in step3_required_outputs:
        path = ACTIVE_STEP3_RUN / name
        print_check(path.exists(), f"Step 3 active output {name}", str(path.relative_to(ROOT)) if path.exists() else "missing")
        all_ok = all_ok and path.exists()

    domain_dfi_path = ACTIVE_STEP3_DOMAIN_RUN / "ml_dfi_probability_full_valid_predictor_domain.tif"
    print_check(
        domain_dfi_path.exists(),
        "Step 3 active output domain DFI raster",
        str(domain_dfi_path.relative_to(ROOT)) if domain_dfi_path.exists() else "missing",
    )
    all_ok = all_ok and domain_dfi_path.exists()

    predictions_path = ACTIVE_STEP3_RUN / "validation_predictions.csv"
    if predictions_path.exists():
        import pandas as pd

        predictions_df = pd.read_csv(predictions_path)
        has_probability_range = predictions_df["predicted_probability"].between(0, 1).all()
        print_check(has_probability_range, "Step 3 validation probability range", f"rows={len(predictions_df)}")
        all_ok = all_ok and bool(has_probability_range)

    all_ok = check_step4_config() and all_ok
    all_ok = check_step5_pds_config() and all_ok
    all_ok = check_step6_ldp_config() and all_ok
    return all_ok


def load_config(config_path: Path) -> dict[str, object]:
    with config_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a JSON object: {config_path}")
    return payload


def check_step4_config() -> bool:
    config_path = ROOT / "steps" / "step4_deposition_zone" / "config.default.json"
    cfg = load_config(config_path)
    reference = resolve_config_path(config_path, cfg.get("dem_fel_raster"))
    inputs = [
        ("Step 4 D-Infinity flow", resolve_config_path(config_path, cfg.get("dinf_flow_raster"))),
        ("Step 4 source raster", resolve_config_path(config_path, cfg.get("source_raster"))),
        ("Step 4 DFI raster", resolve_config_path(config_path, cfg.get("dfi_raster"))),
    ]
    alpha_raster = resolve_config_path(config_path, cfg.get("alpha_raster"))
    if str(alpha_raster):
        inputs.append(("Step 4 alpha raster", alpha_raster))
    return check_raster_alignment(reference, inputs)


def check_step5_pds_config() -> bool:
    config_path = ROOT / "steps" / "step5_post_depositional_spread_PDS" / "config.default.json"
    cfg = load_config(config_path)
    reference = resolve_config_path(config_path, cfg.get("dem_raster"))
    inputs = [
        ("Step 5 PDS runout mask", resolve_config_path(config_path, cfg.get("runout_mask_raster"))),
        ("Step 5 PDS DFI raster", resolve_config_path(config_path, cfg.get("dfi_raster"))),
        ("Step 5 PDS source mask", resolve_config_path(config_path, cfg.get("source_mask_raster"))),
        ("Step 5 PDS stream mask", resolve_config_path(config_path, cfg.get("stream_mask_raster"))),
        (
            "Step 5 PDS source contributing area",
            resolve_config_path(config_path, cfg.get("source_contributing_area_raster")),
        ),
    ]
    return check_raster_alignment(reference, inputs)


def check_step6_ldp_config() -> bool:
    config_path = ROOT / "steps" / "step6_landslide_damming_potential_LDP" / "config.default.json"
    cfg = load_config(config_path)
    reference = resolve_config_path(config_path, cfg.get("dem_fel_raster"))
    inputs = [
        ("Step 6 D-Infinity flow", resolve_config_path(config_path, cfg.get("dinf_flow_raster"))),
        ("Step 6 runout mask", resolve_config_path(config_path, cfg.get("runout_mask_raster"))),
        ("Step 6 stream mask", resolve_config_path(config_path, cfg.get("stream_raster"))),
        ("Step 6 slope raster", resolve_config_path(config_path, cfg.get("slope_raster"))),
    ]
    source_value = cfg.get("source_raster")
    if isinstance(source_value, str) and source_value.strip():
        inputs.append(("Step 6 source QA raster", resolve_config_path(config_path, source_value)))
    depositional_value = cfg.get("depositional_mask_raster")
    if isinstance(depositional_value, str) and depositional_value.strip():
        inputs.append(
            ("Step 6 depositional QA raster", resolve_config_path(config_path, depositional_value))
        )
    return check_raster_alignment(reference, inputs)


def main() -> int:
    print(f"Workspace: {ROOT}")
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    print()

    import_ok = check_imports()
    print()
    syntax_ok = check_syntax()
    print()
    outputs_ok = check_sample_outputs()
    print()

    if import_ok and syntax_ok and outputs_ok:
        print("Diagnostics completed without blocking warnings.")
        return 0

    print("Diagnostics found warnings that should be reviewed before model training.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
