"""Shared Step 4 config, raster validation, and reporting helpers.

This module is intentionally routing-engine neutral. The active C++/MPI wrapper
uses it for config, validation, and reporting; the archived
old_step4_deposition_zone.py file is retained only as a historical Python
reference.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from osgeo import gdal, osr
except ModuleNotFoundError:  # pragma: no cover - exercised in non-QGIS Python environments.
    gdal = None  # type: ignore[assignment]
    osr = None  # type: ignore[assignment]

TAUDEM_FLOAT_NODATA = np.float32(-3.4028235e38)
ALPHA_ANGLE_VALID_MIN = 1.0
ALPHA_ANGLE_VALID_MAX = 90.0
FIXED_ALPHA_MIN = 6.0
FIXED_ALPHA_MAX = 72.0
DEFAULT_ALPHA_GAIN_PER_METER = 0.03333
FIXED_DFI_DEADBAND = 0.05
FIXED_BETA_TIE_TOLERANCE = 1e-6
FIXED_FAIL_ON_UNPROCESSED_CELLS = True
FIXED_MAX_UNPROCESSED_FRACTION = 0.0
FIXED_MAX_SOURCE_FRACTION = 0.30
FIXED_MIN_DFI_VALID_FRACTION = 0.95

DEPRECATED_CONFIG_FIELDS = (
    "slope_units",
    "su_id_field",
    "alpha_values_csv",
    "alpha_field",
    "predictions_contract_json",
    "generated_alpha_raster",
)

REMOVED_CONFIG_FIELDS = {
    "output_mode": (
        "output_mode has been removed. "
        "The active Step 4 MPI runtime always writes the complete output bundle."
    ),
    "fail_if_alpha_looks_like_probability": (
        "fail_if_alpha_looks_like_probability is no longer configurable. "
        "Step 4 always fails when source alpha values look like probabilities."
    ),
    "enforce_alpha_max": (
        "enforce_alpha_max is no longer configurable. "
        f"Step 4 always enforces fixed alpha bounds [{FIXED_ALPHA_MIN:g}, {FIXED_ALPHA_MAX:g}] degrees."
    ),
    "alpha_min": (
        "alpha_min is no longer configurable. "
        f"Step 4 uses a fixed minimum dynamic alpha of {FIXED_ALPHA_MIN:g} degrees."
    ),
    "alpha_max": (
        "alpha_max is no longer configurable. "
        f"Step 4 uses a fixed maximum dynamic alpha of {FIXED_ALPHA_MAX:g} degrees."
    ),
    "alpha_plausible_min": (
        "alpha_plausible_min is no longer configurable. "
        "Step 4 accepts only valid alpha angles from 1 to 90 degrees."
    ),
    "alpha_plausible_max": (
        "alpha_plausible_max is no longer configurable. "
        "Step 4 accepts only valid alpha angles from 1 to 90 degrees."
    ),
    "clamp_initial_source_alpha": (
        "clamp_initial_source_alpha is no longer configurable. "
        f"Step 4 always fails when source alpha values are outside fixed bounds [{FIXED_ALPHA_MIN:g}, {FIXED_ALPHA_MAX:g}]."
    ),
    "alpha_step_gain_up": (
        "alpha_step_gain_up is no longer configurable. "
        "Use alpha_gain_per_meter; Step 4 applies the same rate to alpha increases and decreases."
    ),
    "alpha_step_gain_down": (
        "alpha_step_gain_down is no longer configurable. "
        "Use alpha_gain_per_meter; Step 4 applies the same rate to alpha increases and decreases."
    ),
    "alpha_step_gain": (
        "alpha_step_gain has been replaced by alpha_gain_per_meter. "
        "Provide the alpha change rate in degrees per meter."
    ),
    "alpha_gain_reference_distance_m": (
        "alpha_gain_reference_distance_m has been removed. "
        "Step 4 now uses alpha_gain_per_meter directly."
    ),
    "clamp_dfi_to_unit_interval": (
        "clamp_dfi_to_unit_interval is no longer configurable. "
        "Step 4 always requires valid DFI values to already be within [0, 1] and fails if they are not."
    ),
    "dfi_deadband": (
        "dfi_deadband is no longer configurable. "
        f"Step 4 uses a fixed DFI deadband of {FIXED_DFI_DEADBAND:g}."
    ),
    "beta_tie_tolerance": (
        "beta_tie_tolerance is no longer configurable. "
        f"Step 4 uses a fixed beta tie tolerance of {FIXED_BETA_TIE_TOLERANCE:g}."
    ),
    "fail_on_unprocessed_cells": (
        "fail_on_unprocessed_cells is no longer configurable. "
        "Step 4 always fails when unprocessed routing cells exceed the fixed allowance."
    ),
    "max_unprocessed_fraction": (
        "max_unprocessed_fraction is no longer configurable. "
        "Step 4 uses a fixed allowance of 0.0."
    ),
    "require_binary_source_raster": (
        "require_binary_source_raster is no longer configurable. "
        "Step 4 always requires source rasters to contain only 0, 1, or NoData."
    ),
    "max_source_fraction": (
        "max_source_fraction is no longer configurable. "
        "Step 4 uses a fixed warning threshold of 0.30."
    ),
    "min_dfi_valid_fraction": (
        "min_dfi_valid_fraction is no longer configurable. "
        "Step 4 uses a fixed warning threshold of 0.95."
    ),
    "constant_alpha_angle": (
        "constant_alpha_angle is no longer supported. "
        "Step 4 now requires an aligned alpha_raster input."
    ),
}

ACTIVE_CONFIG_FIELDS = {
    "dem_fel_raster",
    "dinf_flow_raster",
    "source_raster",
    "dfi_raster",
    "alpha_raster",
    "proportion_threshold",
    "dfi_mid",
    "alpha_gain_per_meter",
    "write_debug_rasters",
    "output_dynamic_alpha",
    "output_beta_angle",
    "output_dfs",
    "output_mask",
    "output_depositional_mask",
    "output_summary_json",
    "observed_full_landslide_footprint_raster",
    "observed_depositional_zone_raster",
    "validation_domain_raster",
    "output_validation_summary_csv",
    "output_validation_summary_json",
}


def require_osgeo() -> None:
    """Fail clearly when a real GIS run is attempted outside QGIS/OSGeo Python."""

    if gdal is None or osr is None:
        raise RuntimeError(
            "Step 4 raster execution requires GDAL Python bindings. "
            "Run this script through run_step4.bat after setting OSGEO4W_ROOT, "
            "or add the QGIS Python launcher to PATH."
        )
@dataclass(frozen=True)
class Step4Params:
    """User-facing settings loaded from the JSON config."""

    dem_fel_raster: str
    dinf_flow_raster: str
    source_raster: str
    dfi_raster: str
    output_mask: str
    output_dynamic_alpha: str = ""
    output_beta_angle: str = ""
    output_dfs: str = ""
    alpha_raster: str = ""
    output_depositional_mask: str = ""
    output_summary_json: str = ""
    observed_full_landslide_footprint_raster: str = ""
    observed_depositional_zone_raster: str = ""
    validation_domain_raster: str = ""
    output_validation_summary_csv: str = ""
    output_validation_summary_json: str = ""
    proportion_threshold: float = 0.2
    dfi_mid: float = 0.5
    alpha_gain_per_meter: float = DEFAULT_ALPHA_GAIN_PER_METER
    write_debug_rasters: bool = False
    deprecated_fields_present: tuple[str, ...] = ()


@dataclass(frozen=True)
class RasterSpec:
    """Raster array plus the grid metadata needed for alignment checks."""

    path: str
    rows: int
    cols: int
    geotransform: tuple[float, float, float, float, float, float]
    projection_wkt: str
    nodata: float | None
    array: np.ndarray


@dataclass(frozen=True)
class RasterMetadata:
    """Raster grid metadata without loading pixel arrays."""

    path: str
    rows: int
    cols: int
    geotransform: tuple[float, float, float, float, float, float]
    projection_wkt: str
    nodata: float | None


@dataclass
class DynamicAlphaCoreResult:
    """All output rasters and counters produced by the propagation engine."""

    dynamic_alpha_out: np.ndarray
    beta_angle_out: np.ndarray
    dfs_out: np.ndarray
    mask_out: np.ndarray
    depositional_mask_out: np.ndarray
    accepted_beta_out: np.ndarray | None
    parent_dynamic_alpha_out: np.ndarray | None
    valid_cells: int
    terrain_valid_cells: int
    alpha_valid_cells: int
    source_cells: int
    non_source_terrain_valid_cells: int
    runout_cells: int
    pure_depositional_cells: int
    processed_cells: int
    remaining_dependency_cells: int
    remaining_dependency_fraction: float
    candidate_evaluations: int
    accepted_candidate_propagations: int
    source_cells_without_valid_alpha: int
    terrain_valid_cells_without_valid_dfi: int
    candidate_cells_skipped_missing_dfi: int
    dfi_clamped_count: int

def _parse_bool(value: object, *, field_name: str, default: bool = False) -> bool:
    raw = default if value is None else value
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"{field_name} must be a boolean value.")


def _require_text(cfg: dict[str, Any], field_name: str) -> str:
    raw = str(cfg.get(field_name, "")).strip()
    if not raw:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return raw


def _resolve_config_path(config_dir: Path, value: object) -> str:
    """Resolve relative paths from the config file folder; keep absolute paths unchanged."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    return str((config_dir / path).resolve())


def load_step4_params(config_path: str) -> Step4Params:
    """Read the JSON config and validate the simplified Step 4 interface."""

    cfg_path = Path(config_path).resolve()
    with cfg_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a JSON object: {cfg_path}")
    cfg = {str(key): value for key, value in payload.items()}
    for field_name, message in REMOVED_CONFIG_FIELDS.items():
        if field_name in cfg:
            raise ValueError(message)
    unknown_fields = sorted(set(cfg) - ACTIVE_CONFIG_FIELDS - set(DEPRECATED_CONFIG_FIELDS))
    if unknown_fields:
        raise ValueError(
            "Unknown Step 4 config field(s): "
            + ", ".join(unknown_fields)
            + ". Correct spelling errors or remove unsupported settings."
        )
    config_dir = cfg_path.parent

    deprecated_fields = tuple(
        field for field in DEPRECATED_CONFIG_FIELDS if str(cfg.get(field, "")).strip()
    )
    if deprecated_fields:
        print(
            "Warning: deprecated legacy alpha-table fields are ignored by standalone Step 4: "
            + ", ".join(deprecated_fields)
        )

    params = Step4Params(
        dem_fel_raster=_resolve_config_path(config_dir, _require_text(cfg, "dem_fel_raster")),
        dinf_flow_raster=_resolve_config_path(config_dir, _require_text(cfg, "dinf_flow_raster")),
        source_raster=_resolve_config_path(config_dir, _require_text(cfg, "source_raster")),
        dfi_raster=_resolve_config_path(config_dir, _require_text(cfg, "dfi_raster")),
        alpha_raster=_resolve_config_path(config_dir, _require_text(cfg, "alpha_raster")),
        output_mask=_resolve_config_path(config_dir, _require_text(cfg, "output_mask")),
        output_dynamic_alpha=_resolve_config_path(
            config_dir,
            _require_text(cfg, "output_dynamic_alpha"),
        ),
        output_beta_angle=_resolve_config_path(
            config_dir,
            _require_text(cfg, "output_beta_angle"),
        ),
        output_dfs=_resolve_config_path(
            config_dir,
            _require_text(cfg, "output_dfs"),
        ),
        output_depositional_mask=_resolve_config_path(
            config_dir,
            _require_text(cfg, "output_depositional_mask"),
        ),
        output_summary_json=_resolve_config_path(
            config_dir,
            _require_text(cfg, "output_summary_json"),
        ),
        observed_full_landslide_footprint_raster=_resolve_config_path(
            config_dir, cfg.get("observed_full_landslide_footprint_raster", "")
        ),
        observed_depositional_zone_raster=_resolve_config_path(
            config_dir, cfg.get("observed_depositional_zone_raster", "")
        ),
        validation_domain_raster=_resolve_config_path(config_dir, cfg.get("validation_domain_raster", "")),
        output_validation_summary_csv=_resolve_config_path(
            config_dir, cfg.get("output_validation_summary_csv", "")
        ),
        output_validation_summary_json=_resolve_config_path(
            config_dir, cfg.get("output_validation_summary_json", "")
        ),
        proportion_threshold=float(cfg.get("proportion_threshold", 0.2)),
        dfi_mid=float(cfg.get("dfi_mid", 0.5)),
        alpha_gain_per_meter=float(cfg.get("alpha_gain_per_meter", DEFAULT_ALPHA_GAIN_PER_METER)),
        write_debug_rasters=_parse_bool(
            cfg.get("write_debug_rasters"), field_name="write_debug_rasters", default=False
        ),
        deprecated_fields_present=deprecated_fields,
    )
    validate_scalar_params(params)
    validate_path_contract(params)
    return params


def validate_scalar_params(params: Step4Params) -> None:
    """Catch impossible numeric settings before any raster work starts."""

    if not math.isfinite(float(params.proportion_threshold)):
        raise ValueError("proportion_threshold must be finite.")
    if not (0.0 <= float(params.proportion_threshold) <= 1.0):
        raise ValueError("proportion_threshold must be within [0, 1].")
    if not math.isfinite(float(params.dfi_mid)):
        raise ValueError("dfi_mid must be finite.")
    if not (0.0 <= float(params.dfi_mid) <= 1.0):
        raise ValueError("dfi_mid must be within [0, 1].")
    if not math.isfinite(float(params.alpha_gain_per_meter)):
        raise ValueError("alpha_gain_per_meter must be finite.")
    if float(params.alpha_gain_per_meter) < 0.0:
        raise ValueError("alpha_gain_per_meter must be >= 0.")


def _normalized_contract_path(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def step4_input_paths(params: Step4Params) -> dict[str, str]:
    return {
        "dem_fel_raster": params.dem_fel_raster,
        "dinf_flow_raster": params.dinf_flow_raster,
        "source_raster": params.source_raster,
        "dfi_raster": params.dfi_raster,
        "alpha_raster": params.alpha_raster,
        "observed_full_landslide_footprint_raster": params.observed_full_landslide_footprint_raster,
        "observed_depositional_zone_raster": params.observed_depositional_zone_raster,
        "validation_domain_raster": params.validation_domain_raster,
    }


def step4_output_paths(params: Step4Params) -> dict[str, str]:
    outputs = {
        "output_dynamic_alpha": params.output_dynamic_alpha,
        "output_beta_angle": params.output_beta_angle,
        "output_dfs": params.output_dfs,
        "output_mask": params.output_mask,
        "output_depositional_mask": params.output_depositional_mask,
        "output_summary_json": params.output_summary_json,
        "output_validation_summary_csv": params.output_validation_summary_csv,
        "output_validation_summary_json": params.output_validation_summary_json,
    }
    debug_paths = build_debug_output_paths(params.output_dynamic_alpha, params.write_debug_rasters)
    if debug_paths["parent_dynamic_alpha"]:
        outputs["output_parent_dynamic_alpha"] = debug_paths["parent_dynamic_alpha"]
    return outputs


def validate_path_contract(params: Step4Params) -> None:
    """Prevent destructive path aliasing before any output is opened."""

    normalized_inputs = {
        _normalized_contract_path(path): label
        for label, path in step4_input_paths(params).items()
        if path
    }
    normalized_outputs: dict[str, str] = {}
    for label, path in step4_output_paths(params).items():
        if not path:
            continue
        normalized = _normalized_contract_path(path)
        if normalized in normalized_inputs:
            raise ValueError(
                f"{label} must not overwrite input {normalized_inputs[normalized]}: {path}"
            )
        if normalized in normalized_outputs:
            raise ValueError(
                f"{label} duplicates output {normalized_outputs[normalized]}: {path}"
            )
        normalized_outputs[normalized] = label

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_raster(path: str, label: str) -> RasterSpec:
    """Read one raster band as float64 and keep its grid metadata."""

    require_osgeo()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} raster does not exist: {path}")
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Could not open {label} raster: {path}")
    if ds.RasterCount < 1:
        raise RuntimeError(f"{label} raster has no bands: {path}")

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    if arr is None:
        raise RuntimeError(f"Failed reading raster array for {label}: {path}")

    gt_raw = ds.GetGeoTransform(can_return_null=True)
    if gt_raw is None or len(gt_raw) != 6:
        raise RuntimeError(f"{label} raster has missing or invalid geotransform: {path}")

    return RasterSpec(
        path=os.path.abspath(path),
        rows=int(ds.RasterYSize),
        cols=int(ds.RasterXSize),
        geotransform=tuple(float(v) for v in gt_raw),  # type: ignore[arg-type]
        projection_wkt=ds.GetProjectionRef() or "",
        nodata=None if band.GetNoDataValue() is None else float(band.GetNoDataValue()),
        array=np.asarray(arr, dtype=np.float64),
    )


def read_raster_metadata(path: str, label: str) -> RasterMetadata:
    """Read only grid metadata for fast preflight and output verification."""

    require_osgeo()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} raster does not exist: {path}")
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Could not open {label} raster: {path}")
    if ds.RasterCount < 1:
        raise RuntimeError(f"{label} raster has no bands: {path}")
    band = ds.GetRasterBand(1)
    gt_raw = ds.GetGeoTransform(can_return_null=True)
    if gt_raw is None or len(gt_raw) != 6:
        raise RuntimeError(f"{label} raster has missing or invalid geotransform: {path}")
    return RasterMetadata(
        path=os.path.abspath(path),
        rows=int(ds.RasterYSize),
        cols=int(ds.RasterXSize),
        geotransform=tuple(float(v) for v in gt_raw),  # type: ignore[arg-type]
        projection_wkt=ds.GetProjectionRef() or "",
        nodata=None if band.GetNoDataValue() is None else float(band.GetNoDataValue()),
    )


def srs_from_wkt(wkt: str) -> osr.SpatialReference:
    require_osgeo()
    srs = osr.SpatialReference()
    if wkt:
        srs.ImportFromWkt(wkt)
    if hasattr(srs, "SetAxisMappingStrategy"):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def geotransforms_equal(
    a: tuple[float, float, float, float, float, float],
    b: tuple[float, float, float, float, float, float],
    tol: float = 1e-9,
) -> bool:
    return all(abs(a[i] - b[i]) <= tol for i in range(6))




CRS_NUMERIC_TOLERANCE = 1e-6
CRS_PROJECTION_PARAMETERS = (
    "latitude_of_origin",
    "central_meridian",
    "scale_factor",
    "false_easting",
    "false_northing",
    "standard_parallel_1",
    "standard_parallel_2",
    "longitude_of_center",
    "latitude_of_center",
)


def _authority_pair(srs: osr.SpatialReference) -> tuple[str, str] | None:
    clone = srs.Clone()
    try:
        clone.AutoIdentifyEPSG()
    except Exception:
        pass
    name = clone.GetAuthorityName(None)
    code = clone.GetAuthorityCode(None)
    if name and code:
        return (str(name).upper(), str(code))
    return None


def _numeric_srs_value(srs: osr.SpatialReference, method_name: str) -> float | None:
    try:
        value = getattr(srs, method_name)()
    except Exception:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric_values_match(left: float | None, right: float | None) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) <= CRS_NUMERIC_TOLERANCE


def spatial_references_equivalent(reference_wkt: str, other_wkt: str) -> bool:
    """Compare CRS objects while tolerating harmless WKT precision differences.

    Some derived rasters in this workspace carry the same PRS92 / UTM Zone 51N
    grid but differ in spheroid inverse-flattening precision by a few decimal
    places. GDAL/OSR IsSame can reject that metadata even when the projection,
    units, and numeric parameters are effectively identical.
    """

    left = srs_from_wkt(reference_wkt)
    right = srs_from_wkt(other_wkt)
    if bool(left.IsSame(right)):
        return True

    left_auth = _authority_pair(left)
    right_auth = _authority_pair(right)
    if left_auth is not None and left_auth == right_auth:
        return True

    if bool(left.IsProjected()) != bool(right.IsProjected()):
        return False
    if bool(left.IsGeographic()) != bool(right.IsGeographic()):
        return False

    left_projection = (left.GetAttrValue("PROJECTION") or "").lower()
    right_projection = (right.GetAttrValue("PROJECTION") or "").lower()
    if left_projection != right_projection:
        return False

    for method_name in ("GetSemiMajor", "GetSemiMinor", "GetInvFlattening", "GetLinearUnits", "GetAngularUnits"):
        if not _numeric_values_match(
            _numeric_srs_value(left, method_name),
            _numeric_srs_value(right, method_name),
        ):
            return False

    for parameter_name in CRS_PROJECTION_PARAMETERS:
        try:
            left_value = float(left.GetProjParm(parameter_name, 0.0))
            right_value = float(right.GetProjParm(parameter_name, 0.0))
        except Exception:
            continue
        if abs(left_value - right_value) > CRS_NUMERIC_TOLERANCE:
            return False

    return True


def validate_alignment(reference: RasterSpec, other: RasterSpec, label: str) -> None:
    """Reject rasters that do not share the DEM grid exactly."""

    if reference.rows != other.rows or reference.cols != other.cols:
        raise ValueError(
            f"{label} size mismatch: expected {reference.cols}x{reference.rows}, "
            f"got {other.cols}x{other.rows} ({other.path})."
        )
    if not geotransforms_equal(reference.geotransform, other.geotransform):
        raise ValueError(f"{label} geotransform mismatch with reference grid: {other.path}")
    if not spatial_references_equivalent(reference.projection_wkt, other.projection_wkt):
        raise ValueError(f"{label} CRS mismatch with reference grid: {other.path}")


def assert_projected_metric_grid(reference: RasterSpec) -> tuple[float, float]:
    """Confirm the grid is projected, north-up, and measured in map units."""

    gt = reference.geotransform
    if abs(gt[2]) > 1e-12 or abs(gt[4]) > 1e-12:
        raise ValueError("Rotated/sheared rasters are not supported. Use north-up rasters.")
    dx = abs(float(gt[1]))
    dy = abs(float(gt[5]))
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("Invalid pixel size in reference raster.")
    srs = srs_from_wkt(reference.projection_wkt)
    if bool(srs.IsGeographic()):
        raise ValueError("Geographic CRS is not supported. Use projected rasters in meters.")
    if not bool(srs.IsProjected()):
        raise ValueError("Step 4 requires projected CRS rasters.")
    return dx, dy


def valid_mask(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    mask = np.isfinite(arr)
    if nodata is not None:
        mask &= ~np.isclose(arr, nodata, atol=0.0, rtol=0.0)
    return mask
def _validation_pixel_area(reference: RasterSpec) -> float:
    return abs(float(reference.geotransform[1]) * float(reference.geotransform[5]))


def _optional_validation_raster(
    path: str,
    reference: RasterSpec,
    label: str,
    warnings: list[str],
) -> RasterSpec | None:
    if not path:
        warnings.append(f"{label} was not provided; skipping related validation.")
        return None
    if not os.path.exists(path):
        warnings.append(f"{label} is missing: {path}")
        return None
    try:
        raster = read_raster(os.path.abspath(path), label)
        validate_alignment(reference, raster, label)
    except Exception as exc:
        warnings.append(f"{label} could not be used for validation: {exc}")
        return None
    return raster


def _metric_dict(
    *,
    group: str,
    predicted: np.ndarray,
    observed: np.ndarray,
    valid: np.ndarray,
    pixel_area: float,
    warnings: list[str],
    notes: list[str],
) -> dict[str, Any]:
    positive_pred = (np.asarray(predicted) > 0) & valid
    positive_obs = (np.asarray(observed) > 0) & valid
    observed_positive_count = int(np.count_nonzero(positive_obs))
    predicted_positive_count = int(np.count_nonzero(positive_pred))
    if int(np.count_nonzero(valid)) == 0:
        warnings.append(f"{group}: validation domain contains no valid cells.")
    if observed_positive_count == 0:
        warnings.append(f"{group}: observed raster contains no positive cells in the validation domain.")
    if predicted_positive_count == 0:
        warnings.append(f"{group}: predicted raster contains no positive cells in the validation domain.")

    tp = int(np.count_nonzero(positive_pred & positive_obs))
    fp = int(np.count_nonzero(positive_pred & (~positive_obs)))
    fn = int(np.count_nonzero((~positive_pred) & positive_obs))
    tn = int(np.count_nonzero((~positive_pred) & (~positive_obs) & valid))

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    false_positive_rate = fp / (fp + tn) if (fp + tn) else None
    false_negative_rate = fn / (fn + tp) if (fn + tp) else None
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else None

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "f1": f1,
        "iou_jaccard": iou,
        "validation_cells": int(np.count_nonzero(valid)),
        "predicted_positive_cells": predicted_positive_count,
        "observed_positive_cells": observed_positive_count,
        "predicted_positive_area": float(predicted_positive_count * pixel_area),
        "observed_positive_area": float(observed_positive_count * pixel_area),
        "intersection_area": float(tp * pixel_area),
        "overprediction_area": float(fp * pixel_area),
        "underprediction_area": float(fn * pixel_area),
        "warnings": warnings,
        "notes": notes,
    }


def run_validation_metrics(
    params: Step4Params,
    reference: RasterSpec,
    core_result: DynamicAlphaCoreResult,
    source: RasterSpec,
) -> dict[str, Any]:
    """Validate full runout and pure depositional outputs when observed rasters are configured."""

    pixel_area = _validation_pixel_area(reference)
    output: dict[str, Any] = {
        "runout_footprint_validation": None,
        "depositional_zone_validation": None,
        "warnings": [],
    }
    shared_warnings: list[str] = output["warnings"]
    domain_raster = None
    if params.validation_domain_raster:
        domain_raster = _optional_validation_raster(
            params.validation_domain_raster,
            reference,
            "validation_domain_raster",
            shared_warnings,
        )

    domain_positive: np.ndarray | None = None
    if domain_raster is not None:
        domain_positive = valid_mask(domain_raster.array, domain_raster.nodata) & (domain_raster.array > 0)
        if not bool(np.any(domain_positive)):
            shared_warnings.append("validation_domain_raster contains no positive cells; validation may be empty.")

    observed_full = _optional_validation_raster(
        params.observed_full_landslide_footprint_raster,
        reference,
        "observed_full_landslide_footprint_raster",
        shared_warnings,
    )
    if observed_full is not None:
        valid = valid_mask(observed_full.array, observed_full.nodata)
        if domain_positive is not None:
            valid &= domain_positive
        runout_warnings: list[str] = []
        output["runout_footprint_validation"] = _metric_dict(
            group="runout_footprint_validation",
            predicted=core_result.mask_out,
            observed=observed_full.array,
            valid=valid,
            pixel_area=pixel_area,
            warnings=runout_warnings,
            notes=[
                "Predicted raster is the full modeled runout mask.",
                "Observed raster is the full clean landslide footprint.",
                "Source cells inside the observed landslide footprint count as TP when predicted as runout.",
            ],
        )

    observed_deposition = _optional_validation_raster(
        params.observed_depositional_zone_raster,
        reference,
        "observed_depositional_zone_raster",
        shared_warnings,
    )
    if observed_deposition is not None:
        valid = valid_mask(observed_deposition.array, observed_deposition.nodata)
        depositional_notes = [
            "Predicted raster is the pure modeled depositional mask: full runout excluding source cells.",
            "Observed raster is the observed depositional zone.",
        ]
        if domain_positive is not None:
            valid &= domain_positive
        source_domain = valid_mask(source.array, source.nodata) & (source.array > 0)
        valid &= ~source_domain
        depositional_notes.append(
            "Source cells were excluded from depositional validation because the source raster is available."
        )
        depositional_warnings: list[str] = []
        if params.output_depositional_mask and os.path.abspath(params.output_depositional_mask) == os.path.abspath(params.output_mask):
            depositional_warnings.append(
                "output_depositional_mask is the same path as output_mask; depositional validation should not use the full runout mask."
            )
        output["depositional_zone_validation"] = _metric_dict(
            group="depositional_zone_validation",
            predicted=core_result.depositional_mask_out,
            observed=observed_deposition.array,
            valid=valid,
            pixel_area=pixel_area,
            warnings=depositional_warnings,
            notes=depositional_notes,
        )
    return output


def write_validation_summaries(params: Step4Params, validation: dict[str, Any]) -> None:
    if params.output_validation_summary_json:
        _ensure_parent_dir(params.output_validation_summary_json)
        with open(params.output_validation_summary_json, "w", encoding="utf-8") as handle:
            json.dump(validation, handle, indent=2)
    if params.output_validation_summary_csv:
        _ensure_parent_dir(params.output_validation_summary_csv)
        rows: list[dict[str, Any]] = []
        for group in ("runout_footprint_validation", "depositional_zone_validation"):
            metrics = validation.get(group)
            if not isinstance(metrics, dict):
                continue
            row = {"validation_group": group}
            for key in (
                "tp",
                "fp",
                "tn",
                "fn",
                "precision",
                "recall",
                "specificity",
                "false_positive_rate",
                "false_negative_rate",
                "f1",
                "iou_jaccard",
                "validation_cells",
                "predicted_positive_cells",
                "observed_positive_cells",
                "predicted_positive_area",
                "observed_positive_area",
                "intersection_area",
                "overprediction_area",
                "underprediction_area",
            ):
                row[key] = metrics.get(key)
            row["warnings"] = " | ".join(str(item) for item in metrics.get("warnings", []))
            row["notes"] = " | ".join(str(item) for item in metrics.get("notes", []))
            rows.append(row)
        fields = [
            "validation_group",
            "tp",
            "fp",
            "tn",
            "fn",
            "precision",
            "recall",
            "specificity",
            "false_positive_rate",
            "false_negative_rate",
            "f1",
            "iou_jaccard",
            "validation_cells",
            "predicted_positive_cells",
            "observed_positive_cells",
            "predicted_positive_area",
            "observed_positive_area",
            "intersection_area",
            "overprediction_area",
            "underprediction_area",
            "warnings",
            "notes",
        ]
        with open(params.output_validation_summary_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

def derive_related_output_path(base_raster_path: str, replacement: str) -> str:
    directory = os.path.dirname(os.path.abspath(base_raster_path))
    stem, ext = os.path.splitext(os.path.basename(base_raster_path))
    return os.path.join(directory, f"{stem}_{replacement}{ext or '.tif'}")


def build_debug_output_paths(output_dynamic_alpha: str, write_debug_rasters: bool) -> dict[str, str]:
    if not write_debug_rasters:
        return {"parent_dynamic_alpha": ""}
    return {
        "parent_dynamic_alpha": derive_related_output_path(output_dynamic_alpha, "parent_dynamic_alpha"),
    }

