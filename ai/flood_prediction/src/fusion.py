"""Uncertainty-aware evidence aggregation for ResQTwin.

This module keeps satellite inundation evidence separate from environmental
risk. It does not calculate a combined flood probability.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio


FUSION_VERSION = "0.1.0"


def _parse_timestamp(value, field_name, required=True):
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required.")
        return None

    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a valid ISO-8601 timestamp.")

    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone.")

    return parsed

def _expected_risk_category(score):
    """Return the category defined by the Model 2 scoring boundaries."""
    if score is None:
        return "unavailable"
    if score < 25:
        return "low"
    if score < 50:
        return "moderate"
    if score < 75:
        return "high"
    return "very_high"

def _is_nodata(values, nodata):
    if nodata is None:
        return np.zeros(values.shape, dtype=bool)
    if np.isnan(nodata):
        return np.isnan(values)
    return values == nodata


def _summarize_satellite(assessment, fusion_time):
    required = (
        "probability_raster_path",
        "mask_raster_path",
        "inference_timestamp",
        "threshold",
        "model_version",
    )
    for field in required:
        if field not in assessment or assessment[field] is None:
            raise ValueError(f"Satellite assessment is missing '{field}'.")

    inference_time = _parse_timestamp(
        assessment["inference_timestamp"], "inference_timestamp"
    )
    acquisition_time = _parse_timestamp(
        assessment.get("acquisition_timestamp"),
        "acquisition_timestamp",
        required=False,
    )

    try:
        threshold = float(assessment["threshold"])
    except (TypeError, ValueError):
        raise ValueError("Satellite threshold must be numeric.")

    if not 0.0 < threshold < 1.0:
        raise ValueError("Satellite threshold must be between 0 and 1.")

    probability_path = Path(assessment["probability_raster_path"])
    mask_path = Path(assessment["mask_raster_path"])

    if not probability_path.is_file():
        raise FileNotFoundError(f"Probability raster not found: {probability_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask raster not found: {mask_path}")

    with rasterio.open(probability_path) as probability_src, \
            rasterio.open(mask_path) as mask_src:

        if probability_src.count != 1 or mask_src.count != 1:
            raise ValueError("Both satellite rasters must contain exactly one band.")

        if probability_src.crs is None or mask_src.crs is None:
            raise ValueError("Both satellite rasters must have a CRS.")

        if (
            probability_src.crs != mask_src.crs
            or probability_src.transform != mask_src.transform
            or probability_src.width != mask_src.width
            or probability_src.height != mask_src.height
        ):
            raise ValueError(
                "Probability and mask rasters are not spatially aligned."
            )

        declared_crs = assessment.get("crs")
        if declared_crs and probability_src.crs != rasterio.crs.CRS.from_string(
            declared_crs
        ):
            raise ValueError("Declared satellite CRS does not match raster CRS.")

        probability = probability_src.read(1)
        mask = mask_src.read(1)

        probability_valid = (
            probability_src.read_masks(1) > 0
        ) & np.isfinite(probability) & ~_is_nodata(
            probability, probability_src.nodata
        )

        mask_valid = (
            mask_src.read_masks(1) > 0
        ) & ~_is_nodata(mask, mask_src.nodata) & (mask != 255)

        valid = probability_valid & mask_valid

        if not valid.any():
            raise ValueError("Satellite rasters contain no overlapping valid pixels.")

        valid_probabilities = probability[valid]
        valid_mask = mask[valid]

        if np.any((valid_probabilities < 0) | (valid_probabilities > 1)):
            raise ValueError("Valid satellite probabilities must be between 0 and 1.")

        if not np.all(np.isin(valid_mask, [0, 1])):
            raise ValueError("Valid satellite mask values must be 0 or 1.")

        expected_mask = (
            valid_probabilities >= threshold
        ).astype(np.uint8)

        inconsistent_pixels = int(
            np.count_nonzero(valid_mask != expected_mask)
        )

        if inconsistent_pixels > 0:
            raise ValueError(
                "Satellite mask is inconsistent with the probability "
                f"raster and threshold: {inconsistent_pixels} mismatched pixels."
            )

        valid_count = int(valid.sum())
        flooded_count = int((valid_mask == 1).sum())
        flooded_fraction = flooded_count / valid_count

        pixel_area_map_units = abs(
            probability_src.transform.a * probability_src.transform.e
            - probability_src.transform.b * probability_src.transform.d
        )

        return {
            "availability": "available",
            "inference_timestamp": inference_time.isoformat(),
            "acquisition_timestamp": (
                acquisition_time.isoformat() if acquisition_time else None
            ),
            "age_since_inference_hours": round(
                (fusion_time - inference_time).total_seconds() / 3600, 2
            ),
            "freshness_status": "not_assessed_no_validated_threshold",
            "crs": probability_src.crs.to_string(),
            "width": probability_src.width,
            "height": probability_src.height,
            "valid_pixel_count": valid_count,
            "flooded_pixel_count": flooded_count,
            "flooded_pixel_fraction": round(flooded_fraction, 6),
            "mean_flood_probability": round(
                float(np.mean(valid_probabilities)), 6
            ),
            "maximum_flood_probability": round(
                float(np.max(valid_probabilities)), 6
            ),
            "threshold": threshold,
            "model_version": str(assessment["model_version"]),
            "pixel_area_map_units_squared": pixel_area_map_units,
            "validity_note": (
                "Statistics describe overlapping valid raster pixels only; "
                "they do not represent the percentage of Chennai flooded."
            ),
        }



def fuse_assessments(
    satellite_assessment,
    environmental_assessment,
    fusion_timestamp=None,
):
    """Aggregate satellite and environmental evidence without score averaging."""

    now = (
        _parse_timestamp(fusion_timestamp, "fusion_timestamp")
        if fusion_timestamp is not None
        else datetime.now(timezone.utc)
    )

    warnings = []
    satellite = None
    environmental = None

    # --------------------------------------------------
    # 1. Process satellite assessment
    # --------------------------------------------------
    if satellite_assessment is not None:
        satellite = _summarize_satellite(satellite_assessment, now)

        if satellite["age_since_inference_hours"] < 0:
            warnings.append(
                "Satellite inference timestamp is in the future."
            )

    # --------------------------------------------------
    # 2. Process environmental assessment
    # --------------------------------------------------
    if environmental_assessment is not None:
        environmental = dict(environmental_assessment)

        score = environmental.get("environmental_risk_score")

        if score is not None:
            try:
                score = float(score)
            except (TypeError, ValueError):
                raise ValueError(
                    "Environmental risk score must be numeric or null."
                )

            if not 0 <= score <= 100:
                raise ValueError(
                    "Environmental risk score must be between 0 and 100."
                )

            environmental["environmental_risk_score"] = score

        # Validate score/category consistency.
        expected_category = _expected_risk_category(score)
        supplied_category = environmental.get("risk_category")

        if supplied_category != expected_category:
            raise ValueError(
                "Environmental risk category does not match its score. "
                f"Expected '{expected_category}', got '{supplied_category}'."
            )

        # Validate environmental prediction timestamp.
        prediction_time = _parse_timestamp(
            environmental.get("prediction_timestamp"),
            "environmental prediction_timestamp",
        )

        environmental["prediction_timestamp"] = (
            prediction_time.isoformat()
        )

        # Preserve environmental warnings.
        warnings.extend(environmental.get("warnings") or [])

        # Determine environmental availability.
        environmental["availability"] = (
            "unavailable" if score is None else "available"
        )

    # --------------------------------------------------
    # 3. Determine individual source availability
    # --------------------------------------------------
    satellite_available = satellite is not None

    environmental_available = (
        environmental is not None
        and environmental["availability"] == "available"
    )

    # --------------------------------------------------
    # 4. Determine overall availability
    # --------------------------------------------------
    if satellite_available and environmental_available:
        overall_status = "both_sources_available"

    elif satellite_available:
        overall_status = "satellite_only"

    elif environmental_available:
        overall_status = "environmental_only"

    else:
        overall_status = "insufficient_data"

    # --------------------------------------------------
    # 5. Preserve satellite freshness uncertainty
    # --------------------------------------------------
    if satellite is not None:
        warnings.append(
            "Satellite freshness is not assessed because no validated "
            "freshness threshold is configured."
        )

    # --------------------------------------------------
    # 6. Flag future satellite timestamps
    # --------------------------------------------------
    if (
        satellite is not None
        and satellite["age_since_inference_hours"] < 0
    ):
        overall_status = "temporal_validation_required"

    # --------------------------------------------------
    # 7. Preserve environmental missing-input information
    # --------------------------------------------------
    if environmental is not None:
        warnings.extend(
            environmental.get("missing_inputs") or []
        )

    # Remove duplicate warnings and normalize their representation.
    warnings = sorted(
        set(str(item) for item in warnings)
    )

    # --------------------------------------------------
    # 8. Return fusion assessment
    # --------------------------------------------------
    return {
        "fusion_timestamp": now.isoformat(),
        "fusion_version": FUSION_VERSION,

        "satellite_inundation": satellite,
        "environmental_risk": environmental,

        "overall_status": overall_status,

        "uncertainty_status": (
            "source_uncertainties_preserved_no_combined_probability"
        ),

        "data_quality": {
            "satellite_available": satellite is not None,
            "environmental_assessment_provided": (
                environmental is not None
            ),
            "warnings_count": len(warnings),
        },

        "warnings": warnings,

        "interpretation": (
            "Satellite inundation statistics and environmental risk are "
            "separate evidence streams. No combined flood probability or "
            "calibrated overall risk score is produced."
        ),
    }
