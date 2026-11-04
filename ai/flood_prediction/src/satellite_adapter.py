"""Adapter for converting Model 1 metadata into Fusion input."""

import json
from datetime import datetime
from pathlib import Path


def load_satellite_assessment(metadata_path):
    """Load Model 1 metadata and create a Fusion-compatible assessment."""

    metadata_path = Path(metadata_path)

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Inference metadata not found: {metadata_path}"
        )

    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid inference metadata JSON: {exc}"
        ) from exc

    if not isinstance(metadata, dict):
        raise ValueError("Inference metadata must be a JSON object.")

    required_fields = [
        "inference_completed_at",
        "threshold",
        "model_version",
        "outputs",
        "scene_metadata",
    ]

    for field in required_fields:
        if field not in metadata or metadata[field] is None:
            raise ValueError(
                f"Metadata is missing required field: {field}"
            )

    # Validate inference timestamp.
    timestamp = metadata["inference_completed_at"]

    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ValueError("Invalid inference timestamp.")

    try:
        parsed_timestamp = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError(
            "Inference timestamp must be valid ISO-8601."
        ) from exc

    if parsed_timestamp.tzinfo is None:
        raise ValueError(
            "Inference timestamp must include timezone information."
        )

    # Validate threshold.
    threshold = metadata["threshold"]

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 <= threshold <= 1
    ):
        raise ValueError(
            "Threshold must be a numeric value between 0 and 1."
        )

    # Validate model version.
    model_version = metadata["model_version"]

    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("Model version must be a non-empty string.")

    # Validate output paths.
    outputs = metadata["outputs"]

    if not isinstance(outputs, dict):
        raise ValueError("Metadata outputs must be a JSON object.")

    for field in ["probability_raster", "mask_raster"]:
        if not isinstance(outputs.get(field), str) or not outputs[field].strip():
            raise ValueError(
                f"Metadata is missing a valid output path: {field}"
            )

    probability_path = Path(outputs["probability_raster"])
    mask_path = Path(outputs["mask_raster"])

    if not probability_path.is_file():
        raise FileNotFoundError(
            f"Probability raster not found: {probability_path}"
        )

    if not mask_path.is_file():
        raise FileNotFoundError(
            f"Mask raster not found: {mask_path}"
        )

    # Validate scene metadata.
    scene = metadata["scene_metadata"]

    if not isinstance(scene, dict):
        raise ValueError(
            "Scene metadata must be a JSON object."
        )

    crs = scene.get("crs")

    if not isinstance(crs, str) or not crs.strip():
        raise ValueError(
            "Scene metadata is missing a valid CRS."
        )

    return {
        "probability_raster_path": str(probability_path),
        "mask_raster_path": str(mask_path),
        "inference_timestamp": timestamp,
        "acquisition_timestamp": metadata.get("acquisition_timestamp"),
        "threshold": threshold,
        "model_version": model_version,
        "crs": crs,
    }