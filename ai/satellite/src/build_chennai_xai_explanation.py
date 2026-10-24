import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[3]

PROB_PATH = ROOT / "data/processed/chennai/sentinel1/gcc/inference/flood_probability.tif"
MASK_PATH = ROOT / "data/processed/chennai/sentinel1/gcc/inference/flood_mask.tif"
CAM_PATH = ROOT / "data/processed/chennai/explainability/gradcam_tile.tif"

OUT_DIR = ROOT / "data/processed/chennai/explainability/day7"
JSON_PATH = OUT_DIR / "explanation_summary.json"
MD_PATH = OUT_DIR / "explanation_summary.md"


def valid_pixels(array, nodata):
    valid = np.isfinite(array)
    if nodata is not None:
        valid &= array != nodata
    return valid


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(PROB_PATH) as prob, \
         rasterio.open(MASK_PATH) as mask, \
         rasterio.open(CAM_PATH) as cam:

        if (prob.crs != mask.crs or
                prob.transform != mask.transform or
                prob.shape != mask.shape):
            raise ValueError("Probability and mask rasters do not share a grid.")

        if prob.crs != cam.crs:
            raise ValueError("Grad-CAM CRS differs from source rasters.")

        col_f, row_f = (~prob.transform) * (cam.transform.c, cam.transform.f)
        row, col = round(row_f), round(col_f)

        if abs(row_f - row) > 1e-6 or abs(col_f - col) > 1e-6:
            raise ValueError("Grad-CAM origin is not aligned to the source grid.")

        if (row < 0 or col < 0 or
                row + cam.height > prob.height or
                col + cam.width > prob.width):
            raise ValueError("Grad-CAM window falls outside source rasters.")

        window = Window(col, row, cam.width, cam.height)
        p = prob.read(1, window=window)
        m = mask.read(1, window=window)
        c = cam.read(1)

        pv = valid_pixels(p, prob.nodata)
        mv = valid_pixels(m, mask.nodata)
        cv = valid_pixels(c, cam.nodata)
        joint = pv & mv & cv
        predicted_flood = joint & (m == 1)
        predicted_nonflood = joint & (m == 0)

        if not joint.any():
            raise ValueError("No pixels are valid across all three rasters.")

        if not predicted_flood.any():
            raise ValueError("No jointly valid predicted-flood pixels in Grad-CAM tile.")

        # Descriptive statistics only; these are not calibrated confidence values.
        cam_valid = c[cv]
        high_threshold = float(np.quantile(cam_valid, 0.90))
        high_attr = joint & (c >= high_threshold)
        high_attr_count = int(high_attr.sum())
        high_attr_flood_count = int((high_attr & (m == 1)).sum())

        def mean_or_none(values):
            return float(np.mean(values)) if values.size else None

        p_joint = p[joint]
        c_joint = c[joint]
        p_flood = p[predicted_flood]
        p_nonflood = p[predicted_nonflood]
        cam_flood = c[predicted_flood]
        cam_nonflood = c[predicted_nonflood]

        summary = {
            "schema_version": "1.0",
            "project": "ResQTwin",
            "case_study": "Chennai urban flooding",
            "artifact_type": "single_tile_xai_explanation",
            "status": "generated",
            "inputs": {
                "probability_raster": str(PROB_PATH.relative_to(ROOT)),
                "predicted_mask_raster": str(MASK_PATH.relative_to(ROOT)),
                "gradcam_raster": str(CAM_PATH.relative_to(ROOT)),
            },
            "spatial_reference": {
                "crs": str(prob.crs),
                "pixel_size_m": [abs(float(prob.transform.a)),
                                 abs(float(prob.transform.e))],
                "source_window": {
                    "row_offset": int(row),
                    "column_offset": int(col),
                    "height_pixels": int(cam.height),
                    "width_pixels": int(cam.width),
                },
                "grid_alignment_verified": True,
            },
            "prediction_summary": {
                "valid_joint_pixels": int(joint.sum()),
                "predicted_flood_pixels": int(predicted_flood.sum()),
                "predicted_nonflood_pixels": int(predicted_nonflood.sum()),
                "predicted_flood_fraction": float(predicted_flood.sum() / joint.sum()),
                "probability_min": float(p_joint.min()),
                "probability_max": float(p_joint.max()),
                "probability_mean": float(p_joint.mean()),
                "mean_probability_predicted_flood": mean_or_none(p_flood),
                "mean_probability_predicted_nonflood": mean_or_none(p_nonflood),
            },
            "attribution_summary": {
                "method": "Grad-CAM",
                "target_layer": "dec1",
                "target": "mean flood logit over the existing predicted-flood region",
                "valid_pixels": int(cv.sum()),
                "minimum": float(cam_valid.min()),
                "maximum": float(cam_valid.max()),
                "mean": float(cam_valid.mean()),
                "mean_attribution_predicted_flood": mean_or_none(cam_flood),
                "mean_attribution_predicted_nonflood": mean_or_none(cam_nonflood),
                "top_decile_threshold": high_threshold,
                "top_decile_pixels_in_joint_valid_area": high_attr_count,
                "top_decile_predicted_flood_fraction": (
                    float(high_attr_flood_count / high_attr_count)
                    if high_attr_count else None
                ),
            },
            "interpretation": [
                "The flood mask represents the model's predicted class, not verified ground truth.",
                "Grad-CAM indicates relative model sensitivity for the selected target; it does not establish causation.",
                "Probability values are model outputs and have not been established as calibrated confidence.",
                "This is one selected tile, not a city-wide explanation or a validated emergency recommendation.",
            ],
            "limitations": [
                "Statistics describe only the spatially aligned selected tile and valid pixels.",
                "The Grad-CAM target is defined using the existing predicted-flood region.",
                "No independent ground-truth validation is performed by this explanation builder.",
                "Attribution and probability statistics alone do not establish real-world flood extent or safety.",
            ],
        }

    JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pred = summary["prediction_summary"]
    attr = summary["attribution_summary"]
    spatial = summary["spatial_reference"]

    report = f"""# ResQTwin — Human-Readable XAI Explanation

## Status and scope

- Status: Generated
- Case study: Chennai urban flooding
- Scope: One georeferenced Grad-CAM tile; not a city-wide assessment.
- CRS: {spatial['crs']}
- Pixel size: {spatial['pixel_size_m'][0]:g} m
- Source window: row {spatial['source_window']['row_offset']}, column {spatial['source_window']['column_offset']}, {spatial['source_window']['height_pixels']} × {spatial['source_window']['width_pixels']} pixels
- Grid alignment: verified against the probability and predicted-mask rasters.

## Model prediction summary

- Jointly valid pixels: {pred['valid_joint_pixels']:,}
- Predicted-flood pixels: {pred['predicted_flood_pixels']:,}
- Predicted-nonflood pixels: {pred['predicted_nonflood_pixels']:,}
- Predicted-flood fraction among jointly valid pixels: {pred['predicted_flood_fraction']:.2%}
- Model-output range on jointly valid pixels: {pred['probability_min']:.4f}–{pred['probability_max']:.4f}
- Mean model output on jointly valid pixels: {pred['probability_mean']:.4f}
- Mean output in predicted-flood pixels: {pred['mean_probability_predicted_flood']:.4f}
- Mean output in predicted-nonflood pixels: {pred['mean_probability_predicted_nonflood']:.4f}

These are descriptive model-output statistics, not calibrated confidence estimates.

## Attribution summary

- Method: Grad-CAM
- Target layer: `dec1`
- Target: mean flood logit over the existing predicted-flood region
- Valid Grad-CAM pixels: {attr['valid_pixels']:,}
- Attribution range: {attr['minimum']:.4f}–{attr['maximum']:.4f}
- Mean attribution over valid Grad-CAM pixels: {attr['mean']:.4f}
- Mean attribution in predicted-flood pixels: {attr['mean_attribution_predicted_flood']:.4f}
- Mean attribution in predicted-nonflood pixels: {attr['mean_attribution_predicted_nonflood']:.4f}
- Top-decile attribution threshold: {attr['top_decile_threshold']:.4f}
- Predicted-flood fraction among top-decile pixels within the jointly valid area: {attr['top_decile_predicted_flood_fraction']:.2%}

Grad-CAM describes relative model sensitivity for this selected target. It does not show causal influence or prove that flooding is present.

## Human-readable interpretation

The selected tile contains a predicted-flood fraction of {pred['predicted_flood_fraction']:.1%} among jointly valid pixels. The Grad-CAM raster identifies relative attribution strengths for the model's mean flood-logit target within the existing predicted-flood region. These measurements describe the model's output and attribution in this tile only; they do not independently verify the flood boundary or establish an emergency action.

## Limitations and safety caveats

""" + "".join(f"- {item}\n" for item in summary["limitations"]) + """
- Do not interpret model-output values as calibrated probabilities without a separate calibration evaluation.
- Do not use this artifact alone to make evacuation, routing, or other emergency decisions.
- Human review and independent evidence are required before operational use.

## Input artifacts

""" + "".join(f"- `{value}`\n" for value in summary["inputs"].values())

    MD_PATH.write_text(report, encoding="utf-8")

    # Re-open and validate the artifacts so a successful write is not the only check.
    reloaded = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert reloaded["spatial_reference"]["grid_alignment_verified"] is True
    assert reloaded["prediction_summary"]["valid_joint_pixels"] > 0
    assert reloaded["attribution_summary"]["valid_pixels"] > 0
    assert "Limitations and safety caveats" in MD_PATH.read_text(encoding="utf-8")

    print("EXPLANATION BUILDER: PASS")
    print("JSON:", JSON_PATH.relative_to(ROOT))
    print("Markdown:", MD_PATH.relative_to(ROOT))
    print("Grid alignment:", reloaded["spatial_reference"]["grid_alignment_verified"])
    print("Joint valid pixels:", reloaded["prediction_summary"]["valid_joint_pixels"])
    print("Predicted flood fraction:",
          f"{reloaded['prediction_summary']['predicted_flood_fraction']:.2%}")
    print("Caveats present: True")


if __name__ == "__main__":
    main()
