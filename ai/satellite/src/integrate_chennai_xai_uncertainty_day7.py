import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[3]
DAY7 = ROOT / "data/processed/chennai/explainability/day7"
JSON_PATH = DAY7 / "explanation_summary.json"
MD_PATH = DAY7 / "explanation_summary.md"

PATHS = {
    "probability": ROOT / "data/processed/chennai/sentinel1/gcc/inference/flood_probability.tif",
    "mask": ROOT / "data/processed/chennai/sentinel1/gcc/inference/flood_mask.tif",
    "gradcam": ROOT / "data/processed/chennai/explainability/gradcam_tile.tif",
    "predictive_entropy": ROOT / "data/processed/chennai/uncertainty/day6/flood_predictive_entropy.tif",
    "confidence_proxy": ROOT / "data/processed/chennai/uncertainty/day6/flood_confidence_proxy.tif",
    "candidate_entropy": ROOT / "data/processed/chennai/uncertainty/day6/candidate_inundation_entropy.tif",
    "candidate_categories": ROOT / "data/processed/chennai/uncertainty/day6/candidate_uncertainty_categories.tif",
}

MARKER = "## Uncertainty summary (Day 6 outputs)"


def valid_mask(src, arr, window=None):
    if window is None:
        dataset_valid = src.dataset_mask() > 0
    else:
        dataset_valid = src.dataset_mask(window=window) > 0

    if dataset_valid.shape != arr.shape:
        raise ValueError(
            f"Validity-mask shape {dataset_valid.shape} does not match "
            f"array shape {arr.shape} for {src.name}"
        )

    valid = dataset_valid & np.isfinite(arr)
    if src.nodata is not None:
        valid &= arr != src.nodata
    return valid


def stats(arr, valid):
    values = arr[valid]
    if values.size == 0:
        return {
            "valid_pixels": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
        }
    return {
        "valid_pixels": int(values.size),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }


def main():
    if not JSON_PATH.exists() or not MD_PATH.exists():
        raise FileNotFoundError("Run the Day 7 explanation builder first.")

    summary = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    window_info = summary["spatial_reference"]["source_window"]
    row = int(window_info["row_offset"])
    col = int(window_info["column_offset"])
    height = int(window_info["height_pixels"])
    width = int(window_info["width_pixels"])
    window = Window(col, row, width, height)

    with rasterio.open(PATHS["probability"]) as prob, \
         rasterio.open(PATHS["mask"]) as mask, \
         rasterio.open(PATHS["gradcam"]) as cam:

        reference = (prob.crs, prob.transform, prob.shape)
        for name, src in (("mask", mask), ("Grad-CAM", cam)):
            if name == "mask":
                same = (src.crs, src.transform, src.shape) == reference
                if not same:
                    raise ValueError("Mask raster does not match probability grid.")
            else:
                if src.crs != prob.crs:
                    raise ValueError("Grad-CAM CRS does not match probability raster.")
                c_f, r_f = (~prob.transform) * (src.transform.c, src.transform.f)
                if abs(c_f - round(c_f)) > 1e-6 or abs(r_f - round(r_f)) > 1e-6:
                    raise ValueError("Grad-CAM is not aligned to source pixel grid.")

        if (row < 0 or col < 0 or row + height > prob.height or
                col + width > prob.width):
            raise ValueError("Stored explanation window is outside source grid.")

        p = prob.read(1, window=window)
        m = mask.read(1, window=window)
        c = cam.read(1)

        joint = (
            valid_mask(prob, p, window)
            & valid_mask(mask, m, window)
            & valid_mask(cam, c)
        )

        uncertainty = {}
        for name in ("predictive_entropy", "confidence_proxy",
                     "candidate_entropy", "candidate_categories"):
            path = PATHS[name]
            with rasterio.open(path) as src:
                if (src.crs, src.transform, src.shape) != reference:
                    raise ValueError(f"{name} raster does not match source grid.")
                arr = src.read(1, window=window)
                valid = valid_mask(src, arr, window) & joint

                if name == "predictive_entropy":
                    uncertainty[name] = {
                        "description": "Normalized binary entropy derived from the flood score.",
                        "interpretation": "Higher values indicate a score closer to 0.5; not a calibrated error probability.",
                        **stats(arr, valid),
                    }
                elif name == "confidence_proxy":
                    uncertainty[name] = {
                        "description": "Absolute distance from 0.5, scaled as abs(2*p - 1).",
                        "interpretation": "A mathematical score-separation proxy, not independently validated calibrated confidence.",
                        **stats(arr, valid),
                    }
                elif name == "candidate_entropy":
                    uncertainty[name] = {
                        "description": "Entropy values defined only on valid candidate-inundation pixels.",
                        "interpretation": "NoData means unavailable/not a candidate measurement, not zero uncertainty.",
                        **stats(arr, valid),
                    }
                else:
                    candidate_valid = valid & np.isin(arr, [1, 2, 3])
                    vals = arr[candidate_valid]
                    counts = {str(category): int((vals == category).sum())
                              for category in (1, 2, 3)}
                    uncertainty[name] = {
                        "description": "Heuristic entropy categories for valid candidate pixels.",
                        "category_labels": {"1": "low entropy", "2": "medium entropy", "3": "high entropy"},
                        "thresholds": {
                            "low": "entropy < 0.33",
                            "medium": "0.33 <= entropy < 0.66",
                            "high": "entropy >= 0.66",
                        },
                        "valid_candidate_pixels_in_tile": int(vals.size),
                        "category_counts": counts,
                        "interpretation": "Categories are heuristic indicators, not probabilities of prediction error.",
                    }

    uncertainty_contract = {
        "source": "Existing Day 6 uncertainty outputs",
        "spatial_alignment_verified": True,
        "scope": "Same selected Grad-CAM tile and jointly valid pixels where applicable.",
        "predictive_entropy_formula": "-p*log2(p) - (1-p)*log2(1-p), with numerical clipping in the generator.",
        "confidence_proxy_formula": "abs(2*p - 1)",
        "candidate_entropy_coverage_note": "Candidate entropy is defined only on valid candidate-inundation pixels. NoData or missing values must not be interpreted as zero uncertainty.",
        "metrics": uncertainty,
        "limitations": [
            "Entropy is a model-derived uncertainty indicator, not a calibrated probability of error.",
            "The confidence proxy is not independently calibrated or validated confidence.",
            "Candidate categories are heuristic thresholds and apply only to valid candidate pixels.",
            "Uncertainty and Grad-CAM attribution are distinct quantities and must not be merged into one score.",
        ],
    }

    summary["uncertainty_summary"] = uncertainty_contract
    JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = MD_PATH.read_text(encoding="utf-8")
    if MARKER in md:
        md = md.split(MARKER)[0].rstrip() + "\n\n"

    ent = uncertainty["predictive_entropy"]
    conf = uncertainty["confidence_proxy"]
    cand = uncertainty["candidate_entropy"]
    cats = uncertainty["candidate_categories"]

    section = f"""## Uncertainty summary (Day 6 outputs)

These measurements are from the existing uncertainty rasters, sampled over the selected Grad-CAM tile. They are separate from Grad-CAM attribution.

### Predictive entropy

- Valid pixels: {ent['valid_pixels']:,}
- Minimum / maximum: {ent['minimum']:.4f} / {ent['maximum']:.4f}
- Mean: {ent['mean']:.4f}

Formula: normalized binary entropy, `-p*log2(p) - (1-p)*log2(1-p)`. Higher values indicate model scores closer to 0.5. This is not a calibrated probability of prediction error.

### Confidence proxy

- Valid pixels: {conf['valid_pixels']:,}
- Minimum / maximum: {conf['minimum']:.4f} / {conf['maximum']:.4f}
- Mean: {conf['mean']:.4f}

Formula: `abs(2*p - 1)`. This measures distance from the 0.5 decision boundary and is not independently validated calibrated confidence.

### Candidate-inundation uncertainty

- Valid candidate-entropy pixels in this tile: {cand['valid_pixels']:,}
- Entropy minimum / maximum / mean: {cand['minimum']} / {cand['maximum']} / {cand['mean']}
- Valid candidate-category pixels in this tile: {cats['valid_candidate_pixels_in_tile']:,}
- Category counts: low (1) = {cats['category_counts']['1']:,}; medium (2) = {cats['category_counts']['2']:,}; high (3) = {cats['category_counts']['3']:,}

Thresholds: low entropy `< 0.33`; medium entropy `0.33–<0.66`; high entropy `>=0.66`. These are heuristic categories. Candidate entropy is only defined for valid candidate-inundation pixels; NoData must not be interpreted as zero uncertainty.

### Interpretation boundary

Flood prediction describes the model's output; entropy and the confidence proxy are model-derived indicators; Grad-CAM describes relative attribution for the selected target. They are distinct quantities and none independently verifies real-world flooding or establishes an emergency action.

"""
    MD_PATH.write_text(md.rstrip() + "\n\n" + section, encoding="utf-8")

    # Validate persisted output.
    reloaded = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    md_check = MD_PATH.read_text(encoding="utf-8")
    assert reloaded["uncertainty_summary"]["spatial_alignment_verified"]
    assert reloaded["uncertainty_summary"]["metrics"]["predictive_entropy"]["valid_pixels"] > 0
    assert reloaded["uncertainty_summary"]["metrics"]["confidence_proxy"]["valid_pixels"] > 0
    assert MARKER in md_check
    assert "not a calibrated probability" in md_check
    assert "NoData must not be interpreted as zero uncertainty" in md_check

    print("UNCERTAINTY INTEGRATION: PASS")
    print("JSON:", JSON_PATH.relative_to(ROOT))
    print("Markdown:", MD_PATH.relative_to(ROOT))
    print("Spatial alignment verified: True")
    print("Tile entropy mean:", f"{ent['mean']:.6f}")
    print("Tile confidence-proxy mean:", f"{conf['mean']:.6f}")
    print("Valid candidate pixels in tile:", cats["valid_candidate_pixels_in_tile"])
    print("Grad-CAM kept distinct from uncertainty: True")


if __name__ == "__main__":
    main()
