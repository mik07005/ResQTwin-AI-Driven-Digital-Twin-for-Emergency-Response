
from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROBABILITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "gcc"
    / "inference"
    / "flood_probability.tif"
)

CANDIDATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "persistent_water"
    / "candidate_new_inundation_filtered.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "uncertainty"
    / "day6"
)

ENTROPY_PATH = OUTPUT_DIR / "flood_predictive_entropy.tif"
CONFIDENCE_PATH = OUTPUT_DIR / "flood_confidence_proxy.tif"
CANDIDATE_ENTROPY_PATH = (
    OUTPUT_DIR / "candidate_inundation_entropy.tif"
)

FLOAT_NODATA = -9999.0


def same_grid(src_a, src_b):
    return (
        src_a.crs == src_b.crs
        and src_a.width == src_b.width
        and src_a.height == src_b.height
        and np.allclose(
            tuple(src_a.transform),
            tuple(src_b.transform),
            atol=1e-9,
        )
    )


def main():
    with rasterio.open(PROBABILITY_PATH) as prob_src, \
            rasterio.open(CANDIDATE_PATH) as candidate_src:

        # Validate the grids before processing.
        if not same_grid(prob_src, candidate_src):
            raise ValueError(
                "Probability and candidate rasters do not share a grid."
            )

        if prob_src.crs is None or not prob_src.crs.is_projected:
            raise ValueError(
                "Area calculation requires a projected CRS."
            )

        linear_units = (prob_src.crs.linear_units or "").lower()
        if linear_units not in ("metre", "meter"):
            raise ValueError(
                f"Expected projected metre units, got: {linear_units}"
            )

        probability = prob_src.read(1).astype(np.float32)
        candidate_values = candidate_src.read(1)

        # Use the source validity mask, not np.isfinite() alone.
        probability_valid = (
            (prob_src.dataset_mask() > 0)
            & np.isfinite(probability)
            & (probability >= 0.0)
            & (probability <= 1.0)
        )

        candidate_valid = candidate_src.dataset_mask() > 0
        candidate_pixels = (
            (candidate_values == 1)
            & candidate_valid
            & probability_valid
        )

        if not probability_valid.any():
            raise ValueError("No valid probability pixels found.")

        if not candidate_pixels.any():
            raise ValueError(
                "No valid candidate pixels overlap valid probability data."
            )

        # Binary entropy normalized to [0, 1].
        p = np.clip(probability, 1e-7, 1.0 - 1e-7)

        entropy_values = -(
            p * np.log2(p)
            + (1.0 - p) * np.log2(1.0 - p)
        )

        confidence_values = np.abs(2.0 * probability - 1.0)

        # Fill outputs with a distinct NoData value first.
        entropy = np.full(
            probability.shape, FLOAT_NODATA, dtype=np.float32
        )
        confidence = np.full(
            probability.shape, FLOAT_NODATA, dtype=np.float32
        )
        candidate_entropy = np.full(
            probability.shape, FLOAT_NODATA, dtype=np.float32
        )

        # Valid zero-valued measurements remain valid data.
        entropy[probability_valid] = entropy_values[probability_valid]
        confidence[probability_valid] = confidence_values[probability_valid]
        candidate_entropy[candidate_pixels] = entropy_values[candidate_pixels]

        profile = prob_src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=FLOAT_NODATA,
            compress="deflate",
            predictor=2,
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        outputs = (
            (ENTROPY_PATH, entropy),
            (CONFIDENCE_PATH, confidence),
            (CANDIDATE_ENTROPY_PATH, candidate_entropy),
        )

        for output_path, values in outputs:
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(values, 1)

        # Statistics are calculated only over valid source pixels.
        valid_entropy = entropy_values[probability_valid]
        valid_confidence = confidence_values[probability_valid]
        candidate_entropy_values = entropy_values[candidate_pixels]

        transform = prob_src.transform
        pixel_area_km2 = abs(
            transform.a * transform.e
            - transform.b * transform.d
        ) / 1_000_000.0

        candidate_area_km2 = (
            int(candidate_pixels.sum()) * pixel_area_km2
        )

        print()
        print("=" * 64)
        print("DAY 6 - FLOOD UNCERTAINTY OUTPUTS")
        print("=" * 64)

        print("CRS:", prob_src.crs)
        print("Linear units:", prob_src.crs.linear_units)
        print("Raster dimensions:", prob_src.width, "x", prob_src.height)
        print("NoData value:", FLOAT_NODATA)
        print()
        print("Valid probability pixels:", int(probability_valid.sum()))
        print("Invalid probability pixels:", int((~probability_valid).sum()))
        print("Valid candidate pixels:", int(candidate_pixels.sum()))
        print("Filtered candidate area (km2):", round(candidate_area_km2, 4))

        print()
        print("Mean binary entropy:", float(valid_entropy.mean()))
        print("Median binary entropy:", float(np.median(valid_entropy)))
        print("Mean confidence proxy:", float(valid_confidence.mean()))
        print("Median confidence proxy:", float(np.median(valid_confidence)))
        print("Mean candidate entropy:", float(candidate_entropy_values.mean()))
        print(
            "Median candidate entropy:",
            float(np.median(candidate_entropy_values)),
        )

        print()
        print("Entropy output:", ENTROPY_PATH)
        print("Confidence proxy output:", CONFIDENCE_PATH)
        print("Candidate entropy output:", CANDIDATE_ENTROPY_PATH)

        print()
        print("Interpretation:")
        print("- Entropy and confidence are model-score-derived proxies.")
        print("- They are not calibrated probabilities of model error.")
        print("- Candidate inundation is not confirmed ground-truth flooding.")
        print("=" * 64)


if __name__ == "__main__":
    main()
