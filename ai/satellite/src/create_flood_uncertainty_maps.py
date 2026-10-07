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
)

ENTROPY_PATH = OUTPUT_DIR / "flood_predictive_entropy.tif"
CONFIDENCE_PATH = OUTPUT_DIR / "flood_confidence_proxy.tif"
CANDIDATE_ENTROPY_PATH = (
    OUTPUT_DIR / "candidate_inundation_entropy.tif"
)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(PROBABILITY_PATH) as prob_src, \
         rasterio.open(CANDIDATE_PATH) as candidate_src:

        probability = prob_src.read(1).astype(np.float32)
        candidate = candidate_src.read(1) == 1

        # -------------------------------------------------------------
        # Verify spatial grids
        # -------------------------------------------------------------

        if prob_src.crs != candidate_src.crs:
            raise ValueError("CRS mismatch.")

        if prob_src.width != candidate_src.width:
            raise ValueError("Width mismatch.")

        if prob_src.height != candidate_src.height:
            raise ValueError("Height mismatch.")

        if not np.allclose(
            tuple(prob_src.transform),
            tuple(candidate_src.transform),
            atol=1e-9,
        ):
            raise ValueError("Transform/grid mismatch.")

        # -------------------------------------------------------------
        # Identify valid probability pixels
        # -------------------------------------------------------------

        valid = np.isfinite(probability)

        # Probability should lie in [0, 1].
        probability = np.clip(probability, 1e-7, 1.0 - 1e-7)

        # -------------------------------------------------------------
        # Predictive entropy
        #
        # Maximum entropy occurs at p = 0.5.
        # Normalize to [0, 1].
        # -------------------------------------------------------------

        entropy = -(
            probability * np.log2(probability)
            + (1.0 - probability)
            * np.log2(1.0 - probability)
        )

        entropy = np.where(valid, entropy, 0.0)

        # -------------------------------------------------------------
        # Confidence proxy
        #
        # 1 = very confident
        # 0 = maximally ambiguous
        # -------------------------------------------------------------

        confidence = np.abs(2.0 * probability - 1.0)
        confidence = np.where(valid, confidence, 0.0)

        # -------------------------------------------------------------
        # Entropy specifically over candidate inundation
        # -------------------------------------------------------------

        candidate_entropy = np.where(
            candidate & valid,
            entropy,
            0.0
        )

        profile = prob_src.profile.copy()

        profile.update(
            dtype="float32",
            count=1,
            nodata=0.0,
            compress="deflate",
            predictor=2,
        )

        # -------------------------------------------------------------
        # Save full entropy
        # -------------------------------------------------------------

        with rasterio.open(
            ENTROPY_PATH,
            "w",
            **profile
        ) as dst:

            dst.write(entropy.astype(np.float32), 1)

        # -------------------------------------------------------------
        # Save confidence
        # -------------------------------------------------------------

        with rasterio.open(
            CONFIDENCE_PATH,
            "w",
            **profile
        ) as dst:

            dst.write(confidence.astype(np.float32), 1)

        # -------------------------------------------------------------
        # Save candidate-only entropy
        # -------------------------------------------------------------

        with rasterio.open(
            CANDIDATE_ENTROPY_PATH,
            "w",
            **profile
        ) as dst:

            dst.write(candidate_entropy.astype(np.float32), 1)

        # -------------------------------------------------------------
        # Statistics
        # -------------------------------------------------------------

        valid_entropy = entropy[valid]
        valid_confidence = confidence[valid]

        candidate_valid = candidate & valid

        print()
        print("=" * 60)
        print("FLOOD UNCERTAINTY / CONFIDENCE RESULTS")
        print("=" * 60)

        print()
        print("Valid probability pixels:", int(valid.sum()))

        print()
        print("Entropy range:",
              float(valid_entropy.min()),
              "to",
              float(valid_entropy.max()))

        print("Mean entropy:",
              float(valid_entropy.mean()))

        print("Median entropy:",
              float(np.median(valid_entropy)))

        print()
        print("Confidence range:",
              float(valid_confidence.min()),
              "to",
              float(valid_confidence.max()))

        print("Mean confidence:",
              float(valid_confidence.mean()))

        print("Median confidence:",
              float(np.median(valid_confidence)))

        if candidate_valid.any():

            ce = entropy[candidate_valid]

            print()
            print("Candidate inundation pixels:",
                  int(candidate_valid.sum()))

            print("Candidate entropy mean:",
                  float(ce.mean()))

            print("Candidate entropy median:",
                  float(np.median(ce)))

            print("Candidate entropy P25:",
                  float(np.percentile(ce, 25)))

            print("Candidate entropy P75:",
                  float(np.percentile(ce, 75)))

        print()
        print("Entropy output:")
        print(ENTROPY_PATH)

        print()
        print("Confidence output:")
        print(CONFIDENCE_PATH)

        print()
        print("Candidate entropy output:")
        print(CANDIDATE_ENTROPY_PATH)

        print()
        print("=" * 60)


if __name__ == "__main__":
    main()