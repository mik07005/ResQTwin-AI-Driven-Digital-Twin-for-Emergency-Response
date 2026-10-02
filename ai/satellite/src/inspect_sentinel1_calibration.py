from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


ROOT = Path(__file__).resolve().parents[3]

XML = ROOT / (
    "data/raw/chennai/sentinel1/cog_safe/"
    "S1A_IW_GRDH_1SDV_20211116T003212_20211116T003237_040589_04D094_667C_COG.SAFE/"
    "annotation/calibration/"
    "calibration-s1a-iw-grd-vv-20211116t003212-20211116t003237-040589-04d094-001-cog.xml"
)


def local_name(tag):
    return tag.split("}")[-1]


def parse_numbers(text):
    return np.fromstring(text.strip(), sep=" ")


def main():
    print("=== Sentinel-1 Calibration Inspection ===")
    print(f"XML: {XML}")

    if not XML.exists():
        raise FileNotFoundError(XML)

    root = ET.parse(XML).getroot()

    calibration_vectors = []

    for elem in root.iter():
        if local_name(elem.tag) != "calibrationVector":
            continue

        data = {}

        for child in elem:
            tag = local_name(child.tag)

            if tag in {"line", "azimuthTime"}:
                data[tag] = child.text.strip()
            elif tag in {"pixel", "sigmaNought", "betaNought", "gamma"}:
                values = parse_numbers(child.text)
                data[tag] = values

        calibration_vectors.append(data)

    print(f"\nCalibration vectors: {len(calibration_vectors)}")

    if len(calibration_vectors) != 27:
        raise RuntimeError(
            f"Expected 27 calibration vectors, "
            f"found {len(calibration_vectors)}"
        )

    lines = np.array(
        [int(v["line"]) for v in calibration_vectors]
    )

    print(f"Lines: {lines.tolist()}")

    first = calibration_vectors[0]

    print("\nFirst calibration vector:")
    print(f"  Line: {first['line']}")
    print(f"  Pixels: {len(first['pixel'])}")
    print(f"  sigmaNought values: {len(first['sigmaNought'])}")
    print(f"  betaNought values: {len(first['betaNought'])}")
    print(f"  gamma values: {len(first['gamma'])}")

    # Verify LUT dimensions.
    for i, vector in enumerate(calibration_vectors):
        n_pixels = len(vector["pixel"])

        for name in ("sigmaNought", "betaNought", "gamma"):
            if len(vector[name]) != n_pixels:
                raise RuntimeError(
                    f"Vector {i}: {name} length "
                    f"{len(vector[name])} != pixel length {n_pixels}"
                )

    print("\nLUT dimension check: PASS")

    # Inspect sigma-nought range across the complete LUT.
    sigma = np.concatenate(
        [v["sigmaNought"] for v in calibration_vectors]
    )

    beta = np.concatenate(
        [v["betaNought"] for v in calibration_vectors]
    )

    gamma = np.concatenate(
        [v["gamma"] for v in calibration_vectors]
    )

    print("\nComplete LUT statistics:")
    print(
        f"  sigmaNought: "
        f"min={sigma.min():.6f}, "
        f"max={sigma.max():.6f}"
    )
    print(
        f"  betaNought:  "
        f"min={beta.min():.6f}, "
        f"max={beta.max():.6f}"
    )
    print(
        f"  gamma:       "
        f"min={gamma.min():.6f}, "
        f"max={gamma.max():.6f}"
    )

    print("\nCalibration XML validation: PASS")


if __name__ == "__main__":
    main()