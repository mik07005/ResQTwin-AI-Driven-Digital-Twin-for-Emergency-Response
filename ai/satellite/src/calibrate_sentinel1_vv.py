from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[3]

VV_TIFF = (
    ROOT
    / "data/raw/chennai/sentinel1"
    / "cog_safe/S1A_IW_GRDH_1SDV_20211116T003212_20211116T003237_040589_04D094_667C_COG.SAFE"
    / "measurement/s1a-iw-grd-vv-20211116t003212-20211116t003237-040589-04d094-001-cog.tiff"
)

CAL_XML = (
    ROOT
    / "data/raw/chennai/sentinel1"
    / "cog_safe/S1A_IW_GRDH_1SDV_20211116T003212_20211116T003237_040589_04D094_667C_COG.SAFE"
    / "annotation/calibration/calibration-s1a-iw-grd-vv-20211116t003212-20211116t003237-040589-04d094-001-cog.xml"
)

OUTPUT = (
    ROOT
    / "data/processed/chennai/sentinel1"
    / "vv_sigma0_linear.tif"
)

CHUNK_LINES = 512


def strip_namespace(tag):
    return tag.split("}")[-1]


def parse_space_separated(text, dtype=float):
    return np.fromstring(text.strip(), sep=" ", dtype=dtype)


def load_calibration_lut(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    vectors = []

    for elem in root.iter():
        if strip_namespace(elem.tag) != "calibrationVector":
            continue

        data = {}

        for child in elem:
            name = strip_namespace(child.tag)

            if name == "line":
                data["line"] = int(child.text.strip())

            elif name == "pixel":
                data["pixel"] = parse_space_separated(child.text, np.int32)

            elif name == "sigmaNought":
                data["sigmaNought"] = parse_space_separated(child.text)

        if {"line", "pixel", "sigmaNought"} <= data.keys():
            vectors.append(data)

    if not vectors:
        raise RuntimeError("No calibration vectors found.")

    lines = np.array([v["line"] for v in vectors], dtype=np.float64)
    pixels = vectors[0]["pixel"].astype(np.float64)

    sigma = np.vstack([v["sigmaNought"] for v in vectors]).astype(np.float64)

    if sigma.shape != (len(lines), len(pixels)):
        raise RuntimeError(
            f"Unexpected LUT shape: {sigma.shape}; "
            f"expected {(len(lines), len(pixels))}"
        )

    return lines, pixels, sigma


def interpolate_sigma(lines, pixels, lut, source_lines, source_pixels):
    """
    Bilinear interpolation of sigmaNought calibration coefficients.

    source_lines and source_pixels are 2-D arrays containing
    original radar image coordinates.
    """

    # Interpolate in range direction for every calibration line.
    range_interp = np.empty(
        (len(lines), source_lines.shape[0], source_lines.shape[1]),
        dtype=np.float32,
    )

    for i in range(len(lines)):
        range_interp[i] = np.interp(
            source_pixels,
            pixels,
            lut[i],
        )

    # Interpolate between calibration lines.
    result = np.empty_like(source_lines, dtype=np.float32)

    for row in range(source_lines.shape[0]):
        line_value = source_lines[row, 0]

        if line_value <= lines[0]:
            result[row] = range_interp[0, row]
        elif line_value >= lines[-1]:
            result[row] = range_interp[-1, row]
        else:
            upper = np.searchsorted(lines, line_value)
            lower = upper - 1

            fraction = (
                line_value - lines[lower]
            ) / (
                lines[upper] - lines[lower]
            )

            result[row] = (
                range_interp[lower, row] * (1.0 - fraction)
                + range_interp[upper, row] * fraction
            )

    return result


def main():
    print("=== Sentinel-1 VV Radiometric Calibration ===")
    print(f"Input:  {VV_TIFF}")
    print(f"XML:    {CAL_XML}")
    print(f"Output: {OUTPUT}")
    print()

    lines, pixels, sigma_lut = load_calibration_lut(CAL_XML)

    print(f"Calibration vectors: {len(lines)}")
    print(f"Calibration samples: {len(pixels)}")
    print(f"Line range: {lines[0]:.0f} -> {lines[-1]:.0f}")
    print(f"Pixel range: {pixels[0]:.0f} -> {pixels[-1]:.0f}")

    print(
        f"sigmaNought LUT: "
        f"min={sigma_lut.min():.6f}, "
        f"max={sigma_lut.max():.6f}"
    )
    print()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(VV_TIFF) as src:

        print("Source raster:")
        print(f"  Size: {src.width} x {src.height}")
        print(f"  dtype: {src.dtypes[0]}")
        print(f"  nodata: {src.nodata}")
        print()

        profile = src.profile.copy()
        profile.update(
            dtype="float32",
            count=1,
            nodata=0.0,
            compress="deflate",
            predictor=3,
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        total_valid = 0
        total_zero = 0

        global_min = np.inf
        global_max = -np.inf
        global_sum = 0.0
        global_sum_sq = 0.0

        with rasterio.open(OUTPUT, "w", **profile) as dst:

            for row_start in range(0, src.height, CHUNK_LINES):

                row_end = min(
                    row_start + CHUNK_LINES,
                    src.height,
                )

                window = rasterio.windows.Window(
                    0,
                    row_start,
                    src.width,
                    row_end - row_start,
                )

                dn = src.read(
                    1,
                    window=window,
                )

                dn = dn.astype(np.float32)

                rows = np.arange(
                    row_start,
                    row_end,
                    dtype=np.float64,
                )

                cols = np.arange(
                    0,
                    src.width,
                    dtype=np.float64,
                )

                source_lines = np.broadcast_to(
                    rows[:, None],
                    dn.shape,
                )

                source_pixels = np.broadcast_to(
                    cols[None, :],
                    dn.shape,
                )

                sigma_coeff = interpolate_sigma(
                    lines,
                    pixels,
                    sigma_lut,
                    source_lines,
                    source_pixels,
                )

                valid = dn > 0

                sigma0 = np.zeros_like(
                    dn,
                    dtype=np.float32,
                )

                # Sentinel-1 GRD calibration:
                #
                # sigma0 = DN^2 / sigmaNought^2
                #
                sigma0[valid] = (
                    dn[valid] ** 2
                    / sigma_coeff[valid] ** 2
                )

                dst.write(
                    sigma0,
                    1,
                    window=window,
                )

                valid_values = sigma0[valid]

                total_valid += valid_values.size
                total_zero += np.count_nonzero(~valid)

                if valid_values.size:
                    global_min = min(
                        global_min,
                        float(valid_values.min()),
                    )

                    global_max = max(
                        global_max,
                        float(valid_values.max()),
                    )

                    global_sum += float(
                        valid_values.sum(dtype=np.float64)
                    )

                    global_sum_sq += float(
                        np.square(valid_values).sum(
                            dtype=np.float64
                        )
                    )

                print(
                    f"Processed rows "
                    f"{row_start:5d}-{row_end - 1:5d} "
                    f"({row_end / src.height * 100:6.2f}%)"
                )

    mean = global_sum / total_valid

    variance = (
        global_sum_sq / total_valid
        - mean * mean
    )

    std = np.sqrt(max(variance, 0.0))

    print()
    print("=== Calibration Complete ===")
    print(f"Output: {OUTPUT}")
    print()
    print("Output statistics:")
    print(f"  Valid pixels: {total_valid:,}")
    print(f"  Zero pixels:  {total_zero:,}")
    print(f"  Min sigma0:   {global_min:.8f}")
    print(f"  Max sigma0:   {global_max:.8f}")
    print(f"  Mean sigma0:  {mean:.8f}")
    print(f"  Std sigma0:   {std:.8f}")

    print()
    print("Validation:")

    if global_min >= 0:
        print("  Non-negative sigma0: PASS")
    else:
        print("  Non-negative sigma0: FAIL")

    if np.isfinite(global_min) and np.isfinite(global_max):
        print("  Finite output range: PASS")
    else:
        print("  Finite output range: FAIL")

    print()
    print("VV calibration finished.")


if __name__ == "__main__":
    main()