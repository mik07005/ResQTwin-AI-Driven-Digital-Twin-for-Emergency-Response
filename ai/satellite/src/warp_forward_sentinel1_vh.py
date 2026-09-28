from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.transform import from_origin
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

SRC = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "vh_sigma0_linear.tif"
)

ANNOTATION = (
    ROOT
    / "data"
    / "raw"
    / "chennai"
    / "sentinel1"
    / "cog_safe"
    / "S1A_IW_GRDH_1SDV_20211116T003212_20211116T003237_040589_04D094_667C_COG.SAFE"
    / "annotation"
    / "s1a-iw-grd-vh-20211116t003212-20211116t003237-040589-04d094-002-cog.xml"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "vh_sigma0_chennai_utm44n.tif"
)


# ============================================================
# Chennai basin AOI
# ============================================================

MIN_LON = 79.1666667
MIN_LAT = 12.6666667
MAX_LON = 80.4166667
MAX_LAT = 13.6666667

DST_CRS = "EPSG:32644"
RESOLUTION = 10.0

# Safety margin around source region
SOURCE_MARGIN_PIXELS = 100


# ============================================================
# Parse Sentinel-1 GCPs
# ============================================================

def parse_gcps(xml_path):

    root = ET.parse(xml_path).getroot()

    gcps = []

    for node in root.iter():

        if not node.tag.endswith("geolocationGridPoint"):
            continue

        line = node.find(".//{*}line")
        pixel = node.find(".//{*}pixel")
        lat = node.find(".//{*}latitude")
        lon = node.find(".//{*}longitude")

        if all(v is not None for v in (line, pixel, lat, lon)):

            gcps.append(
                GroundControlPoint(
                    row=float(line.text),
                    col=float(pixel.text),
                    x=float(lon.text),
                    y=float(lat.text),
                )
            )

    return gcps


# ============================================================
# Build regular GCP interpolators
# ============================================================

def build_interpolators(gcps):

    rows = sorted(set(g.row for g in gcps))
    cols = sorted(set(g.col for g in gcps))

    lon_grid = np.zeros(
        (len(rows), len(cols)),
        dtype=np.float64,
    )

    lat_grid = np.zeros(
        (len(rows), len(cols)),
        dtype=np.float64,
    )

    row_index = {
        value: i
        for i, value in enumerate(rows)
    }

    col_index = {
        value: i
        for i, value in enumerate(cols)
    }

    for gcp in gcps:

        r = row_index[gcp.row]
        c = col_index[gcp.col]

        lon_grid[r, c] = gcp.x
        lat_grid[r, c] = gcp.y

    lon_interp = RegularGridInterpolator(
        (rows, cols),
        lon_grid,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    lat_interp = RegularGridInterpolator(
        (rows, cols),
        lat_grid,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    return (
        np.asarray(rows),
        np.asarray(cols),
        lon_interp,
        lat_interp,
    )


# ============================================================
# Locate source pixels covering Chennai
# ============================================================

def find_source_window(
    lon_interp,
    lat_interp,
    width,
    height,
):

    # Coarse source grid used only to identify
    # which portion of the Sentinel-1 scene
    # intersects the Chennai AOI.

    sample_cols = np.linspace(
        0,
        width - 1,
        1000,
    )

    sample_rows = np.linspace(
        0,
        height - 1,
        700,
    )

    cc, rr = np.meshgrid(
        sample_cols,
        sample_rows,
    )

    points = np.column_stack(
        [
            rr.ravel(),
            cc.ravel(),
        ]
    )

    lon = lon_interp(points).reshape(
        rr.shape
    )

    lat = lat_interp(points).reshape(
        rr.shape
    )

    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
    )

    inside = (
        valid
        & (lon >= MIN_LON)
        & (lon <= MAX_LON)
        & (lat >= MIN_LAT)
        & (lat <= MAX_LAT)
    )

    if not inside.any():

        raise RuntimeError(
            "Could not locate Chennai AOI "
            "inside Sentinel-1 scene."
        )

    source_cols = cc[inside]
    source_rows = rr[inside]

    min_col = max(
        0,
        int(np.floor(source_cols.min()))
        - SOURCE_MARGIN_PIXELS,
    )

    max_col = min(
        width - 1,
        int(np.ceil(source_cols.max()))
        + SOURCE_MARGIN_PIXELS,
    )

    min_row = max(
        0,
        int(np.floor(source_rows.min()))
        - SOURCE_MARGIN_PIXELS,
    )

    max_row = min(
        height - 1,
        int(np.ceil(source_rows.max()))
        + SOURCE_MARGIN_PIXELS,
    )

    return (
        min_col,
        min_row,
        max_col,
        max_row,
    )


# ============================================================
# Main
# ============================================================

print("=== Sentinel-1 VH Forward-Geolocation Warp ===")

print(f"Input:      {SRC}")
print(f"Annotation: {ANNOTATION}")
print(f"Output:     {OUTPUT}")

print("\nChennai AOI:")
print(
    f"  Longitude: "
    f"{MIN_LON} -> {MAX_LON}"
)

print(
    f"  Latitude:  "
    f"{MIN_LAT} -> {MAX_LAT}"
)


# ============================================================
# Parse GCPs
# ============================================================

gcps = parse_gcps(ANNOTATION)

print(f"\nGCP count: {len(gcps)}")

if len(gcps) != 210:

    raise RuntimeError(
        f"Expected 210 GCPs, found {len(gcps)}"
    )


# ============================================================
# Open calibrated VH
# ============================================================

with rasterio.open(SRC) as src:

    width = src.width
    height = src.height

    print("\nSource:")
    print(
        f"  Size: "
        f"{width} x {height}"
    )

    print(
        f"  dtype: "
        f"{src.dtypes[0]}"
    )

    print(
        f"  CRS: "
        f"{src.crs}"
    )


    # ========================================================
    # Build geolocation interpolators
    # ========================================================

    (
        gcp_rows,
        gcp_cols,
        lon_interp,
        lat_interp,
    ) = build_interpolators(gcps)

    print("\nGCP grid:")
    print(
        f"  Rows: "
        f"{len(gcp_rows)}"
    )

    print(
        f"  Columns: "
        f"{len(gcp_cols)}"
    )


    # ========================================================
    # Find source window
    # ========================================================

    (
        min_col,
        min_row,
        max_col,
        max_row,
    ) = find_source_window(
        lon_interp,
        lat_interp,
        width,
        height,
    )

    source_width = (
        max_col
        - min_col
        + 1
    )

    source_height = (
        max_row
        - min_row
        + 1
    )

    print(
        "\nSource window covering Chennai:"
    )

    print(
        f"  Columns: "
        f"{min_col} -> {max_col}"
    )

    print(
        f"  Rows:    "
        f"{min_row} -> {max_row}"
    )

    print(
        f"  Size:    "
        f"{source_width} x "
        f"{source_height}"
    )


    # ========================================================
    # Geographic -> UTM transformation
    # ========================================================

    to_utm = Transformer.from_crs(
        "EPSG:4326",
        DST_CRS,
        always_xy=True,
    )

    corners = [
        (MIN_LON, MIN_LAT),
        (MIN_LON, MAX_LAT),
        (MAX_LON, MIN_LAT),
        (MAX_LON, MAX_LAT),
    ]

    utm_points = [
        to_utm.transform(
            lon,
            lat,
        )
        for lon, lat in corners
    ]

    xs = [
        point[0]
        for point in utm_points
    ]

    ys = [
        point[1]
        for point in utm_points
    ]

    left = min(xs)
    right = max(xs)
    bottom = min(ys)
    top = max(ys)

    out_width = int(
        np.ceil(
            (right - left)
            / RESOLUTION
        )
    )

    out_height = int(
        np.ceil(
            (top - bottom)
            / RESOLUTION
        )
    )

    dst_transform = from_origin(
        left,
        top,
        RESOLUTION,
        RESOLUTION,
    )

    print("\nDestination:")
    print(
        f"  CRS: "
        f"{DST_CRS}"
    )

    print(
        f"  Resolution: "
        f"{RESOLUTION} m"
    )

    print(
        f"  Size: "
        f"{out_width} x "
        f"{out_height}"
    )

    print(
        f"  Pixels: "
        f"{out_width * out_height:,}"
    )


    # ========================================================
    # Read calibrated VH source subset
    # ========================================================

    window = rasterio.windows.Window(
        min_col,
        min_row,
        source_width,
        source_height,
    )

    source_data = src.read(
        1,
        window=window,
    ).astype(np.float32)

    print("\nSource subset:")

    print(
        f"  Nonzero: "
        f"{np.count_nonzero(source_data):,}"
    )

    print(
        f"  Min: "
        f"{source_data.min():.8f}"
    )

    print(
        f"  Max: "
        f"{source_data.max():.8f}"
    )


    # ========================================================
    # Build source geolocation grid
    # ========================================================

    local_cols = np.arange(
        min_col,
        max_col + 1,
        dtype=np.float64,
    )

    local_rows = np.arange(
        min_row,
        max_row + 1,
        dtype=np.float64,
    )

    cc, rr = np.meshgrid(
        local_cols,
        local_rows,
    )

    points = np.column_stack(
        [
            rr.ravel(),
            cc.ravel(),
        ]
    )

    lon = lon_interp(
        points
    ).reshape(
        rr.shape
    )

    lat = lat_interp(
        points
    ).reshape(
        rr.shape
    )

    print("\nGeolocation grid:")

    print(
        f"  Shape: "
        f"{lon.shape}"
    )

    print(
        f"  Longitude: "
        f"{np.nanmin(lon):.6f} -> "
        f"{np.nanmax(lon):.6f}"
    )

    print(
        f"  Latitude:  "
        f"{np.nanmin(lat):.6f} -> "
        f"{np.nanmax(lat):.6f}"
    )


    # ========================================================
    # Convert geolocation to UTM
    # ========================================================

    utm_x, utm_y = to_utm.transform(
        lon,
        lat,
    )

    print("\nUTM geolocation:")

    print(
        f"  X: "
        f"{np.nanmin(utm_x):.2f} -> "
        f"{np.nanmax(utm_x):.2f}"
    )

    print(
        f"  Y: "
        f"{np.nanmin(utm_y):.2f} -> "
        f"{np.nanmax(utm_y):.2f}"
    )


    # ========================================================
    # Forward scatter resampling
    # ========================================================

    destination = np.zeros(
        (
            out_height,
            out_width,
        ),
        dtype=np.float32,
    )

    # Source UTM coordinates -> destination
    # raster coordinates.

    dst_cols = (
        (utm_x - left)
        / RESOLUTION
    )

    dst_rows = (
        (top - utm_y)
        / RESOLUTION
    )

    valid = (
        np.isfinite(dst_cols)
        & np.isfinite(dst_rows)
        & np.isfinite(source_data)
        & (source_data > 0)
        & (dst_cols >= 0)
        & (dst_cols < out_width)
        & (dst_rows >= 0)
        & (dst_rows < out_height)
    )

    print("\nResampling:")

    print(
        f"  Candidate source pixels: "
        f"{valid.size:,}"
    )

    print(
        f"  Valid source pixels: "
        f"{valid.sum():,}"
    )

    rows_out = np.rint(
        dst_rows[valid]
    ).astype(np.int64)

    cols_out = np.rint(
        dst_cols[valid]
    ).astype(np.int64)

    values = source_data[valid]

    inside = (
        (rows_out >= 0)
        & (rows_out < out_height)
        & (cols_out >= 0)
        & (cols_out < out_width)
    )

    rows_out = rows_out[inside]
    cols_out = cols_out[inside]
    values = values[inside]

    destination[
        rows_out,
        cols_out
    ] = values


# ============================================================
# Write GeoTIFF
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

profile = {
    "driver": "GTiff",
    "height": out_height,
    "width": out_width,
    "count": 1,
    "dtype": "float32",
    "crs": DST_CRS,
    "transform": dst_transform,
    "nodata": 0.0,
    "compress": "deflate",
    "predictor": 3,
    "tiled": True,
    "BIGTIFF": "IF_SAFER",
}

with rasterio.open(
    OUTPUT,
    "w",
    **profile,
) as dst:

    dst.write(
        destination,
        1,
    )

    dst.update_tags(
        satellite="Sentinel-1A",
        acquisition="2021-11-16T00:32:12Z",
        polarization="VH",
        product_type="GRD",
        calibration="sigma0_linear",
        geolocation="Sentinel-1 annotation GCP grid",
        resampling="forward nearest-neighbour",
        source_crs="WGS84 GCP geolocation",
    )


# ============================================================
# Validation
# ============================================================

print("\n=== Validation ===")

with rasterio.open(OUTPUT) as check:

    arr = check.read(1)

    valid = (
        np.isfinite(arr)
        & (arr > 0)
    )

    print(
        f"CRS: "
        f"{check.crs}"
    )

    print(
        f"Size: "
        f"{check.width} x "
        f"{check.height}"
    )

    print(
        f"Resolution: "
        f"{check.res[0]:.3f} x "
        f"{check.res[1]:.3f} m"
    )

    print(
        f"Bounds: "
        f"{check.bounds}"
    )

    print(
        f"Valid pixels: "
        f"{valid.sum():,}"
    )

    print(
        f"Coverage: "
        f"{100 * valid.mean():.4f}%"
    )

    if valid.any():

        values = arr[valid]

        print(
            f"Min: "
            f"{values.min():.8f}"
        )

        print(
            f"Max: "
            f"{values.max():.8f}"
        )

        print(
            f"Mean: "
            f"{values.mean():.8f}"
        )

        print(
            "\nNON-EMPTY OUTPUT: PASS"
        )

    else:

        print(
            "\nNON-EMPTY OUTPUT: FAIL"
        )

        raise RuntimeError(
            "Output contains no valid VH pixels."
        )


print(
    "\n=== VH Forward-Geolocation "
    "Warp Complete ==="
)