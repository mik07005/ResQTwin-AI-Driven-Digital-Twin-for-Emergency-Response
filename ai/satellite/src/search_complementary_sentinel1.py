from pathlib import Path
import json
import requests
import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

AOI_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "aoi"
    / "gcc_sentinel1_uncovered.geojson"
)

# Copernicus Data Space STAC
URL = "https://stac.dataspace.copernicus.eu/v1/search"


# ------------------------------------------------------------
# Load uncovered GCC geometry
# ------------------------------------------------------------

gdf = gpd.read_file(AOI_FILE)

# Convert UTM geometry to WGS84 because STAC spatial coordinates
# are longitude/latitude.
gdf = gdf.to_crs(epsg=4326)

geometry = gdf.geometry.union_all()

print("=" * 70)
print("COMPLEMENTARY SENTINEL-1 SEARCH")
print("=" * 70)

print("AOI bounds (WGS84):")
print(geometry.bounds)

# ------------------------------------------------------------
# Search windows
# ------------------------------------------------------------

windows = [
    ("same-day", "2021-11-16T00:00:00Z/2021-11-17T00:00:00Z"),
    ("one-day-before", "2021-11-15T00:00:00Z/2021-11-16T00:00:00Z"),
    ("one-day-after", "2021-11-17T00:00:00Z/2021-11-18T00:00:00Z"),
    ("two-days-before", "2021-11-14T00:00:00Z/2021-11-15T00:00:00Z"),
    ("two-days-after", "2021-11-18T00:00:00Z/2021-11-19T00:00:00Z"),
]


for label, datetime_range in windows:

    print("\n" + "=" * 70)
    print(f"SEARCH: {label}")
    print(f"Time:   {datetime_range}")
    print("=" * 70)

    payload = {
        "collections": ["sentinel-1-grd"],
        "datetime": datetime_range,
        "intersects": geometry.__geo_interface__,
        "limit": 100,
    }

    try:
        response = requests.post(
            URL,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

    except Exception as e:
        print(f"ERROR: {e}")
        continue

    features = data.get("features", [])

    print(f"Products found: {len(features)}")

    for i, item in enumerate(features, start=1):

        props = item.get("properties", {})

        product_id = item.get("id", "UNKNOWN")

        start = props.get("datetime")
        platform = props.get("platform")
        orbit_state = props.get("sat:orbit_state")
        relative_orbit = props.get("sat:relative_orbit")

        print(f"\n[{i}]")
        print(f"ID:              {product_id}")
        print(f"Datetime:        {start}")
        print(f"Platform:        {platform}")
        print(f"Orbit state:     {orbit_state}")
        print(f"Relative orbit:  {relative_orbit}")

        # Print useful Sentinel-1 properties if available.
        for key in [
            "sar:instrument_mode",
            "sar:polarizations",
            "processing:level",
            "product:type",
        ]:
            if key in props:
                print(f"{key}: {props[key]}")