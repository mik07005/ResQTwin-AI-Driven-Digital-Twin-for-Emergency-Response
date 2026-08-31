import requests
import geopandas as gpd


BASE_URL = (
    "https://gisgcc.chennaicorporation.gov.in/"
    "server/rest/services/GCCDepts/GCC_COLLABORATION_LAYER/MapServer"
)

ROAD_LAYER = f"{BASE_URL}/2"

OUTPUT_FILE = "data/raw/gcc/roads_centreline.geojson"

BATCH_SIZE = 2000


def download_roads():
    all_features = []
    offset = 0

    while True:

        print(f"Downloading records {offset} to {offset + BATCH_SIZE - 1}...")

        params = {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "resultOffset": offset,
            "resultRecordCount": BATCH_SIZE,
            "f": "geojson"
        }

        response = requests.get(
            f"{ROAD_LAYER}/query",
            params=params,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        features = data.get("features", [])

        if not features:
            break

        all_features.extend(features)

        print(f"Received: {len(features)}")

        if len(features) < BATCH_SIZE:
            break

        offset += BATCH_SIZE

    print(f"\nTotal downloaded: {len(all_features)}")

    roads = gpd.GeoDataFrame.from_features(
        all_features,
        crs="EPSG:4326"
    )

    roads.to_file(
        OUTPUT_FILE,
        driver="GeoJSON"
    )

    return roads


if __name__ == "__main__":
    roads = download_roads()

    print("\nFinal shape:", roads.shape)
    print("CRS:", roads.crs)
    print("\nGeometry types:")
    print(roads.geometry.geom_type.value_counts())

    print("\nMissing values:")
    print(roads.isna().sum())

    print("\nInvalid geometries:")
    print((~roads.geometry.is_valid).sum())