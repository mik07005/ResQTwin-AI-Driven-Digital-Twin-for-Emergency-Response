import geopandas as gpd
import matplotlib.pyplot as plt


ROAD_FILE = "data/raw/gcc/roads_centreline.geojson"
INTERSECTION_FILE = "data/processed/gcc/intersection_points.geojson"


roads = gpd.read_file(ROAD_FILE)
intersections = gpd.read_file(INTERSECTION_FILE)

# Use metric CRS
roads = roads.to_crs("EPSG:32644")
intersections = intersections.to_crs("EPSG:32644")


# Select 5 random intersection points
samples = intersections.sample(
    min(5, len(intersections)),
    random_state=42
)


for i, (_, point) in enumerate(samples.iterrows(), start=1):

    # Create a small area around the intersection
    buffer = point.geometry.buffer(100)

    nearby_roads = roads[
        roads.geometry.intersects(buffer)
    ]

    fig, ax = plt.subplots(figsize=(8, 8))

    nearby_roads.plot(
        ax=ax,
        linewidth=1
    )

    gpd.GeoSeries([point.geometry]).plot(
        ax=ax,
        markersize=50
    )

    ax.set_title(
        f"Intersection Sample {i}"
    )

    ax.set_axis_off()

    plt.tight_layout()

    output = (
        f"data/processed/gcc/"
        f"intersection_sample_{i}.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output}")


print("\nInspection complete.")