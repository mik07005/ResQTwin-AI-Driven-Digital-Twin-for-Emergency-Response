import geopandas as gpd
import matplotlib.pyplot as plt


ROADS = "data/raw/gcc/roads_centreline.geojson"
INTERSECTIONS = "data/processed/gcc/intersection_points.geojson"


roads = gpd.read_file(ROADS)
points = gpd.read_file(INTERSECTIONS)


# Work in metric CRS for plotting
roads = roads.to_crs("EPSG:32644")
points = points.to_crs("EPSG:32644")


# Random sample of intersections
sample = points.sample(
    min(300, len(points)),
    random_state=42
)


fig, ax = plt.subplots(figsize=(12, 12))

roads.plot(
    ax=ax,
    linewidth=0.4
)

sample.plot(
    ax=ax,
    markersize=5
)

ax.set_title(
    "Sample of GCC Road Intersections"
)

ax.set_axis_off()

plt.tight_layout()

plt.savefig(
    "data/processed/gcc/intersection_sample.png",
    dpi=200
)

plt.show()

print(
    "Saved: "
    "data/processed/gcc/intersection_sample.png"
)