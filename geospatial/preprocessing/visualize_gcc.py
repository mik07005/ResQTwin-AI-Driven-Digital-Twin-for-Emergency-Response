import geopandas as gpd
import matplotlib.pyplot as plt

# Load downloaded ward boundaries
wards = gpd.read_file(
    "data/raw/gcc/ward_boundary.geojson"
)

print("Loaded wards:", len(wards))
print("CRS:", wards.crs)

# Plot
wards.plot(
    figsize=(10, 10),
    edgecolor="black"
)

plt.title("Greater Chennai Corporation Wards")
plt.axis("equal")
plt.show()