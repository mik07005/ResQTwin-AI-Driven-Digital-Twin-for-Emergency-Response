from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]

gap = gpd.read_file(
    ROOT / "data" / "processed" / "chennai" / "aoi"
    / "gcc_remaining_gap_after_db41.geojson"
)

gcc = gpd.read_file(
    ROOT / "data" / "processed" / "chennai" / "aoi"
    / "gcc_boundary_2025_utm44n.geojson"
)

gap = gap.to_crs("EPSG:32644")
gcc = gcc.to_crs("EPSG:32644")

fig, ax = plt.subplots(figsize=(10, 10))

gcc.boundary.plot(ax=ax, linewidth=1)
gap.plot(ax=ax, alpha=0.7)

ax.set_title("Remaining GCC Area Not Covered by B07B + DB41")
ax.set_xlabel("UTM Easting (m)")
ax.set_ylabel("UTM Northing (m)")

plt.tight_layout()

output = (
    ROOT / "data" / "processed" / "chennai" / "aoi"
    / "remaining_gcc_gap.png"
)

plt.savefig(output, dpi=200)
plt.close()

print(f"Gap area: {gap.geometry.area.sum() / 1_000_000:.6f} km²")
print(f"Saved: {output}")