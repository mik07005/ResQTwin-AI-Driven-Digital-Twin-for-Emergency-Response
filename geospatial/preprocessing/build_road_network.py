import geopandas as gpd

from shapely.ops import unary_union
from shapely.geometry import (
    LineString,
    MultiLineString,
    GeometryCollection,
    Point
)


# ============================================================
# PATHS
# ============================================================

ROAD_FILE = "data/raw/gcc/roads_centreline.geojson"

NODE_OUTPUT = "data/processed/gcc/road_nodes.geojson"
EDGE_OUTPUT = "data/processed/gcc/road_edges.geojson"


# ============================================================
# SETTINGS
# ============================================================

# Coordinates are in EPSG:32644 (metres).
# 6 decimal places = 1 micrometre, so this removes
# floating-point noise without meaningfully changing the network.
COORD_PRECISION = 6


# ============================================================
# HELPER
# ============================================================

def coordinate_key(coordinate):
    """
    Convert a coordinate into a stable dictionary key.

    Rounding avoids floating-point precision problems when
    assigning node IDs to edge endpoints.
    """

    return (
        round(float(coordinate[0]), COORD_PRECISION),
        round(float(coordinate[1]), COORD_PRECISION)
    )


# ============================================================
# START
# ============================================================

print("=" * 60)
print("BUILDING GCC ROAD NETWORK")
print("=" * 60)


# ============================================================
# LOAD ROADS
# ============================================================

roads = gpd.read_file(ROAD_FILE)

print("Original features:", len(roads))
print("CRS:", roads.crs)


# ============================================================
# EXPLODE MULTILINESTRINGS
# ============================================================

segments = roads.explode(
    index_parts=False
).copy()

segments = segments.reset_index(drop=True)

print("Segments after explode:", len(segments))


# ============================================================
# PROJECT TO METRIC CRS
# ============================================================

segments = segments.to_crs("EPSG:32644")

print("Working CRS:", segments.crs)


# ============================================================
# REMOVE EMPTY / INVALID GEOMETRIES
# ============================================================

segments = segments[
    segments.geometry.notna()
    & ~segments.geometry.is_empty
].copy()

# Keep only line geometries
segments = segments[
    segments.geometry.geom_type.isin(
        ["LineString", "MultiLineString"]
    )
].copy()

segments = segments.reset_index(drop=True)

print("Valid segments:", len(segments))


# ============================================================
# NODE THE ROAD NETWORK
# ============================================================

print()
print("Creating noded road network...")
print("This may take some time...")

network = unary_union(segments.geometry)


# ============================================================
# EXTRACT LINEWORK
# ============================================================

if isinstance(network, LineString):

    network_lines = [network]

elif isinstance(network, MultiLineString):

    network_lines = list(network.geoms)

elif isinstance(network, GeometryCollection):

    network_lines = [
        geom
        for geom in network.geoms
        if isinstance(
            geom,
            (LineString, MultiLineString)
        )
    ]

    # Flatten MultiLineStrings inside GeometryCollection
    flattened = []

    for geom in network_lines:

        if isinstance(geom, LineString):
            flattened.append(geom)

        elif isinstance(geom, MultiLineString):
            flattened.extend(list(geom.geoms))

    network_lines = flattened

else:

    raise TypeError(
        f"Unexpected geometry type: {network.geom_type}"
    )


print(
    "Network edges after noding:",
    len(network_lines)
)


# ============================================================
# CREATE EDGE GEODATAFRAME
# ============================================================

edges = gpd.GeoDataFrame(
    {
        "edge_id": range(len(network_lines))
    },
    geometry=network_lines,
    crs="EPSG:32644"
)


# ============================================================
# REMOVE ZERO-LENGTH EDGES
# ============================================================

edges["length_m"] = edges.geometry.length

edges = edges[
    edges["length_m"] > 0
].copy()

edges = edges.reset_index(drop=True)

# Reassign edge IDs after filtering
edges["edge_id"] = range(len(edges))

print(
    "Edges after removing zero-length geometries:",
    len(edges)
)


# ============================================================
# CREATE NODE TABLE DIRECTLY FROM EDGE ENDPOINTS
# ============================================================

print()
print("Creating road nodes...")


coordinate_to_node = {}
node_coordinates = []


for geom in edges.geometry:

    start = coordinate_key(
        geom.coords[0]
    )

    end = coordinate_key(
        geom.coords[-1]
    )

    # Create node for start point if necessary
    if start not in coordinate_to_node:

        node_id = len(node_coordinates)

        coordinate_to_node[start] = node_id

        node_coordinates.append(start)


    # Create node for end point if necessary
    if end not in coordinate_to_node:

        node_id = len(node_coordinates)

        coordinate_to_node[end] = node_id

        node_coordinates.append(end)


print(
    "Unique nodes created:",
    len(node_coordinates)
)


# ============================================================
# CREATE NODE GEODATAFRAME
# ============================================================

nodes = gpd.GeoDataFrame(
    {
        "node_id": range(
            len(node_coordinates)
        )
    },
    geometry=[
        Point(x, y)
        for x, y in node_coordinates
    ],
    crs="EPSG:32644"
)


# ============================================================
# ASSIGN FROM / TO NODE IDs
# ============================================================

print()
print("Assigning node IDs to edges...")


from_nodes = []
to_nodes = []

mapping_failures = 0


for geom in edges.geometry:

    start = coordinate_key(
        geom.coords[0]
    )

    end = coordinate_key(
        geom.coords[-1]
    )

    if start not in coordinate_to_node:
        mapping_failures += 1
        from_nodes.append(-1)
    else:
        from_nodes.append(
            coordinate_to_node[start]
        )

    if end not in coordinate_to_node:
        mapping_failures += 1
        to_nodes.append(-1)
    else:
        to_nodes.append(
            coordinate_to_node[end]
        )


edges["from_node"] = from_nodes
edges["to_node"] = to_nodes


# ============================================================
# CHECK MAPPING
# ============================================================

print(
    "Node mapping failures:",
    mapping_failures
)


if mapping_failures > 0:

    raise RuntimeError(
        "Some edge endpoints could not be mapped "
        "to nodes. Network construction stopped."
    )


# ============================================================
# CHECK SELF-LOOPS
# ============================================================

self_loops = (
    edges["from_node"]
    == edges["to_node"]
).sum()

print(
    "Self-loop edges:",
    self_loops
)


# ============================================================
# FINAL EDGE COLUMNS
# ============================================================

edges = edges[
    [
        "edge_id",
        "from_node",
        "to_node",
        "length_m",
        "geometry"
    ]
].copy()


# ============================================================
# NETWORK STATISTICS
# ============================================================

total_length_km = (
    edges["length_m"].sum()
    / 1000
)

average_length = (
    edges["length_m"].mean()
)

minimum_length = (
    edges["length_m"].min()
)

maximum_length = (
    edges["length_m"].max()
)


# ============================================================
# SAVE AS EPSG:4326
# ============================================================

print()
print("Saving GeoJSON files...")


nodes_out = nodes.to_crs("EPSG:4326")
edges_out = edges.to_crs("EPSG:4326")


nodes_out.to_file(
    NODE_OUTPUT,
    driver="GeoJSON"
)

edges_out.to_file(
    EDGE_OUTPUT,
    driver="GeoJSON"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("NETWORK SUMMARY")
print("=" * 60)

print(
    "Original road features:",
    len(roads)
)

print(
    "Exploded road segments:",
    len(segments)
)

print(
    "Final network edges:",
    len(edges)
)

print(
    "Final network nodes:",
    len(nodes)
)

print(
    "Total network length (km):",
    round(total_length_km, 2)
)

print(
    "Average edge length (m):",
    round(average_length, 2)
)

print(
    "Shortest edge (m):",
    round(minimum_length, 2)
)

print(
    "Longest edge (m):",
    round(maximum_length, 2)
)

print(
    "Self-loop edges:",
    self_loops
)

print(
    "Node mapping failures:",
    mapping_failures
)

print()
print("Saved:")
print(NODE_OUTPUT)
print(EDGE_OUTPUT)

print()
print("Road network construction complete.")