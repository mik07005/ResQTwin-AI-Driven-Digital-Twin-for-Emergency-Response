import os
import geopandas as gpd
import networkx as nx
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

EDGE_FILE = "data/processed/gcc/road_edges_final.geojson"

OUTPUT_EDGE_FILE = "data/processed/gcc/road_edges_routing.geojson"
OUTPUT_ROUTE_FILE = "data/processed/gcc/test_time_route.geojson"

# Baseline speed only.
# This is NOT a traffic model.
DEFAULT_SPEED_KMPH = 30.0

# Optional classification-based baseline speeds.
# Only used if a suitable road-class field exists.
SPEED_BY_CLASS = {
    "motorway": 60.0,
    "trunk": 55.0,
    "primary": 45.0,
    "secondary": 40.0,
    "tertiary": 35.0,
    "residential": 25.0,
    "service": 15.0,
}


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("RESQTWIN BASELINE TRAVEL-TIME MODEL")
print("=" * 60)


# ============================================================
# LOAD
# ============================================================

print("\nLoading final road graph...")

edges = gpd.read_file(EDGE_FILE)

print(f"Edges loaded: {len(edges)}")
print(f"CRS: {edges.crs}")


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("INPUT VALIDATION")
print("=" * 60)

required_columns = ["edge_id", "from_node", "to_node", "length_m"]

missing_columns = [
    col for col in required_columns
    if col not in edges.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

invalid_geometry = (
    edges.geometry.isna()
    | edges.geometry.is_empty
    | (~edges.geometry.is_valid)
)

print(
    "Invalid/empty geometries:",
    int(invalid_geometry.sum())
)

if invalid_geometry.any():
    raise ValueError(
        "Invalid or empty edge geometries found."
    )


# ============================================================
# LENGTH VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("LENGTH VALIDATION")
print("=" * 60)

missing_length = edges["length_m"].isna().sum()

non_positive_length = (
    edges["length_m"] <= 0
).sum()

print("Missing length values:", missing_length)
print("Non-positive lengths:", non_positive_length)

if missing_length > 0:
    raise ValueError("Missing length_m values found.")

if non_positive_length > 0:
    raise ValueError(
        "Non-positive length_m values found."
    )

print(
    f"Minimum length: {edges['length_m'].min():.4f} m"
)
print(
    f"Median length: {edges['length_m'].median():.2f} m"
)
print(
    f"Maximum length: {edges['length_m'].max():.2f} m"
)


# ============================================================
# NODE REFERENCE VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("NODE REFERENCE VALIDATION")
print("=" * 60)

node_ids = set(
    pd.concat(
        [
            edges["from_node"],
            edges["to_node"]
        ]
    ).unique()
)

invalid_from = (~edges["from_node"].isin(node_ids)).sum()
invalid_to = (~edges["to_node"].isin(node_ids)).sum()

print("Invalid from_node references:", invalid_from)
print("Invalid to_node references:", invalid_to)


# ============================================================
# ROAD SPEED ASSIGNMENT
# ============================================================

print("\n" + "=" * 60)
print("BASELINE SPEED MODEL")
print("=" * 60)

# Find a useful road classification column if available.
class_column = None

possible_class_columns = [
    "road_class",
    "highway",
    "highway_type",
    "road_type",
    "type",
]

for column in possible_class_columns:
    if column in edges.columns:
        class_column = column
        break


if class_column is not None:

    print(
        f"Road classification field found: {class_column}"
    )

    def assign_speed(value):

        if pd.isna(value):
            return DEFAULT_SPEED_KMPH

        value = str(value).strip().lower()

        return SPEED_BY_CLASS.get(
            value,
            DEFAULT_SPEED_KMPH
        )

    edges["speed_kmph"] = (
        edges[class_column]
        .apply(assign_speed)
    )

else:

    print(
        "No road classification field found."
    )

    print(
        f"Using baseline speed: "
        f"{DEFAULT_SPEED_KMPH} km/h"
    )

    edges["speed_kmph"] = DEFAULT_SPEED_KMPH


print("\nSpeed statistics:")
print(
    edges["speed_kmph"].describe()
)


# ============================================================
# TRAVEL TIME CALCULATION
# ============================================================

print("\n" + "=" * 60)
print("TRAVEL-TIME CALCULATION")
print("=" * 60)

# hours = distance_km / speed_kmph
# minutes = hours * 60

edges["travel_time_min"] = (
    (edges["length_m"] / 1000.0)
    / edges["speed_kmph"]
    * 60.0
)

print(
    f"Minimum travel time: "
    f"{edges['travel_time_min'].min():.4f} min"
)

print(
    f"Median travel time: "
    f"{edges['travel_time_min'].median():.4f} min"
)

print(
    f"Mean travel time: "
    f"{edges['travel_time_min'].mean():.4f} min"
)

print(
    f"Maximum travel time: "
    f"{edges['travel_time_min'].max():.4f} min"
)


# ============================================================
# BUILD NETWORKX GRAPH
# ============================================================

print("\n" + "=" * 60)
print("BUILDING WEIGHTED ROUTING GRAPH")
print("=" * 60)

G = nx.MultiGraph()

for _, row in edges.iterrows():

    G.add_edge(
        int(row["from_node"]),
        int(row["to_node"]),
        edge_id=int(row["edge_id"]),
        length_m=float(row["length_m"]),
        speed_kmph=float(row["speed_kmph"]),
        travel_time_min=float(
            row["travel_time_min"]
        ),
    )


print("Graph nodes:", G.number_of_nodes())
print("Graph edges:", G.number_of_edges())


# ============================================================
# LARGEST CONNECTED COMPONENT
# ============================================================

print("\n" + "=" * 60)
print("CONNECTIVITY")
print("=" * 60)

components = list(
    nx.connected_components(G)
)

components.sort(
    key=len,
    reverse=True
)

largest_component = components[0]

print(
    "Connected components:",
    len(components)
)

print(
    "Largest component:",
    len(largest_component)
)

print(
    "Largest component coverage:",
    f"{len(largest_component) / G.number_of_nodes() * 100:.2f} %"
)


# ============================================================
# DETERMINISTIC TEST NODE SELECTION
# ============================================================

print("\n" + "=" * 60)
print("TEST ROUTE SELECTION")
print("=" * 60)

# IMPORTANT:
# Both nodes come from the largest connected component.
#
# We deliberately avoid arbitrary disconnected nodes.

largest_nodes = sorted(
    largest_component
)

source = largest_nodes[0]
target = largest_nodes[-1]

print("Source node:", source)
print("Target node:", target)


if source == target:

    raise RuntimeError(
        "Source and target resolved to the same node."
    )


# ============================================================
# DISTANCE-OPTIMAL ROUTE
# ============================================================

print("\n" + "=" * 60)
print("DISTANCE-OPTIMAL ROUTE")
print("=" * 60)

distance_path = nx.shortest_path(
    G,
    source=source,
    target=target,
    weight="length_m"
)

distance_edges = list(
    zip(
        distance_path[:-1],
        distance_path[1:]
    )
)

distance_total = 0.0

for u, v in distance_edges:

    edge_data = G.get_edge_data(u, v)

    best_edge = min(
        edge_data.values(),
        key=lambda x: x["length_m"]
    )

    distance_total += best_edge["length_m"]


print("Route nodes:", len(distance_path))
print("Route edges:", len(distance_edges))
print(
    f"Distance: {distance_total:.2f} m"
)
print(
    f"Distance: {distance_total / 1000:.2f} km"
)


# ============================================================
# TIME-OPTIMAL ROUTE
# ============================================================

print("\n" + "=" * 60)
print("TIME-OPTIMAL ROUTE")
print("=" * 60)

time_path = nx.shortest_path(
    G,
    source=source,
    target=target,
    weight="travel_time_min"
)

time_edges = list(
    zip(
        time_path[:-1],
        time_path[1:]
    )
)

time_total = 0.0
time_distance = 0.0

for u, v in time_edges:

    edge_data = G.get_edge_data(u, v)

    best_edge = min(
        edge_data.values(),
        key=lambda x: x["travel_time_min"]
    )

    time_total += best_edge["travel_time_min"]
    time_distance += best_edge["length_m"]


print("Route nodes:", len(time_path))
print("Route edges:", len(time_edges))
print(
    f"Distance: {time_distance:.2f} m"
)
print(
    f"Distance: {time_distance / 1000:.2f} km"
)
print(
    f"Travel time: {time_total:.2f} min"
)


# ============================================================
# ROUTE COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("ROUTE COMPARISON")
print("=" * 60)

distance_route_time = 0.0

for u, v in distance_edges:

    edge_data = G.get_edge_data(u, v)

    best_edge = min(
        edge_data.values(),
        key=lambda x: x["length_m"]
    )

    distance_route_time += (
        best_edge["travel_time_min"]
    )


print(
    f"Distance-optimal route: "
    f"{distance_total / 1000:.2f} km"
)

print(
    f"Time-optimal route: "
    f"{time_distance / 1000:.2f} km"
)

print(
    f"Distance-route estimated time: "
    f"{distance_route_time:.2f} min"
)

print(
    f"Time-optimal route time: "
    f"{time_total:.2f} min"
)

print(
    f"Time saving: "
    f"{distance_route_time - time_total:.2f} min"
)


# ============================================================
# CREATE TIME-OPTIMAL ROUTE GEOMETRY
# ============================================================

print("\n" + "=" * 60)
print("CREATING TIME-OPTIMAL ROUTE")
print("=" * 60)

route_edge_ids = []

for u, v in time_edges:

    edge_data = G.get_edge_data(u, v)

    best_key, best_edge = min(
        edge_data.items(),
        key=lambda item: item[1]["travel_time_min"]
    )

    route_edge_ids.append(
        best_edge["edge_id"]
    )


route_gdf = edges[
    edges["edge_id"].isin(route_edge_ids)
].copy()

# Preserve traversal order
order_map = {
    edge_id: i
    for i, edge_id in enumerate(route_edge_ids)
}

route_gdf["_route_order"] = (
    route_gdf["edge_id"]
    .map(order_map)
)

route_gdf = route_gdf.sort_values(
    "_route_order"
)

route_gdf = route_gdf.drop(
    columns=["_route_order"]
)


# ============================================================
# SAVE ENRICHED EDGES
# ============================================================

print("\nSaving enriched routing graph...")

edges.to_file(
    OUTPUT_EDGE_FILE,
    driver="GeoJSON"
)

print(
    "Saved:",
    OUTPUT_EDGE_FILE
)


# ============================================================
# SAVE TEST ROUTE
# ============================================================

route_gdf.to_file(
    OUTPUT_ROUTE_FILE,
    driver="GeoJSON"
)

print(
    "Saved:",
    OUTPUT_ROUTE_FILE
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAVEL-TIME MODEL SUMMARY")
print("=" * 60)

print("Edges:", len(edges))
print("Nodes:", G.number_of_nodes())

print(
    f"Total network length: "
    f"{edges['length_m'].sum() / 1000:.2f} km"
)

print(
    f"Baseline speed: "
    f"{DEFAULT_SPEED_KMPH:.1f} km/h"
)

print(
    f"Distance-optimal route: "
    f"{distance_total / 1000:.2f} km"
)

print(
    f"Distance-route time: "
    f"{distance_route_time:.2f} min"
)

print(
    f"Time-optimal route: "
    f"{time_distance / 1000:.2f} km"
)

print(
    f"Time-optimal travel time: "
    f"{time_total:.2f} min"
)

print(
    f"Estimated time saving: "
    f"{distance_route_time - time_total:.2f} min"
)

print("\nOutputs:")
print(OUTPUT_EDGE_FILE)
print(OUTPUT_ROUTE_FILE)

print("\n" + "=" * 60)
print("TRAVEL-TIME MODEL COMPLETE")
print("=" * 60)