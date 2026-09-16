import geopandas as gpd
import pandas as pd
import networkx as nx

from collections import Counter


# ============================================================
# PATHS
# ============================================================

CLEAN_EDGES_FILE = "data/processed/gcc/road_edges_clean.geojson"
CLEAN_NODES_FILE = "data/processed/gcc/road_nodes_clean.geojson"

FINAL_EDGES_FILE = "data/processed/gcc/road_edges_final.geojson"
FINAL_NODES_FILE = "data/processed/gcc/road_nodes_final.geojson"

ACCEPTED_SNAPS_FILE = "data/processed/gcc/accepted_snaps.geojson"
REJECTED_SNAPS_FILE = "data/processed/gcc/rejected_snaps.geojson"


# ============================================================
# SETTINGS
# ============================================================

# Maximum distance we consider a strong snap.
STRONG_SNAP_MAX_M = 0.5

# Warning threshold for suspicious accepted snaps.
WARNING_SNAP_MAX_M = 0.5


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("FINAL ROAD GRAPH SNAP VALIDATION")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading graph files...")

clean_edges = gpd.read_file(CLEAN_EDGES_FILE)
clean_nodes = gpd.read_file(CLEAN_NODES_FILE)

final_edges = gpd.read_file(FINAL_EDGES_FILE)
final_nodes = gpd.read_file(FINAL_NODES_FILE)

accepted_snaps = gpd.read_file(ACCEPTED_SNAPS_FILE)

try:
    rejected_snaps = gpd.read_file(REJECTED_SNAPS_FILE)
except Exception:
    rejected_snaps = None


print("Clean edges:", len(clean_edges))
print("Clean nodes:", len(clean_nodes))

print("Final edges:", len(final_edges))
print("Final nodes:", len(final_nodes))

print("Accepted snaps:", len(accepted_snaps))

if rejected_snaps is not None:
    print("Rejected/review snaps:", len(rejected_snaps))


# ============================================================
# CRS CHECK
# ============================================================

print("\n" + "=" * 60)
print("CRS VALIDATION")
print("=" * 60)

print("Clean edges CRS:", clean_edges.crs)
print("Final edges CRS:", final_edges.crs)
print("Clean nodes CRS:", clean_nodes.crs)
print("Final nodes CRS:", final_nodes.crs)

crs_ok = (
    clean_edges.crs is not None
    and final_edges.crs is not None
    and clean_nodes.crs is not None
    and final_nodes.crs is not None
)

print("CRS present:", "YES" if crs_ok else "NO")


# ============================================================
# BASIC FILE VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("BASIC GEOMETRY VALIDATION")
print("=" * 60)


def geometry_errors(gdf):

    invalid = 0
    empty = 0
    null = 0

    for geom in gdf.geometry:

        if geom is None:
            null += 1

        elif geom.is_empty:
            empty += 1

        elif not geom.is_valid:
            invalid += 1

    return null, empty, invalid


final_null, final_empty, final_invalid = geometry_errors(
    final_edges
)

print("Null geometries:", final_null)
print("Empty geometries:", final_empty)
print("Invalid geometries:", final_invalid)

geometry_ok = (
    final_null == 0
    and final_empty == 0
    and final_invalid == 0
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

print("\n" + "=" * 60)
print("ATTRIBUTE VALIDATION")
print("=" * 60)

required_edge_columns = [
    "edge_id",
    "from_node",
    "to_node",
    "length_m"
]

missing_edge_columns = [
    col
    for col in required_edge_columns
    if col not in final_edges.columns
]

print("Missing required edge columns:", len(missing_edge_columns))

if missing_edge_columns:
    print(missing_edge_columns)
else:
    print("Required edge attributes: OK")


required_node_columns = [
    "node_id"
]

missing_node_columns = [
    col
    for col in required_node_columns
    if col not in final_nodes.columns
]

print("Missing required node columns:", len(missing_node_columns))

if missing_node_columns:
    print(missing_node_columns)
else:
    print("Required node attributes: OK")


# ============================================================
# NODE ID VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("NODE ID VALIDATION")
print("=" * 60)

node_ids = set(
    final_nodes["node_id"].astype(int)
)

from_nodes = set(
    final_edges["from_node"].dropna().astype(int)
)

to_nodes = set(
    final_edges["to_node"].dropna().astype(int)
)

referenced_nodes = from_nodes | to_nodes

invalid_from = from_nodes - node_ids
invalid_to = to_nodes - node_ids

print("Final node IDs:", len(node_ids))
print("Referenced nodes:", len(referenced_nodes))

print("Invalid from_node references:", len(invalid_from))
print("Invalid to_node references:", len(invalid_to))

node_reference_ok = (
    len(invalid_from) == 0
    and len(invalid_to) == 0
)


# ============================================================
# EDGE ID VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("EDGE ID VALIDATION")
print("=" * 60)

edge_ids = final_edges["edge_id"]

duplicate_edge_ids = edge_ids.duplicated().sum()

missing_edge_ids = edge_ids.isna().sum()

print("Duplicate edge IDs:", duplicate_edge_ids)
print("Missing edge IDs:", missing_edge_ids)

edge_id_ok = (
    duplicate_edge_ids == 0
    and missing_edge_ids == 0
)


# ============================================================
# EDGE LENGTH VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("EDGE LENGTH VALIDATION")
print("=" * 60)

lengths = pd.to_numeric(
    final_edges["length_m"],
    errors="coerce"
)

missing_lengths = lengths.isna().sum()

zero_lengths = (
    lengths <= 0
).sum()

negative_lengths = (
    lengths < 0
).sum()

short_edges = (
    lengths <= 0.5
).sum()

print("Missing lengths:", missing_lengths)
print("Zero/negative length edges:", zero_lengths)
print("Edges <= 0.5 m:", short_edges)

print("Minimum length:", lengths.min())
print("Median length:", lengths.median())
print("Mean length:", lengths.mean())
print("Maximum length:", lengths.max())

length_ok = (
    missing_lengths == 0
    and negative_lengths == 0
    and zero_lengths == 0
)


# ============================================================
# SELF LOOP VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("SELF-LOOP VALIDATION")
print("=" * 60)

self_loops = final_edges[
    final_edges["from_node"].astype(int)
    ==
    final_edges["to_node"].astype(int)
].copy()

print("Self-loop edges:", len(self_loops))

if len(self_loops) > 0:

    print(
        self_loops[
            [
                "edge_id",
                "from_node",
                "to_node",
                "length_m"
            ]
        ].head(20).to_string(index=False)
    )

self_loop_ok = len(self_loops) == 0


# ============================================================
# DUPLICATE NODE CONNECTION VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("NODE CONNECTION VALIDATION")
print("=" * 60)


def normalize_pair(row):

    a = int(row["from_node"])
    b = int(row["to_node"])

    if a <= b:
        return (a, b)

    return (b, a)


final_pairs = final_edges.apply(
    normalize_pair,
    axis=1
)

pair_counts = Counter(final_pairs)

parallel_pairs = {
    pair: count
    for pair, count in pair_counts.items()
    if count > 1
}

parallel_edge_count = sum(
    count - 1
    for count in parallel_pairs.values()
)

print("Unique node pairs:", len(pair_counts))
print("Node pairs with parallel edges:", len(parallel_pairs))
print("Extra parallel edges:", parallel_edge_count)

if len(parallel_pairs) > 0:

    print("\nTop parallel connections:")

    top_parallel = sorted(
        parallel_pairs.items(),
        key=lambda x: x[1],
        reverse=True
    )[:15]

    for pair, count in top_parallel:

        print(
            f"{pair[0]} - {pair[1]} : {count} edges"
        )


# ============================================================
# ACCEPTED SNAP VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("ACCEPTED SNAP VALIDATION")
print("=" * 60)

print("Accepted snaps:", len(accepted_snaps))


if len(accepted_snaps) > 0:

    print(
        "Accepted snap columns:",
        list(accepted_snaps.columns)
    )

    if "distance_m" in accepted_snaps.columns:

        snap_distances = pd.to_numeric(
            accepted_snaps["distance_m"],
            errors="coerce"
        )

        print("\nSnap distance statistics:")

        print(
            "Minimum:",
            snap_distances.min(),
            "m"
        )

        print(
            "Median:",
            snap_distances.median(),
            "m"
        )

        print(
            "Mean:",
            snap_distances.mean(),
            "m"
        )

        print(
            "Maximum:",
            snap_distances.max(),
            "m"
        )

        print(
            "Snaps <= 0.1 m:",
            (snap_distances <= 0.1).sum()
        )

        print(
            "Snaps <= 0.5 m:",
            (snap_distances <= 0.5).sum()
        )

        print(
            "Snaps > 0.5 m:",
            (snap_distances > 0.5).sum()
        )

        suspicious_snaps = accepted_snaps[
            snap_distances > WARNING_SNAP_MAX_M
        ]

        print(
            "Suspicious accepted snaps:",
            len(suspicious_snaps)
        )

    else:

        snap_distances = None

        print(
            "WARNING: distance_m attribute not found."
        )

else:

    snap_distances = None

    suspicious_snaps = accepted_snaps


# ============================================================
# ACCEPTED SNAP NODE VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("SNAP NODE REFERENCE VALIDATION")
print("=" * 60)

snap_node_columns = []

for col in [
    "from_node",
    "to_node",
    "node_a",
    "node_b"
]:

    if col in accepted_snaps.columns:
        snap_node_columns.append(col)


if snap_node_columns:

    print(
        "Node reference fields found:",
        snap_node_columns
    )

    for col in snap_node_columns:

        values = pd.to_numeric(
            accepted_snaps[col],
            errors="coerce"
        ).dropna().astype(int)

        invalid = sum(
            value not in node_ids
            for value in values
        )

        print(
            f"{col} invalid references:",
            invalid
        )

else:

    print(
        "No explicit snap node ID fields found."
    )

    print(
        "Snap geometry will be validated against final graph."
    )


# ============================================================
# SNAP CONNECTOR DETECTION
# ============================================================

print("\n" + "=" * 60)
print("SNAP CONNECTOR VALIDATION")
print("=" * 60)

clean_edge_ids = set(
    clean_edges["edge_id"].astype(int)
)

final_edge_ids = set(
    final_edges["edge_id"].astype(int)
)

new_edge_ids = final_edge_ids - clean_edge_ids

print(
    "Original clean edge IDs:",
    len(clean_edge_ids)
)

print(
    "Final edge IDs:",
    len(final_edge_ids)
)

print(
    "New edge IDs:",
    len(new_edge_ids)
)

print(
    "Expected new connector edges:",
    len(accepted_snaps)
)


# ============================================================
# CONNECTIVITY FUNCTION
# ============================================================

def graph_connectivity(edges):

    graph = nx.Graph()

    for row in edges.itertuples():

        u = int(row.from_node)
        v = int(row.to_node)

        graph.add_edge(u, v)

    components = list(
        nx.connected_components(graph)
    )

    components.sort(
        key=len,
        reverse=True
    )

    if len(components) == 0:

        return {
            "components": 0,
            "largest": 0,
            "second": 0,
            "coverage": 0,
            "small": 0
        }

    largest = len(components[0])

    second = (
        len(components[1])
        if len(components) > 1
        else 0
    )

    coverage = (
        largest / len(graph.nodes)
    ) * 100

    small = sum(
        len(component) <= 10
        for component in components
    )

    return {
        "components": len(components),
        "largest": largest,
        "second": second,
        "coverage": coverage,
        "small": small
    }


# ============================================================
# CONNECTIVITY COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("CONNECTIVITY COMPARISON")
print("=" * 60)

clean_connectivity = graph_connectivity(
    clean_edges
)

final_connectivity = graph_connectivity(
    final_edges
)

print("\nCLEAN GRAPH")

print(
    "Connected components:",
    clean_connectivity["components"]
)

print(
    "Largest component:",
    clean_connectivity["largest"]
)

print(
    "Coverage:",
    round(
        clean_connectivity["coverage"],
        2
    ),
    "%"
)

print(
    "Components <= 10 nodes:",
    clean_connectivity["small"]
)


print("\nFINAL GRAPH")

print(
    "Connected components:",
    final_connectivity["components"]
)

print(
    "Largest component:",
    final_connectivity["largest"]
)

print(
    "Coverage:",
    round(
        final_connectivity["coverage"],
        2
    ),
    "%"
)

print(
    "Components <= 10 nodes:",
    final_connectivity["small"]
)


component_change = (
    clean_connectivity["components"]
    -
    final_connectivity["components"]
)

largest_change = (
    final_connectivity["largest"]
    -
    clean_connectivity["largest"]
)

coverage_change = (
    final_connectivity["coverage"]
    -
    clean_connectivity["coverage"]
)

print("\nCONNECTIVITY IMPROVEMENT")

print(
    "Component change:",
    component_change
)

print(
    "Largest component change:",
    largest_change,
    "nodes"
)

print(
    "Coverage change:",
    round(
        coverage_change,
        2
    ),
    "percentage points"
)


# ============================================================
# NODE DEGREE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("FINAL NODE DEGREE VALIDATION")
print("=" * 60)

degree_graph = nx.Graph()

for row in final_edges.itertuples():

    u = int(row.from_node)
    v = int(row.to_node)

    degree_graph.add_edge(u, v)

# Include all final nodes
for node_id in node_ids:

    degree_graph.add_node(node_id)


degrees = dict(
    degree_graph.degree()
)

degree_values = list(
    degrees.values()
)

degree_0 = sum(
    degree == 0
    for degree in degree_values
)

degree_1 = sum(
    degree == 1
    for degree in degree_values
)

degree_2 = sum(
    degree == 2
    for degree in degree_values
)

degree_3_plus = sum(
    degree >= 3
    for degree in degree_values
)

max_degree = max(
    degree_values
) if degree_values else 0

print("Degree 0:", degree_0)
print("Degree 1:", degree_1)
print("Degree 2:", degree_2)
print("Degree >= 3:", degree_3_plus)
print("Maximum degree:", max_degree)

print("Isolated nodes:", degree_0)


# ============================================================
# FINAL NETWORK LENGTH
# ============================================================

print("\n" + "=" * 60)
print("FINAL NETWORK METRICS")
print("=" * 60)

total_length_km = (
    lengths.sum()
    /
    1000
)

print(
    "Total network length:",
    round(total_length_km, 2),
    "km"
)

print(
    "Average edge length:",
    round(lengths.mean(), 2),
    "m"
)

print(
    "Median edge length:",
    round(lengths.median(), 2),
    "m"
)

print(
    "Maximum edge length:",
    round(lengths.max(), 2),
    "m"
)


# ============================================================
# ROUTING GRAPH TEST
# ============================================================

print("\n" + "=" * 60)
print("ROUTING GRAPH TEST")
print("=" * 60)

routing_graph = nx.Graph()

for row in final_edges.itertuples():

    u = int(row.from_node)
    v = int(row.to_node)
    length = float(row.length_m)

    routing_graph.add_edge(
        u,
        v,
        weight=length
    )

for node_id in node_ids:

    routing_graph.add_node(node_id)


routing_nodes = routing_graph.number_of_nodes()
routing_edges = routing_graph.number_of_edges()

print("Routing graph nodes:", routing_nodes)
print("Routing graph edges:", routing_edges)

# Test shortest-path capability inside largest component.
largest_component = max(
    nx.connected_components(routing_graph),
    key=len
)

largest_subgraph = routing_graph.subgraph(
    largest_component
)

if len(largest_component) >= 2:

    test_nodes = list(largest_component)

    source = test_nodes[0]
    target = test_nodes[-1]

    try:

        test_distance = nx.shortest_path_length(
            largest_subgraph,
            source,
            target,
            weight="weight"
        )

        print(
            "Shortest-path test: PASS"
        )

        print(
            "Test route distance:",
            round(test_distance, 2),
            "m"
        )

        routing_test_ok = True

    except nx.NetworkXNoPath:

        print(
            "Shortest-path test: FAIL"
        )

        routing_test_ok = False

else:

    print(
        "Shortest-path test: FAIL"
    )

    routing_test_ok = False


# ============================================================
# FINAL VERDICT
# ============================================================

print("\n" + "=" * 60)
print("FINAL ROUTING READINESS")
print("=" * 60)


checks = {

    "Geometry valid":
        geometry_ok,

    "Node references valid":
        node_reference_ok,

    "Edge IDs valid":
        edge_id_ok,

    "Edge lengths valid":
        length_ok,

    "No self-loops":
        self_loop_ok,

    "Routing graph build":
        routing_nodes > 0
        and routing_edges > 0,

    "Shortest-path test":
        routing_test_ok
}


for name, result in checks.items():

    print(
        f"{name}:",
        "PASS" if result else "FAIL"
    )


all_pass = all(
    checks.values()
)


print("\n" + "=" * 60)

if all_pass:

    print("ROUTING READY: YES")

else:

    print("ROUTING READY: NO")

print("=" * 60)


# ============================================================
# IMPORTANT INTERPRETATION
# ============================================================

print("\nInterpretation:")

if all_pass:

    print(
        "The final road graph passed structural "
        "and routing-readiness validation."
    )

    print(
        "The graph can now be used for "
        "emergency routing and optimization."
    )

else:

    print(
        "The graph still contains one or more "
        "validation failures."
    )

    print(
        "Review the failed checks before "
        "starting routing optimization."
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL VALIDATION SUMMARY")
print("=" * 60)

print(
    "Final nodes:",
    len(final_nodes)
)

print(
    "Final edges:",
    len(final_edges)
)

print(
    "Accepted snaps:",
    len(accepted_snaps)
)

print(
    "Connected components:",
    final_connectivity["components"]
)

print(
    "Largest component:",
    final_connectivity["largest"]
)

print(
    "Largest component coverage:",
    round(
        final_connectivity["coverage"],
        2
    ),
    "%"
)

print(
    "Network length:",
    round(
        total_length_km,
        2
    ),
    "km"
)

print(
    "Self-loops:",
    len(self_loops)
)

print(
    "Shortest-path test:",
    "PASS" if routing_test_ok else "FAIL"
)

print(
    "Routing ready:",
    "YES" if all_pass else "NO"
)

print("\nFinal road graph validation complete.")