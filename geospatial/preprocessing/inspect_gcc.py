import geopandas as gpd
import pandas as pd


ROAD_FILE = "data/raw/gcc/roads_centreline.geojson"


# Load roads
roads = gpd.read_file(ROAD_FILE)

print("=" * 60)
print("BASIC INFORMATION")
print("=" * 60)

print("Shape:", roads.shape)
print("CRS:", roads.crs)

print("\nColumns:")
print(roads.columns.tolist())


print("\n" + "=" * 60)
print("GEOMETRY")
print("=" * 60)

print(roads.geometry.geom_type.value_counts())

print("\nInvalid geometries:")
print((~roads.geometry.is_valid).sum())


print("\n" + "=" * 60)
print("ROAD ID CHECK")
print("=" * 60)

print("Total records:", len(roads))
print("Unique road IDs:", roads["road_id"].nunique())
print("Missing road IDs:", roads["road_id"].isna().sum())

duplicate_ids = roads[
    roads["road_id"].duplicated(keep=False)
]

print("Records with duplicate road IDs:", len(duplicate_ids))


print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(roads.isna().sum())


print("\n" + "=" * 60)
print("ROAD LENGTH CHECK")
print("=" * 60)

length_column = "first_length_of_the_road_in_met"

print("Length statistics:")
print(roads[length_column].describe())

print("\nGeometry length statistics:")
print(roads["st_length(shape)"].describe())


print("\n" + "=" * 60)
print("ROAD WIDTH CHECK")
print("=" * 60)

width_column = "first_average_width_of_the_road"

print(roads[width_column].describe())


print("\n" + "=" * 60)
print("ZONE / WARD")
print("=" * 60)

print("Number of zones:", roads["zone"].nunique())
print("Number of wards:", roads["ward"].nunique())

print("\nRoads per zone:")
print(roads["zone"].value_counts().sort_index())


print("\n" + "=" * 60)
print("ROAD DEPARTMENT")
print("=" * 60)

print(roads["department"].value_counts(dropna=False))


print("\n" + "=" * 60)
print("ROAD SAMPLE")
print("=" * 60)

print(
    roads[
        [
            "road_id",
            "road_name",
            "area_name",
            "locality",
            "zone",
            "ward",
            length_column,
            width_column
        ]
    ].head(10).to_string()
)

print("\n" + "=" * 60)
print("DUPLICATE ROAD ID ANALYSIS")
print("=" * 60)

# Find records with duplicated road IDs
duplicate_roads = roads[
    roads["road_id"].notna()
    & roads["road_id"].duplicated(keep=False)
].copy()

print("Records involved in duplicate road IDs:",
      len(duplicate_roads))

print("Number of duplicated road IDs:",
      duplicate_roads["road_id"].nunique())


# Show some examples
print("\nExample duplicated road IDs:")

duplicate_counts = (
    duplicate_roads["road_id"]
    .value_counts()
    .head(20)
)

print(duplicate_counts)


print("\nDetailed examples:")

example_ids = duplicate_counts.head(5).index

print(
    duplicate_roads[
        duplicate_roads["road_id"].isin(example_ids)
    ][
        [
            "road_id",
            "road_name",
            "area_name",
            "locality",
            "zone",
            "ward",
            "first_length_of_the_road_in_met",
            "first_average_width_of_the_road",
            "geometry"
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 60)
print("BLANK STRING CHECK")
print("=" * 60)

for column in [
    "road_id",
    "road_name",
    "area_name",
    "locality",
    "department",
    "first_length_of_the_road_in_met",
    "first_average_width_of_the_road"
]:
    if roads[column].dtype == "object":
        blank_count = (
            roads[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        print(f"{column}: {blank_count} blank strings")


print("\n" + "=" * 60)
print("MISSING ROAD ID BY ZONE")
print("=" * 60)

print(
    roads[roads["road_id"].isna()]
    ["zone"]
    .value_counts()
    .sort_index()
)


print("\n" + "=" * 60)
print("MISSING WIDTH BY ZONE")
print("=" * 60)

print(
    roads[roads["first_average_width_of_the_road"].isna()]
    ["zone"]
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 60)
print("MISSING WIDTH BY ZONE")
print("=" * 60)

print(
    roads[roads["first_average_width_of_the_road"].isna()]
    ["zone"]
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 60)
print("FEATURE ID CHECK")
print("=" * 60)

print("Total features:", len(roads))
print("Unique objectids:", roads["objectid"].nunique())
print("Missing objectids:", roads["objectid"].isna().sum())

print(
    "Duplicate objectids:",
    roads["objectid"].duplicated().sum()
)


print("\n" + "=" * 60)
print("EXACT GEOMETRY DUPLICATE CHECK")
print("=" * 60)

geometry_duplicates = roads.geometry.to_wkb().duplicated().sum()

print("Exact duplicate geometries:", geometry_duplicates)


print("\n" + "=" * 60)
print("DUPLICATE ROAD ID SUMMARY")
print("=" * 60)

duplicate_summary = (
    roads[roads["road_id"].notna()]
    .groupby("road_id")
    .agg(
        feature_count=("objectid", "count"),
        road_name_count=("road_name", "nunique"),
        geometry_count=("geometry", "nunique"),
        ward_count=("ward", "nunique")
    )
)

print("Road IDs with multiple features:")
print(
    duplicate_summary[
        duplicate_summary["feature_count"] > 1
    ]
    .head(20)
)


print("\n" + "=" * 60)
print("MISSING ROAD ID DETAILS")
print("=" * 60)

missing_id = roads[roads["road_id"].isna()]

print("Total:", len(missing_id))

print("\nBy zone:")
print(
    missing_id["zone"]
    .value_counts()
    .sort_index()
)

print("\nSample:")
print(
    missing_id[
        [
            "objectid",
            "road_name",
            "area_name",
            "locality",
            "zone",
            "ward",
            "department"
        ]
    ]
    .head(20)
    .to_string(index=False)
)