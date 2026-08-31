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