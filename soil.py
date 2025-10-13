import pyodbc
import pandas as pd
import geopandas as gpd
import rioxarray

mdb_path = r"dataset/raw/soil/HWSD2.mdb"

conn_str = (
    r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
    f"DBQ={mdb_path};"
)
conn = pyodbc.connect(conn_str)

# # list all tables
# tables = [t.table_name for t in conn.cursor().tables() if t.table_type == "TABLE"]
# print(tables)

# print("=" * 20)


smu = pd.read_sql("SELECT * FROM HWSD2_SMU", conn)
print(smu.head())
# print(smu.columns)

layers = pd.read_sql("SELECT * FROM HWSD2_LAYERS", conn)
print(layers.head())
# print(layers.columns)

# smu.to_csv("HWSD2_SMU.csv", index=False)
# layers.to_csv("HWSD2_LAYERS.csv", index=False)

def extract_smu_ids(countries):
    world = gpd.read_file("dataset/raw/world/ne_110m_admin_0_countries.shp")
    countries = world[world["NAME"].isin(countries)]

    raster_path = r"dataset/raw/soil/HWSD2.bil"
    rds = rioxarray.open_rasterio(raster_path)

    # ensure CRS match
    countries = countries.to_crs(rds.rio.crs)

    # clip raster to Algeria + Tunisia
    clipped = rds.rio.clip(countries.geometry, from_disk=True)

    smu_ids = pd.Series(clipped.values.flatten())
    smu_ids = smu_ids.dropna().unique().astype(int)
    return smu_ids

smu_ids_algeria = extract_smu_ids(["Algeria"])
# print(len(smu_ids_algeria), smu_ids_algeria)
# print("=" * 20)
smu_ids_tunisia = extract_smu_ids(["Tunisia"])
# print(len(smu_ids_tunisia), smu_ids_tunisia)
# print("=" * 20)

def filter_smu_layers(smu_ids):
    # Filter to selected SMU_IDs
    smu_filtered = smu[smu["HWSD2_SMU_ID"].isin(smu_ids)]
    layers_filtered = layers[layers["HWSD2_SMU_ID"].isin(smu_ids)]

    # Merge attributes
    full = layers_filtered.merge(smu_filtered, on="HWSD2_SMU_ID", how="left")
    return full

full_algeria = filter_smu_layers(smu_ids_algeria)
print(full_algeria.head())
# print(full_algeria.columns)
# print(full_algeria.info())

full_tunisia = filter_smu_layers(smu_ids_tunisia)
print(full_tunisia.head())
# print(full_tunisia.columns)
# print(full_tunisia.info())

full_algeria.to_csv("HWSD2_Algeria.csv", index=False)
full_tunisia.to_csv("HWSD2_Tunisia.csv", index=False)