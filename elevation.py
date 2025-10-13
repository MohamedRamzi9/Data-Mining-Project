import geopandas as gpd

gdf = gpd.read_file("dataset/raw/elevation/gmted2010/gmted2010.shp")

print(gdf.head())
print(gdf.columns)
print(gdf.crs)  # Coordinate Reference System
print("="*20)
print(gdf.info())