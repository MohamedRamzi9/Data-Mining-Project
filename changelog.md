
# 09-11-2025
- added `soil_elevation_merge.ipynb` notebook to merge soil and elevation data based on coordinates
- changed `soil.ipynb, elevation.ipynb` notebooks to save cleaned data as nc files
- added `fire.ipynb` notebook to perform exploratory data analysis on fire data

# 26-10-2025
- renamed some variables and removed unecessary print statements in `elevation.ipynb, soil.ipynb` notebooks for better readability

# 24-10-2025
- added outlier, missing string values and useless and correlated features handling in `elevation.ipynb` notebook

# 20-10-2025
- reorganized `soil.ipynb` 
- added data analysis and visualization for elevation data in `elevation.ipynb` 

# 19-10-2025
- added `soil.ipynb` notebook in `data_processing` folder to perform exploratory data analysis directly on the loaded soil layers data
- removed `dataset` folder and `soil.py, elevation.py` files from `data_processing` folder from git
- cleaned up `soil.ipynb` notebook by separating analysis parts in different cells and adding comments to explain each step

# 18-10-2025
- changed the code in `soil.py` to generate both csv and nc files containing the layers attributes with averaged coordinates for each SMU ID (might revert back to previous version if needed)
- added code in `elevation.py` to extract and save clipped elevation data as both GeoTIFF and NetCDF formats
- generated `soil_full.csv, soil_full.nc, elevation_full.nc, elevation_full.tif` files in the `dataset` folder containing processed data for Algeria and Tunisia together

# 17-10-2025
- added `TODO.md` file
- added a `dataset` folder to put all csv files containing processed data

# 13-10-2025
- added multiple folders and files to the dataset
- created `soil.py, elevation.py` files and finished elevation processing by generating a csv file for each country (Algeria and Tunisia)


# 10-12-2025
- created the repository
- added `project.pdf, changelog.md` files
