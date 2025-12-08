
# 08-12-2025
- added `dbscan.ipynb` notebook to implement DBSCAN clustering algorithm from scratch and using sklearn version 

# 24-11-2025
- added `random_forest.ipynb` notebook to implement Random Forest classification model on the merged dataset
- added performance evaluation metrics and confusion matrix visualization to all supervised learning notebooks (`knn.ipynb, decision_tree.ipynb, random_forest.ipynb`)

# 23-11-2025
- fixed negetive values not being handled properly in `soil.ipynb` notebook
- added `supervised_learning` folder containing `knn.ipynb` and `decision_tree.ipynb` notebooks to implement KNN and Decision Tree classification models on the merged dataset

# 15-11-2025
- modified `fire.ipynb` notebook to generate the rest of the positions not in the fire dataset, and balanced the number of fire and no-fire instances for better model training
- modified `elevation.ipynb` notebook to save the data as .tif files for smaller size and raster operations compatibility
- modified `soil.ipynb` notebook to save the data as .parquet files for smaller size and faster read/write operations
- added `merge_datasets.ipynb` notebook to merge fire, soil and elevation datasets based on coordinates

# 10-11-2025
- added export of cleaned fire data as nc file in `fire.ipynb` notebook

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
