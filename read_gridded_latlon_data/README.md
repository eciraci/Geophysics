#### A set of scripts and procedures useful to process/analyze Geospatial date provided in a gridded format.

##### Create Basins Binary Mask given:
- input basin boundaries saved in esri shapefile format;
- reference Latitude / Longitude grid;

    - create_basin_masks.py (Approach 1 - employ multiprocessing approach with concurrent.futures and Python Shapely)
    - create_basin_masks_geop.py (Approach 2 - employ Spatial Join between GeoPandas Geodataframe Objects.

