#!/usr/bin/env python
"""
# =================================================================================================================== #
# - Written by Enrico Ciraci - 03/24/2020
# =================================================================================================================== #
# - Create Basins Binary Mask given:
# -
# - input basin shapefile;
# - reference lat/lon grid;
# -
# =================================================================================================================== #
# -  INPUT PARAMETERS:
# - "-S", "--basins" - list of basins to consider passed as csv
# -                    (def. 'indus_river', 'lower_indus_river', 'upper_indus_river')
# - "-B", "--buffer" - boundary shapefile buffer used in the intersection operation. (def. 0)
# - "-N", "--nproc" - number of maximum simultaneous processes (def. 32)
# =================================================================================================================== #
#  OUTPUTS - (see introduction)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#     To Run this procedure digit:
#
#     python create_basins_mask_geop.py
# =================================================================================================================== #
# PYTHON DEPENDENCIES:
#	numpy: Scientific Computing Tools For Python (http://www.numpy.org)
#   netCDF4: python/numpy interface to netCDF library (https://pypi.python.org/pypi/netCDF4)
#   getopt: C-style parser for command line options (https://docs.python.org/2/library/getopt.html)
#   shapely: Manipulation and analysis of geometric objects in the Cartesian plane. (https://shapely.readthedocs.
#            io/en/stable/manual.html)
#   geopandas: GeoPandas is an open source project to make working with geospatial data in python easier.
#           (https://geopandas.org/)
#
# PROGRAM DEPENDENCIES:
#   enrico_library: https://github.com/uci-gravity/Enrico
# =================================================================================================================== #
# - UPDATE -
# =================================================================================================================== #
# IMPORTANT:
# =================================================================================================================== #
"""
# - python dependencies
from __future__ import print_function
import os
import sys
import numpy as np
import netCDF4 as nC4
import getopt
import shapefile
from shapely.geometry import Point
from time import time
import geopandas as gpd


def write_point_shapefile(output_path_shp, lat_vect, lon_vect, attribute_vect, attribute_name=''):
    """
    Save a Point Type esri shapefile with a single attribute per point
    :param output_path_shp: absolute path to the output file
    :param lat_vect: vector containing the latitude points coordinates
    :param lon_vect: vector containing the longitude points coordinates
    :param attribute_vect: vector containing the discharge magnitude
    :param attribute_name: name of the attribute
    :return:
    """
    # - python
    w = shapefile.Writer(output_path_shp[:-4], shapeType=1)
    w.autoBalance = 1
    w.field(attribute_name, 'F', 10, 8)
    # -
    for ind in range(0, len(attribute_vect)):
        w.point(np.float(lon_vect[ind]), np.float(lat_vect[ind]))
        w.record(float(attribute_vect[ind]))
    w.close()
    # - create the PRJ file
    with open(output_path_shp[:-3] + 'prj', "w") as prj:
        epsg = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],                ' \
               'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
        prj.write(epsg)


def verify_and_create_dir(abs_path, dir_name):
    """
    Create directory
    :param abs_path:absolute path to the output directory
    :param dir_name: new directory name
    :return: path to the new directory
    """
    directory = os.path.join(abs_path, dir_name)

    if not os.path.exists(directory):
        os.mkdir(directory)
    # -
    return directory


def calculate_area_mask_vect(lat_vect, lon_vect):
    """
    Calculate Latitude/Longitude Area mask at a selected degree resolution grid
    defined employing the input latitude and longitude vector.
    :param lat_vect: latitude vector
    :param lon_vect: latitude vector
    :return: Area Mask - Expressed in cm2
    """
    # - General Parameters
    dtr = np.pi / 180.    # - Coefficient for the deg to radiant conversion
    rad_e = 6.371e8       # -  Earth Radius in cm
    out_lat = lat_vect
    out_lon = lon_vect
    resolution_lat = np.abs(lat_vect[1] - lat_vect[0])
    resolution_lon = np.abs(lon_vect[1] - lon_vect[0])
    area0 = resolution_lat * resolution_lon * dtr * dtr  # - Area of a single point mass
    out_lat_grid, out_lon_grid = np.meshgrid(out_lat, out_lon)
    # - output area mask
    radius = np.sqrt(area0 * np.cos(out_lat_grid * dtr) / np.pi) * rad_e
    out_mask = np.pi * (radius ** 2)  # - Circular Area in cm^2
    # -
    return out_mask


def write_netcdf_mask(binary_mask, area_mask, lat, lon, file_to_save):
    """
    Write spatial field in a netcdf archive
    :param binary_mask: input binary mask
    :param area_mask: regional area mask in cm2
    :param lat: latitude axis
    :param lon: longitude axis
    :param file_to_save: absolute path to the output fi;e
    :return:
    """
    # - Writing the Output file
    rootgrp = nC4.Dataset(file_to_save, mode='w', format='NETCDF4')
    # - Create Variable dimensions
    rootgrp.createDimension('lat', len(lat))
    rootgrp.createDimension('lon', len(lon))
    # - Create output variables:
    var_lat = rootgrp.createVariable('lat', 'f4', 'lat')
    var_lon = rootgrp.createVariable('lon', 'f4', 'lon')
    var_mask = rootgrp.createVariable('area', 'f8', ('lat', 'lon'))
    var_mask_binary = rootgrp.createVariable('binary', 'f8', ('lat', 'lon'))

    # - Longitude attributes
    var_lon.units = 'degree east'
    var_lon.long_name = 'Longitude'
    var_lon.actual_range = [np.min(lon), np.max(lon)]
    var_lon.standard_name = 'longitude'
    var_lon.axis = 'X'
    var_lon.coordinate_defines = 'point'
    # - Latitude Attributes
    var_lat.units = 'degree north'
    var_lat.long_name = 'Latitude'
    var_lat.actual_range = [np.min(lat), np.max(lat)]
    var_lat.standard_name = 'latitude'
    var_lat.axis = 'Y'
    var_lat.coordinate_defines = 'point'
    # - Mask Attributes
    var_mask.units = 'cm2'
    var_mask.var_desc = 'Basin Area Mask'
    var_mask.actual_range = [np.min(area_mask), np.max(area_mask)]
    # - Binary Mask Attributes
    var_mask_binary.units = ''
    var_mask_binary.var_desc = 'basin binary mask'
    var_mask_binary.actual_range = [np.min(binary_mask), np.max(binary_mask)]

    var_lon[:] = lon
    var_lat[:] = lat
    var_mask[:, :] = area_mask
    var_mask_binary[:, :] = binary_mask
    # -  close the netcdf file created
    rootgrp.close()


def main():
    """
    Main Section of the script.
    :return:
    """
    # -- Read the system arguments listed after the program and run the program
    long_options = ['basins=', 'buffer=']
    try:
        optlist = getopt.getopt(sys.argv[1:], 'S:B:', long_options)
    except ValueError:
        optlist = list()

    # - list of the river basin to consider
    basin_list = ['indus_river']
    # - basin boundaries buffer
    buffer_p = 0.
    try:
        for opt, arg in optlist[0]:
            if opt in ("-S", "--basins"):
                # - list of the basins to process
                # - passed as csv
                basin_list = arg.split(',')
            elif opt in ("-B", "--buffer"):
                # - Basin boundary buffer
                buffer_p = float(arg)
    except ValueError:
        pass
    start = time()

    # - input/output data directory
    input_dir = os.path.join('.', 'masks', 'input')
    output_dir = verify_and_create_dir(os.path.join('.', 'masks'), 'output')

    # - lat/lon arrays
    lat_vect = np.arange(-90,  90+1, 1)
    lon_vect = np.arange(-180,  180+1, 1)
    # - create lat lon domain grid
    l_lon, l_lat = np.meshgrid(lon_vect, lat_vect)
    l_lon_vect = l_lon.flatten()
    l_lat_vect = l_lat.flatten()

    # - Create GeoPandas Data Frame containing the reference let/lon grid
    xy_point = gpd.GeoSeries([Point(x, y) for x, y in list(zip(l_lon_vect, l_lat_vect))])
    df1 = gpd.GeoDataFrame({'geometry': xy_point,
                            'df1': np.arange(len(l_lon_vect)),
                            'lon': l_lon_vect, 'lat': l_lat_vect})

    # - calculate a global area mask in cm2 ad the model resolution
    area_mask = np.transpose(calculate_area_mask_vect(lat_vect, lon_vect))

    # - crop the model mask using the selected river mask
    if basin_list:
        print('# - Create Basins Mask.')
        for basin in basin_list:
            print('# - ' + basin)
            # - create output directory
            out_dir_reg = verify_and_create_dir(output_dir, basin)
            # - load basin boundaries mask into a GeoPandas Data Frame
            shp_path = os.path.join(input_dir, basin, basin+'.shp')
            basin_mask_gdf = gpd.read_file(shp_path)
            basin_mask_geoseries = gpd.GeoSeries([x for x in basin_mask_gdf['geometry']])
            df2 = gpd.GeoDataFrame({'geometry': basin_mask_geoseries,
                                    'df2': np.arange(len(basin_mask_gdf['geometry']))})
            df2['geometry'] = df2.geometry.buffer(buffer_p)

            # - Use GeoPandas spatial Join to find the grid points within the considered
            # - basin,
            res_within = gpd.sjoin(df1, df2, op='within')
            tot_ind = list(res_within['df1'])

            # - Save the obtained mask
            out_lat = np.array(l_lat_vect)[tot_ind]
            out_lon = np.array(l_lon_vect)[tot_ind]
            id_y = np.arange(len(out_lon)) + 1
            output_path_shp = os.path.join(out_dir_reg, basin + '.shp')
            write_point_shapefile(output_path_shp, out_lat, out_lon, id_y, attribute_name='id')

            # - Save the mask also in netcdf format
            binary_mask = np.zeros(np.shape(l_lon))
            # - fill the binary mask
            for k in range(0, len(out_lat)):
                binary_mask[np.where(lat_vect == out_lat[k]), np.where(lon_vect == out_lon[k])] = 1.
            output_path_shp = os.path.join(out_dir_reg, basin + '.nc')
            write_netcdf_mask(binary_mask, area_mask*binary_mask, lat_vect, lon_vect, output_path_shp)

    end = time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'Execution Time: ' + "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


# -- run main program
if __name__ == '__main__':
    main()
