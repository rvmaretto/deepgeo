import numpy as np
import os
import subprocess
from osgeo import gdal
from osgeo import ogr
from osgeo import osr


def load_image(filepath, no_data=None):
    img_ds = gdal.Open(filepath)

    if (no_data is None):
        no_data = 0

    img = None
    for i in range(1, img_ds.RasterCount + 1):
        band = img_ds.GetRasterBand(i)
        band_arr = band.ReadAsArray()
        # band_arr[band_arr == no_data] = 0
        band_arr = np.ma.masked_array(band_arr, band_arr == no_data) # TODO: Vefify how to remove this. How to deal with no_data
        if (img is None):
            img = band_arr
        else:
            img = np.ma.dstack((img, band_arr))

    return img


def generate_multi_raster_structure(path_images, band_names=None, no_data=None):
    if not isinstance(path_images, list):
        path_images = [path_images]

    multi_raster_struct = {
        "file_paths": [],
        "band_names": [],
        "raster_arrays": []
    }

    for img_path in path_images:
        multi_raster_struct["file_paths"].append(img_path)

        if no_data is None:
            no_data = 0

        img_ds = gdal.Open(img_path)

        if band_names is None:
            band_names = []
            for i in range(1, img_ds.RasterCount + 1):
                band_name = img_ds.GetRasterBand(i).GetDescription()
                if band_name == '':
                    band_name = "Band_" + str(i)
                band_names.append(band_name)

        img = None
        for i in range(1, img_ds.RasterCount + 1):
            band = img_ds.GetRasterBand(i)
            band_arr = band.ReadAsArray()
            band_arr[band_arr == no_data] = 0
            band_arr = np.ma.masked_array(band_arr,
                                          band_arr == 0)  # TODO: Vefify how to remove this. How to deal with no_data
            if img is None:
                img = band_arr
            else:
                img = np.ma.dstack((img, band_arr))

        multi_raster_struct["raster_arrays"].append(img)
        multi_raster_struct["band_names"] = band_names

    return multi_raster_struct


def load_vector_layer(filename):
    vector_ds = ogr.Open(filename)
    layer = vector_ds.GetLayer()
    return layer


#TODO: Extend this method to other file formats
def merge_vector_layers(files, output_file):
    if not isinstance(files, list):
        raise TypeError("Argument \"files\" must be a list.")

    if len(files) < 2:
        raise Exception("You must provide at least two files.")

    arguments_1 = ['ogr2ogr', '-f', 'ESRI Shapefile', output_file, files[0]]

    if os.path.exists(output_file):
        os.remove(output_file)

    print("Merging Files...")

    ps = subprocess.Popen(arguments_1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = ps.communicate()
    for line in output[0].splitlines():
        print(str(line, 'utf-8'))

    for line in output[1].splitlines():
        print(str(line, 'utf-8'))

    for i in range(1, len(files)):
        arguments_2 = ['ogr2ogr', '-f', 'ESRI Shapefile', '-update', '-append',
                       output_file, files[i]]#, '-nln', 'PRODES']

        ps = subprocess.Popen(arguments_2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = ps.communicate()
        for line in output[0].splitlines():
            print(str(line, 'utf-8'))

        for line in output[1].splitlines():
            print(str(line, 'utf-8'))


# The following code is based in the code available in the following
# link: https://gis.stackexchange.com/questions/264618/reprojecting-and-saving-shapefile-in-gdal
def reproj_shape_to_raster(path_in_shp, path_raster, path_out_shp):
    # Opens the base raster
    base_tiff = gdal.Open(path_raster)

    # shapefile with the from projection
    driver = ogr.GetDriverByName("ESRI Shapefile")
    in_ds =   driver.Open(path_in_shp, 1)
    in_layer = in_ds.GetLayer()

    # set spatial reference and transformation
    sourceprj = in_layer.GetSpatialRef()
    targetprj = osr.SpatialReference(wkt = base_tiff.GetProjection())
    # create the CoordinateTransformation
    coord_transform = osr.CoordinateTransformation(sourceprj, targetprj)

    # create the output layer
    out_driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(path_out_shp):
        out_driver.DeleteDataSource(path_out_shp)
    out_ds = out_driver.CreateDataSource(path_out_shp)
    out_layer = out_ds.CreateLayer('', targetprj, ogr.wkbPolygon)

    # add fields
    in_layerDefn = in_layer.GetLayerDefn()
    for i in range(0, in_layerDefn.GetFieldCount()):
        field_defn = in_layerDefn.GetFieldDefn(i)
        out_layer.CreateField(field_defn)

    # get the output layer's feature definition
    out_layer_defn = out_layer.GetLayerDefn()

    # loop through the input features
    in_feature = in_layer.GetNextFeature()
    while in_feature:
        # get the input geometry
        geom = in_feature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coord_transform)
        geom = ogr.CreateGeometryFromWkb(geom.ExportToWkb())
        # create a new feature
        out_feature = ogr.Feature(out_layer_defn)
        # set the geometry and attribute
        out_feature.SetGeometry(geom)
        for i in range(0, out_layer_defn.GetFieldCount()):
            out_feature.SetField(out_layer_defn.GetFieldDefn(i).GetNameRef(), in_feature.GetField(i))
        # add the feature to the shapefile
        out_layer.CreateFeature(out_feature)
        # dereference the features and get the next input feature
        out_feature = None
        in_feature = in_layer.GetNextFeature()

    out_ds = None
    in_ds = None


def write_chips(output_path, base_raster, pred_struct, output_format='GTiff', dataType=gdal.GDT_UInt16):
    driver = gdal.GetDriverByName(output_format)
    base_ds = gdal.Open(base_raster)

    x_start, pixel_width, _, y_start, _, pixel_height = base_ds.GetGeoTransform()
    x_size = base_ds.RasterXSize
    y_size = base_ds.RasterYSize

    srs = osr.SpatialReference()
    srs.ImportFromWkt(base_ds.GetProjectionRef())

    out_ds = driver.Create(output_path, x_size, y_size, 1, dataType)
    out_ds.SetGeoTransform((x_start, pixel_width, 0, y_start, 0, pixel_height))
    out_ds.SetProjection(srs.ExportToWkt())
    out_band = out_ds.GetRasterBand(1)

    for idx in range(1, len(pred_struct['chips'])):
        chip = pred_struct['chips'][idx]
        chip = np.squeeze(chip)
        x_start = pred_struct['coords'][idx]['upper_row'] - pred_struct['overlap'][0]
        y_start = pred_struct['coords'][idx]['left_col'] - pred_struct['overlap'][1]
        out_band.WriteArray(chip, y_start, x_start)

    out_band.FlushCache()
