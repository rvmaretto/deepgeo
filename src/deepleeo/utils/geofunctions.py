import numpy as np
import gdal
import ogr
import os
import subprocess

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
            if (img is None):
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