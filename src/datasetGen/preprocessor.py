import numpy as np
import gdal
import sys
from os import path

sys.path.insert(0, path.join(path.dirname(__file__),"../"))
import utils.geofunctions as gf

def compute_NDVI(np_raster, parameters):
    # print("Computing NDVI")
    red = np_raster[:,:,parameters["idx_b_red"]]
    nir = np_raster[:,:,parameters["idx_b_nir"]]
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.true_divide(np.subtract(nir, red), np.add(nir, red))
    return ndvi

def compute_EVI(np_raster, parameters):
    # print("Computing EVI")
    if ("factor" in parameters):
        factor = parameters["factor"]
    else:
        factor = 1
    red = np_raster[:,:,parameters["idx_b_red"]] * factor
    nir = np_raster[:,:,parameters["idx_b_nir"]] * factor
    blue = np_raster[:,:,parameters["idx_b_blue"]] * factor

    with np.errstate(divide='ignore', invalid='ignore'):
        divider = nir - red
        dividend = nir + (6.0 * red) - (7.5 * blue) + 1.0
        evi = 2.5 * (np.true_divide(divider, dividend))

    return evi

def compute_EVI2(np_raster, parameters):
    # print("Computing EVI2")
    red = np_raster[:,:,parameters["idx_b_red"]]
    nir = np_raster[:,:,parameters["idx_b_nir"]]

    with np.errstate(divide='ignore', invalid='ignore'):
        divider = np.subtract(nir, red)
        dividend = nir + (2.4 * red) + 1.0
        evi2 = 2.5 * np.true_divide(divider, dividend)

    return evi2

class Preprocessor(object):
    predefIndexes = {
        "ndvi": compute_NDVI,
        "evi": compute_EVI,
        "evi2": compute_EVI2
    }

    sint_bands = {}

    def __init__(self, raster_path, vector_path):
        self.raster_path = raster_path
        self.vector_path = vector_path
        self.raster_array = gf.load_image(raster_path)
        self.img_dataset = gdal.Open(raster_path)
        # self.raster_array = self.img_dataset.ReadAsArray()
        # self.raster_array = np.rollaxis(self.raster_array, 0, start=3)
        

    def compute_indexes(self, parameters):
        for idx, params in parameters.items():
            result = self.predefIndexes[idx](self.raster_array, params)
            num_bands = self.raster_array.shape[2]
            self.sint_bands[idx] = num_bands
            self.raster_array = np.dstack((self.raster_array[:,:,:num_bands], result))

        return self.raster_array

    def get_position_index_band(self, index):
        return self.sint_bands[index]

    def get_index_band(self, index):
        return self.raster_array[:,:,self.get_position_index_band(index)]

    def get_raster_stacked_raster(self):
        return self.raster_array

    def register_new_func(self, name, function):
        self.predefIndexes[name] = function

    def save_index_raster(self, index, out_path):
        driver = gdal.GetDriverByName("GTiff")
        ds_band = self.img_dataset.GetRasterBand(1)
        out_xSize = ds_band.XSize
        out_ySize = ds_band.YSize
        output_ds = driver.Create(out_path,
                                  out_xSize, out_ySize, 1)
        output_ds.SetProjection(self.img_dataset.GetProjection())
        output_ds.SetGeoTransform(self.img_dataset.GetGeoTransform())
        outputBand = output_ds.GetRasterBand(1)
        outputBand.WriteArray(self.get_index_band(index))

    def save_stacked_raster(self, out_path):
        driver = gdal.GetDriverByName("GTiff")
        ds_band = self.img_dataset.GetRasterBand(1)
        out_xSize = ds_band.XSize
        out_ySize = ds_band.YSize
        nbands = self.raster_array.shape[2]
        output_ds = driver.Create(out_path,
                                  out_xSize, out_ySize, nbands)

        for band in range(nbands):
            output_ds.SetProjection(self.img_dataset.GetProjection())
            output_ds.SetGeoTransform(self.img_dataset.GetGeoTransform())
            outputBand = output_ds.GetRasterBand(band + 1)
            outputBand.WriteArray(self.raster_array[:,:,band])