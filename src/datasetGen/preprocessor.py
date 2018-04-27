import numpy as np
import gdal
import sys
from os import path

sys.path.insert(0, path.join(path.dirname(__file__),"../"))
import utils.geofunctions as gf

def computeNDVI(np_raster, parameters):
    print("Computing NDVI")
    red = np_raster[:,:,parameters["idx_b_red"]]
    nir = np_raster[:,:,parameters["idx_b_nir"]]
    ndvi = np.divide((nir-red),(nir+red))
    return ndvi

def computeEVI(np_raster, parameters):
    print("Computing EVI")
    red = np_raster[:,:,parameters["idx_b_red"]]
    nir = np_raster[:,:,parameters["idx_b_nir"]]
    blue = np_raster[:,:,parameters["idx_b_blue"]]
    evi = 2.5 * (np.divide((nir - red), (nir + (6.0 * red) - (7.5 * blue) + 1.0)))
    return evi

class Preprocessor(object):
    predefIndexes = {
        "ndvi": computeNDVI,
        "evi": computeEVI
    }

    def __init__(self, raster_path, vector_path):
        self.raster_path = raster_path
        self.vector_path = vector_path
        self.raster_array = gf.load_image(raster_path)
        # self.img_dataset = gdal.Open(raster_path)
        # self.raster_array = self.img_dataset.ReadAsArray()
        # self.raster_array = np.rollaxis(self.raster_array, 0, start=3)
        

    def compute_indexes(self, indexes, parameters):
        for idx in indexes:
            result = self.predefIndexes[idx](self.raster_array, parameters[idx])
            num_bands = self.raster_array.shape[2]
            self.raster_array = np.dstack((self.raster_array[:,:,:num_bands], result))

        return self.raster_array

    def register_new_func(self, name, function):
        self.predefIndexes[name] = function