import numpy as np
import gdal
import sys
from os import path

sys.path.insert(0, path.join(path.dirname(__file__),"../"))
import utils.geofunctions as gf

def computeNDVI(np_raster, idx_b_red, idx_b_nir):
    print("Computing NDVI")
    red = np_raster[:,:,idx_b_red]
    nir = np_raster[:,:,idx_b_nir]
    ndvi = np.divide((nir-red),(nir+red))
    return ndvi

# def computeEVI(np_raster, idx_b_red, idx_b_nir):
#     print("Computing EVI")

class Preprocessor(object):
    predefIndexes = {
        "ndvi": computeNDVI
        # "evi": computeEVI
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
            if(idx == "ndvi"):
                ndvi = self.predefIndexes[idx](self.raster_array,
                                        parameters[idx]["idx_b_red"],
                                        parameters[idx]["idx_b_nir"])

        num_bands = self.raster_array.shape[2]
        print(num_bands) #TODO: Remove
        print(self.raster_array.shape) #TODO: Remove
        self.raster_array = np.dstack((self.raster_array[:,:,:num_bands], ndvi))
        print(self.raster_array.shape) #TODO: Remove
        return self.raster_array