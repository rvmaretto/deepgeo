import gdal
import math
import numpy as np
import osr
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common.utils as utils


class FilesetChipGenerator(object):
    mandatory_params = ['raster_array','labels_array']
    default_params = {}

    def __init__(self, params):
        params = utils.check_dict_parameters(params, self.mandatory_params, self.default_params)
        self.img_array = params['raster_array']
        self.labeled_array = params['labels_array']

    def generate_chips(self):
        row_size, col_size, nbands = self.img_array.shape
        if self.labeled_array is not None:
            if len(self.labeled_array.shape) == 2:
                self.labeled_array = np.expand_dims(self.labeled_array, -1)
            lbl_row_size, lbl_col_size, _ = self.labeled_array.shape
            if row_size != lbl_row_size or col_size != lbl_col_size:
                raise AssertionError('Raster and labels have different sizes (rows and columns)!')
                
            return {'chips': self.img_array,
                    'labels': self.labeled_array,
                    'overlap': (0,0),
                    'coords': None}
        else:
            return {'chips': self.img_array,
                    'overlap': (0,0),
                    'coords': None}
