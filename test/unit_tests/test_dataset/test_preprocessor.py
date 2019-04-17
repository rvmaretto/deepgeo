from nose.tools import *
from os import path
import sys
# import warnings

sys.path.insert(0, path.join(path.dirname(__file__), '..', '..', '..', 'src'))
import deepgeo.dataset.preprocessor as prep
import deepgeo.common.filesystem as fs


# TODO: Improve these tests to compare numerically the results
class TestPreprocessor():
    def setup(self):
        self.data_dir = path.join(path.dirname(__file__), '..', '..', '..', 'data')
        self.pathVector = path.join(self.data_dir, 'prodes_shp_crop.shp')
        self.pathRaster = path.join(self.data_dir, 'raster_B1_B7.tif')
        self.class_column = 'agregClass'
        self.output_dir = path.join(self.data_dir, 'tests_gen')
        self.preproc = prep.Preprocessor(self.pathRaster, self.pathVector)
        fs.mkdir(self.output_dir)

    def test_compute_indexes_NDVI(self):
        ndvi_raster = self.preproc.compute_indexes({
            'ndvi': {
                'idx_b_red': 3,
                'idx_b_nir': 4
            }
        })
        assert_equal(8, ndvi_raster.shape[2])
        assert_equal(7, self.preproc.get_position_index_band('ndvi'))

    def test_compute_two_indexes(self):
        new_raster = self.preproc.compute_indexes({
            'ndvi': {
                'idx_b_red': 3,
                'idx_b_nir': 4
            },
            'evi': {
                'idx_b_red': 3,
                'idx_b_blue': 1,
                'idx_b_nir': 4
            }
        })
        assert_equal(9, new_raster.shape[2])
        # assert_equal(8, self.preproc.get_position_index_band('ndvi'))
        # assert_equal(7, self.preproc.get_position_index_band('evi'))

    def test_register_new_function(self):
        def subtraction(raster, param):
            return raster[:,:,param['b1']] - raster[:,:,param['b2']]

        self.preproc.register_new_idx_func('func', subtraction)
        new_raster = self.preproc.compute_indexes({
            'func': {
                'b1': 4,
                'b2': 3
            }
        })
        assert_equal(8, new_raster.shape[2])
        assert_equal(7, self.preproc.get_position_index_band('func'))
