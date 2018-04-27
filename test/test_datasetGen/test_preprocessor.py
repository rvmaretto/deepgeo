from nose.tools import *
from os import path
import sys

sys.path.insert(0, path.join(path.dirname(__file__),"../../src"))
import datasetGen.preprocessor as prep
import utils.filesystem as fs

class test_preprocessor():
    def setup(self):
        self.data_dir = path.join(path.dirname(__file__), "../../data")
        self.pathVector = path.join(self.data_dir, "PRODES2016_225-64_REP.shp")
        self.pathRaster = path.join(self.data_dir, "Landsat8_225-64_17-07-2016-B1-7.tif")
        self.class_column = "agregClass"
        self.output_dir = path.join(self.data_dir, "tests_gen")
        self.preproc = prep.Preprocessor(self.pathRaster, self.pathVector)
        fs.mkdir(self.output_dir)

    def test_compute_indexes_NDVI(self):
        parameters = {
            "ndvi": {
                "idx_b_red": 3,
                "idx_b_nir": 4
            }
        }
        ndvi_raster = self.preproc.compute_indexes(["ndvi"], parameters)
        assert_equal(8, ndvi_raster.shape[2])

    def test_compute_two_indexes(self):
        parameters = {
            "ndvi": {
                "idx_b_red": 3,
                "idx_b_nir": 4
            },
            "evi": {
                "idx_b_red": 3,
                "idx_b_blue": 1,
                "idx_b_nir": 4
            }
        }
        new_raster = self.preproc.compute_indexes(["ndvi", "evi"], parameters)
        assert_equal(9, new_raster.shape[2])

    def test_register_new_function(self):
        def subtraction(raster, param):
            return raster[:,:,param["b1"]] - raster[:,:,param["b2"]]

        parameters = {
            "func": {
                "b1": 4,
                "b2": 3
            }
        }

        self.preproc.register_new_func("func", subtraction)
        new_raster = self.preproc.compute_indexes(["func"], parameters)
        assert_equal(8, new_raster.shape[2])
