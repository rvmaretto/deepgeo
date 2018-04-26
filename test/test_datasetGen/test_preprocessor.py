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
        fs.mkdir(self.output_dir)

    def test_compute_indexes(self):
        preproc = prep.Preprocessor(self.pathRaster, self.pathVector)
        parameters = {
            "ndvi": {
                #"raster": "self",
                "idx_b_red": 3,
                "idx_b_nir": 4
            }
        }
        ndvi = preproc.compute_indexes(["ndvi"], parameters)
        print(ndvi.shape)