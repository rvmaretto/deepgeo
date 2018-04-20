from nose.tools import *
import os
import sys
import gdal

sys.path.insert(0, os.path.join(os.path.dirname(__file__),"../../src"))
import utils.geofunctions as gf

class test_geofunctions():

    def setup(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "../../data")
        self.pathVector = os.path.join(self.data_dir, "PRODES2016_225-64_REP.shp")
        self.pathRaster = os.path.join(self.data_dir, "Landsat8_225-64_17-07-2016-R6G5B4.tif")

    def test_load_image(self):
        img = gf.load_image(self.pathRaster)
        assert_equal(7741, img.shape[0])
        assert_equal(7591, img.shape[1])
        assert_equal(3, img.shape[2])

    def test_rasterize_vector_file(self):
        _, class_names = gf.rasterize_vector_file(self.pathVector, "agregClass")
        assert_equal(4, len(class_names))
        assert_equal("DESMATAMENTO", class_names[0])
        assert_equal("FLORESTA", class_names[1])
        assert_equal("HIDROGRAFIA", class_names[2])
        assert_equal("NAO_FLORESTA", class_names[3])

    def test_rasterize_layer(self):
        vector_layer, class_names = gf.rasterize_vector_file(self.pathVector, "agregClass")
        img_ds = gdal.Open(self.pathRaster) # TODO: Is it better to create a class that encapsulate all these methods?
        rasterized_labels = gf.rasterize_layer(vector_layer, img_ds, "agregClass", nodata_val=255)
        assert_equal(7741, rasterized_labels.shape[0])
        assert_equal(7591, rasterized_labels.shape[1])
        print("shape: ", rasterized_labels.shape)