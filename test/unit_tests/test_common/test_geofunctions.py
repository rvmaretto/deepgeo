from nose.tools import *
from os import path
import sys
# import gdal

sys.path.insert(0, path.join(path.dirname(__file__), '..', '..', '..', 'src'))
import deepgeo.common.geofunctions as gf


class test_geofunctions():

    def setup(self):
        self.data_dir = path.join(path.dirname(__file__), '..', '..', '..', 'data')
        self.pathVector = path.join(self.data_dir, 'prodes_shp_crop.shp')
        self.pathRaster = path.join(self.data_dir, 'raster_R6G5B4.tif')

    def test_load_image(self):
        img = gf.load_image(self.pathRaster)
        assert_equal(851, img.shape[0])
        assert_equal(926, img.shape[1])
        assert_equal(3, img.shape[2])

    # def test_load_vector_layer(self):
    #     layer = gf.load_vector_layer(self.pathVector)
    #     layer.ResetReading()
        # layerDefinition = layer.GetLayerDefn()
        # assert_equal(4, layerDefinition.GetFieldCount())

    # def test_rasterize_layer(self):
    #     vector_layer, class_names = gf.rasterize_vector_file(self.pathVector, "agregClass")
    #     img_ds = gdal.Open(self.pathRaster) # TODO: Is it better to create a class that encapsulate all these methods?
    #     rasterized_labels = gf.rasterize_layer(vector_layer, img_ds, "agregClass", nodata_val=255)
    #     assert_equal(7741, rasterized_labels.shape[0])
    #     assert_equal(7591, rasterized_labels.shape[1])
    #     print("shape: ", rasterized_labels.shape)