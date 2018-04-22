from nose.tools import *
from os import path
import sys

sys.path.insert(0, path.join(path.dirname(__file__),"../../src"))
from datasetGen import rasterizer
import utils.filesystem as fs

class test_rasterizer():
    def setup(self):
        self.data_dir = path.join(path.dirname(__file__), "../../data")
        self.pathVector = path.join(self.data_dir, "PRODES2016_225-64_REP.shp")
        self.pathRaster = path.join(self.data_dir, "Landsat8_225-64_17-07-2016-R6G5B4.tif")
        self.class_column = "agregClass"
        self.rasterizer = rasterizer.Rasterizer(self.pathVector, self.pathRaster, self.class_column)
        self.output_dir = path.join(self.data_dir, "tests_gen")
        fs.mkdir(self.output_dir)

    def test_collect_class_names(self):
        self.rasterizer.collect_class_names()
        class_names = self.rasterizer.get_class_names()
        assert_equal(4, len(class_names))
        assert_equal("DESMATAMENTO", class_names[0])
        assert_equal("FLORESTA", class_names[1])
        assert_equal("HIDROGRAFIA", class_names[2])
        assert_equal("NAO_FLORESTA", class_names[3])

    def test_rasterize_layer(self):
        self.rasterizer.collect_class_names()
        self.rasterizer.rasterize_layer()
        rasterized_layer = self.rasterizer.get_labeled_raster()
        assert_equal(7741, rasterized_layer.shape[0])
        assert_equal(7591, rasterized_layer.shape[1])

    def test_save_to_tiff(self):
        output_file = path.join(self.output_dir, "generatedTIFF.tiff")
        self.rasterizer.collect_class_names()
        self.rasterizer.rasterize_layer()
        self.rasterizer.save_labeled_raster_to_gtiff(output_file)
        assert_true(path.exists(output_file) == 1)