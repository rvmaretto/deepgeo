# from nose.tools import *
# from os import path
# import sys
# import numpy as np
# from osgeo import gdal
# from osgeo import ogr
# from matplotlib.colors import ListedColormap
#
# sys.path.insert(0, path.join(path.dirname(__file__), "../../src"))
# import deepgeo.dataset.rasterizer as rstzr
# import deepgeo.dataset.sampleGenerator as sg
# import deepgeo.utils.filesystem as fs
#
#
# # TODO: Create a method rasterizer.execute, that execute the whole proccess.
# class TestSampleGenerator():
#     def setup(self):
#         self.data_dir = path.join(path.dirname(__file__), "../../data")
#         self.pathVector = path.join(self.data_dir, "prodes_shp_crop.shp")
#         self.pathRaster = path.join(self.data_dir, "raster_B1_B7.tif")
#         class_column = "agregClass"
#         rasterizer = rstzr.Rasterizer(self.pathVector, self.pathRaster, class_column)
#         self.output_dir = path.join(self.data_dir, "tests_gen")
#
#         rasterizer.collect_class_names()
#         self.class_names = rasterizer.get_class_names()
#         rasterizer.rasterize_layer()
#         self.rasterized_layer = rasterizer.get_labeled_raster()
#
#         #load Raster
#         self.raster_ds = gdal.Open(self.pathRaster)
#         self.raster_img = self.raster_ds.ReadAsArray()
#         self.raster_img = np.rollaxis(self.raster_img, 0, start=3)
#
#         fs.mkdir(self.output_dir)
#
#     def teardown(self):
#         self.raster_ds = None
#         fs.delete_dir(self.output_dir)
#
#     def test_compute_index(self):
#         smpGen = sg.SampleGenerator(self.raster_img, self.rasterized_layer, self.class_names, self.pathRaster)
#         np.random.seed(0)
#         smpGen.compute_indexes(5)
#         smpIdxes = smpGen.get_sample_indexes()
#         assert_equal(5, len(smpIdxes))
#         assert_equal(509, smpIdxes[0][0])
#         assert_equal(534, smpIdxes[0][1])
#         assert_equal(600, smpIdxes[1][0])
#         assert_equal(642, smpIdxes[1][1])
#         assert_equal(246, smpIdxes[2][0])
#         assert_equal(503, smpIdxes[2][1])
#         assert_equal(48, smpIdxes[3][0])
#         assert_equal(339, smpIdxes[3][1])
#         assert_equal(170, smpIdxes[4][0])
#         assert_equal(428, smpIdxes[4][1])
#
#     def test_extract_windows(self):
#         smpGen = sg.SampleGenerator(self.raster_img, self.rasterized_layer, self.class_names, self.pathRaster)
#         smpGen.compute_indexes(5)
#         smpGen.extract_windows(win_size=5)
#         samples = smpGen.get_samples()
#
#         for window in samples["img_samples"]:
#             assert_equal(5, window.shape[0])
#             assert_equal(5, window.shape[1])
#             assert_equal(7, window.shape[2])
#
#         for window in samples["labels"]:
#             assert_equal(5, window.shape[0])
#             assert_equal(5, window.shape[1])
#
#     def test_save_samples_PNG_without_cmap_and_bands(self):
#         output_path = path.join(self.output_dir, "generated_samples")
#         smpGen = sg.SampleGenerator(self.raster_img, self.rasterized_layer, self.class_names, self.pathRaster)
#         smpGen.compute_indexes(quantity=5)
#         smpGen.extract_windows(win_size=5)
#         smpGen.save_samples_PNG(output_path)
#         assert_true(path.exists(output_path))
#
#         samplesDir = path.join(output_path, "sample_imgs")
#         labelsDir = path.join(output_path, "sample_labels")
#         for pos in range(5):
#             fileName = "sample" + str(pos) + ".png"
#             assert_true(path.exists(path.join(samplesDir, fileName)))
#             assert_true(path.exists(path.join(labelsDir, fileName)))
#
#     def test_save_samples_PNG_without_cmap_with_bands(self):
#         output_path = path.join(self.output_dir, "generated_samples")
#         smpGen = sg.SampleGenerator(self.raster_img, self.rasterized_layer, self.class_names, self.pathRaster)
#         smpGen.compute_indexes(quantity=5)
#         smpGen.extract_windows(win_size=5)
#         smpGen.save_samples_PNG(output_path, r_g_b=[5,4,3])
#         assert_true(path.exists(output_path))
#
#         samplesDir = path.join(output_path, "sample_imgs")
#         labelsDir = path.join(output_path, "sample_labels")
#         for pos in range(5):
#             fileName = "sample" + str(pos) + ".png"
#             assert_true(path.exists(path.join(samplesDir, fileName)))
#             assert_true(path.exists(path.join(labelsDir, fileName)))
#
#     def test_save_samples_PNG_all_parameters(self):
#         output_path = path.join(self.output_dir, "generated_samples")
#         smpGen = sg.SampleGenerator(self.raster_img, self.rasterized_layer, self.class_names, self.pathRaster)
#         smpGen.compute_indexes(quantity=5)
#         smpGen.extract_windows(win_size=5)
#
#         colorMap = ListedColormap(["red", "green", "blue", "yellow"])
#         smpGen.save_samples_PNG(output_path, colorMap=colorMap, r_g_b=[5,4,3])
#         assert_true(path.exists(output_path))
#
#         samplesDir = path.join(output_path, "sample_imgs")
#         labelsDir = path.join(output_path, "sample_labels")
#         for pos in range(5):
#             fileName = "sample" + str(pos) + ".png"
#             assert_true(path.exists(path.join(samplesDir, fileName)))
#             assert_true(path.exists(path.join(labelsDir, fileName)))
#
#     def test_save_samples_NPZ(self):
#         output_path = path.join(self.output_dir, "samples_dataset.npz")
#         smpGen = sg.SampleGenerator(self.raster_img, self.rasterized_layer, self.class_names, self.pathRaster)
#         smpGen.compute_indexes(quantity=5)
#         smpGen.extract_windows(win_size=5)
#
#         smpGen.save_samples_NPZ(output_path)
#         assert_true(path.exists(output_path))
#         # TODO: Verify here the contents of the file. Compare to a reference
#
#     def test_save_samples_SHP(self):
#         output_path = path.join(self.output_dir, "samples.shp")
#         smpGen = sg.SampleGenerator(self.raster_img, self.rasterized_layer, self.class_names, self.pathRaster)
#         smpGen.compute_indexes(quantity=5)
#         smpGen.extract_windows(win_size=5)
#
#         smpGen.save_samples_SHP(output_path)
#         assert_true(path.exists(output_path))
#
#         out_ds = ogr.Open(output_path)
#         layer = out_ds.GetLayer()
#
#         assert_equal(5, layer.GetFeatureCount())
#
#         for feature in layer:
#             geom = feature.GetGeometryRef()
#             assert_equal(22500.0, geom.GetArea())
#
#         out_ds.Destroy()