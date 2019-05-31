import numpy as np
from os import path
import sys
from osgeo import gdal
from osgeo import ogr

sys.path.insert(0, path.join(path.dirname(__file__),"../"))
import dataset.imageutils as iutils
# import utils.geofunctions as gf


class Rasterizer(object):
    def __init__(self, vector_file, in_raster_file, class_column='class', classes_interest=None,
                 non_class_name='non_class'):  # , nodata_val=255):
        self.vector_path = vector_file
        self.raster_path = in_raster_file
        self.class_column = class_column
        self.no_data = 0
        self.base_raster = gdal.Open(self.raster_path) #gf.load_image(self.raster_path)
        if not classes_interest is None:
            self.classes_interest = [non_class_name] + classes_interest
        else:
            self.classes_interest = None
        self.non_class = non_class_name
        self.class_names = []

    def get_class_names(self):
        if self.classes_interest is None:
            return self.class_names
        else:
            return self.classes_interest

    def collect_class_names(self):
        vector_ds = ogr.Open(self.vector_path)
        vector_layer = vector_ds.GetLayer()
        vector_layer.ResetReading()
        unique_labels = set()
        while True:
            feature = vector_layer.GetNextFeature()
            if feature is None:
                break
            name = feature.GetField(self.class_column)
            unique_labels.add(name)

        # Close DataSource Connection
        vector_ds.Destroy()
        self.class_names = []
        for name in sorted(unique_labels):
            self.class_names.append(name)

    def rasterize_label(self, vector_layer):
        mem_drv = gdal.GetDriverByName('MEM')
        mem_raster = mem_drv.Create(
            '',
            self.base_raster.RasterXSize,
            self.base_raster.RasterYSize,
            1,
            gdal.GDT_Int16
        )
        mem_raster.SetProjection(self.base_raster.GetProjection())
        mem_raster.SetGeoTransform(self.base_raster.GetGeoTransform())
        mem_band = mem_raster.GetRasterBand(1)
        mem_band.Fill(self.no_data)
        mem_band.SetNoDataValue(self.no_data)

        err = gdal.RasterizeLayer(
            mem_raster,
            [1],
            vector_layer,
            None,
            None,
            [1],
            options=['ALL_TOUCHED']
        )

        assert(err == gdal.CE_None)
        return mem_raster.ReadAsArray()

    def rasterize_layer(self):
        vector_ds = ogr.Open(self.vector_path)
        vector_layer = vector_ds.GetLayer()
        self.labeled_raster = np.ma.masked_all((self.base_raster.RasterYSize, 
                                self.base_raster.RasterXSize, 1),
                                dtype=np.int32)

        for lid, label in enumerate(self.class_names):
            if self.classes_interest is None:
                value = self.class_names.index(label) + 1
            else:
                if label in self.classes_interest:
                    value = self.classes_interest.index(label) + 1
                else:
                    value = self.classes_interest.index(self.non_class) + 1
            
            vector_layer.SetAttributeFilter("%s='%s'" % (str(self.class_column), str(label)))
            limg = self.rasterize_label(vector_layer)
            self.labeled_raster[limg == 1] = value

        # Close DataSource Connection
        vector_ds.Destroy()

    def get_labeled_raster(self):
        return self.labeled_raster

    def execute(self):
        self.collect_class_names()
        self.rasterize_layer()

    def remove_labels_under_cloud(self, pos_qa=0, new_label=0):
        img_arr = self.base_raster.ReadAsArray()
        img_arr = np.rollaxis(img_arr, 0, 3)
        cl_mask = iutils.compute_cloud_mask(img_arr, pos_qa)

        mask = np.expand_dims(cl_mask, -1)
        mask = mask == 1
        self.labeled_raster[mask] = new_label

    def save_labeled_raster_to_gtiff(self, path_tiff):
        driver = gdal.GetDriverByName('GTiff')
        out_x_size = self.base_raster.GetRasterBand(1).XSize
        out_y_size = self.base_raster.GetRasterBand(1).YSize
        output_ds = driver.Create(path_tiff, out_x_size, out_y_size, 1)
        output_ds.SetProjection(self.base_raster.GetProjection())
        output_ds.SetGeoTransform(self.base_raster.GetGeoTransform())
        output_band = output_ds.GetRasterBand(1)
        output_band.WriteArray(np.ma.filled(self.labeled_raster[:, :, 0], self.no_data))
        output_band.FlushCache()
        output_ds = None