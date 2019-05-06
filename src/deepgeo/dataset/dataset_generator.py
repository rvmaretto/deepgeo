import numpy as np
import math
import os
import sys
import scipy.misc
import pylab as pl
from osgeo import gdal
from osgeo import ogr
from osgeo import osr  # TODO: Verify if it is really necessary? If I get the SRID from the Raster I still need this?

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))
import common.filesystem as fs
import dataset.sequential_chips as seqchips
import dataset.random_chips as rdmchips


class DatasetGenerator(object):
    strategies = {
        'sequential': seqchips.SequentialChipGenerator,
        'random': rdmchips.RandomChipGenerator
    }

    def __init__(self, path_img, labeled_img, class_names, strategy='sequential', base_raster_path=None):
        self.strategy = strategy
        self.ref_img = path_img
        self.labeled_img = labeled_img
        self.class_names = class_names
        self.base_raster_path = base_raster_path

    def generate_chips(self):
        params = {'raster_array': self.ref_img,
                  'labeled_array': self.labeled_img,
                  'win_size': 128}
        self.chips_struct = self.strategies[self.strategy](params).generate_chips()

    def get_samples(self):
        return self.chips_struct
        #     {
        #     'images': self.samples_img,
        #     'labels': self.samples_labels,
        #     'classes': self.class_names
        # }

    def save_samples_PNG(self, path, color_map=None, r_g_b=[1, 2, 3]):
        for pos in range(len(self.samples_img)):
            samples_dir = os.path.join(path, 'sample_imgs')
            labels_dir = os.path.join(path, 'sample_labels')
            fs.mkdir(samples_dir)
            fs.mkdir(labels_dir)
            file_name = 'sample' + str(pos) + '.png'
            scipy.misc.imsave(os.path.join(samples_dir, file_name), self.samples_img[pos][:, :, r_g_b])
            if color_map is None:
                scipy.misc.imsave(os.path.join(labels_dir, file_name), self.samples_labels[pos][:, :, 0])
            else:
                pl.imsave(fname=os.path.join(labels_dir, file_name), arr=self.samples_labels[pos][:, :, 0],
                          cmap=color_map)

    def save_samples_NPZ(self, path, no_data=255):
        if os.path.exists(path):
            os.remove(path)
        np.savez(path,
                 images=self.samples_img,
                 labels=np.ma.filled(self.samples_labels, no_data),
                 classes=np.array(self.class_names))
