import numpy as np
import math
import os
import sys
import scipy.misc
import pylab as pl
import gdal

sys.path.insert(0, "../")
import utils.filesystem as fs

class SampleGenerator(object):

    def __init__(self, path_img, labeled_img, class_names, base_raster_path=None):
        self.ref_img = path_img
        self.labeled_img = labeled_img
        self.class_names = class_names
        self.base_raster_path = base_raster_path

    def compute_sample_indexes(self, quantity):
        """
        Sample quantity indices in the labeled image
        """
        sample_candidates = np.transpose(np.nonzero(~self.labeled_img.mask))
        indices = np.random.choice(np.arange(len(sample_candidates)), quantity, replace=False)
        self.ij_samples = sample_candidates[indices]

    def get_sample_indexes(self):
        return self.ij_samples

    def compute_window_coords(self, coord):
        window_coords = {}
        window_coords["upperLin"] = coord[0] - math.floor(self.win_size / 2)
        window_coords["lowerLin"] = coord[0] + math.ceil(self.win_size / 2)
        window_coords["rightCol"] = coord[1] - math.floor(self.win_size / 2)
        window_coords["leftCol"] = coord[1] + math.ceil(self.win_size / 2)

        return window_coords

    def extract_windows(self, win_size):
        self.samples_img = []
        self.samples_labels = []
        self.win_size = win_size

        # count = 0
        for coord in self.ij_samples:
            window = self.compute_window_coords(coord)
            sampleImg = self.ref_img[window["upperLin"]:window["lowerLin"], window["rightCol"]:window["leftCol"]]
            sampleLabel = self.labeled_img[window["upperLin"]:window["lowerLin"], window["rightCol"]:window["leftCol"]]
            self.samples_img.append(sampleImg)
            self.samples_labels.append(sampleLabel)

    def getSamples(self):
        return {
            "img_samples": self.samples_img,
            "labels": self.samples_labels
        }

    def save_samples_PNG(self, path, colorMap=None, r_g_b=[1,2,3]):
        for pos in range(len(self.samples_img)):
            samplesDir = os.path.join(path, "sample_imgs")
            labelsDir = os.path.join(path, "sample_labels")
            fs.mkdir(samplesDir)
            fs.mkdir(labelsDir)
            fileName = "sample" + str(pos) + ".png"
            scipy.misc.imsave(os.path.join(samplesDir, fileName), self.samples_img[pos][:, :, r_g_b])
            if(colorMap is None):
                scipy.misc.imsave(os.path.join(labelsDir, fileName), self.samples_labels[pos])
            else:
                pl.imsave(fname=os.path.join(labelsDir, fileName), arr=self.samples_labels[pos], cmap=colorMap)

    def save_samples_NPZ(self, path, noDataValue=255):
        np.savez(path,
                 img_samples = self.samples_img,
                 label_samples = np.ma.filled(self.samples_labels, noDataValue),
                 class_names=np.array(self.class_names))

    # TODO: From here the methods must be reviewed. How to open the raster dataset to get the geo coordinates?
    def generate_windows_geo_coords(self):
        if(self.base_raster_path is None):
            raise RuntimeError("Base raster path is None. It must exists to generate geographic coordinates.")
        else:
            img_ds = gdal.Open(self.base_raster_path)

        transform = img_ds.GetGeoTransform()

        xOrigin = transform[0]
        yOrigin = transform[3]
        pixelWidth = transform[1]
        pixelHeight = transform[5]

        self.geo_coords = []
        for pos in range(len(self.ij_samples)):
            coord = self.ij_samples[pos]
            window = self.compute_window_coords(coord)

            leftX = xOrigin + (window["leftCol"] * pixelWidth)
            rightX = xOrigin + (window["rightCol"] * pixelWidth)
            upperY = yOrigin + (window["upperLin"] * pixelHeight)
            lowerY = yOrigin + (window["lowerLin"] * pixelHeight)

        img_ds = None

    def save_samples_SHP(self, path):
        if (self.base_raster_path is None):
            raise RuntimeError("Base raster path is None. It must exists to generate geographic coordinates.")
        else:
            img_ds = gdal.Open(self.base_raster_path)
        transform = img_ds.GetGeoTransform()

        xOrigin = transform[0]
        yOrigin = transform[3]
        pixelWidth = transform[1]
        pixelHeight = transform[5]

        img_ds = None