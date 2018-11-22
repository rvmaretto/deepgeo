import numpy as np
import math
import os
import sys
import scipy.misc
import pylab as pl
from osgeo import gdal
from osgeo import ogr
from osgeo import osr #TODO: Verify if it is really necessary? If I get the SRID from the Raster I still need this?

sys.path.insert(0, os.path.join(os.path.dirname(__file__),"../"))
import utils.filesystem as fs
import dataset.data_augment as daug
import dataset.sequencialchips as seqchips

#TODO: Review this class to work with strategies.

class ChipsGenerator(object):
    strategies = {
        "sequential": seqchips.generate_sequential_chips
    }

    def __init__(self, path_img, labeled_img, class_names, base_raster_path=None):
        self.ref_img = path_img
        self.labeled_img = labeled_img
        self.class_names = class_names
        self.base_raster_path = base_raster_path

    def compute_sample_indexes(self, quantity, class_of_interest=None):
        """
        Sample quantity indices in the labeled image
        """
        if class_of_interest is None:
            self.sample_candidates = np.transpose(np.nonzero(~self.labeled_img.mask))
        else:
            label_interest = self.class_names.index(class_of_interest)
            self.sample_candidates = np.transpose(np.nonzero(np.logical_and(~self.labeled_img.mask,
                                                                       self.labeled_img == label_interest)))
        indices = np.random.choice(np.arange(len(self.sample_candidates)), quantity, replace=False)
        self.ij_samples = self.sample_candidates[indices]

    def get_sample_indexes(self):
        return self.ij_samples

    def compute_window_coords(self, coord):
        window_coords = {}
        window_coords["upperLin"] = coord[0] - math.floor(self.win_size / 2)
        window_coords["lowerLin"] = coord[0] + math.ceil(self.win_size / 2)
        window_coords["rightCol"] = coord[1] + math.floor(self.win_size / 2)
        window_coords["leftCol"] = coord[1] - math.ceil(self.win_size / 2)

        # TODO: Review this. Is there a better way to do this?
        if(window_coords["upperLin"] < 0):
            window_coords["upperLin"] = 0
            window_coords["lowerLin"] = self.win_size
        if(window_coords["leftCol"] < 0):
            window_coords["leftCol"] = 0
            window_coords["rightCol"] = self.win_size
        if(window_coords["lowerLin"] > self.labeled_img.shape[0]):
            window_coords["lowerLin"] = self.labeled_img.shape[0]
            window_coords["upperLin"] = window_coords["lowerLin"] - self.win_size
        if(window_coords["rightCol"] > self.labeled_img.shape[1]):
            window_coords["rightCol"] = self.labeled_img.shape[1]
            window_coords["leftCol"] = window_coords["rightCol"] - self.win_size

        return window_coords

    def extract_windows(self, win_size):
        self.samples_img = []
        self.samples_labels = []
        self.win_size = win_size

        count = 0
        for coord in self.ij_samples:
            window = self.compute_window_coords(coord)
            sampleImg = self.ref_img[window["upperLin"]:window["lowerLin"], window["leftCol"]:window["rightCol"]]
            sampleLabel = self.labeled_img[window["upperLin"]:window["lowerLin"], window["leftCol"]:window["rightCol"]]
            # if np.count_nonzero(sampleLabel.mask) == 0:
            #     self.samples_img.append(sampleImg)
            #     self.samples_labels.append(sampleLabel)
            # else:
            #     # print("Chip with No data")
            #     # print(np.unique(sampleLabel))
            while np.count_nonzero(sampleLabel.mask) != 0:
                indice = np.random.choice(np.arange(len(self.sample_candidates)), 1, replace=False)
                coord = self.sample_candidates[indice][0]
                self.ij_samples[count] = coord
                window = self.compute_window_coords(coord)
                sampleImg = self.ref_img[window["upperLin"]:window["lowerLin"], window["leftCol"]:window["rightCol"]]
                sampleLabel = self.labeled_img[window["upperLin"]:window["lowerLin"], window["leftCol"]:window["rightCol"]]

            self.samples_img.append(sampleImg)
            self.samples_labels.append(sampleLabel)
            count = count + 1

        self.generate_windows_geo_coords()

    def applyDataAugmentation(self, rot_angles=[90, 180, 270], rotation=True, flip=True):
        rot_imgs = daug.rotate_images(self.samples_img, rot_angles)
        rot_labels = daug.rotate_images(self.samples_labels, rot_angles)
        flip_imgs = daug.flip_images(self.samples_img)
        flip_labels = daug.flip_images(self.samples_labels)

        self.samples_img = np.concatenate(self.samples_img, rot_imgs)
        self.samples_img = np.concatenate(self.samples_img, flip_imgs)
        self.samples_labels = np.concatenate(self.samples_labels, rot_labels)
        self.samples_labels = np.concatenate(self.samples_labels, flip_labels)

    def getSamples(self):
        return {
            "images": self.samples_img,
            "labels": self.samples_labels,
            "classes": self.class_names
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
                scipy.misc.imsave(os.path.join(labelsDir, fileName), self.samples_labels[pos][:,:,0])
            else:
                pl.imsave(fname=os.path.join(labelsDir, fileName), arr=self.samples_labels[pos][:,:,0], cmap=colorMap)

    def save_samples_NPZ(self, path, noDataValue=255):
        if os.path.exists(path):
            os.remove(path)
        # print("UNIQUE: ", np.unique(self.samples_labels))
        np.savez(path,
                 images = self.samples_img,
                 labels= np.ma.filled(self.samples_labels, noDataValue),
                 classes=np.array(self.class_names))

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
            geo_coord = []
            coord = self.ij_samples[pos]
            window = self.compute_window_coords(coord)

            upperY = yOrigin + (window["upperLin"] * pixelHeight)
            lowerY = yOrigin + (window["lowerLin"] * pixelHeight)
            leftX = xOrigin + (window["leftCol"] * pixelWidth)
            rightX = xOrigin + (window["rightCol"] * pixelWidth)

            geo_coord.append(upperY)
            geo_coord.append(lowerY)
            geo_coord.append(leftX)
            geo_coord.append(rightX)
            self.geo_coords.append(geo_coord)

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

        driver = ogr.GetDriverByName("ESRI Shapefile")

        if os.path.exists(path):
            driver.DeleteDataSource(path)

        prj = img_ds.GetProjection()
        srs = osr.SpatialReference(wkt=prj)

        # create the data source
        output_ds = driver.CreateDataSource(path)
        layer_name = os.path.splitext(os.path.basename(path))[0]
        layer = output_ds.CreateLayer(layer_name, srs, ogr.wkbPolygon)

        for pos in range(len(self.geo_coords)):
            coord = self.geo_coords[pos]
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(coord[2], coord[0])
            ring.AddPoint(coord[2], coord[1])
            ring.AddPoint(coord[3], coord[1])
            ring.AddPoint(coord[3], coord[0])
            ring.AddPoint(coord[2], coord[0])

            polygon = ogr.Geometry(ogr.wkbPolygon)
            polygon.AddGeometry(ring)

            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetGeometry(polygon)
            layer.CreateFeature(feature)

            feature.Destroy()
            polygon.Destroy()
            ring.Destroy()

        output_ds.Destroy()
        img_ds = None