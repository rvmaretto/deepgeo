
import numpy as np
import math
import scipy.misc
from osgeo import gdal
from osgeo import ogr
import sys
import os
# from matplotlib.colors import ListedColormap
import pylab as pl

sys.path.insert(0, "../")
import functions.filesystem as fs

""" 
TODO: Ao inves de gerar um mosaico, rotacionar a imagem na hora de
      coletar as amostras e em seguida descartar.
TODO: The number of samples passed as parameter should be the total number or the number of samples
      per each rotation angle?
"""

class SampleGenerator(object):
    m_rotationAngles = []
    m_imgDataset = []
    m_labeledImg = []
    m_classNames = []

    def __init__(self, img, labeledImg, classNames):
        self.m_imgDataset = img
        self.m_labeledImg = labeledImg
        self.m_classNames = classNames
        # TODO: Rotation angles must be in the DatasetGenerator
        # if(rotationAngles is None):
        #     self.m_rotationAngles = [0]
        # else:
        #     self.m_rotationAngles = rotationAngles
        # TODO: Put this in the DatasetGenerator class
        #if(isinstance(img, list)):
        # else:
        #     self.m_imgDataset = [img]

    def computeSampleIndexes(self, quantity):
        """
        Sample quantity indices in the labeled image
        """
        sample_candidates = np.transpose(np.nonzero(~self.m_labeledImg.mask))
        indices = np.random.choice(np.arange(len(sample_candidates)), quantity, replace=False)
        self.m_ijSamples = sample_candidates[indices]

    def getSampleIndexes(self):
        return self.m_ijSamples

    # TODO: Is it really necessary?
    # def extractLabels(self):
    #     indexes = self.m_labeledImg[self.m_ijSamples[:,0], self.m_ijSamples[:,1]].filled(255)
    #     assert not np.any(indexes == 255), "Unlabeled points"
    #     return indexes

    def extractWindows(self, win_size, saveOnDisk=False, path="./"):
        self.m_samplesImg = []
        self.m_samplesLabels = []
        self.win_size = win_size

        # count = 0
        for coord in self.m_ijSamples:
            window = self.compute_window_coords(coord)
            sampleImg = self.m_imgDataset[window["upperLin"]:window["lowerLin"], window["rightCol"]:window["leftCol"]]
            sampleLabel = self.m_labeledImg[window["upperLin"]:window["lowerLin"], window["rightCol"]:window["leftCol"]]
            self.m_samplesImg.append(sampleImg)
            self.m_samplesLabels.append(sampleLabel)

    def getSamples(self):
        return self.m_samplesImg, self.m_samplesLabels

    # TODO: Verify this method.
    # TODO: Reimplement
    def saveSamplesToNPZ(self, path, noDataValue=255):
        np.savez(path,
                 imgSamples = self.m_samplesImg,
                 labelSamples = np.ma.filled(self.m_samplesLabels, noDataValue))
                 # id2label=np.array(classNames))  #TODO: Do I have to save this together with the data?

    def saveSamplesToPNG(self, path, colorMap=None):
        for pos in range(len(self.m_samplesImg)):
            samplesDir = os.path.join(path, "sample_imgs")
            labelsDir = os.path.join(path, "sample_labels")
            fs.mkdir(samplesDir)
            fs.mkdir(labelsDir)
            fileName = "sample" + str(pos) + ".png"
            scipy.misc.imsave(os.path.join(samplesDir, fileName), self.m_samplesImg[pos])
            if(colorMap is None):
                scipy.misc.imsave(os.path.join(labelsDir, fileName), self.m_samplesLabels[pos])
            else:
                pl.imsave(fname=os.path.join(labelsDir, fileName), arr=self.m_samplesLabels[pos], cmap=colorMap)

    # TODO: Reimplement
    def saveSamplesToGeoTiff(self, path):
        transform = self.m_imgDataset.GetGeoTransform()

        xOrigin = transform[0]
        yOrigin = transform[3]
        pixelWidth = transform[1]
        pixelHeight = -transform[5]

        for pos in range(len(self.m_ijSamples)):
            samplesDir = os.path.join(path, "sample_imgs")
            labelsDir = os.path.join(path, "sample_labels")
            fs.mkdir(samplesDir)
            fs.mkdir(labelsDir)
            fileName = "sample" + str(pos) + ".tiff"

            coord = self.m_ijSamples[pos]
            window = self.compute_window_coords(coord)

            upperX = window["upperX"] / pixelWidth + xOrigin
            leftY = window["leftY"] / pixelHeight + yOrigin
            lowerX = window["lowerX"] / pixelWidth + xOrigin
            rightY = window["rightY"] / pixelHeight + yOrigin
            coordsWindow = [upperX, leftY, lowerX, rightY]

            new_transform = transform
            new_transform[0] = upperX
            new_transform[3] = leftY

            # TODO: Verify if it will work
            out_ds = gdal.Warp(fileName, self.m_imgDataset, format="GTiff", outputBounds=coordsWindow,
                               xRes = pixelWidth, yRes = pixelHeight, dstSRS = self.m_imgDataset.GetProjectionRef(),
                               resampleAlg=gdal.GRA_NearestNeighbour, option=["COMPRESS=DEFLATE"])
            out_ds = None
            #TODO: Verify gdal.Translate function and CreateCopy
            # driver = gdal.GetDriverByName("GTiff")
            # driver.Create(fileName, self.win_size, self.win_size, self.m_imgDataset.RasterCount,
            #               self.m_imgDataset.GetRasterBand(1).DataType)
            # driver.CreateCopy(fileName, self.m_samplesImg[pos])


    def compute_window_coords(self, coord):
        window_coords = {}
        window_coords["upperLin"] = coord[0] - math.floor(self.win_size / 2)
        window_coords["lowerLin"] = coord[0] + math.ceil(self.win_size / 2)
        window_coords["rightCol"] = coord[1] - math.floor(self.win_size / 2)
        window_coords["leftCol"] = coord[1] + math.ceil(self.win_size / 2)

        return window_coords