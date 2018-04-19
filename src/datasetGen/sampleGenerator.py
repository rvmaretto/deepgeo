
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

# TODO: Verificar como configurar a execucao automatica dos testes no github
""" 
TODO: Se o consumo de memoria com os mosaicos ficar muito alto, rotacionar a imagem na hora de
      coletar as amostras e em seguida descartar.
TODO: The number of samples passed as parameter should be the total number or the number of samples
      per each rotation angle?
"""

class SampleGenerator(object):
    m_rotationAngles = []
    m_imgDataset = []
    m_labeledImg = []
    m_classNames = []

    def __init__(self, img, labeledImg, classNames, rotationAngles=None):
        self.m_imgDataset = img
        self.m_labeledImg = labeledImg
        self.m_classNames = classNames
        if(rotationAngles is None):
            self.m_rotationAngles = [0]
        else:
            self.m_rotationAngles = rotationAngles
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
        
    def extractLabels(self):
        y = self.m_labeledImg[self.m_ijSamples[:,0], self.m_ijSamples[:,1]].filled(255)
        assert not np.any(y == 255), "Unlabeled points"
        return y

    # @TODO(Raian): Put in this class the data augmentation functionalities
    def extractWindows(self, win_size, saveOnDisk=False, path="./"):
        self.m_samplesImg = []
        self.m_samplesLabels = []

        # count = 0
        for coord in self.m_ijSamples:
            upperX = coord[0] - math.floor(win_size / 2)
            lowerX = coord[0] + math.ceil(win_size / 2)
            rightY = coord[1] - math.floor(win_size / 2)
            leftY = coord[1] + math.ceil(win_size / 2)
            sampleImg = self.m_imgDataset[upperX:lowerX, rightY:leftY]
            sampleLabel = self.m_labeledImg[upperX:lowerX, rightY:leftY]
            self.m_samplesImg.append(sampleImg)
            self.m_samplesLabels.append(sampleLabel)

    def getSamples(self):
        return self.m_samplesImg, self.m_samplesLabels

    # TODO: Verify this method.
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

# @TODO(Raian): Implement here a way to compute NDVI and another indexes. Allowing that the user passes its own function
# to compute desired different indexes.
