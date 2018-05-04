import numpy as np
import math
import os
import sys
import scipy.misc
import pylab as pl

sys.path.insert(0, "../")
import utils.filesystem as fs

class SampleGenerator(object):

    def __init__(self, img, labeledImg, classNames):
        self.m_ref_img = img
        self.m_labeledImg = labeledImg
        self.m_classNames = classNames

    def compute_sample_indexes(self, quantity):
        """
        Sample quantity indices in the labeled image
        """
        sample_candidates = np.transpose(np.nonzero(~self.m_labeledImg.mask))
        indices = np.random.choice(np.arange(len(sample_candidates)), quantity, replace=False)
        self.m_ijSamples = sample_candidates[indices]

    def get_sample_indexes(self):
        return self.m_ijSamples

    def compute_window_coords(self, coord):
        window_coords = {}
        window_coords["upperLin"] = coord[0] - math.floor(self.win_size / 2)
        window_coords["lowerLin"] = coord[0] + math.ceil(self.win_size / 2)
        window_coords["rightCol"] = coord[1] - math.floor(self.win_size / 2)
        window_coords["leftCol"] = coord[1] + math.ceil(self.win_size / 2)

        return window_coords

    def extract_windows(self, win_size):
        self.m_samples_img = []
        self.m_samples_labels = []
        self.win_size = win_size

        # count = 0
        for coord in self.m_ijSamples:
            window = self.compute_window_coords(coord)
            sampleImg = self.m_ref_img[window["upperLin"]:window["lowerLin"], window["rightCol"]:window["leftCol"]]
            sampleLabel = self.m_labeledImg[window["upperLin"]:window["lowerLin"], window["rightCol"]:window["leftCol"]]
            self.m_samples_img.append(sampleImg)
            self.m_samples_labels.append(sampleLabel)

    def getSamples(self):
        return { "img_samples": self.m_samples_img,
                 "labels": self.m_samples_labels }

    def save_samples_PNG(self, path, colorMap=None, r_g_b=[1,2,3]):
        for pos in range(len(self.m_samples_img)):
            samplesDir = os.path.join(path, "sample_imgs")
            labelsDir = os.path.join(path, "sample_labels")
            fs.mkdir(samplesDir)
            fs.mkdir(labelsDir)
            fileName = "sample" + str(pos) + ".png"
            scipy.misc.imsave(os.path.join(samplesDir, fileName), self.m_samples_img[pos][:,:,r_g_b])
            if(colorMap is None):
                scipy.misc.imsave(os.path.join(labelsDir, fileName), self.m_samples_labels[pos])
            else:
                pl.imsave(fname=os.path.join(labelsDir, fileName), arr=self.m_samples_labels[pos], cmap=colorMap)
