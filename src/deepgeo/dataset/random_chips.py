import math
import numpy as np


def compute_indexes(self, quantity, class_of_interest=None):
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


def compute_window_coords(self, coord):
    window_coords = {}
    window_coords['upperLin'] = coord[0] - math.floor(self.win_size / 2)
    window_coords['lowerLin'] = coord[0] + math.ceil(self.win_size / 2)
    window_coords['rightCol'] = coord[1] + math.floor(self.win_size / 2)
    window_coords['leftCol'] = coord[1] - math.ceil(self.win_size / 2)

    # TODO: Review this. Is there a better way to do this?
    if window_coords['upperLin'] < 0:
        window_coords['upperLin'] = 0
        window_coords['lowerLin'] = self.win_size
    if window_coords['leftCol'] < 0:
        window_coords['leftCol'] = 0
        window_coords['rightCol'] = self.win_size
    if window_coords['lowerLin'] > self.labeled_img.shape[0]:
        window_coords['lowerLin'] = self.labeled_img.shape[0]
        window_coords['upperLin'] = window_coords['lowerLin'] - self.win_size
    if window_coords['rightCol'] > self.labeled_img.shape[1]:
        window_coords['rightCol'] = self.labeled_img.shape[1]
        window_coords['leftCol'] = window_coords['rightCol'] - self.win_size

    return window_coords


def extract_windows(win_size):
    samples_img = []
    samples_labels = []
    win_size = win_size

    count = 0
    for coord in self.ij_samples:
        window = self.compute_window_coords(coord)
        sampleImg = self.ref_img[window['upperLin']:window['lowerLin'], window['leftCol']:window['rightCol']]
        sampleLabel = self.labeled_img[window['upperLin']:window['lowerLin'], window['leftCol']:window['rightCol']]

        while np.count_nonzero(sampleLabel.mask) != 0:
            indice = np.random.choice(np.arange(len(self.sample_candidates)), 1, replace=False)
            coord = self.sample_candidates[indice][0]
            self.ij_samples[count] = coord
            window = self.compute_window_coords(coord)
            sampleImg = self.ref_img[window['upperLin']:window['lowerLin'], window['leftCol']:window['rightCol']]
            sampleLabel = self.labeled_img[window['upperLin']:window['lowerLin'],
                          window['leftCol']:window['rightCol']]

        self.samples_img.append(sampleImg)
        self.samples_labels.append(sampleLabel)
        count = count + 1

    self.generate_windows_geo_coords()


def generate_random_chips():
    return True