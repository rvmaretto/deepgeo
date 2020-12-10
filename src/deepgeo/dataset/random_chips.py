import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common.utils as utils


class RandomChipGenerator(object):
    mandatory_params = ['raster_array', 'labels_array', 'win_size', 'quantity', 'class_names']
    default_params = {'class_of_interest': None,
                      'remove_no_data': None}

    def __init__(self, params):
        params = utils.check_dict_parameters(params, self.mandatory_params, self.default_params)
        self.ref_img = params['raster_array']
        self.labeled_img = params['labels_array']
        self.win_size = params['win_size']
        self.class_of_interest = params['class_of_interest']
        self.quantity = params['quantity']
        self.class_names = params['class_names']

    def compute_indexes(self):
        """
        Sample quantity indices in the labeled image
        """
        if self.class_of_interest is None:
            self.sample_candidates = np.transpose(np.nonzero(~self.labeled_img.mask))
        elif isinstance(self.class_of_interest, list):
            self.sample_candidates = []
            for clazz in self.class_of_interest:
                label_interest = self.class_names.index(clazz)
                self.sample_candidates.append(np.transpose(np.nonzero(np.logical_and(~self.labeled_img.mask,
                                                                                self.labeled_img == label_interest))))
            self.sample_candidates = np.concatenate(self.sample_candidates, axis=0)
        else:
            label_interest = self.class_names.index(self.class_of_interest)
            self.sample_candidates = np.transpose(np.nonzero(np.logical_and(~self.labeled_img.mask,
                                                                            self.labeled_img == label_interest)))
        indices = np.random.choice(np.arange(len(self.sample_candidates)), self.quantity, replace=False)
        self.ij_samples = self.sample_candidates[indices]
        print(self.ij_samples, type(self.ij_samples))

    def compute_window_coords(self, coord):
        window_coords = {}
        window_coords['upper_row'] = coord[0] - math.floor(self.win_size / 2)
        window_coords['lower_row'] = coord[0] + math.ceil(self.win_size / 2)
        window_coords['right_col'] = coord[1] + math.floor(self.win_size / 2)
        window_coords['left_col'] = coord[1] - math.ceil(self.win_size / 2)

        # TODO: Review this. Is there a better way to do this?
        if window_coords['upper_row'] < 0:
            window_coords['upper_row'] = 0
            window_coords['lower_row'] = self.win_size
        if window_coords['left_col'] < 0:
            window_coords['left_col'] = 0
            window_coords['right_col'] = self.win_size
        if window_coords['lower_row'] > self.labeled_img.shape[0]:
            window_coords['lower_row'] = self.labeled_img.shape[0]
            window_coords['upper_row'] = window_coords['lower_row'] - self.win_size
        if window_coords['right_col'] > self.labeled_img.shape[1]:
            window_coords['right_col'] = self.labeled_img.shape[1]
            window_coords['left_col'] = window_coords['right_col'] - self.win_size

        return window_coords

    def extract_windows(self, coord):
        window = self.compute_window_coords(coord)
        sample_img = self.ref_img[window['upper_row']:window['lower_row'], window['left_col']:window['right_col']]
        sample_label = self.labeled_img[window['upper_row']:window['lower_row'], window['left_col']:window['right_col']]

        return sample_img, sample_label, window

    def generate_chips(self):
        self.compute_indexes()
        samples_img, samples_label, windows = [np.asarray(a) for a in zip(*map(self.extract_windows, self.ij_samples))]
        return {'chips': samples_img,
                'labels': samples_label,
                'coords': windows}

    # def extract_windows(self, coord):
    # samples_img = []
    # samples_labels = []
    # count = 0
    # for coord in self.ij_samples:
    #     window = self.compute_window_coords(coord)
    #     sampleImg = self.ref_img[window['upper_row']:window['lower_row'], window['left_col']:window['right_col']]
    #     sampleLabel = self.labeled_img[window['upper_row']:window['lower_row'], window['left_col']:window['right_col']]
    #
    #     while np.count_nonzero(sampleLabel.mask) != 0:
    #         indice = np.random.choice(np.arange(len(self.sample_candidates)), 1, replace=False)
    #         coord = self.sample_candidates[indice][0]
    #         self.ij_samples[count] = coord
    #         window = self.compute_window_coords(coord)
    #         sampleImg = self.ref_img[window['upper_row']:window['lower_row'], window['left_col']:window['right_col']]
    #         sampleLabel = self.labeled_img[window['upper_row']:window['lower_row'],
    #                       window['left_col']:window['right_col']]
    #
    #     self.samples_img.append(sampleImg)
    #     self.samples_labels.append(sampleLabel)
    #     count = count + 1
    #
    # self.generate_windows_geo_coords()