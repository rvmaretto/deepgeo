import math
import os
import sys
import numpy as np
import geopandas
import rasterio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common.utils as utils


class CentroidsChipGenerator(object):
    mandatory_params = ['raster_array', 'labels_array', 'win_size', 'shp_path', 'labels_tif']
    default_params = {'class_of_interest': None,
                      'remove_no_data': None}

    def __init__(self, params):
        params = utils.check_dict_parameters(params, self.mandatory_params, self.default_params)
        self.ref_img = params['raster_array']
        self.labeled_img = params['labels_array']
        self.win_size = params['win_size']
        # self.class_of_interest = params['class_of_interest']
        # self.quantity = params['quantity']
        # self.class_names = params['class_names']
        self.shp_path = params['shp_path']
        self.labels_tif = params['labels_tif']

    def compute_indexes(self):
        # opens the shapefile and get points coordinates
        shp = geopandas.read_file(self.shp_path)
        coordinates = np.zeros([len(shp),2], dtype=np.float64)
        coordinates[:,0] = shp['geometry'].x
        coordinates[:,1] = shp['geometry'].y

        # open labels raster to get its diemnsions and bounding box
        labels = rasterio.open(self.labels_tif)
        box = labels.bounds
        delta_x = box[2]-box[0] 
        delta_y = box[3]-box[1]

        # converts the points coordinates to image coordinates
        samples = np.zeros([len(coordinates), 3], dtype=np.int64)
        samples[:,0] = (np.asarray((coordinates[:,1]-box[1])*labels.height/delta_y, dtype=np.int64)*-1)+labels.height
        samples[:,1] = np.asarray((coordinates[:,0]-box[0])*labels.width/delta_x, dtype=np.int64)
        
        # checks if the chips to be created are completely within the images
        check1 = samples[:,1]+(self.win_size/2)>=labels.width
        check2 = samples[:,1]-(self.win_size/2)<0
        check3 = samples[:,0]+(self.win_size/2)>=labels.height
        check4 = samples[:,0]-(self.win_size/2)<0

        check = check1+check2+check3+check4

        print('%d valid chips found, %d invalid chips (out of bounds).' % (len(check)-np.sum(check), np.sum(check)))
        print('Proceeding with valid chips...')

        # final result
        self.ij_samples = samples[np.invert(check)]

    def compute_window_coords(self, coord):
        window_coords = {}
        window_coords['upper_row'] = coord[0] - math.floor(self.win_size / 2)
        window_coords['lower_row'] = coord[0] + math.ceil(self.win_size / 2)
        window_coords['right_col'] = coord[1] + math.floor(self.win_size / 2)
        window_coords['left_col']  = coord[1] - math.ceil(self.win_size / 2)

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