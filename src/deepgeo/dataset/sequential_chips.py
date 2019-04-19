import gdal
import math
import numpy as np
import osr
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import common.utils as utils


class SequentialChipGenerator(object):
    mandatory_params = ['raster_array', 'win_size']
    default_params = {'labels_array': None,
                      'overlap': (0, 0),
                      'class_of_interest': None,
                      'perc_discard_nd': None,  # TODO: Allow here to define the threshold percentage of no_data pixels to remove the chip
                      'no_data': 0}

    def __init__(self, params):
        params = utils.check_dict_parameters(params, self.mandatory_params, self.default_params)
        self.img_array = params['raster_array']
        self.labeled_array = params['labels_array']
        self.win_size = params['win_size']
        self.overlap = params['overlap']
        self.class_of_interest = params['class_of_interest']
        self.perc_discard_nd = params['perc_discard_nd']
        self.no_data = params['no_data']

    def compute_indexes(self):
        row_size, col_size, nbands = self.img_array.shape
        if self.labeled_array is not None:
            lbl_row_size, lbl_col_size, _ = self.labeled_array.shape

            if row_size != lbl_row_size or col_size != lbl_col_size:
                raise AssertionError('Raster and labels have different sizes (rows and columns)!')

        self.win_coords = []
        for col_start in range(0, col_size, self.win_size - self.overlap[0]):
            col_end = col_start + self.win_size

            if col_end > col_size:
                col_end = col_size
                col_start = col_end - self.win_size
                # print('ROW_START = ', row_start, 'ROW_END = ', row_end)

            for row_start in range(0, row_size, self.win_size - self.overlap[1]):
                row_end = row_start + self.win_size

                if row_end > row_size:
                    row_end = row_size
                    row_start = row_end - self.win_size
                    # print('COL_START = ', col_start, 'COL_END = ', col_end)

                self.win_coords.append({'upper_row': row_start, 'lower_row': row_end, 'left_col': col_start, 'right_col': col_end})

    def extract_windows(self, win_coord):
        sample_img = self.img_array[win_coord['upper_row']:win_coord['lower_row'],
                                    win_coord['left_col']:win_coord['right_col']]
        if self.labeled_array is not None:
            sample_lbl = self.labeled_array[win_coord['upper_row']:win_coord['lower_row'],
                                               win_coord['left_col']:win_coord['right_col']]
        else:
            sample_lbl = None
        return sample_img, sample_lbl, win_coord

    def generate_chips(self):
        self.compute_indexes()
        samples_img, samples_labels, windows = [np.asarray(a) for a in zip(*map(self.extract_windows, self.win_coords))]

        if self.labeled_array is not None:
            # if self.perc_discard_nd is not None:
            #     positions_remove = []
            #     total_pixels = math.pow(self.win_size, 2)
            #     for i in range(0, len(samples_labels)):
            #         perc_no_data = np.count_nonzero(samples_labels[i].mask == 0) / total_pixels
            #         if perc_no_data > self.perc_discard_nd:
            #             positions_remove.append(i)
            return {'chips': samples_img,
                    'labels': samples_labels,
                    'win_coords': windows}
        else:
            return {'chips': samples_img,
                    'win_coords': windows}



# def generate_sequential_chips(img_array, chip_size=286, overlap=(0, 0), remove_no_data=True):
#     x_size, y_size, nbands = img_array.shape
#     # print('Raster size: (', x_size, ', ', y_size, ', ', nbands, ')')
#
#     struct = {'chips': [], 'coords': []}
#     for y_start in range(0, y_size, chip_size - overlap[0]):
#         y_end = y_start + chip_size
#
#         if y_end > y_size:
#             y_end = y_size
#             y_start = y_end - chip_size
#             # print('XSTART = ', x_start, 'XEND = ', x_end)
#
#         for x_start in range(0, x_size, chip_size - overlap[1]):
#             x_end = x_start + chip_size
#
#             if x_end > x_size:
#                 x_end = x_size
#                 x_start = x_end - chip_size
#                 # print('YSTART = ', y_start, 'YEND = ', y_end)
#
#             chip_array = img_array[x_start:x_end, y_start:y_end, :]
#
#             struct['chips'].append(chip_array)
#             struct['coords'].append({'x_start': x_start, 'x_end': x_end, 'y_start': y_start, 'y_end': y_end})
#
#     return struct


def write_chips(output_path, base_raster, pred_struct, output_format='GTiff', dataType=gdal.GDT_UInt16):
    driver = gdal.GetDriverByName(output_format)
    base_ds = gdal.Open(base_raster)

    x_start, pixel_width, _, y_start, _, pixel_height = base_ds.GetGeoTransform()
    x_size = base_ds.RasterXSize
    y_size = base_ds.RasterYSize

    srs = osr.SpatialReference()
    srs.ImportFromWkt(base_ds.GetProjectionRef())

    out_ds = driver.Create(output_path, x_size, y_size, 1, dataType)
    out_ds.SetGeoTransform((x_start, pixel_width, 0, y_start, 0, pixel_height))
    out_ds.SetProjection(srs.ExportToWkt())
    out_band = out_ds.GetRasterBand(1)

    for idx in range(1, len(pred_struct['chips'])):
        chip = pred_struct['chips'][idx]
        chip = np.squeeze(chip)
        out_band.WriteArray(chip, pred_struct['coords'][idx]['y_start'], pred_struct['coords'][idx]['x_start'])

    out_band.FlushCache()