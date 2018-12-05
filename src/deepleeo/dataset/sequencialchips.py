# TODO: Review this method.
# TODO: Try to plot the chips in the already implemented approach (file train_fcn.ipynb). See Marciano approach
# To plot the chips
# def generate_sequential_chips(img_path, nodata_value, chip_size, pad_size, offset_list=[(0, 0)], \
#                               rotate=False, flip=False, remove_chips_wnodata=True,
#                               chips_data_np=None, chips_expect_np=None):
#     index = 0
#     chip_data_list = []
#     chip_expect_list = []
#     input_img_ds = gdal.Open(img_path)
#
#     for x_offset_percent, y_offset_percent in offset_list:
#         x_offset = int(chip_size * (x_offset_percent / 100.0))
#         y_offset = int(chip_size * (y_offset_percent / 100.0))
#
#         input_positions = get_predict_positions(input_img_ds.RasterXSize, input_img_ds.RasterYSize, \
#                                                 chip_size, pad_size, x_offset, y_offset)
#
#         for input_position in input_positions:
#             chip_data, _ = get_predict_data(input_img_ds, input_position, pad_size)
#
#             chip_data, chip_expect = split_data(chip_data, pad_size)
#             xsize, ysize, _ = chip_expect.shape
#
#             if (chip_size == xsize and chip_size == ysize) and (
#                     not remove_chips_wnodata or float(np.min(chip_expect)) != float(nodata_value)):
#                 if (float(np.max(chip_expect)) > float(0.0)):  # Only include chips with some object
#                     chip_expect[chip_expect != 1] = 0  # convert all other class to pixel == 0
#                     # chip_data_aux = chip_augmentation(chip_data, rotate, flip)
#                     # chip_expect_aux = chip_augmentation(chip_expect, rotate, flip)
#                     chip_data_aux = chip_data  #TODO: Remove this line and the following
#                     chip_expect_aux = chip_expect
#                     nchips = len(chip_data_aux)
#
#                     if (chips_data_np is not None):
#                         chips_data_np[index:index + nchips, :, :, :] = np.stack(chip_data_aux)
#                         chips_expect_np[index:index + nchips, :, :, :] = np.stack(chip_expect_aux)
#                     else:
#                         chip_data_list = chip_data_list + chip_data_aux
#                         chip_expect_list = chip_expect_list + chip_expect_aux
#
#                     index = index + nchips
#
#     return chip_data_list, chip_expect_list
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import gdal
import osr
import skimage
from skimage import exposure

#TODO: How to put this as an strategy to the chipGenerator?
def generate_sequential_chips(img_array, chip_size=286, overlap=(0, 0), remove_no_data=True):
    x_size, y_size, nbands = img_array.shape
    # print("Raster size: (", x_size, ", ", y_size, ", ", nbands, ")")

    struct = {"chips": [], "coords": []}
    for y_start in range(0, y_size, chip_size - overlap[0]):
        y_end = y_start + chip_size

        if y_end > y_size:
            y_end = y_size
            y_start = y_end - chip_size
            # print("XSTART = ", x_start, "XEND = ", x_end)

        for x_start in range(0, x_size, chip_size - overlap[1]):
            x_end = x_start + chip_size

            if x_end > x_size:
                x_end = x_size
                x_start = x_end - chip_size
                # print("YSTART = ", y_start, "YEND = ", y_end)

            chip_array = img_array[x_start:x_end, y_start:y_end, :]

            struct["chips"].append(chip_array)
            struct["coords"].append({"x_start": x_start, "x_end": x_end, "y_start": y_start, "y_end": y_end})

    return struct

def plot_chips(chips, raster_array, bands=[1, 2, 3], contrast=False):
    fig,ax = plt.subplots(1, figsize=(12, 12))

    # Display the image
    raster_img = skimage.img_as_ubyte(raster_array)
    if contrast:
        for band in bands:
            p2, p98 = np.percentile(raster_img[:, :, band], (2, 98))
            raster_img[:, :, band] = exposure.rescale_intensity(raster_img[:, :, band], in_range=(p2, p98))

    if len(bands) == 3:
        ax.imshow(raster_img[:, :, bands])
    else:
        ax.imshow(raster_img[:, :, bands[0]])

    for coord in chips["coords"]:
        width = coord["y_end"] - coord["y_start"]
        height = coord["x_end"] - coord["x_start"]
        rect = patches.Rectangle((coord["y_start"], coord["x_start"]), width, height,
                                  edgecolor="blue", facecolor="none")
        ax.add_patch(rect)


def write_chips(output_path, base_raster, pred_struct, output_format="GTiff", dataType=gdal.GDT_UInt16):
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

    for idx in range(1, len(pred_struct["chips"])):
        chip = pred_struct["chips"][idx]
        chip = np.squeeze(chip)
        out_band.WriteArray(chip, pred_struct["coords"][idx]["y_start"], pred_struct["coords"][idx]["x_start"])

    out_band.FlushCache()