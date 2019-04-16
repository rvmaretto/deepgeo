import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import gdal
import osr
import skimage
from skimage import exposure


class SequentialChipGenerator(object):
    def __init__(self, params):
        self.ref_img = params['raster_array']
        self.labeled_img = params['shp_input']
        self.win_size = params['win_size']
        self.class_of_interest = params['class_of_interest']
        self.overlap = params['overlap']
        self.remove_no_data = params['remove_no_data']  # TODO: Allow here to define the threshold percentage of no_data to remove the chip


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


def plot_chips(chips, raster_array, bands=[1, 2, 3], contrast=False, chipscolor="blue"):
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

    plt.axis('off')

    for coord in chips["coords"]:
        width = coord["y_end"] - coord["y_start"]
        height = coord["x_end"] - coord["x_start"]
        rect = patches.Rectangle((coord["y_start"], coord["x_start"]), width, height,
                                  edgecolor=chipscolor, facecolor="none")
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