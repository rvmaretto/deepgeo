import numpy as np
import gdal
import ogr


def load_image(filepath, no_data=None):
    img_ds = gdal.Open(filepath)

    if (no_data is None):
        no_data = 0

    img = None
    for i in range(1, img_ds.RasterCount + 1):
        band = img_ds.GetRasterBand(i)
        band_arr = band.ReadAsArray()
        band_arr = np.ma.masked_array(band_arr, band_arr == no_data)
        if (img is None):
            img = band_arr
        else:
            img = np.ma.dstack((img, band_arr))

    return img

def load_vector_layer(filename):
    vector_ds = ogr.Open(filename)
    layer = vector_ds.GetLayer()
    return layer
