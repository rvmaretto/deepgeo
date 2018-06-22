import numpy as np
import numpy.ma as ma
import gdal
import ogr

def load_image(filename):
    #print("--- Loading Raster ---")
    rgb_dataset = gdal.Open(filename)
    numBands = rgb_dataset.RasterCount
    #print("   NUM Bands: ", numBands)
    img_rgb = rgb_dataset.ReadAsArray()
    # print("   Shape before roll: ", img_rgb.shape)
    img_rgb= np.rollaxis(img_rgb, 0, start=3)
    # print("   Shape after roll: ", img_rgb.shape)
    img_rgb = ma.array(img_rgb[:,:,:numBands])#, mask=mask)
    # convert it to float because it's easier for us after
    img_rgb = ma.array(img_rgb.astype(np.float32) / 255.0)#, mask=mask)
    return img_rgb

def load_vector_layer(filename):
    vector_ds = ogr.Open(filename)
    layer = vector_ds.GetLayer()
    return layer
