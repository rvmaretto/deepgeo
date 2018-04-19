import numpy as np
import numpy.ma as ma
import gdal
import ogr

def load_image(filename):
    print("--- Loading Raster ---")
    rgb_dataset = gdal.Open(filename)
    numBands = rgb_dataset.RasterCount
    print("   NUM Bands: ", numBands)
    img_rgb = rgb_dataset.ReadAsArray()
    print("   Shape before roll: ", img_rgb.shape)
    img_rgb= np.rollaxis(img_rgb, 0, start=3)
    print("   Shape after roll: ", img_rgb.shape)
    # mask = img_rgb[:,:,(numBands - 1)] == 0
    # mask = np.stack([mask, mask, mask])
    img_rgb = ma.array(img_rgb[:,:,:numBands])#, mask=mask)
    # convert it to float because it's easier for us after
    img_rgb = ma.array(img_rgb.astype(np.float32) / 255.0)#, mask=mask)
    return img_rgb

# TODO: Verify here. This method will just load the shape file and the class names. Or it will rasterize and generate a
# numpy array? It should be called load_... or rasterize_...? Verify the best way to reimplement the file "rasterize_truth"
def rasterize_vector_file(filename, label_column="class"):
    print("--- Loading Shape File ---")
    vector_ds = ogr.Open(filename)
    layer = vector_ds.GetLayer()
    print("--- Collecting Labels ---")
    layer.ResetReading()
    unique_labels = set()
    while True:
        feature = layer.GetNextFeature()
        if feature is None:
            break
        name = feature.GetField(label_column)
        # unique_labels.add(np.string_(name))  # TODO: Verify why I needed this. Is it needed to put in a npz file?
        unique_labels.add(name)

    print("Labels loaded:")
    class_names = []
    for name in sorted(unique_labels):
        # print("\t\"", str(name, encoding="UTF-8"), "\"")  # TODO: The same as the last TODO
        print("\t\"", name, "\"")
        class_names.append(name)

    return layer, class_names