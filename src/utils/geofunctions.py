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

# TODO: Verify why it is breaking in this method. Solution is probably in the metnod "rasterizePolygons"
def rasterize_layer(vector_layer, model_raster, class_column="class", nodata_val=255):
    print("--- Rasterizing Vector Layer ---")
    mem_drv = gdal.GetDriverByName('MEM')
    mem_raster = mem_drv.Create(
        '',
        model_raster.RasterXSize,
        model_raster.RasterYSize,
        1,
        gdal.GDT_Int16
    )
    mem_raster.SetProjection(model_raster.GetProjection())
    mem_raster.SetGeoTransform(model_raster.GetGeoTransform())
    mem_band = mem_raster.GetRasterBand(1)
    mem_band.Fill(nodata_val)
    mem_band.SetNoDataValue(nodata_val)

    err = gdal.RasterizeLayer(
        mem_raster,
        [1],
        vector_layer,
        None,
        None,
        [1],
        options=[class_column]
    )

    assert(err == gdal.CE_None)
    return mem_raster.ReadAsArray()