from osgeo import gdal
from osgeo import ogr
import os
import subprocess
import fiona
import rasterio
import rasterio.mask
# from rasterio.merge import merge

def stack_bands(files, output_img, band_names=None):#, no_data=-9999, format="GTiff", dtype=gdal.GDT_Int16):
    if not isinstance(files, list):
        raise TypeError("Argument \"files\" must be a list.")
    
    if len(files) < 2:
        raise Exception("You must provide at least two .tiff files.")

    if band_names is None:
        band_names = []
        for i in range(0, len(files )): #TODO: Verify this
            ds = gdal.Open(files[i])
            name = ds.GetRasterBand(1).GetDescription()
            if name == '':
                name = "band_" + str(i)
            band_names.append(name)

    if os.path.exists(output_img):
        os.remove(output_img)

    outvrt = '/vsimem/stacked.vrt' #/vsimem is special in-memory virtual "directory"
    outds = gdal.BuildVRT(outvrt, files, separate=True)

    count = 1
    for i in range(0, len(files)):
        ds = gdal.Open(files[i])
        for j in range(1, ds.RasterCount + 1):
            outds.GetRasterBand(count).SetNoDataValue(ds.GetRasterBand(j).GetNoDataValue())
            outds.GetRasterBand(count).SetMetadata(ds.GetRasterBand(j).GetMetadata())
            outds.GetRasterBand(count).SetDescription(band_names[count - 1])
            count = count + 1

    ds = gdal.Open(files[0])
    outds.SetGeoTransform(ds.GetGeoTransform())
    outds.SetProjection(ds.GetProjection())
    gdal.Translate(output_img, outds, options=['COMPRESS=LZW'])

def clip_img_by_extent(img_file, reference_shp, output_img):
    if os.path.exists(output_img):
        os.remove(output_img)

    # Get a Layer's Extent
    vector_driver = ogr.GetDriverByName("ESRI Shapefile")
    vector_ds = vector_driver.Open(reference_shp, 0) # 0=Read-only, 1=Read-Write
    vector_layer = vector_ds.GetLayer()
    min_x, max_x, min_y, max_y = vector_layer.GetExtent()

    raster_to_clip = gdal.Open(img_file)
    projection = raster_to_clip.GetProjectionRef()

    gdal.Warp(output_img, raster_to_clip, format="GTiff",
                       outputBounds=[min_x, min_y, max_x, max_y],
                       dstSRS=projection, resampleAlg=gdal.GRA_NearestNeighbour,
                       options=['COMPRESS=LZW'])

    vector_ds.Destroy()
    raster_to_clip = None

def stack_temporal_images(files, output_img, band_names=None):
    if not isinstance(files, list):
        raise TypeError("Argument \"files\" must be a list.")

    if len(files) < 2:
        raise Exception("You must provide at least two .tiff files.")

    if os.path.exists(output_img):
        os.remove(output_img)

    num_bands = 0
    for file in files:
        inputds = gdal.Open(file)
        num_bands = num_bands + inputds.RasterCount
        inputds = None

    drv = gdal.GetDriverByName("GTiff")
    inputds = gdal.Open(files[0])
    out_xSize = inputds.RasterXSize
    out_ySize = inputds.RasterYSize
    datatype = inputds.GetRasterBand(1).DataType

    outds = drv.Create(output_img, out_xSize, out_ySize, num_bands, datatype, options=['COMPRESS=LZW'])

    count = 1
    for i in range(0, len(files)):
        inputds = gdal.Open(files[i])

        for j in range(1, inputds.RasterCount + 1):
            band = inputds.GetRasterBand(j)
            band_arr = band.ReadAsArray()
            outds.GetRasterBand(count).WriteArray(band_arr)
            outds.GetRasterBand(count).SetNoDataValue(inputds.GetRasterBand(j).GetNoDataValue())
            outds.GetRasterBand(count).SetMetadata(inputds.GetRasterBand(j).GetMetadata())
            outds.GetRasterBand(count).SetDescription(band_names[count - 1])
            count = count + 1
        inputds = None

    ds = gdal.Open(files[0])
    outds.SetGeoTransform(ds.GetGeoTransform())
    outds.SetProjection(ds.GetProjection())
    outds = None

def mosaic_images(files, output_file, band_names=None):
    if not isinstance(files, list):
        raise TypeError("Argument \"files\" must be a list.")
    
    if len(files) < 2:
        raise Exception("You must provide at least two .tiff files.")

    print("Mosaicing images:")
    for file_name in files:
        print(" >", file_name)

    arguments = ['gdal_merge.py', '-o', output_file, '-co','COMPRESS=LZW', '-q', '-v']

    for file_name in files:
        arguments.append(file_name)

    if os.path.exists(output_file):
        os.remove(output_file)

    ps = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = ps.communicate()
    for line in output[0].splitlines():
        print(str(line, 'utf-8'))

    for line in output[1].splitlines():
        print(str(line, 'utf-8'))

    print("Setting Metadata...")
    input_ds = gdal.Open(files[0])
    out_ds = gdal.Open(output_file, gdal.GA_Update)

    if band_names is None:
        band_names = []
        for band in range(1, input_ds.RasterCount + 1):
            name = input_ds.GetRasterBand(band).GetDescription()
            if name == '':
                name = "band_" + str(band)
            band_names.append(name)

    for band in range(1, input_ds.RasterCount + 1):
        out_band = out_ds.GetRasterBand(band)
        out_band.SetNoDataValue(input_ds.GetRasterBand(band).GetNoDataValue())
        out_band.SetDescription(band_names[band - 1])
        out_band = None

    out_ds = None
    input_ds = None

def clip_by_aggregated_polygons(in_raster_path, shape_file, output_path, band_names=None, no_data=None):
    if band_names is None:
        band_names = []
        ds = gdal.Open(in_raster_path)
        for i in range(1, ds.RasterCount):
            name = ds.GetRasterBand(i).GetDescription()
            if name == '':
                name = "band_" + str(i - 1)
            band_names.append(name)

    if no_data is None:
        ds = gdal.Open(in_raster_path)
        no_data = ds.GetRasterBand(1).GetNoDataValue()

    with fiona.open(shape_file, "r") as shapefile:
        features = [feature["geometry"] for feature in shapefile]

    with rasterio.open(in_raster_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, features, crop=True, nodata=no_data)
        out_meta = src.meta.copy()

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    if os.path.exists(output_path):
        os.remove(output_path)

    with rasterio.open(output_path, "w", **out_meta) as dest:
        for id, name in enumerate(band_names):
            dest.set_band_description(id + 1, name)
        dest.write(out_image)