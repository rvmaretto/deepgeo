from osgeo import gdal
import os
import subprocess
# import rasterio
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

    for i in range(1, len(files) + 1):
        ds = gdal.Open(files[i-1])
        outds.GetRasterBand(i).SetNoDataValue(ds.GetRasterBand(1).GetNoDataValue())
        outds.GetRasterBand(i).SetMetadata(ds.GetRasterBand(1).GetMetadata())
        outds.GetRasterBand(i).SetDescription(band_names[i - 1])

    ds = gdal.Open(files[0])
    outds.SetGeoTransform(ds.GetGeoTransform())
    outds.SetProjection(ds.GetProjection())
    gdal.Translate(output_img, outds)

#TODO: Verify band names and other metadata if they are Ok.
def mosaic_images(files, output_file, band_names=None):
    if not isinstance(files, list):
        raise TypeError("Argument \"files\" must be a list.")
    
    if len(files) < 2:
        raise Exception("You must provide at least two .tiff files.")

    print("Mosaicing images:")
    for file_name in files:
        print(" >", file_name)

    # in_ds = gdal.Open(files[0])
    # no_data_value = in_ds.GetRasterBand(1).GetNoDataValue()
    arguments = ['gdal_merge.py', '-o', output_file, '-q', '-v']

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