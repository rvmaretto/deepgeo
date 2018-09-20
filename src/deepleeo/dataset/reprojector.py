import os
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

# The following code is based in the code available in the following
# link: https://gis.stackexchange.com/questions/264618/reprojecting-and-saving-shapefile-in-gdal
def reproj_shape_to_raster(path_in_shp, path_raster, path_out_shp):
    # Opens the base raster
    base_tiff = gdal.Open(path_raster)

    #shapefile with the from projection
    driver = ogr.GetDriverByName("ESRI Shapefile")
    in_ds =   driver.Open(path_in_shp, 1)
    in_layer = in_ds.GetLayer()

    #set spatial reference and transformation
    sourceprj = in_layer.GetSpatialRef()
    targetprj = osr.SpatialReference(wkt = base_tiff.GetProjection())
    # create the CoordinateTransformation
    coord_transform = osr.CoordinateTransformation(sourceprj, targetprj)

    # create the output layer
    out_driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(path_out_shp):
        out_driver.DeleteDataSource(path_out_shp)
    out_ds = out_driver.CreateDataSource(path_out_shp)
    out_layer = out_ds.CreateLayer('', targetprj, ogr.wkbPolygon)

    # add fields
    in_layerDefn = in_layer.GetLayerDefn()
    for i in range(0, in_layerDefn.GetFieldCount()):
        field_defn = in_layerDefn.GetFieldDefn(i)
        out_layer.CreateField(field_defn)

    # get the output layer's feature definition
    out_layer_defn = out_layer.GetLayerDefn()

    # loop through the input features
    in_feature = in_layer.GetNextFeature()
    while in_feature:
        # get the input geometry
        geom = in_feature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coord_transform)
        geom = ogr.CreateGeometryFromWkb(geom.ExportToWkb())
        # create a new feature
        out_feature = ogr.Feature(out_layer_defn)
        # set the geometry and attribute
        out_feature.SetGeometry(geom)
        for i in range(0, out_layer_defn.GetFieldCount()):
            out_feature.SetField(out_layer_defn.GetFieldDefn(i).GetNameRef(), in_feature.GetField(i))
        # add the feature to the shapefile
        out_layer.CreateFeature(out_feature)
        # dereference the features and get the next input feature
        out_feature = None
        in_feature = in_layer.GetNextFeature()

    out_ds = None
    in_ds = None