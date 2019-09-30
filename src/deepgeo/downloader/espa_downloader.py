import folium
import getpass
import geopandas as gpd
import json
import numpy as np
import pandas as pd
import requests


class EspaDownloader(object):
    def __init__(self):
        self.ls_grid = 'https://prd-wret.s3-us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip'
        self.espa_catalog = 'https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_8_C1.csv.gz'

    def espa_api(self, endpoint, verb='get', body=None, uauth=None):
        """ Suggested simple way to interact with the ESPA JSON REST API """
        host = 'https://espa.cr.usgs.gov/api/v1/'
        auth_tup = uauth if uauth else (self.username, self.password)
        response = getattr(requests, verb)(host + endpoint, auth=auth_tup, json=body)
        print('{} {}'.format(response.status_code, response.reason))
        data = response.json()
        if isinstance(data, dict):
            messages = data.pop("messages", None)
            if messages:
                print(json.dumps(messages, indent=4))
        try:
            response.raise_for_status()
        except Exception as e:
            print(e)
            return None
        else:
            return data

    def authenticate(self, username=None, password=None):
        if username is not None:
            self.username = username
        else:
            self.username = input('earthexplorer user name:')

        if password is not None:
            self.password = password
        else:
            self.password = getpass.getpass()

    def get_intersections(self, roi):
        self.roi = gpd.read_file(roi)
        self.wrs_intersection = self.ls_grid[self.ls_grid.intersects(self.roi.geometry[0])]
        self.paths, self.rows = self.wrs_intersection['PATH'].values, self.wrs_intersection['ROW'].values

    def plot_intersections(self):
        # Get the center of the map
        xy = np.asarray(self.roi.centroid[0].xy).squeeze()
        center = list(xy[::-1])

        # Select a zoomsurface arc mouse
        zoom = 6

        # Create the most basic OSM folium map
        m = folium.Map(location=center, zoom_start=zoom, control_scale=True)

        # Add the bounds GeoDataFrame in red
        m.add_child(folium.GeoJson(self.roi.__geo_interface__, name='Area of Study',
                                   style_function=lambda x: {'color': 'red', 'alpha': 0}))

        # Iterate through each Polygon of paths and rows intersecting the area
        for i, row in self.wrs_intersection.iterrows():
            # Create a string for the name containing the path and row of this Polygon
            name = 'path: %03d, row: %03d' % (row.PATH, row.ROW)
            # Create the folium geometry of this Polygon
            g = folium.GeoJson(row.geometry.__geo_interface__, name=name)
            # Add a folium Popup object with the name string
            g.add_child(folium.Popup(name))
            # Add the object to the map
            g.add_to(m)

        folium.LayerControl().add_to(m)
        # m.save('./images/10/wrs.html')
        m  # TODO: Check this
