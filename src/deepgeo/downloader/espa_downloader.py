import folium
import getpass
import geopandas as gpd
import json
import numpy as np
import pandas as pd
import requests


class EspaDownloader(object):
    catalog_files = {'Landsat_8_OLI': 'https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_8_C1.csv.gz',
                     'Landsat_7_ETM+': 'https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_ETM_C1.csv.gz',
                     'Landsat_4_5_TM': 'https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_TM_C1.csv.gz',
                     'Landsat_1_5_MSS': 'https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_MSS_C1.csv.gz'}

    wrs_files = {'wrs_2': 'https://prd-wret.s3-us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip',
                 'wrs_1': 'https://prd-wret.s3-us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS1_descending_0.zip'}

    def __init__(self, sensor='Landsat_8_OLI'):
        if sensor in ['Landsat_8_OLI', 'Landsat_7_ETM+', 'Landsat_4_5_TM']:
            self.ls_grid = gpd.GeoDataFrame.from_file(self.wrs_files['wrs_2'])
        elif sensor in ['Landsat_1_5_MSS']:
            self.ls_grid = gpd.GeoDataFrame.from_file(self.wrs_files['wrs_1'])

        self.espa_catalog_file = self.catalog_files[sensor]
        self.espa_scenes = pd.read_csv(self.espa_catalog_file)

    def __call_espa_api(self, endpoint, verb='get', body=None, uauth=None):
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

        print('Authenticating on /api/v1/user')
        self.__call_espa_api('user', uauth=(self.username, self.password))

    def get_intersections(self, roi):
        self.roi = gpd.read_file(roi)
        self.wrs_intersection = self.ls_grid[self.ls_grid.intersects(self.roi.geometry[0])]
        self.paths, self.rows = self.wrs_intersection['PATH'].values, self.wrs_intersection['ROW'].values
        return self.paths, self.rows

    def plot_intersections(self, path_html=None):
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
        if path_html is not None:
            m.save('path_html')
        return m

    def consult_dates(self, start_date, end_date=None, min_cloud_cover=99, strategy='min_cloud_cover'):
        # Empty list to add the images
        bulk_list = []
        ids_list = []
        not_found_list = []

        # Iterate through paths and rows
        for path, row in zip(self.paths, self.rows):

            print('Path:', path, 'Row:', row)

            if end_date is None:
                end_date = start_date

            if isinstance(start_date, dict):
                st_date = start_date[{'%03d_%03d'.format(path, row)}]
                ed_date = end_date[{'%03d_%03d'.format(path, row)}]
            else:
                st_date = start_date
                ed_date = end_date

            if isinstance(st_date, str):
                st_date = start_date.split('-')
                if len(st_date[0]) < 4:
                    st_date = st_date.reverse()
                st_date = '-'.join(st_date)

                ed_date = start_date.split('-')
                if len(ed_date[0]) < 4:
                    ed_date = ed_date.reverse()
                ed_date = '-'.join(ed_date)

            # Filter the Landsat ESPA table for images matching path, row, cloudcover, processing state, and dates.
            scenes = self.espa_scenes[(self.espa_scenes.path == path) & (self.espa_scenes.row == row) &
                                      (self.espa_scenes.cloudCover <= min_cloud_cover) &
                                      (self.espa_scenes.acquisitionDate >= st_date) &
                                      (self.espa_scenes.acquisitionDate <= ed_date) &
                                      (~self.espa_scenes.LANDSAT_PRODUCT_ID.str.contains('_RT'))]

            print('  -> Found {} images\n'.format(len(scenes)))

            # If any scenes exists, select the one that have the minimum cloudCover.
            if len(scenes):
                if strategy == 'min_cloud_cover':
                    scene = scenes.sort_values('cloudCover').iloc[0]

                    # Add the selected scene to the bulk download list.
                    bulk_list.append(scene)
                    ids_list.append(scene.LANDSAT_PRODUCT_ID)
                elif strategy == 'all':
                    for scene in scenes.itertuples():
                        bulk_list.append(scene)
                        ids_list.append(scene.LANDSAT_PRODUCT_ID)
            else:
                not_found_list.append('%03d/%03d' % (path, row))

        return pd.DataFrame(bulk_list), ids_list, not_found_list

    def get_available_projections(self):
        print('Getting projections from /api/v1/projections')
        projs = self.__call_espa_api('projections')
        # print(json.dumps(projs.keys()))
        print(projs.keys())
