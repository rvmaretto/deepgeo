import folium
import getpass
import geopandas as gpd
import json
import os
import requests
import sys
import wget
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, '../src')
import deepgeo.common.filesystem as fs

class EspaDownloader(object):
    catalog_files = {'Landsat_8_OLI': 'https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_8_C1.csv.gz',
                     'Landsat_7_ETM+': 'https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_ETM_C1.csv.gz',
                     'Landsat_4_5_TM': 'https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_TM_C1.csv.gz',
                     'Landsat_1_5_MSS': 'https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_MSS_C1.csv.gz'}

    wrs_files = {'wrs_2': 'https://prd-wret.s3-us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip',
                 'wrs_1': 'https://prd-wret.s3-us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS1_descending_0.zip'}

    projections = {'lonlat': {'lonlat': None}}

    def __init__(self, sensor='Landsat_8_OLI'):
        if sensor in ['Landsat_8_OLI', 'Landsat_7_ETM+', 'Landsat_4_5_TM']:
            self.ls_grid = gpd.GeoDataFrame.from_file(self.wrs_files['wrs_2'])
        elif sensor in ['Landsat_1_5_MSS']:
            self.ls_grid = gpd.GeoDataFrame.from_file(self.wrs_files['wrs_1'])

        self.espa_catalog_file = self.catalog_files[sensor]
        self.espa_scenes = pd.read_csv(self.espa_catalog_file, compression='gzip', parse_dates=['acquisitionDate'])

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
        if isinstance(roi, str):
            self.roi = gpd.read_file(roi)
        else:
            self.roi = roi

        self.wrs_intersection = self.ls_grid[self.ls_grid.intersects(self.roi.geometry[0])]
        self.paths, self.rows = self.wrs_intersection['PATH'].values, self.wrs_intersection['ROW'].values
        return self.paths, self.rows

    def set_paths_rows(self, paths, rows):
        self.paths = paths
        self.rows = rows

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

    def consult_dates(self, start_date, end_date=None, max_cloud_cover=100, strategy='min_cloud_cover'):
        # Empty list to add the images
        bulk_list = []
        self.ids_list = []
        self.not_found_list = []

        # Iterate through paths and rows
        for path, row in zip(self.paths, self.rows):

            if end_date is None:
                end_date = start_date

            if isinstance(start_date, dict):
                st_date = start_date['%03d_%03d' % (path, row)]
                ed_date = end_date['%03d_%03d' % (path, row)]
            else:
                st_date = start_date
                ed_date = end_date

            # if isinstance(st_date, str):
            st_date = st_date.split('-')
            if len(st_date[0]) < 4:
                st_date = st_date.reverse()
            st_date = '-'.join(st_date)
            st_date = datetime.strptime(st_date, '%Y-%m-%d')

            ed_date = ed_date.split('-')
            if len(ed_date[0]) < 4:
                ed_date = ed_date.reverse()
            ed_date = '-'.join(ed_date)
            ed_date = datetime.strptime(ed_date, '%Y-%m-%d')

            print('Path:', path, '- Row:', row, '- Start date: ', st_date, '- End date: ', ed_date)

            # Filter the Landsat ESPA table for images matching path, row, cloudcover, processing state, and dates.
            scenes = self.espa_scenes[(self.espa_scenes.path == path) & (self.espa_scenes.row == row) &
                                      (self.espa_scenes.cloudCover <= max_cloud_cover) &
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
                    self.ids_list.append(scene.LANDSAT_PRODUCT_ID)
                elif strategy == 'all':
                    for scene in scenes.itertuples():
                        bulk_list.append(scene)
                        self.ids_list.append(scene.LANDSAT_PRODUCT_ID)
            else:
                self.not_found_list.append('%03d/%03d' % (path, row))
            self.bulk_list = pd.DataFrame(bulk_list)

        return self.bulk_list, self.ids_list, self.not_found_list

    def get_available_products(self, ids_list):
        print('Getting available products from /api/v1/available-products')
        avail_prods = self.__call_espa_api('available-products', body=dict(inputs=ids_list))
        print(json.dumps(avail_prods, indent=4))

    def get_available_projections(self):
        print('Getting projections from /api/v1/projections')
        self.projs = self.__call_espa_api('projections')
        # print(json.dumps(projs.keys()))
        print(self.projs.keys())
        return self.projs

    def generate_order(self, products, file_format='gtiff', projection=None, verbose=False):
        print('GET /api/v1/available-products')
        self.order = self.__call_espa_api('available-products', body=dict(inputs=self.ids_list))
        # if verbose:
            # print(json.dumps(self.order, indent=4))

        # Replace the available products that was returned with what we want
        for sensor in self.order.keys():
            if isinstance(self.order[sensor], dict) and self.order[sensor].get('inputs'):
                if set(self.ids_list) & set(self.order[sensor]['inputs']):
                    self.order[sensor]['products'] = products
        if 'date_restricted'in self.order:
            del self.order['date_restricted']

        # Add in the rest of the order information
        if projection is not None:
            if isinstance(projection, str):
                self.order['projection'] = self.projections[projection]
            else:
                self.order['projection'] = projection
        self.order['format'] = file_format
        self.order['resampling_method'] = 'cc'
        self.order['note'] = 'DeepGeo Download!!'

        # Notice how it has changed from the original call available-products
        if verbose:
            print(json.dumps(self.order, indent=4))

    def place_order(self):
        # Place the order
        print('POST /api/v1/order')
        resp = self.__call_espa_api('order', verb='post', body=self.order)
        print(json.dumps(resp, indent=4))

    def list_orders(self):
        print('GET /api/v1/list-orders')
        filters = {"status": ["complete", "ordered"]}  # Here, we ignore any purged orders
        self.orders_list = self.__call_espa_api('list-orders', body=filters)
        print(json.dumps(self.orders_list, indent=4))
        return self.orders_list

    def check_order_status(self, orderid=None):
        if orderid is not None:
            if isinstance(orderid, list):
                orderid = orderid
            else:
                orderid = [orderid]
        else:
            orderid = self.orders_list

        for id in orderid:
            print('GET /api/v1/order-status/{}'.format(id))
            resp = self.__call_espa_api('order-status/{}'.format(id))
            print(json.dumps(resp, indent=4))

    def download_order(self, orderid, output_dir):
        fs.mkdir(output_dir)

        print('GET /api/v1/item-status/{0}'.format(orderid))
        resp = self.__call_espa_api('item-status/{0}'.format(orderid), body={'status': 'complete'})
        #print(json.dumps(resp[orderid], indent=4))

        # Once the order is completed or partially completed, can get the download url's
        for item in resp[orderid]:
            url = item.get('product_dload_url')
            print("Downloading URL: {0}".format(url))
            filename = os.path.basename(url)
            output_file = os.path.join(output_dir, filename)
            wget.download(item.get('product_dload_url'), out=output_file)
