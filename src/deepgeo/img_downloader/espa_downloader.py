import getpass
import geopandas as gpd
import json
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
        wrs_intersection = self.ls_grid[self.ls_grid.intersects(self.roi.geometry[0])]
        paths, rows = wrs_intersection['PATH'].values, wrs_intersection['ROW'].values
        return paths, rows