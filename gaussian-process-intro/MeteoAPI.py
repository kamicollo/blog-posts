import http.client
import json
import geopandas as gpd
from pathlib import Path
import os
import hashlib
from shapely.geometry import Point
import pandas as pd

class MeteoAPI:
    
    def __init__(self, use_cache = False, cache_dir = None):
        assert not use_cache or (use_cache and cache_dir is not None), "Cache requested, but no directory provided"
        assert not use_cache or (use_cache and Path(cache_dir).is_dir()), "Cache requested, but invalid directory provided"
        assert not use_cache or (use_cache and os.access(cache_dir, os.W_OK | os.X_OK)), "Cache requested, but directory not writable"
        self.use_cache = use_cache
        self.cache_dir = cache_dir        
    
    def get_connection(self):
        return http.client.HTTPSConnection('api.meteo.lt')
    
    def get_api_results(self, url):
        if self.use_cache:
            hash_id = hashlib.md5(url.encode('utf-8')).hexdigest()
            filename = self.cache_dir + "/" + hash_id + ".json"
            #if file exists - get results
            if Path(filename).is_file():
                with open(filename, "r") as f:
                    resp = json.load(f)
            else:
                #otherwise, retrieve from API
                with open(filename, 'w') as f:                    
                    resp = self.get_response(url)                    
                    if resp is not None: #save if a good response
                        json.dump(resp, f)
        else:
            resp = self.get_response(url)
            
        return resp if resp is not None else []
                    
    
    def get_response(self, url):
        conn = self.get_connection()            
        conn.request('GET',url)
        resp = conn.getresponse()
        
        if resp.status == 200:
            result = json.loads(resp.read())
        else:
            result = None
            
        conn.close()        
        return result

    def get_places(self):        
            result = self.get_api_results('/v1/places')
            self.places = [p['code'] for p in result]            
            return self.places
        
    def get_forecast(self, place_code):
        assert place_code in self.places, "Unknown place code"
        url = '/v1/places/{}/forecasts/long-term'.format(place_code)
        result = self.get_api_results(url)
        if len(result) > 0 and 'forecastTimestamps' in result:
            
            #initialize dataframe
            fc = gpd.GeoDataFrame(result['forecastTimestamps'])

            #assign place attributes
            fc['location'] = result['place']['name']
            fc['lat'] = result['place']['coordinates']['latitude']
            fc['lon'] = result['place']['coordinates']['longitude']
            
            #type conversions - numeric columns
            num_cols = list(set(fc.columns).difference(['forecastTimeUtc', 'location', 'conditionCode']))
            fc[num_cols] = fc[num_cols].astype(float)

            #type conversions - datetime
            fc['forecastTimeUtc'] = pd.to_datetime(fc['forecastTimeUtc'])

            #type conversion lat/lon to geometry column
            points = [Point(lon, lat) for i,lat,lon in fc[['lat', 'lon']].itertuples()]
            fc = fc.assign(**{'geometry': gpd.GeoSeries(points).set_crs('EPSG:4326')})
            fc = fc.drop(['lat', 'lon'], axis = 1)
                
        else:
            fc = gpd.GeoDataFrame()
        
        return fc