import googlemaps
from datetime import datetime

gmaps = googlemaps.Client(key='AIzaSyCJHUVb9e0rP-h5tPSNzGbSZBI7rMDC6N0')

path_HEB = []
geocode_result_wiess = gmaps.geocode('6340 Main St, Houston, TX')
print(geocode_result_wiess[0]['geometry']['location'])
path_HEB += [geocode_result_wiess[0]['geometry']['location']]

geocode_result_heb = gmaps.geocode('1701 W Alabama, Houston, TX')
print(geocode_result_heb[0]['geometry']['location'])
path_HEB += [geocode_result_heb[0]['geometry']['location']]

results = gmaps.snap_to_roads(path=path_HEB, interpolate=True)

place_id = 'qoy2bBGpGjLik9qrB8R8rg'
place_id2 = 'SoKV6UhaPk-xPNqL_H359A'
path_2 = [(29.72145, -95.41851000000001),(29.721130000000002, -95.41851000000001)]
results2 = gmaps.snap_to_roads(path=path_2, interpolate=True)
print(results)