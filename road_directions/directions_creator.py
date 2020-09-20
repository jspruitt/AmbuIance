"""
Provided code to work with Google maps.
"""

import urllib.request
import math
import simplemap2.simplemap as simplemap
import webbrowser
import googlemaps
from sklearn.neighbors import BallTree
import numpy as np
import random
from EVPath import Graph, EVPath

# Constants
WIDTH = 800
HEIGHT = 500
CTRLWIDTH = 150
_EARTH_RAD = 3440.06479


#####################
# Map GUI
#####################

class MapGUI:
    """
    Class to run the GUI for map pathfinding.
    """

    def __init__(self, name, location, mapdata, pathdata,
                 measle_icon, start_icon, stop_icon, algorithm, google,
                 start_loc, stop_loc):
        # Store icon urls
        self._measle_icon = open(measle_icon)
        self._start_icon = open(start_icon)
        self._stop_icon = open(stop_icon)


        self._algorithm = algorithm
        self._gmaps = google

        # read map data and create markers for each intersection
        self._graph = Graph()
        self._markers = {}
        self._marker_list = []
        self._loc_list = []
        self._road_dictionary = []
        self._gps_path = []

        self._read_map(mapdata)
        self._read_paths(pathdata)

        coords = start_loc
        results = self._gmaps.reverse_geocode(tuple(coords))
        new_loc = results[0]['place_id']
        self._graph.add_node(new_loc)
        self._graph.add_node_attr(new_loc, "lat", coords[0])
        self._graph.add_node_attr(new_loc, "lng", coords[1])
        neighbors = self.nearest_neighbors(coords)
        for nb2 in neighbors:
            self._graph.add_edge(new_loc, nb2)
        self._markers[new_loc] = coords
        self._marker_list += [coords]
        self._start_id = new_loc

        coords = stop_loc
        results = self._gmaps.reverse_geocode(tuple(coords))
        new_loc = results[0]['place_id']
        self._graph.add_node(new_loc)
        self._graph.add_node_attr(new_loc, "lat", coords[0])
        self._graph.add_node_attr(new_loc, "lng", coords[1])
        neighbors = self.nearest_neighbors(coords)
        for nb2 in neighbors:
            self._graph.add_edge(new_loc, nb2)
        self._markers[new_loc] = coords
        self._marker_list += [coords]
        self._stop_id = new_loc


        self.call_algorithm()

        self.draw_graph()

    def _read_map(self, mapdata):
        """
        Read map data and construct graph.  Assumes self._graph and
        self._markers have already been created.
        """
        mapdatafile = open(mapdata)

        for line in mapdatafile.readlines():
            if len(line) > 1:
                fields = line.split(';')
                loc = fields[0]
                nbrs = fields[1].strip()
                lat = float(fields[2].strip())
                lng = float(fields[3].strip())
                name = fields[4].strip()

                self._graph.add_node(loc)
                self._graph.add_node_attr(loc, "lat", lat)
                self._graph.add_node_attr(loc, "lng", lng)

                marker = [lat, lng]
                #marker = self._map.add_marker(name, loc, self._measle_icon,
                                              #(lat, lng), self.click)
                self._marker_list += [marker]
                self._loc_list += [loc]
                self._markers[loc] = marker

                nbr2 = nbrs.split(',')
                for nb1 in nbr2:
                    nb2 = nb1.strip().split(':')
                    self._graph.add_edge(loc, nb2[0])

    def nearest_neighbors(self, start_loc):
        """
        Find nearest point to the location
        """
        a = np.array(self._marker_list)
        ball_tree = BallTree(a, leaf_size=2)
        dist, ind = ball_tree.query([start_loc], k=2)
        return [self._loc_list[i] for i in ind[0]]

    def _read_paths(self, pathdata):
        """
        Read path data and augment graph.  Assumes self._graph has already
        been populated.
        """
        pathdatafile = open(pathdata)

        nodes = self._graph.nodes()

        for line in pathdatafile.readlines():
            fields = line.split(';')
            begin = fields[0].strip()
            end = fields[1].strip()
            if not begin in nodes:
                continue
            if not end in self._graph.get_neighbors(begin):
                continue

            path = []
            for pair in fields[2:]:
                elems = pair.split(":")
                point = (float(elems[0].strip()), float(elems[1].strip()))
                path.append(point)

    def draw_graph(self):
        """
        Draw the entire graph.
        """
        markers = [[29.7174, -95.4018]]
        self._map = simplemap.Map("Rice", markers=markers, points=self._road_dictionary)
        file_url = self._map.write('simplemap/rice.html')
        print('HTML page written to: ' + file_url)
        webbrowser.open(file_url)


        # self._map.clear_lines()
        # for node in self._graph.nodes():
        #     for nbr in self._graph.get_neighbors(node):
        #         start_marker = self._markers[node]
        #         stop_marker = self._markers[nbr]
        #         path = self._graph.get_edge_attr(node, nbr, "path")
        #         self._map.draw_line(start_marker, stop_marker, path)


    def choose_start(self):
        """
        Enter start selection mode.
        """
        self._select_start = True
        self._select_stop = False

    def choose_stop(self):
        """
        Enter stop selection mode.
        """
        self._select_start = False
        self._select_stop = True

    def call_algorithm(self):
        """
        Call provided Alorithm.
        """
        start_id = self._start_id
        stop_id = self._stop_id

        parent = self._algorithm(self._graph, start_id, stop_id)
        self.correct_path(parent, stop_id)
       # self.color_edges(parent)

    def get_path(self):
        directions = []
        for i in range(1, len(self._gps_path) - 1):
            lat_old = self._gps_path[i-1][0]
            lat_now = self._gps_path[i][0]
            lat_new = self._gps_path[i+1][0]

            long_old = self._gps_path[i-1][1]
            long_now = self._gps_path[i][1]
            long_new = self._gps_path[i+1][1]

            # x vector component
            x_old = (lat_old - lat_now)
            x_new = (lat_new - lat_now)

            # y vector component
            y_old = (long_old - long_now)
            y_new = (long_new - long_now)

            l_old = (x_old**2 + y_old**2)**(1./2)
            l_new = (x_new ** 2 + y_new ** 2) ** (1. / 2)

            dot = (x_old * x_new + y_old * y_new)/(l_old * l_new)
            print(dot)
            if dot < -.5:
                directions += [-1]
            elif dot > .5:
                directions += [1]
            else:
                directions += [0]
        return directions

    def correct_path(self, parent, end_node):
        """
        Find the path based on algorithm
        """
        current_node = end_node

        while current_node != None:
            if current_node in self._markers:
                formatted_point = [self._markers[current_node][0], self._markers[current_node][1]]
                self._gps_path += [formatted_point]
            current_node = parent[current_node]

        results = self._gmaps.snap_to_roads(path=self._gps_path, interpolate=True)
        self._road_dictionary = [{'lat':r['location']['latitude'], 'lng': r['location']['longitude']} for r in results]


#####################
# Start GUI
#####################

def start(algorithm, start_node = None, end_node = None):
    """
    Start the GUI.
    """
    # injury site
    start_node = 1

    # Houston Methodist ER
    end_node = [29.7083761,-95.4081554]

    # Find nearest road to gps_location
    # Hospital should already be a node.

    gmaps = googlemaps.Client(key='AIzaSyCJHUVb9e0rP-h5tPSNzGbSZBI7rMDC6N0')

    querey_hospital = gmaps.find_place("Houston Medical Center Emergency Room", 'textquery')
    hospital = gmaps.place(querey_hospital['candidates'][0]['place_id'])
    hosp_coords = [hospital['result']['geometry']['location']['lat'],
                   hospital['result']['geometry']['location']['lat']]

    incident_coords = [29.725471, -95.400051]

    m = MapGUI("Rice University", [29.7174, -95.4018],
           "mapdata.txt",
           "pathdata.txt",
           "measle_blue.jpg",
           "pin_green.png",
           "pin_red.png", algorithm, google = gmaps,

           start_loc = incident_coords,
           stop_loc = hosp_coords)

if __name__ == "__main__":
    start(EVPath)
