"""
Provided code to work with Google maps.
"""

import urllib.request
import math
import comp140_module7_graphs as graphs
import simplemap2.simplemap as simplemap
import webbrowser
import googlemaps
from sklearn.neighbors import BallTree
import numpy as np

# Constants
WIDTH = 800
HEIGHT = 500
CTRLWIDTH = 150
_EARTH_RAD = 3440.06479

#####################
# Distance functions
#####################

def node_name(graph, node):
    """
    Return string to use for name of "node".
    """
    ndname = str(node)
    if graph.has_node_attr(node, "name"):
        nameattr = graph.get_node_attr(node, "name")
        ndname += ' ("' + str(nameattr) + '")'
    return ndname

# map_ functions are for use with actual Google maps
def map_straight_line_distance(id1, id2, graph):
    """
    Receives two nodes and a graph to which those two nodes belong,
    and returns the straightline ("as the crow flies") distance
    between them, in meters.
    """
    # Check for errors
    if not graph.has_node(id1):
        raise ValueError("Node " + str(id1) + " is not in graph")
    if not graph.has_node(id2):
        raise ValueError("Node " + str(id2) + " is not in graph")

    # Get Latitude/Longitude
    lat1 = graph.get_node_attr(id1, "lat")
    lon1 = graph.get_node_attr(id1, "lng")
    lat2 = graph.get_node_attr(id2, "lat")
    lon2 = graph.get_node_attr(id2, "lng")

    # Convert from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Convert to nautical miles
    sinlatpow = pow(math.sin(float(lat2 - lat1) / 2), 2)
    sinlonpow = pow(math.sin(float(lon2 - lon1) / 2), 2)
    aval = sinlatpow + math.cos(lat1) * math.cos(lat2) * sinlonpow
    bval = 2 * math.atan2(math.sqrt(aval), math.sqrt(1 - aval))
    nmiles = _EARTH_RAD * bval

    # Convert to meters
    meters = int(nmiles * 1852)
    return meters

def map_edge_distance(id1, id2, graph):
    """
    Receives two nodes and a graph in which those two nodes are direct
    neighbors, and returns the distance along the edge between them,
    in meters.
    """
    # Check for errors
    if not graph.has_node(id1):
        raise ValueError("Node " + str(id1) + " is not in graph")
    if not graph.has_node(id2):
        raise ValueError("Node " + str(id2) + " is not in graph")
    if not graph.has_edge(id1, id2):
        name1 = node_name(graph, id1)
        name2 = node_name(graph, id2)
        msg = "Graph does not have an edge from node "
        msg += name1 + " to node " + name2
        raise ValueError(msg)

    return graph.get_edge_attr(id1, id2, "dist")

# test_ functions are for use with the test graphs
def test_straight_line_distance(id1, id2, graph):
    """
    Receives two nodes and a graph to which those two nodes belong,
    and returns the straightline distance between them.
    """
    # Check for errors
    if not graph.has_node(id1):
        raise ValueError("Node " + str(id1) + " is not in graph")
    if not graph.has_node(id2):
        raise ValueError("Node " + str(id2) + " is not in graph")

    # Get x, y
    xval1 = graph.get_node_attr(id1, "x")
    yval1 = graph.get_node_attr(id1, "y")
    xval2 = graph.get_node_attr(id2, "x")
    yval2 = graph.get_node_attr(id2, "y")

    # Calculate distance
    dist = math.sqrt((xval2 - xval1) ** 2 + (yval2 - yval1) ** 2)

    return dist

def test_edge_distance(id1, id2, graph):
    """
    Receives two nodes and a graph in which those two nodes are direct
    neighbors, and returns the distance along the edge between them.
    """
    # Check for errors
    if not graph.has_node(id1):
        raise ValueError("Node " + str(id1) + " is not in graph")
    if not graph.has_node(id2):
        raise ValueError("Node " + str(id2) + " is not in graph")
    if not graph.has_edge(id1, id2):
        msg = "Graph does not have an edge from node "
        msg += str(id1) + " to node " + str(id2)
        raise ValueError(msg)

    return graph.get_edge_attr(id1, id2, 'dist')

#####################
# Test graphs
#####################

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def build_line():
    """
    Build a graph that is 5 nodes in a line.
    """
    graph = graphs.DiGraph()
    for idx in range(4):
        graph.add_edge(LETTERS[idx], LETTERS[idx+1])
        graph.add_node_attr(LETTERS[idx], 'x', idx)
        graph.add_node_attr(LETTERS[idx], 'y', 0)
        graph.add_edge_attr(LETTERS[idx], LETTERS[idx+1], 'edgenum', idx)
        graph.add_edge_attr(LETTERS[idx], LETTERS[idx+1], 'dist', 1)

    graph.add_node_attr(LETTERS[4], 'x', idx)
    graph.add_node_attr(LETTERS[4], 'y', 0)
    return graph

def build_clique():
    """
    Build a complete graph with 5 nodes.
    """
    graph = graphs.DiGraph()
    edge_num = 0
    for idx in range(5):
        graph.add_node(LETTERS[idx])
        for idx2 in range(5):
            if (idx != idx2) and (not (idx == 1 and idx2 == 0)):
                graph.add_edge(LETTERS[idx], LETTERS[idx2])
                graph.add_edge_attr(LETTERS[idx], LETTERS[idx2],
                                    'edgenum', edge_num)
                graph.add_edge_attr(LETTERS[idx], LETTERS[idx2], 'dist', 5)
                edge_num += 1

    # Create ring that is closer
    for idx in range(4):
        graph.add_edge_attr(LETTERS[idx], LETTERS[idx+1], 'dist', 1)

    # Set positions
    graph.add_node_attr(LETTERS[0], 'x', 0)
    graph.add_node_attr(LETTERS[0], 'y', 0)
    graph.add_node_attr(LETTERS[1], 'x', 0)
    graph.add_node_attr(LETTERS[1], 'y', 1)
    graph.add_node_attr(LETTERS[2], 'x', 1)
    graph.add_node_attr(LETTERS[2], 'y', 1)
    graph.add_node_attr(LETTERS[3], 'x', 2)
    graph.add_node_attr(LETTERS[3], 'y', 1)
    graph.add_node_attr(LETTERS[4], 'x', 2)
    graph.add_node_attr(LETTERS[4], 'y', 0)

    return graph

def get_grid_nbrs(node, width, height):
    """
    Get the neighbors of node in a width x height grid.
    """
    nbrs = []
    if node % width != width - 1:
        nbrs.append(node + 1)
    if node < (width * (height - 1)):
        nbrs.append(node + width)
    return nbrs

def build_grid():
    """
    Build a graph that is a 3 x 3 grid.
    """
    graph = graphs.DiGraph()
    edge_num = 0
    for idx in range(9):
        graph.add_node(LETTERS[idx])
        graph.add_node_attr(LETTERS[idx], 'x', idx % 3)
        graph.add_node_attr(LETTERS[idx], 'y', 2 - (idx // 3))

        for nbr in get_grid_nbrs(idx, 3, 3):
            graph.add_edge(LETTERS[idx], LETTERS[nbr])
            graph.add_edge_attr(LETTERS[idx], LETTERS[nbr], 'edgenum', edge_num)
            graph.add_edge_attr(LETTERS[idx], LETTERS[nbr], 'dist', 1)
            edge_num += 1

    # Add an extra edge going the other way
    graph.add_edge('H', 'G')
    graph.add_edge_attr('H', 'G', 'edgenum', edge_num)
    graph.add_edge_attr('H', 'G', 'dist', 1)

    # Make some edges longer
    graph.add_edge_attr('D', 'G', 'dist', 5)
    return graph

def build_big_grid():
    """
    Build a graph that is a 10x10 grid.
    """
    graph = graphs.DiGraph()
    edge_num = 0
    for idx in range(100):
        graph.add_node(str(idx))
        graph.add_node_attr(str(idx), 'x', idx % 10)
        graph.add_node_attr(str(idx), 'y', idx // 10)

        for nbr in get_grid_nbrs(idx, 10, 10):
            graph.add_edge(str(idx), str(nbr))
            graph.add_edge_attr(str(idx), str(nbr), 'edgenum', edge_num)
            graph.add_edge(str(nbr), str(idx))
            graph.add_edge_attr(str(nbr), str(idx), 'edgenum', edge_num)

            graph.add_edge_attr(str(idx), str(nbr), 'dist', (edge_num % 3) + 1)
            graph.add_edge_attr(str(nbr), str(idx), 'dist', (edge_num % 3) + 1)
            edge_num += 1

    return graph

def build_tree():
    """
    Build a binary tree of height 3.
    """
    graph = graphs.DiGraph()
    node = 0
    graph.add_node(LETTERS[node])
    edge_num = 0
    for idx in range(3): # height = 3
        for idx2 in range(2**idx):
            graph.add_node(LETTERS[node])
            for idx3 in range(2): # binary tree
                graph.add_edge(LETTERS[node], LETTERS[node + (node + 1) + idx3])
                graph.add_edge_attr(LETTERS[node],
                                    LETTERS[node + (node + 1) + idx3],
                                    'edgenum', edge_num)
                edge_num += 1
            node += 1

    node = 0
    for idx in range(4):
        for idx2 in range(2 ** idx):
            graph.add_node_attr(LETTERS[node], 'x', idx)
            graph.add_node_attr(LETTERS[node], 'y', idx2  * (8 / 2 ** idx))
            node += 1

    # Set distances
    graph.add_edge_attr(LETTERS[0], LETTERS[1], 'dist', 1)
    graph.add_edge_attr(LETTERS[1], LETTERS[3], 'dist', 1)
    graph.add_edge_attr(LETTERS[3], LETTERS[7], 'dist', 1)
    graph.add_edge_attr(LETTERS[4], LETTERS[9], 'dist', 1)
    graph.add_edge_attr(LETTERS[2], LETTERS[5], 'dist', 1)
    graph.add_edge_attr(LETTERS[5], LETTERS[11], 'dist', 1)
    graph.add_edge_attr(LETTERS[6], LETTERS[13], 'dist', 1)

    graph.add_edge_attr(LETTERS[3], LETTERS[8], 'dist', 2)
    graph.add_edge_attr(LETTERS[4], LETTERS[10], 'dist', 2)
    graph.add_edge_attr(LETTERS[5], LETTERS[12], 'dist', 2)

    graph.add_edge_attr(LETTERS[1], LETTERS[4], 'dist', 3)

    graph.add_edge_attr(LETTERS[0], LETTERS[2], 'dist', 10)
    graph.add_edge_attr(LETTERS[2], LETTERS[6], 'dist', 10)
    graph.add_edge_attr(LETTERS[6], LETTERS[14], 'dist', 10)

    return graph

def build_asymmetric1():
    """
    Build an asymmetric graph with 11 nodes.
    """
    graph = graphs.DiGraph()
    nodes = {0: [1], 1: [2, 7], 2: [3], 3: [7, 4],
             4: [5], 6: [7, 10], 7: [8], 8: [9, 10]}
    edge_num = 0
    for node, nbrs in nodes.items():
        for nbr in nbrs:
            graph.add_edge(LETTERS[node], LETTERS[nbr])
            graph.add_edge_attr(LETTERS[node], LETTERS[nbr],
                                'edgenum', edge_num)
            edge_num += 1

    # Set positions
    graph.add_node_attr(LETTERS[0], 'x', 0)
    graph.add_node_attr(LETTERS[0], 'y', 0)
    graph.add_node_attr(LETTERS[1], 'x', 1)
    graph.add_node_attr(LETTERS[1], 'y', 0)
    graph.add_node_attr(LETTERS[2], 'x', 2)
    graph.add_node_attr(LETTERS[2], 'y', 0)
    graph.add_node_attr(LETTERS[3], 'x', 2)
    graph.add_node_attr(LETTERS[3], 'y', 1)
    graph.add_node_attr(LETTERS[4], 'x', 1)
    graph.add_node_attr(LETTERS[4], 'y', 1)
    graph.add_node_attr(LETTERS[5], 'x', 0)
    graph.add_node_attr(LETTERS[5], 'y', 1)
    graph.add_node_attr(LETTERS[6], 'x', 0)
    graph.add_node_attr(LETTERS[6], 'y', 2)
    graph.add_node_attr(LETTERS[7], 'x', 1)
    graph.add_node_attr(LETTERS[7], 'y', 2)
    graph.add_node_attr(LETTERS[8], 'x', 2)
    graph.add_node_attr(LETTERS[8], 'y', 2)
    graph.add_node_attr(LETTERS[9], 'x', 2)
    graph.add_node_attr(LETTERS[9], 'y', 3)
    graph.add_node_attr(LETTERS[10], 'x', 0)
    graph.add_node_attr(LETTERS[10], 'y', 3)

    # Set distances
    graph.add_edge_attr(LETTERS[0], LETTERS[1], 'dist', 1)
    graph.add_edge_attr(LETTERS[1], LETTERS[2], 'dist', 1)
    graph.add_edge_attr(LETTERS[2], LETTERS[3], 'dist', 1)
    graph.add_edge_attr(LETTERS[3], LETTERS[4], 'dist', 1)
    graph.add_edge_attr(LETTERS[4], LETTERS[5], 'dist', 1)

    graph.add_edge_attr(LETTERS[6], LETTERS[7], 'dist', 1)
    graph.add_edge_attr(LETTERS[6], LETTERS[10], 'dist', 1)

    graph.add_edge_attr(LETTERS[1], LETTERS[7], 'dist', 5)
    graph.add_edge_attr(LETTERS[3], LETTERS[7], 'dist', 2)

    graph.add_edge_attr(LETTERS[7], LETTERS[8], 'dist', 1)
    graph.add_edge_attr(LETTERS[8], LETTERS[9], 'dist', 1)

    graph.add_edge_attr(LETTERS[8], LETTERS[10], 'dist', 4)

    return graph

def build_asymmetric2():
    """
    Build an asymetric graph with 10 nodes.
    """
    graph = graphs.DiGraph()
    nodes = {0:[1, 2, 3, 4], 1:[4], 5:[8], 6:[7], 7:[8], 8:[9]}
    edge_num = 0
    for node, nbrs in nodes.items():
        for nbr in nbrs:
            graph.add_edge(LETTERS[node], LETTERS[nbr])
            graph.add_edge_attr(LETTERS[node], LETTERS[nbr],
                                'edgenum', edge_num)
            edge_num += 1

    # set positions
    for idx in range(10):
        graph.add_node_attr(LETTERS[idx], 'x', idx % 3)
        graph.add_node_attr(LETTERS[idx], 'y', 2 - (idx // 3))

    # set distances
    graph.add_edge_attr(LETTERS[0], LETTERS[1], 'dist', 1)
    graph.add_edge_attr(LETTERS[0], LETTERS[2], 'dist', 2)
    graph.add_edge_attr(LETTERS[0], LETTERS[3], 'dist', 1)
    graph.add_edge_attr(LETTERS[0], LETTERS[4], 'dist', 1.75)
    graph.add_edge_attr(LETTERS[1], LETTERS[4], 'dist', 1)
    graph.add_edge_attr(LETTERS[5], LETTERS[8], 'dist', 1)
    graph.add_edge_attr(LETTERS[6], LETTERS[7], 'dist', 1)
    graph.add_edge_attr(LETTERS[7], LETTERS[8], 'dist', 1)
    graph.add_edge_attr(LETTERS[8], LETTERS[9], 'dist', 2.5)

    return graph

def build_asymmetric3():
    """
    Build an asymmetric graph with 11 nodes.
    """
    graph = graphs.DiGraph()
    nodes = {0: [1], 1: [2, 7], 2: [3], 3: [7, 4],
             4: [5], 6: [7, 10], 7: [], 8: [9, 10]}
    edge_num = 0
    for node, nbrs in nodes.items():
        for nbr in nbrs:
            graph.add_edge(LETTERS[node], LETTERS[nbr])
            graph.add_edge_attr(LETTERS[node], LETTERS[nbr],
                                'edgenum', edge_num)
            edge_num += 1

    # Set positions
    graph.add_node_attr(LETTERS[0], 'x', 0)
    graph.add_node_attr(LETTERS[0], 'y', 0)
    graph.add_node_attr(LETTERS[1], 'x', 1)
    graph.add_node_attr(LETTERS[1], 'y', 0)
    graph.add_node_attr(LETTERS[2], 'x', 2)
    graph.add_node_attr(LETTERS[2], 'y', 0)
    graph.add_node_attr(LETTERS[3], 'x', 2)
    graph.add_node_attr(LETTERS[3], 'y', 1)
    graph.add_node_attr(LETTERS[4], 'x', 1)
    graph.add_node_attr(LETTERS[4], 'y', 1)
    graph.add_node_attr(LETTERS[5], 'x', 0)
    graph.add_node_attr(LETTERS[5], 'y', 1)
    graph.add_node_attr(LETTERS[6], 'x', 0)
    graph.add_node_attr(LETTERS[6], 'y', 2)
    graph.add_node_attr(LETTERS[7], 'x', 1)
    graph.add_node_attr(LETTERS[7], 'y', 2)
    graph.add_node_attr(LETTERS[8], 'x', 2)
    graph.add_node_attr(LETTERS[8], 'y', 2)
    graph.add_node_attr(LETTERS[9], 'x', 2)
    graph.add_node_attr(LETTERS[9], 'y', 3)
    graph.add_node_attr(LETTERS[10], 'x', 0)
    graph.add_node_attr(LETTERS[10], 'y', 3)

    # Set distances
    graph.add_edge_attr(LETTERS[0], LETTERS[1], 'dist', 1)
    graph.add_edge_attr(LETTERS[1], LETTERS[2], 'dist', 1)
    graph.add_edge_attr(LETTERS[2], LETTERS[3], 'dist', 1)
    graph.add_edge_attr(LETTERS[3], LETTERS[4], 'dist', 1)
    graph.add_edge_attr(LETTERS[4], LETTERS[5], 'dist', 1)

    graph.add_edge_attr(LETTERS[6], LETTERS[7], 'dist', 1)
    graph.add_edge_attr(LETTERS[6], LETTERS[10], 'dist', 1)

    graph.add_edge_attr(LETTERS[1], LETTERS[7], 'dist', 5)
    graph.add_edge_attr(LETTERS[3], LETTERS[7], 'dist', 2)

    graph.add_edge_attr(LETTERS[8], LETTERS[9], 'dist', 1)
    graph.add_edge_attr(LETTERS[8], LETTERS[10], 'dist', 4)

    return graph

GRAPHS = {'line': build_line,
          'clique': build_clique,
          'grid': build_grid,
          'biggrid': build_big_grid,
          'tree': build_tree,
          'asymmetric1': build_asymmetric1,
          'asymmetric2': build_asymmetric2,
          'asymmetric3': build_asymmetric3}

def load_test_graph(name):
    """
    Given the name of a test graph, return it as a graphs.DiGraph object.
    """
    if not name in GRAPHS:
        raise ValueError("test graph name must be in " + str(list(GRAPHS.keys())))
    return GRAPHS[name]()

#####################
# Map GUI
#####################

class MapGUI:
    """
    Class to run the GUI for map pathfinding.
    """

    def __init__(self, name, location, mapdata, pathdata,
                 measle_icon, start_icon, stop_icon, algorithm, google,
                 start_loc=None, stop_loc=None):
        # Store icon urls
        self._measle_icon = open(measle_icon)
        self._start_icon = open(start_icon)
        self._stop_icon = open(stop_icon)

        self._start_id = start_loc
        self._stop_id = stop_loc

        self._algorithm = algorithm
        self._gmaps = google

        # read map data and create markers for each intersection
        self._graph = graphs.DiGraph()
        self._markers = {}
        self._marker_list = []
        self._loc_list = []
        self._road_dictionary = []


        self._read_map(mapdata)
        self._read_paths(pathdata)

        coords = [29.719573, -95.408499]
        results = self._gmaps.reverse_geocode(tuple(coords))
        new_loc = results[0]['place_id']
        neighbors = self.nearest_neighbors(coords)
        for nb2 in neighbors:
            self._graph.add_edge(new_loc, nb2)
            self._graph.add_edge_attr(new_loc, nb2, "dist", 0)
            self._graph.add_edge_attr(new_loc, nb2, "path", None)
        self._markers[new_loc] = coords
        self._marker_list += [coords]
        self._start_id = new_loc
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
                self._graph.add_node_attr(loc, "name", name)

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
                    self._graph.add_edge_attr(loc, nb2[0], "dist", float(nb2[1]))
                    self._graph.add_edge_attr(loc, nb2[0], "path", None)

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
            self._graph.add_edge_attr(begin, end, "path", path)

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

        parent = self._algorithm(self._graph, start_id)
        self.correct_path(parent, stop_id)
       # self.color_edges(parent)

    def correct_path(self, parent, end_node):
        """
        Find the path based on algorithm
        """

        current_node = end_node

        while current_node != None:
            if current_node in self._markers:
                formatted_point = [self._markers[current_node][0], self._markers[current_node][1]]
                self._road_dictionary += [formatted_point]
            current_node = parent[current_node]

        results = self._gmaps.snap_to_roads(path=self._road_dictionary, interpolate=True)
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

    MapGUI("Rice University", [29.7174, -95.4018],
           "comp140_module7_mapdata.txt",
           "comp140_module7_pathdata.txt",
           "comp140_module7_measle_blue.jpg",
           "comp140_module7_pin_green.png",
           "comp140_module7_pin_red.png", algorithm, google = gmaps,
           start_loc = "boVs8yK4i4UOA25B6cDpiA",
           stop_loc = "WBvF8w13UYfiuUnIIo_n2g")

def bfs(graph, start_node):
    """
    Performs a breadth-first search on graph starting at the
    start_node.
    """
    parents = {start_node: None}
    orders = {start_node: 0}
    queue = []
    queue.append(start_node)
    parent = {start_node}
    while queue:
        next_person = queue.pop(0)
        children = graph.get_neighbors(next_person)
        for child in children:
            if child not in parent:
                parents[child] = next_person
                queue.append(child)
                parent.add(child)
    return parents

def test():

    gmaps = googlemaps.Client(key='AIzaSyCJHUVb9e0rP-h5tPSNzGbSZBI7rMDC6N0')

    results = gmaps.nearest_roads(points=[[29.7083761,-95.4081554]])
    print(results)


start(bfs)