"""
Directed graph class.
"""

class DiGraph:
    """
    Directed graph.
    """

    def __init__(self):
        """
        Initializes an empty graph.
        """
        self._graph = {}

    def __str__(self):
        """
        Returns a string representation of the graph.
        """
        return_str  = "[node]\t[node attrs]\t[edges]\n"
        return_str += "======\t============\t============================\n"
        for node in self.nodes():
            return_str += str(node)
            hasnattrs = False
            for nattrkey in self._graph[node][0]:
                hasnattrs = True
                return_str += "\t" + str(nattrkey) + ": "
                return_str += str(self.get_node_attr(node, nattrkey)) + "\n"
            if not hasnattrs:
                return_str += "\n"
            return_str += "\t\t\t[neighbor]\t[edge attrs]\n"
            return_str += "\t\t\t==========\t============\n"
            for nbr in self.get_neighbors(node):
                return_str += "\t\t\t" + str(nbr) + "\t"*(2 - len(str(nbr))/16)
                haseattrs = False
                ntabs = 3 + max(0, 2 - len(str(nbr))/16)
                for eattrkey in self._graph[node][1][nbr]:
                    if haseattrs:
                        # already printed one attr
                        return_str += "\t" * ntabs
                    return_str += str(eattrkey) + ": "
                    return_str += str(self.get_edge_attr(node, nbr, eattrkey))
                    return_str += "\n"
                    haseattrs = True
                if not haseattrs:
                    return_str += "\n"
        return return_str

    def nodes(self):
        """
        Returns a list of nodes in the graph.
        """
        return list(self._graph.keys())

    def has_node(self, node):
        """
        Returns True if node is in the graph, False otherwise.
        """
        return node in self._graph

    def has_edge(self, node1, node2):
        """
        Returns True if node1 and node2 are in the graph and there is
        an edge from node1 to node2, False otherwise.
        """
        if (node1 not in self._graph) or (node2 not in self._graph):
            return False
        edges = self._graph[node1][1]
        return node2 in edges

    def get_neighbors(self, node):

        """
        Returns the neighbor list for node. Assumes that node is in the graph;
        raises a KeyError if it is not.
        """
        return list(self._graph[node][1].keys())

    def add_node(self, node):
        """
        Adds node to the graph. Does nothing if node is already in the graph.
        """
        if node not in self._graph:
            self._graph[node] = ({}, {})

    def add_edge(self, node1, node2):
        """
        Adds an edge between the two nodes in the graph, adding the nodes
        themselves if they're not already there.
        """
        ## Update the first node's neighbor list
        if node1 not in self._graph:
            self._graph[node1] = ({}, {node2:{}})
        else:
            self._graph[node1][1][node2] = {}

        ## Add node2 to the graph if it's not already there
        if node2 not in self._graph:
            self._graph[node2] = ({}, {})

    def add_node_attr(self, node, key, val):
        """
        Adds the (key, val) pair to the attributes dictionary for the
        given node.  Assumes that node is in the graph; raises a
        KeyError if it is not.
        """
        self._graph[node][0][key] = val

    def add_edge_attr(self, node1, node2, key, val):
        """
        Adds the (key, val) pair to the attributes dictionary for the
        given edge.  Assumes there is an edge between (node1, node2)
        in the graph; raises a KeyError if there is not.
        """
        self._graph[node1][1][node2][key] = val

    def has_node_attr(self, node, key):
        """
        Returns True if the node has an attribute for "key",
        False otherwise.
        """
        if node not in self._graph:
            return False
        return key in self._graph[node][0]

    def get_node_attr(self, node, key):
        """
        Returns the attributes of the given node associated with the given
        key.  Assumes that node is in the graph, and that it has an
        attribute for the given key; raises a KeyError otherwise.
        """
        return self._graph[node][0][key]

    def has_edge_attr(self, node1, node2, key):
        """
        Returns True if the edge from node1 to node2 has an attribute for
        "key", False otherwise.
        """
        if (node1 not in self._graph) or (node2 not in self._graph):
            return False
        if node2 not in self._graph[node1][1]:
            return False
        return key in self._graph[node1][1][node2]

    def get_edge_attr(self, node1, node2, key):
        """
        Given a pair of nodes and a key, returns the attribute of the edge
        between them associated with the given key. Assumes that there
        is an edge between the two nodes, and that an attribute for
        the given key is set.
        """
        return self._graph[node1][1][node2][key]

    def copy(self):
        """
        Returns an identical (deep) copy of the graph.
        """
        g_new = DiGraph()
        for node in self.nodes():
            g_new.add_node(node)
            for key, val in self._graph[node][0].items():
                g_new.add_node_attr(node, key, val)

            for nbr in self.get_neighbors(node):
                g_new.add_edge(node, nbr)
                for key, val in self._graph[node][1][nbr].items():
                    g_new.add_edge_attr(node, nbr, key, val)
        return g_new
