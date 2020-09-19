"""
Map search
"""

import random


def EVPath(graph, start_node, end_node):
    """
    Performs a breadth-first search with randomness introduced
    """
    rac = Queue()
    dist = {}
    parent = {}

    for node in graph.nodes():
        dist[node] = float('inf')
        parent[node] = None

    dist[start_node] = 0
    rac.push(start_node)

    while rac:
        node = rac.pop()
        nbrs = graph.get_neighbors(node)

        #randomly shuffle elements in nbrs to introduce different choices
        random.shuffle(nbrs)

        for nbr in nbrs:
            if dist[nbr] == float('inf'):
                dist[nbr] = dist[node] + 1
                parent[nbr] = node
                rac.push(nbr)
                if nbr == end_node:
                    return parent

    return parent




class Queue:
    """
    A simple implementation of a FIFO queue.
    """

    def __init__(self):
        """
        Initialize the queue.
        """
        self._items = []

    def __len__(self):
        """
        Return number of items in the queue.
        """
        return len(self._items)

    def __str__(self):
        """
        Return a string representing the queue.
        """
        return str(self._items)

    def push(self, item):
        """
        Add item to the queue.
        """
        self._items.append(item)

    def pop(self):
        """
        Return and remove least recently inserted item.

        Assumes that there is at least one element in the queue.  It
        is an error if there is not.  You do not need to check for
        this condition.
        """
        return self._items.pop(0)

    def clear(self):
        """
        Remove all items from the queue.
        """
        self._items = []


"""
Undirected graph class.
"""

class Graph:
    """
    Undirected graph.
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
        return_str = "[node]\n======\n"
        for node in self.nodes():
            return_str += str(node) + "\n"
            return_str += "\t[neighbor]\t\t[attrs]\n"
            return_str += "\t==========\t\t=======\n"
            for nbr in self.get_neighbors(node):
                return_str += "\t" + str(nbr) + "\t"*(2 - len(str(nbr))//16)
                return_str += str(self.get_attrs(node, nbr)) + "\n"
        return return_str

    def nodes(self):
        """
        Returns a list of nodes in the graph.
        """
        return list(self._graph.keys())

    def get_neighbors(self, node):
        """
        Returns the neighbor list for node or raises a KeyError if node is not
        in the graph.
        """
        return list(self._graph[node].keys())

    def add_node(self, node):
        """
        Add node to the graph. Does nothing if node is already in the graph.
        """
        if node not in self._graph:
            self._graph[node] = {}

    def add_edge(self, node1, node2, attrs):
        """
        Add an edge between two nodes in the graph, adding the nodes
        themselves if they're not already there.
        """
        ## Update the first node's neighbor list
        if node1 not in self._graph:
            self._graph[node1] = {node2:attrs}
        elif node2 not in self._graph[node1]:
            self._graph[node1][node2] = attrs
        else:
            self._graph[node1][node2] = self._graph[node1][node2].union(attrs)

        ## Update the second node's neighbor list
        if node2 not in self._graph:
            self._graph[node2] = {node1:attrs}
        elif node1 not in self._graph[node2]:
            self._graph[node2][node1] = attrs
        else:
            self._graph[node2][node1] = self._graph[node1][node2].union(attrs)

    def get_attrs(self, node1, node2):
        """
        Given a pair of nodes, returns the attribute of the edge
        between them.  Assumes that there is an edge between the two
        nodes.
        """
        return self._graph[node1][node2]

    def copy(self):
        """
        Returns an identical (deep) copy of the graph.
        """
        g_new = Graph()
        for node in self.nodes():
            for nbr in self.get_neighbors(node):
                g_new.add_edge(node, nbr, set(self.get_attrs(node, nbr)))
        return g_new


