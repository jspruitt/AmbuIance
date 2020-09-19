import comp140_module7_graphs as graphs
"""
G -- graph = (V, E, RNJun)
S -- start node (element of V)
T -- end node (Element of V)

Output hierarchical graph hg
"""
def fewestTurn(G, S, T):
    hG = graphs.Graph()
    color = dict{}
    for node in G.nodes:
        if node == S:
            color[S] = 0
        else:
            color[node] = 1
    Q = [S]
    hf = S
    while (T not in hG):
        curV = Q.pop(0)
        for neighbor in curV.get_neighbors():
            if color[neighbor] = 1:
                Q.append(neighbor)
            color[neighbor] = 0
        color[curV] = 2
        


