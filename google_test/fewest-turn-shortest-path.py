"""
G -- graph = (V, E, RNJun)
S -- start node (element of V)
T -- end node (Element of V)

Output hierarchical graph hg
"""
def fewestTurn(G, S, T):
    hG = None
    color = dict{}
    for node in G.nodes:
        if node == S:
            color[S] = 0
        else:
            color[node] = 1
    Q = [S]
    hf = S
    while ((hG != None) and (T not in hG)):
        curV = Q.pop(0)

