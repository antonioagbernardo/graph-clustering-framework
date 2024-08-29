#Relaxed Minimum Spanning Tree (RMST) graph
import numpy as np
from igraph import Graph
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import networkx as nx
import itertools

def rmst_graph(X, gamma, k):

    D = euclidean_distances(X, X)
    N = D.shape[0]
    assert k < N
    np.fill_diagonal(D,0)

    adj = np.zeros([N, N])

    D_k = np.sort(D)
    D_k = np.tile(D_k[:,k],(N,1))
    D_k = gamma * (D_k + D_k.T)
    np.fill_diagonal(D_k,0)

    max_weight = np.zeros((N,N))
    G = nx.Graph(D)
    T = nx.minimum_spanning_tree(G)
    path = dict(nx.all_pairs_dijkstra_path(T))
    for i,j in itertools.combinations(range(N),2):
        p = path[i][j]
        path_weight = np.zeros(len(p)-1)
        for k in range(len(p)-1):
            path_weight[k] = T.edges[p[k],p[k+1]]['weight']
        max_weight[i][j] = np.amax(path_weight)
    max_weight = max_weight + max_weight.T
    np.fill_diagonal(max_weight,0)

    adj[D < (max_weight + D_k)] = 1
    np.fill_diagonal(adj,0)

    W = csr_matrix(adj)

    g = Graph.Weighted_Adjacency(W.todense(), mode='undirected', attr='weight', loops=False)

    return g, W
