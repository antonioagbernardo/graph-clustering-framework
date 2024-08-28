import numpy as np
from scipy.spatial.distance import cosine_distances
from scipy.spatial.distance import euclidean_distances
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import igraph

def mst_graph(X, metric):

    if metric == 'cosine':
        D = cosine_distances(X, X)
    else:
        D = euclidean_distances(X, X)
    adj_directed = minimum_spanning_tree(D).toarray()
    adj = adj_directed + adj_directed.T
    adj[adj > 0] = 1
    np.fill_diagonal(adj,0)

    return csr_matrix(adj)

def MST(X, metric):

    if metric == 'cosine':
        D = cosine_distances(X, X)
    else:
        D = euclidean_distances(X, X)
    adj_directed = minimum_spanning_tree(D).toarray()
    adj = adj_directed + adj_directed.T
    adj[adj > 0] = 1
    np.fill_diagonal(adj,0)

    csr_matrix(adj)

    W1 = mst_graph(X,'euclidian')

    g2 = Graph.Weighted_Adjacency(W1.todense(), mode='undirected', attr='weight', loops=False)

    return g2, W1
