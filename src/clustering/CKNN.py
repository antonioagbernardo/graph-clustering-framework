from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from igraph import Graph

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

def cknn_graph(X, delta, k):
    assert k < X.shape[0]

    D = euclidean_distances(X, X)
    N = D.shape[0]
    np.fill_diagonal(D, 0)
    D_k = np.sort(D)

    adj = np.zeros([N, N])
    adj[np.square(D) < delta * delta * np.dot(D_k[:, k].reshape(-1,1), D_k[:, k].reshape(1,-1))] = 1
    np.fill_diagonal(adj, 0)

    adj = adj + mst_graph(X, 'euclidean')

    adj = 0.5 * (adj + adj.T)

    W = csr_matrix(adj)

    g = Graph.Weighted_Adjacency(W.todense(), mode='undirected', attr='weight', loops=False)

    return g, W
