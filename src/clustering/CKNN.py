from MST import mst_graph
from sklearn.metrics import euclidean_distances
import numpy as np
from scipy.sparse import csr_matrix
from igraph import Graph

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
