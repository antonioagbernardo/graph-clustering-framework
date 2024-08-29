from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from igraph import Graph
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

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
  
def components(W):
  g = Graph.Weighted_Adjacency(W.todense(), mode='undirected', attr='weight', loops=False)
  return len(g.connected_components())


def MKNN(X,k,metric):
  if metric == 'euclidean':
    W = kneighbors_graph(X, k, mode='distance', metric='euclidean', include_self=False)
    W = W.minimum(W.T)
    if components(W) > 1:
      W = W + mst_graph(X,'euclidean')
    g2 = Graph.Weighted_Adjacency(W.todense(), mode='undirected', attr='weight', loops=False)
    return g2, W
  else:
    W = kneighbors_graph(X, k, mode='distance', metric='cosine', include_self=False)
    W = W.minimum(W.T)
    if components(W) > 1:
      W = W + mst_graph(X,'cosine')
    g2 = Graph.Weighted_Adjacency(W.todense(), mode='undirected', attr='weight', loops=False)
    return g2, W
