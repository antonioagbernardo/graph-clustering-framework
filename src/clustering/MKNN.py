from MST import mst_graph
from sklearn.neighbors import kneighbors_graph
from igraph import Graph

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
