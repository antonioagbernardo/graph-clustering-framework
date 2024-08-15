#EPSILON

#Código

def Epsilon(X,e):
  W = radius_neighbors_graph(X, e, mode='distance', metric='euclidean', include_self=False)
  W = 0.5 * (W + W.T) # make connectivity symmetric
  g = Graph.Weighted_Adjacency(W.todense(), mode='undirected', attr='weight', loops=False)
  return g, W
