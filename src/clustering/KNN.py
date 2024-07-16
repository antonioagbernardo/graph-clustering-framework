def KNN(X,k,metric):

  if metric == 'euclidean':
    W = kneighbors_graph(X, 4, mode='distance', metric='euclidean', include_self=False)
    W = 0.5 * (W + W.T)
    g = Graph.Weighted_Adjacency(W.todense(), mode='undirected', attr='weight', loops=False)

    return g

  if metric == 'cosine':

    W = kneighbors_graph(X, 4, mode='distance', metric='cosine', include_self=False)
    W = 0.5 * (W + W.T)
    g = Graph.Weighted_Adjacency(W.todense(), mode='undirected', attr='weight', loops=False)

    return g
  
