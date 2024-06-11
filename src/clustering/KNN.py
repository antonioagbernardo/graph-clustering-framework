def KNN(X,k):

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  W_knn = kneighbors_graph(X, n_neighbors=k, mode='connectivity', metric='euclidean', include_self=False)

  W_knn_mutual = kneighbors_graph(X, n_neighbors=k, mode='connectivity', metric='euclidean', include_self=False)

  W_mutual_knn = W_knn.multiply(W_knn_mutual)

  W_mutual_knn = 0.5 * (W_mutual_knn + W_mutual_knn.T)

  g = Graph.Weighted_Adjacency(W_mutual_knn.todense(), mode='undirected', attr='weight', loops=False)

  return g
  
