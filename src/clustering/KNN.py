#KNN

#Código
'''
from sklearn.neighbors import kneighbors_graph
from igraph import Graph

def KNN(X,k,metric):

  if metric == 'euclidean':
    W = kneighbors_graph(X, k, mode='distance', metric='euclidean', include_self=False)
    W = 0.5 * (W + W.T)
    g = Graph.Weighted_Adjacency(W.todense(), mode='undirected', attr='weight', loops=False)

    return g, W

  if metric == 'cosine':

    W = kneighbors_graph(X, k, mode='distance', metric='cosine', include_self=False)
    W = 0.5 * (W + W.T)
    g = Graph.Weighted_Adjacency(W.todense(), mode='undirected', attr='weight', loops=False)

    return g, W
 ''' 
# KNN

# Importações
from sklearn.neighbors import kneighbors_graph as knn_graph
from igraph import Graph

def KNN(X, k, metric):
  
    if metric == 'euclidean':
        W = knn_graph(X, k, mode='distance', metric='euclidean', include_self=False)
        W = 0.5 * (W + W.T)
        g = Graph.Weighted_Adjacency(W.todense(), mode='undirected', attr='weight', loops=False)

        return g, W

    elif metric == 'cosine':
        W = knn_graph(X, k, mode='distance', metric='cosine', include_self=False)
        W = 0.5 * (W + W.T)
        g = Graph.Weighted_Adjacency(W.todense(), mode='undirected', attr='weight', loops=False)

        return g, W

    else:
        raise ValueError("Métrica não suportada. Use 'euclidean' ou 'cosine'.")
