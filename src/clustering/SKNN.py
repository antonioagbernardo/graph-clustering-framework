from sklearn.metrics.pairwise import euclidean_distances
#import igraph
from igraph import Graph
from scipy import spatial

def sknn(X, kmax, mode='distance', metric='euclidean'):
    """Summary or Description of the Function

    Parameters:
    argument1 (int): Description of arg1

    Returns:
    int:Returning value

   """
    # Compute distance matrix
    D = euclidean_distances(X, X)

    # Compute closeness centrality
    closeness = []
    for i in range(0, D.shape[0]):
        closeness.append(1 / sum(D[i,:]))
    closeness = list(zip(list(range(0, D.shape[0])), closeness))
    closeness.sort(key=lambda x:x[1], reverse=False)
    g1 = Graph(D.shape[0])
    kdtree = spatial.KDTree(X)
    weights = []

    # Sequential k-NN based on closseness centrality
    k1 = 3
    while True:
        for i, value in closeness:
            obj_knn = kdtree.query(X[i,:], k=(kmax + 1))
            for key, j in enumerate(obj_knn[1]):
                if i == j or g1.degree(j) >= k1 or D[i, j] <= 0.0:
                    continue
                if g1.are_connected(i, j) == False:
                    g1.add_edge(i, j)
                    weights.append(D[i, j])
                break

        k1 += 1
        degree = np.array(g1.degree())
        indices = np.where(degree < kmax)[0]
        if len(indices) == 0 or k1 > kmax:
            break

    # Constrained k-NN by kmax
    for i in indices:
        obj_knn = kdtree.query(X[i,:], k=(kmax + 1))
        for key, j in enumerate(obj_knn[1]):
            if i == j or g1.degree(j) >= kmax or D[i, j] <= 0.0:
                continue
            if g1.are_connected(i, j) == False:
                g1.add_edge(i, j)
                weights.append(D[i, j])
            if g1.degree(i) >= kmax:
                break

    g1.es['weight'] = weights
    return g1
