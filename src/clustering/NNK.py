# Graph Construction from Data using Non Negative Kernel regression (NNK Graphs)
import numpy as np
from scipy.sparse import coo_matrix
import igraph

def create_directed_KNN_mask(D, knn_param=10, D_type='distance'):
    if D_type == 'similarity':
        directed_KNN_mask = np.argpartition(-D, knn_param + 1, axis=1)[:, 0:knn_param + 1]
    else:
        directed_KNN_mask = np.argpartition(D, knn_param + 1, axis=1)[:, 0:knn_param + 1]
    return directed_KNN_mask


def non_negative_qpsolver(A, b, x_init, x_tol, check_tol=-1, epsilon_low=-1, epsilon_high=-1):
    """
    Solves (1/2)x.T A x - b.T x
    :param x_init: Initial value for solution x
    :param x_tol: Smallest allowed non zero value for x_opt. Values below x_tol are made zero
    :param check_tol: Allowed tolerance for stopping criteria. If negative, uses x_tol value
    :param epsilon_high: maximum value of x during optimization
    :param epsilon_low: minimum value of x during optimization
    :return: x_opt, error
    """
    if epsilon_low < 0:
        epsilon_low = x_tol  # np.finfo(float).eps
    if epsilon_high < 0:
        epsilon_high = x_tol
    if check_tol < 0:
        check_tol = x_tol

    n = A.shape[0]
    # A = A + 1e-6 * np.eye(n)
    max_iter = 50 * n
    itr = 0
    # %%
    x_opt = np.reshape(x_init, (n, 1))
    N = 1.0 * (x_opt > (1 - epsilon_high))  # Similarity too close to 1 (nodes collapse)
    if np.sum(N) > 0:
        x_opt = x_opt * N
        return x_opt[:, 0], 0

    # %%
    non_pruned_elements = x_opt > epsilon_low
    check = 1

    while (check > check_tol) and (itr < max_iter):
        x_opt_solver = np.zeros((n, 1))
        x_opt_solver[non_pruned_elements] = cholesky_solver(
            A[non_pruned_elements[:, 0], :][:, non_pruned_elements[:, 0]], b[non_pruned_elements[:, 0]], tol=x_tol)
        x_opt = x_opt_solver
        itr = itr + 1
        N = x_opt < epsilon_low
        if np.sum(N) > 0:
            check = np.max(np.abs(x_opt[N]))
        else:
            check = 0
        non_pruned_elements = np.logical_and(x_opt > epsilon_low, non_pruned_elements)

    x_opt[x_opt < x_tol] = 0
    return x_opt[:, 0], check


def nnk_graph(G, mask, knn_param, reg=1e-6):
    """
    Function to generate NNK graph given similarity matrix and mask
    :param G: Similarity matrix
    :param mask: each row corresponds to the neighbors to be considered for NNK optimization
    :param knn_param: maximum number of neighbors for each node
    :param reg: weights below this threshold are removed (set to 0)
    :return: Adjacency matrix of size num of nodes x num of nodes
    """
    num_of_nodes = G.shape[0]
    neighbor_indices = np.zeros((num_of_nodes, knn_param))
    weight_values = np.zeros((num_of_nodes, knn_param))
    error_values = np.zeros((num_of_nodes, knn_param))

    for node_i in range(num_of_nodes):
        non_zero_index = np.array(mask[node_i, :])
        non_zero_index = np.delete(non_zero_index, np.where(non_zero_index == node_i))
        G_i = G[np.ix_(non_zero_index, non_zero_index)]
        g_i = G[non_zero_index, node_i]
        x_opt, check = non_negative_qpsolver(G_i, g_i, g_i, reg)
        error_values[node_i, :] = G[node_i, node_i] - 2 * np.dot(x_opt, g_i) + np.dot(x_opt, np.dot(G_i, x_opt))
        try:
            weight_values[node_i, :] = x_opt
        except:
            weight_values[node_i, :] = 1
        try:
            neighbor_indices[node_i, :] = non_zero_index
        except:
            neighbor_indices[node_i, :] = 1

    row_indices = np.expand_dims(np.arange(0, num_of_nodes), 1)
    row_indices = np.tile(row_indices, [1, knn_param])
    adjacency = sparse.coo_matrix((weight_values.ravel(), (row_indices.ravel(), neighbor_indices.ravel())),
                                  shape=(num_of_nodes, num_of_nodes))
    error = sparse.coo_matrix((error_values.ravel(), (row_indices.ravel(), neighbor_indices.ravel())),
                                  shape=(num_of_nodes, num_of_nodes))
    # Alternate way of doing: error_index = sparse.find(error > error.T); adjacency[error_index[0], error_index[
    # 1]] = 0
    adjacency = adjacency.multiply(error < error.T)
    adjacency = adjacency.maximum(adjacency.T)

    adjacency = adjacency + mst_graph(X, 'Euclidean')

    g = Graph.Weighted_Adjacency(adjacency.todense(), mode='undirected', attr='weight', loops=False)

    return g, adjacency
