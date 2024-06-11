def KNN(X,k):

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  W_knn = kneighbors_graph(X, n_neighbors=k, mode='connectivity', metric='euclidean', include_self=False)

  W_knn_mutual = kneighbors_graph(X, n_neighbors=k, mode='connectivity', metric='euclidean', include_self=False)

  W_mutual_knn = W_knn.multiply(W_knn_mutual)

  W_mutual_knn = 0.5 * (W_mutual_knn + W_mutual_knn.T)

  g = Graph.Weighted_Adjacency(W_mutual_knn.todense(), mode='undirected', attr='weight', loops=False)

  return g




'''
1. **Data Standardization:**

   - `StandardScaler()` is a class from scikit-learn used to standardize features by removing the mean and scaling to unit variance.
   - `scaler.fit_transform(X)` fits the scaler to the data `X` and then transforms this data.

2. **Definition of the Number of Neighbors:**

   - Defines the number of neighbors for the Mutual KNN algorithm.

3. **Calculation of the Nearest Neighbors Connectivity Matrix:**

   - `kneighbors_graph` from scikit-learn is used to calculate the connectivity matrix of the nearest neighbors.
   - `mode='connectivity'` indicates that we are interested in the binary connectivity matrix.
   - `metric='euclidean'` specifies that the Euclidean distance is used to measure proximity.
   - `include_self=False` means that the main diagonal (self-elements) is not included.

4. **Calculation of the Reciprocal (Mutual) Nearest Neighbors Connectivity Matrix:**

   - Calculates the reciprocal (mutual) nearest neighbors connectivity matrix, following the same parameters as point 3.

5. **Calculation of the Mutual KNN Connectivity Matrix:**

   - Multiplies the two connectivity matrices element-wise.

6. **Symmetric Connectivity Matrix:**

   - Makes the connectivity matrix symmetric by adding it to its transpose and dividing by 2.

7. **Creation of the Weighted Graph using iGraph:**

   - Converts the weighted connectivity matrix to an undirected weighted graph using the iGraph library.
'''
