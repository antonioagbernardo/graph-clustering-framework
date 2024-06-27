#B-matching

  '''
  X: uma matriz numpy 2D de formato (n_amostras, n_características) representando os pontos de dados.

k: um número inteiro representando o número de vizinhos mais próximos a serem considerados para cada ponto de dados.

b: um número inteiro representando o número de vizinhos para conectar cada ponto de dados no grafo de B-Matching.

A função primeiro calcula o grafo dos k-vizinhos mais próximos para os pontos de dados de entrada usando a função kneighbors_graph do scikit-learn.

Em seguida, cria uma matriz esparsa para armazenar o grafo de B-Matching.

A função então itera sobre cada ponto de dados e seleciona seus b vizinhos mais próximos que também estão entre seus k vizinhos mais próximos. Em seguida, cria arestas entre o ponto de dados e seus b vizinhos mais próximos selecionados no grafo de B-Matching.

Por fim, a função retorna o grafo de B-Matching como uma matriz esparsa.

b_matching assume que b é menor ou igual a k.

  '''

def b_matching(X, k, b, metric='euclidean'):
    # Compute k-nearest neighbors for each node
  knn_graph = kneighbors_graph(X, n_neighbors=k, metric=metric)

    # Create a sparse matrix to store the B-Matching graph
  b_matching_graph = csr_matrix((X.shape[0], X.shape[0]))

    # Iterate over each node
  for i in range(X.shape[0]):
        # Get the k-nearest neighbors for node i
    neighbors = knn_graph[i].nonzero()[1]

        # Select the b-nearest neighbors that are also among the k-nearest neighbors
    b_neighbors = neighbors[:b]

        # Create edges between node i and its b-nearest neighbors
    for j in b_neighbors:
      b_matching_graph[i, j] = 1
      b_matching_graph[j, i] = 1

  g = Graph.Weighted_Adjacency(b_matching_graph.todense(), mode='undirected', attr='weight', loops=False)

  return g
