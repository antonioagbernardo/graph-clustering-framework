# Kmeans x Fastgreedy

## KNN

from sklearn.cluster import KMeans
from igraph import Graph

from clustering import *

#Base artificial
X, y = dt.make_blobs(n_samples = 1000, n_features = 3, centers = 5, cluster_std = 1.1, random_state = 33)


#Aplicando Kmeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

labels_kmeans = kmeans.labels_

#Aplicando Fastgreedy
g, W = KNN(X, k=5, metric='euclidean')

# Convert W to a dense matrix
W_dense = W.toarray()

# Get the edge list from the graph
edges = g.get_edgelist()

# Create a weights vector with the same length as the number of edges
edge_weights = np.zeros(len(edges))

# Assign weights to each edge using the dense matrix
for i, edge in enumerate(edges):
    edge_weights[i] = W_dense[edge[0], edge[1]]

# Now you can pass edge_weights to the community_fastgreedy method
fastgreedy = Graph.community_fastgreedy(g, weights=edge_weights)
labels_fastgreedy = fastgreedy.as_clustering().membership


#Comparando
plt.scatter(X[:, 0], X[:, 1], c=labels_kmeans)
plt.title('Kmeans')
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=labels_fastgreedy)
plt.title('Fastgreedy + KNN')
plt.show()


#NMI (Normalized Mutual Information)
labels_kmeans = kmeans.labels_
labels_fastgreedy = fastgreedy.as_clustering().membership

nmi_kmeans = normalized_mutual_info_score(y, labels_kmeans)
nmi_fastgreedy = normalized_mutual_info_score(y, labels_fastgreedy)

print()
print("NMI K-Means:", nmi_kmeans)
print("NMI Fastgreedy + KNN:", nmi_fastgreedy)
print()

plt.bar(["K-Means", "Fastgreedy + KNN"], [nmi_kmeans, nmi_fastgreedy], color=["blue", "green"])
plt.xlabel("Algoritmo")
plt.ylabel("NMI")
plt.title("Pontuação de NMI")
plt.show()

### Variando o valor de K

# Base artificial
X, y = dt.make_blobs(n_samples=1000, n_features=3, centers=5, cluster_std=1.1, random_state=33)

# Aplicando Kmeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

labels_kmeans = kmeans.labels_

# Variando K no Fastgreedy
ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
nmi_fastgreedy = []

for k in ks:
    g, W = KNN(X, k=k, metric='euclidean')
    # Convert W to a dense matrix
    W_dense = W.toarray()
    # Get the edge list from the graph
    edges = g.get_edgelist()
    # Create a weights vector with the same length as the number of edges
    edge_weights = np.zeros(len(edges))
    # Assign weights to each edge using the dense matrix
    for i, edge in enumerate(edges):
        edge_weights[i] = W_dense[edge[0], edge[1]]
    # Now you can pass edge_weights to the community_fastgreedy method
    fastgreedy = Graph.community_fastgreedy(g, weights=edge_weights)
    labels_fastgreedy = fastgreedy.as_clustering().membership
    nmi_fastgreedy.append(normalized_mutual_info_score(y, labels_fastgreedy))

    # Visualizando Fastgreedy para cada K
    plt.scatter(X[:, 0], X[:, 1], c=labels_fastgreedy)
    plt.title(f'Fastgreedy + KNN (K={k})')
    plt.show()

# Plotando os resultados
plt.plot(ks, nmi_fastgreedy, label='Fastgreedy + KNN')
plt.axhline(normalized_mutual_info_score(y, labels_kmeans), color='r', linestyle='--', label='K-Means')
plt.xlabel('K')
plt.ylabel('NMI')
plt.title('Desempenho do Fastgreedy + KNN em função de K')
plt.legend()
plt.show()

### Variando valor de K mas usando Coceno

# Base artificial
X, y = dt.make_blobs(n_samples=1000, n_features=3, centers=5, cluster_std=1.1, random_state=33)

# Aplicando Kmeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

labels_kmeans = kmeans.labels_

# Variando K no Fastgreedy
ks = list(range(2, 41))
nmi_fastgreedy = []

for k in ks:
    g, W = KNN(X, k=k, metric='cosine')
    # Convert W to a dense matrix
    W_dense = W.toarray()
    # Get the edge list from the graph
    edges = g.get_edgelist()
    # Create a weights vector with the same length as the number of edges
    edge_weights = np.zeros(len(edges))
    # Assign weights to each edge using the dense matrix
    for i, edge in enumerate(edges):
        edge_weights[i] = W_dense[edge[0], edge[1]]
    # Now you can pass edge_weights to the community_fastgreedy method
    fastgreedy = Graph.community_fastgreedy(g, weights=edge_weights)
    labels_fastgreedy = fastgreedy.as_clustering().membership
    nmi_fastgreedy.append(normalized_mutual_info_score(y, labels_fastgreedy))

    # Visualizando Fastgreedy para cada K
    plt.scatter(X[:, 0], X[:, 1], c=labels_fastgreedy)
    plt.title(f'Fastgreedy + KNN (K={k})')
    plt.show()

# Plotando os resultados
plt.plot(ks, nmi_fastgreedy, label='Fastgreedy + KNN')
plt.axhline(normalized_mutual_info_score(y, labels_kmeans), color='r', linestyle='--', label='K-Means')
plt.xlabel('K')
plt.ylabel('NMI')
plt.title('Desempenho do Fastgreedy + KNN em função de K')
plt.legend()
plt.show()

### Variando o ruído

# Variando ruido na base de dados
noise = list(range(2, 11))
nmi_fastgreedy = []
nmi_kmeans =[]

for nois in noise:
    # Base artificial
    X, y = dt.make_blobs(n_samples=1000, n_features=3, centers=5, cluster_std=nois, random_state=33)

    # Aplicando Kmeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)

    labels_kmeans = kmeans.labels_

    nmi_kmeans.append(normalized_mutual_info_score(y, labels_kmeans))

    g, W = KNN(X, k=10, metric='euclidean')
    # Convert W to a dense matrix
    W_dense = W.toarray()
    # Get the edge list from the graph
    edges = g.get_edgelist()
    # Create a weights vector with the same length as the number of edges
    edge_weights = np.zeros(len(edges))
    # Assign weights to each edge using the dense matrix
    for i, edge in enumerate(edges):
        edge_weights[i] = W_dense[edge[0], edge[1]]
    # Now you can pass edge_weights to the community_fastgreedy method
    fastgreedy = Graph.community_fastgreedy(g, weights=edge_weights)
    labels_fastgreedy = fastgreedy.as_clustering().membership
    nmi_fastgreedy.append(normalized_mutual_info_score(y, labels_fastgreedy))

    # Visualizando Fastgreedy para cada Ruido
    plt.scatter(X[:, 0], X[:, 1], c=labels_fastgreedy)
    plt.title(f'Fastgreedy + KNN (noise={nois})')
    plt.show()

# Plotando os resultados
plt.plot(noise, nmi_fastgreedy, label='Fastgreedy + KNN')
plt.plot(noise, nmi_kmeans, label='K-Means')
plt.xlabel('Ruído')
plt.ylabel('NMI')
plt.title('Desempenho do Fastgreedy+KNN e K-Means em função do ruído')
plt.legend()
plt.show()

## MKNN

from sklearn.cluster import KMeans
from igraph import Graph

#Base artificial
X, y = dt.make_blobs(n_samples = 1000, n_features = 3, centers = 5, cluster_std = 1.1, random_state = 33)


#Aplicando Kmeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

labels_kmeans = kmeans.labels_

#Aplicando Fastgreedy
g, W = MKNN(X, k=5, metric='euclidean')

# Convert W to a dense matrix
W_dense = W.toarray()

# Get the edge list from the graph
edges = g.get_edgelist()

# Create a weights vector with the same length as the number of edges
edge_weights = np.zeros(len(edges))

# Assign weights to each edge using the dense matrix
for i, edge in enumerate(edges):
    edge_weights[i] = W_dense[edge[0], edge[1]]

# Now you can pass edge_weights to the community_fastgreedy method
fastgreedy = Graph.community_fastgreedy(g, weights=edge_weights)
labels_fastgreedy = fastgreedy.as_clustering().membership


#Comparando
plt.scatter(X[:, 0], X[:, 1], c=labels_kmeans)
plt.title('Kmeans')
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=labels_fastgreedy)
plt.title('Fastgreedy + MKNN')
plt.show()


#NMI (Normalized Mutual Information)
labels_kmeans = kmeans.labels_
labels_fastgreedy = fastgreedy.as_clustering().membership

nmi_kmeans = normalized_mutual_info_score(y, labels_kmeans)
nmi_fastgreedy = normalized_mutual_info_score(y, labels_fastgreedy)

print()
print("NMI K-Means:", nmi_kmeans)
print("NMI Fastgreedy + MKNN:", nmi_fastgreedy)
print()

plt.bar(["K-Means", "Fastgreedy + MKNN"], [nmi_kmeans, nmi_fastgreedy], color=["blue", "green"])
plt.xlabel("Algoritmo")
plt.ylabel("NMI")
plt.title("Pontuação de NMI")
plt.show()

## Variando o valor de K

# Base artificial
X, y = dt.make_blobs(n_samples=1000, n_features=3, centers=5, cluster_std=1.1, random_state=33)

# Aplicando Kmeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

labels_kmeans = kmeans.labels_

# Variando K no Fastgreedy
ks = list(range(2, 41))
nmi_fastgreedy = []

for k in ks:
    g, W = MKNN(X, k=k, metric='euclidean')
    # Convert W to a dense matrix
    W_dense = W.toarray()
    # Get the edge list from the graph
    edges = g.get_edgelist()
    # Create a weights vector with the same length as the number of edges
    edge_weights = np.zeros(len(edges))
    # Assign weights to each edge using the dense matrix
    for i, edge in enumerate(edges):
        edge_weights[i] = W_dense[edge[0], edge[1]]
    # Now you can pass edge_weights to the community_fastgreedy method
    fastgreedy = Graph.community_fastgreedy(g, weights=edge_weights)
    labels_fastgreedy = fastgreedy.as_clustering().membership
    nmi_fastgreedy.append(normalized_mutual_info_score(y, labels_fastgreedy))

    # Visualizando Fastgreedy para cada K
    plt.scatter(X[:, 0], X[:, 1], c=labels_fastgreedy)
    plt.title(f'Fastgreedy + MKNN (K={k})')
    plt.show()

# Plotando os resultados
plt.plot(ks, nmi_fastgreedy, label='Fastgreedy + MKNN')
plt.axhline(normalized_mutual_info_score(y, labels_kmeans), color='r', linestyle='--', label='K-Means')
plt.xlabel('K')
plt.ylabel('NMI')
plt.title('Desempenho do Fastgreedy + MKNN em função de K')
plt.legend()
plt.show()

## Variando o valor de K mas usando coceno

# Base artificial
X, y = dt.make_blobs(n_samples=1000, n_features=3, centers=5, cluster_std=1.1, random_state=33)

# Aplicando Kmeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

labels_kmeans = kmeans.labels_

# Variando K no Fastgreedy
ks = list(range(2, 41))
nmi_fastgreedy = []

for k in ks:
    g, W = MKNN(X, k=k, metric='cosine')
    # Convert W to a dense matrix
    W_dense = W.toarray()
    # Get the edge list from the graph
    edges = g.get_edgelist()
    # Create a weights vector with the same length as the number of edges
    edge_weights = np.zeros(len(edges))
    # Assign weights to each edge using the dense matrix
    for i, edge in enumerate(edges):
        edge_weights[i] = W_dense[edge[0], edge[1]]
    # Now you can pass edge_weights to the community_fastgreedy method
    fastgreedy = Graph.community_fastgreedy(g, weights=edge_weights)
    labels_fastgreedy = fastgreedy.as_clustering().membership
    nmi_fastgreedy.append(normalized_mutual_info_score(y, labels_fastgreedy))

    # Visualizando Fastgreedy para cada K
    plt.scatter(X[:, 0], X[:, 1], c=labels_fastgreedy)
    plt.title(f'Fastgreedy + MKNN (K={k})')
    plt.show()

# Plotando os resultados
plt.plot(ks, nmi_fastgreedy, label='Fastgreedy + MKNN')
plt.axhline(normalized_mutual_info_score(y, labels_kmeans), color='r', linestyle='--', label='K-Means')
plt.xlabel('K')
plt.ylabel('NMI')
plt.title('Desempenho do Fastgreedy + MKNN em função de K')
plt.legend()
plt.show()
