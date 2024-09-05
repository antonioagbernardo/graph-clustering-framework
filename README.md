# Graph-Clustering-Framework

## Proposal

The main idea is to define a dataset:

![](./pics/fig_conjunto_de_dados_a.png)

Transform the original data into a graph:

![](./pics/fig_conjunto_de_dados_b.jpeg)

Apply a community detection algorithm:

![](./pics/fig_conjunto_de_dados_c.jpeg)

And then project the communities as groups onto the original data:

![](./pics/fig_conjunto_de_dados_d.png)


## Input format
We use `X,y = sklearn.datasets.make_blobs(n_samples, n_features, centers, cluster_std, random_state)`to create an artificial the data base.

We also create a list os colors `colors = ['red', 'green', 'orange', 'yellow', 'blue'......]` the number of colors == centers

You can plot by doing:

~~~python
X, y = dt.make_blobs(n_samples = 1000, n_features = 3, centers = 5, cluster_std = 1.1, random_state = 33)

colors = ['red', 'green', 'orange', 'yellow', 'blue']

plot_grafico3(X, y, colors)

plot_grafico2(X, y, colors)
~~~

![](./pics/3d_artificial_data.png)

![](./pics/2d_artificial_data.png)

## Example

After defining a database (X, y), we can use some of the Graph Construction Algorithm, for example the MKNN (X,k).

~~~python
def components(W):
  g = Graph.Weighted_Adjacency(W.todense(), mode='undirected', attr='weight', loops=False)
  return len(g.connected_components())

def MKNN(X,k):
  W = kneighbors_graph(X, k, mode='distance', metric='euclidean', include_self=False)
  W = W.minimum(W.T)
  if components(W) > 1:
     W = W + mst_graph(X,'euclidean')
  g2 = Graph.Weighted_Adjacency(W.todense(), mode='undirected', attr='weight', loops=False)
  return g2
~~~

Then:

~~~python
g2 = MKNN(X,10)

# Plot

# 2D Graph
plot_2d_graph(g2, y=y, X=X, ind=range(0, g2.vcount()), colors=colors)


#3D Graph
plot_grafico3(X, y, colors, g2)
~~~

![](./pics/2d_mknn_graph.jpeg)

![](./pics/3d_mknn_graph.png)

## How to use the framework

1.Clone the framework repository:
   ~~~python
   git clone https://github.com/Artur-Porto/graph-clustering-framework.git
   ~~~
2.Change to the framework's root directory:
   ~~~python
   cd graph-clustering-framework
   ~~~
3.Install the required dependencies:
  ~~~python
  pip install -r requirements.txt
  ~~~
4.Install the framework locally:
   ~~~python
   pip install .
   ~~~
5.Add Framework Source Directory to Python Path
  ~~~python
  import sys
  import os

  repo_dir = '/content/graph-clustering-framework/src'
  sys.path.insert(0, repo_dir)
~~~

This code adds the graph clustering framework's source directory to the Python path, allowing you to import and use the framework's classes and functions in your current Python session.

6.Import the necessary clustering classes:
  ~~~python
  from clustering.KNN import KNN
  from clustering.Epsilon import Epsilon
  from clustering.SKNN import sknn
  from clustering.MST import MST
  from clustering.MKNN import MKNN
  from clustering.CKNN import cknn_graph
  from clustering.GBLP import GBLP
  from clustering.RMST import rmst_graph
  from clustering.B_matching import b_matching
  ~~~
7.Import the graph visualization functions:
  ~~~python
  from graph_visualization.plot_2d_graph import plot_2d_graph
  from graph_visualization.plot_grafico2 import plot_grafico2
  from graph_visualization.plot_grafico3 import plot_grafico3
  ~~~
8.Get some data(example)
  ~~~python
  import sklearn.datasets as dt

  X, y = dt.make_blobs(n_samples=1000, n_features=3, centers=5, cluster_std=1.1, random_state=33)
  ~~~
9.Use the desired clustering method (e.g., MKNN):
  ~~~python
  g, W = MKNN(X, 5, 'euclidean')
  ~~~
10.Visualize the resulting graph:
~~~python
colors = ['red', 'green', 'orange', 'yellow', 'blue']
plot_2d_graph(g, y=y, X=X, ind=range(0, g.vcount()), colors=colors)
plot_grafico2(X, y, colors, g)
plot_grafico3(X, y, colors, g)
~~~

## Iris Dataset
I also used Iris Dataset to make some graphs and clustering.
### Upload and Visualization

~~~python
Iris_dataset = pd.read_csv('Iris_dataset.csv')
px.scatter_3d(Iris_dataset, "sepal_length", "sepal_width", "petal_length",
             color="species", color_discrete_map = {"Joly": "blue", "Bergeron": "violet", "Coderre":"pink"})
~~~

### Graph constructions
~~~python
X = np.array(Iris_dataset[['sepal_length', 'sepal_width','petal_length', 'petal_width']])

y = Iris_dataset['species']
le = LabelEncoder()
y = le.fit_transform(y)

g, W = MST(X,'cosine')

print('Score Wrong Edges:', score_we(W, y))


# Plot

# Grafo 2D
plot_2d_graph(g, y=y, X=X, ind=range(0, g.vcount()), colors=colors)

plot_grafico2(X,y,colors,g)

#Grafo 3D
plot_grafico3(X, y, colors, g)
~~~
![](./pics/iris1.jpeg)
![](./pics/iris2.png)
![](./pics/iris3.png)

### Clustering example 
~~~python
X = np.array(Iris_dataset[['sepal_length', 'sepal_width', 'petal_width']])

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
~~~
![](./pics/iriskmeans.png)
![](./pics/irisfastgreedy.png)
![](./pics/irisnmi.png)

## NLP
You can also use the graph´s constructions methods to make graphs for NLP analytics. 
This example is a modified version of the code you find in: https://github.com/llSourcell/word_vectors_game_of_thrones-LIVE/blob/master/Thrones2Vec.ipynb
Remember you can analyse any txt text following this steps.

~~~python

## Importações

from __future__ import absolute_import, division, print_function

import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re

import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

## Set up Log

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

## Download NLTK tokenizer models

nltk.download("punkt")
nltk.download("stopwords")

## Load Data(Books)

book_filenames = sorted(glob.glob("/content/*.txt"))

print("Found books:")
book_filenames

## Combine the books into one string

corpus_raw = u""
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()

## Lemmatize and remove stopwords

import nltk
from nltk.corpus import stopwords
import re

# Carrega a lista de stopwords em inglês
stop_words = set(stopwords.words('english'))

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw_sentences = tokenizer.tokenize(corpus_raw)

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    # Remove stopwords
    words = [word.lower() for word in words if word.lower() not in stop_words]
    return words

#sentence where each word is tokenized
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))

print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))

token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))

## Train Word2Vec

#ONCE we have vectors
#step 3 - build model
#3 main tasks that vectors help with
#DISTANCE, SIMILARITY, RANKING

# Dimensionality of the resulting word vectors.
#more dimensions, more computationally expensive to train
#but also more accurate
#more dimensions = more generalized
num_features = 300
# Minimum word count threshold.
min_word_count = 3

# Number of threads to run in parallel.
#more workers, faster we train
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words.
#0 - 1e-5 is good for this
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
#random number generator
#deterministic, good for debugging
seed = 1


thrones2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    vector_size=num_features ,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

thrones2vec.build_vocab(sentences)

print("Word2Vec vocabulary length:", len(thrones2vec.wv.key_to_index))

## Train

thrones2vec.train(sentences, total_examples=thrones2vec.corpus_count, epochs=10)

## Save

if not os.path.exists("trained"):
    os.makedirs("trained")

thrones2vec.save(os.path.join("trained", "thrones2vec.w2v"))

## Load

thrones2vec = w2v.Word2Vec.load(os.path.join("trained", "thrones2vec.w2v"))

## Dimension reduction

tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

all_word_vectors_matrix = thrones2vec.wv.vectors

all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

Points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[thrones2vec.wv.key_to_index[word]])
            for word in thrones2vec.wv.key_to_index
        ]
    ],
    columns=["word", "x", "y"]
)

Points.head()

## General View

sns.set_context("poster")

Points.plot.scatter("x", "y", s=10, figsize=(20, 12))

## Zoom into regions

def plot_region(x_bounds, y_bounds):
    slice = Points[
        (x_bounds[0] <= Points.x) &
        (Points.x <= x_bounds[1]) &
        (y_bounds[0] <= Points.y) &
        (Points.y <= y_bounds[1])
    ]

    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, Point in slice.iterrows():
        ax.text(Point.x + 0.05, Point.y + 0.05, Point.word, fontsize=11)

plot_region(x_bounds=(40, 42), y_bounds=(-5, -1))

## Explore Similarities

thrones2vec.wv.most_similar("stark")

## Graph construction

print(all_word_vectors_matrix_2d)
type(all_word_vectors_matrix_2d)

X = all_word_vectors_matrix_2d
g, W= MST(X,'euclidian')

# Plot
# Grafo 2D
plot_2d_graph(g, X=X)
~~~
![](./pics/nlp1.png)
![](./pics/nlp2.jpeg)
## Kmens x Fastegreedy

I can also evaluate the performance of a clustering algorithm, in the example below I used the Fastgreedy community detection algorithm and the K-NN graph construction method. I used a synthetic dataset with varying levels of noise. 

~~~python

# Variando ruido na base de dados
noise = list(range(2, 11))
nmi_fastgreedy = []
nmi_kmeans =[]
execution_times_kmeans = []
execution_times_fastg = []

for nois in noise:
    start_time_kmeans = time.time()

    # Base artificial
    X, y = dt.make_blobs(n_samples=1000, n_features=3, centers=5, cluster_std=nois, random_state=33)

    # Aplicando Kmeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)

    labels_kmeans = kmeans.labels_

    nmi_kmeans.append(normalized_mutual_info_score(y, labels_kmeans))

    end_time_kmeans = time.time()
    execution_times_kmeans.append(end_time_kmeans - start_time_kmeans)

    start_time_fastg = time.time()

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

    end_time_fastg = time.time()
    execution_times_fastg.append(end_time_fastg - start_time_fastg)

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
~~~
![](./pics/ruidokmeansfastgreedy.png)

We can also, fix the database and variate K values
~~~python
# Base artificial
X, y = dt.make_blobs(n_samples=1000, n_features=3, centers=5, cluster_std=1.1, random_state=33)

# Aplicando Kmeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

labels_kmeans = kmeans.labels_

# Variando K no Fastgreedy
ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
nmi_fastgreedy = []
execution_times = []

for k in ks:
    start_time = time.time()

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

    end_time = time.time()
    execution_times.append(end_time - start_time)

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

~~~
![](./pics/kmeansfastgreedykvar3.png)
.
.
.
![](./pics/kmeansfastgreedykvar10.png)
![](./pics/kmeansfastgreedykvar.png)
## References
>KNN

  >> Eppstein, D., Paterson, M.S. & Yao, F.F. On Nearest-Neighbor Graphs. Discrete Comput Geom 17, 263–282 (1997).

  >>Maier, Markus & Hein, Matthias & Luxburg, Ulrike. (2007). Cluster Identification in Nearest-Neighbor Graphs. Proceedings of the 18th International Confererence on Algorithmic   Learning Theory (ALT 2007), 196-210 (2007). 4754. 196-210. 10.1007/978-3-540-75225-7_18.

>Epsilon

 >> M. Maier, U. von Luxburg, and M. Hein. Influence of graph construction on graph-based clustering measures. The Neural Information Processing Systems, 22:1025–1032, 2009

 >> Partha Pratim Talukdar, "Topics in Graph Construction for Semi-Supervised Learning", Technical Report, University of Pennsylvania. August 2009.

>MkNN

  >>Kohei Ozaki, Masashi Shimbo, Mamoru Komachi, and Yuji Matsumoto. 2011. Using the mutual k-nearest neighbor graphs for semi-supervised classification of natural language data.   In Proceedings of the Fifteenth Conference on Computational Natural Language Learning (CoNLL '11). Association for Computational Linguistics, USA, 154–162.

  >>Gowda, K. Chidananda, and G. Krishna. "Agglomerative clustering using the concept of mutual nearest neighbourhood." Pattern recognition 10.2 (1978): 105-112.

  >>Brito, Maria R., et al. "Connectivity of the mutual k-nearest-neighbor graph in clustering and outlier detection." Statistics & Probability Letters 35.1 (1997): 33-42.

  >>Maier, Markus & Hein, Matthias & Luxburg, Ulrike. (2007). Cluster Identification in Nearest-Neighbor Graphs. Proceedings of the 18th International Confererence on Algorithmic   Learning Theory (ALT 2007), 196-210 (2007). 4754. 196-210. 10.1007/978-3-540-75225-7_18.

>B-matching

  >>Tony Jebara, Jun Wang, and Shih-Fu Chang. 2009. Graph construction and b-matching for semi-supervised learning. In Proceedings of the 26th Annual International Conference on     Machine Learning (ICML '09).

 >> Huang, Bert, and Tony Jebara. "Fast b-matching via sufficient selection belief propagation." Proceedings of the Fourteenth International Conference on Artificial Intelligence   and Statistics. JMLR Workshop and Conference Proceedings, 2011.

 >> Pothen, A., Ferdous, S., & Manne, F. (2019). Approximation algorithms in combinatorial scientific computing. Acta Numerica, 28, 541-633.

 >> Yasuhiro Fujiwara, Atsutoshi Kumagai, Sekitoshi Kanai, Yasutoshi Ida, and Naonori Ueda. 2020. Efficient Algorithm for the b-Matching Graph. In Proceedings of the 26th ACM       SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20).

 >> Khan, Arif and Pothen, Alex and Patwary, Md. Mostofa Ali and Satish, Nadathur and Sundaram, Narayanan and Manne, Fredrik and Halappanavar, Mahantesh and Dubey, Pradeep.       
  (2016). Efficient Approximation Algorithms for Weighted b-Matching. SIAM Journal on Scientific Computing. 38. S593-S619.

>GBLP

  >>Berton, Lilian, Jorge Valverde-Rebaza, and Alneu de Andrade Lopes. "Link prediction in graph construction for supervised and semi-supervised learning." 2015 International Joint Conference on Neural Networks (IJCNN). IEEE, 2015.
  
>S-KNN

  >>Didier A Vega-Oliveros and Lilian Berton and Andre Mantini Eberle and Alneu de Andrade Lopes and Liang Zhao. Regular graph construction for semi-supervised learning, Journal    of Physics: Conference Series, p. 012022, 2014.
  
>MST

  >>R. L. Graham and P. Hell, "On the History of the Minimum Spanning Tree Problem," in Annals of the History of Computing, vol. 7, no. 1, pp. 43-57, Jan.-March 1985, doi:10.1109/MAHC.1985.10011.

  >>Cormen, Thomas H., et al. Introduction to algorithms. MIT press, 2009.

  >>Pettie, Seth and Ramachandran, Vijaya. An Optimal Minimum Spanning Tree Algorithm, Automata, Languages and Programming, p. 49--60, 2000.

>RMST

  >>Beguerisse-Díaz, Mariano, Borislav Vangelov, and Mauricio Barahona. Finding role communities in directed networks using role-based similarity, markov stability and the 
  relaxed minimum spanning tree. 2013 IEEE Global Conference on Signal and Information Processing. IEEE, 2013.
  
>CKNN

  >>Tyrus Berry, Timothy Sauer. Consistent manifold representation for topological data analysis. Foundations of Data Science, 2019, 1 (1) : 1-38. doi: 10.3934/fods.2019001
