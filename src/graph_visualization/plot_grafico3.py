import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import igraph
from mpl_toolkits.mplot3d import Axes3D

def plot_grafico3(X, y, colors, G=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1],  X[:, 2], c=y, cmap=ListedColormap(colors), edgecolor='k')
    if G is not None:
        for edge in G.es():
            i, j = edge.tuple[0], edge.tuple[1]
            line = np.array([X[i], X[j]])
            ax.plot(line[:, 0], line[:, 1],  line[:, 2], color='black')
    plt.show()
