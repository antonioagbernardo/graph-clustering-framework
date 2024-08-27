#Function used to vizualize the data base (Poinst (X,y) in space)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import igraph

def plot_grafico2(X, y, colors, G=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(colors), edgecolor='k')
    if G is not None:
        for edge in G.es():
            i, j = edge.tuple[0], edge.tuple[1]
            line = numpy.array([X[i], X[j]])
            ax.plot(line[:, 0], line[:, 1], color='black')
    plt.show()
