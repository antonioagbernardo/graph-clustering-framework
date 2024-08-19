def plot_image(X, y, ann=False, cmap=False):
  plt.figure(1, figsize=(18, 3))
  size = [70] * X.shape[0]
  edgecolor = ['k'] * X.shape[0]
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor=edgecolor, s=size, alpha=0.8, marker='o')
  if ann:
    for i, txt in enumerate(range(0, X.shape[0])):
        plt.annotate(txt, (X[i, 0], X[i, 1]))
  plt.show()
