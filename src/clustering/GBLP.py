# Graph-based on Link Prediction (GBLP) using MST/RMST and WCN
from MST import mst_graph


class Predictor(object):

    def __init__(self, W, h=2, eligible=None, excluded=None):

        sources, targets = W.nonzero()
        edgelist = zip(sources.tolist(), targets.tolist())
        weights = np.array(W[sources, targets])[0]
        self.G = Graph(W.shape[0], edgelist, edge_attrs={"weight": weights})
        self.W = W
        self.adjlist = list(map(set, self.G.get_adjlist()))
        self.eligible_attr = eligible
        self.excluded = [] if excluded is None else excluded
        self.h = h

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def add_edges(self):
        raise NotImplementedError

    def neighbourhood(self, v):
        """Get k-neighbourhood of node n"""

        if self.h == 1:
            return self.adjlist[v]
        neighbors = self.G.neighborhood(vertices=v, order=self.h)
        return neighbors

    def eligible(self, u, v):
        """Check if link between nodes u and v is eligible
        Eligibility allows us to ignore some nodes/links for link prediction.
        """

        return self.eligible_node(u) and self.eligible_node(v) and u != v

    def eligible_node(self, v):
        """Check if node v is eligible
        Eligibility allows us to ignore some nodes/links for link prediction.
        """

        if self.eligible_attr is None:
            return True
        return self.G.vs[v][self.eligible_attr]

    def eligible_nodes(self):
        """Get list of eligible nodes
        Eligibility allows us to ignore some nodes/links for link prediction.
        """

        return [v for v in self.G if self.eligible_node(v)]

    def likely_pairs(self):
        """
        Yield node pairs from the same neighbourhood
        Arguments
        ---------
        k : int
            size of the neighbourhood (e.g., if k = 2, the neighbourhood
            consists of all nodes that are two links away)
        """

        for a in range(0, self.G.vcount()):
            if not self.eligible_node(a):
                continue
            for b in self.neighbourhood(a):
                if not self.eligible_node(b):
                    continue
                yield (a, b)


class WeightedCommonNeighbours(Predictor):

    def __init__(self, *args, **kwargs):
        self.top_links = kwargs.pop('top_links')
        self.links = None
        super().__init__(*args, **kwargs)

    def predict(self):

        self.links = dict()
        for i, j in self.likely_pairs():
            if self.G[i, j] or i == j:
                continue
            _sum = 0.0
            cn = self.adjlist[i].intersection(self.adjlist[j])
            for z in cn:
                _sum += (self.G[i, z] + self.G[j, z]) / 2
            flag = self.links.get((i, j), False)
            if i < j and not flag:
                self.links[(i, j)] = _sum
            elif not flag:
                self.links[(j, i)] = _sum
        return self.links

    def add_edges(self):

        self.top_links = self.W.count_nonzero() * self.top_links
        for itr, vertices in enumerate(sorted(self.links, key=self.links.get, reverse=True)):
            i, j = vertices
            if itr > self.top_links:
                break
            self.W[i, j] = self.links[(i, j)]

        return self.W


def GBLP (X) :
  W = mst_graph(X, 'euclidean')
  model = WeightedCommonNeighbours(W, h=2, top_links=0.1)
  links = model.predict()
  top_links = W.count_nonzero() * 0.1
  for itr, vertices in enumerate(sorted(links, key=links.get, reverse=True)):
    i, j = vertices
    if itr > top_links:
      break
    W[i, j] = links[(i, j)]

  W = 0.5 * (W + W.T) #make symmetric

  g = Graph.Weighted_Adjacency(W.todense(), mode='undirected', attr='weight', loops=False)

  return g, W


