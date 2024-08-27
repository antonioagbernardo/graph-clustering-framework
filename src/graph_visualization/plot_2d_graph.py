import igraph
from igraph import plot
import matplotlib.pyplot as plt
from IPython.display import display, SVG
def plot_2d_graph(g, y=None, X=None, ind=None, colors=None):

  vertex_color = ['white'] * g.vcount()
  if ind:
    for ind in ind:
        vertex_color[ind] = colors[y[ind]]

  if X is not None:
    lyt = igraph.layout.Layout(list(zip(X[:, 0], X[:, 1])))
    lyt.mirror([1])
  else:
    lyt = g.layout("fr")
  visual_style = {}
  visual_style['bbox'] = [350, 350]
  visual_style['vertex_color'] = vertex_color
  visual_style['vertex_size'] = 5
  visual_style['edge_width'] = 0.5
  visual_style['layout'] = lyt
  visual_style['vertex_label_color'] = 'black'
  svg = plot(g, **visual_style)._repr_svg_()[0]
  display(SVG(svg))
