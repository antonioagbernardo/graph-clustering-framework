def plot_graph(G, X=None, y=None, class_color=None, vertex_color=None, bbox=[200, 150]):

  if vertex_color is not None:
      vertex_color = vertex_color
  else:
      vertex_color = ['white'] * g.vcount()

  if class_color is not None:
    for v in g.vs():
        vertex_color[v.index] = class_color[y[v.index]]

  if X is not None:
    lyt = layout.Layout(list(zip(X[:, 0], X[:, 1])))
    lyt.mirror([1])
  else:
    lyt = G.layout("fr")
  visual_style = {}
  visual_style['bbox'] = bbox
  visual_style['vertex_color'] = vertex_color
  visual_style['vertex_size'] = 5
  visual_style['edge_width'] = 0.2
  visual_style['edge_curved'] = 0.2
  visual_style['layout'] = lyt
  visual_style['vertex_label_color'] = 'black'
  visual_style['vertex_frame_color'] = 'black'
  visual_style['vertex_frame_width'] = 0.5
  svg = plot(G, **visual_style)._repr_svg_()[0]
  display(SVG(svg))
