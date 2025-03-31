import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable

def initialize_figure(palette='bwr'):
  '''Initialize a subplots figure with some default values.'''
  sns.set_style('whitegrid', {'axis.grid': False})
  sns.set_palette(palette)
  fig, ax = plt.subplots()
  ax.axis('off')
  return fig, ax
  
def render_line(data, num_tokens, tokens, scores, scores_2, y_pos_pct, fig, ax, fontsize=16, name='monospace', alpha=0.6,
  offset_factor=1.29):
  '''Draw a single line of text.'''
  
  #cmap = ListedColormap(sns.color_palette())
  #print(cmap)
  cmap = ListedColormap(sns.color_palette("light:#52DE0C"))
  #print(cmap)
  r = fig.canvas.get_renderer()
  #print(data)
  fig_size_pix = fig.get_size_inches()[0] * fig.dpi
  #num_tokens = len(data[0])
  x_pos_pct = 0.
  x_pos_pix = 0.
  idx = 0
  
  while (x_pos_pix < fig_size_pix) and (idx < num_tokens) and scores:
    props = dict(alpha=alpha, facecolor=cmap(scores[0]), pad=0., lw=0.)
    if scores_2[0] == 0:
        #print("get score 0")
        #print(tokens[0])
        props = dict(alpha=alpha,facecolor="White", pad=0., lw=0.)
    t = ax.text(x_pos_pct, y_pos_pct, tokens[0] + ' ', bbox=props, horizontalalignment='left',
                fontsize=fontsize, verticalalignment='bottom', name=name)
    x_pos_pix += (offset_factor * t.get_window_extent(renderer=r).width)
    x_pos_pct = x_pos_pix / fig_size_pix
    #data.pop(0)
    scores.pop(0)
    tokens.pop(0)
    scores_2.pop(0)
    idx += 1
    
  return fig, ax, t, tokens, scores

def render(tokens, scores, sigma=10, fontsize=16, name='monospace', alpha=0.6, offset_factor=1.29, 
  typical_wpl=10, palette='bwr', filesave='test.png'):
  '''Render highlighted text.

  Args:
    tokens: list of tokens to render
    scores: list or 1d-array of scores between 0. and 1., one per token
    sigma: sigmoid parameter for squishing scores towards 0/1. Higher values make the
      function more step-like.
    fontsize:  text font size
    name:  font family name
    alpha: text box transparency between 0. and 1.
    offset_factor: fudge factor for aligning text boxes.
    typical_wpl: desired (approximate) number of words per line
    palette: Seaborn or matplotlib color palette

  Returns:
    fig, ax: figure and axis handles
  '''

  fig, ax = initialize_figure(palette)
  scores_2 = list(scores)
  # Transform the scores using sigmoid function.
  scores = 1. / (1. + np.exp( -2 * sigma * (scores - 0.5)))

  # Get the dimensions needed to align text.
  props = dict(facecolor='white', pad=0.0, lw=0.)
  t_tmp = ax.text(0., 0., 'typical?', bbox=props, fontsize=fontsize, name=name, alpha=0.)
  extent = t_tmp.get_window_extent(renderer=fig.canvas.get_renderer())
  
  # Set figure dimensions.
  fig_height_pix = 1. * extent.height * (1. * len(tokens) / typical_wpl)
  fig_width_pix = 1. * extent.width * typical_wpl 
  fig.set_size_inches(fig_width_pix / fig.dpi, fig_height_pix / fig.dpi)

  # Plot text lines.
  data = zip(tokens, scores)
  num_tokens = len(tokens)
  tokens = list(tokens)
  scores = list(scores)
  #print(scores)
  #print(tokens)
  line_num = 0
   
  while len(scores)>0:
    y_pos_pct = 1. - (offset_factor * line_num * extent.height / fig_height_pix)
    fig, ax, t, tokens, scores = render_line(data, num_tokens, tokens, scores, scores_2, y_pos_pct, fig, ax, fontsize=fontsize,
      name=name, alpha=alpha, offset_factor=offset_factor)
    line_num += 1
  #plt.savefig(filesave)
  fig, (ax1) = plt.subplots(1, 1, figsize=(12, 0.2))
  plt.colorbar(ScalarMappable(cmap=ListedColormap(sns.color_palette("light:#52DE0C"))), label='Features (> more important)', orientation='horizontal', cax=ax1) 
  
  return fig, ax
