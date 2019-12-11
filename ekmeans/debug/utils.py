import numpy as np
import os

from bokeh.io import export_png, save
from glob import glob
from ipywidgets import interactive, IntSlider
from IPython.display import display, Image
from soydata.visualize import scatterplot


def draw_scatterplot(X, labels, title=None, toolbar_location=None, n_labels=-1, show_inline=True):

    """
    It draws scatterplot. The color of not assigned points is `rightgrey`.
    And each cluster is painted with a different color.

    Arguments
    ---------
    X : numpy.ndarray
        Shape = (n_data,2)
    labels : numpy.ndarray
        Cluster index, shape = (n_data,)
    r : int
        Round index
    i : int
        Iteration index
    show_inline : Boolean
        If True, it shows scatterplot, then returns figure
        Else, it just returns the figure

    Returns
    -------
    p : bokeh.plotting.figure.Figure
        Scatterplot figure
    """
    coverage = 100 * np.where(labels >= 0)[0].shape[0] / labels.shape[0]
    if title is not None:
        title = f'{title} (covered {coverage:.4}%)'
    else:
        title = f'{coverage:.4}%'

    data_indices = np.where(labels >= 0)[0]
    noise_indices = np.where(labels == -1)[0]

    p = scatterplot(X[data_indices], labels=labels[data_indices], size=3,
        title=title, show_inline=False, toolbar_location=toolbar_location, n_labels=n_labels)
    p = scatterplot(X[noise_indices], size=3, color='lightgrey',
        p=p, show_inline=show_inline)

    return p

def draw_scatterplots_batch(X, log_dir, n_labels=-1,
    figure_dir=None, height=600, width=600, figure_type='png'):

    if figure_type == 'png':
        toolbar_location = None
        export = export_png
    elif figure_type == 'html':
        toolbar_location = 'right'
        export = save
    else:
        raise ValueError("`figure_type` must be one of ['png', 'html']")

    figures = []
    figpaths = []
    paths = glob(f'{log_dir}/*labels.txt')
    paths = sort_paths(paths)

    for path in paths:
        filename = path.split("/")[-1][:-4]
        labels = load_label(path)
        prefix = path_to_prefix(path)
        fig = draw_scatterplot(X, labels, prefix, n_labels=n_labels, show_inline=False)
        fig.height = height
        fig.width = width
        figures.append(fig)
        figpaths.append(f'{figure_dir}/{filename}.{figure_type}')

    if figure_dir is not None:
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        for fig, figpath in zip(figures, figpaths):
            export(fig, figpath)
            print(f'saved {figpath}')

    return figures

def path_to_prefix(filename):
    filename = filename.split('/')[-1]
    if filename[:5] == 'round':
        r, i, _ = filename.split('/')[-1].split('_')
        r = int(r[5:])
        i = int(i[4:])
        return f'round {r}, iter {i}'
    return filename.rsplit('.txt')[0]

def sort_paths(paths):
    paths_sub0 = [p for p in paths if p.split('/')[-1][:5] == 'round']
    paths_sub1 = [p for p in paths if p.split('/')[-1][:5] != 'round']

    def key(path):
        r, i, _ = path.split('/')[-1].split('_')
        return int(r[5:]), int(i[4:])

    paths_sub0 = sorted(paths_sub0, key=key)
    paths_sub1 = sorted(paths_sub1)
    return paths_sub0 + paths_sub1

def load_label(path):
    return np.loadtxt(path)

def prepare_ipython_image_slider(image_dir):
    paths = sort_paths(glob(f'{image_dir}/*.png'))

    def select(index):
        display(Image(paths[index]))

    slider = IntSlider(min=0, max=len(paths)-1, step=1, value=0)
    widget = interactive(select, index=slider)
    return widget
