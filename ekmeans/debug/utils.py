import numpy as np
from soydata.visualize import scatterplot


def draw_scatterplot(X, labels, r=-1, i=-1, show_inline=True):
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
    if r >= 0 and i >= 0:
        title = f'round {r}, iter = {i} (covered {coverage:.4}%)'
    else:
        title = f'{coverage:.4}%'

    data_indices = np.where(labels >= 0)[0]
    noise_indices = np.where(labels == -1)[0]

    p = scatterplot(X[data_indices], labels=labels[data_indices], size=3,
        title=title, show_inline=False, toolbar_location=None)
    p = scatterplot(X[noise_indices], size=3, color='lightgrey',
        p=p, show_inline=show_inline)

    return p

def draw_scatterplots_from_files(log_dir):

    raise NotImplemented
