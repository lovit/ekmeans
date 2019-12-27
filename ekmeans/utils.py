from datetime import datetime
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances
from sklearn.utils.extmath import safe_sparse_dot
from time import time


def merge_close_clusters(centers, labels, threshold, metric='euclidean'):
    n_clusters, n_terms = centers.shape
    cluster_size = np.bincount(labels[np.where(labels >= 0)[0]], minlength=n_clusters)
    sorted_indices, _ = zip(*sorted(enumerate(cluster_size), key=lambda x:-x[1]))

    groups = _grouping_with_centers(centers, threshold, sorted_indices, metric)
    centers_ = np.dot(np.diag(cluster_size), centers)

    n_groups = len(groups)

    centers_new = []
    labels_new = -1 * np.ones(labels.shape[0], dtype=np.int)
    for g_idx, cluster_indices in enumerate(groups):
        # make new centroid
        centroid_sum = centers_[cluster_indices].sum(axis=0)
        group_size = cluster_size[cluster_indices].sum()
        # TODO: check empty cluster
        if group_size == 0:
            continue
        # update group centroid
        centroid = centroid_sum / group_size
        centers_new.append(centroid)
        # update group labels
        for c_idx in cluster_indices:
            data_indices = np.where(labels == c_idx)[0]
            labels_new[data_indices] = g_idx
    centers_new = np.vstack(centers_new)
    return centers_new, labels_new, groups

def _closest_group(groups, c, pdist, max_dist):
    dist_ = 1
    closest = None
    for g, idxs in enumerate(groups):
        dist = pdist[idxs, c].mean()
        if dist > max_dist:
            continue
        if dist_ > dist:
            dist_ = dist
            closest = g
    return closest

def _grouping_with_centers(centers, max_dist, sorted_indices, metric):
    pdist = pairwise_distances(centers, metric=metric)
    return _grouping_with_pdist(pdist, max_dist, sorted_indices)

def _grouping_with_pdist(pdist, max_dist, sorted_indices):
    groups = [[sorted_indices[0]]]
    for c in sorted_indices[1:]:
        g = _closest_group(groups, c, pdist, max_dist)
        # create new group
        if g is None:
            groups.append([c])
        # assign c to g
        else:
            groups[g].append(c)
    return groups

def filter_infrequents(min_size, labels, centers=None):
    if min_size <= 0:
        return labels, centers
    centers_ = []
    labels_ = -np.ones(labels.shape[0], dtype=labels.dtype)
    label_new = 0
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        if (indices.shape[0] < min_size) or (label == -1):
            continue
        labels_[indices] = label_new
        label_new += 1
        if centers is not None:
            centers_.append(centers[label])
    if centers is None:
        centers_ = None
    else:
        centers_ = np.vstack(centers_)
    return labels_, centers_

def inner_product(X, Y):
    """
    Parameters
    ---------
    X : scipy.sparse.matrix
        shape = (n, p)
    Y : numpy.ndarray or scipy.sparse.matrix
        shape = (p, m)

    It returns
    ----------
    Z : scipy.sparse.matrix
        shape = (n, m)
    """
    return safe_sparse_dot(X, Y, dense_output=False)

def check_convergence(centers, labels, centers_, labels_, tol, metric):
    """
    Check whether k-means is converged or not. 
    If the proportion of reassigned points is smaller than `tol` or
    the difference between `centers` and `centers_` is `tol`,
    it decides to stop training.

    Arguments
    ---------
    centers : numpy.ndarray
        Centroid vectors of current step t    
    labels : numpy.ndarray
        Cluster index, shape = (n_data,)
    centers_ : numpy.ndarray
        Centroid vectors of next step t+1, same shape with `centers`
    labels : numpy.ndarray
        Updated cluster index, shape = (n_data,)
    tol : float
        tolerance parameter
    metric : str
        Distance metric

    Returns
    -------
    diff : float
        Difference between the two centroids
    n_cnahges : int
        Number of re-assigned points
    early_stop : Boolean
        Flag of early-stop
    """
    n_data = labels.shape[0]
    reassign_threshold = n_data * tol
    difference_threshold = tol
    diff = paired_distances(centers, centers_, metric=metric).mean()
    n_changes = np.where(labels != labels_)[0].shape[0]
    early_stop = (diff < difference_threshold) or (n_changes < reassign_threshold)
    return diff, n_changes, early_stop

def verbose_message(i_iter, max_iter, diff, n_changes, n_assigneds,
    n_clusters, inner_dist, early_stop, begin_time, prefix=''):
    """
    Arguments
    ---------
    i_iter : int
        Iteration index
    max_iter : int
        Last iteration index
    diff : float
        Centroid difference
    n_changes : int
        Number of re-assigned points
    n_assigneds : int
        Number of assigned points
    n_clusters : int
        Number of non-empty clusters
    inner_dist : float
        Average inner distance
    early_stop : Boolean
        Flag of early-stop
    begin_time : float
        UNIX time of training begin time

    Returns
    -------
    strf : str
        String formed verbose message
    """
    elapsed_t = time() - begin_time
    remain_t = (max_iter - i_iter) * elapsed_t / i_iter
    ct = as_minute(elapsed_t)
    rt = as_minute(remain_t)
    if rt:
        rt = f'(-{rt})'
    t = f'{ct} {rt}'.strip()
    strf = f'[{prefix}iter: {i_iter}/{max_iter}] #changes: {n_changes}, diff: {diff:.4}, inner: {inner_dist:.4}'
    if n_assigneds > 0:
        strf += f', #assigned: {n_assigneds}'
    if n_clusters > 0:
        strf += f', #clusters: {n_clusters}'
    if t:
        strf += ', time: '+t
    if early_stop:
        strf += f'\nEarly-stop'
    return strf

def as_minute(sec):
    """
    It transforms second to string formed min-sec

    Usage
    -----
        >>> as_minute(153.3)
        $ '2m 33s'

        >>> as_minute(3.21)
        $ '3s'
    """
    m, s = int(sec // 60), int(sec % 60)
    strf = ''
    if m > 0:
        strf += f'{m}m'
    if s > 1:
        strf += ((' ' if strf else '') + f'{s}s')
    return strf

def now():
    now = datetime.now()
    return datetime.strftime(now, '%y-%m-%d_%H-%M-%S')
