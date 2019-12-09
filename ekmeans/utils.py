import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils.extmath import safe_sparse_dot


def merge_close_clusters(centers, labels, threshold):
    n_clusters, n_terms = centers.shape
    cluster_size = np.bincount(labels[np.where(labels >= 0)[0]], minlength=n_clusters)
    sorted_indices, _ = zip(*sorted(enumerate(cluster_size), key=lambda x:-x[1]))

    groups = _grouping_with_centers(centers, threshold, sorted_indices)
    centers_ = np.dot(np.diag(cluster_size), centers)

    n_groups = len(groups)
    group_centers = np.zeros((n_groups, n_terms))
    for g, idxs in enumerate(groups):
        sum_ = centers_[idxs].sum(axis=0)
        mean = sum_ / cluster_size[idxs].sum()
        group_centers[g] = mean

    labels_ = -1 * np.ones(labels.shape[0], dtype=np.int)
    for m_idx, c_idxs in enumerate(groups):
        for c_idx in c_idxs:
            idxs = np.where(labels == c_idx)[0]
            labels_[idxs] = m_idx
    return group_centers, labels_, groups

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

def _grouping_with_centers(centers, max_dist, sorted_indices):
    pdist = pairwise_distances(centers, metric='cosine')
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

def print_status(i_round, i_iter, labels, n_changed):
    "deprecated"
    n_samples = labels.shape[0]
    n_clusters = np.where(np.unique(labels) >= 0)[0].shape[0]
    n_assigned = np.where(labels >= 0)[0].shape[0]
    print('[round #{}, iter #{}] clusters = {}, changes = {}, assigned = {} / {}'.format(
        i_round, i_iter, n_clusters, n_changed, n_assigned, n_samples))

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

def verbose_message(i_iter, max_iter, diff, n_changes, n_assigneds, inner_dist, early_stop, begin_time):
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
    comsumed_t = time() - begin_time
    remain_t = (max_iter - i_iter) * comsumed_t / i_iter
    ct = as_minute(comsumed_t)
    rt = as_minute(remain_t)
    if rt:
        rt = f'(-{rt})'
    t = f'{ct} {rt}'.strip()
    strf = f'[iter: {i_iter}/{max_iter}] #changes: {n_changes}, diff: {diff:.4}, inner: {inner_dist:.4}'
    if n_assigned > 0:
        strf += f'#assigned: {n_assigneds}'
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
