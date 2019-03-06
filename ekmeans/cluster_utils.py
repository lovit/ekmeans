import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils.extmath import safe_sparse_dot


def compatify(centers, labels):
    centers_ = []
    labels_ = -1 * np.ones(labels.shape[0], dtype=np.int)
    for cluster in range(centers.shape[0]):
        idxs = np.where(labels == cluster)[0]
        if idxs.shape[0] > 0:
            labels_[idxs] = len(centers_)
            centers_.append(centers[cluster])
    centers_ = np.asarray(np.vstack(centers_))
    return centers_, labels_

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
