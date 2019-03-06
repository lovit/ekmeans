from collections import Counter
import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.utils import check_array
from sklearn.utils import check_random_state

from .cluster_utils import compatify
from .cluster_utils import merge_close_clusters
from .cluster_utils import print_status


class EKMeans:
    def __init__(self, n_clusters, epsilon=0.4, max_depth=5, min_size=2,
        max_iter=30, tol=0.0001, init='random', metric='cosine',
        random_state=None, postprocessing=False, verbose=True, debug_dir=None):

        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.min_size = min_size
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.metric = metric
        self.random_state = random_state
        self.postprocessing = postprocessing
        self.verbose = verbose
        self.debug_dir = debug_dir

    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : sparse matrix, shape = [n_samples, n_features]
            New data to transform.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def fit(self, X):
        self._check_fit_data(X)
        random_state = check_random_state(self.random_state)

        self.cluster_centers_, self.labels_ = \
            ek_means(
                X, n_clusters = self.n_clusters, epsilon = self.epsilon,
                init = self.init, max_iter = self.max_iter, max_depth= self.max_depth,
                tol = self.tol, random_state = random_state, metric = self.metric,
                min_size = self.min_size, postprocessing = self.postprocessing,
                verbose = self.verbose
            )
        return self

    def predict(self, X):
        labels, dist = pairwise_distances_argmin_min(
            X, self.cluster_centers_, metric=self.metric)
        labels[np.where(dist >= self.epsilon)[0]] = -1
        return labels

    def transform(self, X):
        if not hasattr(self, 'cluster_centers_'):
            raise ValueError('Train model first using EKMeans.fit(X)')
        return pairwise_distances(X, self.cluster_centers_, metric=self.metric)

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = check_array(X, accept_sparse='csr',
                dtype=[np.float64, np.float32])
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))

def ek_means(X, n_clusters, epsilon, max_depth, init, max_iter, tol,
    random_state, metric, min_size, postprocessing, verbose):
    """
    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse.matrix
    n_clusters : int
        Number of clusters for each round
    epsilon : float
        Maximum distance between centroid and points that allowed to assign.
    max_depth : int
        Maximum number of basic epsilon k-means
    init : str or callable
        Initializer method
    max_iter : int
        Maximum number of iteration for basic epsilon k-means
    tol : float
        Convergence threshold. Proportion of re-assigned points.
    random_state : int or None
        Random seed
    metric : str
        Distance metric
    min_size : int
        Minumum cluster size for basic epsilon k-means
    verbose : Boolean
        If True, verbose mode on

    Returns
    -------
    centers : numpy.ndarray
        shape = (-1, n_features)
    labels : numpy.ndarray
        shape = (n_samples)

    Notes
    -----
    label -1 means not cluster assigned points (outliers)
    """

    n_samples = X.shape[0]
    cum_clusters = 0
    centers = []
    labels = -1 * np.ones(n_samples, dtype=np.int)
    sub_to_idx = np.asarray(range(n_samples), dtype=np.int)

    for depth in range(1, max_depth + 1):
        # for each base ek-means
        sub_centers, sub_labels = ek_means_base(X, n_clusters, epsilon, min_size,
            init, max_iter, tol, random_state, metric, verbose, depth)

        # store labels
        assigned_idxs = np.where(sub_labels >= 0)[0]
        sub_labels[assigned_idxs] += cum_clusters
        labels[sub_to_idx[assigned_idxs]] = sub_labels[assigned_idxs]

        # store centroids
        centers.append(sub_centers)
        cum_clusters += sub_centers.shape[0]

        if verbose:
            print('  - num cumulative clusters = {}'.format(cum_clusters))

        sub_to_idx_ = np.where(sub_labels == -1)[0]
        X = X[sub_to_idx_]

        sub_to_idx = sub_to_idx[sub_to_idx_]

        # check whether execute additional round
        if assigned_idxs.shape[0] <= min_size:
            break

    centers = np.asarray(np.vstack(centers))

    if postprocessing:
        if verbose:
            print('Post-processing: merging close clusters ...', end='')
        centers, labels, merge_to_indpt = merge_close_clusters(
            centers, labels, epsilon)
        if verbose:
            print('\rPost-processing: merging close clusters was done')

    labels = flush(X, centers, labels, sub_to_idx, epsilon, metric)
    return centers, labels

def ek_means_base(X, n_clusters, epsilon, min_size, init, max_iter, tol,
    random_state, metric, verbose, depth):
    """
    Returns
    -------
    centers : numpy.ndarray
        Centroid vector. shape = (n_clusters, X.shape[1])
    labels : numpy.ndarray
        Cluster labels
    """

    # set convergence threshold
    n_samples = X.shape[0]
    tol_ = int(n_samples * tol)

    # initialize parameters
    labels = -1 * np.ones(n_samples)
    centers = initialize(X, n_clusters, init, random_state)

    for i_iter in range(1, max_iter + 1):
        # reassign & update centroid
        new_labels, dist = reassign(X, centers, epsilon, min_size, metric)
        centers = update_centroid(X, centers, new_labels)

        # update labels
        n_changed = np.where(labels != new_labels)[0].shape[0]
        labels = new_labels

        # check convergence
        if verbose:
            print_status(depth, i_iter, labels, n_changed)

        if n_changed <= tol_:
            if verbose:
                print('  - Early stoped. (converged)')
            break

    centers, labels = compatify(centers, labels)
    return centers, labels

def reassign(X, centers, epsilon, min_size, metric):
    # find closest cluster
    labels, dist = pairwise_distances_argmin_min(X, centers, metric=metric)

    # epsilon filtering
    labels[np.where(dist >= epsilon)[0]] = -1

    # size filtering
    for label, size in Counter(labels).items():
        if size < min_size:
            centers[label] = 0
            labels[np.where(labels == label)[0]] = -1

    return labels, dist

def update_centroid(X, centers, labels):
    for cluster in np.unique(labels):
        idxs = np.where(labels == cluster)[0]
        centers[cluster] = np.asarray(X[idxs,:].sum(axis=0)) / idxs.shape[0]
    return centers

def flush(X, centers, labels, sub_to_idx, epsilon, metric):
    # set min_size = 0
    sub_labels, _ = reassign(X, centers, epsilon, 0, metric)
    assigned_idxs = np.where(sub_labels >= 0)[0]
    labels[sub_to_idx[assigned_idxs]] = sub_labels[assigned_idxs]
    return labels

def initialize(X, n_clusters, init, random_state):
    if isinstance(init, str) and init == 'random':
        seeds = random_state.permutation(X.shape[0])[:n_clusters]
        if sp.sparse.issparse(X):
            centers = X[seeds,:].todense()
        else:
            centers = X[seeds,:]
    elif hasattr(init, '__array__'):
        centers = np.array(init, dtype=X.dtype)
        if centers.shape[0] != n_clusters:
            raise ValueError('the number of customized initial points '
                'should be same with n_clusters parameter')
    elif callable(init):
        centers = init(X, n_clusters, random_state=random_state)
        centers = np.asarray(centers, dtype=X.dtype)
    else:
        raise ValueError("init method should be "
            "['random', 'callable', 'numpy.ndarray']")
    return centers
