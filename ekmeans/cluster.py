from collections import Counter
import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.utils import check_array
from time import time

from ekmeans.utils import merge_close_clusters
from ekmeans.utils import print_status
from ekmeans.utils import check_convergence
from ekmeans.utils import verbose_message
from ekmeans.logger import initialize_logger


class EKMeans:
    def __init__(self, n_clusters, epsilon=0.4, max_depth=5, min_size=2,
        max_iter=30, tol=0.0001, init='random', metric='cosine',
        random_state=None, postprocessing=False, verbose=True, log_dir=None):

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
        self.logger = initialize_logger(log_dir)

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
                verbose = self.verbose, logger = self.logger
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

def ekmeans(X, n_init, metric, epsilon, min_size, max_depth,
    max_iter, tol, init, random_state, verbose, logger=None):

    """
    Arguments
    ---------
    X : numpy.ndarray or scipy.sparse.csr_matrix
        Training data
    n_init : int
        Number of newly initialized clusters
    metric : str
        Distance metric
    epsilon : float
        Maximum distance from centroid to belonging data.
        The points distant more than epsilon are not assigned to any cluster.
    min_size : int
        Minimum number of assigned points.
        The clusters of which size is smaller than the value are disintegrated.
    max_depth : int
        Maximum number of rounds
    max_iter : int
        Maximum number of repetition
    tol : float
        Convergence threshold. if the distance between previous centroid
        and updated centroid is smaller than `tol`, it stops training step.
    init : str, callable, or numpy.ndarray
        Initialization method
    random_state : int or None
        Random seed
    verbose : Boolean
        If True, it shows training progress.
    logger : Logger
        If not None, logging all cluster lables for each round and iteration

    Returns
    -------
    centers : numpy.ndarray
        Centroid vectors, shape = (n_clusters, X.shape[1])
    labels : numpy.ndarray
        Integer list, shape = (X.shape[0],)
    """
    n_clusters = 0
    centers = None
    labels = -np.ones(X.shape[0])

    for depth in range(1, max_depth + 1):
        if centers is None:
            centers = initialize(X, n_init, init, random_state)
        else:
            indices = np.where(labels == -1)[0]
            Xs = X[indices]
            centers_new = initialize(Xs, n_init, init, random_state)
            centers = np.vstack([centers, centers_new])

        prefix = f'round: {depth}/{max_depth} '
        centers, labels = ekmeans_core(X, centers, metric, labels,
            max_iter, tol, epsilon, min_size, verbose, prefix, logger)

    # TODO
    # postprocessing: merge similar clusters

    return centers, labels

def ekmeans_core(X, centers, metric, labels, max_iter,
    tol, epsilon, min_size, verbose, prefix='', logger=None):

    """
    Arguments
    ---------
    X : numpy.ndarray or scipy.sparse.csr_matrix
        Training data
    centers : numpy.ndarray
        Initialized centroid vectors
    metric : str
        Distance metric
    labels : numpy.ndarray
        Cluster index list, shape=(n_data,)
    max_iter : int
        Maximum number of repetition
    tol : float
        Convergence threshold. if the distance between previous centroid
        and updated centroid is smaller than `tol`, it stops training step.
    epsilon : float
        Maximum distance from centroid to belonging data.
        The points distant more than epsilon are not assigned to any cluster.
    min_size : int
        Minimum number of assigned points.
        The clusters of which size is smaller than the value are disintegrated.
    verbose : Boolean
        If True, it shows training progress.
    prefix : str
        Verbose prefix
    logger : Logger
        If not None, logging all cluster lables for each round and iteration

    Returns
    -------
    centers : numpy.ndarray
        Centroid vectors, shape = (n_clusters, X.shape[1])
    labels : numpy.ndarray
        Integer list, shape = (X.shape[0],)
    """
    begin_time = time()

    # repeat
    for i_iter in range(1, max_iter + 1):

        # training
        labels_, dist = reassign(X, centers, metric, epsilon, min_size)
        centers_ = update_centroid(X, centers, labels_)

        # average distance only with assigned points
        assigned_indices = np.where(labels_ >= 0)[0]
        inner_dist = dist[assigned_indices].mean()
        n_assigned = assigned_indices.shape[0]
        n_clusters = np.unique(labels_).shape[0]

        # convergence check
        diff, n_changes, early_stop = check_convergence(
            centers, labels, centers_, labels_, tol, metric)

        centers = centers_
        labels = labels_

        # verbose
        if verbose:
            strf = verbose_message(i_iter, max_iter, diff, n_changes, n_assigned,
                n_clusters, inner_dist, early_stop, begin_time, prefix)
            print(strf)

        # TODO
        # logging

        if early_stop:
            break

    return centers, labels

def kmeans(X, n_clusters, metric, init='random', random_state=None,
    max_iter=10, tol=0.001, verbose=False, logger=None):

    """
    Arguments
    ---------
    X : numpy.ndarray or scipy.sparse.csr_matrix
        Training data
    n_clusters : int
        Number of clusters
    metric : str
        Distance metric
    init : str, callable, or numpy.ndarray
        Initialization method
    random_state : int or None
        Random seed
    max_iter : int
        Maximum number of repetition
    tol : float
        Convergence threshold. if the distance between previous centroid
        and updated centroid is smaller than `tol`, it stops training step.
    verbose : Boolean
        If True, it shows training progress.
    logger : Logger
        If not None, logging all cluster lables for each round and iteration

    Returns
    -------
    centers : numpy.ndarray
        Centroid vectors, shape = (n_clusters, X.shape[1])
    labels : numpy.ndarray
        Integer list, shape = (X.shape[0],)
    """

    # initialize
    centers = initialize(X, n_clusters, init, random_state)
    labels = -np.ones(X.shape[0])

    # train
    centers, labels = kmeans_core(X, centers, metric,
        labels, max_iter, tol, verbose, logger)

    return centers, labels

def kmeans_core(X, centers, metric, labels, max_iter, tol, verbose, logger=None):
    """
    Arguments
    ---------
    X : numpy.ndarray or scipy.sparse.csr_matrix
        Training data
    centers : numpy.ndarray
        Initialized centroid vectors
    metric : str
        Distance metric
    labels : numpy.ndarray
        Cluster index list, shape=(n_data,)
    max_iter : int
        Maximum number of repetition
    tol : float
        Convergence threshold. if the distance between previous centroid
        and updated centroid is smaller than `tol`, it stops training step.
    verbose : Boolean
        If True, it shows training progress.
    logger : Logger
        If not None, logging all cluster lables for each round and iteration

    Returns
    -------
    centers : numpy.ndarray
        Centroid vectors, shape = (n_clusters, X.shape[1])
    labels : numpy.ndarray
        Integer list, shape = (X.shape[0],)
    """
    begin_time = time()

    # repeat
    for i_iter in range(1, max_iter + 1):

        # training
        labels_, dist = reassign(X, centers, metric)
        centers_ = update_centroid(X, centers, labels_)

        # convergence check
        diff, n_changes, early_stop = check_convergence(
            centers, labels, centers_, labels_, tol, metric)

        centers = centers_
        labels = labels_

        # verbose
        if verbose:
            strf = verbose_message(i_iter, max_iter, diff, n_changes,
                -1, -1, dist.mean(), early_stop, begin_time)
            print(strf)

        # TODO
        # logging

        if early_stop:
            break

    return centers, labels

def reassign(X, centers, metric, epsilon=0, min_size=0, do_filter=True):
    """
    Arguments
    ---------
    X : numpy.ndarray or scipy.sparse.csr_matrix
        Training data
    centers : numpy.ndarray
        Centroid vectors
    metric : str
        Distance metric
    epsilon : float
        Maximum distance from centroid to belonging data.
        The points distant more than epsilon are not assigned to any cluster.
    min_size : int
        Minimum number of assigned points.
        The clusters of which size is smaller than the value are disintegrated.
    do_filter : Boolean
        If True, it executes `epsilon` & `min_size` based filtering.
        Else, it works like k-means.

    Returns
    -------
    labels : numpy.ndarray
        Integer list, shape = (X.shape[0],)
        Not assigned points have -1
    dist : numpy.ndarray
        Distance from their corresponding cluster centroid
    """
    # find closest cluster
    labels, dist = pairwise_distances_argmin_min(X, centers, metric=metric)

    if (not do_filter) or (epsilon == 0 and min_size <= 1):
        return labels, dist

    # epsilon filtering
    labels[np.where(dist >= epsilon)[0]] = -1

    cluster_size = np.bincount(
        labels[np.where(labels >= 0)[0]],
        minlength = centers.shape[0]
    )

    # size filtering
    for label, size in enumerate(cluster_size):
        if size < min_size:
            labels[np.where(labels == label)[0]] = -1

    return labels, dist

def update_centroid(X, centers, labels):
    """
    Arguments
    ---------
    X : numpy.ndarray or scipy.sparse.csr_matrix
        Training data
    centers : numpy.ndarray
        Centroid vectors of current step t
    labels : numpy.ndarray
        Integer list, shape = (X.shape[0],)

    Returns
    -------
    centers_ : numpy.ndarray
        Updated centroid vectors
    """
    n_clusters = centers.shape[0]
    centers_ = np.zeros(centers.shape, dtype=np.float)
    cluster_size = np.bincount(
        labels[np.where(labels >= 0)[0]],
        minlength = n_clusters
    )

    for label, size in enumerate(cluster_size):
        if size == 0:
            centers_[label] = centers[label]
        else:
            idxs = np.where(labels == label)[0]
            centers_[label] = np.asarray(X[idxs,:].sum(axis=0)) / idxs.shape[0]
    return centers_

def compatify(centers, labels):
    """
    Remove centroids of empty cluster from `centers`, and re-numbering cluster index

    Arguments
    ---------
    centers : numpy.ndarray
        Centroid vectors
    labels : numpy.ndarray
        Integer list, shape = (n_data,)

    Returns
    -------
    centers_ : numpy.ndarray
        Compatified centroid vectors
    labels_ : numpy.ndarray
        Re-numbered cluster index list, shape = (n_data,)

    Usage
    -----
        >>> centers = np.random.random_sample((10, 2))
        >>> labels = np.concatenate([
                np.random.randint(low=2, high=5, size=(100,)),
                np.random.randint(low=8, high=10, size=(100,))
            ])

        >>> np.unique(labels)
        $ array([2, 3, 4, 8, 9])

        >>> centers, labels = compatify(centers, labels)
        >>> np.unique(labels)
        $ [0 1 2 3 4]

        >>> centers

        $ [[0.44844521 0.44927703]
           [0.43318219 0.15770172]
           [0.7164269  0.60108245]
           [0.49072922 0.41281989]
           [0.61217156 0.20791931]]
    """
    centers_ = []
    labels_ = -1 * np.ones(labels.shape[0], dtype=np.int)
    for cluster in range(centers.shape[0]):
        idxs = np.where(labels == cluster)[0]
        if idxs.shape[0] > 0:
            labels_[idxs] = len(centers_)
            centers_.append(centers[cluster])
    centers_ = np.asarray(np.vstack(centers_))
    return centers_, labels_

def flush(X, centers, labels, sub_to_idx, epsilon, metric):
    "deprecated"
    # set min_size = 0
    sub_labels, _ = reassign(X, centers, epsilon, 0, metric)
    assigned_idxs = np.where(sub_labels >= 0)[0]
    labels[sub_to_idx[assigned_idxs]] = sub_labels[assigned_idxs]
    return labels

def initialize(X, n_clusters, init, random_state):
    """
    Arguments
    ---------
    X : numpy.ndarray or scipy.sparse.csr_matrix
        Training data
    n_clusters : int
        Number of clusters
    init : str, callable, or numpy.ndarray
        Initialization method
    random_state : int or None
        Random seed

    Returns
    -------
    centers : numpy.ndarray
        Initialized centroid vectors, shape = (n_clusters, X.shape[1])
    """
    np.random.seed(random_state)
    if isinstance(init, str) and init == 'random':
        seeds = np.random.permutation(X.shape[0])[:n_clusters]
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
