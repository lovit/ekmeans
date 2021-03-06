import math
import numpy as np
import os
import scipy as sp
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.utils import check_array
from time import time

from ekmeans.utils import check_convergence
from ekmeans.utils import verbose_message
from ekmeans.utils import as_minute
from ekmeans.utils import filter_infrequents
from ekmeans.utils import now
from ekmeans.utils import merge_close_clusters
from ekmeans.logger import build_logger


class EKMeans:
    def __init__(self, n_clusters, metric='euclidean', epsilon=0.6, min_size=3, max_depth=10,
        coverage=0.95, coarse_iter=5, max_iter=5, tol=0.0001, init='random',
        random_state=None, postprocessing=False, merge_similar=False,
        warm_start=False, verbose=True):

        if merge_similar:
            postprocessing = False

        self.n_clusters = n_clusters
        self.metric = metric
        self.epsilon = epsilon
        self.min_size = min_size
        self.max_depth = max_depth
        self.coverage = coverage
        self.coarse_iter = coarse_iter
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.postprocessing = postprocessing
        self.merge_similar = merge_similar
        self.warm_start = warm_start
        self.verbose = verbose

        self.cluster_centers_ = None
        self.depth_begin = 0

    def fit_predict(self, X, min_size=-1, log_dir=None, time_prefix=True):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Arguments
        ---------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            Training data, shape=(n_samples, n_features)
        min_size : int
            Minimum cluster size threshold to use at the final filtering.
            Default value -1 means that cluster sizes are larger than `self.min_size`
        log_dir : str or None
            Directory path to log files
            If not None, it records changes of labels at each iteration step.
        time_prefix : Boolean
            If True, it saves logs and temporal labels at `log_dir/yy-mm-dd_hh-mm-ss`
            Else, it saves them at `log_dir/`

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, min_size, log_dir, time_prefix).labels_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def fit(self, X, min_size=-1, log_dir=None, time_prefix=True):
        """Compute cluster centers.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Arguments
        ---------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            Training data, shape=(n_samples, n_features)
        min_size : int
            Minimum cluster size threshold to use at the final filtering.
            Default value -1 means that cluster sizes are larger than `self.min_size`
        log_dir : str or None
            Directory path to log files
            If not None, it records changes of labels at each iteration step.
        time_prefix : Boolean
            If True, it saves logs and temporal labels at `log_dir/yy-mm-dd_hh-mm-ss`
            Else, it saves them at `log_dir/`

        Returns
        -------
        self : EKmeans
            Trained model instance
        """
        self._check_fit_data(X)
        logger = build_logger(log_dir, self, time_prefix)

        if self.warm_start:
            cluster_centers_ = self.cluster_centers_
        else:
            cluster_centers_ = None

        centers, labels_ = \
            ekmeans(X, self.n_clusters, self.metric, self.epsilon,
                self.min_size, self.max_depth, self.coverage,
                self.coarse_iter, self.max_iter, self.tol, self.init,
                self.random_state, self.verbose, cluster_centers_,
                self.depth_begin, self.merge_similar, logger)

        if self.warm_start:
            self.depth_begin += self.max_depth

        if self.postprocessing:
            n_before = np.where(np.unique(labels_) >= 0)[0].shape[0]
            centers, labels_, _ = \
                merge_close_clusters(centers, labels_, self.epsilon, self.metric)
            n_after = np.where(np.unique(labels_) >= 0)[0].shape[0]
            if self.verbose:
                print(f'Merged similar clusters. num clusters {n_before} -> {n_after}')

        n_before = np.where(labels_ >= 0)[0].shape[0]
        labels_, centers = filter_infrequents(min_size, labels_, centers)
        n_after = np.where(labels_ >= 0)[0].shape[0]
        if (n_after < n_before) and (self.verbose):
            n_data = X.shape[0]
            p_before = 100 * n_before / n_data
            p_after = 100 * n_after / n_data
            print(f'Filter out small clusters of which size is smaller than {min_size}.\n'\
                  f'Assigned points: {n_before} -> {n_after} ({p_before:.4}% -> {p_after:.4}%)')

        if self.verbose:
            n_clusters = np.where(np.unique(labels_) >= 0)[0].shape[0]
            print(f'Found {n_clusters} clusters')

        if logger is not None:
            logger.log(-1, -1, labels_, path=f'{logger.log_dir}/labels.txt')

        self.cluster_centers_ = centers
        self.labels_ = labels_
        return self

    def predict(self, X, min_size=-1):
        labels, dist = pairwise_distances_argmin_min(
            X, self.cluster_centers_, metric=self.metric)
        labels[np.where(dist >= self.epsilon)[0]] = -1
        labels, _ = filter_infrequents(min_size, labels, None)
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

def ekmeans(X, n_init, metric, epsilon, min_size, max_depth, coverage,
    coarse_iter, max_iter, tol, init, random_state, verbose,
    centers=None, depth_begin=0, merge_similar=False, logger=None):

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
    coverage : float
        Target percentage of the data allocated to corresponding cluster.
    coarse_iter : int
        The number of iteration for k-means only with not-assigned data
    max_iter : int
        Maximum number of repetition
    tol : float
        Convergence threshold. if the distance between previous centroid
        and updated centroid is smaller than `tol`, it stops training step.
    init : str, callable, or numpy.ndarray
        Initialization method
        Available choices : ['random', 'callable', 'numpy.ndarray', 'kmeans++']
    random_state : int or None
        Random seed
    verbose : Boolean
        If True, it shows training progress.
    centers : numpy.ndarray or None
        If it is not None, reuse it as initial centroid vectors
    depth_begin : int
        Depth begin index
    merge_similar : Boolean
        If True, it merges similar clusters after execute `kmeans_core` for each round
    logger : Logger
        If not None, logging all cluster lables for each round and iteration

    Returns
    -------
    centers : numpy.ndarray
        Centroid vectors, shape = (n_clusters, X.shape[1])
    labels : numpy.ndarray
        Integer list, shape = (X.shape[0],)
    """
    if max_depth <= 0:
        max_depth_ = math.ceil(n_data / n_init)
    else:
        max_depth_ = max_depth

    n_clusters = 0
    n_data = X.shape[0]
    labels = -np.ones(n_data, dtype=np.int)
    t = time()

    for depth in range(depth_begin + 1, max_depth_ + depth_begin + 1):
        if centers is None:
            centers = initialize(X, n_init, metric, init, random_state)

        if depth == depth_begin + 1:
            max_iter_ = max_iter + coarse_iter
        else:
            max_iter_ = max_iter

        if (depth > depth_begin + 1) and (coarse_iter > 0):
            indices = np.where(labels == -1)[0]
            Xs = X[indices]
            centers_new = initialize(Xs, n_init, metric, init, random_state)
            sub_labels = -np.ones(indices.shape[0], dtype=np.int)
            prefix = f'round: {depth}/{max_depth + depth_begin} coarse-'

            # no-logging
            centers_new, sub_labels, _ = ekmeans_core(Xs, centers_new, metric, sub_labels,
                coarse_iter, tol, epsilon, min_size, verbose, prefix, -1, None)

            labels[indices] = sub_labels
            centers = np.vstack([centers, centers_new])

        # logging coarse learning
        if (logger is not None) and (coarse_iter > 0) and (depth > depth_begin + 1):
            logger.log(depth, 0, labels, f'{now()}  [round: {depth}/{max_depth_ + depth_begin}] save coarse learning')

        prefix = f'round: {depth}/{max_depth + depth_begin} full-'
        centers, labels, i_iter = ekmeans_core(X, centers, metric, labels,
            max_iter_, tol, epsilon, min_size, verbose, prefix, depth, logger)

        if merge_similar:
            n_before = centers.shape[0]
            centers, labels, _ = merge_close_clusters(centers, labels, epsilon, metric)
            n_after = centers.shape[0]
            message = f'[round: {depth}/{max_depth_ + depth_begin}] merged similar clusters: {n_before} -> {n_after}'
            if verbose:
                print(message)
            if (logger is not None):
                logger.log(depth, i_iter+1, labels, message)

        n_assigned = np.where(labels >= 0)[0].shape[0]
        percent = n_assigned / n_data
        if verbose:
            percent_strf = f'{100 * percent:.4}%'
            t_strf = as_minute(time() - t)
            if t_strf:
                t_strf = f', time: {t_strf}'
            print(f'[round: {depth}/{max_depth + depth_begin}] #assigned: {n_assigned} ({percent_strf}){t_strf}\n')
            # develop
            n_unique = np.where(np.unique(labels) >= 0)[0].shape[0]
            print(f'[dev], unique labels = {n_unique}, center shape = {centers.shape}\n')

        if (coverage > 0) and (percent > coverage):
            print(f'Reached the target coverage {100 * coverage:.4}%')
            break

    if logger is not None:
        logger.save_messages()

    return centers, labels

def ekmeans_core(X, centers, metric, labels, max_iter,
    tol, epsilon, min_size, verbose, prefix='', depth=-1, logger=None):

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
    depth : int
        Round index which used for logging
    logger : Logger
        If not None, logging all cluster lables for each round and iteration

    Returns
    -------
    centers : numpy.ndarray
        Centroid vectors, shape = (n_clusters, X.shape[1])
    labels : numpy.ndarray
        Integer list, shape = (X.shape[0],)
    i_iter : int
        Number of excuted iterations
    """
    begin_time = time()
    n_clusters = centers.shape[0]

    # repeat
    i_iter = 0
    for i_iter in range(1, max_iter + 1):

        # training
        labels_, dist = reassign(X, centers, metric, epsilon, min_size)
        centers_, cluster_size = update_centroid(X, centers, labels_)

        # average distance only with assigned points
        assigned_indices = np.where(labels_ >= 0)[0]
        inner_dist = dist[assigned_indices].mean()
        n_assigned = assigned_indices.shape[0]
        n_clusters = np.where(np.unique(labels_) >= 0)[0].shape[0]

        # convergence check
        diff, n_changes, early_stop = check_convergence(
            centers, labels, centers_, labels_, tol, metric)
        if i_iter == max_iter:
            early_stop = False

        # reinitialize empty clusters
        if (i_iter < max_iter) and (np.where(cluster_size == 0)[0].shape[0] > 0):
            centers_ = reinitialize_empty_clusters_with_notassigned(
                X, centers_, cluster_size, labels_)

        centers = centers_
        labels = labels_

        # verbose
        strf = verbose_message(i_iter, max_iter, diff, n_changes, n_assigned,
            n_clusters, inner_dist, early_stop, begin_time, prefix)
        if verbose:
            print(strf)

        # logging
        if logger is not None:
            logger.log(depth, i_iter, labels, f'{now()}  {strf}')

        if early_stop:
            break

    return centers, labels, i_iter

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
        Available choices : ['random', 'callable', 'numpy.ndarray', 'kmeans++']
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
    centers = initialize(X, n_clusters, metric, init, random_state)
    labels = -np.ones(X.shape[0], dtype=np.int)

    # train
    centers, labels, _ = kmeans_core(X, centers, metric,
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
    i_iter : int
        Number of excuted iterations
    """
    begin_time = time()
    n_clusters = centers.shape[0]

    # repeat
    i_iter = 0
    for i_iter in range(1, max_iter + 1):

        # training
        labels_, dist = reassign(X, centers, metric)
        centers_, cluster_size = update_centroid(X, centers, labels_)

        # average distance only with assigned points
        assigned_indices = np.where(labels_ >= 0)[0]
        inner_dist = dist[assigned_indices].mean()
        n_assigned = assigned_indices.shape[0]
        n_clusters = np.where(np.unique(labels_) >= 0)[0].shape[0]

        # convergence check
        diff, n_changes, early_stop = check_convergence(
            centers, labels, centers_, labels_, tol, metric)
        if i_iter == max_iter:
            early_stop = False

        # reinitialize empty clusters
        n_empty_clusters = np.where(cluster_size == 0)[0].shape[0]
        if n_empty_clusters > 0:
            centers_ = reinitialize_empty_cluster_with_distant(
                X, centers_, cluster_size, dist)

        centers = centers_
        labels = labels_

        # verbose
        strf = verbose_message(i_iter, max_iter, diff, n_changes, n_assigned,
            n_clusters, inner_dist, early_stop, begin_time, prefix='kmeans')
        if verbose:
            print(strf)

        # logging
        if logger is not None:
            logger.log(0, i_iter, labels, f'{now()}  {strf}')

        if early_stop:
            break

    return centers, labels, i_iter

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
        Centroid vectors of current step
    labels : numpy.ndarray
        Integer list, shape = (X.shape[0],)

    Returns
    -------
    centers_ : numpy.ndarray
        Updated centroid vectors
    cluster_size : numpy.ndarray
        Shape = (n_clusters,)
    """
    n_clusters = centers.shape[0]
    centers_ = np.zeros(centers.shape, dtype=np.float)
    cluster_size = np.bincount(
        labels[np.where(labels >= 0)[0]],
        minlength = n_clusters
    )

    for label, size in enumerate(cluster_size):
        if size == 0:
            centers_[label] == centers[label]
        else:
            idxs = np.where(labels == label)[0]
            centers_[label] = np.asarray(X[idxs,:].sum(axis=0)) / idxs.shape[0]
    return centers_, cluster_size

def reinitialize_empty_cluster_with_distant(X, centers, cluster_size, dist):
    """
    Reinitialize empty clusters with random sampling from distant points

    Arguments
    ---------
    X : numpy.ndarray or scipy.sparse.csr_matrix
        Training data
    centers : numpy.ndarray
        Centroid vectors
    cluster_size : numpy.ndarray
        Shape = (n_clsuters,)
    dist : numpy.ndarray
        Distance from data and corresponding centroid

    Returns
    -------
    centers : numpy.ndarray
        Partially reinitialized centroid vectors
    """
    cluster_indices = np.where(cluster_size == 0)[0]
    n_empty = cluster_indices.shape[0]
    data_indices = dist.argsort()[-n_empty:]
    initials = X[data_indices,:]
    if sp.sparse.issparse(initials):
        initials = np.asarray(initials.todense())
    centers[cluster_indices,:] = initials
    return centers

def reinitialize_empty_clusters_with_notassigned(X, centers, cluster_size, labels):
    """
    Reinitialize empty clusters with random sampling from not-assigned points

    Arguments
    ---------
    X : numpy.ndarray or scipy.sparse.csr_matrix
        Training data
    centers : numpy.ndarray
        Centroid vectors
    cluster_size : numpy.ndarray
        Shape = (n_clsuters,)
    labels : numpy.ndarray
        Cluster indices, shape = (n_data,)

    Returns
    -------
    centers : numpy.ndarray
        Partially reinitialized centroid vectors
    """
    cluster_indices = np.where(cluster_size == 0)[0]
    n_empty = cluster_indices.shape[0]
    data_indices = np.where(labels == -1)[0]
    data_indices = np.random.permutation(data_indices)[:n_empty]
    initials = X[data_indices,:]
    if sp.sparse.issparse(initials):
        initials = np.asarray(initials.todense())
    centers[cluster_indices,:] = initials
    return centers

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

def initialize(X, n_clusters, metric, init, random_state):
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
    elif init == 'kmeans++':
        centers, _ = kmeanspp(X, n_clusters, metric)
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
            "['random', 'callable', 'numpy.ndarray', 'kmeans++']")
    return centers

def kmeanspp(X, n_clusters, metric):
    """
    Arguments
    ---------
    X : numpy.ndarray or scipy.sparse.csr_matrix
        Training data
    n_clusters : int
        Number of clusters
    metric : str
        Distance metric

    Returns
    -------
    centers : numpy.ndarray
        Initialized centroid vectors, shape = (n_clusters, X.shape[1])

    Usage
    -----
    With dense matrix

        >>> import numpy as np

        >>> n_clusters = 5
        >>> metric = 'euclidean'
        >>> z = np.random.random_sample((1000, 2))

        >>> centers, seeds = kmeanspp(z, n_clusters, metric)

    With sparse matrix

        >>> from scipy.sparse import csr_matrix

        >>> nnz = 1000
        >>> rows = np.random.randint(0, 100, nnz)
        >>> cols = np.random.randint(0, 500, nnz)
        >>> data = np.ones(nnz)
        >>> z = csr_matrix((data, (rows, cols)))

        >>> centers, seeds = kmeanspp(z, n_clusters, metric)
    """
    n_data, n_features = X.shape
    if n_data < n_clusters:
        raise ValueError('The length of data must be larger than `n_clusters`')
    if n_data == n_clusters:
        if sp.sparse.issparse(X):
            return X.todense()
        return X

    # initialize
    seeds = np.zeros(n_clusters, dtype=np.int)
    seed = np.random.randint(n_data)
    seeds[0] = seed
    dist = pairwise_distances(X[seed].reshape(1,-1), X, metric=metric).reshape(-1)

    # iterate
    for i in range(1, n_clusters):
        # define prob
        prob = dist ** 2
        prob /= prob.sum()
        # update seed
        seed = np.random.choice(n_data, 1, p=prob)
        seeds[i] = seed
        # update dist
        if i < n_clusters -1:
            dist_ = pairwise_distances(X[seed].reshape(1,-1), X, metric=metric).reshape(-1)
            dist = np.vstack([dist, dist_]).min(axis=0)

    if sp.sparse.issparse(X):
        centers = X[seeds,:].todense()
    else:
        centers = X[seeds]

    return centers, seeds
