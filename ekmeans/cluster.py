from collections import Counter
import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot


class EKMeans:
    def __init__(self, n_clusters, epsilon=0.4, max_depth=5, min_size=2,
        max_iter=30, tol=0.0001, init='random', metric='cosine',
        random_state=None, verbose=True):

        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.min_size = min_size
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.metric = metric
        self.random_state = random_state
        self.verbose = verbose

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

        self.cluster_centers_, self.labels_, self.inertia_, = \
            ek_means(
                X, n_clusters = self.n_clusters, epsilon = self.epsilon,
                init = self.init, max_iter = self.max_iter, tol = self.tol,
                random_state = random_state, metric = self.metric,
                min_size = self.min_size, verbose = self.verbose
            )
        return self

    def predict(self, X):
        raise NotImplemented

    def transform(self, X):
        raise NotImplemented

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = check_array(X, accept_sparse='csr',
                dtype=[np.float64, np.float32])
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))

def ek_means(X, n_clusters, epsilon, max_depth, init, max_iter, tol,
    random_state, metric, min_size, verbose):
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
    center_stack = []
    labels = -1 * np.ones(n_samples)
    idx_to_row = np.asarray(range(n_samples), dtype=np.int)

    for depth in range(1, max_depth + 1):
        # each round
        centers, labels, X = ek_means_base(X, n_clusters, epsilon, min_size,
            init, max_iter, tol, random_state, metric, verbose, depth)
        # flushing
        # TODO

        # check whether execute additional round
    raise NotImplemented

def ek_means_base(X, n_clusters, epsilon, min_size, init, max_iter, tol,
    random_state, metric, verbose, depth):
    """
    Returns
    -------
    centers : numpy.ndarray
        Centroid vector. shape = (n_clusters, X.shape[1])
    labels : numpy.ndarray
        Labels
    Xr : numpy.ndarray or scipy.sparse.matrix
        Xr.shape[0] < X.shape[0]
        Submatrix of cluster not assigned points
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
        if n_changed <= tol_:
            if verbose:
                print('Early stoped. (converged)')
                print_status(depth, i_iter, labels, n_changed)
            break

        if verbose:
            print_status(depth, i_iter, labels, n_changed)

    Xr = X[np.where(labels == -1)[0]]
    return centers, labels, Xr

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

def initialize(X, n_clusters, init, random_state):
    if isinstance(init, str) and init == 'random':
        seeds = random_state.permutation(X.shape[0])[:n_clusters]
        centers = X[seeds,:].todense()
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

def print_status(i_round, i_iter, labels, n_changed):
    n_samples = labels.shape[0]
    n_clusters = np.where(np.unique(labels) >= 0)[0].shape[0]
    n_assigned = np.where(labels >= 0)[0].shape[0]
    print('[round #{}, iter #{}] n cluster = {}, n changes = {}, assigned = {} / {}'.format(
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
