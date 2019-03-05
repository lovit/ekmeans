import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot


class EKMeans:
    def __init__(self, n_clusters, max_iter=30, tol=0.0001, epsilon, init='random',
        algorithm='ekmeans', metric='cosine', random_state=None, verbose=True):

        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.init = init
        self.algorithm = algorithm
        self.metric = metric
        self.random_state = random_state
        self.verbose = verbose

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : sparse matrix, shape = [n_samples, n_features]
            New data to transform.
        y : Ignored

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels_

    def fit_transform(self, X):
        raise NotImplemented

    def fit(self, X):

        self._check_fit_data(X)
        random_state = check_random_state(self.random_state)

        self.cluster_centers_, self.labels_, self.inertia_, = \
            ek_means(
                X, n_clusters = self.n_clusters, init = self.init,
                max_iter = self.max_iter, tol = self.tol,
                random_state = random_state, algorithm = self.algorithm,
                verbose = self.verbose,
            )

        return self

    def predict(self, X):
        raise NotImplemented

    def transform(self, X):
        raise NotImplemented

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32])
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))

def ek_means(X, n_clusters, init, max_iter, tol, random_state = random_state,
    algorithm, verbose):
    raise NotImplemented

def inner_product(X, Y):
    """
    Arguments
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
