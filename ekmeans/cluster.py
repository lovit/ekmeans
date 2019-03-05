class EKMeans:
    def __init__(self, num_clusters, epsilon, initialize='random', metric='cosine'):
        self.num_clusters = num_clusters
        self.epsilon = epsilon
        self.initialize = initialize
        self.metric = metric

    def fit_predict(self, X):
        raise NotImplemented

    def fit_transform(self, X):
        raise NotImplemented

    def fit(self, X):
        raise NotImplemented

    def predict(self, X):
        raise NotImplemented

    def transform(self, X):
        raise NotImplemented

