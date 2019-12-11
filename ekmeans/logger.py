import json
import numpy as np
import os


def initialize_logger(log_dir):
    if log_dir == None:
        return None

    log_dir = '{}/{}/'.format(log_dir, get_excution_time())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return Logger(log_dir)

class Logger:
    def __init__(self, log_dir, n_samples):
        self.log_dir = log_dir
        self.centers = []
        self.cum_clusters = 0
        self.labels = -1 * np.ones(n_samples, dtype=np.int)

    def log_configure(self, ekmeans):
        params = {
            'n_clusters' : ekmeans.n_clusters,
            'epsilon' : ekmeans.epsilon,
            'max_depth': ekmeans.max_depth,
            'min_size': ekmeans.min_size,
            'max_iter': ekmeans.max_iter,
            'tol': ekmeans.tol,
            'init': ekmeans.init,
            'metric': ekmeans.metric,
            'random_state': str(ekmeans.random_state),
            'postprocessing': ekmeans.postprocessing
        }
        path = '{}/configure.json'.format(self.log_dir)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

    def log(self, suffix, labels=None, centers=None, sub_to_idx=None):
        if labels is None:
            labels_ = self.labels.copy()
            centers_ = np.vstack(self.centers)
        else:
            labels_, centers_ = self.markup(labels, centers, sub_to_idx)

        path = '{}/{}_label.txt'.format(self.log_dir, suffix)
        np.savetxt(path, labels_, '%d')
        path = '{}/{}_center.txt'.format(self.log_dir, suffix)
        np.savetxt(path, centers_, '%.8f')

    def cumulate(self, centers, labels, sub_to_idx):
        self.centers.append(centers)
        self.labels[sub_to_idx] = labels
        self.cum_clusters += centers.shape[0]

    def markup(self, labels, centers, sub_to_idx):
        labels = labels.copy()
        labels[np.where(labels >= 0)[0]] += self.cum_clusters
        labels_ = self.labels.copy()
        labels_[sub_to_idx] = labels
        centers_ = np.vstack([c for c in self.centers] + [centers])
        return labels_, centers_
