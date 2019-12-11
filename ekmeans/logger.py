import json
import numpy as np
import os

from .utils import now


def build_logger(log_dir, ekmeans):
    if log_dir == None:
        return None

    log_dir = '{}/{}/'.format(log_dir, get_excution_time())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return Logger(log_dir, parameters)

def get_attributes(ekmeans):
    parameters = {
        'n_clusters': ekmeans.n_clusters,
        'metric': ekmeans.metric,
        'epsilon': ekmeans.epsilon,
        'min_size': ekmeans.min_size,
        'max_depth': ekmeans.max_depth,
        'coverage': ekmeans.coverage,
        'coarse_iter': ekmeans.coarse_iter,
        'max_iter': ekmeans.max_iter,
        'tol': ekmeans.tol,
        'init': ekmeans.init if isinstance(ekmeans.init, str) else 'callable',
        'random_state': ekmeans.random_state,
        'postprocessing': ekmeans.postprocessing if isinstance(ekmeans.postprocessing, str) else 'callable',
    }
    return parameters

class Logger:
    def __init__(self, log_dir, parameters):
        self.log_dir = log_dir

        # save configuration
        path = f'{log_dir}/configure.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

    def log(self, depth, iter, labels, centers):
        path = f'{self.log_dir}/{depth}_{iter}_label.txt'
        np.savetxt(path, labels_, '%d')
        path = f'{self.log_dir}/{depth}_{iter}_center.csv'
        np.savetxt(path, centers_, '%.8f')
