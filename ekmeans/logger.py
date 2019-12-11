import json
import numpy as np
import os

from .utils import now


def build_logger(log_dir, ekmeans, time_prefix=True):
    if log_dir == None:
        return None

    if time_prefix:
        log_dir = '{}/{}/'.format(log_dir, now())
    if log_dir[-1] != '/':
        log_dir += '/'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    parameters = get_attributes(ekmeans)
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
        self.messages = []
        print(f'Logging labels and verbose message at {log_dir}')

        # save configuration
        path = f'{log_dir}/configure.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(parameters, f, ensure_ascii=False, indent=2)

    def log(self, depth, iter, labels, message=None, path=None):
        if path is None:
            path = f'{self.log_dir}/round{depth}_iter{iter}_label.txt'
        np.savetxt(path, labels, '%d')
        if message is not None:
            self.messages.append(message)

    def save_messages(self, path=None):
        if (path is None) or (not path):
            path = f'{self.log_dir}/logs.txt'
        with open(path, 'w', encoding='utf-8') as f:
            for msg in self.messages:
                f.write(f'{msg.strip()}\n')
