import os
import logging
from math import floor
from datetime import datetime
from random import randint
import inspect
import string
import numpy as np
import pickle
import yaml


def prepare_logger(log_dir):
    logger = logging.getLogger(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_dir + "info.log")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    # logger.addHandler(ch)


def log_inputs(defaults, log_dir):
    with open(log_dir + 'meta.yaml', mode='w+') as f:
        yaml.dump(defaults, f, allow_unicode=True)
    save_object(defaults, filepath=log_dir + 'defaults.pkl')


def get_default_args(func):
    signature = inspect.signature(func)
    output = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    return output


def generate_log_dir(log_dir='./logs/'):
    log_dir = add_timestamp(log_dir)
    log_dir = add_random_characters(log_dir + '_')
    log_dir += '/'
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def add_timestamp(beginning='./params_'):
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = beginning + time_stamp
    return output_file


def add_random_characters(beginning, size_to_add=10):
    letters = string.ascii_letters
    stamp = ''
    r = np.random.choice(a=len(letters), size=size_to_add)
    for i in range(size_to_add):
        stamp += letters[r[i]]
    return beginning + stamp


def add_timestamp_with_random(beginning='./params_', ending='.pkl'):
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_stamp += '_' + str(randint(1, int(1.e5)))
    output_file = beginning + time_stamp + ending
    return output_file


def fill_in_residuals(residuals, maxcgiters):
    r_norm = np.zeros(shape=maxcgiters)
    if residuals[-1] > 0.0:
        iters_n = residuals.shape[0] - 1
        r_norm = residuals
    else:
        iters_n = np.argmin(residuals)
        iters_n -= 1
        r_norm += residuals[iters_n]
        r_norm[:iters_n + 1] = residuals[:iters_n + 1]
    return r_norm, iters_n


def compute_residual_norms(residuals, maxcgiters):
    r_norm = np.zeros(shape=maxcgiters)
    r_norm += np.linalg.norm(residuals[-1], axis=0)
    for i in range(len(residuals)):
        r_norm[i] = np.linalg.norm(residuals[i], axis=0)
    return r_norm


def print_time_taken(delta, text='Experiment took: ', logger=None):
    minutes = floor(delta / 60)
    seconds = delta - minutes * 60
    message = text + f'{minutes:4d} min and {seconds:4.2f} sec'
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def load_object(filepath):
    with open(file=filepath, mode='rb') as f:
        obj = pickle.load(file=f)
    return obj


def save_object(obj, filepath, use_highest=True):
    protocol = pickle.HIGHEST_PROTOCOL if use_highest else pickle.DEFAULT_PROTOCOL
    with open(file=filepath, mode='wb') as f:
        pickle.dump(obj=obj, file=f, protocol=protocol)


def get_red_colors():
    colors = {
        3: ['#fee0d2', '#fc9272', '#de2d26'],
        4: ['#fee5d9', '#fcae91', '#fb6a4a', '#cb181d'],
        5: ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'],
        6: ['#fee5d9', '#fcbba1', '#fc9272', '#fb6a4a', '#de2d26', '#a50f15'],
        7: [
            '#fee5d9', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#99000d'
        ],
    }
    return colors


def get_blue_colors():
    colors = {
        3: ['#deeb7', '#9ecae1', '#3181bd'],
        4: ['#eff3ff', '#bdd7e7', '#6baed6', '#2171b5'],
        5: ['#eff3ff', '#bdd7e7', '#6baed6', '#3182bd', '#08519c'],
    }
    return colors
