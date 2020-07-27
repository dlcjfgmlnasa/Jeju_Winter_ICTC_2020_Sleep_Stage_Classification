# -*- coding:utf-8 -*-
import os
import numpy as np
from sklearn.model_selection import KFold


def train_test_validation_split(path, n_splits=10):
    # Cross Validation
    files = np.array([os.path.join(path, name) for name in os.listdir(path)])
    kf = KFold(n_splits=n_splits)

    total_train_paths, total_validation_path, total_test_path = [], [], []
    for train_index, test_index in kf.split(files):
        train_paths = files[train_index[:-3]]
        validation_paths = files[train_index[-3:]]
        test_paths = files[test_index]

        total_train_paths.append(train_paths)
        total_validation_path.append(validation_paths)
        total_test_path.append(test_paths)

    return total_train_paths, total_validation_path, total_test_path


def drop_week_one_part(total_x, total_y):
    # Wake Label 일부 삭제
    step = 10

    total_list = list(total_y)
    total_size = len(total_list)

    # 처음 부분의 Wake Label 일부
    start_idx, end_idx = 0, 0
    for i in total_list:
        if i == 0:
            start_idx += 1
        else:
            break

    # 마지막 부분의 Wake Label 일부
    total_list.reverse()
    for i in total_list:
        if i == 0:
            end_idx += 1
        else:
            break

    total_x = total_x[start_idx - step:total_size - end_idx + step, :]
    total_y = total_y[start_idx - step:total_size - end_idx + step]
    return total_x, total_y


class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
