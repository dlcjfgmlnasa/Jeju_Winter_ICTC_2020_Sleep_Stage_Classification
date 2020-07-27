# -*- coding:utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset
from src.utils import drop_week_one_part


class TorchDataset(Dataset):
    def __init__(self, paths, d_type, train=True):
        super().__init__()
        self.x, self.y = self.get_dataset(paths=paths, d_type=d_type, train=train)
        self.d_type = d_type

    @staticmethod
    def get_dataset(paths, d_type, train):
        total_x, total_y = [], []
        for path in paths:
            mat = np.load(path)
            x, y = mat['x'], mat['y']

            if train:
                x, y = drop_week_one_part(x, y)

            x_temp = []
            if 'eeg' in d_type:
                x_temp.append(x[:, 0, :])
            if 'eog' in d_type:
                x_temp.append(x[:, 1, :])
            if 'emg' in d_type:
                x_temp.append(x[:, 2, :])
            x_sample = np.stack(x_temp, axis=1)
            total_x.append(x_sample)
            total_y.append(y)

        total_x = np.concatenate(total_x, axis=0)
        total_y = np.concatenate(total_y, axis=0)
        return total_x, total_y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        sample_x = torch.tensor(self.x[item], dtype=torch.float32)
        sample_y = torch.tensor(self.y[item], dtype=torch.long)
        return sample_x, sample_y
