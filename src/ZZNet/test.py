# -*- coding:utf-8 -*-
import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.ZZNet.model import ZZLet
from src.ZZNet.utils import TorchDataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import scikitplot as sk_plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=os.path.join('..', '..', 'data', 'physionet-sleep-npy'), type=str)
    parser.add_argument('--model_dir', default=os.path.join('.', 'ckpt', 'eog', 'best_model.pth'), type=str)
    parser.add_argument('--accuracy_dir', default=os.path.join('.', 'result', 'eog_accuracy.csv'), type=str)
    parser.add_argument('--kappa_dir', default=os.path.join('.', 'result', 'eog_kappa.csv'), type=str)
    return parser.parse_args()


class Tester(object):
    def __init__(self, args):
        self.args = args
        self.ckpt = torch.load(self.args.model_dir)
        self.model = self.load_model()
        self.model.to(device)
        self.model.eval()

    def load_model(self):
        model = ZZLet(**self.ckpt['parameter'])
        model.load_state_dict(self.ckpt['model_state_dict'])
        return model

    def get_test_loader(self):
        test_path = self.ckpt['test_path']
        d_type = self.ckpt['d_type']
        dataset = TorchDataset(paths=test_path, d_type=d_type, train=False)
        x = torch.tensor(dataset.x, dtype=torch.float32).to(device)
        y = torch.tensor(dataset.y, dtype=torch.long).to(device)
        return x, y

    def accuracy(self):
        result = []
        with torch.no_grad():
            test_path = self.ckpt['test_path']
            d_type = self.ckpt['d_type']

            total_accuracy = 0
            for path in test_path:
                dataset = TorchDataset(paths=[path], d_type=d_type, train=False)
                x = torch.tensor(dataset.x, dtype=torch.float32).to(device)
                y = torch.tensor(dataset.y, dtype=torch.long).to(device)

                out = self.model(x)
                out = torch.argmax(out, dim=-1)
                accuracy = torch.mean(torch.eq(out, y).to(dtype=torch.float32)).item()
                total_accuracy += accuracy

                result.append({
                    'path': os.path.basename(path),
                    'accuracy': accuracy
                })

            total_accuracy /= len(test_path)
        print('average accuracy : {}'.format(total_accuracy))
        result_df = pd.DataFrame(result)
        result_df.to_csv(self.args.accuracy_dir, index=False)

    def kappa(self):
        result = []
        with torch.no_grad():
            test_path = self.ckpt['test_path']
            d_type = self.ckpt['d_type']

            total_kappa = 0
            for path in test_path:
                dataset = TorchDataset(paths=[path], d_type=d_type, train=False)
                x = torch.tensor(dataset.x, dtype=torch.float32).to(device)
                y = torch.tensor(dataset.y, dtype=torch.long).to(device).cpu().numpy()

                out = self.model(x)
                out = torch.argmax(out, dim=-1).cpu().numpy()
                kappa_value = cohen_kappa_score(y, out)
                total_kappa += kappa_value

                result.append({
                    'path': os.path.basename(path),
                    'kappa': kappa_value
                })

            total_kappa /= len(test_path)
        print('average kappa : {}'.format(total_kappa))
        result_df = pd.DataFrame(result)
        result_df.to_csv(self.args.kappa_dir, index=False)

    def normalization_confusion_matrix(self):
        x, y = self.get_test_loader()
        with torch.no_grad():
            out = self.model(x)
            out = torch.argmax(out, dim=-1).numpy()
            y = y.numpy()
            matrix = confusion_matrix(y, out)

            # Normalization
            norm_matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
            sk_plt.metrics.plot_confusion_matrix(y, out, normalize=True)
            plt.show()
            return norm_matrix

    def sleep_stage(self):
        for filename in self.ckpt['validation_path'][3:]:
            mat = np.load(filename)
            x, y = mat['x'], mat['y']
            plt.subplot(211)
            plt.title('Hypnogram manually scored by a sleep expert')
            plt.plot(y)
            plt.yticks(range(0, 5))
            plt.subplot(212)

            print(x.shape)
            print(y.shape)
            # plt.show()
            # plt.title('Hypnogram automatically scored by Network')
            # x = torch.tensor(x, dtype=torch.float32)
            # x = x.unsqueeze(dim=1)
            # out = self.model(x)
            # plt.plot(out.argmax(dim=-1).numpy())
            # plt.yticks(range(0, 5))
            # plt.show()
            exit()


if __name__ == '__main__':
    arguments = get_args()
    Tester(arguments).kappa()
    # print(norm_confusion_matrix)
