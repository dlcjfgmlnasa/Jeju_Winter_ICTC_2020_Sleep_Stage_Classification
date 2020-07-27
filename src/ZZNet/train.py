# -*- coding:utf-8 -*-
import os
import torch
import argparse
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from src.ZZNet.utils import TorchDataset
from src.ZZNet.model import ZZLet
from src.utils import train_test_validation_split, EarlyStopping
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    d_type = ['eeg']
    parser.add_argument('--data_dir', default=os.path.join('..', '..', 'data', 'physionet-sleep-npy'), type=str)
    parser.add_argument('--save_dir', default=os.path.join('.', 'ckpt', d_type[0]), type=str)
    parser.add_argument('--d_type', default=d_type, type=str, choices=['eeg', 'eog', 'emg'])

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=400, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--patience', default=200, type=int)
    parser.add_argument('--sampling_rate', default=100, type=int)

    parser.add_argument('--f1', type=int, default=10)
    parser.add_argument('--f2', type=int, default=20)
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--n_channels', type=int, default=len(d_type))
    parser.add_argument('--rnn_layers', type=float, default=2)
    parser.add_argument('--cnn_dropout', type=float, default=0.25)
    parser.add_argument('--rnn_dropout', type=float, default=0.5)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.total_train_paths, self.total_test_paths, self.total_validation_paths = \
            train_test_validation_split(args.data_dir)
        self.model = ZZLet(f1=args.f1, f2=args.f2, d=args.d, channel_size=args.n_channels,
                           sampling_rate=args.sampling_rate, rnn_layers=args.rnn_layers,
                           cnn_dropout_rate=args.cnn_dropout, rnn_dropout=args.rnn_dropout,
                           seq_len=args.seq_len, classes=5)
        self.model.to(device)
        self.optimizer = opt.AdamW(self.model.parameters(),
                                   lr=args.learning_rate)

        self.model_state_dict = self.model.state_dict()
        self.optimizer_state_dict = self.optimizer.state_dict()

        self.early_stopping = EarlyStopping(patience=args.patience)

    def train(self, train_paths, validation_paths, test_paths, ckpt_dir, writer):
        args = self.args
        # Train Dataset & DataLoader
        train_dataset = TorchDataset(paths=train_paths, d_type=args.d_type)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # Validation Dataset
        validation_dataset = TorchDataset(paths=validation_paths, d_type=args.d_type)

        # Test Dataset
        test_dataset = TorchDataset(paths=test_paths, d_type=args.d_type)

        total_iter = 0
        for epoch in range(args.epochs):
            for i, data in enumerate(train_dataloader):
                self.model.train()
                train_x, train_y = data
                train_x, train_y = train_x.to(device), train_y.to(device)

                t_out = self.model(train_x)
                t_accuracy, t_loss = self.get_accuracy_loss(output=t_out, target=train_y)

                self.optimizer.zero_grad()
                t_loss.backward()
                self.optimizer.step()

                if i % 10 == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_x, val_y = validation_dataset.x, validation_dataset.y
                        val_x = torch.tensor(val_x, dtype=torch.float32).to(device)
                        val_y = torch.tensor(val_y, dtype=torch.long).to(device)

                        test_x, test_y = test_dataset.x, test_dataset.y
                        test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
                        test_y = torch.tensor(test_y, dtype=torch.long).to(device)

                        v_out, t_out = self.model(val_x), self.model(test_x)
                        v_accuracy, v_loss = self.get_accuracy_loss(output=v_out, target=val_y)
                        t_accuracy, t_loss = self.get_accuracy_loss(output=t_out, target=test_y)

                        # TensorBoard Writer
                        writer.add_scalar('train/loss', t_loss.item(), total_iter)
                        writer.add_scalar('train/accuracy', t_accuracy.item(), total_iter)
                        writer.add_scalar('validation/loss', v_loss.item(), total_iter)
                        writer.add_scalar('validation/accuracy', v_accuracy.item(), total_iter)
                        writer.add_scalar('test/loss', t_loss.item(), total_iter)
                        writer.add_scalar('test/accuracy', t_accuracy.item(), total_iter)

                        print('[Epoch] : {0:2d}  '
                              '[Iteration] : {1:4d}  '
                              '[Train Acc] : {2:.4f}  '
                              '[Train Loss] : {3:.4f}    '
                              '[Val Acc] : {4:.4f}    '
                              '[Val Loss] : {5:.4f}    '
                              '[Test Acc] : {6:.4f}    '
                              '[Test Loss]: {7:.4f}    '.format(epoch, i, t_accuracy.item(), t_loss.item(),
                                                                v_accuracy.item(), v_loss.item(),
                                                                t_accuracy.item(), t_loss.item()))
                        self.early_stopping(val_loss=v_loss.item())

                if self.early_stopping.early_stop:
                    save_dir = os.path.join(ckpt_dir, 'best_model.pth')
                    self.model_save(save_dir=save_dir, train_paths=train_paths,
                                    test_paths=test_paths, validation_paths=validation_paths, epoch=epoch)
                    exit()

                    total_iter += 1

            save_dir = os.path.join(ckpt_dir, 'model_{}.pth'.format(epoch))
            self.model_save(save_dir=save_dir, train_paths=train_paths,
                            test_paths=test_paths, validation_paths=validation_paths, epoch=epoch)

        save_dir = os.path.join(ckpt_dir, 'model_{}.pth'.format(args.epochs))
        self.model_save(save_dir=save_dir, train_paths=train_paths,
                        test_paths=test_paths, validation_paths=validation_paths, epoch=epoch)

    def get_accuracy_loss(self, output, target):
        loss = self.criterion(input=output, target=target)
        output = torch.argmax(output, dim=-1)
        accuracy = torch.mean(torch.eq(target, output).to(dtype=torch.float32))
        return accuracy, loss

    def model_save(self, save_dir, train_paths, test_paths, validation_paths, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'batch_size': self.args.batch_size,
            'train_path': train_paths,
            'test_path': test_paths,
            'validation_path': validation_paths,
            'd_type': self.args.d_type,
            'parameter':  {'f1': self.args.f1, 'f2': self.args.f2, 'd': self.args.d,
                           'channel_size': self.args.n_channels, 'sampling_rate': self.args.sampling_rate,
                           'rnn_layers': self.args.rnn_layers, 'cnn_dropout_rate': self.args.cnn_dropout,
                           'rnn_dropout': self.args.rnn_dropout, 'seq_len': self.args.seq_len, 'classes': 5}
        }, os.path.join(save_dir))


if __name__ == '__main__':
    arguments = get_args()

    total_train_paths, total_validation_path, total_test_path = train_test_validation_split(arguments.data_dir)
    for fold, (train_paths_, validation_paths_, test_paths_) in enumerate(zip(total_train_paths,
                                                                              total_validation_path,
                                                                              total_test_path)):
        # Checkpoint & Tensorboard
        tensorboard_writer = SummaryWriter('runs/{}/{}_fold'.format(arguments.d_type[0], fold))
        ckpt_path = os.path.join(arguments.save_dir, 'fold_{}'.format(fold))
        os.makedirs(ckpt_path)

        trainer = Trainer(arguments)
        trainer.train(
            train_paths=train_paths_,
            validation_paths=validation_paths_,
            test_paths=test_paths_,
            ckpt_dir=ckpt_path,
            writer=tensorboard_writer
        )


