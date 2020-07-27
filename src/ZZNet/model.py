# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class ZZLet(nn.Module):
    def __init__(self, f1, f2, d, channel_size, sampling_rate,
                 rnn_layers, cnn_dropout_rate, rnn_dropout, seq_len, classes):
        super().__init__()
        self.rnn_units = f2 * 3
        self.cnn = EEGNet(f1=f1, f2=f2, d=d, channel_size=channel_size,
                          dropout_rate=cnn_dropout_rate, sampling_rate=sampling_rate, classes=5)
        self.rnn = nn.LSTM(input_size=f2 * 3, hidden_size=f2 * 3, num_layers=rnn_layers,
                           dropout=rnn_dropout, bidirectional=True)
        self.fc = nn.Linear(
            in_features=(seq_len * f2 * 3),
            out_features=classes
        )

    def forward(self, x):
        b = x.size()[0]
        # EEGNet (Convolution Neural Network)
        cnn_outs = []
        for sample_x in torch.split(x, split_size_or_sections=100, dim=-1):
            sample_x = sample_x.unsqueeze(dim=1)
            cnn_out = self.cnn(sample_x)
            cnn_out = cnn_out.view([b, -1])
            cnn_outs.append(cnn_out)
        cnn_outs = torch.stack(cnn_outs, dim=1)

        # Recurrent Neural Network
        rnn_outs, _ = self.rnn(cnn_outs)

        # Skip-Connected
        rnn_outs = rnn_outs[:, :, :self.rnn_units] + rnn_outs[:, :, self.rnn_units:] + cnn_outs
        rnn_outs = rnn_outs.view([b, -1])
        out = self.fc(rnn_outs)
        return out


class EEGNet(nn.Module):
    def __init__(self, f1, f2, d, channel_size, dropout_rate, sampling_rate, classes):
        super(EEGNet, self).__init__()
        self.classes = classes
        self.cnn = nn.Sequential()
        half_sampling_rate = sampling_rate // 2

        self.cnn.add_module(
            name='conv_temporal',
            module=nn.Conv2d(
                in_channels=1,
                out_channels=f1,
                kernel_size=(1, half_sampling_rate),
                stride=1,
                bias=False,
                padding=(0, half_sampling_rate // 2)
            )
        )
        self.cnn.add_module(
            name='batch_normalization_1',
            module=nn.BatchNorm2d(f1)
        )
        self.cnn.add_module(
            name='conv_spatial',
            module=nn.Conv2d(
                in_channels=f1,
                out_channels=f1 * d,
                kernel_size=(channel_size, 1),
                stride=1,
                bias=False,
                groups=f1,
                padding=(0, 0),
            )
        )
        self.cnn.add_module(
            name='batch_normalization_2',
            module=nn.BatchNorm2d(f1 * d)
        )
        self.cnn.add_module(
            name='activation1',
            module=nn.ELU()
        )
        self.cnn.add_module(
            name='average_pool_2d_1',
            module=nn.AvgPool2d(
                kernel_size=(1, 4)
            )
        )
        self.cnn.add_module(
            name='dropout_rate1',
            module=nn.Dropout(dropout_rate)
        )
        self.cnn.add_module(
            name='conv_separable_point',
            module=nn.Conv2d(
                in_channels=f1 * d,
                out_channels=f2,
                kernel_size=(1, 1),
                stride=1,
                bias=False,
                padding=(0, 0),
            )
        )
        self.cnn.add_module(
            name='batch_normalization_3',
            module=nn.BatchNorm2d(f2),
        )
        self.cnn.add_module(
            name='activation2',
            module=nn.ELU()
        )
        self.cnn.add_module(
            name='average_pool_2d_2',
            module=nn.AvgPool2d(
                kernel_size=(1, 8)
            )
        )

    def forward(self, x):
        out = self.cnn(x)
        return out
