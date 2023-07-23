# kuzu.py
# COMP9444, CSE, UNSW
## Some programming statements refer to https://pytorch.org/tutorials/beginner/nn_tutorial.html#refactor-using-nn-linear

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

# insert my own code

batch_size = 28 * 28
output_size = 10


def format_dimension(x):
    x = x.view(x.size()[0], -1)
    return x


class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.lin = nn.Linear(in_features=batch_size, out_features=output_size)

    def forward(self, x):
        x = format_dimension(x)
        x = self.lin(x)
        x = torch.nn.functional.log_softmax(x, dim=1, dtype=None)
        return x  # CHANGE CODE HERE


hidden_nodes = 300


# hidden_nodes_2 = 100


class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.lin1 = nn.Linear(in_features=batch_size, out_features=hidden_nodes)
        self.lin2 = nn.Linear(in_features=hidden_nodes, out_features=output_size)
        # self.lin3 = nn.Linear(in_features=hidden_nodes_2, out_features=output_size)

    def forward(self, x):
        x = format_dimension(x)
        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        # x = torch.tanh(x)
        # x = self.lin3(x)
        x = torch.nn.functional.log_softmax(x, dim=1, dtype=None)
        return x  # CHANGE CODE HERE


temp_channels = 16
hidden_features = 600
kernel = 5
padding_size = 2


class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=temp_channels, kernel_size=kernel, padding=padding_size)
        self.conv2 = nn.Conv2d(in_channels=temp_channels, out_channels=2 * temp_channels, kernel_size=kernel,
                               padding=padding_size)
        self.lin1 = nn.Linear(in_features=1152, out_features=hidden_features)
        self.lin2 = nn.Linear(in_features=hidden_features, out_features=output_size)
        self.pool = nn.MaxPool2d(kernel_size=kernel, padding=padding_size)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = format_dimension(x)
        x = self.lin1(x)
        x = torch.nn.functional.relu(x)
        x = self.lin2(x)
        x = torch.nn.functional.log_softmax(x, dim=1, dtype=None)
        return x  # CHANGE CODE HERE
