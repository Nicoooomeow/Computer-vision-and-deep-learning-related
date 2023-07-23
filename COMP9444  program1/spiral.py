# spiral.py
# COMP9444, CSE, UNSW
# Some programming statements refer to https://pytorch.org/tutorials/beginner/nn_tutorial.html#refactor-using-nn-linear

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cmath

input_node = 2
output_node = 1


class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        self.lin1 = nn.Linear(input_node, num_hid)
        self.lin2 = nn.Linear(num_hid, output_node)

    def forward(self, input_data):
        x = input_data[:, 0]
        y = input_data[:, -1]
        input_data = transform(x, y)
        input_data = self.lin1(input_data)
        self.hidden_1 = torch.tanh(input_data)
        output = self.lin2(self.hidden_1)
        output = torch.sigmoid(output)
        # output = 0 * input[:, 0]  # CHANGE CODE HERE
        return output


def transform(x, y):
    a = torch.atan2(y, x)
    r = torch.sqrt(x * x + y * y)
    output_data = torch.stack((r, a), 1)
    return output_data


class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        self.lin1 = nn.Linear(input_node, num_hid)  # hidden nodes
        self.lin2 = nn.Linear(num_hid, num_hid * 2)
        self.lin3 = nn.Linear(num_hid * 2, output_node)

    def forward(self, input):
        input = self.lin1(input)
        self.hidden_1 = torch.tanh(input)
        self.hidden_2 = self.lin2(self.hidden_1)
        self.hidden_2 = torch.tanh(self.hidden_2)
        temp = self.lin3(self.hidden_2)
        output = torch.sigmoid(temp)
        return output
        # output = 0 * input[:, 0]  # CHANGE CODE HERE
        # return output


def shape_format(x):
    x = x.shape[0]
    return x


def modified(x_range, y_range, grid, net, layer_num, node):
    with torch.no_grad():
        net.eval()
        net(grid)
        if layer_num == 1:
            output = net.hidden_1[:, node]
        else:
            output = net.hidden_2[:, node]
        net.train()
        predict = (output >= 0.5).float()
        plt.clf()
        plt.pcolormesh(x_range, y_range, predict.cpu().view(shape_format(y_range), shape_format(x_range)), cmap='Wistia')


def graph_hidden(net, layer, node):  # copy from spiral_main
    plt.clf()
    # INSERT CODE HERE
    xrange = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
    yrange = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
    xcoord = xrange.repeat(shape_format(yrange))
    ycoord = torch.repeat_interleave(yrange, shape_format(xrange), dim=0)
    grid = torch.cat((xcoord.unsqueeze(1), ycoord.unsqueeze(1)), 1)
    modified(xrange, yrange, grid, net, layer, node)
