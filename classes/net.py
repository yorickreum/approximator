import torch
from torch import nn

import approximator


class ApproximationNet(nn.Module):
    r""" Network to calculate pressure head h
     and hereby solve RRE """
    # optimizer = None
    losses = []
    pde_loss = []
    bc_top_losses = []
    bc_bottom_losses = []
    bc_initial_losses = []

    # net input 1 = t
    # net input 2 = z
    def __init__(self, n_hidden_layers, n_neurons_per_layer, device=approximator.DEVICE):
        super(ApproximationNet, self).__init__()
        self.device = device

        self.hidden1 = torch.nn.Linear(2, n_neurons_per_layer, bias=True)  # hidden layer
        self.hidden1.to(device=self.device, dtype=approximator.DTYPE)
        # torch.nn.init.xavier_normal_(self.hidden1.weight)

        self.hidden_layers = nn.ModuleList([])
        for i in range(n_hidden_layers):
            new_hidden_layer = torch.nn.Linear(n_neurons_per_layer, n_neurons_per_layer, bias=True)
            new_hidden_layer.to(device=self.device, dtype=approximator.DTYPE)
            # torch.nn.init.xavier_normal_(new_hidden_layer.weight)
            self.hidden_layers.append(new_hidden_layer)  # hidden layer

        self.predict = torch.nn.Linear(n_neurons_per_layer, 1, bias=True)  # output layer
        self.predict.to(device=self.device, dtype=approximator.DTYPE)
        # torch.nn.init.xavier_normal_(self.predict.weight)

    def forward(self, x):
        # @TODO check / maybe change sign of tanh
        x = self.hidden1(x)
        # no activation function for first hidden layer to avoid problems if x is not normalized
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        # x = -torch.exp(self.predict(x))
        x = self.predict(x)
        return x
