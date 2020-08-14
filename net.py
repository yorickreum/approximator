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
    def __init__(self, n_hidden_layers, n_hidden_neurons, device=approximator.DEVICE):
        super(ApproximationNet, self).__init__()
        self.device = device

        self.hidden1 = torch.nn.Linear(2, n_hidden_neurons, bias=True)  # hidden layer
        self.hidden1.to(device=self.device)
        # torch.nn.init.xavier_normal_(self.hidden1.weight)

        self.hidden_layers = nn.ModuleList([])
        for i in range(n_hidden_layers):
            new_hidden_layer = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons, bias=True)
            new_hidden_layer.to(device=self.device)
            # torch.nn.init.xavier_normal_(new_hidden_layer.weight)
            self.hidden_layers.append(new_hidden_layer)  # hidden layer

        self.predict = torch.nn.Linear(n_hidden_neurons, 1,  bias=True)  # output layer
        self.predict.to(device=self.device)
        # torch.nn.init.xavier_normal_(self.predict.weight)

    def forward(self, x):
        # @TODO check / maybe change sign of tanh
        x = torch.tanh(self.hidden1(x))  # tanh activation function for first hidden layer
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        # x = -torch.exp(self.predict(x))
        x = self.predict(x)
        return x
