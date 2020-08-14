import itertools

import torch

import approximator
from approximator.net import ApproximationNet

import numpy as np

from approximator.problem import Problem


class Model:

    def __init__(self, problem: Problem, discretization, n_hidden, n_neurons, device=approximator.DEVICE):
        print("Device: ", device)
        self.device = device

        self.net = ApproximationNet(n_hidden_layers=n_hidden,
                                    n_hidden_neurons=n_neurons,
                                    device=self.device)
        if approximator.DTYPE == torch.double:
            self.net.double()
        self.net = self.net.to(device=self.device)

        self.constraints = problem.constraints

        x_space, y_space = \
            np.arange(problem.domain.x_min, problem.domain.x_max, discretization.x_step), \
            np.arange(problem.domain.y_min, problem.domain.y_max, discretization.y_step)

        self.constrained_inputs = []
        for constraint in self.constraints:
            constrained_input = []
            for r in itertools.product(x_space, y_space):
                x, y = [r[0], r[1]]
                if constraint.conditionf(x, y):
                    constrained_input += [[x, y]]
            self.constrained_inputs += [torch.tensor(
                constrained_input,
                dtype=approximator.DTYPE,
                requires_grad=True,
                device=self.device)]

    def loss_func(self):
        predictions = [self.net(constrained_input) for constrained_input in self.constrained_inputs]
        residuals = [self.constraints[i].residualf(self.constrained_inputs[i], prediction)
                     for i, prediction in enumerate(predictions)]
        reduced_residuals = [torch.mean(residual) for residual in residuals]
        loss = sum(reduced_residuals)
        return loss

    # @TODO train on mini-batches, see https://stackoverflow.com/a/45118712/8666556
