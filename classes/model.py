import itertools

import torch

import approximator
from approximator.classes.discretization import Discretization
from approximator.classes.net import ApproximationNet

import numpy as np

from approximator.classes.problem import Problem


class Model:

    def __init__(self, problem: Problem, discretization: Discretization, n_hidden, n_neurons,
                 device=approximator.DEVICE):
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
            discretization.get_x_space(problem.domain.x_min, problem.domain.x_max), \
            discretization.get_y_space(problem.domain.y_min, problem.domain.y_max)

        self.constrained_inputs = []
        for constraint in self.constraints:
            constrained_input = []
            for r in itertools.product(x_space, y_space):
                x, y = [r[0], r[1]]
                if constraint.conditionf(x, y):
                    constrained_input += [[x, y]]
            if len(constrained_input) == 0:
                raise RuntimeWarning("Constraint condition was never met in discretized domain!")
            self.constrained_inputs += [torch.tensor(
                constrained_input,
                dtype=approximator.DTYPE,
                requires_grad=True,
                device=self.device)]

        intersection = set.intersection(*[set(ci) for ci in self.constrained_inputs])
        if len(intersection) > 0:
            raise RuntimeWarning("Multiple constraint conditions apply for at least one point, "
                                 "this can lead to unpredictable behaviour.")

    def loss_func(self):
        predictions = [self.net(constrained_input) for constrained_input in self.constrained_inputs]
        residuals = [self.constraints[i].residualf(self.constrained_inputs[i], prediction)
                     for i, prediction in enumerate(predictions)]
        reduced_residuals = [torch.mean(residual) for residual in residuals]
        loss = sum(reduced_residuals)
        return loss

    # @TODO train on mini-batches, see https://stackoverflow.com/a/45118712/8666556
