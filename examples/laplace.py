import torch
import numpy as np

import approximator
from approximator.classes.approximation import Approximation
from approximator.classes.discretization import Discretization, StepsDiscretization
from approximator.classes.constraint import Constraint
from approximator.classes.problem import Problem, Domain
from approximator.utils.visualization import plot_approximation

problem = Problem(
    Domain(
        x_min=0,
        x_max=1,
        y_min=0,
        y_max=1
    ),
    [
        Constraint(
            lambda x, y: x == 0 or x == 1 or y == 0,
            lambda x, y, prediction: (prediction - 0) ** 2
        ),
        Constraint(
            lambda x, y: y == 1,
            lambda x, y, prediction: (prediction - torch.sin(np.pi * x)) ** 2
        ),
        Constraint(
            lambda x, y: not (x == 0 or x == 1 or y == 0 or y == 1),
            lambda input, prediction: laplace(input, prediction) ** 2
        )
    ]
)

approximation = Approximation(
    problem=problem,
    discretization=StepsDiscretization(
        x_steps=40,
        y_steps=40
    ),
    n_hidden_layers=5,
    n_neurons_per_layer=10,
    learning_rate=.001,
    epochs=int(1e4)
)


def laplace(input, prediction):
    ones = torch.unsqueeze(torch.ones(len(input), dtype=approximator.DTYPE, device=approximator.DEVICE), 1)
    prediction_d = torch.autograd.grad(prediction, input, create_graph=True, grad_outputs=ones)[0]

    prediction_dx = prediction_d[:, 0:1]
    prediction_dy = prediction_d[:, 1:2]

    prediction_dxx = torch.autograd.grad(prediction_dx, input, create_graph=True, grad_outputs=ones, )
    prediction_dxx = (prediction_dxx[0])[:, 0:1]

    prediction_dyy = (torch.autograd.grad(prediction_dy, input, create_graph=True, grad_outputs=ones, ))
    prediction_dyy = (prediction_dyy[0])[:, 1:2]

    prediction_laplace = prediction_dxx + prediction_dyy

    return prediction_laplace


def plot_laplace():
    approximation.train()
    print('quick check result:' + str(approximation.use(.5, .5)))
    plot_approximation(approximation)
