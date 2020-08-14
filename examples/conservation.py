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
            lambda x, y: 1 >= x >= 0 == y,
            lambda x, y, prediction: (prediction - (x ** 2 + torch.exp(- x ** 2))) ** 2
        ),
        Constraint(
            lambda x, y: not (1 >= x >= 0 == y),
            lambda input, prediction: conservation(input, prediction) ** 2
        )
    ]
)

approximation = Approximation(
    problem=problem,
    discretization=StepsDiscretization(
        x_steps=100,
        y_steps=100
    ),
    n_hidden_layers=5,
    n_neurons_per_layer=10,
    learning_rate=.001,
    epochs=int(1e3)
)


def conservation(input, prediction):
    input_x = input[:, 0:1]
    input_y = input[:, 1:2]

    ones = torch.unsqueeze(torch.ones(len(input), dtype=approximator.DTYPE, device=approximator.DEVICE), 1)
    prediction_d = torch.autograd.grad(prediction, input, create_graph=True, grad_outputs=ones)[0]

    prediction_dx = prediction_d[:, 0:1]
    prediction_dy = prediction_d[:, 1:2]

    prediction_conservation = \
        input_x * prediction_dx + prediction_dy - input_x * input_y

    return prediction_conservation


def plot_conservation():
    approximation.train()
    print('quick check result:' + str(approximation.use(.5, .5)))
    plot_approximation(approximation)
