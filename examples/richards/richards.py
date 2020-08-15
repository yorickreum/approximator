import torch
import numpy as np

import approximator
from approximator.classes.approximation import Approximation
from approximator.classes.discretization import Discretization, StepsDiscretization
from approximator.classes.constraint import Constraint
from approximator.classes.problem import Problem, Domain
from approximator.examples.richards.func_lib import get_res
from approximator.utils.visualization import plot_approximation

problem = Problem(
    Domain(
        x_min=0,  # x is time t in s
        x_max=100,
        y_min=0,  # y is elevation z in cm
        y_max=40  # attention: in SimPEG this is probably 39?
    ),
    [
        Constraint(
            lambda x, y: x == 0,
            lambda x, y, prediction: (prediction - (1.02 * y - 61.5)) ** 2  # linear starting conditions
        ),
        Constraint(
            lambda x, y: y == 40,
            lambda x, y, prediction: (prediction + 20.7) ** 2
        ),
        Constraint(
            lambda x, y: y == 0,
            lambda x, y, prediction: (prediction + 61.5) ** 2
        ),
        Constraint(
            lambda x, y: not (x == 0 or y == 40 or y == 0),
            lambda input, prediction: get_res(input, prediction) ** 2
        )
    ]
)

approximation = Approximation(
    problem=problem,
    discretization=StepsDiscretization(
        x_steps=400,
        y_steps=200
    ),
    n_hidden_layers=10,
    n_neurons_per_layer=40,
    learning_rate=.0001,
    epochs=int(1e5)
)


def plot_richards(train=False):
    approximation.train() if train else approximation.load()
    print('quick check result:')
    print([str(approximation.use(0, z)) for z in range(41)])
    plot_approximation(approximation)
