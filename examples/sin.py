import torch

from approximator.classes.approximation import Approximation
from approximator.classes.discretization import Discretization, StepSizeDiscretization
from approximator.classes.constraint import Constraint
from approximator.classes.problem import Domain, Problem
from approximator.utils.visualization import plot_approximation

problem = Problem(
    Domain(
        x_min=0,
        x_max=1,
        y_min=0,
        y_max=1
    ),
    [
        Constraint(nope, lambda x, y: True, lambda x, y, prediction: (prediction - torch.sin(x) * torch.cos(y)) ** 2)
    ]
)

approximation = Approximation(
    problem=problem,
    discretization=StepSizeDiscretization(
        x_step=.01,
        y_step=.01
    ),
    n_hidden_layers=5,
    n_neurons_per_layer=10,
    learning_rate=.001,
    epochs=int(500)
)


def plot_sin():
    approximation.train()
    print('result:' + str(approximation.use(.5, .5)))
    plot_approximation(approximation)
