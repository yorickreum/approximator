import torch

from approximator.approximation import Approximation, Discretization
from approximator.constraint import Constraint
from approximator.problem import Problem, Domain
from approximator.visualization import plot_approximation

problem = Problem(
    Domain(
        x_min=0,
        x_max=1,
        y_min=0,
        y_max=1
    ),
    [
        Constraint(
            lambda x, y: True,
            lambda x, y, prediction: (prediction - torch.sin(x) * torch.cos(y))**2
        )
    ]
)

approximation = Approximation(
    problem=problem,
    discretization=Discretization(
        x_step=.05,
        y_step=.05
    ),
    n_hidden_layers=5,
    n_neurons=10,
    learning_rate=.001,
    epochs=int(1e3)
)

approximation.train()

print('result:' + str(approximation.use(.5, .5)))

plot_approximation(approximation)