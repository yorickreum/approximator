import math
import os
from shutil import copyfile

import torch

from approximator.classes.approximation import Approximation
from approximator.classes.constraint import Constraint
from approximator.classes.discretization import StepsDiscretization, RandomStepsDiscretization
from approximator.classes.net import ApproximationNet
from approximator.classes.problem import Problem, Domain
from approximator.examples.heat.func_lib import get_res
from approximator.utils.visualization import plot_approximation, plot_approximation_residuals, \
    plot_approximation_deviation

problem = Problem(
    Domain(
        x_min=0,  # x is time t in s
        x_max=25,
        y_min=0,  # y is elevation z in cm
        y_max=10
    ),
    [
        Constraint(identifier="initial condition", condition=lambda x, y: x == 0,
                   residual=lambda x, y, prediction: (prediction - (torch.sin(math.pi * y / 10))) ** 2),
        Constraint(identifier="upper boundary", condition=lambda x, y: y == 10,
                   residual=lambda x, y, prediction: (prediction - 0) ** 2),
        Constraint(identifier="lower boundary", condition=lambda x, y: y == 0,
                   residual=lambda x, y, prediction: (prediction - 0) ** 2),
        Constraint(identifier="pde", condition=lambda x, y: not (x == 0 or y == 10 or y == 0),
                   residual=lambda input, prediction: get_res(input, prediction) ** 2)
    ]
)

approximation_net = ApproximationNet(n_hidden_layers=10, n_neurons_per_layer=40)

approximation = Approximation(
    problem=problem,
    net=approximation_net
)


def train_heat(pretrained_model=None):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if pretrained_model is None:
        current_model = 1
    else:
        current_model = pretrained_model + 1
        approximation.load(os.path.join(dir_path, f"trained_models/{pretrained_model}/net.pt"))
    os.mkdir(os.path.join(dir_path, f"trained_models/{current_model}/"))
    copyfile(os.path.join(dir_path, f"heat.py"),
             os.path.join(dir_path, f"trained_models/{current_model}/config.py"))
    approximation.train(
        learning_rate=5e-5,
        epochs=int(2e3),
        discretization=RandomStepsDiscretization(
            x_steps=100,
            y_steps=100,
            x_additional=[0],
            y_additional=[0, 10]
        )
    )
    print('quick check result:')
    print([str(approximation.use(0, z)) for z in range(41)])
    copyfile(f"./run/net.pt", os.path.join(dir_path, f"trained_models/{current_model}/net.pt"))


def plot_heat(trained_model=1):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    approximation.load(os.path.join(dir_path, f"trained_models/{trained_model}/net.pt"))
    print('quick check result:')
    print([str(approximation.use(0, z)) for z in range(41)])
    plot_approximation(approximation)


def plot_heat_res(trained_model=1):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    approximation.load(os.path.join(dir_path, f"trained_models/{trained_model}/net.pt"))
    plot_approximation_residuals(approximation)


def plot_heat_difference(trained_model=1):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    approximation.load(os.path.join(dir_path, f"trained_models/{trained_model}/net.pt"))

    def analytically(t, z):
        return math.exp(-math.pi ** 2 * t / 1600) * math.sin(math.pi * z / 40)

    plot_approximation_deviation(approximation, analytically)
