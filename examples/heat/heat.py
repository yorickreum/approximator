import math
import os
from shutil import copyfile

import torch

import numpy as np

from approximator.classes.approximation import Approximation
from approximator.classes.constraint import Constraint
from approximator.classes.discretization import StepsDiscretization, RandomStepsDiscretization
from approximator.classes.net import ApproximationNet
from approximator.classes.problem import Problem, Domain
from approximator.examples.heat.func_lib import get_res
from approximator.utils.visualization import plot_approximation, plot_approximation_residuals, \
    plot_approximation_deviation, plot_domain

problem = Problem(
    Domain(
        x_min=0,  # x is time
        x_max=1,
        y_min=-1,  # y is spacial
        y_max=+1
    ),
    [
        Constraint(identifier="initial condition", condition=lambda x, y: x == 0,
                   residual=lambda x, y, prediction: (prediction -
                                                      (torch.sin(math.pi * y / 2 + math.pi / 2))) ** 2),
        Constraint(identifier="upper boundary", condition=lambda x, y: y == -1,
                   residual=lambda x, y, prediction: (prediction - 0) ** 2),
        Constraint(identifier="lower boundary", condition=lambda x, y: y == +1,
                   residual=lambda x, y, prediction: (prediction - 0) ** 2),
        Constraint(identifier="pde", condition=lambda x, y: not (x == 0 or y == -1 or y == +1),
                   residual=lambda input, prediction: (get_res(input, prediction)) ** 2)
    ]
)

approximation_net = ApproximationNet(n_hidden_layers=2, n_neurons_per_layer=25)

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
        learning_rate=1e-3,
        epochs=int(5e4),
        discretization=StepsDiscretization(
            x_steps=100,
            y_steps=200,
            x_additional=[0],
            y_additional=[-1, 1],
        ),
        verbose_output=True
    )
    print('quick check result:')
    print([str(approximation.use(0, (z / 10) - 1)) for z in range(21)])
    copyfile(f"./run/net.pt", os.path.join(dir_path, f"trained_models/{current_model}/net.pt"))


def plot_heat(trained_model=1):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    approximation.load(os.path.join(dir_path, f"trained_models/{trained_model}/net.pt"))
    print('quick check result:')
    print([str(approximation.use(0, z)) for z in range(41)])
    plot_approximation(approximation, xlabel="t", ylabel="z", title="Approximated solution of the heat equation")


def plot_heat_res(trained_model=1):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    approximation.load(os.path.join(dir_path, f"trained_models/{trained_model}/net.pt"))
    plot_approximation_residuals(approximation, xlabel="t", ylabel="z", title="Residuals")


def plot_heat_difference(trained_model=1):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    approximation.load(os.path.join(dir_path, f"trained_models/{trained_model}/net.pt"))

    def analytically(t, z):
        return math.exp(-math.pi ** 2 * t / 4) * math.cos(math.pi * z / 2)

    plot_approximation_deviation(approximation, analytically, xlabel="t", ylabel="z", title="Difference to "
                                                                                            "analytical solution")


def plot_heat_direct_difference(trained_model=1):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    approximation.load(os.path.join(dir_path, f"trained_models/{trained_model}/net.pt"))

    def analytically(t, z):
        return math.exp(-math.pi ** 2 * t / 4) * math.cos(math.pi * z / 2)

    def deviation(x, y):
        # try:
        #     return (appoximation.use(x, y) / ref_func(x, y)) - 1
        # except ZeroDivisionError:
        #     return None
        return math.fabs(approximation.use(x, y) - analytically(x, y))

    plot_domain(
        approximation,
        deviation,
        xlabel="t", ylabel="z", title="Difference to "
                                      "analytical solution")


def calculate_accuracy(trained_model=1):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    approximation.load(os.path.join(dir_path, f"trained_models/{trained_model}/net.pt"))

    # evaluate accuracy
    def analytically(t, z):
        return math.exp(-math.pi ** 2 * t / 4) * math.cos(math.pi * z / 2)

    x_space = np.linspace(0, 1, 100)
    y_space = np.linspace(-1, 1, 100)
    z_space = [math.fabs(approximation.use(x, y) - analytically(x, y)) for x in x_space for y in y_space]

    accuracy = sum(z_space) / len(z_space)
    print("accuracy: " + str(accuracy))
    return accuracy
