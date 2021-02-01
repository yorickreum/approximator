import math
import os
from shutil import copyfile

import torch

from approximator.classes.approximation import Approximation
from approximator.classes.constraint import Constraint
from approximator.classes.discretization import StepsDiscretization, RandomStepsDiscretization
from approximator.classes.net import ApproximationNet
from approximator.classes.problem import Problem, Domain
from approximator.examples.richards.func_lib import get_res
from approximator.utils.visualization import plot_approximation, plot_approximation_residuals

problem = Problem(
    Domain(
        x_min=0,  # x is time, t in h
        x_max=.2,
        y_min=0,  # y is spacial, z in m
        y_max=.4
    ),
    [
        Constraint(
            condition=lambda x, y: x == 0,
            residual=lambda x, y, prediction: (prediction - (1.02 * y - .615)) ** 2,  # linear starting conditions
            identifier="initial condition"
        ),
        Constraint(
            condition=lambda x, y: y == 0,
            residual=lambda x, y, prediction: (prediction - (-.615)) ** 2,
            identifier="upper boundary"
        ),
        Constraint(
            condition=lambda x, y: y == .4,
            residual=lambda x, y, prediction: (prediction - (-.207)) ** 2,
            identifier="lower boundary"
        ),
        Constraint(
            condition=lambda x, y: not (x == 0 or y == 0 or y == .4),
            residual=lambda input, prediction: (get_res(input, prediction)) ** 2,
            identifier="pde"
        )
    ]
)

approximation_net = ApproximationNet(n_hidden_layers=20, n_neurons_per_layer=40)

approximation = Approximation(
    problem=problem,
    net=approximation_net
)


def train_richards(pretrained_model=None):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if pretrained_model is None:
        current_model = 1
    else:
        current_model = pretrained_model + 1
        approximation.load(os.path.join(dir_path, f"trained_models/{pretrained_model}/net.pt"))
    os.mkdir(os.path.join(dir_path, f"trained_models/{current_model}/"))
    copyfile(os.path.join(dir_path, f"richards.py"),
             os.path.join(dir_path, f"trained_models/{current_model}/config.py"))
    approximation.train(
        learning_rate=1e-3,
        epochs=int(2e3),  # 1e4 boundaries 3e4 all 1e4 random steps
        discretization=StepsDiscretization(
            x_steps=100,
            y_steps=100,
            x_additional=[0],
            y_additional=[0, .4]
        )
    )
    print('quick check result:')
    print([str(approximation.use(0, (z / 10) - 1)) for z in range(21)])
    copyfile(f"./run/net.pt", os.path.join(dir_path, f"trained_models/{current_model}/net.pt"))


def plot_richards(trained_model=1):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    approximation.load(os.path.join(dir_path, f"trained_models/{trained_model}/net.pt"))
    print('quick check result:')
    print([str(approximation.use(0, z)) for z in range(41)])
    plot_approximation(approximation, xlabel="t", ylabel="z", title="Approximated solution of the richards equation")


def plot_richards_res(trained_model=1):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    approximation.load(os.path.join(dir_path, f"trained_models/{trained_model}/net.pt"))
    plot_approximation_residuals(approximation, xlabel="t", ylabel="z", title="Residuals")
