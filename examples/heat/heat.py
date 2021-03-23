import math
import os
import time
from pathlib import Path
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
                   prepone=True,
                   residual=lambda x, y, prediction: (prediction -
                                                      (torch.sin(math.pi * y / 2 + math.pi / 2))) ** 2),
        Constraint(identifier="upper boundary", condition=lambda x, y: y == -1,
                   prepone=True,
                   residual=lambda x, y, prediction: (prediction - 0) ** 2),
        Constraint(identifier="lower boundary", condition=lambda x, y: y == +1,
                   prepone=True,
                   residual=lambda x, y, prediction: (prediction - 0) ** 2),
        Constraint(identifier="pde", condition=lambda x, y: not (x == 0 or y == -1 or y == +1),
                   prepone=False,
                   residual=lambda input, prediction: (get_res(input, prediction)) ** 2)
    ]
)

approximation_net = ApproximationNet(n_hidden_layers=2, n_neurons_per_layer=25)

approximation = Approximation(
    problem=problem,
    net=approximation_net
)


def train_heat(pretrained_model=None):
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    config_identifier = os.environ['CONFIGIDENTIFIER']
    if pretrained_model is None:
        current_model = 1
    else:
        current_model = pretrained_model + 1
        approximation.load(os.path.join(dir_path, f"trained_models/{pretrained_model}/net.pt"))
    trained_model_dirpath = Path(f"trained_models/{config_identifier}/{current_model}/")
    dir_path.joinpath(trained_model_dirpath).mkdir(parents=True, exist_ok=True)
    copyfile(os.path.join(dir_path, f"heat.py"),
             os.path.join(dir_path, trained_model_dirpath, f"config_heat.py"))
    copyfile(os.path.join(dir_path, f"func_lib.py"),
             os.path.join(dir_path, trained_model_dirpath, f"config_func_lib.py"))
    time_start = int(time.time() * 1000)
    approximation.train(
        learning_rate=1e-3,
        epochs=int(1e7),
        pretraining_patience=int(5e4),
        training_patience=int(5e4),
        checkpoint_dir_path="/ramdisk",
        discretization=StepsDiscretization(
            x_steps=100,
            y_steps=200,
            x_additional=[0],
            y_additional=[-1, 1],
        ),
        verbose_output=True
    )
    time_end = int(time.time() * 1000)
    print('quick check result:')
    print([str(approximation.use(0, (z / 10) - 1)) for z in range(21)])
    copyfile(f"./run/net.pt", os.path.join(dir_path, trained_model_dirpath, f"net.pt"))
    training_summary = np.array([np.concatenate([
        [time_start],
        [time_end],
        [approximation.pretraining_best_loss if not (approximation.pretraining_best_loss is None) else float('nan')],
        [approximation.pretraining_best_loss_epoch if not (approximation.pretraining_best_loss_epoch is None) else -1],
        [approximation.training_best_loss if not (approximation.training_best_loss is None) else float('nan')],
        [approximation.training_best_loss_epoch if not (approximation.training_best_loss_epoch is None) else -1],
        [t.item() for t in approximation.latest_residuals]
    ]).ravel()])
    np.savetxt(
        os.path.join(dir_path, trained_model_dirpath, f"training.csv"),
        training_summary,
        delimiter=",",
        fmt=(",".join(np.concatenate([
            ["%.i", "%.i"],
            ["%.18e", "%.i"],
            ["%.18e", "%.i"],
            ["%.18e" for _ in approximation.latest_residuals]
        ]).ravel())),
        header=(",".join(np.concatenate([
            ["time start", "time end"],
            ["pretraining best loss", "pretraining best loss epoch"],
            ["training best loss", "training best loss epoch"],
            ["residual " + str(index + 1) for index, _ in
             enumerate(approximation.latest_residuals)]
        ])))
    )


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
