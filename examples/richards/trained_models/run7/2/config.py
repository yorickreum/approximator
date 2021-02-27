import math
import os
from shutil import copyfile

import pandas
import torch

from approximator.classes.approximation import Approximation
from approximator.classes.constraint import Constraint
from approximator.classes.discretization import StepsDiscretization, RandomStepsDiscretization
from approximator.classes.net import ApproximationNet
from approximator.classes.problem import Problem, Domain
from approximator.examples.richards.func_lib import get_res
from approximator.utils.visualization import plot_approximation, plot_approximation_residuals, plot_x_y_z

domain = Domain(
        x_min=0,  # x is time, t in h
        x_max=.2,
        y_min=0,  # y is spacial, z in m
        y_max=.4
    )
problem = Problem(
        domain,
        [
            Constraint(identifier="initial condition", condition=lambda x, y: x == 0,
                       residual=lambda x, y, prediction: (prediction - (- .615)) ** 2),
            Constraint(identifier="lower boundary", condition=lambda x, y: y == domain.y_min,
                       residual=lambda x, y, prediction: (prediction - (-.615)) ** 2),
            Constraint(identifier="upper boundary", condition=lambda x, y: y == domain.y_max and x > 0,
                       residual=lambda x, y, prediction: (prediction - (-.207)) ** 2),
            Constraint(identifier="pde", condition=lambda x, y: not (x == 0 or y == domain.y_min or y == domain.y_max),
                       residual=lambda input, prediction: (get_res(input, prediction)) ** 2)
        ]
    )

approximation_net = ApproximationNet(n_hidden_layers=8, n_neurons_per_layer=16)

approximation = Approximation(
    problem=problem,
    net=approximation_net
)


def load_trained_model(trained_model):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    approximation.load(os.path.join(dir_path, f"trained_models/{trained_model}/net.pt"))


def train_richards(pretrained_model=None):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if pretrained_model is None:
        current_model = 1
    else:
        current_model = pretrained_model + 1
        load_trained_model(pretrained_model)
    os.mkdir(os.path.join(dir_path, f"trained_models/{current_model}/"))
    copyfile(os.path.join(dir_path, f"richards.py"),
             os.path.join(dir_path, f"trained_models/{current_model}/config.py"))
    approximation.train(
        learning_rate=1e-3,
        epochs=int(5e3),  # 1e4 boundaries 3e4 all 1e4 random steps
        discretization=RandomStepsDiscretization(
            x_steps=50,
            y_steps=50,
            x_additional=[0],
            y_additional=[domain.y_min, domain.y_max],
        )
    )
    print('quick check result:')
    print([str(approximation.use(0, (z / 10) - 1)) for z in range(21)])
    copyfile(f"./run/net.pt", os.path.join(dir_path, f"trained_models/{current_model}/net.pt"))


def plot_richards(trained_model=1):
    load_trained_model(trained_model)
    print('quick check result:')
    print([str(approximation.use(0, z)) for z in range(41)])
    plot_approximation(approximation, xlabel="t / h", ylabel="z / m",
                       title="Approximated solution of the richards equation")


def plot_richards_res(trained_model=1):
    load_trained_model(trained_model)
    plot_approximation_residuals(approximation, xlabel="t / h", ylabel="z / m", title="Residuals")


def plot_richards_difference(trained_model=1, simpeg_csv=''):
    load_trained_model(trained_model)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    simpeg_results = pandas.read_csv(os.path.join(dir_path, simpeg_csv), index_col="t")

    t_space_s_int = simpeg_results.index
    z = simpeg_results.columns
    h_space_m = []
    for t_i, t_s in enumerate(t_space_s_int):
        h_space_m += [[]]
        for z_cm, h_cm in (simpeg_results.iloc[t_s]).iteritems():
            z_m = float(z_cm) / 100
            t_h = t_s / 3600
            h_space_m[t_i] += [approximation.use(t_h, z_m) - (h_cm / 100)]

    t_space_h = [float(t_i) / 3600 for t_i in t_space_s_int]
    z_space_m = [float(z_i) / 100 for z_i in z]

    plot_x_y_z(z_space_m, t_space_h, h_space_m, xlabel="t / h", ylabel="z / m", title="Difference to SimPEG")


def plot_richards_celialike(trained_model=1, t=0.1, z_bottom=0, z_top=0.4):
    load_trained_model(trained_model)
    steps = 100
    import numpy as np
    z_lin_space = np.linspace(z_top, z_bottom, steps)
    net_vals = [approximation.use(t, z) for z in z_lin_space]
    import matplotlib.pyplot as plt
    plot = plt.plot(
        (z_top - z_lin_space),
        net_vals
    )
    plt.title(f'h / m at t = {t} h = {t * 3600} s')
    plt.xlabel("z / m")
    plt.ylabel("h / m")
    plt.show()
