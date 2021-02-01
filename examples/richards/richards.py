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

import numpy as np

domain = Domain(
    x_min=0,  # x is time, t in h
    x_max=.2,
    y_min=0,  # y is spacial, z in m
    y_max=.4
)
problem = Problem(
    domain,
    [
        Constraint(
            condition=lambda x, y: x == 0,
            # residual=lambda x, y, prediction: (prediction - (1.02 * y - .615)) ** 2,  # linear initial conditions
            residual=lambda x, y, prediction: (prediction - (- .615)) ** 2,  # celia initial conditions
            identifier="initial condition"
        ),
        Constraint(
            condition=lambda x, y: y == domain.y_min,
            residual=lambda x, y, prediction: (prediction - (-.615)) ** 2,
            identifier="lower boundary"
        ),
        Constraint(
            condition=lambda x, y: y == domain.y_max,
            # residual=lambda x, y, prediction: (prediction - (-.207)) ** 2,
            residual=lambda x, y, prediction: (prediction - (-.207 - .408 * torch.exp(-(x / 0.0001) ** 2))) ** 2,
            identifier="upper boundary"
        ),
        Constraint(
            condition=lambda x, y: not (x == 0 or y == domain.y_min or y == domain.y_max),
            residual=lambda input, prediction: (get_res(input, prediction)) ** 2,
            identifier="pde"
        )
    ]
)

# hyperstudy for heat: 2 hidden, with 25 neurons
# approximation_net = ApproximationNet(n_hidden_layers=2, n_neurons_per_layer=25)
# hyperstudy for richards: 8 hidden, with 21 neurons
approximation_net = ApproximationNet(n_hidden_layers=8, n_neurons_per_layer=21)
# --> funktioniert, loss 1e-5

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
        epochs=int(1e5),  # 1e4 boundaries 3e4 all 1e4 random steps
        discretization=StepsDiscretization(
            x_steps=100,
            y_steps=100,
            x_additional=[domain.x_min],
            y_additional=[domain.y_min, domain.y_max],
        )
    )
    print('quick check result:')
    print([str(approximation.use(0, (z / 10) - 1)) for z in range(21)])
    copyfile(f"./run/net.pt", os.path.join(dir_path, f"trained_models/{current_model}/net.pt"))


def plot_richards(trained_model=1):
    load_trained_model(trained_model)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    t_space_h, z_space_m = \
        np.linspace(problem.domain.x_min, problem.domain.x_max, 100), \
        np.linspace(problem.domain.y_min, problem.domain.y_max, 100)
    h_space_m = []
    for z_i, z in enumerate(z_space_m):
        h_space_m += [[]]
        for t_i, t in enumerate(t_space_h):
            h_space_m[z_i] += [approximation.use(t, z)]

    # plot_x_y_z(z_space_m, t_space_h, h_space_m, xlabel="t / h", ylabel="z / m", title="Difference to SimPEG")

    plot_x_y_z(
        list(map((lambda x: x * 60), t_space_h)),
        list(map((lambda x: x * 100), z_space_m)),
        list(map((lambda l: list(map((lambda x: x * 100), l))), h_space_m)),
        xlabel="Time $t$ [min]", ylabel="Depth $z$ [cm]",
        title="Approximator solution for $h$ [cm]")


def plot_richards_res(trained_model=1):
    load_trained_model(trained_model)

    x_resolution = 100
    y_resolution = 100

    x_space, y_space = \
        np.linspace(problem.domain.x_min, problem.domain.x_max, x_resolution), \
        np.linspace(problem.domain.y_min, problem.domain.y_max, y_resolution)
    z_space = []
    for y_i, y in enumerate(y_space):
        z_space += [[]]
        for x_i, x in enumerate(x_space):
            z_space[y_i] += [approximation.res(x, y)]

    x_space_min = x_space * 60
    y_space_cm = y_space * 100
    z_space_cm = np.array(z_space) * 100

    plot_x_y_z(
        x_space_min,
        y_space_cm,
        z_space_cm,
        xlabel="t / min",
        ylabel="z / cm",
        title="PDE residuals [cm]")


def plot_richards_difference(trained_model=1, simpeg_csv=''):
    load_trained_model(trained_model)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    simpeg_results = pandas.read_csv(
        os.path.join(dir_path, f'./results_celia_720s.csv'),
        names=[str(i) for i in range(41)])

    h_vals_simpeg = (simpeg_results.to_numpy()).T  # h in cm
    t_vals_simpeg = [i for i in range(721)]  # t in s
    z_vals_simpeg = [i for i in range(41)]  # z in cm

    h_total = []
    for t_index, t in enumerate(t_vals_simpeg):
        h_total += [[]]
        for z_index, z in enumerate(z_vals_simpeg):
            # h_total[t_index] += [approximation.use(t / 3600, z / 100)]
            # h_total[t_index] += [h_vals_simpeg[z_index, t_index]]
            h_total[t_index] += [
                approximation.use(t / 3600, z / 100) * 100 - h_vals_simpeg[z_index, t_index]
            ]

    t_space_min = [i / 60 for i in t_vals_simpeg]
    z_space_cm = [i for i in z_vals_simpeg]
    h_vals_cm = np.array(h_total).T

    plot_x_y_z(
        t_space_min,
        z_space_cm,
        h_vals_cm,
        xlabel="$t$ [min]", ylabel="$z$ [cm]",
        title="Difference of $h$ [cm]")


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
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    plt.title(f'h / m at t = {t} h = {t * 60} min')
    plt.xlabel("z / m")
    plt.ylabel("h / m")
    plt.show()
