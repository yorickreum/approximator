import os
import shutil
import time
from pathlib import Path
from shutil import copyfile

import os
from shutil import copyfile

import numpy as np
import pandas
import torch

from approximator.classes.approximation import Approximation
from approximator.classes.constraint import Constraint
from approximator.classes.discretization import StepsDiscretization
from approximator.classes.net import ApproximationNet
from approximator.classes.problem import Problem, Domain
from approximator.examples.richards.func_lib import get_res
from approximator.utils.visualization import plot_x_y_z

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
                   prepone=True,
                   residual=lambda x, y, prediction: (prediction - (- .615)) ** 2),
        Constraint(identifier="lower boundary", condition=lambda x, y: y == domain.y_min,
                   prepone=True,
                   residual=lambda x, y, prediction: (prediction - (-.615)) ** 2),
        # Constraint(identifier="upper boundary", condition=lambda x, y: y == domain.y_max,
        #            prepone=True,
        #            residual=lambda x, y, prediction: (prediction - (
        #                    -.207 - .408 * torch.exp(-(x / 0.01) ** 2))) ** 2),
        Constraint(identifier="upper boundary", condition=lambda x, y: y == domain.y_max,
                   prepone=True,
                   residual=lambda x, y, prediction: (prediction - (-.207)) ** 2),
        Constraint(identifier="pde", condition=lambda x, y: not (x == 0 or y == domain.y_min or y == domain.y_max),
                   prepone=False,
                   residual=lambda input, prediction: (get_res(input, prediction)) ** 2)
    ]
)

# hyperstudy for heat: 2 hidden, with 25 neurons
# approximation_net = ApproximationNet(n_hidden_layers=2, n_neurons_per_layer=25)
# hyperstudy for richards: 8 hidden, with 21 neurons
# --> funktioniert, loss 1e-5
# hyperstudy for richards, study_1598064702.csv, # 65: 6 hidden, with 19 neurons
approximation_net = ApproximationNet(n_hidden_layers=6, n_neurons_per_layer=19)

approximation = Approximation(
    problem=problem,
    net=approximation_net
)


def load_trained_model(trained_model):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_identifier = os.environ['CONFIGIDENTIFIER']
    approximation.load(os.path.join(dir_path, f"trained_models/{config_identifier}/{trained_model}/net.pt"))


def train_richards(pretrained_model=None):
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    if pretrained_model is None:
        current_model = 1
    else:
        current_model = pretrained_model + 1
        load_trained_model(pretrained_model)
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    # current_model = os.environ['TIMESTEPS'] + "-" + os.environ['SPACESTEPS']
    config_identifier = os.environ['CONFIGIDENTIFIER']
    trained_model_dirpath = Path(f"trained_models/{config_identifier}/{current_model}/")
    if trained_model_dirpath.exists() and trained_model_dirpath.is_dir():
        shutil.rmtree(trained_model_dirpath)  # clean up if training was already started once before
    dir_path.joinpath(trained_model_dirpath).mkdir(parents=True, exist_ok=True)
    copyfile(os.path.join(dir_path, f"richards.py"),
             os.path.join(dir_path, trained_model_dirpath, f"config_richards.py"))
    copyfile(os.path.join(dir_path, f"func_lib.py"),
             os.path.join(dir_path, trained_model_dirpath, f"config_func_lib.py"))
    time_start = int(time.time() * 1000)
    approximation.train(
        learning_rate=1e-3,
        epochs=int(1e7),  # 1e4 boundaries 3e4 all 1e4 random steps
        pretraining_patience=int(1e5),
        training_patience=int(1e5),
        checkpoint_dir_path="/ramdisk",
        model_path=os.path.join(dir_path, trained_model_dirpath, f"net.pt"),
        losses_path=os.path.join(dir_path, trained_model_dirpath, f"losses.csv"),
        discretization=StepsDiscretization(
            x_steps=100,
            y_steps=100,
            x_additional=[domain.x_min],
            y_additional=[domain.y_min, domain.y_max],
        )
    )
    time_end = int(time.time() * 1000)
    print('quick check result:')
    y_testspace = np.linspace(problem.domain.y_min, problem.domain.y_max, 20)
    print([str(approximation.use(domain.x_min, y)) for y in y_testspace])
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
