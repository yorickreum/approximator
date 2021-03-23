import os
from pathlib import Path

import numpy as np
import pandas

from approximator.examples.heat.heat import approximation
from approximator.examples.richards.richards import problem
from approximator.utils.visualization import plot_x_y_z

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
# config_id = "richards-5v2"
config_id = "2021-03-19-21-23-53-510723"
fig_dir_path = dir_path.joinpath(f"figs/{config_id}")
fig_dir_path.mkdir(parents=False, exist_ok=True)
# run_path = Path(f"{config_id}/1")
run_path = Path(f"{config_id}")

approximation.load(dir_path.joinpath(run_path).joinpath("net.pt"))


def plot_richards_approximation(approximation):
    t_space_h, z_space_m = \
        np.linspace(problem.domain.x_min, problem.domain.x_max, 100), \
        np.linspace(problem.domain.y_min, problem.domain.y_max, 100)
    h_space_m = []
    for z_i, z in enumerate(z_space_m):
        h_space_m += [[]]
        for t_i, t in enumerate(t_space_h):
            h_space_m[z_i] += [approximation.use(t, z)]

    plot_x_y_z(
        list(map((lambda x: x * 60), t_space_h)),
        list(map((lambda x: x * 100), z_space_m)),
        list(map((lambda l: list(map((lambda x: x * 100), l))), h_space_m)),
        xlabel="Time $t$ [min]", ylabel="Elevation $z$ [cm]",
        title="Approximated solution for $h$ [cm]",
        dir_path=fig_dir_path,
        show=False
    )


plot_richards_approximation(approximation)


def plot_richards_residuum(approximation):
    t_space_h, z_space_m = \
        np.linspace(problem.domain.x_min, problem.domain.x_max, 100), \
        np.linspace(problem.domain.y_min, problem.domain.y_max, 100)
    h_space_m = []
    for z_i, z in enumerate(z_space_m):
        h_space_m += [[]]
        for t_i, t in enumerate(t_space_h):
            h_space_m[z_i] += [approximation.res(t, z)]

    plot_x_y_z(
        list(map((lambda x: x * 60), t_space_h)),
        list(map((lambda x: x * 100), z_space_m)),
        list(map((lambda l: list(map((lambda x: x * 100), l))), h_space_m)),
        xlabel="Time $t$ [min]", ylabel="Elevation $z$ [cm]",
        title="Total residuum",
        dir_path=fig_dir_path,
        show=False
    )


plot_richards_residuum(approximation)


def plot_richards_difference(approximation, simpeg_csv=''):
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
        xlabel="Time $t$ [min]", ylabel="Elevation $z$ [cm]",
        title="Difference of $h$ [cm] to SimPEG",
        dir_path=fig_dir_path,
        show=False
    )


plot_richards_difference(approximation)
