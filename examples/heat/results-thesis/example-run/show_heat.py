import numpy as np
import math
import os
from pathlib import Path

from approximator.examples.heat.heat import approximation
from approximator.utils.visualization import plot_approximation_residuals, plot_approximation, \
    plot_approximation_deviation

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
fig_dir_path = dir_path.joinpath("figs")
fig_dir_path.mkdir(parents=False, exist_ok=True)
# run_path = Path(f"heat-3/1")
run_path = Path(f"2021-03-15 10:58:29.442608")

approximation.load(dir_path.joinpath(run_path).joinpath("net.pt"))

plot_approximation(
    approximation,
    xlabel="$t$", ylabel="$z$",
    title="Approximated solution of the heat equation",
    dir_path=fig_dir_path,
    show=False
)

plot_approximation_residuals(
    approximation,
    xlabel="$t$", ylabel="$z$",
    title="Residuals",
    dir_path=fig_dir_path,
    show=False
)


def analytically(t, z):
    return math.exp(-math.pi ** 2 * t / 4) * math.cos(math.pi * z / 2)


plot_approximation_deviation(
    approximation,
    analytically,
    xlabel="$t$", ylabel="$z$",
    title="Difference to analytical solution",
    dir_path=fig_dir_path,
    show=False
)


def calculate_accuracy(approximation):
    # evaluate accuracy
    x_space = np.linspace(0, 1, 100)
    y_space = np.linspace(-1, 1, 100)
    z_space = [math.fabs(approximation.use(x, y) - analytically(x, y)) for x in x_space for y in y_space]

    print("min accuracy: " + str(min(z_space)))
    print("max accuracy: " + str(max(z_space)))

    accuracy = sum(z_space) / len(z_space)
    print("accuracy: " + str(accuracy))
    return accuracy


calculate_accuracy(approximation)