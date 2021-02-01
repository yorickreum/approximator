from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from approximator.classes.approximation import Approximation


def plot_x_y_z(x_space, y_space, z_space, xlabel="x", ylabel="y", title=""):
    z_resolution = 100  # contours

    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'figure.autolayout': True})  # or: plt.tight_layout()
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    fig, ax = plt.subplots()
    cs = ax.contourf(x_space, y_space, z_space, levels=z_resolution)
    # ax.contour(cs, colors='k')
    cbar = fig.colorbar(cs)  # Make a colorbar for the ContourSet returned by the contourf call
    ax.grid(c='k', ls='-', alpha=0.3)  # Plot grid.
    ax.set_title(title, size=16, pad=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(f'./run/plt_{title.replace(" ", "_")}.pdf')
    plt.savefig(f'./run/plt_{title.replace(" ", "_")}.png')
    plt.show()


def plot_domain(approximation: Approximation, func: Callable, xlabel="x", ylabel="y", title=""):
    problem = approximation.problem

    x_resolution = 100
    y_resolution = 100

    x_space, y_space = \
        np.linspace(problem.domain.x_min, problem.domain.x_max, x_resolution), \
        np.linspace(problem.domain.y_min, problem.domain.y_max, y_resolution)
    z_space = []
    for y_i, y in enumerate(y_space):
        z_space += [[]]
        for x_i, x in enumerate(x_space):
            z_space[y_i] += [func(x, y)]

    plot_x_y_z(x_space, y_space, z_space, xlabel, ylabel, title)


def plot_approximation(approximation: Approximation, xlabel="x", ylabel="y", title=""):
    plot_domain(approximation, approximation.use, xlabel=xlabel, ylabel=ylabel, title=title)


def plot_approximation_residuals(approximation: Approximation, xlabel="x", ylabel="y", title=""):
    plot_domain(approximation, approximation.res, xlabel=xlabel, ylabel=ylabel, title=title)


def plot_approximation_deviation(approximation: Approximation, ref_func: Callable, xlabel="x", ylabel="y", title=""):
    def deviation(x, y):
        # try:
        #     return (approximation.use(x, y) / ref_func(x, y)) - 1
        # except ZeroDivisionError:
        #     return None
        return approximation.use(x, y) - ref_func(x, y)

    plot_domain(
        approximation,
        deviation,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title)
