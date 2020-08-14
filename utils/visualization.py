import matplotlib.pyplot as plt
import numpy as np

from approximator.classes.approximation import Approximation


def plot_approximation(appoximation: Approximation):
    x_resolution = 40
    y_resolution = 40
    z_resolution = 40

    problem = appoximation.problem
    # discretization = appoximation.discretization

    x_space, y_space = \
        np.linspace(problem.domain.x_min, problem.domain.x_max, x_resolution), \
        np.linspace(problem.domain.y_min, problem.domain.y_max, y_resolution)
    z_space = []
    for y_i, y in enumerate(y_space):
        z_space += [[]]
        for x_i, x in enumerate(x_space):
            z_space[y_i] += [appoximation.use(x, y)]

    fig, ax = plt.subplots()
    cs = ax.contourf(x_space, y_space, z_space, levels=z_resolution)
    ax.contour(cs, colors='k')
    cbar = fig.colorbar(cs)  # Make a colorbar for the ContourSet returned by the contourf call
    ax.grid(c='k', ls='-', alpha=0.3)  # Plot grid.
    plt.show()
