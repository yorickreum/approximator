import matplotlib.pyplot as plt
import numpy as np

from approximator.approximation import Approximation


def plot_approximation(appoximation: Approximation):
    problem = appoximation.problem
    discretization = appoximation.discretization

    x_space, y_space = \
        np.arange(problem.domain.x_min, problem.domain.x_max, discretization.x_step), \
        np.arange(problem.domain.y_min, problem.domain.y_max, discretization.y_step)
    z_space = []
    for y_i, y in enumerate(y_space):
        z_space += [[]]
        for x_i, x in enumerate(x_space):
            z_space[y_i] += [appoximation.use(x, y)]

    fig, ax = plt.subplots()
    cs = ax.contourf(x_space, y_space, z_space)
    ax.contour(cs, colors='k')
    cbar = fig.colorbar(cs)  # Make a colorbar for the ContourSet returned by the contourf call
    ax.grid(c='k', ls='-', alpha=0.3)  # Plot grid.
    plt.show()
