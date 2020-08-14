from approximator.classes.approximation import Approximation
from approximator.classes.discretization import Discretization, StepSizeDiscretization
from approximator.classes.constraint import Constraint
from approximator.classes.problem import Problem, Domain
from approximator.utils.visualization import plot_approximation


def in_circle(x, y):
    r = 0.2
    a, b = 0.5, 0.5
    return ((x - a) ** 2 + (y - b) ** 2) ** (1 / 2) <= r


problem = Problem(
    Domain(
        x_min=0,
        x_max=1,
        y_min=0,
        y_max=1
    ),
    [
        Constraint(
            lambda x, y: in_circle(x, y),
            lambda x, y, prediction: (prediction - 1) ** 2
        ),
        Constraint(
            lambda x, y: not in_circle(x, y),
            lambda x, y, prediction: (prediction - 0) ** 2
        )
    ]
)

approximation = Approximation(
    problem=problem,
    discretization=StepSizeDiscretization(
        x_step=.01,
        y_step=.01
    ),
    n_hidden_layers=5,
    n_neurons_per_layer=10,
    learning_rate=.001,
    epochs=int(2e3)
)


def plot_circle():
    approximation.train()
    print('quick check result:' + str(approximation.use(.5, .5)))
    plot_approximation(approximation)
