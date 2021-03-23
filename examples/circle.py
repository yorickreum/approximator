from approximator.classes.approximation import Approximation
from approximator.classes.discretization import Discretization, StepSizeDiscretization
from approximator.classes.constraint import Constraint
from approximator.classes.net import ApproximationNet
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
        Constraint("in circle constraint", lambda x, y: in_circle(x, y),
                   lambda x, y, prediction: (prediction - 1) ** 2),
        Constraint("out of circle constraint", lambda x, y: not in_circle(x, y),
                   lambda x, y, prediction: (prediction - 0) ** 2)
    ]
)

approximation_net = ApproximationNet(n_hidden_layers=5, n_neurons_per_layer=10)

approximation = Approximation(
    problem=problem,
    net=approximation_net
)


def plot_circle():
    approximation.train(
        discretization=StepSizeDiscretization(
            x_step=.01,
            y_step=.01
        ),
        learning_rate=.001,
        epochs=int(2e3)
    )
    print('quick check result:' + str(approximation.use(.5, .5)))
    plot_approximation(approximation)


plot_circle()