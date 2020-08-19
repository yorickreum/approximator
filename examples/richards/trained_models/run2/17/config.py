import os
from shutil import copyfile

from approximator.classes.approximation import Approximation
from approximator.classes.constraint import Constraint
from approximator.classes.discretization import StepsDiscretization, RandomStepsDiscretization
from approximator.classes.problem import Problem, Domain
from approximator.examples.richards.func_lib import get_res
from approximator.utils.visualization import plot_approximation

problem = Problem(
    Domain(
        x_min=0,  # x is time t in s
        x_max=100,
        y_min=0,  # y is elevation z in cm
        y_max=40  # attention: in SimPEG this is probably 39?
    ),
    [
        Constraint(
            lambda x, y: x == 0,
            lambda x, y, prediction: (prediction - (1.02 * y - 61.5)) ** 2  # linear starting conditions
        ),
        Constraint(
            lambda x, y: y == 40,
            lambda x, y, prediction: (prediction + 20.7) ** 2
        ),
        Constraint(
            lambda x, y: y == 0,
            lambda x, y, prediction: (prediction + 61.5) ** 2
        ),
        Constraint(
            lambda x, y: not (x == 0 or y == 40 or y == 0),
            lambda input, prediction: get_res(input, prediction) ** 2
        )
    ]
)

approximation = Approximation(
    problem=problem,
    discretization=RandomStepsDiscretization(
        x_steps=100,
        y_steps=40,
        x_certain=[0],
        y_certain=[0, 40]
    ),
    n_hidden_layers=10,
    n_neurons_per_layer=40,
    learning_rate=.00001,
    epochs=int(2e3)
)


def train_richards(pretrained_model=16):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if pretrained_model is None:
        current_model = 1
    else:
        current_model = pretrained_model + 1
        approximation.load(os.path.join(dir_path, f"trained_models/{pretrained_model}/net.pt"))
    os.mkdir(os.path.join(dir_path, f"trained_models/{current_model}/"))
    copyfile(os.path.join(dir_path, f"richards.py"),
             os.path.join(dir_path, f"trained_models/{current_model}/config.py"))
    approximation.train()
    print('quick check result:')
    print([str(approximation.use(0, z)) for z in range(41)])
    copyfile(f"./run/net.pt", os.path.join(dir_path, f"trained_models/{current_model}/net.pt"))


def plot_richards(train=False):
    approximation.train() if train else approximation.load()
    print('quick check result:')
    print([str(approximation.use(0, z)) for z in range(41)])
    plot_approximation(approximation)
