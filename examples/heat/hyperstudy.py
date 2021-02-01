import math
import os
import time

import numpy as np
import optuna
import torch

from approximator.classes.approximation import Approximation
from approximator.classes.constraint import Constraint
from approximator.classes.discretization import StepsDiscretization
from approximator.classes.net import ApproximationNet
from approximator.classes.problem import Problem, Domain
from approximator.examples.heat.func_lib import get_res


def objective(trial):
    # generate model, train it
    problem = Problem(
        Domain(
            x_min=0,  # x is time
            x_max=1,
            y_min=-1,  # y is spacial
            y_max=+1
        ),
        [
            Constraint(
                condition=lambda x, y: x == 0,
                residual=lambda x, y, prediction: (prediction -
                                                   (torch.sin(math.pi * y / 2 + math.pi / 2))) ** 2,
                identifier="initial condition"
            ),
            Constraint(
                condition=lambda x, y: y == -1,
                residual=lambda x, y, prediction: (prediction - 0) ** 2,
                identifier="upper boundary"
            ),
            Constraint(
                condition=lambda x, y: y == +1,
                residual=lambda x, y, prediction: (prediction - 0) ** 2,
                identifier="lower boundary"
            ),
            Constraint(
                condition=lambda x, y: not (x == 0 or y == -1 or y == +1),
                residual=lambda input, prediction: (get_res(input, prediction)) ** 2,
                identifier="pde"
            )
        ]
    )

    approximation_net = ApproximationNet(
        n_hidden_layers=trial.suggest_int("n_hidden_layers", 1, 20),
        n_neurons_per_layer=trial.suggest_int("n_neurons_per_layer", 1, 40)
    )

    approximation = Approximation(
        problem=problem,
        net=approximation_net
    )

    approximation.train(
        learning_rate=1e-3,
        epochs=int(1e3),
        discretization=StepsDiscretization(
            x_steps=100,
            y_steps=200,
            x_additional=[0],
            y_additional=[-1, 1],
        ),
        verbose_output=True
    )

    # evaluate accuracy
    def analytically(t, z):
        return math.exp(-math.pi ** 2 * t / 4) * math.cos(math.pi * z / 2)

    x_space = np.linspace(0, 1, 100)
    y_space = np.linspace(-1, 1, 100)
    z_space = [math.fabs(approximation.use(x, y) - analytically(x, y)) for x in x_space for y in y_space]

    accuracy = sum(z_space) / len(z_space)

    return accuracy


def heat_hyperstudy():
    study_id = str(int(time.time()))
    print("study id: " + study_id)
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # storage_file = os.path.join(dir_path, f"optuna/net.pt")
    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///examples/heat/optuna/study_{study_id}.db"
    )
    study.optimize(
        objective,
        n_trials=800,
        # timeout=(30 * 60),  # in seconds, for complete study not per trial
        show_progress_bar=True
    )

    pruned_trials = [t for t in study.trials if t.state == optuna.study.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.study.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # export study as csv
    df = study.trials_dataframe()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df.to_csv(os.path.join(dir_path, f"optuna/study_{study_id}.csv"))
