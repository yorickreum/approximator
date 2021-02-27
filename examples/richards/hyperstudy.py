import os
import time

import optuna

from approximator.classes.approximation import Approximation
from approximator.classes.constraint import Constraint
from approximator.classes.discretization import StepsDiscretization
from approximator.classes.net import ApproximationNet
from approximator.classes.problem import Problem, Domain
from approximator.examples.heat.func_lib import get_res


def objective(trial):
    # generate model, train it
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
                       residual=lambda x, y, prediction: (prediction - (- .615)) ** 2),
            Constraint(identifier="lower boundary", condition=lambda x, y: y == domain.y_min,
                       residual=lambda x, y, prediction: (prediction - (-.615)) ** 2),
            Constraint(identifier="upper boundary", condition=lambda x, y: y == domain.y_max and x > 0,
                       residual=lambda x, y, prediction: (prediction - (-.207)) ** 2),
            Constraint(identifier="pde", condition=lambda x, y: not (x == 0 or y == domain.y_min or y == domain.y_max),
                       residual=lambda input, prediction: (get_res(input, prediction)) ** 2)
        ]
    )

    approximation_net = ApproximationNet(
        n_hidden_layers=trial.suggest_int("n_hidden_layers", 5, 10),
        n_neurons_per_layer=trial.suggest_int("n_neurons_per_layer", 15, 25)
    )

    approximation = Approximation(
        problem=problem,
        net=approximation_net
    )

    approximation.train(
        learning_rate=1e-3,
        epochs=int(1e4),
        discretization=StepsDiscretization(
            x_steps=100,  # x_steps=trial.suggest_int("x_steps", 10, 100),
            y_steps=100,  # y_steps=trial.suggest_int("y_steps", 10, 100),
            x_additional=[0],
            y_additional=[domain.y_min, domain.y_max],
        ),
        verbose_output=False
    )

    accuracy = min(approximation.losses[-10:])

    return accuracy


def richards_hyperstudy():
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=20,
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
    df.to_csv(os.path.join(dir_path, f"optuna/study_{str(int(time.time()))}.csv"))
