import math
import os
import time
from pathlib import Path

import numpy as np
import optuna
import torch

from approximator.classes.approximation import Approximation
from approximator.classes.constraint import Constraint
from approximator.classes.discretization import StepsDiscretization
from approximator.classes.net import ApproximationNet
from approximator.classes.problem import Problem, Domain
from approximator.examples.richards.func_lib import get_res

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))


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
                       prepone=True,
                       residual=lambda x, y, prediction: (prediction - (- .615)) ** 2),
            Constraint(identifier="lower boundary", condition=lambda x, y: y == domain.y_min,
                       prepone=True,
                       residual=lambda x, y, prediction: (prediction - (-.615)) ** 2),
            Constraint(identifier="upper boundary", condition=lambda x, y: y == domain.y_max,
                       prepone=True,
                       residual=lambda x, y, prediction: (prediction - (
                               -.207 - .408 * torch.exp(-(x / 0.001) ** 2))) ** 2),
            Constraint(identifier="pde", condition=lambda x, y: not (x == 0 or y == domain.y_min or y == domain.y_max),
                       prepone=False,
                       residual=lambda input, prediction: (get_res(input, prediction)) ** 2)
        ]
    )

    suggested_n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 20)
    suggested_n_neurons_per_layer = trial.suggest_int("n_neurons_per_layer", 1, 40)

    approximation_net = ApproximationNet(
        n_hidden_layers=suggested_n_hidden_layers,
        n_neurons_per_layer=suggested_n_neurons_per_layer
    )

    approximation = Approximation(
        problem=problem,
        net=approximation_net
    )

    trial_dir = dir_path.joinpath(f"./trials/{trial.datetime_start}")
    trial_dir.mkdir(exist_ok=True)

    time_start = int(time.time() * 1000)
    approximation.train(
        learning_rate=1e-3,
        epochs=int(1e7),
        pretraining_patience=int(5e4),
        training_patience=int(5e4),
        checkpoint_dir_path="/ramdisk",
        model_path=trial_dir.joinpath(f"./net.pt"),
        losses_path=trial_dir.joinpath(f"./losses.csv"),
        discretization=StepsDiscretization(
            x_steps=100,
            y_steps=100,
            x_additional=[domain.x_min],
            y_additional=[domain.y_min, domain.y_max],
        ),
        verbose_output=False
    )
    time_end = int(time.time() * 1000)

    accuracy = (approximation.pretraining_best_loss if not (approximation.pretraining_best_loss is None) else float('nan'))

    training_summary = np.array([np.concatenate([
        [time_start],
        [time_end],
        [approximation.pretraining_best_loss if not (approximation.pretraining_best_loss is None) else float('nan')],
        [approximation.pretraining_best_loss_epoch if not (approximation.pretraining_best_loss_epoch is None) else -1],
        [approximation.training_best_loss if not (approximation.training_best_loss is None) else float('nan')],
        [approximation.training_best_loss_epoch if not (approximation.training_best_loss_epoch is None) else -1],
        [accuracy],
        [t.item() for t in approximation.latest_residuals]
    ]).ravel()])
    np.savetxt(
        trial_dir.joinpath(f"./training.csv"),
        training_summary,
        delimiter=",",
        fmt=(",".join(np.concatenate([
            ["%.i", "%.i"],
            ["%.18e", "%.i"],
            ["%.18e", "%.i"],
            ["%.18e"],
            ["%.18e" for _ in approximation.latest_residuals],
        ]).ravel())),
        header=(",".join(np.concatenate([
            ["time start", "time end"],
            ["pretraining best loss", "pretraining best loss epoch"],
            ["training best loss", "training best loss epoch"],
            ["accuracy"],
            ["residual " + str(index + 1) for index, _ in
             enumerate(approximation.latest_residuals)]
        ])))
    )

    return accuracy


def richards_hyperstudy():
    study_id = str(int(time.time()))
    print("study id: " + study_id)
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # storage_file = os.path.join(dir_path, f"optuna/net.pt")

    # study = optuna.create_study(
    #     direction="minimize",
    #     storage=f"sqlite:///examples/richards/optuna/study_{study_id}.db"
    # )
    study = optuna.load_study(
        study_name="kubernetes",
        storage=f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@{os.environ['STUDREUM_HYPER_POSTGRES_SERVICE_HOST']}:5432/{os.environ['POSTGRES_DB']}"
    )
    study.optimize(
        objective,
        n_trials=2,
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

    # time
    now = str(time.time())
    studyname = f"study_{study_id}_t_{now}"
    # export study
    # as csv
    df = study.trials_dataframe()
    df.to_csv(dir_path.joinpath(f"optuna/{studyname}.csv"))
