import logging
import sys
import time

import optuna
from optuna.samplers import TPESampler

# Add stream handler of stdout to show the messages
# optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "example-study"
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(
    study_name=study_name, storage=storage_name, sampler=TPESampler()
)


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", 1, 10)
    time.sleep(0.001)
    return (x - 2) ** 2 + y


study.optimize(objective, n_trials=100)
print(study.best_params)
