import optuna


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print(study.best_params)
print(type(study.best_params))
print(type(study.best_value))
