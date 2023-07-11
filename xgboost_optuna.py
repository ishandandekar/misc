import optuna
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


def tune_xgboost_classifier(features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    def objective(trial):
        params = {
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "num_class": len(set(target)),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 0.01, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
        }

        model = xgb.XGBClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False,
            callbacks=[
                optuna.integration.XGBoostPruningCallback(
                    trial, "validation_0-mlogloss"
                )
            ],
        )

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    best_model = xgb.XGBClassifier(**best_params)

    best_model.fit(features, target)

    # Create 'figures' directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)

    # Visualizations
    param_importances = optuna.visualization.plot_param_importances(study)
    param_importances.write_image("figures/param_importances.png")

    parallel_coordinate = optuna.visualization.plot_parallel_coordinate(study)
    parallel_coordinate.write_image("figures/parallel_coordinate.png")

    return best_model, best_params
