import optuna
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os


def tune_xgboost_classifier(X_train, X_test, y_train, y_test):
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 0.01, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "early_stopping_rounds": 10,
            "random_state": 42,
        }

        model = xgb.XGBClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
            callbacks=[
                optuna.integration.XGBoostPruningCallback(trial, "validation_0-auc")
            ],
        )

        y_pred = model.predict(X_test)
        fscore = f1_score(y_test, y_pred)
        return fscore

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    best_model = xgb.XGBClassifier(**best_params)

    best_model.fit(X_train, y_train)

    # Create 'figures' directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)

    # Visualizations
    param_importances = optuna.visualization.plot_param_importances(study)
    param_importances.to_html("param_importances.html")

    parallel_coordinate = optuna.visualization.plot_parallel_coordinate(study)
    parallel_coordinate.to_html("parallel_coordinate.png")

    return best_model, best_params


X, y = load_iris(return_X_y=True, as_frame=True)
X, y = X[:100], y[:100]
print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
best_model, best_params = tune_xgboost_classifier(X_train, X_test, y_train, y_test)
print("\n\n")
print("#" * 10, end="\n\n")
print(best_params)
