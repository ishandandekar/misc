import optuna
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import pandas as pd


def tune_random_forest_classifier(features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", [None, "sqrt", "log2"]
            ),
            "random_state": 42,
        }

        model = RandomForestClassifier(**params)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    best_model = RandomForestClassifier(**best_params)

    best_model.fit(features, target)

    # Create 'figures' folder if it doesn't exist
    os.makedirs("figures", exist_ok=True)

    # Visualizations
    param_importances = optuna.visualization.plot_param_importances(study)
    param_importances.write_image("figures/param_importances.png")

    parallel_coordinate = optuna.visualization.plot_parallel_coordinate(study)
    parallel_coordinate.write_image("figures/parallel_coordinate.png")

    return best_model, best_params


iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
target = iris.target

# Use the tune_random_forest_classifier function
best_model, best_params = tune_random_forest_classifier(data, target)
