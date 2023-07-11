import optuna
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def tune_random_forest_regressor(features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state": 42,
        }

        model = RandomForestRegressor(**params)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    best_model = RandomForestRegressor(**best_params)

    best_model.fit(features, target)

    # Visualizations
    fig, ax = plt.subplots(figsize=(8, 6))
    optuna.visualization.plot_param_importances(study, ax=ax)
    plt.show()

    optuna.visualization.plot_parallel_coordinate(study)
    plt.show()

    return best_model, best_params


# Load Boston House Prices dataset
boston = load_boston()
data = boston.data
target = boston.target

# Use the tune_random_forest_regressor function
best_model, best_params = tune_random_forest_regressor(data, target)

# Print the best parameters found
print("Best Parameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Make predictions using the best model
# (Example: Predict the target variable for the first 5 data points)
predictions = best_model.predict(data[:5])
print("Predictions:")
print(predictions)
