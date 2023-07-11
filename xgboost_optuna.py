import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_diabetes


def tune_xgboost_regressor(features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.1, random_state=42
    )

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": "gbtree",
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 0.01, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
            "callbacks": [
                optuna.integration.XGBoostPruningCallback(trial, "validation_0-rmse")
            ],
        }

        model = xgb.XGBRegressor(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False,
        )

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    best_model = xgb.XGBRegressor(**best_params)

    best_model.fit(features, target)

    return best_model, best_params


# Load Iris dataset
raw = load_diabetes()
data = pd.DataFrame(data=raw.data, columns=raw.feature_names)
target = raw.target

# Use the tune_xgboost_regressor function
best_model, best_params = tune_xgboost_regressor(data, target)

# Print the best parameters found
print("Best Parameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Use the best model for predictions
# (Example: Predict the target variable for the first 5 data points)
predictions = best_model.predict(data.iloc[:5])
print("Predictions:")
print(predictions)
