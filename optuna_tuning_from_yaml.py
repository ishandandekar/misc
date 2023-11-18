import optuna
import yaml
from box import Box
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load hyperparameters from YAML file using Box for easy access
with open("hyperparameters.yaml", "r") as file:
    hyperparameters = Box(yaml.safe_load(file))

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Define the search space (hyperparameter grid) based on YAML file
def create_study(hyperparameters):
    study = optuna.create_study(direction="maximize")
    study.parameters = {
        key: getattr(optuna.distributions, hyperparameter.distribution)(
            hyperparameter.low, hyperparameter.high
        )
        for key, hyperparameter in hyperparameters.items()
    }
    return study


# Function to adjust hyperparameter values before suggesting
def adjust_hyperparameter_value(trial, key, distribution, low, high):
    if distribution == "FloatDistribution":
        return trial.suggest_float(key, low, high)
    elif distribution == "IntDistribution":
        return trial.suggest_int(key, low, high)
    elif distribution == "FloatDistribution":
        return trial.suggest_float(key, low, high)
    elif distribution == "IntDistribution":
        return trial.suggest_int(key, low, high)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")


# Define the objective function
def objective(trial, hyperparameters):
    # Access the search space parameters
    hyperparameter_values = {
        key: adjust_hyperparameter_value(
            trial,
            key,
            hyperparameter.distribution,
            hyperparameter.low,
            hyperparameter.high,
        )
        for key, hyperparameter in hyperparameters.items()
    }

    # Create a RandomForestClassifier with the suggested hyperparameters
    model = RandomForestClassifier(random_state=42, **hyperparameter_values)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy as the objective to be maximized
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


if __name__ == "__main__":
    # Create the study and define the search space
    study = create_study(hyperparameters)

    # Optimize the objective function
    objective_func = lambda trial: objective(trial, hyperparameters)
    study.optimize(objective_func, n_trials=10)

    # Print the best hyperparameters and their corresponding accuracy
    print("Best trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
