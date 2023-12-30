from optuna.trial import Trial
import optuna
from pprint import pprint
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from box import Box, BoxList
from pathlib import Path
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
ModelFactory = {
    "lr": LogisticRegression,
    "knn": KNeighborsClassifier,
    "svm": SVC,
    "voting": VotingClassifier,
}

params_filepath = Path("hparams.yaml")
with open(params_filepath, "r") as f_in:
    hparams = yaml.safe_load(f_in)
    hparams = Box(hparams.get("tune"))
models: BoxList = hparams.get("models").to_list()
for model_param_item in models:
    model_name, model_params = (
        list(model_param_item.keys())[0],
        list(model_param_item.values())[0].get("params"),
    )

    def objective(trial: Trial):
        params = dict()
        for k, v in model_params.items():
            args = v.get("args")
            args["name"] = k

            if v.get("strategy") == "float":
                params[k] = trial.suggest_float(**args)
            elif v.get("strategy") == "int":
                params[k] = trial.suggest_int(**args)
            elif v.get("strategy") == "cat":
                params[k] = trial.suggest_categorical(**args)
        model = ModelFactory.get(model_name)
        model = model(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=hparams.get("n_trials"))
    pprint(study.best_params)
