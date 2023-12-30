from pprint import pprint
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from box import Box
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

params_filepath = Path("params.yaml")
with open(params_filepath, "r") as f_in:
    params = yaml.safe_load(f_in)
    params = Box(params)
model_names, models = params.models.keys(), params.models
for model_name in model_names:
    model_params = models.get(model_name).params
    if model_name == "voting":
        estimators_list = list()
        for estimator in model_params.estimators:
            model_, model_params_ = (
                list(estimator.keys())[0],
                list(estimator.values())[0].params,
            )
            model__ = ModelFactory.get(model_)
            model__ = model__(**model_params_)
            estimators_list.append((model_, model__))
        model_params["estimators"] = estimators_list
    model = ModelFactory.get(model_name)
    model: BaseEstimator = model(**model_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"Model --> {model_name}\tAccuracy --> {accuracy_score(y_test, preds)}")
