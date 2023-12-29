from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

models = [KNeighborsClassifier, DummyClassifier]
params = {}
for model in models:
    try:
        model = model(**params)
        print(model.get_params())
    except Exception as e:
        print(e)
