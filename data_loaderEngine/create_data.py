from sklearn.datasets import load_iris

X, _ = load_iris(return_X_y=True, as_frame=True)
X.columns = ["sl", "sw", "pl", "pw"]
X.to_csv("data/iris.csv", index=False)
