import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True, as_frame=True)
y = pd.DataFrame(y, columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=420, stratify=y
)
print(type(X_train), type(X_test), type(y_train), type(y_test))
