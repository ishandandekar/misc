import polars as pl

df_v1 = pl.DataFrame(
    {
        "a": list(range(1, 51, 1)),
        "b": list(range(1, 51, 1)),
        "c": list(range(1, 51, 1)),
    }
)
df_v2 = pl.DataFrame({"a": [2], "b": [4], "c": [6]})
df_vertical_concat = pl.concat(
    [
        df_v1,
        df_v2,
    ],
    how="vertical",
)
print(df_vertical_concat.shape)
print(type(df_vertical_concat))

from sklearn.model_selection import train_test_split

X, y = df_vertical_concat.drop("c"), df_vertical_concat.select("c")
X_ds, test_X, y_train, y_test = train_test_split(X, y, test_size=0.1)
print(type(X_ds))
print(type(test_X))
print(type(y_train))
