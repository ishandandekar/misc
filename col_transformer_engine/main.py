import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
)
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data["TotalCharges"] = data["TotalCharges"].replace(to_replace=" ", value="0")

X, y = data.drop(["Churn"], axis=1), data[["Churn"]]
CAT_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET = ["Churn"]
CAT_COLS_OHE = ["PaymentMethod", "Contract", "InternetService"]
CAT_COLS_OE = list(set(CAT_COLS) - set(CAT_COLS_OHE))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)
transformer = ColumnTransformer(
    [
        ("num_scaler", StandardScaler(), NUM_COLS),
        ("cat_ohe", OneHotEncoder(), CAT_COLS_OHE),
        ("cat_oe", OrdinalEncoder(), CAT_COLS_OE),
    ]
)
transformer.fit(X=X_train)
X_train_transformed = transformer.transform(X_train)
print(type(X_train_transformed))
print(X_train_transformed.shape)
X_test_transformed = transformer.transform(X_test)
print(type(X_test_transformed))
print(X_test_transformed.shape)

train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
transformer_new = ColumnTransformer(
    [
        ("num_scaler", StandardScaler(), NUM_COLS),
        ("cat_ohe", OneHotEncoder(), CAT_COLS_OHE),
        ("cat_oe", OrdinalEncoder(), CAT_COLS_OE),
        ("target_oe", OrdinalEncoder(), TARGET),
    ]
)
transformer_new.fit(train_data)
train_data_trans = transformer_new.transform(train_data)
print(type(train_data_trans))
print(train_data_trans.shape)

import polars as pl

df = (
    pl.scan_csv(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv", dtypes={"TotalCharges": pl.String}
    )
    .with_columns(
        pl.col("TotalCharges").str.replace(pattern=" ", value="0").alias("TotalCharges")
    )
    .collect()
)
train_data, test_data = train_test_split(df, test_size=0.1, random_state=42)
transformer_new = ColumnTransformer(
    transformers=[
        ("num_scaler", StandardScaler(), NUM_COLS),
        ("cat_ohe", OneHotEncoder(), CAT_COLS_OHE),
        ("cat_oe", OrdinalEncoder(), CAT_COLS_OE),
        ("target_oe", OrdinalEncoder(), TARGET),
    ],
)
transformer_new.fit(train_data.to_pandas())
train_data_trans = transformer_new.transform(train_data.to_pandas())
print(type(train_data_trans))
print(train_data_trans.shape)
