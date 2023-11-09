import shap
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load Boston House Prices dataset
boston = load_boston()
data = boston.data
target = boston.target
feature_names = boston.feature_names

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)

# Train a tuned XGBoost Regressor
best_params = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 100,
}
model = xgb.XGBRegressor(**best_params)
model.fit(X_train, y_train)

# Create a SHAP explainer object
explainer = shap.Explainer(model)

# Calculate SHAP values for the testing set
shap_values = explainer.shap_values(X_test)

# Plot summary plot of SHAP values
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
