import shap
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

# Load Boston House Prices dataset
boston = load_boston()
data = boston.data
target = boston.target
feature_names = boston.feature_names

# Train a Random Forest Regressor
model = RandomForestRegressor()
model.fit(data, target)

# Create a SHAP explainer object
explainer = shap.Explainer(model)

# Calculate SHAP values for all data points
shap_values = explainer.shap_values(data)

# Plot summary plot of SHAP values
shap.summary_plot(shap_values, data, feature_names=feature_names)
