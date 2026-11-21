import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

print("\nRANDOM FOREST MODEL")
print("===================")

# Load training and testing datasets
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

test_months = pd.read_csv("data/processed/test_months.csv")


# Create a Random Forest model for HHS care
print("\nTraining Random Forest model for HHS Care...")

care_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

care_model.fit(
    X_train,
    y_train["Average_HHS_Care"]
)

# Predict HHS care values
care_predictions = care_model.predict(X_test)

# Create a Random Forest model for discharges
print("\nTraining Random Forest model for Discharges...")

discharge_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

discharge_model.fit(
    X_train,
    y_train["Average_Discharges"]
)

# Predict discharge values
discharge_predictions = discharge_model.predict(X_test)

# Get actual values
actual_care = y_test["Average_HHS_Care"].values
actual_discharges = y_test["Average_Discharges"].values

# Calculate evaluation metrics
def calculate_metrics(actual, predicted):

    mae = np.mean(
        np.abs(actual - predicted)
    )

    rmse = np.sqrt(
        np.mean((actual - predicted) ** 2)
    )

    return mae, rmse

# Evaluate HHS care predictions
care_mae, care_rmse = calculate_metrics(
    actual_care,
    care_predictions
)

# Evaluate discharge predictions
discharge_mae, discharge_rmse = calculate_metrics(
    actual_discharges,
    discharge_predictions
)

# Prepare forecast results
forecast_results = pd.DataFrame({
    "Month": test_months["Month"],

    "Actual_HHS_Care": actual_care,
    "RF_Care_Prediction": care_predictions,

    "Actual_Discharges": actual_discharges,
    "RF_Discharge_Prediction": discharge_predictions
})

print("\nRANDOM FOREST FORECAST RESULTS")
print("-----------------------------")

print(
    forecast_results.round(2).to_string(index=False)
)

# Prepare model performance results
model_comparison = pd.DataFrame({
    "Model": ["Random Forest"],

    "Care_MAE": [care_mae],
    "Care_RMSE": [care_rmse],

    "Discharge_MAE": [discharge_mae],
    "Discharge_RMSE": [discharge_rmse]
})

print("\nMODEL PERFORMANCE")
print("-----------------")

print(
    model_comparison.round(2).to_string(index=False)
)

# Check which features are important for prediction
care_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": care_model.feature_importances_
})

care_importance = care_importance.sort_values(
    by="Importance",
    ascending=False
)

discharge_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": discharge_model.feature_importances_
})

discharge_importance = discharge_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nIMPORTANT FEATURES FOR HHS CARE")
print("--------------------------------")

print(care_importance.round(4).to_string(index=False))

print("\nIMPORTANT FEATURES FOR DISCHARGES")
print("----------------------------------")

print(discharge_importance.round(4).to_string(index=False))

# Create results folder
Path("results").mkdir(exist_ok=True)

# Save forecast results
forecast_results.to_csv(
    "results/random_forest_forecast_results.csv",
    index=False
)

# Save model comparison
model_comparison.to_csv(
    "results/random_forest_model_comparison.csv",
    index=False
)

# Save feature importance
care_importance.to_csv(
    "results/random_forest_care_feature_importance.csv",
    index=False
)

discharge_importance.to_csv(
    "results/random_forest_discharge_feature_importance.csv",
    index=False
)

print("\nRandom Forest forecast results saved successfully!")
print("Model comparison saved successfully!")
print("Feature importance files saved successfully!")

print("\nRandom Forest modelling completed successfully!")
