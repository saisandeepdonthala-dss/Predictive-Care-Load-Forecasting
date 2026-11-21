import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

print("\nEXPONENTIAL SMOOTHING MODEL")
print("===========================")

# Load training and testing data
y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Load test month information
test_months = pd.read_csv("data/processed/test_months.csv")

# Function to train the model and make predictions
def train_model(train_data, test_size, target_name):

    print(f"\nTraining model for {target_name}...")

    # Create the Exponential Smoothing model
    model = ExponentialSmoothing(
        train_data,
        trend="add",
        seasonal=None,
        initialization_method="estimated"
    )

    # Train the model
    fitted_model = model.fit()

    # Predict the next seven months
    predictions = fitted_model.forecast(test_size)

    return predictions

# Forecast HHS care
care_predictions = train_model(
    y_train["Average_HHS_Care"],
    len(y_test),
    "HHS Care"
)

# Forecast discharges
discharge_predictions = train_model(
    y_train["Average_Discharges"],
    len(y_test),
    "Discharges"
)

# Get actual values
actual_care = y_test["Average_HHS_Care"].values
actual_discharges = y_test["Average_Discharges"].values

# Convert predictions into arrays
care_predictions = np.array(care_predictions)
discharge_predictions = np.array(discharge_predictions)

# Calculate MAE and RMSE
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
    "ES_Care_Prediction": care_predictions,
    "Actual_Discharges": actual_discharges,
    "ES_Discharge_Prediction": discharge_predictions
})

print("\nEXPONENTIAL SMOOTHING FORECAST RESULTS")
print("--------------------------------------")

print(
    forecast_results.round(2).to_string(index=False)
)

# Prepare model performance results
model_comparison = pd.DataFrame({
    "Model": ["Exponential Smoothing"],

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

# Create results folder
Path("results").mkdir(exist_ok=True)

# Save forecast results
forecast_results.to_csv(
    "results/exponential_smoothing_forecast_results.csv",
    index=False
)

# Save model comparison
model_comparison.to_csv(
    "results/exponential_smoothing_model_comparison.csv",
    index=False
)

print("\nForecast results saved successfully!")
print("Model comparison saved successfully!")

print("\nExponential Smoothing modelling completed successfully!")

