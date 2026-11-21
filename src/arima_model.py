import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

print("\nARIMA FORECASTING MODEL")
print("=======================")

# Load training and testing data
y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Load test month information
test_months = pd.read_csv("data/processed/test_months.csv")

# Function to train ARIMA and make forecasts
def train_arima(train_data, test_size, target_name):

    print(f"\nTraining ARIMA model for {target_name}...")

    # Use ARIMA with order (1, 1, 1)
    model = ARIMA(
        train_data,
        order=(1, 1, 1)
    )

    # Fit the model
    fitted_model = model.fit()

    # Forecast the required number of months
    predictions = fitted_model.forecast(
        steps=test_size
    )

    return predictions

# Forecast HHS care
care_predictions = train_arima(
    y_train["Average_HHS_Care"],
    len(y_test),
    "HHS Care"
)

# Forecast discharges
discharge_predictions = train_arima(
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

# Function to calculate evaluation metrics
def calculate_metrics(actual, predicted):

    mae = np.mean(
        np.abs(actual - predicted)
    )

    rmse = np.sqrt(
        np.mean((actual - predicted) ** 2)
    )

    return mae, rmse

# Evaluate HHS care model
care_mae, care_rmse = calculate_metrics(
    actual_care,
    care_predictions
)

# Evaluate discharge model
discharge_mae, discharge_rmse = calculate_metrics(
    actual_discharges,
    discharge_predictions
)

# Create forecast results
forecast_results = pd.DataFrame({
    "Month": test_months["Month"],
    "Actual_HHS_Care": actual_care,
    "ARIMA_Care_Prediction": care_predictions,
    "Actual_Discharges": actual_discharges,
    "ARIMA_Discharge_Prediction": discharge_predictions
})

print("\nARIMA FORECAST RESULTS")
print("----------------------")

print(
    forecast_results.round(2).to_string(index=False)
)

# Create model performance table
model_comparison = pd.DataFrame({
    "Model": ["ARIMA"],

    "Care_MAE": [care_mae],
    "Care_RMSE": [care_rmse],

    "Discharge_MAE": [discharge_mae],
    "Discharge_RMSE": [discharge_rmse]
})

print("\nARIMA MODEL PERFORMANCE")
print("-----------------------")

print(
    model_comparison.round(2).to_string(index=False)
)

# Create results folder
Path("results").mkdir(exist_ok=True)

# Save forecast results
forecast_results.to_csv(
    "results/arima_forecast_results.csv",
    index=False
)

# Save model performance
model_comparison.to_csv(
    "results/arima_model_comparison.csv",
    index=False
)

print("\nARIMA forecast results saved successfully!")
print("ARIMA model comparison saved successfully!")

print("\nARIMA modelling completed successfully!")
