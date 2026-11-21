import pandas as pd
import numpy as np
from pathlib import Path

print("\nBASELINE FORECASTING MODELS")
print("===========================")

# Load training and testing data
y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Load test month information
test_months = pd.read_csv("data/processed/test_months.csv")

# Store actual test values
actual_care = y_test["Average_HHS_Care"].values
actual_discharges = y_test["Average_Discharges"].values

# Store the last three training values
last_care_values = y_train["Average_HHS_Care"].tail(3).values
last_discharge_values = y_train["Average_Discharges"].tail(3).values

# Create lists to store predictions
naive_care_predictions = []
naive_discharge_predictions = []

moving_care_predictions = []
moving_discharge_predictions = []

# Previous values used by the forecasting models
care_history = list(last_care_values)
discharge_history = list(last_discharge_values)

# Generate predictions one month at a time
for i in range(len(y_test)):

    # Naive forecast uses the previous month's actual value
    care_prediction = care_history[-1]
    discharge_prediction = discharge_history[-1]

    naive_care_predictions.append(care_prediction)
    naive_discharge_predictions.append(discharge_prediction)

    # Moving average uses the previous three actual values
    care_average = np.mean(care_history[-3:])
    discharge_average = np.mean(discharge_history[-3:])

    moving_care_predictions.append(care_average)
    moving_discharge_predictions.append(discharge_average)

    # Update history with the actual value after each month
    # This is a one-step-ahead evaluation
    care_history.append(actual_care[i])
    discharge_history.append(actual_discharges[i])

# Create a results table
results = pd.DataFrame({
    "Month": test_months["Month"],

    "Actual_HHS_Care": actual_care,
    "Naive_Care_Prediction": naive_care_predictions,
    "Moving_Average_Care": moving_care_predictions,

    "Actual_Discharges": actual_discharges,
    "Naive_Discharge_Prediction": naive_discharge_predictions,
    "Moving_Average_Discharges": moving_discharge_predictions
})

print("\nFORECAST RESULTS")
print("----------------")

print(results.round(2).to_string(index=False))

# Calculate evaluation metrics
def calculate_metrics(actual, predicted):

    mae = np.mean(np.abs(actual - predicted))

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    # MAPE is not calculated here because it can be unstable
    # when actual values are zero or close to zero

    return mae, rmse


# Evaluate HHS care predictions
naive_care_mae, naive_care_rmse = calculate_metrics(
    actual_care,
    naive_care_predictions
)

moving_care_mae, moving_care_rmse = calculate_metrics(
    actual_care,
    moving_care_predictions
)

# Evaluate discharge predictions
naive_discharge_mae, naive_discharge_rmse = calculate_metrics(
    actual_discharges,
    naive_discharge_predictions
)

moving_discharge_mae, moving_discharge_rmse = calculate_metrics(
    actual_discharges,
    moving_discharge_predictions
)

# Display model performance
model_comparison = pd.DataFrame({
    "Model": ["Naive", "Moving Average"],

    "Care_MAE": [
        naive_care_mae,
        moving_care_mae
    ],

    "Care_RMSE": [
        naive_care_rmse,
        moving_care_rmse
    ],

    "Discharge_MAE": [
        naive_discharge_mae,
        moving_discharge_mae
    ],

    "Discharge_RMSE": [
        naive_discharge_rmse,
        moving_discharge_rmse
    ]
})

print("\nMODEL PERFORMANCE COMPARISON")
print("----------------------------")

print(model_comparison.round(2).to_string(index=False))

# Save forecast results
Path("results").mkdir(exist_ok=True)

results.to_csv(
    "results/baseline_forecast_results.csv",
    index=False
)

model_comparison.to_csv(
    "results/baseline_model_comparison.csv",
    index=False
)

print("\nBaseline forecast results saved successfully!")
print("Model comparison saved successfully!")

print("\nBaseline modelling completed successfully!")
