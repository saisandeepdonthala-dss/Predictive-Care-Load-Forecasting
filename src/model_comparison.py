import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def calculate_metrics(actual, predicted):
    """Calculate MAE and RMSE for model evaluation."""

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse

def main():

    print("\nSTANDARDIZED MODEL COMPARISON")
    print("=============================")

    # Load monthly and featured datasets
    monthly_data = pd.read_csv(
        "data/processed/monthly_data.csv",
        parse_dates=["Month"]
    )

    featured_data = pd.read_csv(
        "data/processed/featured_data.csv",
        parse_dates=["Month"]
    )

    monthly_data = monthly_data.sort_values("Month").reset_index(drop=True)
    featured_data = featured_data.sort_values("Month").reset_index(drop=True)

    # Use the last 7 months for testing
    test_months = monthly_data["Month"].iloc[-7:].tolist()

    feature_columns = [
        "Care_Lag_1",
        "Care_Lag_2",
        "Care_Lag_3",
        "Discharge_Lag_1",
        "Discharge_Lag_2",
        "Discharge_Lag_3",
        "Care_Rolling_Mean_3",
        "Discharge_Rolling_Mean_3",
        "Month_Number",
        "Quarter_Number"
    ]

    results = []

    print("\nTesting Period:")
    print(test_months[0].strftime("%Y-%m"), "to",
          test_months[-1].strftime("%Y-%m"))

    # Evaluate every test month
    for test_month in test_months:

        print("\nEvaluating:", test_month.strftime("%Y-%m"))

        # Use only observations available before the test month
        history = monthly_data[
            monthly_data["Month"] < test_month
        ].copy()

        actual_row = monthly_data[
            monthly_data["Month"] == test_month
        ].iloc[0]

        actual_care = actual_row["Average_HHS_Care"]
        actual_discharge = actual_row["Average_Discharges"]

        care_history = history["Average_HHS_Care"]
        discharge_history = history["Average_Discharges"]

        predictions = {}

        # 1. Naive Model
        predictions["Naive"] = {
            "Care": care_history.iloc[-1],
            "Discharge": discharge_history.iloc[-1]
        }

        # 2. Moving Average Model
        predictions["Moving Average"] = {
            "Care": care_history.tail(3).mean(),
            "Discharge": discharge_history.tail(3).mean()
        }

        # 3. ARIMA Model
        arima_care = ARIMA(
            care_history,
            order=(1, 1, 1)
        ).fit()

        arima_discharge = ARIMA(
            discharge_history,
            order=(1, 1, 1)
        ).fit()

        predictions["ARIMA"] = {
            "Care": arima_care.forecast(steps=1).iloc[0],
            "Discharge": arima_discharge.forecast(steps=1).iloc[0]
        }

        # 4. Exponential Smoothing Model
        es_care = ExponentialSmoothing(
            care_history,
            trend="add",
            seasonal=None,
            initialization_method="estimated"
        ).fit()

        es_discharge = ExponentialSmoothing(
            discharge_history,
            trend="add",
            seasonal=None,
            initialization_method="estimated"
        ).fit()

        predictions["Exponential Smoothing"] = {
            "Care": es_care.forecast(steps=1).iloc[0],
            "Discharge": es_discharge.forecast(steps=1).iloc[0]
        }

        # Get machine learning training data
        ml_train = featured_data[
            featured_data["Month"] < test_month
        ].copy()

        ml_test = featured_data[
            featured_data["Month"] == test_month
        ]

        if ml_test.empty:
            raise ValueError(
                f"Features not found for {test_month.strftime('%Y-%m')}"
            )

        X_train = ml_train[feature_columns]
        X_test = ml_test[feature_columns]

        # 5. Random Forest Model
        rf_care = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        rf_discharge = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        rf_care.fit(X_train, ml_train["Average_HHS_Care"])
        rf_discharge.fit(X_train, ml_train["Average_Discharges"])

        predictions["Random Forest"] = {
            "Care": rf_care.predict(X_test)[0],
            "Discharge": rf_discharge.predict(X_test)[0]
        }

        # 6. Gradient Boosting Model
        gb_care = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        )

        gb_discharge = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        )

        gb_care.fit(X_train, ml_train["Average_HHS_Care"])
        gb_discharge.fit(X_train, ml_train["Average_Discharges"])

        predictions["Gradient Boosting"] = {
            "Care": gb_care.predict(X_test)[0],
            "Discharge": gb_discharge.predict(X_test)[0]
        }

        # Store results for every model
        for model_name, prediction in predictions.items():

            results.append({
                "Month": test_month,
                "Model": model_name,
                "Actual_Care": actual_care,
                "Predicted_Care": max(0, prediction["Care"]),
                "Actual_Discharges": actual_discharge,
                "Predicted_Discharges": max(
                    0, prediction["Discharge"]
                )
            })

    # Create detailed forecast results
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/standardized_forecast_results.csv",
        index=False
    )

    print("\nSTANDARDIZED FORECAST RESULTS")
    print("-----------------------------")
    print(results_df.round(2).to_string(index=False))

    # Calculate final comparison metrics
    comparison = []

    for model_name in results_df["Model"].unique():

        model_data = results_df[
            results_df["Model"] == model_name
        ]

        care_mae, care_rmse = calculate_metrics(
            model_data["Actual_Care"],
            model_data["Predicted_Care"]
        )

        discharge_mae, discharge_rmse = calculate_metrics(
            model_data["Actual_Discharges"],
            model_data["Predicted_Discharges"]
        )

        comparison.append({
            "Model": model_name,
            "Care_MAE": care_mae,
            "Care_RMSE": care_rmse,
            "Discharge_MAE": discharge_mae,
            "Discharge_RMSE": discharge_rmse
        })

    comparison_df = pd.DataFrame(comparison)

    comparison_df = comparison_df.sort_values(
        by="Care_MAE"
    ).reset_index(drop=True)

    comparison_df.to_csv(
        "results/final_model_comparison.csv",
        index=False
    )

    print("\nFINAL MODEL COMPARISON")
    print("----------------------")
    print(comparison_df.round(2).to_string(index=False))

    print("\nBEST MODEL FOR CARE FORECASTING:")
    print(comparison_df.iloc[0]["Model"])

    best_discharge_model = comparison_df.loc[
        comparison_df["Discharge_MAE"].idxmin(),
        "Model"
    ]

    print("\nBEST MODEL FOR DISCHARGE FORECASTING:")
    print(best_discharge_model)

    print("\nModel comparison completed successfully.")

if __name__ == "__main__":
    main()
