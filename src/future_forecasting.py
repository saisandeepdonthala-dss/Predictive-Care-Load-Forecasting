import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

def main():

    print("\nFUTURE CARE LOAD AND DISCHARGE FORECASTING")
    print("==========================================")

    # Load monthly dataset
    data = pd.read_csv(
        "data/processed/monthly_data.csv",
        parse_dates=["Month"]
    )

    data = data.sort_values("Month").reset_index(drop=True)

    # Select forecast horizon
    forecast_months = 6

    # Prepare historical values
    care_history = data["Average_HHS_Care"].tolist()
    discharge_history = data["Average_Discharges"].tolist()

    # ARIMA - HHS Care Forecast
    print("\nTraining ARIMA model for HHS Care...")
    
    care_model = ARIMA(
        care_history,
        order=(1, 1, 1)
    )

    care_fitted = care_model.fit()
    care_predictions = care_fitted.forecast(
        steps=forecast_months
    )
    care_predictions = np.maximum(care_predictions, 0)

    # Random Forest - Discharge Forecast
    print("\nTraining Random Forest model for Discharges...")

    # Create training features from historical observations
    features = []
    targets = []

    for i in range(3, len(data)):
        row = [
            discharge_history[i - 1],
            discharge_history[i - 2],
            discharge_history[i - 3],
            care_history[i - 1],
            care_history[i - 2],
            care_history[i - 3],
            np.mean(discharge_history[i - 3:i]),
            np.mean(care_history[i - 3:i]),
            data.loc[i, "Month"].month,
            data.loc[i, "Month"].quarter
        ]

        features.append(row)
        targets.append(discharge_history[i])

    X_train = pd.DataFrame(
        features,
        columns=[
            "Discharge_Lag_1",
            "Discharge_Lag_2",
            "Discharge_Lag_3",
            "Care_Lag_1",
            "Care_Lag_2",
            "Care_Lag_3",
            "Discharge_Rolling_Mean_3",
            "Care_Rolling_Mean_3",
            "Month_Number",
            "Quarter_Number"
        ]
    )

    y_train = np.array(targets)

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # Generate Future Forecasts
    future_dates = pd.date_range(
        start=data["Month"].iloc[-1] + pd.offsets.MonthBegin(1),
        periods=forecast_months,
        freq="MS"
    )
    future_discharges = []

    for i in range(forecast_months):
        next_date = future_dates[i]
        row = [[
            discharge_history[-1],
            discharge_history[-2],
            discharge_history[-3],
            care_history[-1],
            care_history[-2],
            care_history[-3],
            np.mean(discharge_history[-3:]),
            np.mean(care_history[-3:]),
            next_date.month,
            next_date.quarter
        ]]

        predicted_discharge = rf_model.predict(row)[0]
        predicted_discharge = max(0, predicted_discharge)
        future_discharges.append(predicted_discharge)

        # Update history for the next forecast
        discharge_history.append(predicted_discharge)
        care_history.append(care_predictions[i])

    # Create Forecast Results
    forecast_results = pd.DataFrame({
        "Month": future_dates,
        "Predicted_HHS_Care": care_predictions,
        "Predicted_Discharges": future_discharges
    })

    forecast_results = forecast_results.round(2)

    print("\nFUTURE FORECAST RESULTS")
    print("-----------------------")
    print(forecast_results.to_string(index=False))

    # Save results
    forecast_results.to_csv(
        "results/future_forecast_results.csv",
        index=False
    )

    print("\nForecast saved successfully:")
    print("results/future_forecast_results.csv")
    print("\nFuture forecasting completed successfully.")

if __name__ == "__main__":
    main()
