import pandas as pd
import matplotlib.pyplot as plt
import os

def create_intervals(data, actual_column, predicted_column):
    """Create approximate 80% empirical prediction intervals."""

    errors = data[actual_column] - data[predicted_column]
    lower_error = errors.quantile(0.10)
    upper_error = errors.quantile(0.90)

    data["Lower_Bound"] = (
        data[predicted_column] + lower_error
    ).clip(lower=0)

    data["Upper_Bound"] = (
        data[predicted_column] + upper_error
    ).clip(lower=0)

    return data

def main():

    print("\nPREDICTION INTERVAL ANALYSIS")
    print("============================")

    os.makedirs("results", exist_ok=True)

    # Load standardized forecast results
    results = pd.read_csv(
        "results/standardized_forecast_results.csv",
        parse_dates=["Month"]
    )

    # Find the best model for each target
    comparison = pd.read_csv(
        "results/final_model_comparison.csv"
    )

    best_care_model = comparison.loc[
        comparison["Care_MAE"].idxmin(), "Model"
    ]

    best_discharge_model = comparison.loc[
        comparison["Discharge_MAE"].idxmin(), "Model"
    ]

    print("\nBest Care Model:", best_care_model)
    print("Best Discharge Model:", best_discharge_model)

    # Care prediction intervals
    care_data = results[
        results["Model"] == best_care_model
    ].copy()

    care_data = create_intervals(
        care_data,
        "Actual_Care",
        "Predicted_Care"
    )

    care_data.to_csv(
        "results/care_prediction_intervals.csv",
        index=False
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        care_data["Month"],
        care_data["Actual_Care"],
        marker="o",
        label="Actual HHS Care"
    )

    plt.plot(
        care_data["Month"],
        care_data["Predicted_Care"],
        marker="s",
        linestyle="--",
        label=f"Predicted HHS Care ({best_care_model})"
    )

    plt.fill_between(
        care_data["Month"],
        care_data["Lower_Bound"],
        care_data["Upper_Bound"],
        alpha=0.2,
        label="Approximate 80% Error Interval"
    )

    plt.title("HHS Care Forecast with Approximate Error Intervals")
    plt.xlabel("Month")
    plt.ylabel("Average Number of Children")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "results/care_prediction_interval.png"
    )
    plt.show()

    # Discharge prediction intervals
    discharge_data = results[
        results["Model"] == best_discharge_model
    ].copy()

    discharge_data = create_intervals(
        discharge_data,
        "Actual_Discharges",
        "Predicted_Discharges"
    )

    discharge_data.to_csv(
        "results/discharge_prediction_intervals.csv",
        index=False
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        discharge_data["Month"],
        discharge_data["Actual_Discharges"],
        marker="o",
        label="Actual Discharges"
    )

    plt.plot(
        discharge_data["Month"],
        discharge_data["Predicted_Discharges"],
        marker="s",
        linestyle="--",
        label=f"Predicted Discharges ({best_discharge_model})"
    )

    plt.fill_between(
        discharge_data["Month"],
        discharge_data["Lower_Bound"],
        discharge_data["Upper_Bound"],
        alpha=0.2,
        label="Approximate 80% Error Interval"
    )

    plt.title("Discharge Forecast with Approximate Error Intervals")
    plt.xlabel("Month")
    plt.ylabel("Average Number of Discharges")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "results/discharge_prediction_interval.png"
    )
    plt.show()

    print("\nPREDICTION INTERVALS COMPLETED")
    print("------------------------------")
    print("Generated Files:")
    print("1. results/care_prediction_intervals.csv")
    print("2. results/discharge_prediction_intervals.csv")
    print("3. results/care_prediction_interval.png")
    print("4. results/discharge_prediction_interval.png")

if __name__ == "__main__":
    main()
