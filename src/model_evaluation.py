import pandas as pd
import matplotlib.pyplot as plt
import os

def main():

    print("\nMODEL EVALUATION AND VISUALIZATION")
    print("=================================")

    # Create results folder if it does not exist
    os.makedirs("results", exist_ok=True)

    # Load standardized forecast results
    results = pd.read_csv(
        "results/standardized_forecast_results.csv",
        parse_dates=["Month"]
    )

    # Load model comparison results
    comparison = pd.read_csv(
        "results/final_model_comparison.csv"
    )

    # Chart 1: Actual vs Predicted HHS Care
    best_care_model = comparison.loc[
        comparison["Care_MAE"].idxmin(), "Model"
    ]

    care_data = results[
        results["Model"] == best_care_model
    ]

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

    plt.title("Actual vs Predicted HHS Care")
    plt.xlabel("Month")
    plt.ylabel("Average Number of Children")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/actual_vs_predicted_care.png")
    plt.show()

    # Chart 2: Actual vs Predicted Discharges
    best_discharge_model = comparison.loc[
        comparison["Discharge_MAE"].idxmin(), "Model"
    ]

    discharge_data = results[
        results["Model"] == best_discharge_model
    ]

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

    plt.title("Actual vs Predicted HHS Discharges")
    plt.xlabel("Month")
    plt.ylabel("Average Number of Discharges")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/actual_vs_predicted_discharges.png")
    plt.show()

    # Chart 3: Model MAE Comparison
    comparison = comparison.sort_values("Care_MAE")

    plt.figure(figsize=(12, 6))
    plt.bar(
        comparison["Model"],
        comparison["Care_MAE"]
    )

    plt.title("Model Comparison - HHS Care MAE")
    plt.xlabel("Forecasting Model")
    plt.ylabel("Mean Absolute Error")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("results/model_mae_comparison.png")
    plt.show()

    print("\nMODEL EVALUATION COMPLETED")
    print("--------------------------")
    print("Best Care Model:", best_care_model)
    print("Best Discharge Model:", best_discharge_model)
    print("\nGenerated Charts:")
    print("1. results/actual_vs_predicted_care.png")
    print("2. results/actual_vs_predicted_discharges.png")
    print("3. results/model_mae_comparison.png")

if __name__ == "__main__":
    main()
