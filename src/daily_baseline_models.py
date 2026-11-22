import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("DAILY BASELINE MODELS")
print("=" * 50)

# Create output folder
output_dir = Path("results/daily_baseline")
output_dir.mkdir(parents=True, exist_ok=True)

datasets = {
    "care": {
        "train": "data/processed/daily_split/care_train.csv",
        "test": "data/processed/daily_split/care_test.csv",
        "target": "Children in HHS Care"
    },
    "discharge": {
        "train": "data/processed/daily_split/discharge_train.csv",
        "test": "data/processed/daily_split/discharge_test.csv",
        "target": "Children discharged from HHS Care"
    }
}

all_results = []

for name, paths in datasets.items():

    print(f"\nTraining baseline models for {name.upper()}")
    print("-" * 50)

    train = pd.read_csv(paths["train"], parse_dates=["Date"])
    test = pd.read_csv(paths["test"], parse_dates=["Date"])

    target = paths["target"]

    train = train.sort_values("Date").reset_index(drop=True)
    test = test.sort_values("Date").reset_index(drop=True)

    # Use only historical observed target values
    history = train[target].astype(float).tolist()

    predictions = []

    for _, row in test.iterrows():

        actual = float(row[target])

        # Naive: previous observed target value
        naive_prediction = history[-1]

        # Moving average: average of previous 7 observed records
        recent_values = history[-7:]
        moving_average_prediction = np.mean(recent_values)

        predictions.append({
            "Date": row["Date"],
            "Actual": actual,
            "Naive_Prediction": naive_prediction,
            "Moving_Average_Prediction": moving_average_prediction
        })

        # Walk-forward evaluation:
        # Actual value becomes available for the next test prediction
        history.append(actual)

    results = pd.DataFrame(predictions)

    # Calculate performance metrics
    for model in ["Naive", "Moving_Average"]:

        actual_values = results["Actual"]
        predicted_values = results[f"{model}_Prediction"]

        mae = mean_absolute_error(
            actual_values,
            predicted_values
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual_values,
                predicted_values
            )
        )

        all_results.append({
            "Target": name,
            "Model": model,
            "MAE": mae,
            "RMSE": rmse,
            "Test_Records": len(results)
        })

        print(f"\n{model} Model")
        print("MAE:", round(mae, 2))
        print("RMSE:", round(rmse, 2))

    # Save individual predictions
    results.to_csv(
        output_dir / f"{name}_baseline_predictions.csv",
        index=False
    )

    print("\nSaved predictions for:", name.upper())

# Save combined model comparison
comparison = pd.DataFrame(all_results)

comparison.to_csv(
    output_dir / "daily_baseline_comparison.csv",
    index=False
)

print("\nDAILY BASELINE MODEL COMPARISON")
print("=" * 50)
print(comparison.round(2).to_string(index=False))

print("\nSaved: results/daily_baseline/daily_baseline_comparison.csv")
print("\nDaily baseline modelling completed!")
