import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("DAILY FORECAST VISUALIZATION")
print("=" * 50)

output_dir = Path("results/daily_visualizations")
output_dir.mkdir(parents=True, exist_ok=True)

datasets = {
    "care": {
        "file": "results/daily_baseline/care_baseline_predictions.csv",
        "best_model": "Naive",
        "title": "Daily HHS Care: Actual vs Predicted",
        "ylabel": "Children in HHS Care"
    },
    "discharge": {
        "file": "results/daily_baseline/discharge_baseline_predictions.csv",
        "best_model": "Moving_Average",
        "title": "Daily HHS Discharges: Actual vs Predicted",
        "ylabel": "Children Discharged"
    }
}

for name, config in datasets.items():

    print(f"\nCreating visualization for {name.upper()}")

    df = pd.read_csv(
        config["file"],
        parse_dates=["Date"]
    )

    prediction_column = f"{config['best_model']}_Prediction"

    plt.figure(figsize=(14, 6))

    plt.plot(
        df["Date"],
        df["Actual"],
        label="Actual",
        linewidth=2
    )

    plt.plot(
        df["Date"],
        df[prediction_column],
        label=f"{config['best_model']} Prediction",
        linestyle="--",
        linewidth=2
    )

    plt.title(config["title"])
    plt.xlabel("Date")
    plt.ylabel(config["ylabel"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = output_dir / f"{name}_actual_vs_predicted.png"

    plt.savefig(filename, dpi=300)
    plt.close()

    print("Saved:", filename)

print("\nDaily forecast visualizations completed!")
