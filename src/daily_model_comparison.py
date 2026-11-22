import pandas as pd
from pathlib import Path

print("DAILY MODEL COMPARISON")
print("=" * 55)

files = [
    "results/daily_baseline/daily_baseline_comparison.csv",
    "results/daily_random_forest/daily_random_forest_comparison.csv",
    "results/daily_gradient_boosting/daily_gradient_boosting_comparison.csv"
]

output_dir = Path("results/daily_model_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

dataframes = []

for file in files:

    df = pd.read_csv(file)
    dataframes.append(df)

# Combine all model results
comparison = pd.concat(
    dataframes,
    ignore_index=True
)

# Sort by target and MAE
comparison = comparison.sort_values(
    ["Target", "MAE"]
).reset_index(drop=True)

print("\nCOMPLETE MODEL PERFORMANCE")
print("-" * 55)
print(comparison.round(2).to_string(index=False))

# Identify best model for each target
best_models = comparison.loc[
    comparison.groupby("Target")["MAE"].idxmin()
].reset_index(drop=True)

print("\nBEST MODEL FOR EACH TARGET")
print("-" * 55)
print(best_models.round(2).to_string(index=False))

# Save complete comparison
comparison.to_csv(
    output_dir / "all_daily_model_comparison.csv",
    index=False
)

# Save best models
best_models.to_csv(
    output_dir / "best_daily_models.csv",
    index=False
)

print("\nSaved:")
print(output_dir / "all_daily_model_comparison.csv")
print(output_dir / "best_daily_models.csv")

print("\nDaily model comparison completed!")
