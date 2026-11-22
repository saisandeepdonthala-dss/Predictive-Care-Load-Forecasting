import pandas as pd
from pathlib import Path

print("DAILY FEATURE QUALITY ANALYSIS")
print("=" * 45)

df = pd.read_csv(
    "data/processed/daily_featured_data.csv"
)

# Keep actual reporting dates for the analysis
observed = df[
    df["Reporting_Status"] == "Observed"
].copy()

features = [
    "Care_Lag_1D",
    "Care_Lag_7D",
    "Care_Lag_14D",
    "Care_Rolling_Mean_7D",
    "Care_Rolling_Variance_7D",
    "Care_Rolling_Mean_14D",
    "Care_Rolling_Variance_14D",
    "Discharge_Lag_1D",
    "Discharge_Lag_7D",
    "Discharge_Lag_14D",
    "Discharge_Rolling_Mean_7D",
    "Discharge_Rolling_Variance_7D",
    "Discharge_Rolling_Mean_14D",
    "Discharge_Rolling_Variance_14D",
    "Net_Flow_Pressure",
    "Day_Of_Week",
    "Month_Number",
]

# Calculate feature availability
report = pd.DataFrame({
    "Feature": features,
    "Available_Records": [
        observed[col].notna().sum()
        for col in features
    ],
    "Missing_Records": [
        observed[col].isna().sum()
        for col in features
    ]
})

report["Availability_Percentage"] = (
    report["Available_Records"] / len(observed) * 100
).round(2)

report = report.sort_values(
    "Availability_Percentage",
    ascending=False
)

print("\nObserved Records:", len(observed))
print("\nFeature Availability:")
print(report.to_string(index=False))

# Save report
Path("results").mkdir(exist_ok=True)

report.to_csv(
    "results/daily_feature_availability.csv",
    index=False
)

print("\nReport saved:")
print("results/daily_feature_availability.csv")

print("\nAnalysis completed successfully!")
