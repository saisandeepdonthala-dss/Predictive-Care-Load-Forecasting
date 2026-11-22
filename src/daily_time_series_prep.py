import pandas as pd
import os

# File paths
INPUT_FILE = "data/processed/cleaned_data.csv"
OUTPUT_FILE = "data/processed/daily_observed_data.csv"
REPORT_FILE = "results/daily_data_quality_report.csv"

print("DAILY TIME-SERIES PREPARATION")
print("=" * 40)

# Step 1: Load cleaned data
df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])

# Step 2: Sort and validate dates
df = df.sort_values("Date").reset_index(drop=True)

if df["Date"].duplicated().any():
    raise ValueError("Duplicate dates found. Resolve them before proceeding.")

# Step 3: Identify daily reporting gaps
full_dates = pd.date_range(
    start=df["Date"].min(),
    end=df["Date"].max(),
    freq="D"
)

missing_dates = full_dates.difference(df["Date"])

# Step 4: Preserve observed records only
# Missing flow values are NOT replaced with zero.
# Missing stock values are NOT interpolated automatically.
df["Reporting_Status"] = "Observed"

# Step 5: Calculate reporting intervals
df["Days_Since_Previous_Report"] = (
    df["Date"].diff().dt.days
)

# Step 6: Generate quality report
quality_report = pd.DataFrame({
    "Metric": [
        "First Reporting Date",
        "Last Reporting Date",
        "Expected Calendar Days",
        "Observed Reporting Days",
        "Missing Calendar Days",
        "Reporting Coverage Percentage",
        "Duplicate Dates"
    ],
    "Value": [
        str(df["Date"].min().date()),
        str(df["Date"].max().date()),
        len(full_dates),
        df["Date"].nunique(),
        len(missing_dates),
        round(len(df) / len(full_dates) * 100, 2),
        int(df["Date"].duplicated().sum())
    ]
})

# Step 7: Create output directories
os.makedirs("data/processed", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Step 8: Save observed daily data
df.to_csv(OUTPUT_FILE, index=False)

# Step 9: Save quality report
quality_report.to_csv(REPORT_FILE, index=False)

# Step 10: Display results
print("\nDATA QUALITY SUMMARY")
print(quality_report.to_string(index=False))

print("\nMissing dates are preserved as unobserved.")
print("No zero filling or automatic interpolation applied.")

print("\nFiles saved:")
print(OUTPUT_FILE)
print(REPORT_FILE)

print("\nDaily preparation completed successfully!")
