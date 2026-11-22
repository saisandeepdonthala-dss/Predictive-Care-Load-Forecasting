import pandas as pd
import os

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").copy()

# Monthly average indicators
monthly = (
    df.set_index("Date")
    .resample("ME")
    .agg({
        "Children in HHS Care": "mean",
        "Children discharged from HHS Care": "mean",
        "Children transferred out of CBP custody": "mean"
    })
)

# Calculate month-to-month care-load change
monthly["Care_Change"] = monthly["Children in HHS Care"].diff()

monthly["Care_Change_Percent"] = (
    monthly["Children in HHS Care"].pct_change() * 100
)

# Identify unusual care-load increases
monthly["Care_Warning"] = monthly["Care_Change_Percent"].apply(
    lambda x: (
        "High Increase"
        if x >= 10
        else "Moderate Increase"
        if x >= 5
        else "Normal / Decrease"
        if pd.notna(x)
        else "Insufficient History"
    )
)

# Identify potential flow pressure
monthly["Net_Flow"] = (
    monthly["Children transferred out of CBP custody"]
    - monthly["Children discharged from HHS Care"]
)

monthly["Flow_Warning"] = monthly["Net_Flow"].apply(
    lambda x: (
        "Potential Flow Pressure"
        if x > 0
        else "Discharges Exceed Transfers"
        if x < 0
        else "Balanced"
    )
)

# Save results
os.makedirs("results", exist_ok=True)
monthly.to_csv("results/early_warning_results.csv")

print("\nEARLY WARNING ANALYSIS")
print("======================")
print(monthly.tail(12).round(2))

print("\nResults saved to results/early_warning_results.csv")
