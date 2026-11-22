import pandas as pd
from pathlib import Path

print("DAILY MODEL PREPARATION")
print("=" * 45)

# 1. Load daily feature dataset
df = pd.read_csv(
    "data/processed/daily_featured_data.csv",
    parse_dates=["Date"]
)

# 2. Keep actual observed reporting dates
df = df[
    df["Reporting_Status"] == "Observed"
].copy()

df = df.sort_values("Date")

# 3. Select target variables
targets = [
    "Children in HHS Care",
    "Children discharged from HHS Care"
]

# 4. Select historical and calendar features
features = [
    "Care_Lag_7D",
    "Care_Lag_14D",
    "Discharge_Lag_7D",
    "Discharge_Lag_14D",
    "Care_Rolling_Mean_7D",
    "Care_Rolling_Variance_7D",
    "Discharge_Rolling_Mean_7D",
    "Discharge_Rolling_Variance_7D",
    "Day_Of_Week",
    "Month_Number"
]

# 5. Create separate datasets for each target
Path("data/processed").mkdir(exist_ok=True)

for target in targets:

    print("\nPreparing target:", target)

    selected = df[
        ["Date", target] + features
    ].copy()

    before = len(selected)

    # Remove rows with missing target or features
    selected = selected.dropna()

    after = len(selected)

    print("Observed records:", before)
    print("Usable records:", after)
    print("Removed records:", before - after)

    # Save target-specific dataset
    filename = (
        "data/processed/"
        + (
            "daily_care_model_data.csv"
            if target == "Children in HHS Care"
            else "daily_discharge_model_data.csv"
        )
    )

    selected.to_csv(filename, index=False)

    print("Saved:", filename)

    print(
        "Date range:",
        selected["Date"].min(),
        "to",
        selected["Date"].max()
    )

print("\nDaily model preparation completed!")
