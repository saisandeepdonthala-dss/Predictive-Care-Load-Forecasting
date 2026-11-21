import pandas as pd
from pathlib import Path

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Sort records by date
df = df.sort_values("Date")

print("\nFEATURE ENGINEERING")
print("===================")

# Create output folder
Path("data/processed").mkdir(exist_ok=True)

# 1. Prepare monthly data
print("\nPreparing monthly data...")

monthly_data = df.groupby(
    df["Date"].dt.to_period("M")
).agg(
    Average_HHS_Care=(
        "Children in HHS Care", "mean"
    ),
    Average_Discharges=(
        "Children discharged from HHS Care", "mean"
    ),
    Record_Count=("Date", "count")
)

monthly_data.index.name = "Month"

print("\nMonthly Dataset:")
print(monthly_data.head())

# Save monthly dataset
monthly_data.to_csv(
    "data/processed/monthly_data.csv"
)

# 2. Create lag features
print("\nCreating lag features...")

featured_data = monthly_data.copy()

# Previous months' HHS care values
featured_data["Care_Lag_1"] = (
    featured_data["Average_HHS_Care"].shift(1)
)

featured_data["Care_Lag_2"] = (
    featured_data["Average_HHS_Care"].shift(2)
)

featured_data["Care_Lag_3"] = (
    featured_data["Average_HHS_Care"].shift(3)
)

# Previous months' discharge values
featured_data["Discharge_Lag_1"] = (
    featured_data["Average_Discharges"].shift(1)
)

featured_data["Discharge_Lag_2"] = (
    featured_data["Average_Discharges"].shift(2)
)

featured_data["Discharge_Lag_3"] = (
    featured_data["Average_Discharges"].shift(3)
)

# 3. Create rolling average features
print("\nCreating rolling average features...")

featured_data["Care_Rolling_Mean_3"] = (
    featured_data["Average_HHS_Care"]
    .shift(1)
    .rolling(window=3)
    .mean()
)

featured_data["Discharge_Rolling_Mean_3"] = (
    featured_data["Average_Discharges"]
    .shift(1)
    .rolling(window=3)
    .mean()
)

# 4. Add month and quarter information
print("\nAdding month and quarter information...")

featured_data["Month_Number"] = (
    featured_data.index.month
)

featured_data["Quarter_Number"] = (
    featured_data.index.quarter
)

# 5. Remove rows with missing lag values
featured_data = featured_data.dropna()

# Convert Month index into a normal column
featured_data = featured_data.reset_index()

# Save month in readable format
featured_data["Month"] = (
    featured_data["Month"].astype(str)
)

# 6. Display final dataset

print("\nFEATURE ENGINEERING RESULTS")
print("---------------------------")

print("\nOriginal Monthly Dataset Shape:")
print(monthly_data.shape)

print("\nFinal Featured Dataset Shape:")
print(featured_data.shape)

print("\nFinal Dataset Columns:")
print(featured_data.columns.tolist())

print("\nFirst Five Records:")
print(featured_data.head().round(2).to_string(index=False))

# 7. Save featured dataset
featured_data.to_csv(
    "data/processed/featured_data.csv",
    index=False
)

print("\nFeature engineering completed successfully!")

print("Monthly dataset saved:")
print("data/processed/monthly_data.csv")

print("\nFeatured dataset saved:")
print("data/processed/featured_data.csv")
