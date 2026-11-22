import pandas as pd
from pathlib import Path

print("DAILY FEATURE ENGINEERING")
print("=" * 45)

# 1. Load prepared daily observations
df = pd.read_csv(
    "data/processed/daily_observed_data.csv",
    parse_dates=["Date"]
)

df = df.sort_values("Date").set_index("Date")

# 2. Create complete calendar index
calendar = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq="D"
)

daily = df.reindex(calendar)
daily.index.name = "Date"

# 3. Preserve missing dates
daily["Reporting_Status"] = (
    daily["Reporting_Status"].fillna("Missing")
)

print("\nCalendar Dataset Shape:", daily.shape)
print(
    "Observed dates:",
    (daily["Reporting_Status"] == "Observed").sum()
)
print(
    "Missing dates:",
    (daily["Reporting_Status"] == "Missing").sum()
)

# 4. Define target and flow variables
care = daily["Children in HHS Care"]
discharges = daily["Children discharged from HHS Care"]
transfers = daily["Children transferred out of CBP custody"]

# 5. Create calendar-based lag features
print("\nCreating calendar-based lag features...")

for lag in [1, 7, 14]:
    daily[f"Care_Lag_{lag}D"] = care.shift(lag)
    daily[f"Discharge_Lag_{lag}D"] = discharges.shift(lag)
    daily[f"Transfer_Lag_{lag}D"] = transfers.shift(lag)

# 6. Create rolling features with coverage checks
print("\nCreating rolling features with coverage checks...")

for window, minimum_observations in [(7, 5), (14, 10)]:

    # Care observation count in preceding calendar window
    care_count = (
        care.shift(1)
        .rolling(window=window, min_periods=1)
        .count()
    )

    # Discharge observation count
    discharge_count = (
        discharges.shift(1)
        .rolling(window=window, min_periods=1)
        .count()
    )

    daily[f"Care_Observed_Count_{window}D"] = care_count
    daily[f"Discharge_Observed_Count_{window}D"] = discharge_count

    daily[f"Care_Coverage_{window}D"] = (
        care_count / window
    )

    daily[f"Discharge_Coverage_{window}D"] = (
        discharge_count / window
    )

    # Rolling statistics only when minimum coverage is satisfied
    daily[f"Care_Rolling_Mean_{window}D"] = (
        care.shift(1)
        .rolling(
            window=window,
            min_periods=minimum_observations
        )
        .mean()
    )

    daily[f"Care_Rolling_Variance_{window}D"] = (
        care.shift(1)
        .rolling(
            window=window,
            min_periods=minimum_observations
        )
        .var()
    )

    daily[f"Discharge_Rolling_Mean_{window}D"] = (
        discharges.shift(1)
        .rolling(
            window=window,
            min_periods=minimum_observations
        )
        .mean()
    )

    daily[f"Discharge_Rolling_Variance_{window}D"] = (
        discharges.shift(1)
        .rolling(
            window=window,
            min_periods=minimum_observations
        )
        .var()
    )

# 7. Create flow-based indicator
print("\nCreating flow-based indicator...")

daily["Net_Flow_Pressure"] = transfers - discharges

# 8. Add calendar features
print("\nCreating calendar features...")

daily["Day_Of_Week"] = daily.index.day_of_week
daily["Month_Number"] = daily.index.month
daily["Quarter_Number"] = daily.index.quarter
daily["Day_Of_Month"] = daily.index.day

# 9. Save feature dataset
Path("data/processed").mkdir(exist_ok=True)

output_file = "data/processed/daily_featured_data.csv"

daily.reset_index().to_csv(
    output_file,
    index=False
)

# 10. Display results
print("\nDAILY FEATURE ENGINEERING RESULTS")
print("-" * 45)

print("Final Dataset Shape:", daily.shape)

print("\nRolling Feature Availability:")

for window in [7, 14]:
    care_available = (
        daily[f"Care_Rolling_Mean_{window}D"].notna().sum()
    )

    discharge_available = (
        daily[f"Discharge_Rolling_Mean_{window}D"].notna().sum()
    )

    print(
        f"{window}-day Care rolling mean available:",
        care_available
    )

    print(
        f"{window}-day Discharge rolling mean available:",
        discharge_available
    )

print("\nFeature Columns:")
print(daily.columns.tolist())

print("\nFirst 10 Rows:")
print(daily.head(10).round(2).to_string())

print("\nOutput saved:", output_file)
print("\nDaily feature engineering completed!")
