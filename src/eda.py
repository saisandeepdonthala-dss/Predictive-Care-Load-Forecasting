
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Convert Date column into datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Arrange records based on date
df = df.sort_values("Date").reset_index(drop=True)

# Create results folder if it does not exist
Path("results").mkdir(exist_ok=True)

print("\nEXPLORATORY DATA ANALYSIS")
print("=========================")

# 1. Check dataset information
print("\nDataset Shape:")
print(df.shape)

# 2. Display basic statistical details
print("\nStatistical Summary:")
print(df.describe().round(2))

# 3. Check the difference between reporting dates
df["Date_Gap"] = df["Date"].diff().dt.days

print("\nReporting Date Gap Analysis:")
print(df["Date_Gap"].describe())

print("\nNumber of records by date gap:")
print(df["Date_Gap"].value_counts().sort_index())

# 4. HHS care load trend
plt.figure(figsize=(12, 5))

plt.plot(df["Date"], df["Children in HHS Care"])

plt.title("Children in HHS Care Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Children")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("results/hhs_care_trend.png")
plt.close()

# 5. Discharge trend
plt.figure(figsize=(12, 5))

plt.plot(
    df["Date"],
    df["Children discharged from HHS Care"]
)

plt.title("Children Discharged from HHS Care Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Discharges")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("results/discharge_trend.png")
plt.close()

# 6. Check the relationship between numerical columns
numeric_df = df.drop(
    columns=["Date_Gap"]
).select_dtypes(include="number")

correlation = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation.round(2))

plt.figure(figsize=(10, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Dataset Variables")
plt.tight_layout()

plt.savefig("results/correlation_heatmap.png")
plt.close()

# 7. Check the distribution of HHS care load
plt.figure(figsize=(10, 5))

sns.histplot(
    df["Children in HHS Care"],
    kde=True
)

plt.title("Distribution of Children in HHS Care")
plt.xlabel("Number of Children")
plt.tight_layout()

plt.savefig("results/hhs_care_distribution.png")
plt.close()

# 8. Display HHS care load details
print("\nHHS CARE LOAD TREND")
print("-------------------")

print("First recorded value:",
      df["Children in HHS Care"].iloc[0])

print("Last recorded value:",
      df["Children in HHS Care"].iloc[-1])

print("Minimum care load:",
      df["Children in HHS Care"].min())

print("Maximum care load:",
      df["Children in HHS Care"].max())

# 9. Display discharge details
print("\nDISCHARGE TREND")
print("----------------")

print("First recorded value:",
      df["Children discharged from HHS Care"].iloc[0])

print("Last recorded value:",
      df["Children discharged from HHS Care"].iloc[-1])

print("Minimum discharges:",
      df["Children discharged from HHS Care"].min())

print("Maximum discharges:",
      df["Children discharged from HHS Care"].max())

# 10. Check yearly average values
print("\nYEAR-WISE AVERAGE")
print("-----------------")

df["Year"] = df["Date"].dt.year

yearly_average = df.groupby("Year")[
    [
        "Children in HHS Care",
        "Children discharged from HHS Care"
    ]
].mean().round(2)

print(yearly_average)

# 11. Check the number of records in each year
print("\nYEAR-WISE RECORD COUNT")
print("----------------------")

print(df.groupby("Year").size())

# 12. Display reporting dates with gaps greater than one day
print("\nREPORTING DATE GAPS")
print("-------------------")

gaps = df.loc[
    df["Date_Gap"] > 1,
    ["Date", "Date_Gap"]
]

print(gaps.to_string(index=False))

print("\nTotal gaps greater than one day:", len(gaps))

# 13. Monthly reporting analysis
print("\nMONTHLY REPORTING ANALYSIS")
print("--------------------------")

df["Month"] = df["Date"].dt.to_period("M")

monthly_analysis = df.groupby("Month").agg(
    Record_Count=("Date", "count"),
    Average_Care=("Children in HHS Care", "mean"),
    Average_Discharge=(
        "Children discharged from HHS Care", "mean"
    )
).round(2)

print(monthly_analysis.to_string())

# Save monthly results
monthly_analysis.to_csv(
    "results/monthly_reporting_analysis.csv"
)

print("\nMonthly analysis saved successfully!")

# 14. Check the major change between January and March 2025
print("\n2025 DATA INVESTIGATION")
print("-----------------------")

selected_columns = [
    "Date",
    "Children transferred out of CBP custody",
    "Children in HHS Care",
    "Children discharged from HHS Care"
]

investigation = df[
    (df["Date"] >= "2025-01-01") &
    (df["Date"] <= "2025-03-31")
]

print("\nJanuary to March 2025 records:")

print(
    investigation[selected_columns].to_string(index=False)
)

print("\nMonthly minimum and maximum values:")

print(
    investigation.groupby(
        investigation["Date"].dt.to_period("M")
    )[
        [
            "Children in HHS Care",
            "Children discharged from HHS Care"
        ]
    ].agg(["min", "max"]).round(2)
)

# 15. Check all columns during the 2025 change
print("\n2025 TRANSITION - FULL DATA CHECK")
print("---------------------------------")

print(investigation.to_string(index=False))

print("\nMonthly averages for all numerical columns:")

print(
    investigation.groupby(
        investigation["Date"].dt.to_period("M")
    ).mean(numeric_only=True).round(2)
)

# 16. Quarterly trend analysis
print("\nQUARTERLY TREND ANALYSIS")
print("------------------------")

df["Quarter"] = df["Date"].dt.to_period("Q")

quarterly_analysis = df.groupby("Quarter").agg(
    Average_HHS_Care=("Children in HHS Care", "mean"),
    Average_Discharges=(
        "Children discharged from HHS Care", "mean"
    ),
    Minimum_HHS_Care=("Children in HHS Care", "min"),
    Maximum_HHS_Care=("Children in HHS Care", "max"),
    Record_Count=("Date", "count")
).round(2)

print(quarterly_analysis)

# 17. Create a quarterly trend chart
quarterly_plot = quarterly_analysis.copy()

quarterly_plot.index = quarterly_plot.index.astype(str)

plt.figure(figsize=(12, 6))

plt.plot(
    quarterly_plot.index,
    quarterly_plot["Average_HHS_Care"],
    marker="o",
    label="Average HHS Care"
)

plt.plot(
    quarterly_plot.index,
    quarterly_plot["Average_Discharges"],
    marker="o",
    label="Average Discharges"
)

plt.title("Quarterly Trends in HHS Care and Discharges")
plt.xlabel("Quarter")
plt.ylabel("Average Number of Children")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("results/quarterly_trend.png")
plt.close()

print("\nQuarterly trend chart saved successfully!")

# Final message
print("\nEDA completed successfully!")
print("All charts and analysis files are saved in the results folder.")
