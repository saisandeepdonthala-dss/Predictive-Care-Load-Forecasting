import pandas as pd
import os

# Load cleaned daily dataset
data_path = "data/processed/cleaned_data.csv"

df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Date"])

# Sort records chronologically
df = df.sort_values("Date").copy()

# Calculate monthly average flow indicators
monthly = df.set_index("Date").resample("ME").agg({
    "Children transferred out of CBP custody": "mean",
    "Children discharged from HHS Care": "mean",
    "Children in HHS Care": "mean"
})

# Calculate net flow pressure proxy
monthly["Net_Flow"] = (
    monthly["Children transferred out of CBP custody"]
    - monthly["Children discharged from HHS Care"]
)

# Classify monthly flow
monthly["Flow_Status"] = monthly["Net_Flow"].apply(
    lambda x: (
        "Higher transfer-out volume"
        if x > 0
        else "Higher discharge volume"
        if x < 0
        else "Balanced"
    )
)

# Save results
os.makedirs("results", exist_ok=True)

monthly.to_csv(
    "results/intake_discharge_balance.csv"
)

print("\nINTAKE VS DISCHARGE ANALYSIS")
print("============================")
print(monthly.tail(12).round(2))

print("\nAnalysis saved successfully.")
