import pandas as pd
from pathlib import Path

print("\nMODEL PREPARATION")
print("=================")

# Load the dataset created during feature engineering
df = pd.read_csv("data/processed/featured_data.csv")

print("\nDataset loaded successfully!")
print("Dataset Shape:", df.shape)

# Make sure the data is arranged by month
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

# Select the input features for the models
feature_columns = [
    "Care_Lag_1",
    "Care_Lag_2",
    "Care_Lag_3",
    "Discharge_Lag_1",
    "Discharge_Lag_2",
    "Discharge_Lag_3",
    "Care_Rolling_Mean_3",
    "Discharge_Rolling_Mean_3",
    "Month_Number",
    "Quarter_Number"
]

# Select the two values we want to predict
target_columns = [
    "Average_HHS_Care",
    "Average_Discharges"
]

# Separate input data and target data
X = df[feature_columns]
y = df[target_columns]

print("\nInput Features Shape:", X.shape)
print("Target Data Shape:", y.shape)

# Keep the month information for checking predictions later
month_data = df[["Month"]]

# Split the data in chronological order
# Use the first 80% for training and the remaining 20% for testing
split_point = int(len(df) * 0.80)

X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]

y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]

train_months = month_data.iloc[:split_point]
test_months = month_data.iloc[split_point:]

print("\nTRAINING AND TESTING DETAILS")
print("----------------------------")

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

print("\nTraining Period:")
print(train_months["Month"].min(), "to", train_months["Month"].max())

print("\nTesting Period:")
print(test_months["Month"].min(), "to", test_months["Month"].max())

# Create folder to save the prepared data
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Save training and testing features
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)

# Save actual target values
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

# Save month information separately
train_months.to_csv("data/processed/train_months.csv", index=False)
test_months.to_csv("data/processed/test_months.csv", index=False)

print("\nAll prepared datasets saved successfully!")

print("\nModel preparation completed successfully!")
