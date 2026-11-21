import pandas as pd

# Load the dataset
file_path = "data/raw/HHS_Unaccompanied_Alien_Children_Program.csv"

df = pd.read_csv(file_path)

print("\nDATASET INFORMATION")
print("-------------------")

# Display original dataset size
print("Original Dataset Shape:", df.shape)

# Remove completely empty rows for inspection
df = df.dropna(how="all")

print("After Removing Empty Rows:", df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display first five records
print("\nFirst Five Records:")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display data types
print("\nData Types:")
print(df.dtypes)

# Convert dates for inspection
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%B %d, %Y",
    errors="coerce"
)

print("\nDate Range:")
print("Start Date:", df["Date"].min())
print("End Date:", df["Date"].max())

# Check duplicate dates
print("\nDuplicate Dates:", df["Date"].duplicated().sum())

print("\nDataset inspection completed.")
