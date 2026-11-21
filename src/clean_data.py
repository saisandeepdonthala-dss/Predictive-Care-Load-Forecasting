import pandas as pd

# Load the original dataset
file_path = "data/raw/HHS_Unaccompanied_Alien_Children_Program.csv"

df = pd.read_csv(file_path)

print("DATA CLEANING STARTED")
print("---------------------")

# Step 1: Remove completely empty rows
df = df.dropna(how="all")

print("Rows after removing empty rows:", len(df))

# Step 2: Convert Date column into datetime format
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%B %d, %Y",
    errors="coerce"
)

# Step 3: Convert HHS Care column into numeric format
df["Children in HHS Care"] = pd.to_numeric(
    df["Children in HHS Care"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

# Step 4: Check for invalid values
print("\nMissing values after conversion:")
print(df.isnull().sum())

# Step 5: Sort dates from oldest to newest
df = df.sort_values(by="Date").reset_index(drop=True)

# Step 6: Check duplicate dates
print("\nDuplicate dates:", df["Date"].duplicated().sum())

# Step 7: Display cleaned dataset
print("\nCleaned Dataset:")
print(df.head())

# Step 8: Display data types
print("\nUpdated Data Types:")
print(df.dtypes)

# Step 9: Display date range
print("\nDate Range:")
print("Start Date:", df["Date"].min())
print("End Date:", df["Date"].max())

# Step 10: Save cleaned dataset
output_path = "data/processed/cleaned_data.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("File Location:", output_path)
print("Final Dataset Shape:", df.shape)
