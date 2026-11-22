import pandas as pd
from pathlib import Path

print("DAILY MODEL TRAIN-TEST SPLIT")
print("=" * 45)

files = {
    "care": "data/processed/daily_care_model_data.csv",
    "discharge": "data/processed/daily_discharge_model_data.csv"
}

Path("data/processed/daily_split").mkdir(
    parents=True,
    exist_ok=True
)

for name, filename in files.items():

    print(f"\nPreparing {name.upper()} dataset")

    df = pd.read_csv(
        filename,
        parse_dates=["Date"]
    )

    # Ensure chronological order
    df = df.sort_values("Date").reset_index(drop=True)

    # Chronological 80/20 split
    split_index = int(len(df) * 0.80)

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    print("Total records:", len(df))
    print("Training records:", len(train))
    print("Testing records:", len(test))

    print(
        "Training period:",
        train["Date"].min(),
        "to",
        train["Date"].max()
    )

    print(
        "Testing period:",
        test["Date"].min(),
        "to",
        test["Date"].max()
    )

    # Save separate files
    train.to_csv(
        f"data/processed/daily_split/{name}_train.csv",
        index=False
    )

    test.to_csv(
        f"data/processed/daily_split/{name}_test.csv",
        index=False
    )

print("\nChronological splitting completed!")
