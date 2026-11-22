import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("DAILY RANDOM FOREST MODEL")
print("=" * 50)

# Output folder
output_dir = Path("results/daily_random_forest")
output_dir.mkdir(parents=True, exist_ok=True)

# Datasets and target columns
datasets = {
    "care": {
        "train": "data/processed/daily_split/care_train.csv",
        "test": "data/processed/daily_split/care_test.csv",
        "target": "Children in HHS Care"
    },
    "discharge": {
        "train": "data/processed/daily_split/discharge_train.csv",
        "test": "data/processed/daily_split/discharge_test.csv",
        "target": "Children discharged from HHS Care"
    }
}

# Historical features only
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

comparison_results = []

for name, paths in datasets.items():

    print(f"\nTraining Random Forest for {name.upper()}")
    print("-" * 50)

    train = pd.read_csv(paths["train"], parse_dates=["Date"])
    test = pd.read_csv(paths["test"], parse_dates=["Date"])

    target = paths["target"]

    # Sort chronologically
    train = train.sort_values("Date").reset_index(drop=True)
    test = test.sort_values("Date").reset_index(drop=True)

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    # Initialize Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train, y_train)

    # Generate predictions
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    print("\nModel Performance")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))

    # Save predictions
    prediction_df = pd.DataFrame({
        "Date": test["Date"],
        "Actual": y_test,
        "Predicted": predictions
    })

    prediction_df.to_csv(
        output_dir / f"{name}_random_forest_predictions.csv",
        index=False
    )

    # Save feature importance
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    importance_df.to_csv(
        output_dir / f"{name}_feature_importance.csv",
        index=False
    )

    print("\nTop 5 Important Features:")
    print(importance_df.head(5).to_string(index=False))

    comparison_results.append({
        "Target": name,
        "Model": "Random Forest",
        "MAE": mae,
        "RMSE": rmse,
        "Test_Records": len(test)
    })

# Save combined results
comparison = pd.DataFrame(comparison_results)

comparison.to_csv(
    output_dir / "daily_random_forest_comparison.csv",
    index=False
)

print("\nDAILY RANDOM FOREST COMPARISON")
print("=" * 50)
print(comparison.round(2).to_string(index=False))

print("\nDaily Random Forest modelling completed!")
