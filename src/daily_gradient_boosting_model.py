import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("DAILY GRADIENT BOOSTING MODEL")
print("=" * 50)

output_dir = Path("results/daily_gradient_boosting")
output_dir.mkdir(parents=True, exist_ok=True)

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

all_results = []

for name, paths in datasets.items():

    print(f"\nTraining Gradient Boosting for {name.upper()}")
    print("-" * 50)

    train = pd.read_csv(paths["train"], parse_dates=["Date"])
    test = pd.read_csv(paths["test"], parse_dates=["Date"])

    target = paths["target"]

    train = train.sort_values("Date").reset_index(drop=True)
    test = test.sort_values("Date").reset_index(drop=True)

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_samples_split=5,
        min_samples_leaf=2,
        loss="squared_error",
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    print("\nModel Performance")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))

    prediction_df = pd.DataFrame({
        "Date": test["Date"],
        "Actual": y_test,
        "Predicted": predictions
    })

    prediction_df.to_csv(
        output_dir / f"{name}_gradient_boosting_predictions.csv",
        index=False
    )

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

    all_results.append({
        "Target": name,
        "Model": "Gradient Boosting",
        "MAE": mae,
        "RMSE": rmse,
        "Test_Records": len(test)
    })

comparison = pd.DataFrame(all_results)

comparison.to_csv(
    output_dir / "daily_gradient_boosting_comparison.csv",
    index=False
)

print("\nDAILY GRADIENT BOOSTING COMPARISON")
print("=" * 50)
print(comparison.round(2).to_string(index=False))

print("\nDaily Gradient Boosting modelling completed!")
