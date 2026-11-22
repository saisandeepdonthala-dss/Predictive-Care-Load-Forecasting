import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

print("TIME-SERIES DECOMPOSITION")
print("=" * 45)

# Load monthly dataset
file_path = "data/processed/monthly_data.csv"
df = pd.read_csv(file_path)

# Convert Month into datetime index
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").set_index("Month")

# Create output directory
os.makedirs("results/decomposition", exist_ok=True)

# Variables to decompose
variables = {
    "HHS_Care": "Average_HHS_Care",
    "Discharges": "Average_Discharges"
}

for name, column in variables.items():

    print(f"\nDecomposing {name}...")

    series = df[column].asfreq("MS")

    # Additive decomposition with 12-month seasonality
    decomposition = seasonal_decompose(
        series,
        model="additive",
        period=12,
        extrapolate_trend="period"
    )

    # Save components
    components = pd.DataFrame({
        "Observed": decomposition.observed,
        "Trend": decomposition.trend,
        "Seasonal": decomposition.seasonal,
        "Residual": decomposition.resid
    })

    output_csv = (
        f"results/decomposition/{name}_decomposition.csv"
    )
    components.to_csv(output_csv)

    # Plot components
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(
        f"{name.replace('_', ' ')} - Time Series Decomposition",
        fontsize=14
    )
    fig.tight_layout()

    output_plot = (
        f"results/decomposition/{name}_decomposition.png"
    )

    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output_csv)
    print("Saved:", output_plot)

print("\nDECOMPOSITION COMPLETED SUCCESSFULLY!")
