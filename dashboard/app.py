import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Page Configuration
st.set_page_config(
    page_title="Predictive Care Load Forecasting",
    page_icon="📊",
    layout="wide"
)

# Dashboard Title
st.title("Predictive Forecasting of Care Load & Placement Demand")

st.markdown(
    "An interactive dashboard for analyzing HHS care levels, "
    "discharge patterns, and forecasting model performance."
)

# Check Required Files
comparison_path = "results/final_model_comparison.csv"
forecast_path = "results/standardized_forecast_results.csv"
future_forecast_path = "results/future_forecast_results.csv"

required_files = [
    comparison_path,
    forecast_path,
    future_forecast_path
]

for file_path in required_files:
    if not os.path.exists(file_path):
        st.error(
            f"Required file is missing: {file_path}. "
            "Please run the relevant Python scripts first."
        )
        st.stop()

# Load Data
comparison = pd.read_csv(comparison_path)
forecasts = pd.read_csv(
    forecast_path,
    parse_dates=["Month"]
)

future_forecasts = pd.read_csv(
    future_forecast_path,
    parse_dates=["Month"]
)

# Load prediction interval results
care_intervals = pd.read_csv(
    "results/care_prediction_intervals.csv",
    parse_dates=["Month"]
)

discharge_intervals = pd.read_csv(
    "results/discharge_prediction_intervals.csv",
    parse_dates=["Month"]
)

# Find Best Models
best_care_model = comparison.loc[
    comparison["Care_MAE"].idxmin(), "Model"
]

best_discharge_model = comparison.loc[
    comparison["Discharge_MAE"].idxmin(), "Model"
]

# Key Performance Indicators
st.header("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Forecasting Models",
    len(comparison)
)

col2.metric(
    "Best Care Model",
    best_care_model
)

col3.metric(
    "Best Discharge Model",
    best_discharge_model
)

col4.metric(
    "Testing Months",
    forecasts["Month"].nunique()
)
st.divider()

# Model Performance Comparison
st.header("Model Performance Comparison")

st.dataframe(
    comparison.round(2),
    use_container_width=True,
    hide_index=True
)

# Interactive Model Selection
st.header("Interactive Forecast Analysis")
model_options = comparison["Model"].tolist()
selected_model = st.selectbox(
    "Select Forecasting Model",
    model_options,
    index=model_options.index(best_care_model)
)

# HHS Care Forecast
st.subheader("HHS Care Forecast")

care_data = forecasts[
    forecasts["Model"] == selected_model
].copy()

care_chart_data = care_data.melt(
    id_vars=["Month"],
    value_vars=["Actual_Care", "Predicted_Care"],
    var_name="Type",
    value_name="Children"
)

fig_care = px.line(
    care_chart_data,
    x="Month",
    y="Children",
    color="Type",
    markers=True,
    title=f"Actual vs Predicted HHS Care - {selected_model}"
)

st.plotly_chart(
    fig_care,
    use_container_width=True
)

# HHS Discharge Forecast
st.subheader("HHS Discharge Forecast")

discharge_data = forecasts[
    forecasts["Model"] == selected_model
].copy()

discharge_chart_data = discharge_data.melt(
    id_vars=["Month"],
    value_vars=["Actual_Discharges", "Predicted_Discharges"],
    var_name="Type",
    value_name="Discharges"
)

fig_discharge = px.line(
    discharge_chart_data,
    x="Month",
    y="Discharges",
    color="Type",
    markers=True,
    title=f"Actual vs Predicted Discharges - {selected_model}"
)

st.plotly_chart(
    fig_discharge,
    use_container_width=True
)

# Care Forecasting Error Comparison
st.header("Care Forecasting Error Comparison")

care_mae_chart = px.bar(
    comparison.sort_values("Care_MAE"),
    x="Model",
    y="Care_MAE",
    title="HHS Care Model Comparison - MAE",
    text_auto=".2f"
)

st.plotly_chart(
    care_mae_chart,
    use_container_width=True
)

# Discharge Forecasting Error Comparison
st.header("Discharge Forecasting Error Comparison")

discharge_mae_chart = px.bar(
    comparison.sort_values("Discharge_MAE"),
    x="Model",
    y="Discharge_MAE",
    title="Discharge Model Comparison - MAE",
    text_auto=".2f"
)

st.plotly_chart(
    discharge_mae_chart,
    use_container_width=True
)

# Care Prediction Interval
st.divider()
st.header("Prediction Interval Analysis")
st.subheader("HHS Care Prediction Interval")

fig_care_interval = px.line(
    care_intervals,
    x="Month",
    y=["Lower_Bound", "Upper_Bound", "Predicted_Care"],
    markers=True,
    title="HHS Care Forecast with Approximate 80% Error Range"
)

st.plotly_chart(
    fig_care_interval,
    use_container_width=True
)

# Discharge Prediction Interval
st.subheader("HHS Discharge Prediction Interval")

fig_discharge_interval = px.line(
    discharge_intervals,
    x="Month",
    y=["Lower_Bound", "Upper_Bound", "Predicted_Discharges"],
    markers=True,
    title="HHS Discharge Forecast with Approximate 80% Error Range"
)

st.plotly_chart(
    fig_discharge_interval,
    use_container_width=True
)

# Future Forecast
st.divider()
st.header("Future Care Load and Discharge Forecast")
st.markdown(
    "Explore the model-generated forecasts beyond the last date "
    "available in the dataset."
)

forecast_horizon = st.slider(
    "Select Forecast Horizon (Months)",
    min_value=1,
    max_value=len(future_forecasts),
    value=len(future_forecasts)
)

selected_future = future_forecasts.head(
    forecast_horizon
).copy()

# Future HHS Care Forecast
st.subheader("Future HHS Care Forecast")

fig_future_care = px.line(
    selected_future,
    x="Month",
    y="Predicted_HHS_Care",
    markers=True,
    title="Predicted HHS Care Levels"
)

st.plotly_chart(
    fig_future_care,
    use_container_width=True
)

# Future Discharge Forecast
st.subheader("Future Discharge Forecast")

fig_future_discharge = px.line(
    selected_future,
    x="Month",
    y="Predicted_Discharges",
    markers=True,
    title="Predicted HHS Discharges"
)

st.plotly_chart(
    fig_future_discharge,
    use_container_width=True
)

# Future Forecast Results Table
st.subheader("Future Forecast Results")

st.dataframe(
    selected_future.round(2),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "These are model-generated estimates beyond the dataset's "
    "last date. They are not official projections or current forecasts."
)

# Footer
st.divider()

st.caption(
    "Project: Predictive Forecasting of Care Load & Placement Demand | "
    "Forecasts are analytical estimates, not official projections."
)
