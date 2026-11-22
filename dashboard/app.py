import streamlit as st
import pandas as pd
import plotly.express as px
import os

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Predictive Care Load Forecasting",
    page_icon="📊",
    layout="wide"
)

# DASHBOARD TITLE
st.title("Predictive Forecasting of Care Load & Placement Demand")

st.markdown(
    "An interactive dashboard for analyzing HHS care levels, "
    "discharge patterns, forecasting model performance, "
    "and potential care-flow pressure."
)

# CHECK REQUIRED FILES
comparison_path = "results/final_model_comparison.csv"
forecast_path = "results/standardized_forecast_results.csv"
future_forecast_path = "results/future_forecast_results.csv"
care_interval_path = "results/care_prediction_intervals.csv"
discharge_interval_path = "results/discharge_prediction_intervals.csv"

required_files = [
    comparison_path,
    forecast_path,
    future_forecast_path,
    care_interval_path,
    discharge_interval_path
]

for file_path in required_files:
    if not os.path.exists(file_path):
        st.error(
            f"Required file is missing: {file_path}. "
            "Please run the relevant Python scripts first."
        )
        st.stop()

# LOAD DATA
comparison = pd.read_csv(comparison_path)

forecasts = pd.read_csv(
    forecast_path,
    parse_dates=["Month"]
)

future_forecasts = pd.read_csv(
    future_forecast_path,
    parse_dates=["Month"]
)

care_intervals = pd.read_csv(
    care_interval_path,
    parse_dates=["Month"]
)

discharge_intervals = pd.read_csv(
    discharge_interval_path,
    parse_dates=["Month"]
)


# FIND BEST MODELS
best_care_model = comparison.loc[
    comparison["Care_MAE"].idxmin(), "Model"
]

best_discharge_model = comparison.loc[
    comparison["Discharge_MAE"].idxmin(), "Model"
]

# KEY PERFORMANCE INDICATORS
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

# MODEL PERFORMANCE COMPARISON
st.header("Model Performance Comparison")
st.dataframe(
    comparison.round(2),
    use_container_width=True,
    hide_index=True
)

# INTERACTIVE MODEL SELECTION
st.header("Interactive Forecast Analysis")

model_options = comparison["Model"].tolist()

selected_model = st.selectbox(
    "Select Forecasting Model",
    model_options,
    index=model_options.index(best_care_model)
)

# HHS CARE FORECAST
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

# HHS DISCHARGE FORECAST
st.subheader("HHS Discharge Forecast")

discharge_data = forecasts[
    forecasts["Model"] == selected_model
].copy()

discharge_chart_data = discharge_data.melt(
    id_vars=["Month"],
    value_vars=[
        "Actual_Discharges",
        "Predicted_Discharges"
    ],
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

# CARE FORECASTING ERROR COMPARISON
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

# DISCHARGE FORECASTING ERROR COMPARISON
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

# FORECAST UNCERTAINTY QUANTIFICATION
st.divider()
st.header("Forecast Uncertainty Quantification")
st.markdown(
    "Approximate empirical error ranges calculated from "
    "historical model evaluation residuals."
)

# HHS CARE UNCERTAINTY
st.subheader("HHS Care Forecast Uncertainty")
fig_care_interval = px.line(
    care_intervals,
    x="Month",
    y=[
        "Actual_Care",
        "Predicted_Care",
        "Lower_Bound",
        "Upper_Bound"
    ],
    markers=True,
    title=(
        "Actual vs Predicted HHS Care "
        "with Approximate 80% Error Range"
    ),
    labels={
        "value": "Average Number of Children",
        "Month": "Month",
        "variable": "Measurement"
    }
)

st.plotly_chart(
    fig_care_interval,
    use_container_width=True
)

# HHS DISCHARGE UNCERTAINTY
st.subheader("HHS Discharge Forecast Uncertainty")
fig_discharge_interval = px.line(
    discharge_intervals,
    x="Month",
    y=[
        "Actual_Discharges",
        "Predicted_Discharges",
        "Lower_Bound",
        "Upper_Bound"
    ],
    markers=True,
    title=(
        "Actual vs Predicted HHS Discharges "
        "with Approximate 80% Error Range"
    ),
    labels={
        "value": "Average Number of Discharges",
        "Month": "Month",
        "variable": "Measurement"
    }
)

st.plotly_chart(
    fig_discharge_interval,
    use_container_width=True
)

# NUMERICAL UNCERTAINTY SUMMARY
st.subheader("Uncertainty Range Summary")
care_width = (
    care_intervals["Upper_Bound"] -
    care_intervals["Lower_Bound"]
).mean()

discharge_width = (
    discharge_intervals["Upper_Bound"] -
    discharge_intervals["Lower_Bound"]
).mean()

col1, col2 = st.columns(2)

col1.metric(
    "Average Care Error-Range Width",
    f"{care_width:.2f}"
)

col2.metric(
    "Average Discharge Error-Range Width",
    f"{discharge_width:.2f}"
)

st.markdown("**Interpretation**")
st.write(
    "The error-range width represents the average distance "
    "between the estimated lower and upper bounds. "
    "A wider range indicates greater historical forecast uncertainty."
)

st.caption(
    "These are approximate empirical 80% error ranges based on "
    "the 10th and 90th percentiles of historical test residuals. "
    "They are not formal statistical prediction intervals. "
    "The evaluation contains only seven test months, so the "
    "estimated uncertainty is preliminary and should be "
    "interpreted cautiously."
)

# FUTURE FORECAST
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

# FUTURE HHS CARE FORECAST
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

# FUTURE DISCHARGE FORECAST
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

# FUTURE FORECAST RESULTS
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

# INTAKE VS DISCHARGE BALANCE ANALYSIS
st.divider()
st.header("Intake vs Discharge Balance Analysis")
balance_path = "results/intake_discharge_balance.csv"
if os.path.exists(balance_path):
    balance_df = pd.read_csv(balance_path)
    balance_df["Date"] = pd.to_datetime(
        balance_df["Date"]
    )

    st.markdown(
        "Monthly comparison of CBP transfer-outs and "
        "HHS discharges to monitor potential care-flow pressure."
    )

    fig_balance = px.line(
        balance_df,
        x="Date",
        y=[
            "Children transferred out of CBP custody",
            "Children discharged from HHS Care"
        ],
        markers=True,
        title="Monthly Transfer-Out vs HHS Discharge"
    )

    st.plotly_chart(
        fig_balance,
        use_container_width=True
    )

    fig_net_flow = px.bar(
        balance_df,
        x="Date",
        y="Net_Flow",
        color="Flow_Status",
        title="Monthly Net Flow Pressure Proxy"
    )

    st.plotly_chart(
        fig_net_flow,
        use_container_width=True
    )

    st.dataframe(
        balance_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "CBP transfer-outs are not confirmed HHS admissions. "
        "This comparison is a flow-pressure proxy, not an exact "
        "measure of HHS intake minus exits or population change. "
        "Values are monthly averages, not monthly totals."
    )

else:
    st.warning(
        "Run the intake-discharge analysis first."
    )

# EARLY-WARNING INDICATORS
st.divider()
st.header("Early-Warning Indicators for Healthcare Planners")
warning_path = "results/early_warning_results.csv"
if os.path.exists(warning_path):

    warning_df = pd.read_csv(warning_path)
    warning_df["Date"] = pd.to_datetime(
        warning_df["Date"]
    )

    st.markdown(
        "Historical monitoring indicators to identify changes "
        "in HHS care load and potential flow pressure."
    )

    # CARE LOAD CHANGE CHART
    fig_warning = px.bar(
        warning_df,
        x="Date",
        y="Care_Change_Percent",
        color="Care_Warning",
        title="Monthly HHS Care Load Change (%)",
        labels={
            "Care_Change_Percent": "Change (%)",
            "Date": "Month"
        }
    )

    st.plotly_chart(
        fig_warning,
        use_container_width=True
    )

    # FLOW WARNING SUMMARY
    flow_counts = (
        warning_df["Flow_Warning"]
        .value_counts()
        .reset_index()
    )

    flow_counts.columns = [
        "Warning Type",
        "Number of Months"
    ]

    fig_flow_warning = px.bar(
        flow_counts,
        x="Warning Type",
        y="Number of Months",
        color="Warning Type",
        title="Historical Flow Warning Summary"
    )

    st.plotly_chart(
        fig_flow_warning,
        use_container_width=True
    )

    st.subheader("Early-Warning Results")
    st.dataframe(
        warning_df.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Warning thresholds are analytical assumptions, not official "
        "HHS alert standards. These indicators describe historical "
        "changes and do not constitute validated future predictions. "
        "Flow warnings use the transfer-out versus discharge proxy."
    )

else:
    st.warning(
        "Run the early-warning analysis first."
    )

# DAILY FORECASTING ANALYSIS
st.divider()
st.header("Daily Forecasting Analysis")

st.markdown(
    "Daily model evaluation using observed reporting dates. "
    "These results are separate from the monthly forecasting analysis."
)

daily_comparison_path = (
    "results/daily_model_comparison/all_daily_model_comparison.csv"
)

daily_best_path = (
    "results/daily_model_comparison/best_daily_models.csv"
)

daily_care_path = (
    "results/daily_baseline/care_baseline_predictions.csv"
)

daily_discharge_path = (
    "results/daily_baseline/discharge_baseline_predictions.csv"
)

daily_files = [
    daily_comparison_path,
    daily_best_path,
    daily_care_path,
    daily_discharge_path
]

if all(os.path.exists(file) for file in daily_files):

    daily_comparison = pd.read_csv(daily_comparison_path)
    daily_best = pd.read_csv(daily_best_path)

    daily_care = pd.read_csv(
        daily_care_path,
        parse_dates=["Date"]
    )

    daily_discharge = pd.read_csv(
        daily_discharge_path,
        parse_dates=["Date"]
    )

    # DAILY MODEL PERFORMANCE
    st.subheader("Daily Model Performance Comparison")

    st.dataframe(
        daily_comparison.round(2),
        use_container_width=True,
        hide_index=True
    )

    # BEST DAILY MODELS
    st.subheader("Best Daily Model for Each Target")

    col1, col2 = st.columns(2)

    care_best = daily_best[
        daily_best["Target"] == "care"
    ]

    discharge_best = daily_best[
        daily_best["Target"] == "discharge"
    ]

    if not care_best.empty:
        col1.metric(
            "Best Daily Care Model",
            care_best.iloc[0]["Model"],
            f"MAE: {care_best.iloc[0]['MAE']:.2f}"
        )

    if not discharge_best.empty:
        col2.metric(
            "Best Daily Discharge Model",
            discharge_best.iloc[0]["Model"],
            f"MAE: {discharge_best.iloc[0]['MAE']:.2f}"
        )

    # CARE FORECAST
    st.subheader("Daily HHS Care: Actual vs Predicted")

    fig_daily_care = px.line(
        daily_care,
        x="Date",
        y=["Actual", "Naive_Prediction"],
        markers=True,
        labels={
            "value": "Number of Children",
            "Date": "Reporting Date",
            "variable": "Measurement"
        },
        title="Daily HHS Care - Naive Baseline"
    )

    st.plotly_chart(
        fig_daily_care,
        use_container_width=True
    )

    # DISCHARGE FORECAST
    st.subheader("Daily HHS Discharges: Actual vs Predicted")

    fig_daily_discharge = px.line(
        daily_discharge,
        x="Date",
        y=["Actual", "Moving_Average_Prediction"],
        markers=True,
        labels={
            "value": "Number of Discharges",
            "Date": "Reporting Date",
            "variable": "Measurement"
        },
        title="Daily HHS Discharges - Moving Average Baseline"
    )

    st.plotly_chart(
        fig_daily_discharge,
        use_container_width=True
    )

    # DAILY MODEL ERROR COMPARISON
    st.subheader("Daily Model MAE Comparison")

    fig_daily_mae = px.bar(
        daily_comparison.sort_values(
            ["Target", "MAE"]
        ),
        x="Model",
        y="MAE",
        color="Target",
        barmode="group",
        text_auto=".2f",
        title="Daily Forecasting Model Comparison - MAE"
    )

    st.plotly_chart(
        fig_daily_mae,
        use_container_width=True
    )

    st.caption(
        "Daily evaluation uses 102 chronological test observations "
        "per target. Predictions are evaluated at observed reporting "
        "dates, which are not necessarily consecutive calendar days. "
        "The displayed baseline predictions use walk-forward evaluation. "
        "These are historical backtest results, not future forecasts."
    )

else:
    st.warning(
        "Daily forecasting files are missing. "
        "Run the daily model preparation, training, and comparison scripts first."
    )

# FOOTER
st.divider()

st.caption(
    "Project: Predictive Forecasting of Care Load & Placement Demand | "
    "Forecasts are analytical estimates, not official projections."
)
