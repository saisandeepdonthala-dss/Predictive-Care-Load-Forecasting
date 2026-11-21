# Project Report: Predictive Forecasting of Care Load & Placement Demand

## 1. Executive Summary

This project analyzes historical data from the Unaccompanied Alien Children (UAC) Program to understand trends in care load and child discharges. It develops forecasting models to estimate future care load and discharge demand.

The project uses Python, Pandas, NumPy, Scikit-learn, Statsmodels, and Streamlit. Six forecasting approaches are evaluated, and the best-performing models are selected separately for care load and discharges.

## 2. Problem Statement

Understanding changes in care load and discharge demand can support operational planning and resource allocation.

The main objectives are:
- Analyze historical HHS care and discharge trends.
- Identify patterns and relationships among available variables.
- Develop and compare forecasting models.
- Generate future estimates of care load and discharges.
- Present results through an interactive dashboard.

## 3. Dataset Description

**Dataset:** HHS Unaccompanied Alien Children Program

**Source:** Publicly available HHS program data.

- Original records: 1,170
- Records after removing empty rows: 720
- Time period: January 2023 – December 2025
- Columns: 6
- Missing values after cleaning: 0
- Duplicate dates: 0

The dataset includes date, HHS care, discharges, and other operational measures.

## 4. Data Cleaning and Preprocessing

The following steps were performed:

1. Removed completely empty rows.
2. Converted the Date column into datetime format.
3. Removed commas from numeric values and converted them into numeric types.
4. Sorted records chronologically.
5. Checked for missing values and duplicate dates.
6. Saved the cleaned dataset as `data/processed/cleaned_data.csv`.

## 5. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset and identify trends.

Key observations:
- Average HHS care was approximately 6,061.
- Average discharges were approximately 173.
- HHS care ranged from 1,972 to 11,516.
- Discharges ranged from 0 to 505.
- Both care load and discharges showed a substantial decline in their yearly averages between 2023 and 2025.
- HHS care and discharges showed a strong positive correlation of approximately 0.92.

Correlation indicates association and does not establish causation.

Visualizations included care trends, discharge trends, correlation heatmap, distribution analysis, and quarterly trends.

## 6. Feature Engineering

The dataset was aggregated into monthly averages.

Additional features included:
- Previous one-, two-, and three-month care values.
- Previous one-, two-, and three-month discharge values.
- Three-month rolling averages.
- Month number.
- Quarter number.

The resulting feature dataset was saved as `data/processed/featured_data.csv`.

## 7. Model Development

Six forecasting approaches were evaluated:

1. Naive Forecasting
2. Moving Average
3. ARIMA (1,1,1)
4. Exponential Smoothing
5. Random Forest
6. Gradient Boosting

The data was evaluated chronologically, using June–December 2025 as the seven-month test period.

A rolling one-month-ahead evaluation approach was used to compare model predictions consistently.

## 8. Model Performance

The models were evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

| Model | Care MAE | Care RMSE | Discharge MAE | Discharge RMSE |
|---|---:|---:|---:|---:|
| ARIMA | 126.81 | 162.15 | 3.35 | 3.87 |
| Naive | 145.79 | 163.43 | 2.96 | 3.50 |
| Random Forest | 152.31 | 179.22 | 2.56 | 2.97 |
| Exponential Smoothing | 213.66 | 232.49 | 7.01 | 7.97 |
| Moving Average | 216.32 | 226.70 | 3.77 | 4.20 |
| Gradient Boosting | 218.47 | 291.68 | 6.44 | 7.97 |

### Best-performing models

- **HHS Care:** ARIMA, with an MAE of 126.81.
- **Discharges:** Random Forest, with an MAE of 2.56.

The best model differs by forecasting target, so separate models were selected.

## 9. Future Forecasting

ARIMA was used for HHS care forecasting, while Random Forest was used for discharge forecasting.

The project generated six monthly estimates beyond the dataset's December 2025 cutoff, covering January–June 2026.

| Month | Forecast HHS Care | Forecast Discharges |
|---|---:|---:|
| Jan 2026 | 2532.54 | 21.30 |
| Feb 2026 | 2472.38 | 22.78 |
| Mar 2026 | 2513.85 | 18.13 |
| Apr 2026 | 2485.26 | 15.92 |
| May 2026 | 2504.97 | 15.63 |
| Jun 2026 | 2491.38 | 15.57 |

These are model-generated projections beyond the available dataset, not verified actual values or current forecasts.

## 10. Prediction Interval Analysis

Approximate 80% error ranges were calculated using the 10th and 90th percentiles of the seven test-period residuals.

These ranges provide an indication of historical prediction errors. They are not formal statistical prediction intervals, and their reliability is limited by the small test sample.

## 11. Streamlit Dashboard

An interactive Streamlit dashboard was developed to present the analysis and forecasting results.

Dashboard features:
- Key performance indicators.
- Model performance comparison.
- Interactive model selection.
- Historical care and discharge trends.
- Forecast analysis.
- Error comparison charts.
- Approximate prediction interval visualization.
- Future forecast horizon selector.
- Forecast result tables.

The dashboard can be launched using:

`streamlit run dashboard/app.py`

## 12. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- Streamlit
- Plotly
- Joblib

## 13. Limitations

- The dataset contains 720 records and ends in December 2025.
- Reporting dates are irregular, so monthly averages are used for modeling.
- The test period contains only seven months.
- Forecasts beyond the available data may accumulate errors.
- Approximate prediction ranges are based on a small number of residuals.
- Historical relationships do not establish the causes of changes in care load or discharges.
- Forecasts should not be treated as official operational projections.

## 14. Conclusion

This project demonstrates an end-to-end forecasting workflow, from data cleaning and exploratory analysis to feature engineering, model development, evaluation, and interactive dashboard creation.

ARIMA achieved the lowest care-load MAE, while Random Forest achieved the lowest discharge MAE in the selected test period.

The dashboard provides an accessible way to explore historical trends, compare model performance, and review forecast estimates.

The project can be extended with additional historical data, longer evaluation periods, and further forecasting techniques to improve reliability.
