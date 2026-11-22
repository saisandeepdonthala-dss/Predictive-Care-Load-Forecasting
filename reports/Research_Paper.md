# Project Report: Predictive Forecasting of Care Load & Placement Demand

## 1. Executive Summary

This project analyzes historical data from the Unaccompanied Alien Children (UAC) Program to understand trends in HHS care levels, child discharges, and potential care-flow pressure.

The main objective is to develop forecasting models that can estimate care levels and discharge demand using historical operational data.

The project follows an end-to-end data science workflow, including data cleaning, exploratory data analysis, feature engineering, time-series analysis, machine learning, model evaluation, and dashboard development.

Two forecasting approaches were explored:

- Monthly forecasting using six statistical and machine learning approaches.
- Daily forecasting using historical reporting dates, lag features, rolling statistics, and baseline and machine learning models.

For monthly forecasting, ARIMA achieved the lowest MAE for HHS care, while Random Forest performed best for discharges.

For the daily experiment, the Naive model achieved the lowest care MAE, while the Moving Average model achieved the lowest discharge MAE.

An interactive Streamlit dashboard presents historical analysis, model comparisons, uncertainty estimates, future estimates, early-warning indicators, and daily model evaluation.

## 2. Problem Statement

Understanding changes in care load and discharge demand can support operational planning and resource allocation.

The main objectives are:

- Analyze historical HHS care and discharge trends.
- Identify patterns and relationships among available variables.
- Develop and compare forecasting models.
- Generate estimates of future care load and discharge demand.
- Analyze potential differences between transfer-outs and discharges.
- Identify historical early-warning indicators.
- Present analytical results through an interactive dashboard.

## 3. Dataset Description

**Dataset:** HHS Unaccompanied Alien Children Program

**Source:** Publicly available HHS program data.

Dataset characteristics:

- Original records: 1,170
- Records after removing completely empty rows: 720
- Reporting period: January 12, 2023 – December 21, 2025
- Columns: 6
- Missing values in the cleaned source columns: 0
- Duplicate dates: 0

The dataset includes:

1. Date
2. Children apprehended and placed in CBP custody
3. Children in CBP custody
4. Children transferred out of CBP custody
5. Children in HHS Care
6. Children discharged from HHS Care

The data contains irregular reporting dates. Therefore, missing calendar dates were treated as unobserved reporting days rather than zero-value observations.

## 4. Data Cleaning and Preprocessing

The following steps were performed:

1. Removed completely empty rows.
2. Converted the Date column into datetime format.
3. Removed commas from numeric values and converted them into numeric types.
4. Sorted records chronologically.
5. Checked for missing values and duplicate dates.
6. Saved the cleaned dataset as `data/processed/cleaned_data.csv`.

Additional daily data-quality checks were performed.

The full calendar period contains 1,075 days, of which 720 have observed records.

- Observed reporting days: 720
- Missing calendar days: 355
- Reporting coverage: 66.98%
- Duplicate dates: 0

Missing dates were preserved as unobserved values. No automatic zero-filling or interpolation was applied to the original daily observations.

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

Visualizations included:

- HHS care trends.
- Discharge trends.
- Correlation heatmap.
- Distribution analysis.
- Quarterly and yearly trends.

## 6. Monthly Feature Engineering

The cleaned dataset was aggregated into monthly averages.

The monthly dataset contains 36 records.

The following features were created:

- Previous one-, two-, and three-month care values.
- Previous one-, two-, and three-month discharge values.
- Three-month rolling averages.
- Month number.
- Quarter number.

The resulting feature dataset was saved as `data/processed/featured_data.csv`.

## 7. Monthly Forecasting Model Development

Six forecasting approaches were evaluated:

1. Naive Forecasting
2. Moving Average
3. ARIMA (1,1,1)
4. Exponential Smoothing
5. Random Forest
6. Gradient Boosting

The data was evaluated chronologically, using June–December 2025 as the seven-month test period.

A rolling one-month-ahead evaluation approach was used. Statistical models were refitted as appropriate, and machine learning models were trained using historical observations available before each test month.

## 8. Monthly Model Performance

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

| Model | Care MAE | Care RMSE | Discharge MAE | Discharge RMSE |
|---|---:|---:|---:|---:|
| ARIMA | 126.81 | 162.15 | 3.35 | 3.87 |
| Naive | 145.79 | 163.43 | 2.96 | 3.50 |
| Random Forest | 152.31 | 179.22 | 2.56 | 2.97 |
| Exponential Smoothing | 213.66 | 232.49 | 7.01 | 7.97 |
| Moving Average | 216.32 | 226.70 | 3.77 | 4.20 |
| Gradient Boosting | 218.47 | 291.68 | 6.44 | 7.97 |

### Best-performing monthly models

- **HHS Care:** ARIMA, with an MAE of 126.81.
- **HHS Discharges:** Random Forest, with an MAE of 2.56.

The best model differs by forecasting target, so separate models were selected.

These results are preliminary because the test period contains only seven months.

## 9. Monthly Future Forecasting

ARIMA was selected for HHS care forecasting, while Random Forest was selected for discharge forecasting.

Six monthly estimates were generated beyond the dataset's December 2025 cutoff, covering January–June 2026.

| Month | Forecast HHS Care | Forecast Discharges |
|---|---:|---:|
| Jan 2026 | 2532.54 | 21.30 |
| Feb 2026 | 2472.38 | 22.78 |
| Mar 2026 | 2513.85 | 18.13 |
| Apr 2026 | 2485.26 | 15.92 |
| May 2026 | 2504.97 | 15.63 |
| Jun 2026 | 2491.38 | 15.57 |

These are model-generated estimates beyond the available dataset. They are not verified actual values, official projections, or current forecasts.

## 10. Prediction Uncertainty Analysis

Approximate 80% empirical error ranges were calculated using the 10th and 90th percentiles of historical test residuals.

The purpose of this analysis is to provide an indication of historical forecast errors and uncertainty.

Important limitations:

- The ranges are not formal statistical prediction intervals.
- Only seven test-month residuals were available.
- The estimated uncertainty should therefore be interpreted cautiously.

The dashboard displays the approximate ranges and their average widths for care and discharge forecasts.

## 11. Daily Time-Series Data Preparation

A separate daily forecasting workflow was developed to examine the original reporting observations without converting them into monthly averages.

The daily preparation process included:

1. Sorting observations by date.
2. Checking duplicate dates.
3. Creating a complete calendar index for data-quality analysis.
4. Preserving unobserved dates as missing values.
5. Calculating the number of days since the previous report.
6. Measuring reporting coverage and missing-date patterns.

The resulting daily observed dataset was saved as `data/processed/daily_observed_data.csv`.

The daily analysis retained the irregular reporting structure instead of assuming that every calendar day had a recorded value.

## 12. Time-Series Decomposition

Monthly HHS care and discharge data were decomposed into:

- Observed values.
- Trend component.
- Seasonal component.
- Residual component.

An additive decomposition with a 12-month seasonal period was used.

The analysis helped examine longer-term changes and possible annual patterns.

However, the dataset contains only 36 monthly observations, equivalent to three annual cycles. Therefore, seasonal patterns should be treated as exploratory rather than strong evidence of recurring annual behavior.

Decomposition outputs were saved under `results/decomposition/`.

## 13. Daily Feature Engineering

Daily calendar-based features were created separately from the monthly feature dataset.

The following features were generated:

**Lag features**
- Care lag: 1, 7, and 14 calendar days.
- Discharge lag: 1, 7, and 14 calendar days.
- Transfer-out lag: 1, 7, and 14 calendar days.

**Rolling features**
- Seven-day rolling mean.
- Seven-day rolling variance.
- Fourteen-day rolling mean.
- Fourteen-day rolling variance.

**Calendar features**
- Day of week.
- Month number.
- Quarter.
- Day of month.

A flow-pressure indicator was also calculated as:

Transfer-outs − Discharges

This is an analytical proxy only. CBP transfer-outs are not confirmed HHS admissions, and this calculation does not represent exact HHS population change.

Rolling features were calculated using previous calendar dates, with coverage checks to account for missing observations.

The resulting dataset was saved as `data/processed/daily_featured_data.csv`.

## 14. Daily Model Preparation and Evaluation

Separate datasets were prepared for HHS care and HHS discharges.

Rows without the required historical features were excluded.

Each target dataset contained 508 usable observations.

A chronological 80/20 split was applied:

| Dataset | Training Records | Testing Records |
|---|---:|---:|
| HHS Care | 406 | 102 |
| HHS Discharges | 406 | 102 |

Training period: February 5, 2023 – April 23, 2025.

Testing period: April 24, 2025 – December 21, 2025.

The daily experiment evaluates target values at observed reporting dates using historical lagged features. Because reporting dates are irregular, this should not be interpreted as a forecast for every consecutive calendar day.

## 15. Daily Model Development

Four daily forecasting approaches were evaluated:

1. Naive Forecasting
2. Moving Average
3. Random Forest
4. Gradient Boosting

The Naive baseline uses the previous observed target value.

The Moving Average baseline uses the average of the previous seven observed target values.

Random Forest and Gradient Boosting were trained using historical lagged and calendar features.

The machine learning models used the following principal settings:

- Random Forest: 100 estimators and maximum tree depth of 5.
- Gradient Boosting: 100 estimators, learning rate of 0.05, and maximum tree depth of 2.

The daily baseline predictions were evaluated using a walk-forward approach over the test observations.

## 16. Daily Model Performance

The daily models were evaluated using MAE and RMSE.

### HHS Care

| Model | MAE | RMSE |
|---|---:|---:|
| Naive | 14.59 | 28.63 |
| Moving Average | 47.03 | 68.48 |
| Random Forest | 96.32 | 117.31 |
| Gradient Boosting | 136.77 | 170.47 |

### HHS Discharges

| Model | MAE | RMSE |
|---|---:|---:|
| Moving Average | 3.81 | 4.79 |
| Naive | 5.23 | 6.43 |
| Random Forest | 6.20 | 7.42 |
| Gradient Boosting | 6.67 | 8.53 |

### Best-performing daily models

- **HHS Care:** Naive, with an MAE of 14.59.
- **HHS Discharges:** Moving Average, with an MAE of 3.81.

The simple baseline models outperformed the machine learning models in this daily experiment.

This suggests that recent observed values provided a useful reference for the tested observations.

The results should be interpreted with care because the baseline models use walk-forward evaluation, while the machine learning models use a static chronological training and testing split. The evaluation protocols are not perfectly identical, so a standardized comparison would be useful in future work.

## 17. Daily Forecast Visualizations

Actual-versus-predicted visualizations were generated for both targets.

The selected daily baseline models were:

- Naive for HHS care.
- Moving Average for HHS discharges.

The visualizations are available at:

- `results/daily_visualizations/care_actual_vs_predicted.png`
- `results/daily_visualizations/discharge_actual_vs_predicted.png`

These charts show historical backtest performance on observed reporting dates. They are not future daily forecasts.

## 18. Intake vs Discharge Balance Analysis

A separate analysis was performed to compare monthly CBP transfer-outs and HHS discharges.

The objective was to identify periods in which transfer-outs and discharges differed substantially.

The analysis uses the difference between the two measures as a potential flow-pressure indicator.

Important interpretation:

- CBP transfer-outs are not confirmed HHS admissions.
- The comparison is a flow-pressure proxy, not an exact calculation of HHS intake minus exits.
- The values represent monthly averages, not monthly totals.
- The proxy should not be interpreted as exact population growth or decline.

The analysis results were saved as `results/intake_discharge_balance.csv`.

## 19. Early-Warning Indicators

Historical early-warning indicators were developed to monitor changes in care load and potential flow pressure.

The analysis includes:

- Changes in HHS care levels.
- Percentage changes in care load.
- Historical flow-pressure indicators.
- Warning categories based on analytical thresholds.

These thresholds are project assumptions and are not official HHS alert standards.

The indicators describe historical changes and have not been validated as future surge predictions.

The results were saved as `results/early_warning_results.csv`.

## 20. Streamlit Dashboard

An interactive Streamlit dashboard was developed and deployed to present the project's analytical results.

Dashboard features include:

- Key performance indicators.
- Monthly model performance comparison.
- Interactive forecasting model selection.
- HHS care and discharge forecast charts.
- Forecast error comparison.
- Approximate uncertainty visualization.
- Future forecast horizon selector.
- Future forecast result tables.
- Intake versus discharge balance analysis.
- Historical early-warning indicators.
- Daily model performance comparison.
- Best daily model for each target.
- Daily actual-versus-predicted visualizations.
- Daily model MAE comparison.

The dashboard can be launched locally using:

`streamlit run dashboard/app.py`

**Live dashboard:** https://predictive-care-load-forecasting.streamlit.app/

**GitHub repository:** https://github.com/saisandeepdonthala-dss/Predictive-Care-Load-Forecasting

## 21. Technologies Used

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

## 22. Limitations

The project has several limitations:

1. The dataset contains 720 observed reporting records and ends in December 2025.
2. Reporting dates are irregular, and 355 calendar dates have no recorded observations.
3. Daily forecasting is evaluated at observed reporting dates rather than every calendar day.
4. The monthly model evaluation contains only seven test months.
5. The daily dataset contains 508 usable observations after feature preparation.
6. The daily baseline and machine learning models do not use perfectly identical evaluation protocols.
7. Approximate monthly uncertainty ranges are based on a small number of historical residuals.
8. Only three annual cycles are available for monthly seasonal decomposition.
9. CBP transfer-outs and HHS discharges do not provide an exact measure of HHS population change.
10. The dataset does not contain individual-level placement records, length-of-stay information, or official facility capacity thresholds.
11. Early-warning thresholds are analytical assumptions and have not been validated as operational alert standards.
12. Forecasts beyond the available data may accumulate errors and should not be treated as official operational projections.

## 23. Future Improvements

The project can be extended through:

- Additional historical observations.
- More consistent daily reporting data.
- Standardized walk-forward evaluation across all daily models.
- SARIMA and other time-series approaches.
- Multi-horizon forecasting.
- More robust uncertainty estimation.
- Scenario-based forecasting.
- Additional external variables where reliable data is available.
- Validation of early-warning thresholds with domain experts.
- Evaluation against actual future observations when they become available.

## 24. Conclusion

This project demonstrates an end-to-end forecasting and analytics workflow for examining HHS care levels, discharge patterns, and potential care-flow pressure.

The monthly forecasting experiment found ARIMA to be the best-performing model for HHS care and Random Forest to be the best-performing model for discharges.

The daily experiment found that simple baseline approaches performed better than the tested machine learning models on the selected evaluation data.

The project also includes time-series decomposition, data-quality analysis, historical early-warning indicators, intake-discharge comparisons, and approximate uncertainty analysis.

The deployed Streamlit dashboard brings these results together in an interactive interface.

Overall, the project demonstrates how historical operational data can be transformed into analytical insights and forecasting estimates that may support further research and planning. The results are exploratory and should be validated with additional data before being used for operational decisions.