# Predictive Forecasting of Care Load & Placement Demand

## Project Overview

This project focuses on analyzing and forecasting the number of unaccompanied children in the care of the U.S. Department of Health and Human Services (HHS).

The objective is to study historical care levels, discharge patterns, and reporting trends using statistical time-series techniques and machine learning models. The project compares multiple forecasting approaches and develops analytical estimates that can support care capacity planning and resource management.

An interactive Streamlit dashboard presents historical trends, model performance, forecasting results, prediction uncertainty, and early-warning indicators.

## Project Objectives

- Analyze historical HHS care and discharge data.
- Understand trends and patterns in care levels and discharges.
- Perform data cleaning and exploratory data analysis.
- Examine irregular reporting intervals and missing dates.
- Develop monthly and daily forecasting datasets.
- Create time-based and lag-based features.
- Build and compare statistical and machine learning models.
- Evaluate forecasting performance using MAE and RMSE.
- Generate care load and discharge estimates.
- Analyze intake and discharge patterns.
- Develop historical early-warning indicators.
- Build an interactive Streamlit dashboard.
- Present findings through a research report and executive summary.

## Dataset Description

The project uses the HHS Unaccompanied Alien Children Program dataset.

The dataset contains the following columns:

- Date
- Children apprehended and placed in CBP custody
- Children in CBP custody
- Children transferred out of CBP custody
- Children in HHS Care
- Children discharged from HHS Care

The original dataset contains 1,170 rows. After removing completely empty rows and cleaning the data, 720 records were retained.

The available reporting dates range from January 2023 to December 2025.

### Data Coverage

- Original dataset: 1,170 rows
- Retained records: 720
- Monthly observations: 36
- Daily calendar dates: 1,075
- Observed reporting dates: 720
- Missing reporting dates: 355
- Daily reporting coverage: approximately 66.98%

Missing reporting dates are treated as unobserved records rather than zero values. This avoids introducing artificial values into the forecasting analysis.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- Plotly
- Streamlit
- SQLite

## Project Structure

```text
Predictive_Care_Load_Forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── inspect_data.py
│   ├── clean_data.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── model_preparation.py
│   ├── baseline_models.py
│   ├── arima_model.py
│   ├── exponential_smoothing_model.py
│   ├── random_forest_model.py
│   ├── gradient_boosting_model.py
│   ├── model_comparison.py
│   ├── model_evaluation.py
│   ├── prediction_intervals.py
│   ├── future_forecasting.py
│   ├── daily_time_series_prep.py
│   ├── time_series_decomposition.py
│   ├── daily_feature_engineering.py
│   ├── daily_feature_quality.py
│   ├── daily_model_preparation.py
│   ├── daily_model_split.py
│   ├── daily_baseline_models.py
│   ├── daily_random_forest_model.py
│   ├── daily_gradient_boosting_model.py
│   ├── daily_model_comparison.py
│   ├── daily_forecast_visualization.py
│   ├── intake_discharge_analysis.py
│   └── early_warning_analysis.py
│
├── models/
│
├── results/
│
├── dashboard/
│   └── app.py
│
├── reports/
│   ├── Project_Report.md
│   └── Executive_Summary.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Project Methodology

### 1. Data Inspection and Cleaning

- Inspected the original HHS dataset.
- Removed completely empty rows.
- Converted date values into datetime format.
- Converted numeric columns into appropriate data types.
- Sorted records chronologically.
- Checked missing values and duplicate dates.
- Created a cleaned dataset for further analysis.

### 2. Exploratory Data Analysis

Analyzed the dataset to understand:

- HHS care level trends.
- Discharge patterns.
- Relationships between dataset variables.
- Yearly and monthly trends.
- Reporting frequency and gaps.
- Changes in care levels over time.

The analysis also examined the relationship between HHS care levels and discharges. A strong historical association was observed, but this does not establish causation.

### 3. Monthly Feature Engineering

Created monthly average datasets and generated:

- Lag features.
- Three-month rolling averages.
- Month number.
- Quarter number.

The monthly dataset contains 36 observations and is used for monthly forecasting and model comparison.

### 4. Monthly Forecasting Models

The following six forecasting approaches were developed:

1. Naive Forecasting
2. Moving Average
3. ARIMA
4. Exponential Smoothing
5. Random Forest
6. Gradient Boosting

### 5. Daily Time-Series Analysis

A daily calendar-based dataset was developed to examine reporting patterns and short-term changes.

The analysis included:

- Daily time-series preparation.
- Reporting gap analysis.
- Time-series decomposition.
- Lag and rolling statistical features.
- Calendar-based features.
- Feature quality checks.
- Chronological train-test splitting.

The daily modeling dataset contained 508 usable records after feature preparation.

The chronological split consisted of:

- Training records: 406
- Testing records: 102

Missing dates were not automatically treated as zero observations.

### 6. Daily Forecasting Models

The following models were developed for daily care and discharge forecasting:

1. Naive Forecasting
2. Moving Average
3. Random Forest
4. Gradient Boosting

The daily baseline models were also evaluated to understand whether more complex machine learning models improved forecasting performance.

### 7. Model Evaluation

Model performance was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Monthly forecasting used a chronological training and testing approach with rolling one-month-ahead evaluation.

Daily forecasting used a chronological split, with baseline and machine learning models evaluated using their respective forecasting procedures.

Because the evaluation procedures differ, monthly and daily model errors should not be compared as directly equivalent results.

### 8. Prediction Uncertainty

Approximate 80% empirical error ranges were calculated using historical test residuals for monthly forecasts.

These ranges provide an indication of forecasting uncertainty. They are not formal statistical prediction intervals.

### 9. Intake and Discharge Analysis

An additional analysis examined:

- Children transferred out of CBP custody.
- HHS care levels.
- Children discharged from HHS care.
- Historical changes in intake and discharge patterns.

Transfers from CBP and HHS discharges provide useful indicators of movement through the care system. However, they do not represent a complete population accounting of all admissions and exits.

### 10. Early-Warning Indicators

Historical indicators were developed to help identify changes in care load and potential flow pressure.

The analysis considers changes in:

- HHS care levels.
- Discharge activity.
- Historical movement patterns.

Warning thresholds are analytical assumptions used for exploratory monitoring. They are not official HHS alert standards or validated operational triggers.

### 11. Future Forecasting

Monthly forecasting models generated six-month out-of-sample estimates for:

- HHS care levels.
- HHS discharges.

The forecast period extends beyond the final dataset reporting date of December 2025.

These are model-generated historical estimates for January to June 2026, not verified actual values or official HHS projections.

## Monthly Model Performance

The following results were obtained during the evaluation period from June to December 2025.

| Model | Care MAE | Care RMSE | Discharge MAE | Discharge RMSE |
|---|---:|---:|---:|---:|
| ARIMA | 126.81 | 162.15 | 3.35 | 3.87 |
| Naive | 145.79 | 163.43 | 2.96 | 3.50 |
| Random Forest | 152.31 | 179.22 | 2.56 | 2.97 |
| Exponential Smoothing | 213.66 | 232.49 | 7.01 | 7.97 |
| Moving Average | 216.32 | 226.70 | 3.77 | 4.20 |
| Gradient Boosting | 218.47 | 291.68 | 6.44 | 7.97 |

### Best Monthly Models

- **HHS Care Forecast:** ARIMA (MAE: 126.81)
- **HHS Discharge Forecast:** Random Forest (MAE: 2.56)

The evaluation contains only seven test months, so the results should be interpreted as preliminary.

## Daily Model Performance

The daily models were evaluated on a chronological test dataset containing 102 records.

### Daily Baseline Results

| Model | Care MAE | Care RMSE | Discharge MAE | Discharge RMSE |
|---|---:|---:|---:|---:|
| Naive | 14.59 | 28.63 | 5.23 | Not reported |
| Moving Average | 47.03 | Not reported | 3.81 | 4.79 |

### Daily Machine Learning Results

| Model | Care MAE | Discharge MAE |
|---|---:|---:|
| Random Forest | 96.32 | 6.20 |
| Gradient Boosting | 136.77 | 6.67 |

### Best Daily Models

- **Daily HHS Care Forecast:** Naive baseline (MAE: 14.59)
- **Daily HHS Discharge Forecast:** Moving Average baseline (MAE: 3.81)

The baseline models performed better than the tested machine learning models on the daily evaluation. This demonstrates that more complex models do not necessarily produce better forecasts, particularly with limited and irregularly reported data.

## Key Findings

- HHS care levels experienced substantial changes during the study period.
- Reporting intervals were irregular, with 355 missing dates in the daily calendar.
- Monthly and daily forecasting approaches produced different model performance results.
- ARIMA achieved the lowest monthly MAE for HHS care forecasting.
- Random Forest achieved the lowest monthly MAE for HHS discharge forecasting.
- Daily Naive forecasting performed best for care levels.
- Daily Moving Average forecasting performed best for discharges.
- More complex machine learning models did not consistently outperform baseline models.
- Historical care and discharge levels showed a strong association.
- Early-warning indicators can help explore historical changes in care load and flow pressure.

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard with the following features:

- Key Performance Indicators (KPIs).
- Historical HHS care and discharge trends.
- Forecasting model comparison.
- Interactive model selection.
- Care forecast visualization.
- Discharge forecast visualization.
- MAE comparison charts.
- Prediction uncertainty visualizations.
- Future forecast charts.
- Forecast horizon selector from 1 to 6 months.
- Future forecast results table.
- Daily forecasting analysis.
- Early-warning indicators.
- Historical intake and discharge analysis.

### Live Dashboard

[Open the Predictive Care Load Forecasting Dashboard](https://predictive-care-load-forecasting.streamlit.app/)

## Research and Documentation

The project includes two main documentation deliverables:

- **Project Research Report:** Covers data preparation, exploratory analysis, forecasting methodology, model evaluation, daily analysis, findings, and limitations.
- **Executive Summary:** Presents the main findings and recommendations in a concise format for government and planning stakeholders.

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/saisandeepdonthala-dss/Predictive-Care-Load-Forecasting.git
```

### Step 2: Open the Project Folder

```bash
cd Predictive-Care-Load-Forecasting
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 6: Prepare the Dataset

Place the original HHS dataset inside:

```text
data/raw/
```

Run the relevant data inspection, cleaning, and preparation scripts before running analyses that depend on their outputs.

### Step 7: Run the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser.

**Note:** The dashboard depends on the processed datasets and result files generated by the project scripts. Ensure the required files are available before launching it.

## Limitations

- The dataset contains irregular reporting intervals.
- The daily calendar includes unobserved reporting dates.
- The monthly dataset contains only 36 observations.
- The monthly evaluation period contains only seven test months.
- The daily and monthly evaluation procedures differ.
- Changes in reporting patterns or operational conditions may affect model performance.
- Future forecast errors may accumulate over time.
- Prediction uncertainty ranges are approximate empirical estimates, not formal statistical confidence intervals.
- Early-warning thresholds are analytical assumptions, not official HHS standards.
- The dataset does not provide individual admission and discharge histories, length of stay, staffing levels, or official facility capacity.
- Intake and discharge comparisons are analytical indicators, not a complete population accounting.
- Forecasts are analytical estimates and should not be treated as official HHS projections.

## Conclusion

This project demonstrates how time-series analysis and machine learning can be used to study historical HHS care levels and discharge patterns.

By developing monthly and daily forecasting approaches, comparing statistical and machine learning models, and examining intake, discharge, and early-warning indicators, the project provides an analytical framework for understanding changes in care demand.

The interactive Streamlit dashboard presents the results in an accessible format, allowing users to explore historical trends, model performance, forecasting estimates, and potential changes in care load.

The project highlights the importance of data quality, appropriate model selection, and careful interpretation of forecasting results when using historical data to support planning decisions.

## Author

**Data Science Project**

**Project:** Predictive Forecasting of Care Load & Placement Demand

**Technologies:** Python, Pandas, NumPy, Scikit-learn, Statsmodels, Time-Series Forecasting, Machine Learning, Streamlit

## Project Links

- **GitHub Repository:** https://github.com/saisandeepdonthala-dss/Predictive-Care-Load-Forecasting
- **Live Dashboard:** https://predictive-care-load-forecasting.streamlit.app/