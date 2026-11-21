# Predictive Forecasting of Care Load & Placement Demand

## Project Overview

This project focuses on predicting the number of unaccompanied children in the care of the U.S. Department of Health and Human Services (HHS).

The main goal is to analyze historical care levels and discharge patterns, compare different forecasting models, and generate estimates that can support care capacity planning and resource management.

An interactive Streamlit dashboard is also developed to display historical predictions, model performance, prediction intervals, and future forecasts.

## Project Objectives

- Analyze historical HHS care and discharge data.
- Understand trends and patterns in care levels.
- Perform data cleaning and exploratory data analysis.
- Create time-based features for forecasting.
- Develop and compare multiple forecasting models.
- Evaluate models using MAE and RMSE.
- Generate future care load and discharge estimates.
- Build an interactive Streamlit dashboard.

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
│   └── future_forecasting.py
│
├── models/
│
├── results/
│
├── dashboard/
│   └── app.py
│
├── reports/
│   └── Project_Report.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Project Methodology

### 1. Data Inspection and Cleaning

- Inspected the original dataset.
- Removed completely empty rows.
- Converted date values into datetime format.
- Converted numeric columns into appropriate data types.
- Sorted records chronologically.
- Checked missing values and duplicate dates.

### 2. Exploratory Data Analysis

Performed analysis to understand:

- HHS care level trends.
- Discharge patterns.
- Correlations between dataset variables.
- Yearly and monthly trends.
- Quarterly reporting patterns.
- Changes in care levels over time.

### 3. Feature Engineering

Created monthly average datasets and generated:

- Lag features.
- Three-month rolling averages.
- Month number.
- Quarter number.

### 4. Forecasting Models

The following six models were developed and compared:

1. Naive Forecasting
2. Moving Average
3. ARIMA
4. Exponential Smoothing
5. Random Forest
6. Gradient Boosting

### 5. Model Evaluation

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

A chronological train-test split and rolling one-month-ahead evaluation were used.

### 6. Prediction Intervals

Approximate 80% empirical error ranges were calculated using historical test residuals to show uncertainty around the selected model predictions.

These ranges are indicative and are not formal statistical prediction intervals.

### 7. Future Forecasting

Generated six-month out-of-sample estimates for:

- HHS care levels.
- HHS discharges.

The forecasts extend beyond the dataset's final reporting date of December 2025.

## Model Performance

The following results were obtained during the evaluation period from June to December 2025.

| Model | Care MAE | Care RMSE | Discharge MAE | Discharge RMSE |
|---|---:|---:|---:|---:|
| ARIMA | 126.81 | 162.15 | 3.35 | 3.87 |
| Naive | 145.79 | 163.43 | 2.96 | 3.50 |
| Random Forest | 152.31 | 179.22 | 2.56 | 2.97 |
| Exponential Smoothing | 213.66 | 232.49 | 7.01 | 7.97 |
| Moving Average | 216.32 | 226.70 | 3.77 | 4.20 |
| Gradient Boosting | 218.47 | 291.68 | 6.44 | 7.97 |

### Best Performing Models

- **HHS Care Forecast:** ARIMA
- **HHS Discharge Forecast:** Random Forest

These results are based on seven test months and should be treated as preliminary.

## Streamlit Dashboard

The project includes an interactive dashboard with the following features:

- Key Performance Indicators (KPIs).
- Forecasting model comparison.
- Interactive model selection.
- HHS care forecast visualization.
- Discharge forecast visualization.
- Care and discharge MAE comparison charts.
- Prediction interval visualizations.
- Future forecast charts.
- Forecast horizon selector from 1 to 6 months.
- Future forecast results table.

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

### Step 6: Run the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser.

**Note:** Run the data preparation and forecasting scripts first to generate the required processed datasets and result files. Place the original dataset in `data/raw/`.

## Key Findings

- HHS care levels showed a major decline during early 2025.
- Discharge patterns also changed during the analysis period.
- ARIMA achieved the lowest MAE for HHS care forecasting.
- Random Forest achieved the lowest MAE for discharge forecasting.
- The six-month forecast provides an example of how historical trends can be used for planning estimates.

## Limitations

- The dataset contains irregular reporting intervals.
- The evaluation period contains only seven test months.
- Changes in reporting patterns or operational conditions may affect model performance.
- Future forecast errors may accumulate over time.
- Prediction intervals are approximate empirical ranges, not formal statistical confidence intervals.
- Forecasts are analytical estimates and should not be treated as official HHS projections.

## Conclusion

This project demonstrates how time-series analysis and machine learning can be used to study HHS care levels and discharge patterns.

By comparing statistical and machine learning models, the project identifies suitable models for forecasting care load and discharge demand.

The interactive dashboard presents the results in a simple and understandable way, helping users explore historical trends, model performance, and future estimates.

## Author

**Data Science Project**

**Project:** Predictive Forecasting of Care Load & Placement Demand

**Tools:** Python, Machine Learning, Time Series Forecasting, Streamlit