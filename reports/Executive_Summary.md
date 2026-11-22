# Executive Summary
## Predictive Forecasting of Care Load & Placement Demand

**Project:** Unified Mentor Internship Project  
**Domain:** Data Science, Time-Series Forecasting and Predictive Analytics  
**Dataset:** U.S. Department of Health and Human Services (HHS) Unaccompanied Alien Children Program

### 1. Project Overview

This project uses historical data from the U.S. Department of Health and Human Services (HHS) to study and forecast the number of unaccompanied children in HHS care.

The main objective is to turn historical reporting data into useful analytical insights that can help planners understand changes in care levels, monitor discharge activity, and explore potential future demand.

The project applies data analysis, statistical forecasting, and machine-learning techniques. An interactive Streamlit dashboard presents the findings.

### 2. Dataset at a Glance

The dataset contains records from **January 12, 2023, to December 21, 2025**.

Key observations:

- 720 observed reporting records.
- 6 original data columns.
- 355 calendar dates without observations.
- 66.98% reporting coverage across the full calendar period.
- No duplicate reporting dates were identified.

Missing dates were treated as unobserved records rather than zero-valued observations. This avoids creating misleading daily activity figures.

### 3. Key Findings

- Average HHS care level across the observed records was approximately **6,061 children**.
- Average daily discharges were approximately **173 children**.
- HHS care levels ranged from 1,972 to 11,516.
- Discharges ranged from 0 to 505.
- HHS care levels and discharges showed a strong positive historical correlation of approximately 0.92. This is an association and does not establish causation.
- Yearly average care levels and discharges declined between 2023 and 2025.

These findings describe the available historical reporting data and should not be interpreted as a complete measure of operational capacity or individual placement outcomes.

### 4. Forecasting Results

The project compared baseline, statistical, and machine-learning forecasting approaches.

**Monthly forecasting**

- ARIMA achieved the lowest MAE for HHS care forecasting: **126.81**.
- Random Forest achieved the lowest MAE for discharge forecasting: **2.56**.

**Daily forecasting**

- The Naive baseline achieved the lowest care-load MAE: **14.59**.
- The Moving Average baseline achieved the lowest discharge MAE: **3.81**.

MAE represents the average absolute difference between actual and predicted values. Lower values indicate better accuracy.

The monthly test set contained only seven observations. Daily baseline and machine-learning evaluations also used different evaluation procedures. Therefore, the results are preliminary and should be validated using a standardized evaluation process before operational use.

### 5. Planning and Operational Recommendations

The analysis suggests several ways historical data and forecasting tools could support planning:

1. **Monitor care levels regularly:** Track changes in the number of children in HHS care and identify unusual increases or decreases.
2. **Review intake and discharge trends together:** Monitor transfers into HHS care alongside discharges to understand potential flow pressure.
3. **Use forecasts as planning support:** Consider model estimates alongside current operational information rather than treating them as exact future counts.
4. **Monitor data completeness:** Record reporting gaps and improve consistency so that future forecasting models can use more reliable daily observations.
5. **Validate warning indicators:** Test proposed alert thresholds against additional historical data and operational feedback before using them for formal decisions.

### 6. Dashboard and Project Deliverables

The project includes:

- Exploratory data analysis and data-quality checks.
- Monthly and daily feature engineering.
- Statistical and machine-learning model comparisons.
- Forecasting and approximate uncertainty analysis.
- Historical intake and discharge analysis.
- Early-warning indicators.
- Interactive Streamlit dashboard.
- Research report and this executive summary.

**Live dashboard:** https://predictive-care-load-forecasting.streamlit.app/

**GitHub repository:** https://github.com/saisandeepdonthala-dss/Predictive-Care-Load-Forecasting

### 7. Limitations and Responsible Use

The dataset contains periodic aggregate reporting rather than individual-level admission, discharge, or placement records. It does not provide complete information about length of stay, staffing, or official facility capacity.

The reporting series also contains missing calendar dates. Daily forecasts are therefore based on a subset of observed dates, and the monthly evaluation has a limited test period.

The future estimates generated beyond the December 2025 dataset cutoff are model outputs, not verified actual figures. Early-warning thresholds are analytical assumptions, not official HHS standards. Approximate uncertainty ranges should not be interpreted as formally validated confidence intervals.

The dashboard and models are intended for analytical exploration and decision support, not automated operational decisions.

### 8. Conclusion

This project demonstrates how historical HHS reporting data can be analyzed using time-series methods and machine-learning techniques to understand care-load patterns and explore future demand.

The results provide a foundation for more consistent monitoring, improved forecasting, and evidence-based planning. With more complete reporting data, longer validation periods, and operational information, the forecasting framework could be developed further to support more reliable planning decisions.