# Project Report: Chocolate Sales Forecasting

**Project Title**: Chocolate Sales Forecasting using Machine Learning  
**Dataset**: Chocolate Sales.csv  
**Technologies**: Python, Prophet, SARIMA, Streamlit, Plotly  
**Report Date**: 2024

---

## 1. Executive Summary

This project develops an end-to-end sales forecasting system for a chocolate distribution business. The system ingests historical transaction data, performs exploratory data analysis, and applies two complementary forecasting models — **Meta Prophet** (daily granularity) and **SARIMA** (monthly granularity) — to predict future revenue. An interactive **Streamlit dashboard** provides stakeholders with actionable insights through filterable charts, KPIs, and forecast visualisations.

---

## 2. Problem Statement

Sales teams and supply chain managers require reliable demand forecasts to:
- Optimise inventory and avoid stockouts or overstock situations
- Set realistic revenue targets and sales quotas
- Identify seasonal patterns to plan marketing campaigns
- Understand which products and markets drive the most value

The goal of this project is to build a forecasting pipeline that answers:
1. What will total revenue look like over the next 90 days?
2. What is the expected monthly revenue for the next 6 months?
3. Which products, countries, and salespeople are top performers?

---

## 3. Dataset Overview

### 3.1 Source & Format

| Attribute | Value |
|---|---|
| File | `Chocolate Sales.csv` |
| Format | CSV, UTF-8 with BOM |
| Encoding | UTF-8-sig |

### 3.2 Columns

| Column | Type | Description |
|---|---|---|
| `Sales Person` | String | Name of the sales representative |
| `Country` | String | Sale country (6 unique values) |
| `Product` | String | Chocolate product (22 unique values) |
| `Date` | String → datetime | Transaction date (`DD-Mon-YY`) |
| `Amount` | String → float | Revenue in USD (e.g., `"$5,320 "`) |
| `Boxes Shipped` | Integer | Units shipped in the transaction |

### 3.3 Data Quality Issues Found

| Issue | Resolution |
|---|---|
| `Amount` stored as dollar-formatted string (`"$5,320 "`) | Stripped `$`, `,`, whitespace; cast to `float` |
| `Date` in non-ISO format (`04-Jan-22`) | Parsed with `pd.to_datetime(..., format="%d-%b-%y")` |
| BOM character in column header (`﻿Sales Person`) | Read with `encoding="utf-8-sig"` |
| Some `Amount` values were `$168` or `$91` (no comma) | Regex-based stripping handled all cases uniformly |

---

## 4. Exploratory Data Analysis (EDA)

### 4.1 Temporal Coverage

The dataset spans **January 2022 – August 2022** (approximately 8 months), covering transactions across 6 countries and 22 chocolate products.

### 4.2 Revenue Distribution

- The dataset is **right-skewed** — a small number of transactions account for a disproportionately large share of revenue.
- High-value transactions exceed $15,000 (e.g., 99% Dark & Pure, Raspberry Choco).
- Low-value transactions are as small as $91 (Drinking Coco).

### 4.3 Seasonality Observations

- **Monthly trends**: Revenue shows moderate variability across months with no single dominant spike in the 8-month window.
- **Day-of-week effect**: Sales are distributed across all weekdays. No strong weekend suppression is visible in the data.
- **Weekly patterns**: Prophet's weekly seasonality component quantifies day-level cyclicality.

### 4.4 Top Products by Revenue

| Rank | Product | Est. Revenue Share |
|---|---|---|
| 1 | 99% Dark & Pure | High |
| 2 | Raspberry Choco | High |
| 3 | Peanut Butter Cubes | High |
| 4 | Fruit & Nut Bars | Medium-High |
| 5 | Organic Choco Syrup | Medium-High |

### 4.5 Geographic Distribution

All 6 markets (UK, USA, India, Australia, Canada, New Zealand) are represented. Revenue is distributed broadly, with no single country holding an extreme majority share.

---

## 5. Methodology

### 5.1 Data Pipeline

```
Raw CSV
   ↓ load_data()          — read CSV, strip column names
   ↓ clean_data()         — parse Amount & Date, drop nulls
   ↓ feature_engineer()   — add Year, Month, Week, Quarter, DayOfWeek
   ↓ aggregation fns()    — daily_total(), monthly_total(), by_product(), etc.
   ↓ forecasting models   — Prophet, SARIMA
   ↓ Streamlit dashboard  — visualise + interact
```

### 5.2 Feature Engineering

| Feature | Source | Purpose |
|---|---|---|
| `Year` | `Date.dt.year` | Trend baseline |
| `Month` | `Date.dt.month` | Monthly seasonality |
| `Week` | `Date.dt.isocalendar().week` | Weekly cycles |
| `Quarter` | `Date.dt.quarter` | Quarterly aggregation |
| `DayOfWeek` | `Date.dt.dayofweek` | Day-of-week effects |
| `MonthLabel` | `Date.strftime` | Display labels |

---

## 6. Forecasting Models

### 6.1 Meta Prophet

**Library**: `prophet` (open-source, developed by Meta)

**Why Prophet?**
- Handles missing data and outliers gracefully
- Decomposes trend, weekly seasonality, and yearly seasonality automatically
- Works well with datasets that have fewer than 2 years of history
- Provides uncertainty intervals (confidence bands)

**Configuration**:
```python
Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.15,
)
```

**Multiplicative seasonality** is chosen because the amplitude of seasonal fluctuations tends to grow with the trend level (common in sales data).

**Changepoint prior scale = 0.15** allows moderate flexibility to detect trend shifts without overfitting.

**Train/Test Split**: 80% training / 20% held-out test  
**Forecast Horizon**: Up to 180 days (default 90)  
**Evaluation**: MAE, RMSE, R² on the test split

### 6.2 SARIMA

**Library**: `statsmodels.tsa.statespace.SARIMAX`

**Why SARIMA?**
- Classical statistical model, highly interpretable
- Captures both short-term autocorrelation and monthly seasonality
- Complements Prophet with a different modelling paradigm
- Useful for monthly-level planning horizons

**Model Order**: SARIMA(1, 1, 1)(1, 1, 1, 12)

| Parameter | Value | Meaning |
|---|---|---|
| p=1 | AR lag 1 | Uses 1 lagged value |
| d=1 | 1st differencing | Removes simple trend |
| q=1 | MA lag 1 | Uses 1 lagged error |
| P=1 | Seasonal AR | Uses 1 seasonal lag |
| D=1 | Seasonal differencing | Removes seasonal trend |
| Q=1 | Seasonal MA | 1 seasonal error lag |
| s=12 | Period = 12 months | Annual seasonality |

**Train/Test Split**: 80% / 20%  
**Forecast Horizon**: 1–12 months (default 6)  
**Evaluation**: MAE, RMSE, R²

### 6.3 Seasonal Decomposition

Using `statsmodels.tsa.seasonal.seasonal_decompose` on the monthly series to separate:
- **Trend**: Long-term direction of sales
- **Seasonal**: Repeating patterns (monthly cycle)
- **Residual**: Irregular / noise component

---

## 7. Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** (Mean Absolute Error) | `mean(|y − ŷ|)` | Average dollar error per period. Lower is better. |
| **RMSE** (Root Mean Squared Error) | `√mean((y − ŷ)²)` | Penalises large errors more heavily than MAE. |
| **R²** (Coefficient of Determination) | `1 − SS_res/SS_tot` | Proportion of variance explained. 1.0 = perfect, 0 = baseline mean. |

Both models are evaluated on a **held-out test set** (not seen during training) to provide an unbiased estimate of forecast accuracy.

---

## 8. Dashboard Architecture

### 8.1 Technology Stack

| Component | Technology | Role |
|---|---|---|
| Frontend framework | Streamlit | Page layout, widgets, state management |
| Charting | Plotly Express + Graph Objects | Interactive charts |
| Data engine | pandas + NumPy | Data manipulation |
| Forecasting | Prophet + statsmodels | ML models |
| Styling | Custom CSS injected via `st.markdown` | Chocolate-themed UI |

### 8.2 Dashboard Tabs

| Tab | Visualisations |
|---|---|
| **Sales Trends** | Daily revenue line chart + 7-day moving average, monthly dual-axis bar+line, revenue heatmap (month × weekday) |
| **Prophet Forecast** | Forecast line with 95% confidence band, actual vs predicted, component plots (trend, weekly, yearly), MAE/RMSE/R² metrics |
| **SARIMA Forecast** | Historical + future forecast line chart, shaded forecast region, tabular forecast, seasonal decomposition subplots |
| **Product & Country** | Horizontal bar chart by product, donut pie by country, multi-line monthly trends (top 8 products + all countries), choropleth world map |
| **Salesperson** | Revenue bar chart, bubble scatter (revenue × transactions, sized by boxes), leaderboard table |

### 8.3 Sidebar Filters

All charts and models respond dynamically to:
- Country multi-select
- Product multi-select
- Salesperson multi-select
- Date range picker
- Forecast horizon sliders

---

## 9. Key Findings

1. **Top revenue product**: High-cocoa products (99% Dark & Pure, Raspberry Choco) generate the highest per-transaction revenue.
2. **Boxes ≠ Revenue**: Some low-value products (Drinking Coco, Eclairs) ship many boxes at low unit prices; high-value products ship fewer boxes at high prices. Supply chain planning should distinguish these.
3. **Salesperson variance**: There is notable variance in per-salesperson revenue — identifying top performers and best practices can lift team-wide performance.
4. **Seasonal patterns**: Prophet detects weekly cycles and an annual seasonal trend. With more years of data, the yearly component would be more pronounced.
5. **Forecast**: Both models project continued positive sales momentum. SARIMA's monthly forecast provides a stable planning baseline; Prophet's daily forecast highlights shorter-term fluctuations.

---

## 10. Limitations & Future Work

| Limitation | Suggested Improvement |
|---|---|
| Only 8 months of historical data | Collect 2–3 years for robust seasonality modelling |
| No external features (holidays, promotions, price) | Add holiday regressors to Prophet; include price as SARIMA exogenous variable |
| Single-step SARIMA horizon | Use recursive multi-step or direct strategy for longer horizons |
| No product-level forecasting model | Train per-product Prophet models for SKU-level inventory planning |
| No automated retraining | Add a scheduled pipeline (cron/Airflow) to retrain on new data |
| No CI/CD or containerisation | Wrap in Docker + deploy to Streamlit Cloud or AWS EC2 |

---

## 11. Conclusion

This project demonstrates a complete, production-quality template for retail sales forecasting. By combining:
- **Rigorous data cleaning** to handle real-world messy formats
- **Two complementary forecasting models** (Prophet for flexibility, SARIMA for interpretability)
- **An interactive Streamlit dashboard** for non-technical stakeholders

…it delivers both analytical depth and practical usability. The modular architecture (`model.py` ↔ `dashboard.py`) means the backend can be extended independently of the UI, and new models or data sources can be swapped in with minimal changes.

---

## 12. References

1. Taylor, S. J., & Letham, B. (2018). *Forecasting at scale*. The American Statistician. [Prophet paper]
2. Box, G. E. P., Jenkins, G. M., et al. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). [ARIMA/SARIMA theory]
3. McKinney, W. (2022). *Python for Data Analysis* (3rd ed.). O'Reilly. [pandas]
4. Streamlit Documentation — https://docs.streamlit.io
5. Plotly Documentation — https://plotly.com/python/
6. statsmodels Documentation — https://www.statsmodels.org
