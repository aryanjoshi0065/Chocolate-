# 🍫 Chocolate Sales Forecasting

An end-to-end **sales forecasting** project built with Python, using
**Meta Prophet** and **SARIMA** for time-series modelling and
**Streamlit + Plotly** for the interactive frontend dashboard.

---

## 📁 Project Structure

```
archive/
├── Chocolate Sales.csv            ← Original dataset (required here)
└── chocolate_sales_forecast/
    ├── dashboard.py               ← Streamlit frontend (run this)
    ├── model.py                   ← Backend: data cleaning + ML models
    ├── requirements.txt           ← Python dependencies
    ├── README.md                  ← This file
    └── report.md                  ← Project report
```

---

## 📦 Dataset

| Column | Description |
|---|---|
| `Sales Person` | Name of the sales representative |
| `Country` | Country of the sale (UK, USA, India, Australia, Canada, New Zealand) |
| `Product` | Chocolate product name (22 unique products) |
| `Date` | Transaction date (format: `DD-Mon-YY`) |
| `Amount` | Revenue from the transaction (USD, e.g. `$5,320`) |
| `Boxes Shipped` | Number of boxes shipped in the transaction |

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10 or 3.11 recommended
- pip package manager

### 2. Install dependencies

```bash
cd chocolate_sales_forecast
pip install -r requirements.txt
```

> **Note on Prophet**: Prophet requires a C++ compiler.  
> On Windows: install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first.  
> Alternatively install via conda: `conda install -c conda-forge prophet`

### 3. Run the dashboard

```bash
streamlit run dashboard.py
```

The app will open automatically at `http://localhost:8501`.

### 4. Run the backend standalone (optional)

```bash
python model.py
```

This prints a pipeline summary to the console.

---

## 🔮 Forecasting Models

### Prophet (Daily Revenue Forecast)
- **Library**: `prophet` (Meta / Facebook)
- **Target**: Daily total revenue
- **Config**: Multiplicative seasonality, weekly + yearly components, changepoint prior scale 0.15
- **Horizon**: Configurable 30–180 days (default 90 days)
- **Train/Test split**: 80% train / 20% test

### SARIMA (Monthly Revenue Forecast)
- **Library**: `statsmodels.tsa.statespace.SARIMAX`
- **Order**: SARIMA(1,1,1)(1,1,1,12)
- **Target**: Monthly aggregated revenue
- **Horizon**: Configurable 1–12 months (default 6 months)
- **Train/Test split**: 80% / 20%

### Seasonal Decomposition
- **Library**: `statsmodels.tsa.seasonal.seasonal_decompose`
- Decomposes monthly revenue into **Trend**, **Seasonal**, and **Residual** components

---

## 📊 Dashboard Features

| Tab | Content |
|---|---|
| 📈 Sales Trends | Daily revenue line + 7-day MA, monthly bar chart, day-of-week heatmap |
| 🔮 Prophet Forecast | Forecast chart with confidence bands, component plots, MAE/RMSE/R² metrics |
| 📊 SARIMA Forecast | Monthly forecast chart, forecast table, seasonal decomposition |
| 🏷️ Product & Country | Bar charts by product, pie chart by country, monthly trend lines, choropleth map |
| 👤 Salesperson | Revenue leaderboard, scatter bubble chart, performance table |

### Sidebar Filters
- **Country** — multi-select
- **Product** — multi-select
- **Sales Person** — multi-select
- **Date Range** — date picker
- **Forecast Horizon** — slider (30–180 days for Prophet, 1–12 months for SARIMA)
- **Seasonal Decomposition** — toggle

---

## 📐 Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| MAE | `mean(|y - ŷ|)` | Average absolute error in $ |
| RMSE | `√mean((y - ŷ)²)` | Penalises large errors more |
| R² | `1 - SS_res/SS_tot` | Proportion of variance explained (1.0 = perfect) |

---

## 🛠️ Technical Stack

| Layer | Technology |
|---|---|
| Data processing | pandas, NumPy |
| Forecasting | Prophet, statsmodels (SARIMAX) |
| ML utilities | scikit-learn (metrics) |
| Frontend | Streamlit |
| Visualisation | Plotly Express + Graph Objects |
| File format | CSV (UTF-8 with BOM) |

---

## ⚠️ Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: prophet` | `pip install prophet` or `conda install -c conda-forge prophet` |
| `FileNotFoundError: Chocolate Sales.csv` | Make sure the CSV is in the **parent** folder of `chocolate_sales_forecast/` |
| Prophet install fails on Windows | Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| SARIMA R² shows N/A | Too few test points (filter selects very little data) — widen the date range |
| Decomposition unavailable | Requires at least 2 full seasonal periods; reduce filters to get more data |

---

## 📄 License

This project is for educational and demonstration purposes.
Dataset source: Sample chocolate sales data.
