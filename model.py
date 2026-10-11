"""
model.py — Chocolate Sales Forecasting Backend
================================================
Responsibilities:
  - Load and clean the CSV dataset
  - Engineer time-series features
  - Aggregate sales by date, product, country, salesperson
  - Train Prophet and ARIMA/SARIMA forecasting models
  - Return forecast DataFrames consumed by the Streamlit dashboard

Usage (standalone):
    python model.py
"""

import os
import re
import warnings
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not installed. Install via: pip install prophet")

# Statsmodels SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. DATA LOADING & CLEANING
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent.parent / "Chocolate Sales.csv"


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load and return the raw chocolate sales DataFrame."""
    filepath = Path(path) if path else DATA_PATH
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    log.info(f"Loaded {len(df):,} rows from {filepath.name}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame:
    - Strip & rename columns
    - Parse Amount (remove '$', ',', spaces) → float
    - Parse Date → datetime
    - Drop null rows
    """
    df = df.copy()

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Parse Amount: "$5,320 " → 5320.0
    df["Amount"] = (
        df["Amount"]
        .astype(str)
        .str.replace(r"[\$,\s]", "", regex=True)
        .str.strip()
        .replace("", np.nan)
        .astype(float)
    )

    # Parse Date: "04-Jan-22" → datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%y", errors="coerce")

    # Drop rows with missing critical fields
    df = df.dropna(subset=["Date", "Amount"])
    df = df.reset_index(drop=True)

    log.info(f"Clean shape: {df.shape} | Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based feature columns."""
    df = df.copy()
    df["Year"]        = df["Date"].dt.year
    df["Month"]       = df["Date"].dt.month
    df["Week"]        = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"]   = df["Date"].dt.dayofweek
    df["Quarter"]     = df["Date"].dt.quarter
    df["MonthLabel"]  = df["Date"].dt.strftime("%b %Y")
    return df


# ---------------------------------------------------------------------------
# 2. AGGREGATION HELPERS
# ---------------------------------------------------------------------------

def daily_total(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total Amount & Boxes by day."""
    agg = (
        df.groupby("Date", as_index=False)
          .agg(Amount=("Amount", "sum"), Boxes=("Boxes Shipped", "sum"))
          .sort_values("Date")
    )
    return agg


def monthly_total(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by calendar month."""
    df2 = df.copy()
    df2["Month_Start"] = df2["Date"].dt.to_period("M").dt.to_timestamp()
    agg = (
        df2.groupby("Month_Start", as_index=False)
           .agg(Amount=("Amount", "sum"), Boxes=("Boxes Shipped", "sum"))
           .sort_values("Month_Start")
    )
    return agg


def by_product(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Product", as_index=False).agg(
        Amount=("Amount", "sum"),
        Boxes=("Boxes Shipped", "sum"),
        Transactions=("Amount", "count")
    ).sort_values("Amount", ascending=False)


def by_country(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Country", as_index=False).agg(
        Amount=("Amount", "sum"),
        Boxes=("Boxes Shipped", "sum"),
        Transactions=("Amount", "count")
    ).sort_values("Amount", ascending=False)


def by_salesperson(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Sales Person", as_index=False).agg(
        Amount=("Amount", "sum"),
        Boxes=("Boxes Shipped", "sum"),
        Transactions=("Amount", "count")
    ).sort_values("Amount", ascending=False)


def monthly_by_product(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["Month_Start"] = df2["Date"].dt.to_period("M").dt.to_timestamp()
    agg = (
        df2.groupby(["Month_Start", "Product"], as_index=False)
           .agg(Amount=("Amount", "sum"), Boxes=("Boxes Shipped", "sum"))
           .sort_values(["Product", "Month_Start"])
    )
    return agg


def monthly_by_country(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["Month_Start"] = df2["Date"].dt.to_period("M").dt.to_timestamp()
    agg = (
        df2.groupby(["Month_Start", "Country"], as_index=False)
           .agg(Amount=("Amount", "sum"), Boxes=("Boxes Shipped", "sum"))
           .sort_values(["Country", "Month_Start"])
    )
    return agg


# ---------------------------------------------------------------------------
# 3. PROPHET FORECASTING
# ---------------------------------------------------------------------------

def prophet_forecast(
    ts: pd.DataFrame,
    periods: int = 90,
    freq: str = "D",
    extra_regressors: Optional[list] = None,
) -> dict:
    """
    Fit a Prophet model on ts (columns: ds, y) and return forecast.

    Returns:
        dict with keys: model, forecast, mae, rmse, r2, actuals
    """
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet is not installed.")

    ts = ts.copy().rename(columns={"Date": "ds", "Amount": "y"})
    ts = ts[["ds", "y"]].dropna().sort_values("ds")

    # Train / test split (last 20% as test)
    split = int(len(ts) * 0.8)
    train, test = ts.iloc[:split], ts.iloc[split:]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.15,
    )

    if extra_regressors:
        for reg in extra_regressors:
            model.add_regressor(reg)

    model.fit(train)

    # Evaluate on test
    test_future = model.make_future_dataframe(periods=len(test), freq=freq, include_history=False)
    test_future["ds"] = test["ds"].values
    test_pred = model.predict(test_future)

    y_true = test["y"].values
    y_pred = test_pred["yhat"].values
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # Full future forecast
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    log.info(f"Prophet → MAE={mae:,.2f} | RMSE={rmse:,.2f} | R²={r2:.3f}")
    return {
        "model":    model,
        "forecast": forecast,
        "mae":      mae,
        "rmse":     rmse,
        "r2":       r2,
        "actuals":  ts,
        "train":    train,
        "test":     test,
    }


# ---------------------------------------------------------------------------
# 4. SARIMA FORECASTING
# ---------------------------------------------------------------------------

def sarima_forecast(
    monthly_ts: pd.DataFrame,
    periods: int = 6,
) -> dict:
    """
    Fit a SARIMA(1,1,1)(1,1,1,12) model on monthly aggregated data.

    Returns:
        dict with keys: model_result, forecast_df, mae, rmse, r2
    """
    ts = monthly_ts.copy().set_index("Month_Start")["Amount"].asfreq("MS").fillna(0)

    split = max(int(len(ts) * 0.8), len(ts) - 3)
    train, test = ts.iloc[:split], ts.iloc[split:]

    order        = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

    try:
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)
    except Exception as e:
        log.warning(f"SARIMA fit failed ({e}). Falling back to ARIMA(1,1,1).")
        model = SARIMAX(train, order=(1, 1, 1), enforce_stationarity=False)
        result = model.fit(disp=False)

    # Evaluate on test
    test_pred = result.forecast(steps=len(test))
    y_true = test.values
    y_pred = test_pred.values
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")

    # Future forecast
    future_pred = result.forecast(steps=len(test) + periods)
    last_date = ts.index[-1]
    future_index = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=periods,
        freq="MS",
    )
    forecast_df = pd.DataFrame({"Month_Start": future_index, "Amount": future_pred.values[-periods:]})

    log.info(f"SARIMA → MAE={mae:,.2f} | RMSE={rmse:,.2f} | R²={r2:.3f}")
    return {
        "model_result": result,
        "forecast_df":  forecast_df,
        "mae":          mae,
        "rmse":         rmse,
        "r2":           r2,
        "train":        train,
        "test":         test,
        "history":      ts,
    }


# ---------------------------------------------------------------------------
# 5. SEASONAL DECOMPOSITION
# ---------------------------------------------------------------------------

def decompose_series(monthly_ts: pd.DataFrame, model: str = "multiplicative") -> object:
    """Return statsmodels DecomposeResult for the monthly series."""
    ts = monthly_ts.copy().set_index("Month_Start")["Amount"].asfreq("MS").fillna(0)
    # multiplicative requires all positive values
    if (ts <= 0).any():
        model = "additive"
    result = seasonal_decompose(ts, model=model, period=3, extrapolate_trend="freq")
    return result


# ---------------------------------------------------------------------------
# 6. CONVENIENCE PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(path: Optional[str] = None) -> dict:
    """
    Full end-to-end pipeline. Returns a dict with all DataFrames and models
    ready for the dashboard to consume.
    """
    raw  = load_data(path)
    df   = feature_engineer(clean_data(raw))

    monthly = monthly_total(df)
    daily   = daily_total(df)

    # Prophet on daily totals
    prophet_result = prophet_forecast(daily, periods=90, freq="D")

    # SARIMA on monthly totals
    sarima_result = sarima_forecast(monthly, periods=6)

    # Decomposition
    decomp = decompose_series(monthly)

    return {
        "df":             df,
        "daily":          daily,
        "monthly":        monthly,
        "by_product":     by_product(df),
        "by_country":     by_country(df),
        "by_salesperson": by_salesperson(df),
        "monthly_by_product": monthly_by_product(df),
        "monthly_by_country": monthly_by_country(df),
        "prophet":        prophet_result,
        "sarima":         sarima_result,
        "decomp":         decomp,
    }


# ---------------------------------------------------------------------------
# 7. STANDALONE ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_pipeline()
    print("\n=== Pipeline Summary ===")
    print(f"Total records   : {len(result['df']):,}")
    print(f"Total revenue   : ${result['df']['Amount'].sum():,.2f}")
    print(f"Date range      : {result['df']['Date'].min().date()} → {result['df']['Date'].max().date()}")
    print(f"Products        : {result['df']['Product'].nunique()}")
    print(f"Countries       : {result['df']['Country'].nunique()}")
    print(f"\nProphet  → MAE={result['prophet']['mae']:,.2f}  RMSE={result['prophet']['rmse']:,.2f}  R²={result['prophet']['r2']:.3f}")
    print(f"SARIMA   → MAE={result['sarima']['mae']:,.2f}  RMSE={result['sarima']['rmse']:,.2f}")
    print("\nTop 5 Products by Revenue:")
    print(result["by_product"].head(5).to_string(index=False))
