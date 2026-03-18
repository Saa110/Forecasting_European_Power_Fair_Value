"""
Linear Regression Model — 60-Day Rolling Window
=================================================
Trains OLS on a 60-day rolling window. 
Synthetic quantiles (q10, q90) generated via empirical residual standard deviation.

Usage:
    python src/models/linear_model.py
"""

import sys
import json
import logging
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "linear.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "actual_load_mw", "wind_forecast_mw", "solar_forecast_mw",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "price_lag_24h", "price_lag_168h", "load_lag_24h",
    "renewable_penetration", "price_rolling_mean_24h",
    "price_rolling_std_24h", "net_load_mw",
]

WINDOW_DAYS = 60

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rolling_window_train(df, split_date="2025-12-31"):
    test = df[df.index > split_date].copy()
    test_dates = sorted(test.index.normalize().unique())
    n_days = len(test_dates)
    logger.info(f"Test dates: {n_days} days")

    all_preds = []
    t_start = time.time()

    for i, date in enumerate(test_dates):
        train_start = date - pd.Timedelta(days=WINDOW_DAYS)
        train_window = df[(df.index >= train_start) & (df.index < date)]
        test_day = df[df.index.normalize() == date]

        if len(train_window) < WINDOW_DAYS * 12:
            continue

        X_train = train_window[FEATURE_COLS].fillna(0)
        y_train = train_window["price_eur_mwh"].values
        X_test = test_day[FEATURE_COLS].fillna(0)

        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        preds = model.predict(X_test)
        
        # Empirical residual quantiles from training to simulate q10/q90 intervals
        train_preds = model.predict(X_train)
        residuals = y_train - train_preds
        r_q10 = np.percentile(residuals, 10)
        r_q90 = np.percentile(residuals, 90)
        
        day_preds = {"timestamp": test_day.index, "actual": test_day["price_eur_mwh"].values}
        day_preds["q50"] = preds
        day_preds["q10"] = preds + r_q10
        day_preds["q90"] = preds + r_q90

        all_preds.append(pd.DataFrame(day_preds).set_index("timestamp"))

        if (i + 1) % 5 == 0 or i == 0 or i == n_days - 1:
            elapsed = time.time() - t_start
            logger.info(f"  [{100 * (i + 1) / n_days:5.1f}%] Day {i+1}/{n_days} | ETA: {elapsed / (i + 1) * (n_days - i - 1):.0f}s")

    result = pd.concat(all_preds)
    result["q10"] = result[["q10", "q50"]].min(axis=1)
    result["q90"] = result[["q50", "q90"]].max(axis=1)
    return result

def main():
    logger.info("=" * 60)
    logger.info("Linear Regression (OLS) — 60-Day Rolling Window")
    logger.info("=" * 60)

    df = pd.read_csv(PROCESSED_DIR / "model_ready.csv", index_col="timestamp", parse_dates=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + ["price_eur_mwh"])
    
    preds = rolling_window_train(df)

    y_true = preds["actual"].values
    y_pred = preds["q50"].values

    metrics = {
        "model": "Linear OLS",
        "sMAPE_median": round(smape(y_true, y_pred), 2),
        "MAE_median": round(mae(y_true, y_pred), 2),
    }

    logger.info(f"Linear OLS — sMAPE: {metrics['sMAPE_median']}% | MAE: {metrics['MAE_median']} €/MWh")
    preds.to_csv(PROCESSED_DIR / "linear_predictions.csv")
    with open(LOG_DIR / "linear_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
