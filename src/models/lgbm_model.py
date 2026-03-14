"""
LightGBM Model — 60-Day Rolling Window with Quantile Regression
=================================================================
Trains LightGBM on a 60-day rolling window, producing:
  - 10th percentile (downside risk)
  - 50th percentile (median forecast)
  - 90th percentile (upside risk)

Usage:
    python src/models/lgbm_model.py

Output:
    /data/processed/lgbm_predictions.csv
    /logs/lgbm_metrics.json
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

for d in [LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "lgbm.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "actual_load_mw", "wind_forecast_mw", "solar_forecast_mw",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "price_lag_24h", "price_lag_168h", "load_lag_24h",
    "renewable_penetration", "price_rolling_mean_24h",
    "price_rolling_std_24h", "net_load_mw",
]

WINDOW_DAYS = 60      # Rolling training window
WINDOW_HOURS = WINDOW_DAYS * 24
QUANTILES = [0.10, 0.50, 0.90]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def pinball_loss(y_true, y_pred, quantile):
    delta = y_true - y_pred
    return float(np.mean(np.where(delta >= 0, quantile * delta, (quantile - 1) * delta)))


# ---------------------------------------------------------------------------
# Rolling Window Training
# ---------------------------------------------------------------------------
def rolling_window_train(df, split_date="2025-12-31"):
    """
    60-day rolling window quantile regression.
    For each day in the test set, train on the prior 60 days.
    """
    test = df[df.index > split_date].copy()
    test_dates = sorted(test.index.normalize().unique())

    logger.info(f"Test dates: {len(test_dates)} days")
    logger.info(f"Rolling window: {WINDOW_DAYS} days")

    all_preds = []

    for i, date in enumerate(test_dates):
        # Define training window: [date - 60 days, date)
        train_end = date
        train_start = train_end - pd.Timedelta(days=WINDOW_DAYS)

        train_window = df[(df.index >= train_start) & (df.index < train_end)]
        test_day = df[df.index.normalize() == date]

        if len(train_window) < WINDOW_HOURS * 0.5:  # Need at least 50% of window
            logger.warning(f"  Skip {date.date()}: insufficient training data ({len(train_window)} rows)")
            continue

        X_train = train_window[FEATURE_COLS].values
        y_train = train_window["price_eur_mwh"].values
        X_test = test_day[FEATURE_COLS].values

        day_preds = {"timestamp": test_day.index, "actual": test_day["price_eur_mwh"].values}

        for q in QUANTILES:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            day_preds[f"q{int(q*100)}"] = model.predict(X_test)

        day_df = pd.DataFrame(day_preds).set_index("timestamp")
        all_preds.append(day_df)

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"  Day {i+1}/{len(test_dates)}: {date.date()}")

    result = pd.concat(all_preds)

    # Enforce non-crossing: q10 <= q50 <= q90
    result["q10"] = result[["q10", "q50"]].min(axis=1)
    result["q90"] = result[["q50", "q90"]].max(axis=1)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("LightGBM — 60-Day Rolling Window Quantile Regression")
    logger.info("=" * 60)

    df = pd.read_csv(PROCESSED_DIR / "model_ready.csv", index_col="timestamp", parse_dates=True)
    logger.info(f"Data shape: {df.shape}")

    preds = rolling_window_train(df)

    # Metrics
    y_true = preds["actual"].values
    metrics = {
        "model": "LightGBM",
        "window_days": WINDOW_DAYS,
        "test_size": len(preds),
        "sMAPE_median": round(smape(y_true, preds["q50"].values), 2),
        "MAE_median": round(mae(y_true, preds["q50"].values), 2),
        "pinball_q10": round(pinball_loss(y_true, preds["q10"].values, 0.10), 2),
        "pinball_q50": round(pinball_loss(y_true, preds["q50"].values, 0.50), 2),
        "pinball_q90": round(pinball_loss(y_true, preds["q90"].values, 0.90), 2),
    }

    logger.info(f"LightGBM — sMAPE: {metrics['sMAPE_median']}% | MAE: {metrics['MAE_median']} €/MWh")
    logger.info(f"  Pinball: q10={metrics['pinball_q10']}, q50={metrics['pinball_q50']}, q90={metrics['pinball_q90']}")

    # Check quantile crossing
    crossings = ((preds["q10"] > preds["q50"]) | (preds["q50"] > preds["q90"])).sum()
    metrics["quantile_crossings"] = int(crossings)
    logger.info(f"  Quantile crossings: {crossings}")

    # Save
    preds.to_csv(PROCESSED_DIR / "lgbm_predictions.csv")
    with open(LOG_DIR / "lgbm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Predictions saved → {PROCESSED_DIR / 'lgbm_predictions.csv'}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
