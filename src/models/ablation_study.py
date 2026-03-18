"""
Feature Ablation Study — 60-Day Rolling Window
=================================================
Quantifies the impact of progressively adding fundamental features 
to the LightGBM model. 

Feature sets:
1. Base (Time + Price Autoregression)
2. Base + Demand
3. Base + Supply
4. Full (Base + Demand + Supply)

Usage:
    python src/models/ablation_study.py

Output:
    /data/processed/ablation_metrics.json
    /docs/figures/ablation_chart.png
"""

import sys
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"
FIG_DIR = PROJECT_ROOT / "docs" / "figures"

for d in [LOG_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "ablation.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

WINDOW_DAYS = 60

# ---------------------------------------------------------------------------
# Feature Sets
# ---------------------------------------------------------------------------
FEATURE_SETS = {
    "1_Base": [
        "hour_of_day", "day_of_week", "month", "is_weekend",
        "price_lag_24h", "price_lag_168h", 
        "price_rolling_mean_24h", "price_rolling_std_24h"
    ],
    "2_Base_Plus_Demand": [
        "hour_of_day", "day_of_week", "month", "is_weekend",
        "price_lag_24h", "price_lag_168h", 
        "price_rolling_mean_24h", "price_rolling_std_24h",
        "actual_load_mw", "load_lag_24h", "net_load_mw"
    ],
    "3_Base_Plus_Supply": [
        "hour_of_day", "day_of_week", "month", "is_weekend",
        "price_lag_24h", "price_lag_168h", 
        "price_rolling_mean_24h", "price_rolling_std_24h",
        "wind_forecast_mw", "solar_forecast_mw", "renewable_penetration"
    ],
    "4_Full": [
        "actual_load_mw", "wind_forecast_mw", "solar_forecast_mw",
        "hour_of_day", "day_of_week", "month", "is_weekend",
        "price_lag_24h", "price_lag_168h", "load_lag_24h",
        "renewable_penetration", "price_rolling_mean_24h",
        "price_rolling_std_24h", "net_load_mw",
    ]
}

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------
def run_ablation(df, split_date="2025-12-31"):
    test = df[df.index > split_date].copy()
    test_dates = sorted(test.index.normalize().unique())
    n_days = len(test_dates)

    logger.info(f"Running Ablation on {n_days} test days")

    # Get tuned LGBM params if available
    params = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "random_state": 42, "n_jobs": -1, "verbosity": -1}
    try:
        with open(PROCESSED_DIR / "best_params.json", encoding="utf-8") as f:
            bp = json.load(f).get("lgbm", {})
            if bp:
                params.update(bp)
                logger.info("Loaded tuned LGBM params.")
    except FileNotFoundError:
        pass

    results = {}

    for set_name, features in FEATURE_SETS.items():
        logger.info(f"Evaluating: {set_name} ({len(features)} features)")
        t_start = time.time()
        
        preds_list = []
        actuals_list = []
        
        for i, date in enumerate(test_dates):
            train_start = date - pd.Timedelta(days=WINDOW_DAYS)
            train_window = df[(df.index >= train_start) & (df.index < date)]
            test_day = df[df.index.normalize() == date]

            if len(train_window) < WINDOW_DAYS * 12:
                continue

            X_train = train_window[features]
            y_train = train_window["price_eur_mwh"].values
            X_test = test_day[features]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            
            p = model.predict(X_test)
            preds_list.extend(p)
            actuals_list.extend(test_day["price_eur_mwh"].values)
            
            if (i + 1) % 10 == 0 or i == 0 or i == n_days - 1:
                elapsed = time.time() - t_start
                logger.info(f"    [{100 * (i + 1) / n_days:5.1f}%] Fold {i+1}/{n_days} | ETA: {elapsed / (i + 1) * (n_days - i - 1):.0f}s")
            
        y_true = np.array(actuals_list)
        y_pred = np.array(preds_list)
        
        s = smape(y_true, y_pred)
        m = mae(y_true, y_pred)
        
        results[set_name] = {
            "sMAPE": round(s, 2),
            "MAE": round(m, 2),
            "Time": round(time.time() - t_start, 1)
        }
        logger.info(f"  Result -> sMAPE: {s:.2f}%, MAE: {m:.2f} | Fold Time: {time.time()-t_start:.1f}s")

    return results

def plot_ablation(results):
    logger.info("Plotting ablation chart...")
    names = [k.replace("_", " ")[2:] for k in results.keys()]
    smape_vals = [v["sMAPE"] for v in results.values()]
    mae_vals = [v["MAE"] for v in results.values()]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_ylabel('sMAPE (%)', color=color1)
    ln1 = ax1.plot(names, smape_vals, marker='o', color=color1, linewidth=2, label="sMAPE")
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('MAE (€/MWh)', color=color2)
    ln2 = ax2.plot(names, mae_vals, marker='x', color=color2, linewidth=2, linestyle='--', label="MAE")
    ax2.tick_params(axis='y', labelcolor=color2)

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    plt.title("Ablation Study: Feature Importance Timeline", fontweight="bold")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = FIG_DIR / "ablation_chart.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved chart to {out_path}")


def main():
    logger.info("=" * 60)
    logger.info("Feature Ablation Study (LightGBM)")
    logger.info("=" * 60)

    df = pd.read_csv(PROCESSED_DIR / "model_ready.csv", index_col="timestamp", parse_dates=True)
    
    results = run_ablation(df)
    
    out_path = PROCESSED_DIR / "ablation_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    plot_ablation(results)
    logger.info("Ablation Study Complete.")

if __name__ == "__main__":
    main()
