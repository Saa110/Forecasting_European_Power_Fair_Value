"""
Baseline Modeling & Validation
================================
Establishes rigorous benchmark performance using:
  1. Naive Persistence (same-hour yesterday)
  2. Ridge Regression (ARX with lagged prices, load, wind, solar)

Chronological train/test split:
  - Train: all data up to 2025-12-31
  - Test:  2026-01-01 → latest (most recent ~2.5 months)

Metrics: sMAPE, MAE

Usage:
    python src/models/baseline.py

Output:
    /logs/baseline_metrics.json
    /docs/figures/baseline_forecast_vs_actual.png
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"
FIG_DIR = PROJECT_ROOT / "docs" / "figures"

for d in [LOG_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "baseline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    mask = denominator > 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Data Loading & Splitting
# ---------------------------------------------------------------------------
def load_and_split():
    """Load model_ready.csv and perform chronological train/test split."""
    logger.info("Loading data ...")
    df = pd.read_csv(
        PROCESSED_DIR / "model_ready.csv",
        index_col="timestamp",
        parse_dates=True,
    )
    logger.info(f"Dataset shape: {df.shape}")

    # Chronological split — train on everything up to end of 2025,
    # test on 2026 (most recent ~2.5 months of real market data)
    split_date = "2025-12-31"
    train = df[df.index <= split_date]
    test = df[df.index > split_date]

    logger.info(f"Train: {len(train)} rows ({train.index.min()} → {train.index.max()})")
    logger.info(f"Test:  {len(test)} rows ({test.index.min()} → {test.index.max()})")

    return df, train, test


# ---------------------------------------------------------------------------
# Baseline 1: Naive Persistence
# ---------------------------------------------------------------------------
def naive_persistence(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """
    Forecast = same hour yesterday.
    Uses the price_lag_24h column which is already computed.
    """
    logger.info("Running Naive Persistence baseline ...")
    # price_lag_24h is the price from exactly 24 hours ago
    return test["price_lag_24h"].copy()


# ---------------------------------------------------------------------------
# Baseline 2: Ridge Regression (ARX)
# ---------------------------------------------------------------------------
def ridge_regression(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """
    Ridge regression with exogenous features (ARX-style).
    Inputs: lagged prices, load, wind, solar, temporal features.
    """
    logger.info("Training Ridge Regression (ARX) baseline ...")

    feature_cols = [
        "actual_load_mw",
        "wind_forecast_mw",
        "solar_forecast_mw",
        "hour_of_day",
        "day_of_week",
        "month",
        "is_weekend",
        "price_lag_24h",
        "price_lag_168h",
        "load_lag_24h",
        "renewable_penetration",
        "net_load_mw",
    ]

    target_col = "price_eur_mwh"

    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test = test[feature_cols].values

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Ridge (L2 regularization handles multicollinearity)
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    # Feature importance (coefficient magnitudes)
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_,
        "abs_coef": np.abs(model.coef_),
    }).sort_values("abs_coef", ascending=False)
    logger.info(f"Top features:\n{coef_df.to_string(index=False)}")

    predictions = model.predict(X_test_scaled)
    return pd.Series(predictions, index=test.index, name="ridge_pred")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_forecast_vs_actual(test: pd.DataFrame, preds: dict):
    """Plot 1-2 weeks of forecast vs actual for visual inspection."""
    logger.info("Generating forecast vs actual plot ...")

    # Take first 2 weeks of test set
    plot_end = test.index.min() + pd.Timedelta(days=14)
    plot_data = test[test.index <= plot_end]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Actual prices
    ax.plot(plot_data.index, plot_data["price_eur_mwh"],
            color="#333333", linewidth=1.2, label="Actual", zorder=3)

    # Predictions
    colors = {"naive": "#F44336", "ridge": "#2196F3"}
    for name, pred in preds.items():
        pred_plot = pred[pred.index <= plot_end]
        ax.plot(pred_plot.index, pred_plot.values,
                color=colors.get(name, "#999"), linewidth=0.8,
                alpha=0.8, label=name.title())

    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Price (€/MWh)")
    ax.set_title("Baseline Forecasts vs Actual — First 2 Weeks of Test Set", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    filepath = FIG_DIR / "baseline_forecast_vs_actual.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Baseline Modeling & Validation")
    logger.info("=" * 60)

    df, train, test = load_and_split()
    y_true = test["price_eur_mwh"].values

    # --- Baseline 1: Naive Persistence ---
    naive_pred = naive_persistence(train, test)
    naive_smape = smape(y_true, naive_pred.values)
    naive_mae = mae(y_true, naive_pred.values)
    logger.info(f"Naive Persistence — sMAPE: {naive_smape:.2f}% | MAE: {naive_mae:.2f} €/MWh")

    # --- Baseline 2: Ridge Regression ---
    ridge_pred = ridge_regression(train, test)
    ridge_smape = smape(y_true, ridge_pred.values)
    ridge_mae = mae(y_true, ridge_pred.values)
    logger.info(f"Ridge (ARX)       — sMAPE: {ridge_smape:.2f}% | MAE: {ridge_mae:.2f} €/MWh")

    # --- Save metrics ---
    metrics = {
        "split_date": "2025-12-31",
        "test_size": len(test),
        "train_size": len(train),
        "models": {
            "naive_persistence": {
                "sMAPE": round(naive_smape, 2),
                "MAE": round(naive_mae, 2),
            },
            "ridge_arx": {
                "sMAPE": round(ridge_smape, 2),
                "MAE": round(ridge_mae, 2),
            },
        },
    }

    metrics_path = LOG_DIR / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    # --- Plot ---
    plot_forecast_vs_actual(test, {"naive": naive_pred, "ridge": ridge_pred})

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("Baseline Results:")
    logger.info(f"  {'Model':<20} {'sMAPE':>8} {'MAE':>10}")
    logger.info(f"  {'─'*20} {'─'*8} {'─'*10}")
    logger.info(f"  {'Naive Persistence':<20} {naive_smape:>7.2f}% {naive_mae:>9.2f}")
    logger.info(f"  {'Ridge (ARX)':<20} {ridge_smape:>7.2f}% {ridge_mae:>9.2f}")
    logger.info("=" * 60)

    # Save test predictions for downstream use
    test_preds = test[["price_eur_mwh"]].copy()
    test_preds["naive_pred"] = naive_pred.values
    test_preds["ridge_pred"] = ridge_pred.values
    test_preds.to_csv(PROCESSED_DIR / "baseline_predictions.csv")
    logger.info(f"Predictions saved → {PROCESSED_DIR / 'baseline_predictions.csv'}")


if __name__ == "__main__":
    main()
