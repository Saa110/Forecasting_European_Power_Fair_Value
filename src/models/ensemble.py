"""
Weighted Ensemble + Comparison & Visualization
================================================
Combines LightGBM + XGBoost predictions using RMSE-inverse weighting.
Generates the full model comparison table and fan chart.

Prerequisites:
    - Run lgbm_model.py first  → lgbm_predictions.csv
    - Run xgb_model.py first   → xgb_predictions.csv

Usage:
    python src/models/ensemble.py

Output:
    /data/processed/ensemble_predictions.csv
    /logs/model_metrics.json (full comparison)
    /docs/figures/fan_chart.png
    /docs/figures/model_comparison.png
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ensemble.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def pinball_loss(y_true, y_pred, quantile):
    delta = y_true - y_pred
    return float(np.mean(np.where(delta >= 0, quantile * delta, (quantile - 1) * delta)))

def directional_accuracy(y_true, y_pred):
    """
    Percentage of hours where predicted price direction (up/down vs previous hour)
    matches the actual direction. Critical for trading — a low-MAE forecast that
    gets directions wrong is worse than a high-MAE forecast that gets them right.
    """
    actual_dir = np.diff(y_true) > 0  # True = price went up
    pred_dir = np.diff(y_pred) > 0
    return float(np.mean(actual_dir == pred_dir) * 100)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
def create_ensemble(lgbm_preds, xgb_preds):
    """
    RMSE-inverse weighted ensemble of LightGBM and XGBoost.
    Models with lower RMSE get higher weight.
    """
    # Align on common index
    common_idx = lgbm_preds.index.intersection(xgb_preds.index)
    lgbm = lgbm_preds.loc[common_idx]
    xgb_p = xgb_preds.loc[common_idx]

    y_true = lgbm["actual"].values

    # Compute RMSE for each model's median forecast
    rmse_lgbm = rmse(y_true, lgbm["q50"].values)
    rmse_xgb = rmse(y_true, xgb_p["q50"].values)

    # Inverse RMSE weights
    w_lgbm = (1 / rmse_lgbm) / (1 / rmse_lgbm + 1 / rmse_xgb)
    w_xgb = 1 - w_lgbm

    logger.info(f"Ensemble weights — LightGBM: {w_lgbm:.3f}, XGBoost: {w_xgb:.3f}")
    logger.info(f"  (RMSE LightGBM: {rmse_lgbm:.2f}, XGBoost: {rmse_xgb:.2f})")

    # Weighted blend for each quantile
    ensemble = pd.DataFrame(index=common_idx)
    ensemble["actual"] = y_true

    for q_col in ["q10", "q50", "q90"]:
        ensemble[q_col] = w_lgbm * lgbm[q_col].values + w_xgb * xgb_p[q_col].values

    # Enforce non-crossing
    ensemble["q10"] = ensemble[["q10", "q50"]].min(axis=1)
    ensemble["q90"] = ensemble[["q50", "q90"]].max(axis=1)

    return ensemble, w_lgbm, w_xgb


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_fan_chart(preds, title_suffix="Ensemble"):
    """Fan chart showing 10th-90th percentile confidence interval."""
    # Take 2 weeks of data for readability
    plot_end = preds.index.min() + pd.Timedelta(days=14)
    p = preds[preds.index <= plot_end]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Confidence band
    ax.fill_between(p.index, p["q10"], p["q90"],
                     alpha=0.25, color="#2196F3", label="10th–90th percentile")

    # Median forecast
    ax.plot(p.index, p["q50"], color="#1565C0", linewidth=1.2, label="Median forecast (q50)")

    # Actual
    ax.plot(p.index, p["actual"], color="#333333", linewidth=1.2,
            linestyle="--", label="Actual price")

    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Price (€/MWh)")
    ax.set_title(f"{title_suffix} — Probabilistic Forecast Fan Chart", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    filepath = FIG_DIR / "fan_chart.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved fan chart → {filepath}")


def plot_model_comparison(all_metrics):
    """Bar chart comparing all models on sMAPE and MAE."""
    models = [m["model"] for m in all_metrics]
    smapes = [m["sMAPE"] for m in all_metrics]
    maes = [m["MAE"] for m in all_metrics]

    x = np.arange(len(models))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # sMAPE
    bars1 = ax1.bar(x, smapes, width, color=["#EF5350", "#EF5350", "#42A5F5", "#66BB6A", "#AB47BC"])
    ax1.set_ylabel("sMAPE (%)")
    ax1.set_title("sMAPE Comparison", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, smapes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", fontsize=9)

    # MAE
    bars2 = ax2.bar(x, maes, width, color=["#EF5350", "#EF5350", "#42A5F5", "#66BB6A", "#AB47BC"])
    ax2.set_ylabel("MAE (€/MWh)")
    ax2.set_title("MAE Comparison", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=30, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, maes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    filepath = FIG_DIR / "model_comparison.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved comparison → {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Weighted Ensemble & Full Model Comparison")
    logger.info("=" * 60)

    # Load individual model predictions
    lgbm_preds = pd.read_csv(PROCESSED_DIR / "lgbm_predictions.csv", index_col="timestamp", parse_dates=True)
    xgb_preds = pd.read_csv(PROCESSED_DIR / "xgb_predictions.csv", index_col="timestamp", parse_dates=True)

    logger.info(f"LightGBM predictions: {lgbm_preds.shape}")
    logger.info(f"XGBoost predictions:  {xgb_preds.shape}")

    # Create ensemble
    ensemble_preds, w_lgbm, w_xgb = create_ensemble(lgbm_preds, xgb_preds)

    # Load baseline metrics for comparison
    with open(LOG_DIR / "baseline_metrics.json") as f:
        baseline_metrics = json.load(f)

    # Compute all metrics
    y_true = ensemble_preds["actual"].values
    all_metrics = [
        {
            "model": "Naive Persistence",
            "sMAPE": baseline_metrics["models"]["naive_persistence"]["sMAPE"],
            "MAE": baseline_metrics["models"]["naive_persistence"]["MAE"],
        },
        {
            "model": "Ridge (ARX)",
            "sMAPE": baseline_metrics["models"]["ridge_arx"]["sMAPE"],
            "MAE": baseline_metrics["models"]["ridge_arx"]["MAE"],
        },
        {
            "model": "LightGBM",
            "sMAPE": round(smape(lgbm_preds["actual"].values, lgbm_preds["q50"].values), 2),
            "MAE": round(mae(lgbm_preds["actual"].values, lgbm_preds["q50"].values), 2),
        },
        {
            "model": "XGBoost",
            "sMAPE": round(smape(xgb_preds["actual"].values, xgb_preds["q50"].values), 2),
            "MAE": round(mae(xgb_preds["actual"].values, xgb_preds["q50"].values), 2),
        },
        {
            "model": "Ensemble",
            "sMAPE": round(smape(y_true, ensemble_preds["q50"].values), 2),
            "MAE": round(mae(y_true, ensemble_preds["q50"].values), 2),
        },
    ]

    # Add probabilistic metrics for ensemble
    ensemble_metrics = {
        "model": "Ensemble",
        "weights": {"LightGBM": round(w_lgbm, 3), "XGBoost": round(w_xgb, 3)},
        "test_size": len(ensemble_preds),
        "sMAPE_median": all_metrics[-1]["sMAPE"],
        "MAE_median": all_metrics[-1]["MAE"],
        "pinball_q10": round(pinball_loss(y_true, ensemble_preds["q10"].values, 0.10), 2),
        "pinball_q50": round(pinball_loss(y_true, ensemble_preds["q50"].values, 0.50), 2),
        "pinball_q90": round(pinball_loss(y_true, ensemble_preds["q90"].values, 0.90), 2),
        "quantile_crossings": int(((ensemble_preds["q10"] > ensemble_preds["q50"]) |
                                    (ensemble_preds["q50"] > ensemble_preds["q90"])).sum()),
        "directional_accuracy": round(directional_accuracy(y_true, ensemble_preds["q50"].values), 2),
    }

    # Compute directional accuracy for all models with q50 predictions
    da_lgbm = round(directional_accuracy(lgbm_preds["actual"].values, lgbm_preds["q50"].values), 2)
    da_xgb = round(directional_accuracy(xgb_preds["actual"].values, xgb_preds["q50"].values), 2)
    da_ens = ensemble_metrics["directional_accuracy"]

    # Print comparison table
    logger.info("")
    logger.info(f"  {'Model':<20} {'sMAPE':>8} {'MAE':>10} {'Dir.Acc':>8}")
    logger.info(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*8}")
    for m in all_metrics:
        da = ""
        if m["model"] == "LightGBM":
            da = f"{da_lgbm:>7.1f}%"
        elif m["model"] == "XGBoost":
            da = f"{da_xgb:>7.1f}%"
        elif m["model"] == "Ensemble":
            da = f"{da_ens:>7.1f}%"
        else:
            da = f"{'—':>8}"
        logger.info(f"  {m['model']:<20} {m['sMAPE']:>7.2f}% {m['MAE']:>9.2f} {da}")
    logger.info("")

    # Save
    ensemble_preds.to_csv(PROCESSED_DIR / "ensemble_predictions.csv")
    full_metrics = {
        "comparison": all_metrics,
        "ensemble_detail": ensemble_metrics,
    }
    with open(LOG_DIR / "model_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)

    # Plots
    plot_fan_chart(ensemble_preds)
    plot_model_comparison(all_metrics)

    logger.info("Done.")


if __name__ == "__main__":
    main()
