"""
Weighted Ensemble + Comparison & Visualization
================================================
Combines predictions from multiple models using RMSE-inverse weighting.
Generates the full model comparison table and fan chart.

Prerequisites:
    - Run all models first  → *_predictions.csv

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

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ensemble.log", encoding="utf-8"),
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
    matches the actual direction.
    """
    actual_dir = np.diff(y_true) > 0  # True = price went up
    pred_dir = np.diff(y_pred) > 0
    return float(np.mean(actual_dir == pred_dir) * 100)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
def create_ensemble(preds_dict, top_k=3):
    """
    RMSE-inverse weighted ensemble of the top K best models provided in preds_dict.
    Models with lower RMSE get higher weight.
    """
    # Find common indices
    common_idx = None
    for k, v in preds_dict.items():
        if common_idx is None:
            common_idx = v.index
        else:
            common_idx = common_idx.intersection(v.index)

    aligned = {k: v.loc[common_idx] for k, v in preds_dict.items()}
    # Pick y_true from the first model
    first_key = list(preds_dict.keys())[0]
    y_true = aligned[first_key]["actual"].values

    rmses = {}
    for k, v in aligned.items():
        rmses[k] = rmse(y_true, v["q50"].values)

    # Filter to top K
    sorted_models = sorted(rmses.items(), key=lambda x: x[1])[:top_k]
    best_names = [x[0] for x in sorted_models]
    logger.info(f"Filtering ensemble to top {top_k} models: {best_names}")

    aligned = {k: aligned[k] for k in best_names}
    rmses = {k: rmses[k] for k in best_names}

    inv_rmses = {k: 1 / v for k, v in rmses.items()}
    total_inv = sum(inv_rmses.values())
    weights = {k: v / total_inv for k, v in inv_rmses.items()}

    logger.info("Ensemble weights:")
    for k, w in weights.items():
        logger.info(f"  {k:<10}: {w:.3f} (RMSE: {rmses[k]:.2f})")

    # Weighted blend for each quantile
    ensemble = pd.DataFrame(index=common_idx)
    ensemble["actual"] = y_true

    for q_col in ["q10", "q50", "q90"]:
        ensemble[q_col] = 0.0
        for k, v in aligned.items():
            ensemble[q_col] += weights[k] * v[q_col].values

    # Enforce non-crossing
    ensemble["q10"] = ensemble[["q10", "q50"]].min(axis=1)
    ensemble["q90"] = ensemble[["q50", "q90"]].max(axis=1)

    return ensemble, weights


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_fan_chart(preds, title_suffix="Ensemble"):
    """Fan chart showing 10th-90th percentile confidence interval."""
    plot_end = preds.index.min() + pd.Timedelta(days=14)
    p = preds[preds.index <= plot_end]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.fill_between(p.index, p["q10"], p["q90"],
                     alpha=0.25, color="#2196F3", label="10th–90th percentile")
    ax.plot(p.index, p["q50"], color="#1565C0", linewidth=1.2, label="Median forecast (q50)")
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # sMAPE
    bars1 = ax1.bar(x, smapes, width, color="#42A5F5")
    ax1.set_ylabel("sMAPE (%)")
    ax1.set_title("sMAPE Comparison", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, smapes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", fontsize=8)

    # MAE
    bars2 = ax2.bar(x, maes, width, color="#66BB6A")
    ax2.set_ylabel("MAE (€/MWh)")
    ax2.set_title("MAE Comparison", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, maes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", fontsize=8)

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
    preds_dict = {}
    model_paths = {
        "LightGBM": "lgbm_predictions.csv",
        "XGBoost": "xgb_predictions.csv",
        "CatBoost": "cb_predictions.csv",
        "Linear OLS": "linear_predictions.csv",
        "MLP (NN)": "mlp_predictions.csv"
    }

    for name, filename in model_paths.items():
        try:
            preds_dict[name] = pd.read_csv(PROCESSED_DIR / filename, index_col="timestamp", parse_dates=True)
            logger.info(f"{name} predictions: {preds_dict[name].shape}")
        except FileNotFoundError:
            logger.warning(f"Could not find {filename}. Skipping {name}.")

    if not preds_dict:
        logger.error("No model predictions found. Exiting.")
        return

    # Create ensemble
    ensemble_preds, weights = create_ensemble(preds_dict)
    y_true = ensemble_preds["actual"].values

    # Load baseline metrics for comparison
    with open(LOG_DIR / "baseline_metrics.json", encoding="utf-8") as f:
        baseline_metrics = json.load(f)

    all_metrics = [
        {
            "model": "Naive Persistence",
            "sMAPE": baseline_metrics["models"]["naive_persistence"]["sMAPE"],
            "MAE": baseline_metrics["models"]["naive_persistence"]["MAE"],
            "DirAcc": baseline_metrics["models"]["naive_persistence"].get("DirAcc", "-"),
        },
        {
            "model": "Ridge (ARX)",
            "sMAPE": baseline_metrics["models"]["ridge_arx"]["sMAPE"],
            "MAE": baseline_metrics["models"]["ridge_arx"]["MAE"],
            "DirAcc": baseline_metrics["models"]["ridge_arx"].get("DirAcc", "-"),
        }
    ]

    # Add each constituent model to metrics list
    for name, df in preds_dict.items():
        all_metrics.append({
            "model": name,
            "sMAPE": round(smape(df["actual"].values, df["q50"].values), 2),
            "MAE": round(mae(df["actual"].values, df["q50"].values), 2),
            "DirAcc": round(directional_accuracy(df["actual"].values, df["q50"].values), 2)
        })

    # Add Ensemble
    ens_smape = round(smape(y_true, ensemble_preds["q50"].values), 2)
    ens_mae = round(mae(y_true, ensemble_preds["q50"].values), 2)
    ens_diracc = round(directional_accuracy(y_true, ensemble_preds["q50"].values), 2)

    all_metrics.append({
        "model": "Ensemble",
        "sMAPE": ens_smape,
        "MAE": ens_mae,
        "DirAcc": ens_diracc
    })

    # Detailed Ensemble metrics
    ensemble_metrics = {
        "model": "Ensemble",
        "weights": {k: round(v, 3) for k, v in weights.items()},
        "test_size": len(ensemble_preds),
        "sMAPE_median": ens_smape,
        "MAE_median": ens_mae,
        "pinball_q10": round(pinball_loss(y_true, ensemble_preds["q10"].values, 0.10), 2),
        "pinball_q50": round(pinball_loss(y_true, ensemble_preds["q50"].values, 0.50), 2),
        "pinball_q90": round(pinball_loss(y_true, ensemble_preds["q90"].values, 0.90), 2),
        "quantile_crossings": int(((ensemble_preds["q10"] > ensemble_preds["q50"]) |
                                    (ensemble_preds["q50"] > ensemble_preds["q90"])).sum()),
        "directional_accuracy": ens_diracc,
    }

    # Print comparison table
    logger.info("")
    logger.info(f"  {'Model':<20} {'sMAPE':>8} {'MAE':>10} {'Dir.Acc':>8}")
    logger.info(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*8}")
    for m in all_metrics:
        da = f"{m.get('DirAcc', '—'):>7.1f}%" if "DirAcc" in m else f"{'—':>8}"
        logger.info(f"  {m['model']:<20} {m['sMAPE']:>7.2f}% {m['MAE']:>9.2f} {da}")
    logger.info("")

    # Save outputs
    ensemble_preds.to_csv(PROCESSED_DIR / "ensemble_predictions.csv")
    full_metrics = {
        "comparison": all_metrics,
        "ensemble_detail": ensemble_metrics,
    }
    with open(LOG_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    # Plots
    plot_fan_chart(ensemble_preds)
    plot_model_comparison(all_metrics)

    logger.info("Done.")


if __name__ == "__main__":
    main()
