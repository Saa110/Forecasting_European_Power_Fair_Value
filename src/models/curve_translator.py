"""
Prompt Curve Translation & Trading Signals
============================================
Converts hourly price forecasts into tradable base/peak views with
a hybrid dynamic trading strategy.

Steps:
  1. Aggregate hourly forecasts → Baseload & Peakload (Front-Week)
  2. Compare model fair value against mock traded forward prices
  3. Compute risk premium: RP = F_traded − E[Spot]
  4. Generate probabilistic trading signals with confidence layers:
     - Percentile-based: Forward vs q10/q90 bands
     - Volatility-adaptive conviction sizing
     - Contango/Backwardation classification

Usage:
    python src/models/curve_translator.py

Output:
    /logs/curve_translation.json
    /docs/figures/curve_view.png
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
        logging.FileHandler(LOG_DIR / "curve_translation.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curve Aggregation
# ---------------------------------------------------------------------------
PEAK_HOURS = range(9, 21)  # Hours 09-20 (German peak block)
PEAK_DAYS = range(0, 5)    # Mon=0 to Fri=4


def aggregate_to_curve(preds: pd.DataFrame) -> dict:
    """
    Aggregate hourly forecasts into Baseload and Peakload views.

    Baseload: arithmetic mean of ALL 24 hours
    Peakload: arithmetic mean of hours 09-20 on Mon-Fri
    """
    logger.info("Aggregating hourly forecasts to curve views ...")

    # Weekly grouping
    preds["week"] = preds.index.isocalendar().week.values
    preds["year"] = preds.index.year
    preds["hour"] = preds.index.hour
    preds["dow"] = preds.index.dayofweek

    weeks = []
    for (year, week), group in preds.groupby(["year", "week"]):
        peak_mask = (group["hour"].isin(PEAK_HOURS)) & (group["dow"].isin(PEAK_DAYS))

        base_q50 = group["q50"].mean()
        base_q10 = group["q10"].mean()
        base_q90 = group["q90"].mean()
        base_actual = group["actual"].mean()

        peak_group = group[peak_mask]
        if len(peak_group) > 0:
            peak_q50 = peak_group["q50"].mean()
            peak_q10 = peak_group["q10"].mean()
            peak_q90 = peak_group["q90"].mean()
            peak_actual = peak_group["actual"].mean()
        else:
            peak_q50 = peak_q10 = peak_q90 = peak_actual = np.nan

        weeks.append({
            "year": int(year),
            "week": int(week),
            "start_date": str(group.index.min().date()),
            "end_date": str(group.index.max().date()),
            "hours_in_week": len(group),
            "base_q50": round(base_q50, 2),
            "base_q10": round(base_q10, 2),
            "base_q90": round(base_q90, 2),
            "base_actual": round(base_actual, 2),
            "peak_q50": round(peak_q50, 2) if not np.isnan(peak_q50) else None,
            "peak_q10": round(peak_q10, 2) if not np.isnan(peak_q10) else None,
            "peak_q90": round(peak_q90, 2) if not np.isnan(peak_q90) else None,
            "peak_actual": round(peak_actual, 2) if not np.isnan(peak_actual) else None,
        })

    logger.info(f"  Aggregated into {len(weeks)} weekly curve views")
    return weeks


# ---------------------------------------------------------------------------
# Trading Signal Generation
# ---------------------------------------------------------------------------
def generate_trading_signals(weeks: list, volatility_window: int = 4) -> list:
    """
    Hybrid trading strategy:
      1. Percentile-based: Forward vs q10/q90 bands
      2. Volatility-adaptive confidence
      3. Conviction sizing

    Mock forward prices: actual + small noise (simulates market)
    """
    logger.info("Generating trading signals ...")

    # Compute rolling volatility from base actuals
    base_actuals = [w["base_actual"] for w in weeks]
    rolling_vol = pd.Series(base_actuals).rolling(volatility_window, min_periods=1).std()

    results = []
    for i, week in enumerate(weeks):
        # Mock traded forward price:
        # In production this would come from EEX/ICE
        # We simulate as actual + noise (±5%)
        np.random.seed(int(week["year"]) * 100 + int(week["week"]))
        mock_forward = week["base_actual"] * (1 + np.random.uniform(-0.05, 0.05))
        mock_forward = round(mock_forward, 2)

        fair_value = week["base_q50"]
        q10 = week["base_q10"]
        q90 = week["base_q90"]

        # Risk Premium: RP = Forward - E[Spot]
        risk_premium = round(mock_forward - fair_value, 2)

        # Contango/Backwardation
        curve_structure = "Contango" if risk_premium > 0 else "Backwardation"

        # --- Percentile-based signal ---
        if mock_forward < q10:
            raw_signal = "Strong Buy (Undervalued)"
            signal_strength = 3
        elif mock_forward > q90:
            raw_signal = "Strong Sell (Overvalued)"
            signal_strength = 3
        elif mock_forward < fair_value:
            raw_signal = "Mild Buy"
            signal_strength = 1
        elif mock_forward > fair_value:
            raw_signal = "Mild Sell"
            signal_strength = 1
        else:
            raw_signal = "Hold"
            signal_strength = 0

        # --- Volatility-adaptive confidence ---
        vol = rolling_vol.iloc[i] if i < len(rolling_vol) else rolling_vol.iloc[-1]
        if vol > 0:
            confidence_band = (q90 - q10) / fair_value if fair_value != 0 else 0
        else:
            confidence_band = 0

        if confidence_band > 0.5:
            confidence = "Low (wide interval)"
        elif confidence_band > 0.2:
            confidence = "Medium"
        else:
            confidence = "High (tight interval)"

        # --- Conviction sizing ---
        if signal_strength >= 3 and confidence != "Low (wide interval)":
            conviction = "Full position"
        elif signal_strength >= 1:
            conviction = "Half position"
        else:
            conviction = "No position"

        results.append({
            **week,
            "mock_forward_price": mock_forward,
            "risk_premium": risk_premium,
            "curve_structure": curve_structure,
            "signal": raw_signal,
            "confidence": confidence,
            "conviction": conviction,
            "volatility_4w": round(vol, 2) if not np.isnan(vol) else None,
        })

    logger.info(f"  Generated signals for {len(results)} weeks")
    return results


# ---------------------------------------------------------------------------
# Invalidation Criteria
# ---------------------------------------------------------------------------
def add_invalidation_notes(signals: list) -> list:
    """
    Add invalidation criteria — when the signal should NOT be traded.
    These are heuristic rules that a real desk would refine.
    """
    for s in signals:
        reasons = []

        # 1. Wide confidence interval → low-conviction signal
        if s["confidence"] == "Low (wide interval)":
            reasons.append("Wide prediction interval — market highly uncertain")

        # 2. Extreme actual prices → regime shift
        if s["base_actual"] is not None and (s["base_actual"] > 200 or s["base_actual"] < -10):
            reasons.append("Extreme price regime — model may be unreliable")

        # 3. Risk premium too small to cover slippage
        if abs(s["risk_premium"]) < 2.0:
            reasons.append("Risk premium < 2 €/MWh — insufficient edge after costs")

        s["invalidation_reasons"] = reasons if reasons else ["None — signal valid"]

    return signals


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_curve_view(signals: list):
    """Plot base/peak curve view with trading signals."""
    logger.info("Generating curve view plot ...")

    df = pd.DataFrame(signals)
    df["week_label"] = df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --- Baseload ---
    ax = axes[0]
    x = range(len(df))
    ax.fill_between(x, df["base_q10"], df["base_q90"],
                     alpha=0.2, color="#2196F3", label="10th–90th pct")
    ax.plot(x, df["base_q50"], color="#1565C0", linewidth=1.5, label="Model Fair Value (q50)")
    ax.plot(x, df["base_actual"], color="#333", linewidth=1, linestyle="--", label="Actual Base")
    ax.plot(x, df["mock_forward_price"], color="#FF5722", linewidth=1, marker="o",
            markersize=3, label="Forward Price")
    ax.set_ylabel("€/MWh")
    ax.set_title("Baseload Curve View — Weekly Aggregation", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(df["week_label"].iloc[::2], rotation=45, fontsize=7)

    # --- Trading signals ---
    ax2 = axes[1]
    signal_colors = {
        "Strong Buy (Undervalued)": "#4CAF50",
        "Mild Buy": "#81C784",
        "Hold": "#FFC107",
        "Mild Sell": "#FF8A65",
        "Strong Sell (Overvalued)": "#F44336",
    }
    colors = [signal_colors.get(s, "#999") for s in df["signal"]]
    ax2.bar(x, df["risk_premium"], color=colors, edgecolor="white", linewidth=0.5)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_ylabel("Risk Premium (€/MWh)")
    ax2.set_title("Risk Premium & Trading Signals", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels(df["week_label"].iloc[::2], rotation=45, fontsize=7)

    # Legend for signals
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=s) for s, c in signal_colors.items()]
    ax2.legend(handles=legend_patches, fontsize=7, loc="upper right")

    plt.tight_layout()
    filepath = FIG_DIR / "curve_view.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Prompt Curve Translation & Trading Signals")
    logger.info("=" * 60)

    # Load ensemble predictions
    preds = pd.read_csv(
        PROCESSED_DIR / "ensemble_predictions.csv",
        index_col="timestamp", parse_dates=True,
    )
    logger.info(f"Loaded ensemble predictions: {preds.shape}")

    # Step 1: Aggregate to curve
    weeks = aggregate_to_curve(preds)

    # Step 2: Generate signals
    signals = generate_trading_signals(weeks)

    # Step 3: Invalidation
    signals = add_invalidation_notes(signals)

    # Step 4: Save
    output_path = LOG_DIR / "curve_translation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(signals, f, indent=2)
    logger.info(f"Curve translation saved → {output_path}")

    # Step 5: Plot
    plot_curve_view(signals)

    # Step 6: Print summary table
    logger.info("")
    logger.info(f"  {'Week':<12} {'Base FV':>8} {'Forward':>8} {'RP':>7} {'Signal':<25} {'Conviction':<15}")
    logger.info(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*7} {'─'*25} {'─'*15}")
    for s in signals:
        wk = f"{s['year']}-W{s['week']:02d}"
        logger.info(
            f"  {wk:<12} {s['base_q50']:>7.1f} {s['mock_forward_price']:>7.1f} "
            f"{s['risk_premium']:>6.1f} {s['signal']:<25} {s['conviction']:<15}"
        )
    logger.info("")
    logger.info("Done.")


if __name__ == "__main__":
    main()
