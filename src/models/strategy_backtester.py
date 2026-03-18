"""
Strategy Backtester
====================
Vectorized historical simulation that evaluates whether the ensemble
model's day-ahead price forecasts translate into real profitability
after transaction costs.

Position sizing is capital-aware: each hour, the strategy allocates the
full portfolio to buy/sell as many MWh as the capital can afford at the
current price.  PnL is computed as a *percentage return* on capital
actually at risk — not a raw €/MWh sum.

Caveats (logged and saved to output):
  - Electricity prices are highly autocorrelated (~0.92 lag-1), which
    inflates apparent directional accuracy.
  - No bid-ask spread or slippage modelling beyond the flat fee.
  - 77-day single-market backtest is statistically thin.
  - Walk-forward model, but hyperparameters may be overfit to window.

Usage:
    python src/models/strategy_backtester.py

Output:
    /logs/backtest_results.json
    /docs/figures/equity_curve.png
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
        logging.FileHandler(LOG_DIR / "backtest.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
THRESHOLD = 2.0         # €/MWh — minimum predicted edge to trigger a trade
TRANSACTION_FEE = 0.50  # €/MWh — charged on each signal change
INITIAL_CAPITAL = 10_000.0  # € — real starting capital
RISK_FRACTION = 0.10    # Risk 10% of equity per trade (fractional Kelly)

CAVEATS = [
    "Lag-1 price autocorrelation ~0.92 — directional accuracy is partially "
    "an artifact of price persistence, not pure model alpha.",
    "No bid-ask spread / slippage beyond the flat €0.50/MWh fee.",
    "Single market (DE), 77-day window — results are not statistically robust.",
    "Walk-forward predictions, but hyperparameters may be overfit to this period.",
    "Sharpe Ratio computed from daily PnL × √252 (industry convention).",
]


# ---------------------------------------------------------------------------
# Signal Generation (vectorized)
# ---------------------------------------------------------------------------
def generate_signals(preds: pd.DataFrame) -> pd.Series:
    """
    Compare the model's median forecast (q50) against the previous
    hour's actual price.  At hour t we know actual[t−1] but NOT
    actual[t], so this is free of look-ahead bias.

      +1 (Long)  if  q50[t] > actual[t−1] + threshold
      -1 (Short) if  q50[t] < actual[t−1] − threshold
       0 (Flat)  otherwise
    """
    prev_actual = preds["actual"].shift(1)
    signal = pd.Series(0, index=preds.index, dtype=int)

    signal[preds["q50"] > prev_actual + THRESHOLD] = 1
    signal[preds["q50"] < prev_actual - THRESHOLD] = -1

    # First row has no previous actual → stay flat
    signal.iloc[0] = 0
    return signal


# ---------------------------------------------------------------------------
# PnL Calculation — Capital-Aware
# ---------------------------------------------------------------------------
def compute_pnl(preds: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    """
    Capital-aware PnL engine.

    At each hour the strategy risks the full current equity:
      - position_mwh = equity / price  (how many MWh we can afford)
      - pnl_eur      = position_mwh × price_change − cost

    This produces a genuine *percentage return on capital*.
    """
    n = len(preds)
    equity = np.full(n, np.nan)
    equity[0] = INITIAL_CAPITAL

    net_pnl = np.zeros(n)
    raw_pnl = np.zeros(n)
    costs = np.zeros(n)
    position_mwh = np.zeros(n)

    for t in range(1, n):
        prev_signal = signal.iloc[t - 1]
        price_now = preds["actual"].iloc[t]
        price_prev = preds["actual"].iloc[t - 1]

        if np.isnan(price_prev) or np.isnan(price_now):
            # Skip if data is missing
            equity[t] = equity[t - 1]
            continue

        # 1. Prevent division by near-zero or negative prices
        safe_price_prev = max(price_prev, 10.0) 
        
        # 2. Calculate theoretical position based on equity
        theoretical_pos = (equity[t - 1] * RISK_FRACTION) / safe_price_prev
        
        # 3. Apply a hard Liquidity Limit (e.g., max 10 MWh per hour)
        MAX_POSITION_MWH = 10.0 
        
        pos = min(theoretical_pos, MAX_POSITION_MWH) * abs(prev_signal)
        position_mwh[t] = pos

        # Raw PnL in € = position × price change × direction
        price_change = price_now - price_prev
        raw_pnl[t] = prev_signal * pos * price_change

        # Transaction cost: fee × MWh traded, only on signal flips
        if signal.iloc[t] != signal.iloc[t - 1]:
            costs[t] = TRANSACTION_FEE * pos
        else:
            costs[t] = 0.0

        net_pnl[t] = raw_pnl[t] - costs[t]
        equity[t] = equity[t - 1] + net_pnl[t]

        # Floor equity at 0 (total wipeout)
        if equity[t] < 0:
            equity[t] = 0.0

    results = pd.DataFrame({
        "actual": preds["actual"].values,
        "q50": preds["q50"].values,
        "signal": signal.values,
        "position_mwh": position_mwh,
        "raw_pnl": raw_pnl,
        "cost": costs,
        "net_pnl": net_pnl,
        "equity": equity,
    }, index=preds.index)

    results["cumulative_pnl"] = results["equity"] - INITIAL_CAPITAL

    return results


# ---------------------------------------------------------------------------
# Quant Metrics
# ---------------------------------------------------------------------------
def compute_metrics(results: pd.DataFrame) -> dict:
    """Honest strategy performance metrics."""

    net = results["net_pnl"].iloc[1:]  # skip first NaN row
    traded = net[results["signal"].shift(1) != 0]

    # Total PnL
    final_equity = float(results["equity"].iloc[-1])
    total_pnl = round(final_equity - INITIAL_CAPITAL, 2)
    return_pct = round(total_pnl / INITIAL_CAPITAL * 100, 2)

    # Hit Rate — % of active trading periods with positive PnL
    if len(traded) > 0:
        hit_rate = round(float((traded > 0).mean() * 100), 2)
    else:
        hit_rate = 0.0

    # Max Drawdown — largest peak-to-trough decline in equity
    equity = results["equity"].dropna()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_drawdown_eur = round(float(drawdown.min()), 2)
    # As percentage of peak
    dd_pct = drawdown / running_max
    max_drawdown_pct = round(float(dd_pct.min()) * 100, 2)

    # Sharpe Ratio — DAILY aggregation × √252 (industry standard)
    daily_pnl = results["net_pnl"].resample("D").sum()
    daily_pnl = daily_pnl[daily_pnl.index >= results.index[1]]  # skip partial
    if daily_pnl.std() > 0:
        sharpe = round(float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)), 2)
    else:
        sharpe = 0.0

    # Number of trades (signal flips)
    num_trades = int(results["signal"].diff().abs().gt(0).sum())

    # Total transaction costs
    total_costs = round(float(results["cost"].sum()), 2)

    # Average position size
    active_pos = results["position_mwh"][results["position_mwh"] > 0]
    avg_position = round(float(active_pos.mean()), 1) if len(active_pos) > 0 else 0.0

    # Price autocorrelation (for caveat context)
    autocorr_1 = round(float(results["actual"].autocorr(lag=1)), 3)

    return {
        "initial_capital_eur": INITIAL_CAPITAL,
        "final_equity_eur": round(final_equity, 2),
        "total_pnl_eur": total_pnl,
        "return_pct": return_pct,
        "hit_rate_pct": hit_rate,
        "max_drawdown_eur": max_drawdown_eur,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio_daily": sharpe,
        "num_trades": num_trades,
        "total_transaction_costs_eur": total_costs,
        "avg_position_mwh": avg_position,
        "threshold_eur_mwh": THRESHOLD,
        "transaction_fee_eur_mwh": TRANSACTION_FEE,
        "data_points": len(results),
        "test_days": int((results.index[-1] - results.index[0]).days),
        "price_autocorr_lag1": autocorr_1,
        "caveats": CAVEATS,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_equity_curve(results: pd.DataFrame, metrics: dict):
    """Equity line + drawdown shading."""
    logger.info("Generating equity curve plot ...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1, 1]})

    # ── Panel 1: Equity curve ──
    ax = axes[0]
    ax.plot(results.index, results["equity"], color="#1565C0", linewidth=1.2,
            label=f"Portfolio Equity (Start: €{INITIAL_CAPITAL:,.0f})")
    ax.axhline(y=INITIAL_CAPITAL, color="#999", linewidth=0.7, linestyle="--",
               label="Starting Capital")

    # Shade drawdown
    running_max = results["equity"].cummax()
    ax.fill_between(results.index, results["equity"], running_max,
                    where=results["equity"] < running_max,
                    alpha=0.15, color="#F44336", label="Drawdown")

    ax.set_ylabel("Equity (€)")
    ax.set_title("Strategy Backtester — Capital-Aware Backtest",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # ── Panel 2: Daily PnL ──
    ax2 = axes[1]
    daily_pnl = results["net_pnl"].resample("D").sum()
    colors = np.where(daily_pnl >= 0, "#4CAF50", "#F44336")
    ax2.bar(daily_pnl.index, daily_pnl.values, color=colors, width=0.8)
    ax2.set_ylabel("Daily PnL (€)")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # ── Panel 3: Signal ──
    ax3 = axes[2]
    signal_colors = {1: "#4CAF50", 0: "#FFC107", -1: "#F44336"}
    sc = [signal_colors.get(int(s), "#999") for s in results["signal"]]
    ax3.scatter(results.index, results["signal"], c=sc, s=2, alpha=0.7)
    ax3.set_ylabel("Signal")
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(["Short", "Flat", "Long"])
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Annotate key metrics on the chart
    textstr = (
        f"Return: {metrics['return_pct']:+.1f}%  |  "
        f"PnL: €{metrics['total_pnl_eur']:,.2f}\n"
        f"Hit Rate: {metrics['hit_rate_pct']:.1f}%  |  "
        f"Max DD: {metrics['max_drawdown_pct']:.1f}%\n"
        f"Sharpe (daily, ×√252): {metrics['sharpe_ratio_daily']:.2f}  |  "
        f"Trades: {metrics['num_trades']}\n"
        f"⚠ Price autocorr: {metrics['price_autocorr_lag1']:.2f} — "
        f"see caveats"
    )
    axes[0].annotate(
        textstr, xy=(0.98, 0.02), xycoords="axes fraction",
        fontsize=8, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#ccc", alpha=0.9),
    )

    filepath = FIG_DIR / "equity_curve.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Strategy Backtester (Capital-Aware)")
    logger.info("=" * 60)

    # Load ensemble predictions
    preds = pd.read_csv(
        PROCESSED_DIR / "ensemble_predictions.csv",
        index_col="timestamp", parse_dates=True,
    )
    logger.info(f"Loaded ensemble predictions: {preds.shape}")
    logger.info(f"  Period: {preds.index.min()} → {preds.index.max()}")
    logger.info(f"  Threshold: {THRESHOLD} €/MWh | Fee: {TRANSACTION_FEE} €/MWh")

    # Step 1: Generate signals
    signal = generate_signals(preds)
    long_pct = float((signal == 1).mean() * 100)
    short_pct = float((signal == -1).mean() * 100)
    flat_pct = float((signal == 0).mean() * 100)
    logger.info(f"  Signals — Long: {long_pct:.1f}% | Short: {short_pct:.1f}%"
                f" | Flat: {flat_pct:.1f}%")

    # Step 2: Compute PnL (capital-aware)
    results = compute_pnl(preds, signal)

    # Step 3: Compute metrics
    metrics = compute_metrics(results)

    # Step 4: Save results
    output_path = LOG_DIR / "backtest_results.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Backtest results saved → {output_path}")

    # Step 5: Plot
    plot_equity_curve(results, metrics)

    # Step 6: Print summary
    logger.info("")
    logger.info("  ┌─────────────────────────────────────────────┐")
    logger.info("  │         BACKTEST RESULTS SUMMARY             │")
    logger.info("  ├─────────────────────────────────────────────┤")
    logger.info(f"  │  Initial Capital:   €{metrics['initial_capital_eur']:>10,.2f}       │")
    logger.info(f"  │  Final Equity:      €{metrics['final_equity_eur']:>10,.2f}       │")
    logger.info(f"  │  Total PnL:         €{metrics['total_pnl_eur']:>10,.2f}       │")
    logger.info(f"  │  Return:             {metrics['return_pct']:>9.2f}%       │")
    logger.info(f"  │  Hit Rate:           {metrics['hit_rate_pct']:>9.2f}%       │")
    logger.info(f"  │  Max Drawdown:       {metrics['max_drawdown_pct']:>9.2f}%       │")
    logger.info(f"  │  Sharpe (daily×√252):{metrics['sharpe_ratio_daily']:>10.2f}       │")
    logger.info(f"  │  Trades:             {metrics['num_trades']:>10d}       │")
    logger.info(f"  │  Txn Costs:         €{metrics['total_transaction_costs_eur']:>10,.2f}       │")
    logger.info(f"  │  Avg Position:       {metrics['avg_position_mwh']:>8.1f} MWh       │")
    logger.info("  ├─────────────────────────────────────────────┤")
    logger.info("  │  ⚠  CAVEATS                                 │")
    logger.info(f"  │  Price autocorr (lag-1): {metrics['price_autocorr_lag1']:.3f}           │")
    logger.info("  │  See backtest_results.json for full list     │")
    logger.info("  └─────────────────────────────────────────────┘")
    logger.info("")
    logger.info("Done.")


if __name__ == "__main__":
    main()
