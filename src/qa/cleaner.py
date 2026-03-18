"""
Quality Assurance & Feature Engineering Pipeline
==================================================
Processes raw ENTSO-E / Energy-Charts data into a clean, model-ready dataset.

Steps:
  1. Load raw CSVs from /data/raw/
  2. UTC timezone standardization (handles DST safely)
  3. Resample 15-min load/gen data to hourly (mean aggregation)
  4. Merge all domains on a unified hourly index
  5. Handle missing values (rolling-median / forward-fill)
  6. Outlier flagging (3σ / IQR) — NO deletion of extreme prices
  7. Boundary checks (reject physically impossible values)
  8. Feature engineering (temporal, lags, renewable penetration)
  9. Save model_ready.csv + QA report

Usage:
    python src/qa/cleaner.py

Output:
    /data/processed/model_ready.csv
    /logs/qa_report.txt
    /docs/figures/price_timeseries.png
"""

import os
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"
FIG_DIR = PROJECT_ROOT / "docs" / "figures"

for d in [PROCESSED_DIR, LOG_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "qa_pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QA Report Collector
# ---------------------------------------------------------------------------
class QAReport:
    """Collects QA statistics and writes a summary report."""

    def __init__(self):
        self.entries = []

    def add(self, section: str, message: str):
        self.entries.append(f"[{section}] {message}")
        logger.info(f"  QA: {message}")

    def save(self, path: Path):
        with open(path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("QUALITY ASSURANCE REPORT\n")
            f.write("=" * 60 + "\n\n")
            for entry in self.entries:
                f.write(entry + "\n")
        logger.info(f"QA report saved → {path}")


# ---------------------------------------------------------------------------
# Step 1: Load Raw Data
# ---------------------------------------------------------------------------
def load_raw_data() -> dict:
    """Load all raw CSVs into DataFrames.

    Handles ENTSO-E's mixed UTC-offset timestamps (CET +01:00 / CEST +02:00)
    by forcing conversion to UTC on load via pd.to_datetime(..., utc=True).
    """
    logger.info("Step 1: Loading raw data ...")

    files = {
        "prices": "day_ahead_prices.csv",
        "load": "actual_total_load.csv",
        "wind_solar": "wind_solar_forecast.csv",
    }

    data = {}
    for key, filename in files.items():
        filepath = RAW_DIR / filename
        df = pd.read_csv(filepath)
        # Force UTC conversion — handles mixed offsets from ENTSO-E
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        logger.info(f"  Loaded {key}: {df.shape} | index tz: {df.index.tz}")
        data[key] = df

    return data


# ---------------------------------------------------------------------------
# Step 2: UTC Standardization
# ---------------------------------------------------------------------------
def standardize_utc(df: pd.DataFrame, name: str, report: QAReport) -> pd.DataFrame:
    """Ensure index is timezone-aware UTC. Handles DST safely."""
    # If index isn't a DatetimeIndex yet (e.g. mixed-offset strings),
    # force-convert via pd.to_datetime with utc=True
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
        report.add("UTC", f"{name}: converted non-datetime index to UTC DatetimeIndex")
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
        report.add("UTC", f"{name}: localized naive timestamps to UTC")
    elif str(df.index.tz) != "UTC":
        df.index = df.index.tz_convert("UTC")
        report.add("UTC", f"{name}: converted {df.index.tz} → UTC")
    else:
        report.add("UTC", f"{name}: already in UTC ✓")

    return df


# ---------------------------------------------------------------------------
# Step 2b: DST / Missing-Hour Anomaly Detection
# ---------------------------------------------------------------------------
def detect_dst_anomalies(df: pd.DataFrame, name: str, report: QAReport) -> pd.DataFrame:
    """
    Automated anomaly detection for European DST transitions.

    CET/CEST rules (DE bidding zone):
      - Spring forward (last Sunday of March): 02:00->03:00 CET, lose 1h -> 23-hour day
      - Fall back (last Sunday of October): 03:00->02:00 CEST, gain 1h -> 25-hour day

    Also detects: gaps > 1 hour, duplicate timestamps.
    """
    # --- Check for duplicate timestamps ---
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        report.add("DST/Anomaly", f"{name}: WARNING {n_dupes} duplicate timestamps — keeping first")
        df = df[~df.index.duplicated(keep="first")]
    else:
        report.add("DST/Anomaly", f"{name}: No duplicate timestamps ✓")

    # --- Check for gaps > expected interval ---
    if len(df) > 1:
        time_diffs = pd.Series(df.index).diff().dropna()
        # Use the 95th percentile of diffs to handle mixed-resolution data.
        # E.g., price data has both 15-min (ENTSO-E) and 1-hour (Energy-Charts)
        # intervals. median would be 15 min, but 1h is also normal.
        expected_max = time_diffs.quantile(0.95)
        large_gaps = time_diffs[time_diffs > expected_max * 2]

        if len(large_gaps) > 0:
            report.add("DST/Anomaly",
                f"{name}: Found {len(large_gaps)} gaps exceeding {expected_max * 2}")
            # Limit output to first 20 gaps to keep report clean
            for i, (idx, gap) in enumerate(large_gaps.items()):
                if i >= 20:
                    report.add("DST/Anomaly",
                        f"{name}: ... and {len(large_gaps) - 20} more gaps (truncated)")
                    break
                gap_start = df.index[idx - 1] if idx > 0 else df.index[0]
                gap_hours = gap.total_seconds() / 3600
                report.add("DST/Anomaly",
                    f"{name}: WARNING gap of {gap_hours:.1f}h at {gap_start} "
                    f"(possible DST spring-forward or data outage)")
        else:
            report.add("DST/Anomaly", f"{name}: No unexpected gaps ✓")

    # --- Check for DST-related day-length anomalies ---
    if df.index.tz is not None:
        cet_index = df.index.tz_convert("Europe/Berlin")
        daily_counts = pd.Series(1, index=cet_index).groupby(cet_index.date).count()

        # Detect expected daily counts for each resolution present
        # (e.g., 96 for 15-min, 24 for hourly)
        time_diffs = pd.Series(df.index).diff().dropna()
        unique_intervals = time_diffs.value_counts()
        valid_daily_counts = set()
        for interval, _ in unique_intervals.items():
            hours = interval.total_seconds() / 3600
            if hours > 0:
                valid_daily_counts.add(round(24 / hours))

        report.add("DST/Anomaly",
            f"{name}: Detected resolutions → {sorted(valid_daily_counts)} records/day")

        # A day is anomalous only if it doesn't match ANY expected count ± DST offset
        # DST adds/removes 1 hour → ±4 records for 15-min, ±1 for hourly
        anomalous_short = []
        anomalous_long = []
        for date, count in daily_counts.items():
            is_normal = False
            for expected in valid_daily_counts:
                # Allow ±1 hour equivalent of records
                records_per_hour = expected / 24
                dst_tolerance = records_per_hour + 1  # generous tolerance
                if abs(count - expected) <= dst_tolerance:
                    is_normal = True
                    break
            if not is_normal:
                if count < min(valid_daily_counts) - 2:
                    anomalous_short.append((date, count))
                elif count > max(valid_daily_counts) + 2:
                    anomalous_long.append((date, count))

        # Only report genuine DST days (±1h from expected)
        for expected in valid_daily_counts:
            rph = expected / 24
            spring_count = expected - rph  # lose 1 hour
            fall_count = expected + rph    # gain 1 hour
            spring_days = daily_counts[daily_counts == spring_count]
            fall_days = daily_counts[daily_counts == fall_count]
            for date, count in spring_days.items():
                report.add("DST/Anomaly",
                    f"{name}: DST spring-forward {date} ({int(count)} vs expected {expected})")
            for date, count in fall_days.items():
                report.add("DST/Anomaly",
                    f"{name}: DST fall-back {date} ({int(count)} vs expected {expected})")

        if len(anomalous_short) == 0 and len(anomalous_long) == 0:
            report.add("DST/Anomaly", f"{name}: No unexplained day-length anomalies ✓")

    return df


# ---------------------------------------------------------------------------
# Step 3: Resample to Hourly
# ---------------------------------------------------------------------------
def resample_to_hourly(df: pd.DataFrame, name: str, report: QAReport) -> pd.DataFrame:
    """
    Resample sub-hourly data (e.g., 15-min) to hourly using mean aggregation.
    Prices are already hourly — skip resampling for them.
    """
    # Detect resolution
    if len(df) > 1:
        median_freq = pd.Series(df.index).diff().dropna().median()
        minutes = median_freq.total_seconds() / 60
    else:
        minutes = 60

    if minutes < 55:  # Sub-hourly
        # Keep only numeric columns for resampling
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

        df_resampled = df[numeric_cols].resample("h").mean()

        # Carry forward non-numeric columns (like source_quality)
        if non_numeric_cols:
            df_non_numeric = df[non_numeric_cols].resample("h").first()
            df_resampled = pd.concat([df_resampled, df_non_numeric], axis=1)

        report.add("Resample", f"{name}: {minutes:.0f}-min → hourly ({len(df)} → {len(df_resampled)} rows)")
        return df_resampled
    else:
        report.add("Resample", f"{name}: already hourly ({len(df)} rows) ✓")
        return df


# ---------------------------------------------------------------------------
# Step 4: Merge
# ---------------------------------------------------------------------------
def merge_datasets(data: dict, report: QAReport) -> pd.DataFrame:
    """Merge prices, load, and wind_solar on their shared hourly UTC index."""
    logger.info("Step 4: Merging datasets ...")

    prices = data["prices"].drop(columns=["source_quality"], errors="ignore")
    load = data["load"].drop(columns=["source_quality"], errors="ignore")
    wind_solar = data["wind_solar"].drop(columns=["source_quality"], errors="ignore")

    # Keep source_quality from prices as the canonical source flag
    source_quality = data["prices"][["source_quality"]] if "source_quality" in data["prices"].columns else None

    merged = prices.join(load, how="inner").join(wind_solar, how="inner")

    if source_quality is not None:
        merged = merged.join(source_quality, how="left")

    report.add("Merge", f"Merged shape: {merged.shape}")
    report.add("Merge", f"Date range: {merged.index.min()} → {merged.index.max()}")
    report.add("Merge", f"Inner join retained {len(merged)} rows")

    return merged


# ---------------------------------------------------------------------------
# Step 5: Missing Value Handling
# ---------------------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame, report: QAReport) -> pd.DataFrame:
    """Handle NaNs via rolling-median correction then forward-fill."""
    logger.info("Step 5: Handling missing values ...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Record pre-imputation NaN counts
    nan_before = df[numeric_cols].isna().sum()
    report.add("NaN-Before", f"Missing values per column:\n{nan_before.to_string()}")

    # Rolling median imputation for short gaps (window=24h)
    for col in numeric_cols:
        mask = df[col].isna()
        if mask.any():
            rolling_med = df[col].rolling(window=24, min_periods=1, center=True).median()
            df.loc[mask, col] = rolling_med[mask]

    # Forward-fill any remaining NaNs
    df[numeric_cols] = df[numeric_cols].ffill()

    # Backward-fill for leading NaNs
    df[numeric_cols] = df[numeric_cols].bfill()

    nan_after = df[numeric_cols].isna().sum()
    report.add("NaN-After", f"Missing values after imputation:\n{nan_after.to_string()}")

    return df


# ---------------------------------------------------------------------------
# Step 6: Outlier Flagging (NOT deletion)
# ---------------------------------------------------------------------------
def flag_outliers(df: pd.DataFrame, report: QAReport) -> pd.DataFrame:
    """
    Flag extreme values using IQR method.
    Prices: flag but DO NOT delete — extreme spikes/negatives are real
    physical supply-demand mismatches.
    """
    logger.info("Step 6: Flagging outliers ...")

    df["price_outlier"] = False

    price_col = "price_eur_mwh"
    q1 = df[price_col].quantile(0.25)
    q3 = df[price_col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr

    outlier_mask = (df[price_col] < lower) | (df[price_col] > upper)
    df.loc[outlier_mask, "price_outlier"] = True

    n_outliers = outlier_mask.sum()
    report.add("Outliers", f"Price IQR bounds: [{lower:.2f}, {upper:.2f}] €/MWh")
    report.add("Outliers", f"Price outliers flagged: {n_outliers} ({100*n_outliers/len(df):.2f}%)")
    report.add("Outliers", f"Negative prices: {(df[price_col] < 0).sum()}")
    report.add("Outliers", f"Price range: [{df[price_col].min():.2f}, {df[price_col].max():.2f}] €/MWh")

    return df


# ---------------------------------------------------------------------------
# Step 7: Boundary Checks
# ---------------------------------------------------------------------------
def boundary_checks(df: pd.DataFrame, report: QAReport) -> pd.DataFrame:
    """Flag physically impossible values."""
    logger.info("Step 7: Boundary checks ...")

    # Load should be positive
    bad_load = (df["actual_load_mw"] < 0).sum()
    report.add("Boundary", f"Negative load values: {bad_load}")
    if bad_load > 0:
        df.loc[df["actual_load_mw"] < 0, "actual_load_mw"] = np.nan
        df["actual_load_mw"] = df["actual_load_mw"].ffill()

    # Wind/solar should be non-negative
    for col in ["wind_forecast_mw", "solar_forecast_mw"]:
        bad = (df[col] < 0).sum()
        report.add("Boundary", f"Negative {col}: {bad}")
        if bad > 0:
            df.loc[df[col] < 0, col] = 0

    # Timestamp monotonicity
    is_monotonic = df.index.is_monotonic_increasing
    report.add("Boundary", f"Index strictly monotonic: {is_monotonic}")
    if not is_monotonic:
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        report.add("Boundary", "Fixed: sorted and deduplicated index")

    return df


# ---------------------------------------------------------------------------
# Step 8: Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame, report: QAReport) -> pd.DataFrame:
    """Create temporal, lag, and derived features."""
    logger.info("Step 8: Feature engineering ...")

    # --- Temporal features ---
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek  # 0=Monday
    df["month"] = df.index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # --- Lag features (critical for autoregressive modeling) ---
    df["price_lag_24h"] = df["price_eur_mwh"].shift(24)   # Same hour yesterday
    df["price_lag_168h"] = df["price_eur_mwh"].shift(168)  # Same hour last week
    df["load_lag_24h"] = df["actual_load_mw"].shift(24)

    # --- Renewable penetration ratio ---
    total_renewable = df["wind_forecast_mw"] + df["solar_forecast_mw"]
    df["renewable_penetration"] = total_renewable / df["actual_load_mw"].replace(0, np.nan)
    df["renewable_penetration"] = df["renewable_penetration"].clip(0, 5)  # Cap at 500%

    # --- Rolling statistics ---
    df["price_rolling_mean_24h"] = df["price_eur_mwh"].rolling(24, min_periods=1).mean()
    df["price_rolling_std_24h"] = df["price_eur_mwh"].rolling(24, min_periods=1).std()

    # --- Net load (load minus renewables) ---
    df["net_load_mw"] = df["actual_load_mw"] - total_renewable
    df["net_load_mw"] = df["net_load_mw"].clip(lower=0)

    # Drop rows where lag features create NaNs (first 168 hours = 7 days)
    initial_len = len(df)
    df = df.dropna(subset=["price_lag_168h"])
    dropped = initial_len - len(df)

    features_added = [
        "hour_of_day", "day_of_week", "month", "is_weekend",
        "price_lag_24h", "price_lag_168h", "load_lag_24h",
        "renewable_penetration", "price_rolling_mean_24h",
        "price_rolling_std_24h", "net_load_mw",
    ]
    report.add("Features", f"Added {len(features_added)} features: {features_added}")
    report.add("Features", f"Dropped {dropped} rows due to lag NaNs (first 7 days)")
    report.add("Features", f"Final dataset shape: {df.shape}")

    return df


# ---------------------------------------------------------------------------
# Step 9: Exploratory Figures
# ---------------------------------------------------------------------------
def create_figures(df: pd.DataFrame):
    """Generate exploratory plots for the report."""
    logger.info("Step 9: Generating figures ...")

    # --- Price time series ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Price
    axes[0].plot(df.index, df["price_eur_mwh"], linewidth=0.3, color="#2196F3", alpha=0.8)
    outliers = df[df["price_outlier"] == True]
    axes[0].scatter(outliers.index, outliers["price_eur_mwh"],
                    color="red", s=3, alpha=0.6, label=f"Outliers ({len(outliers)})", zorder=5)
    axes[0].set_ylabel("Price (€/MWh)")
    axes[0].set_title("German Day-Ahead Electricity Prices", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Load
    axes[1].plot(df.index, df["actual_load_mw"] / 1000, linewidth=0.3, color="#4CAF50", alpha=0.8)
    axes[1].set_ylabel("Load (GW)")
    axes[1].set_title("Actual Total Load", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # Renewables
    axes[2].plot(df.index, df["wind_forecast_mw"] / 1000, linewidth=0.3, color="#00BCD4", alpha=0.7, label="Wind")
    axes[2].plot(df.index, df["solar_forecast_mw"] / 1000, linewidth=0.3, color="#FF9800", alpha=0.7, label="Solar")
    axes[2].set_ylabel("Generation (GW)")
    axes[2].set_title("Wind & Solar Generation", fontweight="bold")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "price_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved figure → {FIG_DIR / 'price_timeseries.png'}")

    # --- Price distribution ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["price_eur_mwh"], bins=150, color="#2196F3", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1, label="Zero price")
    ax.set_xlabel("Price (€/MWh)")
    ax.set_ylabel("Frequency")
    ax.set_title("Day-Ahead Price Distribution (DE)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "price_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved figure → {FIG_DIR / 'price_distribution.png'}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("QA & Feature Engineering Pipeline")
    logger.info("=" * 60)

    report = QAReport()

    # Step 1: Load
    data = load_raw_data()

    # Step 2: UTC standardization
    logger.info("Step 2: UTC standardization ...")
    for key in data:
        data[key] = standardize_utc(data[key], key, report)

    # Step 2b: DST / missing-hour anomaly detection
    logger.info("Step 2b: DST / anomaly detection ...")
    for key in data:
        data[key] = detect_dst_anomalies(data[key], key, report)

    # Step 3: Resample to hourly
    logger.info("Step 3: Resampling to hourly ...")
    for key in data:
        data[key] = resample_to_hourly(data[key], key, report)

    # Step 4: Merge
    merged = merge_datasets(data, report)

    # Step 5: Missing values
    merged = handle_missing_values(merged, report)

    # Step 6: Outlier flagging
    merged = flag_outliers(merged, report)

    # Step 7: Boundary checks
    merged = boundary_checks(merged, report)

    # Step 8: Feature engineering
    merged = engineer_features(merged, report)

    # Step 9: Figures
    create_figures(merged)

    # Save final dataset
    output_path = PROCESSED_DIR / "model_ready.csv"
    merged.to_csv(output_path)
    logger.info(f"Final dataset saved → {output_path}")
    report.add("Output", f"Saved: {output_path} | Shape: {merged.shape}")

    # Save QA report
    report.save(LOG_DIR / "qa_report.txt")

    # Print summary
    logger.info("=" * 60)
    logger.info("Pipeline Summary:")
    logger.info(f"  Final shape:    {merged.shape}")
    logger.info(f"  Date range:     {merged.index.min()} → {merged.index.max()}")
    logger.info(f"  Columns:        {list(merged.columns)}")
    logger.info(f"  NaN remaining:  {merged.isna().sum().sum()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
