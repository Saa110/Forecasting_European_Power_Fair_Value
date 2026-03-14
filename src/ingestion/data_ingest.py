"""
Data Ingestion Orchestrator
============================
Resilient data ingestion pipeline with tiered fallback strategy:

    1. ENTSO-E API        (Primary — official EU transparency data)
    2. Energy-Charts API  (Fallback — same data, stabler delivery)
    3. Local CSV Cache    (Last Resort — last known good data)

Every record is tagged with a `source_quality` flag:
    PRIMARY  — fresh data from ENTSO-E
    FALLBACK — fresh data from Energy-Charts
    CACHED   — stale data from local cache (last known good)

Usage:
    python src/ingestion/data_ingest.py

    Optional arguments:
        --market DE_LU          Target bidding zone (default: DE_LU)
        --start  2024-01-01     Start date (default: 2024-01-01)
        --end    2024-12-31     End date (default: 2024-12-31)
        --force-fallback        Skip ENTSO-E, go straight to fallback
        --force-cache           Skip all APIs, use cached data only
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for d in [RAW_DATA_DIR, CACHE_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ingestion.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source-specific pullers
# ---------------------------------------------------------------------------
def _pull_entsoe(market: str, start_str: str, end_str: str) -> dict:
    """Attempt full data pull from ENTSO-E API."""
    from src.ingestion.entsoe_pull import (
        get_client,
        pull_day_ahead_prices,
        pull_actual_total_load,
        pull_wind_solar_forecasts,
    )

    client = get_client()
    start = pd.Timestamp(start_str, tz="Europe/Berlin")
    # ENTSO-E uses exclusive end → add one day
    end = pd.Timestamp(end_str, tz="Europe/Berlin") + pd.Timedelta(days=1)

    return {
        "prices": pull_day_ahead_prices(client, start, end),
        "load": pull_actual_total_load(client, start, end),
        "wind_solar": pull_wind_solar_forecasts(client, start, end),
    }


def _pull_energy_charts(market: str, start_str: str, end_str: str) -> dict:
    """Attempt full data pull from Energy-Charts API."""
    from src.ingestion.energy_charts_pull import pull_all_data

    return pull_all_data(market, start_str, end_str)


def _load_from_cache() -> dict:
    """Load last known good data from the local CSV cache."""
    results = {}
    file_map = {
        "prices": "day_ahead_prices.csv",
        "load": "actual_total_load.csv",
        "wind_solar": "wind_solar_forecast.csv",
    }

    for key, filename in file_map.items():
        filepath = CACHE_DIR / filename
        if filepath.exists():
            df = pd.read_csv(filepath, index_col="timestamp", parse_dates=True)
            logger.info(f"  [Cache] Loaded {len(df)} rows from {filepath}")
            results[key] = df
        else:
            raise FileNotFoundError(
                f"No cached file found at {filepath}. "
                "Cannot serve data — all sources exhausted."
            )

    return results


def _save_to_cache(data: dict):
    """Persist current data as the 'last known good' cache."""
    file_map = {
        "prices": "day_ahead_prices.csv",
        "load": "actual_total_load.csv",
        "wind_solar": "wind_solar_forecast.csv",
    }
    for key, filename in file_map.items():
        if key in data:
            filepath = CACHE_DIR / filename
            data[key].to_csv(filepath)
            logger.info(f"  [Cache] Updated cache: {filepath}")


def _add_source_flag(data: dict, flag: str) -> dict:
    """Add a `source_quality` column to every DataFrame in the dict."""
    for key in data:
        data[key]["source_quality"] = flag
    return data


def _save_raw(data: dict):
    """Save all DataFrames to /data/raw/ as CSVs."""
    file_map = {
        "prices": "day_ahead_prices.csv",
        "load": "actual_total_load.csv",
        "wind_solar": "wind_solar_forecast.csv",
    }
    for key, filename in file_map.items():
        if key in data:
            filepath = RAW_DATA_DIR / filename
            data[key].to_csv(filepath)
            logger.info(f"  Saved {len(data[key])} rows → {filepath}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def ingest_data(
    market: str = "DE_LU",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    force_fallback: bool = False,
    force_cache: bool = False,
) -> dict:
    """
    Main ingestion function with tiered fallback logic.

    Returns:
        dict with keys 'prices', 'load', 'wind_solar', each a DataFrame
        with a `source_quality` column.
    """
    logger.info("=" * 60)
    logger.info("Data Ingestion Orchestrator")
    logger.info(f"Market: {market} | Range: {start_date} → {end_date}")
    logger.info("=" * 60)

    # --- Tier 0: Force cache (for testing / offline mode) ---
    if force_cache:
        logger.info("[Tier 0] Force-cache mode — loading from local cache ...")
        data = _load_from_cache()
        data = _add_source_flag(data, "CACHED")
        _save_raw(data)
        return data

    # --- Tier 1: ENTSO-E (Primary) ---
    if not force_fallback:
        try:
            logger.info("[Tier 1] Attempting ENTSO-E API (primary source) ...")
            data = _pull_entsoe(market, start_date, end_date)
            data = _add_source_flag(data, "PRIMARY")
            _save_raw(data)
            _save_to_cache(data)  # Update cache with fresh data
            logger.info("[Tier 1] ✅ ENTSO-E pull successful.")
            return data
        except Exception as e:
            logger.warning(f"[Tier 1] ❌ ENTSO-E failed: {e}")
            logger.info("[Tier 1] Falling back to Energy-Charts ...")

    # --- Tier 2: Energy-Charts (Fallback) ---
    try:
        logger.info("[Tier 2] Attempting Energy-Charts API (fallback) ...")
        data = _pull_energy_charts(market, start_date, end_date)
        data = _add_source_flag(data, "FALLBACK")
        _save_raw(data)
        _save_to_cache(data)  # Update cache
        logger.info("[Tier 2] ✅ Energy-Charts pull successful.")
        return data
    except Exception as e:
        logger.warning(f"[Tier 2] ❌ Energy-Charts failed: {e}")
        logger.info("[Tier 2] Falling back to local cache ...")

    # --- Tier 3: Local Cache (Last Resort) ---
    try:
        logger.info("[Tier 3] Serving from local CSV cache (last known good) ...")
        data = _load_from_cache()
        data = _add_source_flag(data, "CACHED")
        _save_raw(data)
        logger.warning(
            "[Tier 3] ⚠️  Using CACHED data — results may be stale. "
            "Check /logs/ingestion.log for failure details."
        )
        return data
    except FileNotFoundError as e:
        logger.error(f"[Tier 3] ❌ Cache miss: {e}")
        logger.error(
            "ALL DATA SOURCES EXHAUSTED. Cannot proceed. "
            "Please check your API token and network connectivity."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Resilient data ingestion with fallback sources"
    )
    parser.add_argument("--market", default="DE_LU", help="Bidding zone (default: DE_LU)")
    # Default end = yesterday (most recent complete day of data)
    from datetime import datetime, timedelta
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=yesterday, help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument(
        "--force-fallback", action="store_true",
        help="Skip ENTSO-E, use Energy-Charts directly"
    )
    parser.add_argument(
        "--force-cache", action="store_true",
        help="Skip all APIs, use cached data only"
    )
    args = parser.parse_args()

    data = ingest_data(
        market=args.market,
        start_date=args.start,
        end_date=args.end,
        force_fallback=args.force_fallback,
        force_cache=args.force_cache,
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Ingestion Summary:")
    for key, df in data.items():
        source = df["source_quality"].iloc[0] if "source_quality" in df.columns else "UNKNOWN"
        logger.info(f"  {key:12s}: {len(df):,} rows | source: {source}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
