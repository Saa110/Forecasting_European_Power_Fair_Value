"""
ENTSO-E Data Ingestion Script
==============================
Pulls Day-Ahead market data for the German (DE) bidding zone from
the ENTSO-E Transparency Platform API.

Data domains pulled:
  1. Day-Ahead Prices          (target variable)
  2. Actual Total Load         (demand fundamental)
  3. Day-Ahead Wind Forecast   (supply fundamental)
  4. Day-Ahead Solar Forecast  (supply fundamental)

Timeframe: January 1, 2024 – December 31, 2024
Output:    CSV files per domain in /data/raw/

Usage:
    python src/ingestion/entsoe_pull.py

Requires:
    ENTSOE_API_KEY in .env file
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

API_KEY = os.getenv("ENTSOE_API_KEY")
COUNTRY_CODE = "DE_LU"  # Germany-Luxembourg bidding zone (post-2018 split)

# Date range for data pull (extended to present for real-time relevance)
from datetime import timedelta
START_DATE = "20240101"
_yesterday = (datetime.now() - timedelta(days=1))
END_DATE = (_yesterday + timedelta(days=1)).strftime("%Y%m%d")  # ENTSO-E uses exclusive end dates

# Output directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LOG_DIR = PROJECT_ROOT / "logs"

# Retry configuration for API robustness
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
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
# Helpers
# ---------------------------------------------------------------------------
def get_client():
    """Initialize the ENTSO-E Pandas client with retry-friendly validation."""
    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        logger.error(
            "entsoe-py not installed. Run: pip install entsoe-py"
        )
        sys.exit(1)

    if not API_KEY:
        logger.error(
            "ENTSOE_API_KEY not found in environment. "
            "Copy .env.example to .env and add your token."
        )
        sys.exit(1)

    return EntsoePandasClient(api_key=API_KEY)


def fetch_with_retry(func, *args, **kwargs):
    """
    Wrap an ENTSO-E API call with exponential back-off retry logic.
    The API occasionally returns 503 / 429 errors under load.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = func(*args, **kwargs)
            return data
        except Exception as e:
            logger.warning(
                f"Attempt {attempt}/{MAX_RETRIES} failed: {e}"
            )
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY_SECONDS * attempt
                logger.info(f"Retrying in {wait}s ...")
                time.sleep(wait)
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed for {func.__name__}")
                raise


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    """Save a DataFrame to CSV in the raw data directory."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RAW_DATA_DIR / filename
    df.to_csv(filepath)
    logger.info(f"Saved {len(df)} rows → {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Data Pull Functions
# ---------------------------------------------------------------------------
def pull_day_ahead_prices(client, start, end):
    """Pull Day-Ahead clearing prices (€/MWh) — the TARGET variable."""
    logger.info("Pulling Day-Ahead Prices ...")
    prices = fetch_with_retry(
        client.query_day_ahead_prices, COUNTRY_CODE, start=start, end=end
    )
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name="price_eur_mwh")
    prices.index.name = "timestamp"
    return prices


def pull_actual_total_load(client, start, end):
    """Pull Actual Total Load (MW) — demand fundamental."""
    logger.info("Pulling Actual Total Load ...")
    load = fetch_with_retry(
        client.query_load, COUNTRY_CODE, start=start, end=end
    )
    if isinstance(load, pd.Series):
        load = load.to_frame(name="actual_load_mw")
    elif isinstance(load, pd.DataFrame):
        # Some zones return multiple columns; take 'Actual Load'
        if "Actual Load" in load.columns:
            load = load[["Actual Load"]].rename(
                columns={"Actual Load": "actual_load_mw"}
            )
        else:
            load.columns = ["actual_load_mw"] + [
                f"load_col_{i}" for i in range(1, len(load.columns))
            ]
    load.index.name = "timestamp"
    return load


def pull_wind_solar_forecasts(client, start, end):
    """
    Pull Day-Ahead generation forecasts for Wind and Solar (MW).
    These are the primary supply-side fundamentals driving price
    cannibalization in the German market.
    """
    logger.info("Pulling Day-Ahead Wind & Solar Generation Forecasts ...")
    gen = fetch_with_retry(
        client.query_wind_and_solar_forecast,
        COUNTRY_CODE,
        start=start,
        end=end,
    )

    # The API may return a DataFrame with columns like:
    #   'Solar', 'Wind Offshore', 'Wind Onshore'
    # We consolidate wind into a single column.
    result = pd.DataFrame(index=gen.index)
    result.index.name = "timestamp"

    # Solar
    solar_cols = [c for c in gen.columns if "solar" in c.lower()]
    if solar_cols:
        result["solar_forecast_mw"] = gen[solar_cols].sum(axis=1)
    else:
        logger.warning("No solar forecast columns found in API response")
        result["solar_forecast_mw"] = 0

    # Wind (onshore + offshore)
    wind_cols = [c for c in gen.columns if "wind" in c.lower()]
    if wind_cols:
        result["wind_forecast_mw"] = gen[wind_cols].sum(axis=1)
    else:
        logger.warning("No wind forecast columns found in API response")
        result["wind_forecast_mw"] = 0

    return result


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("ENTSO-E Data Ingestion Pipeline")
    logger.info(f"Market: {COUNTRY_CODE} | Range: {START_DATE} → {END_DATE}")
    logger.info("=" * 60)

    client = get_client()

    # Build timezone-aware date range (ENTSO-E requires tz-aware timestamps)
    start = pd.Timestamp(START_DATE, tz="Europe/Berlin")
    end = pd.Timestamp(END_DATE, tz="Europe/Berlin")

    # --- Pull each data domain ---
    try:
        prices = pull_day_ahead_prices(client, start, end)
        save_csv(prices, "day_ahead_prices.csv")
    except Exception as e:
        logger.error(f"Failed to pull Day-Ahead Prices: {e}")

    try:
        load = pull_actual_total_load(client, start, end)
        save_csv(load, "actual_total_load.csv")
    except Exception as e:
        logger.error(f"Failed to pull Actual Total Load: {e}")

    try:
        wind_solar = pull_wind_solar_forecasts(client, start, end)
        save_csv(wind_solar, "wind_solar_forecast.csv")
    except Exception as e:
        logger.error(f"Failed to pull Wind/Solar Forecasts: {e}")

    logger.info("=" * 60)
    logger.info("Ingestion complete. Raw files saved to: %s", RAW_DATA_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
