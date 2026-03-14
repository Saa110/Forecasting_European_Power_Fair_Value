"""
Energy-Charts Fallback Data Ingestion
======================================
Fallback data source for DE/FR/NL markets when ENTSO-E API is unavailable.

Energy-Charts (api.energy-charts.info) serves the same underlying TSO data
via a stable JSON API. It is free, requires no API key, and has excellent
uptime compared to the ENTSO-E Transparency Platform.

API Reference: https://api.energy-charts.info

Usage:
    This module is called by data_ingest.py when the ENTSO-E pull fails.
    It should NOT be run directly — use data_ingest.py instead.
"""

import logging
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Energy-Charts API Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://api.energy-charts.info"

# Map internal market codes to Energy-Charts bidding zone / country codes
# Price endpoint uses 'bzn' (bidding zone), power endpoint uses 'country'
MARKET_TO_BZN = {
    "DE_LU": "DE-LU",
    "FR": "FR",
    "NL": "NL",
}

MARKET_TO_COUNTRY = {
    "DE_LU": "de",
    "FR": "fr",
    "NL": "nl",
}


# ---------------------------------------------------------------------------
# API Fetch Helpers
# ---------------------------------------------------------------------------
def _fetch_json(endpoint: str, params: dict) -> dict:
    """Fetch JSON from Energy-Charts API with error handling."""
    url = f"{BASE_URL}/{endpoint}"
    logger.info(f"  [Energy-Charts] GET {url} | params={params}")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _unix_to_utc_index(unix_seconds: list) -> pd.DatetimeIndex:
    """Convert a list of UNIX timestamps (seconds) to a UTC DatetimeIndex."""
    return pd.to_datetime(unix_seconds, unit="s", utc=True)


# ---------------------------------------------------------------------------
# Data Pull Functions
# ---------------------------------------------------------------------------
def pull_day_ahead_prices(market: str, start: str, end: str) -> pd.DataFrame:
    """
    Pull Day-Ahead electricity prices from Energy-Charts.

    Args:
        market: Internal market code (e.g., 'DE_LU')
        start:  Start date string 'YYYY-MM-DD'
        end:    End date string 'YYYY-MM-DD'

    Returns:
        DataFrame with UTC DatetimeIndex and 'price_eur_mwh' column.
    """
    bzn = MARKET_TO_BZN[market]
    logger.info(f"  [Energy-Charts] Pulling Day-Ahead Prices for {bzn} ...")

    data = _fetch_json("price", {
        "bzn": bzn,
        "start": start,
        "end": end,
    })

    timestamps = data.get("unix_seconds", [])
    prices = data.get("price", [])

    if not timestamps or not prices:
        raise ValueError(f"Energy-Charts returned empty price data for {bzn}")

    idx = _unix_to_utc_index(timestamps)
    df = pd.DataFrame({"price_eur_mwh": prices}, index=idx)
    df.index.name = "timestamp"

    logger.info(f"  [Energy-Charts] Prices: {len(df)} rows retrieved")
    return df


def pull_actual_total_load(market: str, start: str, end: str) -> pd.DataFrame:
    """
    Pull actual total load (power consumption) from Energy-Charts.

    The /public_power endpoint returns production types including 'Load'.

    Returns:
        DataFrame with UTC DatetimeIndex and 'actual_load_mw' column.
    """
    country = MARKET_TO_COUNTRY[market]
    logger.info(f"  [Energy-Charts] Pulling Total Load for {country} ...")

    data = _fetch_json("public_power", {
        "country": country,
        "start": start,
        "end": end,
    })

    timestamps = data.get("unix_seconds", [])
    production_types = data.get("production_types", [])

    # Find the 'Load' entry in production_types
    load_values = None
    for entry in production_types:
        if isinstance(entry, dict) and entry.get("name", "").lower() == "load":
            load_values = entry.get("data", [])
            break

    if load_values is None:
        raise ValueError(
            f"'Load' production type not found in Energy-Charts response. "
            f"Available types: {[e.get('name') if isinstance(e, dict) else e for e in production_types]}"
        )

    idx = _unix_to_utc_index(timestamps)
    df = pd.DataFrame({"actual_load_mw": load_values}, index=idx)
    df.index.name = "timestamp"

    # Coerce any None / null values to NaN
    df["actual_load_mw"] = pd.to_numeric(df["actual_load_mw"], errors="coerce")

    logger.info(f"  [Energy-Charts] Load: {len(df)} rows retrieved")
    return df


def pull_wind_solar_forecasts(market: str, start: str, end: str) -> pd.DataFrame:
    """
    Pull wind and solar generation data from Energy-Charts.

    Aggregates 'Wind onshore' + 'Wind offshore' into a single wind column,
    and extracts 'Solar' directly.

    Returns:
        DataFrame with UTC DatetimeIndex, 'wind_forecast_mw', 'solar_forecast_mw'.
    """
    country = MARKET_TO_COUNTRY[market]
    logger.info(f"  [Energy-Charts] Pulling Wind & Solar for {country} ...")

    data = _fetch_json("public_power", {
        "country": country,
        "start": start,
        "end": end,
    })

    timestamps = data.get("unix_seconds", [])
    production_types = data.get("production_types", [])
    n = len(timestamps)

    wind_total = [0.0] * n
    solar_total = [0.0] * n

    for entry in production_types:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "").lower()
        values = entry.get("data", [])

        if "wind" in name:
            # Accumulate wind onshore + wind offshore
            for i in range(min(n, len(values))):
                val = values[i]
                if val is not None:
                    wind_total[i] += val

        elif "solar" in name and "storage" not in name:
            # Solar generation (exclude battery storage)
            for i in range(min(n, len(values))):
                val = values[i]
                if val is not None:
                    solar_total[i] += val

    idx = _unix_to_utc_index(timestamps)
    df = pd.DataFrame({
        "wind_forecast_mw": wind_total,
        "solar_forecast_mw": solar_total,
    }, index=idx)
    df.index.name = "timestamp"

    logger.info(f"  [Energy-Charts] Wind+Solar: {len(df)} rows retrieved")
    return df


# ---------------------------------------------------------------------------
# Main Fallback Entry Point
# ---------------------------------------------------------------------------
def pull_all_data(market: str, start_date: str, end_date: str) -> dict:
    """
    Pull all required data domains from Energy-Charts for a given market.

    Args:
        market:     Internal market code (e.g., 'DE_LU')
        start_date: 'YYYY-MM-DD' format
        end_date:   'YYYY-MM-DD' format

    Returns:
        dict with keys: 'prices', 'load', 'wind_solar'
        Each value is a pandas DataFrame.
    """
    if market not in MARKET_TO_BZN:
        raise ValueError(
            f"Energy-Charts fallback not available for market '{market}'. "
            f"Supported: {list(MARKET_TO_BZN.keys())}"
        )

    logger.info(f"[Fallback] Energy-Charts pull for {market} ({start_date} → {end_date})")

    results = {}
    results["prices"] = pull_day_ahead_prices(market, start_date, end_date)
    results["load"] = pull_actual_total_load(market, start_date, end_date)
    results["wind_solar"] = pull_wind_solar_forecasts(market, start_date, end_date)

    return results
