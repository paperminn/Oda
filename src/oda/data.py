"""
Binance OHLCV Data Fetcher with Multi-Timeframe Caching.

Fetches public candlestick (kline) data from the Binance REST API for
multiple timeframes. Results are cached as CSV files and reused if the
cache is less than 24 hours old. Respects configurable rate limits.

Usage:
    from oda.config import Settings
    from oda.data import fetch_all

    settings = Settings.load()
    data: dict[str, pd.DataFrame] = fetch_all(settings)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from oda.config import Settings

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

OHLCV_COLUMNS: list[str] = ["time", "open", "high", "low", "close", "volume"]
"""Ordered column names for parsed OHLCV DataFrames."""

CACHE_MAX_AGE: timedelta = timedelta(hours=24)
"""Maximum age of a cache file before it is considered stale."""

API_TIMEOUT: float = 30.0
"""HTTP request timeout in seconds."""

# ── Cache Helpers ────────────────────────────────────────────────────────────


def _cache_path(symbol: str, interval: str, cache_dir: Path) -> Path:
    """Build the cache file path for a given symbol and interval.

    Args:
        symbol: Trading pair (e.g. ``"BTCUSDT"``).
        interval: Kline interval (e.g. ``"1h"``).
        cache_dir: Root cache directory.

    Returns:
        ``cache_dir / "BTCUSDT_1h.csv"`` (parent directories created).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{symbol}_{interval}.csv"


def _is_cache_fresh(cache_file: Path) -> bool:
    """Return ``True`` if *cache_file* exists and was modified within the last 24 h.

    Args:
        cache_file: Path to a CSV cache file.

    Returns:
        ``True`` if the file is present and less than ``CACHE_MAX_AGE`` old.
    """
    if not cache_file.exists():
        return False

    mtime = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
    age = datetime.now(timezone.utc) - mtime
    return age < CACHE_MAX_AGE


def _load_cache(cache_file: Path) -> pd.DataFrame:
    """Load an OHLCV DataFrame from a CSV cache file.

    Args:
        cache_file: Path to an existing CSV file that contains a ``"time"`` column.

    Returns:
        DataFrame with columns ``OHLCV_COLUMNS`` and parsed timestamps.
    """
    df = pd.read_csv(cache_file, parse_dates=["time"])
    logger.debug("Loaded %d rows from cache: %s", len(df), cache_file)
    return df


def _save_cache(df: pd.DataFrame, cache_file: Path) -> None:
    """Persist an OHLCV DataFrame to a CSV cache file.

    Args:
        df: DataFrame with columns matching ``OHLCV_COLUMNS``.
        cache_file: Destination path (parent directories must exist).
    """
    df.to_csv(cache_file, index=False)
    logger.debug("Saved %d rows to cache: %s", len(df), cache_file)


# ── Binance API ──────────────────────────────────────────────────────────────


def _fetch_binance_klines(
    symbol: str,
    interval: str,
    base_url: str,
    limit: int = 1000,
    end_time_ms: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Fetch raw kline data from Binance and parse it into a list of dicts.

    Calls ``GET /api/v3/klines`` (public endpoint, no API key required).

    Args:
        symbol: Trading pair, e.g. ``"BTCUSDT"``.
        interval: Kline interval, one of ``"15m"``, ``"1h"``, ``"4h"``,
            ``"1d"``, ``"1w"``.
        base_url: Binance REST base URL from config.
        limit: Maximum number of candles to return (default 1000, max 1000).
        end_time_ms: Optional end time in milliseconds. If set, returns
            candles up to (and including) this time.

    Returns:
        List of dicts with keys ``time``, ``open``, ``high``, ``low``,
        ``close``, ``volume``.

    Raises:
        requests.RequestException: On any HTTP or network error.
        ValueError: If the response body is not valid JSON.
    """
    url = f"{base_url}/api/v3/klines"
    params: dict[str, str | int] = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if end_time_ms is not None:
        params["endTime"] = end_time_ms

    logger.debug("GET %s params=%s", url, params)

    response = requests.get(url, params=params, timeout=API_TIMEOUT)
    response.raise_for_status()

    raw_klines: list[list[Any]] = response.json()

    if not isinstance(raw_klines, list):
        raise ValueError(
            f"Unexpected Binance response type: {type(raw_klines).__name__}. "
            f"Expected a list of klines."
        )

    parsed: list[dict[str, Any]] = []
    for k in raw_klines:
        parsed.append(
            {
                "time": pd.to_datetime(k[0], unit="ms"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
        )

    logger.info("Fetched %d klines for %s %s", len(parsed), symbol, interval)
    return parsed


def _fetch_all_historical(
    symbol: str,
    interval: str,
    base_url: str,
    rate_limit_rps: float = 1.0,
    start_date: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Fetch ALL available historical klines with pagination.

    Repeatedly calls Binance klines endpoint with ``endTime`` to paginate
    backward through history. Stops when the start_date is reached or
    Binance returns no more data.

    Args:
        symbol: Trading pair.
        interval: Kline interval.
        base_url: Binance REST base URL.
        rate_limit_rps: Max requests per second.
        start_date: ISO date string (e.g. "2021-01-01"). Stop when data
            reaches or passes this date.

    Returns:
        Complete list of kline dicts, sorted by time ascending, deduplicated.
    """
    all_klines: list[dict[str, Any]] = []
    min_ts_ms: Optional[int] = None
    if start_date:
        min_ts_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)

    end_ms: Optional[int] = None  # None = latest data
    batch_count = 0

    while True:
        batch = _fetch_binance_klines(
            symbol, interval, base_url, limit=1000, end_time_ms=end_ms
        )
        if not batch:
            logger.info("No more klines returned for %s %s after %d batches",
                        symbol, interval, batch_count)
            break

        all_klines.extend(batch)
        batch_count += 1

        # Earliest timestamp in this batch becomes the next endTime
        earliest = min(k["time"] for k in batch)
        earliest_ms = int(earliest.timestamp() * 1000)

        # Stop if we've reached the desired start date
        if min_ts_ms is not None and earliest_ms <= min_ts_ms:
            logger.info("Reached start date after %d batches (%d klines)",
                        batch_count, len(all_klines))
            break

        # Stop if we're not making progress (same timestamp)
        if end_ms is not None and earliest_ms >= end_ms:
            logger.warning("No progress — stopping at %d klines", len(all_klines))
            break

        end_ms = earliest_ms - 1  # Fetch older data in next request

        if batch_count % 5 == 0:
            logger.info("  ... batch %d: %d klines so far, earliest: %s",
                        batch_count, len(all_klines), earliest)

        time.sleep(1.0 / rate_limit_rps)

    # Deduplicate by time
    seen: set[pd.Timestamp] = set()
    deduped: list[dict[str, Any]] = []
    for k in all_klines:
        if k["time"] not in seen:
            seen.add(k["time"])
            deduped.append(k)

    deduped.sort(key=lambda k: k["time"])
    logger.info("Total historical klines for %s %s: %d (deduped from %d)",
                symbol, interval, len(deduped), len(all_klines))
    return deduped


# ── Public API ───────────────────────────────────────────────────────────────


def fetch_ohlcv(
    symbol: str,
    interval: str,
    settings: Settings,
    *,
    force_refresh: bool = False,
    historical: bool = False,
    start_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch OHLCV data for a single symbol/interval pair with caching.

    If a fresh cache exists (``< 24 h``) it is returned immediately.
    Otherwise the data is fetched from Binance and cached to disk.
    On API failure, a stale cache is used as a fallback when available.

    Args:
        symbol: Trading pair, e.g. ``"BTCUSDT"``.
        interval: Candle interval, e.g. ``"1h"``.
        settings: Frozen application settings (provides base URL, cache
            directory, etc.).
        force_refresh: If ``True``, skip the cache and always fetch from
            the API.
        historical: If ``True``, paginate to fetch all available history.
        start_date: ISO date string (e.g. "2021-01-01"). Only used when
            ``historical=True``.

    Returns:
        DataFrame with columns ``time`` (``datetime64[ns]``), ``open``,
        ``high``, ``low``, ``close``, ``volume`` (``float64``).

    Raises:
        requests.RequestException: If the API call fails and no cache
            fallback is available.
    """
    cache_file = _cache_path(symbol, interval, settings.data.cache_dir)

    if not force_refresh and _is_cache_fresh(cache_file):
        logger.info("Cache hit for %s %s → %s", symbol, interval, cache_file)
        return _load_cache(cache_file)

    logger.info("Fetching %s %s from Binance API …", symbol, interval)

    try:
        if historical:
            klines = _fetch_all_historical(
                symbol=symbol,
                interval=interval,
                base_url=settings.data.binance_base_url,
                rate_limit_rps=settings.data.rate_limit_rps,
                start_date=start_date,
            )
        else:
            klines = _fetch_binance_klines(
                symbol=symbol,
                interval=interval,
                base_url=settings.data.binance_base_url,
            )
    except (requests.RequestException, ValueError) as exc:
        logger.error("API fetch failed for %s %s: %s", symbol, interval, exc)

        # Fall back to stale cache when available
        if cache_file.exists():
            logger.warning(
                "Using stale cache for %s %s (API unavailable)", symbol, interval
            )
            return _load_cache(cache_file)

        raise

    df = pd.DataFrame(klines, columns=OHLCV_COLUMNS)
    _save_cache(df, cache_file)
    logger.info("Cached %d rows for %s %s → %s", len(df), symbol, interval, cache_file)

    return df


def fetch_all(
    settings: Settings,
    *,
    historical: bool = False,
    start_date: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for every timeframe listed in **settings.data.intervals**.

    A configurable back-off delay is inserted between successive API
    calls to stay within Binance rate limits.

    Args:
        settings: Frozen application settings.
        historical: If ``True``, paginate each timeframe to fetch all
            available history back to ``start_date``.
        start_date: ISO date string (e.g. "2021-01-01"). Only used with
            ``historical=True``.

    Returns:
        Dictionary mapping each interval string to its OHLCV DataFrame,
        e.g. ``{"15m": df15, "1h": df1h, "4h": df4h, "1d": df1d,
        "1w": df1w}``.
    """
    data: dict[str, pd.DataFrame] = {}
    intervals = settings.data.intervals
    delay: float = 1.0 / settings.data.rate_limit_rps

    for i, interval in enumerate(intervals):
        if i > 0:
            logger.debug("Rate-limit back-off: sleeping %.2f s", delay)
            time.sleep(delay)

        df = fetch_ohlcv(
            symbol=settings.data.symbol,
            interval=interval,
            settings=settings,
            historical=historical,
            start_date=start_date,
        )
        data[interval] = df

    logger.info(
        "Fetched OHLCV for %d timeframes: %s",
        len(data),
        ", ".join(data.keys()),
    )
    return data
