"""
Tests for ``oda.data`` — Binance OHLCV fetcher with multi-TF caching.

Run with::

    python -m pytest tests/test_data.py -v
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from oda.config import BacktestConfig, DataConfig, RiskConfig, Settings, TradingConfig
from oda.data import (
    API_TIMEOUT,
    CACHE_MAX_AGE,
    OHLCV_COLUMNS,
    _cache_path,
    _is_cache_fresh,
    _load_cache,
    _save_cache,
    _fetch_binance_klines,
    fetch_all,
    fetch_ohlcv,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mock_klines(n: int = 3) -> list[list]:
    """Build a minimal Binance klines response with *n* candles.

    Binance returns a list of lists::

        [open_time_ms, open, high, low, close, volume, ...]
    """
    base_ts = 1_700_000_000_000  # ~ Nov 2023 in ms
    rows = []
    for i in range(n):
        ts = base_ts + i * 3_600_000  # +1h per candle
        rows.append(
            [
                ts,
                str(40000.0 + i),   # open
                str(40100.0 + i),   # high
                str(39900.0 + i),   # low
                str(40050.0 + i),   # close
                str(100.0 + i),     # volume
                0, 0, 0, 0, 0, "0",  # unused trailing fields
            ]
        )
    return rows


def _make_settings(tmp_path: Path, **kwargs) -> Settings:
    """Create a Settings dataclass pointing at *tmp_path*."""
    # Build DataConfig defaults, letting kwargs override them.
    dc_kwargs: dict = {
        "symbol": "BTCUSDT",
        "intervals": ("15m", "1h"),
        "cache_dir": tmp_path,
        "binance_base_url": "https://api.binance.com",
        "rate_limit_rps": 10.0,  # fast for tests
    }
    dc_kwargs.update(
        {k: v for k, v in kwargs.items() if k in DataConfig.__dataclass_fields__}
    )
    data_config = DataConfig(**dc_kwargs)
    return Settings(
        data=data_config,
        trading=TradingConfig(),
        risk=RiskConfig(),
        backtest=BacktestConfig(),
        project_root=tmp_path,
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmpdir_path():
    """Yield a temporary directory as a Path, cleaned up afterwards."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def settings(tmpdir_path):
    """A Settings instance with a clean temporary cache directory."""
    return _make_settings(tmpdir_path)


# ── Cache Path Tests ─────────────────────────────────────────────────────────


class TestCachePath:
    """Cache path generation."""

    def test_format(self, settings):
        path = _cache_path("BTCUSDT", "1h", settings.data.cache_dir)
        assert path.name == "BTCUSDT_1h.csv"
        assert path.parent == settings.data.cache_dir

    def test_creates_parent_directory(self, settings):
        sub = settings.data.cache_dir / "nested" / "deep"
        path = _cache_path("BTCUSDT", "1w", sub)
        assert sub.exists()
        assert path.name == "BTCUSDT_1w.csv"

    def test_different_intervals_produce_different_paths(self, settings):
        p15 = _cache_path("BTCUSDT", "15m", settings.data.cache_dir)
        p1h = _cache_path("BTCUSDT", "1h", settings.data.cache_dir)
        assert p15 != p1h
        assert p15.name == "BTCUSDT_15m.csv"
        assert p1h.name == "BTCUSDT_1h.csv"

    def test_path_includes_symbol(self, settings):
        btc = _cache_path("BTCUSDT", "1h", settings.data.cache_dir)
        eth = _cache_path("ETHUSDT", "1h", settings.data.cache_dir)
        assert btc.name == "BTCUSDT_1h.csv"
        assert eth.name == "ETHUSDT_1h.csv"
        assert btc != eth


# ── Cache Freshness Tests ─────────────────────────────────────────────────────


class TestCacheFreshness:
    """_is_cache_fresh logic."""

    def test_missing_file_not_fresh(self, settings):
        path = settings.data.cache_dir / "no_such_file.csv"
        assert not path.exists()
        assert not _is_cache_fresh(path)

    def test_newly_created_file_is_fresh(self, settings):
        path = settings.data.cache_dir / "brand_new.csv"
        path.write_text("dummy")
        assert _is_cache_fresh(path)

    def test_old_file_is_not_fresh(self, settings):
        path = settings.data.cache_dir / "ancient.csv"
        path.write_text("dummy")

        # Set mtime to > 24 h ago
        old_ts = time.time() - (CACHE_MAX_AGE.total_seconds() + 3600)
        os.utime(str(path), (old_ts, old_ts))

        assert not _is_cache_fresh(path)

    def test_exactly_24h_boundary(self, settings):
        """A file that is exactly 24 h old is NOT fresh (strict < check)."""
        path = settings.data.cache_dir / "boundary.csv"
        path.write_text("dummy")

        boundary_ts = time.time() - CACHE_MAX_AGE.total_seconds()
        os.utime(str(path), (boundary_ts, boundary_ts))

        assert not _is_cache_fresh(path), (
            "Exactly 24 h old should be stale (strict less-than)"
        )

    def test_just_under_24h_is_fresh(self, settings):
        """A file just under 24 h old IS fresh."""
        path = settings.data.cache_dir / "almost_stale.csv"
        path.write_text("dummy")

        fresh_ts = time.time() - CACHE_MAX_AGE.total_seconds() + 60
        os.utime(str(path), (fresh_ts, fresh_ts))

        assert _is_cache_fresh(path)


# ── Cache I/O Tests ──────────────────────────────────────────────────────────


class TestCacheIO:
    """_save_cache / _load_cache round-trip."""

    def test_roundtrip(self, settings):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "open": [40000.0, 40100.0],
                "high": [40100.0, 40200.0],
                "low": [39900.0, 40000.0],
                "close": [40050.0, 40150.0],
                "volume": [100.0, 200.0],
            }
        )
        cache_file = settings.data.cache_dir / "test_roundtrip.csv"
        _save_cache(df, cache_file)

        loaded = _load_cache(cache_file)
        pd.testing.assert_frame_equal(df, loaded)

    def test_load_preserves_dtypes(self, settings):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2024-06-15 12:00"]),
                "open": [50000.0],
                "high": [50100.0],
                "low": [49900.0],
                "close": [50050.0],
                "volume": [150.0],
            }
        )
        cache_file = settings.data.cache_dir / "dtype_test.csv"
        _save_cache(df, cache_file)

        loaded = _load_cache(cache_file)
        assert pd.api.types.is_datetime64_any_dtype(loaded["time"])
        for col in ["open", "high", "low", "close", "volume"]:
            assert loaded[col].dtype == "float64", f"{col} dtype mismatch"


# ── Binance API Parsing Tests ────────────────────────────────────────────────


class TestFetchBinanceKlines:
    """_fetch_binance_klines parsing logic."""

    def test_parses_valid_response(self):
        mock_kline_data = _make_mock_klines(5)
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = mock_kline_data
        mock_resp.raise_for_status.return_value = None

        with patch("oda.data.requests.get", return_value=mock_resp) as mock_get:
            result = _fetch_binance_klines(
                symbol="BTCUSDT",
                interval="1h",
                base_url="https://api.binance.com",
            )

        # Verify the API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args.kwargs["params"]["symbol"] == "BTCUSDT"
        assert call_args.kwargs["params"]["interval"] == "1h"
        assert call_args.kwargs["params"]["limit"] == 1000
        assert call_args.kwargs["timeout"] == API_TIMEOUT

        # Verify parsed result
        assert len(result) == 5
        for row in result:
            assert set(row.keys()) == set(OHLCV_COLUMNS)
            assert isinstance(row["time"], pd.Timestamp)
            assert isinstance(row["open"], float)

    def test_empty_response(self):
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None

        with patch("oda.data.requests.get", return_value=mock_resp):
            result = _fetch_binance_klines(
                symbol="BTCUSDT", interval="1h", base_url="https://api.binance.com"
            )

        assert result == []

    def test_raises_on_non_list_response(self):
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = {"code": -1121, "msg": "Invalid symbol"}
        mock_resp.raise_for_status.return_value = None

        with patch("oda.data.requests.get", return_value=mock_resp):
            with pytest.raises(ValueError, match="Unexpected Binance response type"):
                _fetch_binance_klines(
                    symbol="INVALID$$$",
                    interval="1h",
                    base_url="https://api.binance.com",
                )

    def test_raises_on_http_error(self):
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch("oda.data.requests.get", return_value=mock_resp):
            with pytest.raises(requests.HTTPError, match="404"):
                _fetch_binance_klines(
                    symbol="BTCUSDT", interval="1h", base_url="https://bad.url"
                )

    def test_raises_on_network_error(self):
        with patch(
            "oda.data.requests.get",
            side_effect=requests.ConnectionError("Network unreachable"),
        ):
            with pytest.raises(requests.ConnectionError, match="Network"):
                _fetch_binance_klines(
                    symbol="BTCUSDT", interval="1h", base_url="https://api.binance.com"
                )


# ── fetch_ohlcv Tests ────────────────────────────────────────────────────────


class TestFetchOhlcv:
    """fetch_ohlcv caching and fallback behaviour."""

    def test_cache_hit_returns_cached_data(self, settings):
        """When a fresh cache exists, return it without calling the API."""
        cache_file = _cache_path("BTCUSDT", "1h", settings.data.cache_dir)
        df_cached = pd.DataFrame(
            {
                "time": pd.to_datetime(["2024-01-01"]),
                "open": [40000.0],
                "high": [40100.0],
                "low": [39900.0],
                "close": [40050.0],
                "volume": [100.0],
            }
        )
        _save_cache(df_cached, cache_file)

        with patch("oda.data.requests.get") as mock_get:
            result = fetch_ohlcv("BTCUSDT", "1h", settings)
            mock_get.assert_not_called()

        pd.testing.assert_frame_equal(df_cached, result)

    def test_cache_miss_fetches_from_api(self, settings):
        """When no cache exists, fetch from the API and persist."""
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = _make_mock_klines(3)
        mock_resp.raise_for_status.return_value = None

        with patch("oda.data.requests.get", return_value=mock_resp):
            result = fetch_ohlcv("BTCUSDT", "1h", settings)

        assert len(result) == 3
        assert list(result.columns) == OHLCV_COLUMNS

        # The cache file should now exist
        cache_file = _cache_path("BTCUSDT", "1h", settings.data.cache_dir)
        assert cache_file.exists()

    def test_force_refresh_skips_fresh_cache(self, settings):
        """force_refresh=True always fetches from API even if cache is fresh."""
        cache_file = _cache_path("BTCUSDT", "1h", settings.data.cache_dir)
        df_cached = pd.DataFrame(
            {
                "time": pd.to_datetime(["2024-01-01"]),
                "open": [99999.0],
                "high": [99999.0],
                "low": [99999.0],
                "close": [99999.0],
                "volume": [0.0],
            }
        )
        _save_cache(df_cached, cache_file)

        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = _make_mock_klines(2)
        mock_resp.raise_for_status.return_value = None

        with patch("oda.data.requests.get", return_value=mock_resp) as mock_get:
            result = fetch_ohlcv("BTCUSDT", "1h", settings, force_refresh=True)
            mock_get.assert_called_once()

        # Result should be the fresh API data, not the cached placeholder
        assert len(result) == 2

    def test_falls_back_to_stale_cache_on_api_error(self, settings):
        """When the API is unreachable, a stale cache is returned if available."""
        cache_file = _cache_path("BTCUSDT", "1h", settings.data.cache_dir)
        df_stale = pd.DataFrame(
            {
                "time": pd.to_datetime(["2024-01-01"]),
                "open": [40000.0],
                "high": [40100.0],
                "low": [39900.0],
                "close": [40050.0],
                "volume": [100.0],
            }
        )
        _save_cache(df_stale, cache_file)

        # Make the cache stale
        old_ts = time.time() - (CACHE_MAX_AGE.total_seconds() + 7200)
        os.utime(str(cache_file), (old_ts, old_ts))

        with patch(
            "oda.data.requests.get",
            side_effect=requests.ConnectionError("No network"),
        ):
            result = fetch_ohlcv("BTCUSDT", "1h", settings)

        pd.testing.assert_frame_equal(df_stale, result)

    def test_raises_when_no_cache_and_api_fails(self, settings):
        """If there is no cache at all, an API error propagates."""
        with patch(
            "oda.data.requests.get",
            side_effect=requests.ConnectionError("No network"),
        ):
            with pytest.raises(requests.ConnectionError):
                fetch_ohlcv("BTCUSDT", "1h", settings)


# ── fetch_all Tests ──────────────────────────────────────────────────────────


class TestFetchAll:
    """fetch_all multi-timeframe orchestration."""

    def test_returns_dict_keyed_by_interval(self, settings):
        """Should return a dict with one entry per configured interval."""
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = _make_mock_klines(2)
        mock_resp.raise_for_status.return_value = None

        with patch("oda.data.requests.get", return_value=mock_resp):
            data = fetch_all(settings)

        assert isinstance(data, dict)
        assert set(data.keys()) == set(settings.data.intervals)  # ("15m", "1h")
        for df in data.values():
            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == OHLCV_COLUMNS

    def test_respects_rate_limit_delay(self, settings):
        """Verify that time.sleep is called between requests."""
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = _make_mock_klines(1)
        mock_resp.raise_for_status.return_value = None

        with patch("oda.data.requests.get", return_value=mock_resp):
            with patch("oda.data.time.sleep") as mock_sleep:
                fetch_all(settings)

        # With 2 intervals and rate_limit_rps=10 → delay = 0.1 s
        # sleep should be called once (between 1st and 2nd request)
        assert mock_sleep.call_count == 1
        expected_delay = 1.0 / settings.data.rate_limit_rps  # 0.1
        mock_sleep.assert_called_with(expected_delay)

    def test_caches_are_written_for_all_intervals(self, settings):
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = _make_mock_klines(2)
        mock_resp.raise_for_status.return_value = None

        with patch("oda.data.requests.get", return_value=mock_resp):
            fetch_all(settings)

        for interval in settings.data.intervals:
            cache_file = _cache_path("BTCUSDT", interval, settings.data.cache_dir)
            assert cache_file.exists(), f"Missing cache for {interval}"

    def test_single_interval_no_sleep(self, settings):
        """With only one interval, no sleep should be called."""
        # Override intervals to a single value
        single_settings = _make_settings(
            settings.data.cache_dir, intervals=("4h",)
        )

        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = _make_mock_klines(1)
        mock_resp.raise_for_status.return_value = None

        with patch("oda.data.requests.get", return_value=mock_resp):
            with patch("oda.data.time.sleep") as mock_sleep:
                fetch_all(single_settings)

        mock_sleep.assert_not_called()


# ── Integration-style Test ───────────────────────────────────────────────────


class TestLiveIntegration:
    """Optional real API call — skipped when network is unavailable."""

    @pytest.mark.integration
    def test_real_binance_fetch(self, settings):
        """Perform a real fetch from Binance public API.

        Marked ``integration`` so it can be skipped in CI with
        ``pytest -m "not integration"``.
        """
        try:
            df = fetch_ohlcv("BTCUSDT", "1h", settings, force_refresh=True)
        except requests.RequestException as exc:
            pytest.skip(f"No network / Binance unreachable: {exc}")

        assert len(df) > 0, "Expected at least one candle"
        assert list(df.columns) == OHLCV_COLUMNS
        assert pd.api.types.is_datetime64_any_dtype(df["time"])
        assert df["open"].iloc[0] > 0
        print(f"\n✓ Real Binance fetch OK — {len(df)} rows for BTCUSDT 1h")


# ── Main guard ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    # Simple runner when executed directly (no pytest needed)
    print("=" * 60)
    print("oda.data — manual smoke test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        s = _make_settings(tmp)

        # 1. Cache path
        p = _cache_path("BTCUSDT", "1h", s.data.cache_dir)
        print(f"✓ cache_path: {p}")
        assert p.name == "BTCUSDT_1h.csv"

        # 2. Cache freshness
        cf = tmp / "fresh.csv"
        cf.write_text("x")
        assert _is_cache_fresh(cf)
        print("✓ _is_cache_fresh returns True for new file")

        # 3. Save / load roundtrip
        df_orig = pd.DataFrame(
            {
                "time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "open": [40000.0, 40100.0],
                "high": [40100.0, 40200.0],
                "low": [39900.0, 40000.0],
                "close": [40050.0, 40150.0],
                "volume": [100.0, 200.0],
            }
        )
        rtf = tmp / "roundtrip.csv"
        _save_cache(df_orig, rtf)
        df_loaded = _load_cache(rtf)
        pd.testing.assert_frame_equal(df_orig, df_loaded)
        print("✓ save/load roundtrip")

        # 4. Mocked API parse
        mock_k = _make_mock_klines(3)
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = mock_k
        mock_resp.raise_for_status.return_value = None
        with patch("oda.data.requests.get", return_value=mock_resp):
            parsed = _fetch_binance_klines("BTCUSDT", "1h", "https://api.binance.com")
        assert len(parsed) == 3
        print(f"✓ mocked API parse: {len(parsed)} candles")

        # 5. Mocked fetch_all
        with patch("oda.data.requests.get", return_value=mock_resp):
            data = fetch_all(s)
        assert len(data) == 2
        print(f"✓ fetch_all: {list(data.keys())}")

        # 6. Try real Binance fetch
        print("\n--- Real Binance fetch (may fail offline) ---")
        try:
            real_df = fetch_ohlcv("BTCUSDT", "1h", s, force_refresh=True)
            print(f"✓ Real API: {len(real_df)} rows")
        except Exception as exc:
            print(f"⚠ Real API unavailable: {exc}")

        print("\n" + "=" * 60)
        print("All smoke tests passed.")
        print("=" * 60)
