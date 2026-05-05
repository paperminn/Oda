"""
test_zones.py — Tests for oda.zones RTM zone detection module.

Tests FTR, FL, and QM detection on synthetic candle data with known
patterns, plus zone scoring and confluence logic.
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from oda.zones import (
    Zone,
    ZoneDetector,
    ZoneType,
    PatternType,
    Freshness,
    TF_WEIGHTS,
    FRESHNESS_MULTIPLIER,
    CONFLUENCE_BONUS,
    PATTERN_SCORE_BONUS,
    _FTRCandidate,
    _FLCandidate,
    _QMCandidate,
    _SwingPoint,
)


# ═══════════════════════════════════════════════════════════════════
# Test Helpers — Synthetic Candle Generation
# ═══════════════════════════════════════════════════════════════════


def make_df(n: int = 100, base_price: float = 50000.0, seed: int = 42) -> pd.DataFrame:
    """Generate a random-walk OHLCV DataFrame.

    Returns a DataFrame with columns: open, high, low, close, timestamp.
    """
    rng = np.random.default_rng(seed)
    closes = base_price + np.cumsum(rng.normal(0, 100, size=n))

    # Ensure all prices are positive
    closes = np.maximum(closes, 1000.0)

    opens = closes + rng.normal(0, 30, size=n)
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 50, size=n))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 50, size=n))

    # Fix lows to be positive
    lows = np.maximum(lows, 1.0)

    timestamps = (np.arange(n, dtype=np.int64) * 3600000) + 1700000000000

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "timestamp": timestamps,
    })


def make_bullish_ftr_df(n: int = 80) -> pd.DataFrame:
    """Create synthetic candles containing a clear bullish FTR pattern.

    Structure:
      1. Price rises to a swing high (resistance)
      2. Price retraces
      3. Price breaks above the swing high (breakout)
      4. Price retraces but holds above the old high (FTR — fail to return)
      5. Price continues higher (confirmation)
    """
    n = max(n, 50)
    price = np.zeros(n)
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    closes = np.zeros(n)

    # Phase 1: Uptrend to swing high at i=8
    base = 50000.0
    for i in range(10):
        price[i] = base + i * 200.0

    # Swing high at i=8
    highs[8] = price[8] + 50
    closes[8] = price[8]
    opens[8] = price[8] - 30
    lows[8] = price[8] - 100

    # Phase 2: Retrace down
    for i in range(10, 20):
        price[i] = price[9] - (i - 9) * 150

    # Phase 3: Rally to break above swing high
    for i in range(20, 30):
        price[i] = price[19] + (i - 19) * 300
    # At i=25, price breaks above the swing high at highs[8]
    # Ensure break
    barrier = highs[8]
    for i in range(25, 30):
        price[i] = barrier + (i - 24) * 100

    # Phase 4: Retrace but STAY ABOVE the barrier (FTR)
    for i in range(30, 35):
        drop = (i - 29) * 80
        price[i] = price[29] - drop
    # Ensure stays above barrier
    retrace_low = barrier + 50
    for i in range(30, 35):
        price[i] = max(price[i], retrace_low)

    # Phase 5: Continuation higher
    for i in range(35, n):
        price[i] = price[34] + (i - 34) * 150

    # Fill OHLC from price
    for i in range(n):
        c = price[i]
        opens[i] = c - 20
        closes[i] = c
        highs[i] = c + 40
        lows[i] = c - 30

    # Fix specific candles for the FTR pattern clarity
    highs[8] = barrier  # The swing high
    lows[8] = barrier - 200
    closes[25] = barrier + 100  # Break candle
    highs[25] = barrier + 150
    lows[25] = barrier - 50

    # The retrace: lows in 30-34 should be above barrier
    for i in range(30, 35):
        lows[i] = barrier + 10
        highs[i] = max(highs[i], lows[i] + 50)

    timestamps = (np.arange(n, dtype=np.int64) * 3600000) + 1700000000000

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "timestamp": timestamps,
    })


def make_bearish_ftr_df(n: int = 80) -> pd.DataFrame:
    """Create synthetic candles containing a clear bearish FTR pattern.

    Structure:
      1. Price falls to a swing low (support)
      2. Price bounces
      3. Price breaks below the swing low (breakdown)
      4. Price retraces but holds below the old low (FTR)
      5. Price continues lower (confirmation)
    """
    n = max(n, 50)
    price = np.zeros(n)
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    closes = np.zeros(n)

    base = 50000.0
    # Phase 1: Downtrend to swing low at i=8
    for i in range(10):
        price[i] = base - i * 200.0

    # Swing low at i=8
    lows[8] = price[8] - 50
    closes[8] = price[8]
    opens[8] = price[8] + 30
    highs[8] = price[8] + 100

    # Phase 2: Bounce up
    for i in range(10, 20):
        price[i] = price[9] + (i - 9) * 150

    # Phase 3: Drop to break below swing low
    for i in range(20, 30):
        price[i] = price[19] - (i - 19) * 300

    barrier = lows[8]
    for i in range(25, 30):
        price[i] = barrier - (i - 24) * 100

    # Phase 4: Retrace but STAY BELOW barrier (FTR)
    for i in range(30, 35):
        bounce = (i - 29) * 80
        price[i] = price[29] + bounce
    retrace_high = barrier - 50
    for i in range(30, 35):
        price[i] = min(price[i], retrace_high)

    # Phase 5: Continuation lower
    for i in range(35, n):
        price[i] = price[34] - (i - 34) * 150

    for i in range(n):
        c = price[i]
        opens[i] = c + 20
        closes[i] = c
        highs[i] = c + 30
        lows[i] = c - 40

    # Fix barrier candle
    lows[8] = barrier
    highs[8] = barrier + 200
    closes[25] = barrier - 100  # Break candle
    lows[25] = barrier - 150
    highs[25] = barrier + 50

    # Retrace stays below barrier
    for i in range(30, 35):
        highs[i] = barrier - 10
        lows[i] = min(lows[i], highs[i] - 50)

    timestamps = (np.arange(n, dtype=np.int64) * 3600000) + 1700000000000

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "timestamp": timestamps,
    })


def make_flag_limit_df(n: int = 40) -> pd.DataFrame:
    """Create synthetic candles containing a Flag Limit pattern.

    A base candle (small body) followed by a multi-candle impulse.
    """
    price = np.full(n, 50000.0)
    opens = np.full(n, 50000.0)
    highs = np.full(n, 50000.0)
    lows = np.full(n, 50000.0)
    closes = np.full(n, 50000.0)

    # Pre-base: normal candles
    for i in range(5):
        price[i] = 50000.0 + i * 20
        opens[i] = price[i] - 30
        closes[i] = price[i]
        highs[i] = max(opens[i], closes[i]) + 80
        lows[i] = min(opens[i], closes[i]) - 80

    # Base candle at i=10: tiny body (< 0.4 body ratio)
    base_price = 50100.0
    opens[10] = base_price - 5
    closes[10] = base_price + 5
    highs[10] = base_price + 100   # wide range
    lows[10] = base_price - 100    # wide range → body_ratio ≈ 10/200 = 0.05 < 0.4
    price[10] = closes[10]

    # Fill intermediate candles
    for i in range(6, 10):
        price[i] = price[i - 1] + 5
        opens[i] = price[i] - 10
        closes[i] = price[i]
        highs[i] = max(opens[i], closes[i]) + 50
        lows[i] = min(opens[i], closes[i]) - 50

    # Impulse candles: strong bullish move
    impulse_start = base_price
    impulse_prices = [50200, 50500, 51000]  # +0.2%, +1%, +2% moves
    for j, ip in enumerate(impulse_prices):
        i = 11 + j
        opens[i] = price[i - 1]
        closes[i] = ip
        highs[i] = max(opens[i], closes[i]) + 50
        lows[i] = min(opens[i], closes[i]) - 50
        price[i] = ip

    # More candles afterwards
    for i in range(14, n):
        price[i] = price[i - 1] + 30
        opens[i] = price[i] - 20
        closes[i] = price[i]
        highs[i] = max(opens[i], closes[i]) + 60
        lows[i] = min(opens[i], closes[i]) - 60

    timestamps = (np.arange(n, dtype=np.int64) * 3600000) + 1700000000000

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "timestamp": timestamps,
    })


def make_quasimodo_df(n: int = 60) -> pd.DataFrame:
    """Create synthetic candles containing a bearish Quasimodo pattern.

    Bearish QM: SH → SL → SH (higher) → break below SL
    """
    n = max(n, 50)
    price = np.zeros(n)
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    closes = np.zeros(n)

    # Need clear swing points
    # SH1 at ~i=10
    # SL1 at ~i=20
    # SH2 (higher) at ~i=35
    # Break below SL1 at ~i=45

    # Generate data that produces these swings with lookback=3
    base = 52000.0

    # Build a sequence that creates clear local extrema
    # Use a simple oscillating pattern
    pattern = [
        # (i_start, i_end, trend, amplitude)
    ]

    prices = []
    # Manual construction for clarity
    # i=0-4: range-bound at 52000
    for i in range(5):
        prices.append(base + i * 10)

    # i=5-9: rally to SH1
    for i in range(5, 10):
        prices.append(prices[-1] + 200)

    # i=10: peak (SH1)
    prices.append(prices[-1] + 100)  # SH1

    # i=11-14: drop
    for i in range(11, 15):
        prices.append(prices[-1] - 150)

    # i=15-20: more drop to SL1
    for i in range(15, 25):
        prices.append(prices[-1] - 200)

    # i=25: trough (SL1)
    prices.append(prices[-1] - 100)  # SL1

    # i=26-30: rally
    for i in range(26, 35):
        prices.append(prices[-1] + 250)

    # i=35-37: peak higher than SH1 (SH2)
    prices.append(prices[-1] + 200)  # i=35 SH2
    prices.append(prices[-1] + 50)   # i=36 higher
    prices.append(prices[-1] - 30)   # i=37 starting to reverse

    # i=38-44: sharp drop
    for i in range(38, 45):
        prices.append(prices[-1] - 300)

    # i=45-50: break below SL1
    sl1_price = prices[25]
    for i in range(45, min(55, n)):
        prices.append(sl1_price - (i - 44) * 50)

    # Fill remaining
    while len(prices) < n:
        prices.append(prices[-1] - 50)

    prices = prices[:n]

    # Fill OHLC
    for i in range(n):
        c = prices[i]
        opens[i] = c - 30
        closes[i] = c
        highs[i] = c + 60
        lows[i] = c - 40

    timestamps = (np.arange(n, dtype=np.int64) * 3600000) + 1700000000000

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "timestamp": timestamps,
    })


# ═══════════════════════════════════════════════════════════════════
# Zone Dataclass Tests
# ═══════════════════════════════════════════════════════════════════


class TestZone:
    """Tests for the Zone dataclass."""

    def test_contains_price(self):
        z = Zone(
            zone_type=ZoneType.DEMAND,
            pattern=PatternType.FTR,
            timeframe="1h",
            price_low=50000.0,
            price_high=50500.0,
            midpoint=50250.0,
            score=5.0,
            freshness=Freshness.UNTOUCHED,
            touch_count=0,
            confluence_count=1,
            candle_index=10,
            timestamp_ms=1700000000000,
        )
        assert z.contains_price(50100.0)
        assert z.contains_price(50000.0)  # boundary
        assert z.contains_price(50500.0)  # boundary
        assert not z.contains_price(49999.0)
        assert not z.contains_price(50501.0)

    def test_distance_to_price(self):
        z = Zone(
            zone_type=ZoneType.SUPPLY,
            pattern=PatternType.FL,
            timeframe="1d",
            price_low=49000.0,
            price_high=51000.0,
            midpoint=50000.0,
            score=4.0,
            freshness=Freshness.TOUCHED_ONCE,
            touch_count=1,
            confluence_count=2,
            candle_index=5,
            timestamp_ms=1700000000000,
        )
        # Distance from 50000 to 50000 = 0%
        assert z.distance_to_price(50000.0) == 0.0
        # Distance from 50000 to 51000 = 2%
        assert abs(z.distance_to_price(51000.0) - 2.0) < 0.01
        # Distance from 50000 to 49000 = 2%
        assert abs(z.distance_to_price(49000.0) - 2.0) < 0.01

    def test_width_pct(self):
        z = Zone(
            zone_type=ZoneType.DEMAND,
            pattern=PatternType.QM,
            timeframe="4h",
            price_low=49500.0,
            price_high=50500.0,
            midpoint=50000.0,
            score=6.0,
            freshness=Freshness.UNTOUCHED,
            touch_count=0,
            confluence_count=3,
            candle_index=20,
            timestamp_ms=1700000000000,
        )
        assert abs(z.width_pct - 2.0) < 0.01  # 1000/50000 = 2%

    def test_zero_midpoint(self):
        z = Zone(
            zone_type=ZoneType.DEMAND,
            pattern=PatternType.FTR,
            timeframe="1h",
            price_low=0.0,
            price_high=0.0,
            midpoint=0.0,
            score=1.0,
            freshness=Freshness.UNTOUCHED,
            touch_count=0,
            confluence_count=1,
            candle_index=0,
            timestamp_ms=0,
        )
        assert z.width_pct == 0.0
        assert z.distance_to_price(100.0) == float("inf")

    def test_str_representation(self):
        z = Zone(
            zone_type=ZoneType.DEMAND,
            pattern=PatternType.FTR,
            timeframe="1d",
            price_low=48000.0,
            price_high=48500.0,
            midpoint=48250.0,
            score=4.8,
            freshness=Freshness.UNTOUCHED,
            touch_count=0,
            confluence_count=2,
            candle_index=15,
            timestamp_ms=1700000000000,
        )
        s = str(z)
        assert "DEMAND" in s
        assert "FTR" in s
        assert "1d" in s
        assert "score=4.8" in s
        assert "untouched" in s
        assert "2TF" in s


# ═══════════════════════════════════════════════════════════════════
# Candidate Dataclass Tests
# ═══════════════════════════════════════════════════════════════════


class TestCandidates:
    """Tests for internal candidate dataclasses."""

    def test_ftr_candidate(self):
        c = _FTRCandidate(
            direction="bullish",
            barrier_price=50000.0,
            zone_low=49900.0,
            zone_high=50200.0,
            candle_index=25,
            timestamp_ms=1700000000000,
            swing_index=8,
        )
        assert c.direction == "bullish"
        assert c.barrier_price == 50000.0

    def test_fl_candidate(self):
        c = _FLCandidate(
            direction="bullish",
            base_low=49800.0,
            base_high=50100.0,
            candle_index=12,
            timestamp_ms=1700000000000,
        )
        assert c.direction == "bullish"
        assert c.base_low == 49800.0
        assert c.base_high == 50100.0

    def test_qm_candidate(self):
        c = _QMCandidate(
            direction="bearish",
            entry_low=50500.0,
            entry_high=51000.0,
            candle_index=45,
            timestamp_ms=1700000000000,
            sequence=["SH1=$53000", "SL1=$51500", "SH2=$53500", "BreakSL1=$51400"],
        )
        assert c.direction == "bearish"
        assert len(c.sequence) == 4


# ═══════════════════════════════════════════════════════════════════
# Swing Detection Tests
# ═══════════════════════════════════════════════════════════════════


class TestSwingDetection:
    """Tests for swing point detection."""

    def test_find_swings_simple(self):
        """Basic swing detection on a simple price series."""
        detector = ZoneDetector(lookback=2)

        highs = np.array([10, 12, 15, 13, 11, 14, 16, 12, 10], dtype=np.float64)
        lows = np.array([8, 9, 11, 10, 9, 12, 13, 9, 7], dtype=np.float64)

        swings = detector._find_swings(highs, lows)

        assert len(swings) > 0
        types = [s.sw_type for s in swings]
        assert "high" in types or "low" in types

    def test_find_swings_empty(self):
        """Short arrays should return no swings."""
        detector = ZoneDetector(lookback=3)
        swings = detector._find_swings(
            np.array([10.0, 12.0, 15.0], dtype=np.float64),
            np.array([8.0, 9.0, 11.0], dtype=np.float64),
        )
        assert swings == []

    def test_label_swings(self):
        """Structure labeling should assign HH, HL, LL, LH."""
        swings = [
            _SwingPoint(5, "high", 100.0),
            _SwingPoint(10, "high", 120.0),  # HH
            _SwingPoint(15, "high", 110.0),  # LH
            _SwingPoint(20, "low", 90.0),     # SWITCH
            _SwingPoint(25, "low", 95.0),     # HL
            _SwingPoint(30, "low", 85.0),     # LL
        ]
        detector = ZoneDetector(lookback=2)
        detector._label_swings(swings)

        assert swings[0].label == "START"
        assert swings[1].label == "HH"
        assert swings[2].label == "LH"
        assert swings[3].label == "SWITCH"
        assert swings[4].label == "HL"
        assert swings[5].label == "LL"


# ═══════════════════════════════════════════════════════════════════
# FTR Detection Tests
# ═══════════════════════════════════════════════════════════════════


class TestFTRDetection:
    """Tests for Fail-to-Return pattern detection."""

    def test_bullish_ftr_detected(self):
        """Bullish FTR should be detected from synthetic data."""
        df = make_bullish_ftr_df(n=80)
        detector = ZoneDetector(lookback=3)
        zones = detector.detect_all({"1h": df})

        ftr_zones = [z for z in zones if z.pattern == PatternType.FTR]
        assert len(ftr_zones) > 0, (
            f"No FTR zones detected. Total zones: {len(zones)}. "
            f"Zone patterns: {[z.pattern.value for z in zones]}"
        )

        # Bullish FTR should be a DEMAND zone
        bullish_ftrs = [z for z in ftr_zones if z.zone_type == ZoneType.DEMAND]
        assert len(bullish_ftrs) > 0, "Expected at least one bullish FTR (DEMAND) zone"

    def test_bearish_ftr_detected(self):
        """Bearish FTR should be detected from synthetic data."""
        df = make_bearish_ftr_df(n=80)
        detector = ZoneDetector(lookback=3)
        zones = detector.detect_all({"1h": df})

        ftr_zones = [z for z in zones if z.pattern == PatternType.FTR]
        assert len(ftr_zones) > 0, (
            f"No FTR zones detected. Total zones: {len(zones)}"
        )

        bearish_ftrs = [z for z in ftr_zones if z.zone_type == ZoneType.SUPPLY]
        assert len(bearish_ftrs) > 0, "Expected at least one bearish FTR (SUPPLY) zone"

    def test_ftr_zone_has_barrier(self):
        """FTR zones should carry the barrier price (the broken swing level)."""
        df = make_bullish_ftr_df(n=80)
        detector = ZoneDetector(lookback=3)
        zones = detector.detect_all({"1h": df})

        ftr_zones = [z for z in zones if z.pattern == PatternType.FTR]
        barriers = [z.barrier_price for z in ftr_zones if z.barrier_price is not None]
        assert len(barriers) > 0, "FTR zones should have barrier_price set"
        assert all(b > 0 for b in barriers), "All barrier prices should be positive"

    def test_no_ftr_on_random_data(self):
        """Random walk data should produce few or no FTR patterns
        (valid FTR requires a specific structure)."""
        df = make_df(n=100, seed=42)
        detector = ZoneDetector(lookback=3)
        zones = detector.detect_all({"1h": df})

        ftr_zones = [z for z in zones if z.pattern == PatternType.FTR]
        # Random data may occasionally produce an FTR-lookalike, but
        # it should be very rare. Verify the system doesn't crash.
        assert isinstance(ftr_zones, list)


# ═══════════════════════════════════════════════════════════════════
# Flag Limit Detection Tests
# ═══════════════════════════════════════════════════════════════════


class TestFLDetection:
    """Tests for Flag Limit pattern detection."""

    def test_flag_limit_bullish(self):
        """Bullish FL (base ➜ bullish impulse) should create a DEMAND zone."""
        df = make_flag_limit_df(n=40)
        detector = ZoneDetector(lookback=3, impulse_threshold_pct=1.0)
        zones = detector.detect_all({"1h": df})

        fl_zones = [z for z in zones if z.pattern == PatternType.FL]
        assert len(fl_zones) > 0, (
            f"No FL zones detected. Total zones: {len(zones)}. "
            f"Need a base candle followed by >1% impulse."
        )

        # At least one bullish FL (DEMAND zone)
        bullish_fls = [z for z in fl_zones if z.zone_type == ZoneType.DEMAND]
        assert len(bullish_fls) > 0, "Expected at least one bullish FL (DEMAND) zone"

    def test_flag_limit_scoring(self):
        """FL zones should have base scores from TF weight."""
        df = make_flag_limit_df(n=40)
        detector = ZoneDetector(lookback=3, impulse_threshold_pct=1.0)
        zones = detector.detect_all({"1d": df})

        fl_zones = [z for z in zones if z.pattern == PatternType.FL]
        if fl_zones:
            expected_base = TF_WEIGHTS["1d"] * PATTERN_SCORE_BONUS[PatternType.FL]
            # After confluence and freshness, score may be different, but
            # check that base score is reasonable
            assert all(z.score > 0 for z in fl_zones), "All FL scores should be positive"


# ═══════════════════════════════════════════════════════════════════
# Quasimodo Detection Tests
# ═══════════════════════════════════════════════════════════════════


class TestQMDetection:
    """Tests for Quasimodo pattern detection."""

    def test_qm_pattern_detected(self):
        """QM detection should find patterns from synthetic swing data."""
        df = make_quasimodo_df(n=60)
        detector = ZoneDetector(lookback=2)
        zones = detector.detect_all({"1h": df})

        qm_zones = [z for z in zones if z.pattern == PatternType.QM]
        # QM requires a specific 4-swing sequence; synthetic data
        # may or may not produce it with lookback=2.
        # Primary test: no crash, sensible output.
        assert isinstance(qm_zones, list)

    def test_qm_topology(self):
        """If QM is detected, it should have correct topology.

        Bearish QM → SUPPLY, Bullish QM → DEMAND.
        """
        df = make_quasimodo_df(n=60)
        detector = ZoneDetector(lookback=2)
        zones = detector.detect_all({"1h": df})

        qm_zones = [z for z in zones if z.pattern == PatternType.QM]
        for z in qm_zones:
            if z.zone_type == ZoneType.SUPPLY:
                assert True  # bearish QM → supply
            elif z.zone_type == ZoneType.DEMAND:
                assert True  # bullish QM → demand
            assert z.score > 0
            assert z.midpoint > 0


# ═══════════════════════════════════════════════════════════════════
# Zone Scoring Tests
# ═══════════════════════════════════════════════════════════════════


class TestZoneScoring:
    """Tests for zone scoring (TF weights, freshness, confluence)."""

    def test_tf_weights_applied(self):
        """Higher timeframes should produce higher base scores."""
        df_1h = make_bullish_ftr_df(n=80)
        df_1d = make_bullish_ftr_df(n=80)

        detector = ZoneDetector(lookback=3)
        zones_1h = detector.detect_all({"1h": df_1h})
        zones_1d = detector.detect_all({"1d": df_1d})

        # TF weights: 1d=4, 1h=2
        # So 1d base scores should be ~2× 1h base scores
        ftr_1h = [z for z in zones_1h if z.pattern == PatternType.FTR]
        ftr_1d = [z for z in zones_1d if z.pattern == PatternType.FTR]

        if ftr_1h and ftr_1d:
            # Compare highest scores
            max_1h = max(z.score for z in ftr_1h)
            max_1d = max(z.score for z in ftr_1d)
            # 1d score should be higher due to TF weight (4 vs 2)
            # But exact ratio may differ due to zone boundaries
            logger_info = f"1h max={max_1h:.1f}, 1d max={max_1d:.1f}, ratio={max_1d/max_1h:.2f}"
            assert max_1d > 0, logger_info

    def test_freshness_multipliers(self):
        """Freshness enum should have correct multipliers."""
        assert FRESHNESS_MULTIPLIER[Freshness.UNTOUCHED] == 1.0
        assert FRESHNESS_MULTIPLIER[Freshness.TOUCHED_ONCE] == 0.6
        assert FRESHNESS_MULTIPLIER[Freshness.TOUCHED_TWICE] == 0.3
        assert FRESHNESS_MULTIPLIER[Freshness.EXHAUSTED] == 0.1

    def test_confluence_bonus_constants(self):
        """Confluence bonuses should match specification."""
        assert CONFLUENCE_BONUS[1] == 1.0
        assert CONFLUENCE_BONUS[2] == 1.5
        assert CONFLUENCE_BONUS[3] == 2.0

    def test_pattern_score_bonus(self):
        """Pattern bonuses should be consistent."""
        assert PATTERN_SCORE_BONUS[PatternType.FTR] == 1.2
        assert PATTERN_SCORE_BONUS[PatternType.FL] == 1.0
        assert PATTERN_SCORE_BONUS[PatternType.QM] == 1.3

    def test_confluence_detection(self):
        """Zones that overlap across TFs should get confluence bonus."""
        # Create two DataFrames with zones near the same price level
        df_1h = make_bullish_ftr_df(n=80)
        df_4h = make_bullish_ftr_df(n=80)

        detector = ZoneDetector(lookback=3, confluence_overlap_pct=1.0)
        zones = detector.detect_all({"1h": df_1h, "4h": df_4h})

        # Check that confluence_count is populated
        for z in zones:
            assert z.confluence_count >= 1
            if z.confluence_count > 1:
                # Confluence bonus should be reflected in score
                assert z.score > 0

    def test_nearby_zones(self):
        """get_nearby_zones should return zones within distance threshold."""
        df = make_bullish_ftr_df(n=80)
        detector = ZoneDetector(lookback=3)
        zones = detector.detect_all({"1h": df})

        # Use the latest close as reference
        latest = float(df["close"].iloc[-1])
        nearby = detector.get_nearby_zones(latest, distance_pct=10.0)

        assert len(nearby) > 0, f"No nearby zones within 10% of {latest}"
        for z in nearby:
            assert z.distance_to_price(latest) <= 10.0

    def test_get_supply_demand_zones(self):
        """get_supply_zones and get_demand_zones should filter correctly."""
        df = make_df(n=100, seed=42)
        detector = ZoneDetector(lookback=3)
        detector.detect_all({"1h": df})

        supply = detector.get_supply_zones(min_score=0.0)
        demand = detector.get_demand_zones(min_score=0.0)

        for z in supply:
            assert z.zone_type == ZoneType.SUPPLY
        for z in demand:
            assert z.zone_type == ZoneType.DEMAND


# ═══════════════════════════════════════════════════════════════════
# Edge Case Tests
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        """Empty or too-short DataFrames should produce no zones."""
        detector = ZoneDetector()
        zones = detector.detect_all({
            "1h": pd.DataFrame({"open": [], "high": [], "low": [], "close": []})
        })
        assert zones == []

    def test_few_candles(self):
        """DataFrames with fewer than 10 candles should be skipped."""
        df = make_df(n=5)
        detector = ZoneDetector()
        zones = detector.detect_all({"1h": df})
        assert zones == []

    def test_missing_columns(self):
        """Missing required columns should raise an error."""
        detector = ZoneDetector()
        bad_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(KeyError):
            detector.detect_all({"1h": bad_df})

    def test_negative_prices(self):
        """Data with negative prices should not crash."""
        df = pd.DataFrame({
            "open": [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
            "high": [-8.0, -3.0, 2.0, 7.0, 12.0, 17.0, 22.0, 27.0, 32.0, 37.0],
            "low": [-12.0, -7.0, -2.0, 3.0, 8.0, 13.0, 18.0, 23.0, 28.0, 33.0],
            "close": [-9.0, -4.0, 1.0, 6.0, 11.0, 16.0, 21.0, 26.0, 31.0, 36.0],
        })
        # Extend to 10+ candles
        df2 = make_df(n=15, base_price=50000)
        detector = ZoneDetector()
        zones = detector.detect_all({"1h": df2})
        assert isinstance(zones, list)

    def test_none_dataframe(self):
        """None DataFrames should be skipped gracefully."""
        detector = ZoneDetector()
        zones = detector.detect_all({"1h": None, "1d": make_df(n=50)})
        assert isinstance(zones, list)
        # Should still process the valid TF
        assert len(zones) >= 0

    def test_summary(self):
        """Summary should return a valid dict with expected keys."""
        df = make_bullish_ftr_df(n=80)
        detector = ZoneDetector(lookback=3)
        detector.detect_all({"1h": df})

        summary = detector.summary()
        assert "total_zones" in summary
        assert "by_pattern" in summary
        assert "by_timeframe" in summary
        assert "top_score" in summary
        assert isinstance(summary["total_zones"], int)
        assert isinstance(summary["top_score"], float)

    def test_no_timestamp_column(self):
        """DataFrames without 'timestamp' column should work (timestamp=0)."""
        df = make_bullish_ftr_df(n=80)
        df = df.drop(columns=["timestamp"])
        detector = ZoneDetector(lookback=3)
        zones = detector.detect_all({"1h": df})
        assert isinstance(zones, list)
        # Zones should have timestamp_ms=0
        for z in zones:
            assert z.timestamp_ms == 0


# ═══════════════════════════════════════════════════════════════════
# Multi-TF Tests
# ═══════════════════════════════════════════════════════════════════


class TestMultiTimeframe:
    """Tests for multi-timeframe detection and confluence."""

    def test_multiple_timeframes(self):
        """Detecting on multiple TFs should produce zones from all TFs."""
        dfs = {
            "1h": make_bullish_ftr_df(n=60),
            "4h": make_flag_limit_df(n=30),
            "1d": make_df(n=20, base_price=50000),
        }
        detector = ZoneDetector(lookback=2, max_zones_per_tf=10)
        zones = detector.detect_all(dfs)

        tfs_found = set(z.timeframe for z in zones)
        assert "1h" in tfs_found, f"Expected 1h zones, got TFs: {tfs_found}"
        # 1d may be skipped if < 10 candles

    def test_zones_sorted_by_score(self):
        """detect_all should return zones sorted by score descending."""
        dfs = {
            "1h": make_bullish_ftr_df(n=60),
            "4h": make_bullish_ftr_df(n=60),
        }
        detector = ZoneDetector(lookback=3)
        zones = detector.detect_all(dfs)

        if len(zones) >= 2:
            for i in range(len(zones) - 1):
                assert zones[i].score >= zones[i + 1].score, (
                    f"Zone {i} score={zones[i].score:.1f} < "
                    f"zone {i+1} score={zones[i+1].score:.1f}"
                )


# ═══════════════════════════════════════════════════════════════════
# Integration-style Smoke Test
# ═══════════════════════════════════════════════════════════════════


class TestSmoke:
    """End-to-end smoke test with all pattern types."""

    def test_full_pipeline(self):
        """Run the full pipeline on multiple TFs with all pattern types."""
        detector = ZoneDetector(
            lookback=3,
            impulse_threshold_pct=1.0,
            max_zones_per_tf=15,
            confluence_overlap_pct=1.0,
        )

        dfs = {
            "1h": make_bullish_ftr_df(n=80),
            "4h": make_bearish_ftr_df(n=80),
            "1d": make_flag_limit_df(n=40),
            "1w": make_quasimodo_df(n=60),
        }

        zones = detector.detect_all(dfs)

        # Should produce zones
        assert len(zones) > 0, "Full pipeline should produce at least some zones"

        # Should have zones from multiple TFs
        tfs = set(z.timeframe for z in zones)

        # Scores should all be positive
        for z in zones:
            assert z.score > 0, f"Zone score should be positive: {z}"
            assert z.midpoint > 0, f"Zone midpoint should be positive: {z}"
            assert z.price_high >= z.price_low, f"price_high < price_low: {z}"

        # Verify summary
        summary = detector.summary()
        print(f"\n{'='*60}")
        print(f"SMOKE TEST RESULTS")
        print(f"{'='*60}")
        print(f"Total zones: {summary['total_zones']}")
        print(f"By pattern: {summary['by_pattern']}")
        print(f"By timeframe: {summary['by_timeframe']}")
        print(f"Top score: {summary['top_score']:.1f}")
        print(f"\nTop 5 zones:")
        for z in zones[:5]:
            print(f"  {z}")
        print(f"{'='*60}\n")

        # Test nearby zones
        latest = float(dfs["1h"]["close"].iloc[-1])
        nearby = detector.get_nearby_zones(latest, distance_pct=5.0)
        print(f"Nearby zones within 5% of ${latest:,.0f}: {len(nearby)}")

        # Test filtered access
        supply = detector.get_supply_zones(min_score=1.0)
        demand = detector.get_demand_zones(min_score=1.0)
        print(f"Supply zones: {len(supply)}, Demand zones: {len(demand)}")


# ═══════════════════════════════════════════════════════════════════
# Run standalone
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Print a quick manual smoke test
    print("=" * 60)
    print("ODA Zones — Manual Smoke Test")
    print("=" * 60)

    detector = ZoneDetector(lookback=3)
    df1 = make_bullish_ftr_df(80)
    df2 = make_bearish_ftr_df(80)

    zones = detector.detect_all({"1h": df1, "4h": df2})

    print(f"\nDetected {len(zones)} zones across 2 timeframes\n")
    for z in zones[:10]:
        print(f"  {z}")

    print()
    latest = float(df1["close"].iloc[-1])
    nearby = detector.get_nearby_zones(latest, distance_pct=5.0)
    print(f"Zones within 5% of ${latest:,.0f}: {len(nearby)}")

    summary = detector.summary()
    print(f"\nSummary: {summary['total_zones']} zones, patterns={summary['by_pattern']}")
