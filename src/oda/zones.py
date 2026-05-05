"""
RTM Zone Detection — Multi-Timeframe Supply/Demand Zones
=========================================================

Detects three RTM (Read The Market) patterns on OHLCV candle data:

  - **FTR** (Fail to Return): Price breaks a swing level, retests it, and fails
    to return — old resistance becomes new support, old support becomes new
    resistance. Implemented as a 5-step state machine.

  - **FL** (Flag Limit): A small-body "base" candle followed by an impulse move
    in the next 1–3 candles. The base zone acts as supply (bearish) or demand
    (bullish).

  - **QM** (Quasimodo): Structural reversal where price makes a higher high then
    breaks below a prior swing low (bearish QM), or a lower low then breaks
    above a prior swing high (bullish QM).

Zones are scored by timeframe weight, freshness, and multi-TF confluence.

Adapted from:
  - ``z-brain/research/rtm/rtm_detector.py`` (IF Myante FTR/FL/QM logic)
  - ``cryptomecha01-review/strategy/liquidity_map.py`` (confluence scoring)

Usage::

    import pandas as pd
    from oda.zones import ZoneDetector

    detector = ZoneDetector()
    zones = detector.detect_all({
        "1w": weekly_df,
        "1d": daily_df,
        "4h": h4_df,
        "1h": h1_df,
        "15m": m15_df,
    })
    for z in zones[:5]:
        print(z)

Dependencies: dataclasses, numpy, pandas, typing, enum, logging (stdlib only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("oda.zones")


# ═══════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════


class ZoneType(Enum):
    """Supply = resistance (price expected to reject down).
    Demand = support (price expected to reject up)."""

    SUPPLY = "SUPPLY"
    DEMAND = "DEMAND"


class PatternType(Enum):
    """RTM pattern that produced this zone."""

    FTR = "FTR"  # Fail to Return
    FL = "FL"  # Flag Limit
    QM = "QM"  # Quasimodo


class Freshness(Enum):
    """How many times price has revisited the zone since formation."""

    UNTOUCHED = "untouched"
    TOUCHED_ONCE = "touched_once"
    TOUCHED_TWICE = "touched_twice"
    EXHAUSTED = "exhausted"


# ═══════════════════════════════════════════════════════════════════
# Scoring Constants
# ═══════════════════════════════════════════════════════════════════

# Timeframe hierarchy weights — higher TFs dominate
TF_WEIGHTS: dict[str, int] = {
    "1w": 5,
    "1d": 4,
    "4h": 3,
    "1h": 2,
    "15m": 1,
}

# Freshness multiplier: untouched zones are strongest
FRESHNESS_MULTIPLIER: dict[Freshness, float] = {
    Freshness.UNTOUCHED: 1.0,
    Freshness.TOUCHED_ONCE: 0.6,
    Freshness.TOUCHED_TWICE: 0.3,
    Freshness.EXHAUSTED: 0.1,
}

# Confluence bonus when zones stack across multiple timeframes
CONFLUENCE_BONUS: dict[int, float] = {
    1: 1.0,  # Single TF — no bonus
    2: 1.5,  # Two aligned TFs
    3: 2.0,  # Three+ aligned TFs ("stacked zone")
}

# Zone overlap tolerance for confluence detection (0.5% of midpoint)
CONFLUENCE_OVERLAP_TOLERANCE: float = 0.005

# Default body ratio threshold for base candle classification
BASE_BODY_RATIO: float = 0.4

# Maximum base body % for Flag Limit detection (scales by timeframe)
BASE_BODY_THRESHOLDS: dict[str, float] = {
    "1w": 2.0,
    "1d": 1.5,
    "4h": 1.0,
    "1h": 0.8,
    "15m": 0.5,
}

# Minimum impulse move % for Flag Limit
DEFAULT_IMPULSE_THRESHOLD_PCT: float = 1.0

# Pattern-specific score multipliers
PATTERN_SCORE_BONUS: dict[PatternType, float] = {
    PatternType.FTR: 1.2,
    PatternType.FL: 1.0,
    PatternType.QM: 1.3,
}


# ═══════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════


@dataclass
class Zone:
    """A detected supply or demand zone with full scoring metadata.

    Attributes:
        zone_type: SUPPLY or DEMAND.
        pattern: FTR, FL, or QM.
        timeframe: String label (``"1w"``, ``"1d"``, ``"4h"``, ``"1h"``, ``"15m"``).
        price_low: Lower boundary of the zone.
        price_high: Upper boundary of the zone.
        midpoint: (price_low + price_high) / 2.
        score: Computed power score (timeframe_weight × freshness × confluence
               × pattern_bonus).
        freshness: UNTOUCHED, TOUCHED_ONCE, TOUCHED_TWICE, or EXHAUSTED.
        touch_count: How many times price has visited the zone.
        confluence_count: Number of timeframes with overlapping zones.
        candle_index: Index of the origin candle in the source DataFrame.
        timestamp_ms: UTC timestamp of the origin candle (milliseconds).
        barrier_price: For FTR zones — the broken swing level. Optional.
    """

    zone_type: ZoneType
    pattern: PatternType
    timeframe: str
    price_low: float
    price_high: float
    midpoint: float
    score: float
    freshness: Freshness
    touch_count: int
    confluence_count: int
    candle_index: int
    timestamp_ms: int
    barrier_price: Optional[float] = None

    @property
    def width_pct(self) -> float:
        """Zone width as percentage of midpoint."""
        if self.midpoint == 0:
            return 0.0
        return (self.price_high - self.price_low) / self.midpoint * 100

    def contains_price(self, price: float) -> bool:
        """Return True if *price* is within zone boundaries (inclusive)."""
        return self.price_low <= price <= self.price_high

    def distance_to_price(self, price: float) -> float:
        """Distance from *price* to zone midpoint, as percentage of midpoint."""
        if self.midpoint == 0:
            return float("inf")
        return abs(price - self.midpoint) / self.midpoint * 100

    def __repr__(self) -> str:
        return (
            f"Zone({self.zone_type.value}, {self.pattern.value}, "
            f"{self.timeframe}, "
            f"{self.price_low:.2f}–{self.price_high:.2f}, "
            f"score={self.score:.1f})"
        )

    def __str__(self) -> str:
        return (
            f"{self.zone_type.value:7s} {self.pattern.value:4s} "
            f"[{self.timeframe:4s}] "
            f"${self.price_low:,.0f}–${self.price_high:,.0f} "
            f"score={self.score:.1f} "
            f"fresh={self.freshness.value} "
            f"conf={self.confluence_count}TF"
        )


# ── Internal Candidate Dataclasses ─────────────────────────────────


@dataclass
class _FTRCandidate:
    """Raw FTR pattern before zone construction."""

    direction: str  # "bullish" | "bearish"
    barrier_price: float
    zone_low: float
    zone_high: float
    candle_index: int
    timestamp_ms: int
    swing_index: int


@dataclass
class _FLCandidate:
    """Raw Flag Limit pattern before zone construction."""

    direction: str  # "bullish" | "bearish"
    base_low: float
    base_high: float
    candle_index: int
    timestamp_ms: int


@dataclass
class _QMCandidate:
    """Raw Quasimodo pattern before zone construction."""

    direction: str  # "bullish" | "bearish"
    entry_low: float
    entry_high: float
    candle_index: int
    timestamp_ms: int
    sequence: list[str] = field(default_factory=list)


# ── Internal Swing Point ───────────────────────────────────────────


@dataclass
class _SwingPoint:
    """A swing high or low with structure label."""

    index: int
    sw_type: str  # "high" | "low"
    price: float
    label: str = ""  # HH, HL, LH, LL, START, SWITCH


# ═══════════════════════════════════════════════════════════════════
# Zone Detector
# ═══════════════════════════════════════════════════════════════════


class ZoneDetector:
    """Multi-timeframe RTM zone detection engine.

    Detects FTR, FL, and QM patterns on one or more timeframes, scores them
    by timeframe weight, freshness, and cross-TF confluence, and returns a
    ranked list of :class:`Zone` objects.

    Parameters:
        lookback: Number of bars for swing high/low detection (default 3).
        impulse_threshold_pct: Minimum % move to qualify a Flag Limit
            impulse (default 1.0).
        max_zones_per_tf: Maximum zones to retain per timeframe after
            scoring (default 20).
        confluence_overlap_pct: Maximum midpoint distance (% of midpoint)
            for two zones from different TFs to be considered overlapping
            (default 0.5).
    """

    def __init__(
        self,
        lookback: int = 3,
        impulse_threshold_pct: float = 1.0,
        max_zones_per_tf: int = 20,
        confluence_overlap_pct: float = 0.5,
    ) -> None:
        self.lookback = lookback
        self.impulse_threshold_pct = impulse_threshold_pct
        self.max_zones_per_tf = max_zones_per_tf
        self.confluence_overlap_pct = confluence_overlap_pct

        # Stored after detection
        self.zones: list[Zone] = []

    # ── Public API ──────────────────────────────────────────────────

    def detect_all(self, dfs: dict[str, pd.DataFrame]) -> list[Zone]:
        """Run detection on every timeframe, score, and rank.

        Args:
            dfs: Mapping of timeframe label → OHLCV DataFrame.
                Required columns: ``open``, ``high``, ``low``, ``close``.
                Optional: ``timestamp`` (int64 ms UTC).

        Returns:
            List of :class:`Zone` objects sorted by score descending.
        """
        all_candidates: list[Zone] = []

        for tf, df in dfs.items():
            if df is None or len(df) < 10:
                logger.debug("Skipping %s: insufficient data (%s rows)", tf, len(df) if df is not None else 0)
                continue
            zones = self._detect_on_tf(df, tf)
            all_candidates.extend(zones)
            logger.info("%s: %d raw zones detected", tf, len(zones))

        # Apply confluence bonus across timeframes
        all_candidates = self._apply_confluence(all_candidates)

        # Determine latest price for freshness scoring
        latest_price = self._latest_price(dfs)
        if latest_price is not None:
            all_candidates = self._update_freshness(all_candidates, latest_price)

        # Final sort by score descending
        all_candidates.sort(key=lambda z: z.score, reverse=True)
        self.zones = all_candidates
        return all_candidates

    def get_nearby_zones(
        self, price: float, distance_pct: float = 2.0
    ) -> list[Zone]:
        """Return zones whose midpoint is within *distance_pct*% of *price*.

        Args:
            price: Reference price (e.g. current close).
            distance_pct: Maximum distance as percentage of zone midpoint.

        Returns:
            List of nearby :class:`Zone` objects, sorted by distance.
        """
        nearby = [z for z in self.zones if z.distance_to_price(price) <= distance_pct]
        nearby.sort(key=lambda z: z.distance_to_price(price))
        return nearby

    def get_supply_zones(self, min_score: float = 2.0) -> list[Zone]:
        """Return supply (resistance) zones at or above *min_score*."""
        return [
            z
            for z in self.zones
            if z.zone_type == ZoneType.SUPPLY and z.score >= min_score
        ]

    def get_demand_zones(self, min_score: float = 2.0) -> list[Zone]:
        """Return demand (support) zones at or above *min_score*."""
        return [
            z
            for z in self.zones
            if z.zone_type == ZoneType.DEMAND and z.score >= min_score
        ]

    def summary(self) -> dict:
        """Return a detection summary dict."""
        patterns = {pt: 0 for pt in PatternType}
        tfs: dict[str, int] = {}
        for z in self.zones:
            patterns[z.pattern] = patterns.get(z.pattern, 0) + 1
            tfs[z.timeframe] = tfs.get(z.timeframe, 0) + 1
        return {
            "total_zones": len(self.zones),
            "by_pattern": {p.value: c for p, c in patterns.items()},
            "by_timeframe": tfs,
            "top_score": self.zones[0].score if self.zones else 0.0,
        }

    # ── Per-Timeframe Detection ─────────────────────────────────────

    def _detect_on_tf(self, df: pd.DataFrame, tf: str) -> list[Zone]:
        """Run all pattern detectors on a single timeframe and return Zone objects."""
        n = len(df)
        if n < 10:
            return []

        # Extract numpy arrays for speed
        opens = df["open"].to_numpy(dtype=np.float64)
        highs = df["high"].to_numpy(dtype=np.float64)
        lows = df["low"].to_numpy(dtype=np.float64)
        closes = df["close"].to_numpy(dtype=np.float64)
        timestamps = (
            df["timestamp"].to_numpy(dtype=np.int64)
            if "timestamp" in df.columns
            else np.zeros(n, dtype=np.int64)
        )

        # Find swing points (used by FTR and QM)
        swings = self._find_swings(highs, lows)

        # Label swing structure (HH, HL, LH, LL)
        self._label_swings(swings)

        # Detect patterns
        ftr_candidates = self._detect_ftr(highs, lows, closes, timestamps, swings, tf)
        fl_candidates = self._detect_fl(opens, highs, lows, closes, timestamps, tf)
        qm_candidates = self._detect_qm(highs, lows, closes, timestamps, swings, tf)

        # Convert candidates to Zone objects with base scores
        all_zones: list[Zone] = []
        all_zones.extend(self._ftr_candidates_to_zones(ftr_candidates, tf))
        all_zones.extend(self._fl_candidates_to_zones(fl_candidates, tf))
        all_zones.extend(self._qm_candidates_to_zones(qm_candidates, tf))

        # Keep top N per timeframe
        all_zones.sort(key=lambda z: z.score, reverse=True)
        return all_zones[: self.max_zones_per_tf]

    # ── Swing Detection ─────────────────────────────────────────────

    def _find_swings(self, highs: np.ndarray, lows: np.ndarray) -> list[_SwingPoint]:
        """Find swing highs and lows using local extrema.

        A swing high is a bar whose high is greater than the highs of the
        *lookback* bars on either side. Similarly for swing lows.
        """
        lb = self.lookback
        n = len(highs)
        swings: list[_SwingPoint] = []

        # Use sliding window comparisons with numpy
        for i in range(lb, n - lb):
            # Swing high check
            if np.all(highs[i] > highs[i - lb : i]) and np.all(
                highs[i] > highs[i + 1 : i + lb + 1]
            ):
                swings.append(_SwingPoint(i, "high", highs[i]))

            # Swing low check
            if np.all(lows[i] < lows[i - lb : i]) and np.all(
                lows[i] < lows[i + 1 : i + lb + 1]
            ):
                swings.append(_SwingPoint(i, "low", lows[i]))

        # Sort by index for structure labeling
        swings.sort(key=lambda s: s.index)
        return swings

    def _label_swings(self, swings: list[_SwingPoint]) -> None:
        """Label swing points with HH, HL, LH, LL structure."""
        for i, sw in enumerate(swings):
            if i == 0:
                sw.label = "START"
                continue
            prev = swings[i - 1]
            if sw.sw_type == "high" and prev.sw_type == "high":
                sw.label = "HH" if sw.price > prev.price else "LH"
            elif sw.sw_type == "low" and prev.sw_type == "low":
                sw.label = "HL" if sw.price > prev.price else "LL"
            else:
                sw.label = "SWITCH"

    # ── FTR Detection ───────────────────────────────────────────────

    def _detect_ftr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        timestamps: np.ndarray,
        swings: list[_SwingPoint],
        tf: str,
    ) -> list[_FTRCandidate]:
        """Detect Fail-to-Return patterns using a 5-step state machine.

        Steps for bullish FTR (old resistance → new support):
          1. Identify a swing high (barrier).
          2. Price breaks above the barrier.
          3. Price retraces but stays *above* the barrier (fail to return).
          4. Price continues higher from the retrace.
          5. Define the zone around the retrace area.

        Bearish FTR is the mirror: swing low → break below → retrace stays
        below → continuation lower.
        """
        candidates: list[_FTRCandidate] = []
        n = len(highs)

        for sw in swings:
            barrier = sw.price
            bi = sw.index

            if sw.sw_type == "high":
                zone = self._check_bullish_ftr(highs, lows, closes, timestamps, sw, n)
                if zone:
                    candidates.append(zone)
            elif sw.sw_type == "low":
                zone = self._check_bearish_ftr(highs, lows, closes, timestamps, sw, n)
                if zone:
                    candidates.append(zone)

        return candidates

    def _check_bullish_ftr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        timestamps: np.ndarray,
        swing: _SwingPoint,
        n: int,
    ) -> Optional[_FTRCandidate]:
        """Check for bullish FTR: resistance becomes support."""
        barrier = swing.price
        bi = swing.index

        # Step 2: Find first candle that breaks above the barrier
        bi_break: Optional[int] = None
        search_end = min(bi + 30, n)
        for j in range(bi + 1, search_end):
            if highs[j] > barrier:
                bi_break = j
                break
        if bi_break is None:
            return None

        # Step 3: After the break, price must retrace but STAY ABOVE barrier
        post_end = min(bi_break + 20, n)
        post_range = slice(bi_break + 1, post_end)
        if post_end - (bi_break + 1) < 3:
            return None

        retrace_low = float(np.min(lows[post_range]))
        # Price returned to or below barrier → NOT a valid FTR
        if retrace_low <= barrier * 0.999:
            return None

        # Find retrace candle index
        post_lows = lows[post_range]
        r_idx = bi_break + 1 + int(np.argmin(post_lows))

        # Step 4 & 5: Continuation above the retrace high
        cont_end = min(r_idx + 15, n)
        cont_range = slice(r_idx + 1, cont_end)
        if cont_end - (r_idx + 1) == 0:
            return None

        cont_max = float(np.max(highs[cont_range]))
        retrace_high_at_idx = highs[r_idx]
        if cont_max <= max(retrace_low, retrace_high_at_idx):
            return None

        # Define zone around the retrace area
        near_start = max(0, r_idx - 2)
        near_end = min(n, r_idx + 5)
        near_range = slice(near_start, near_end)

        return _FTRCandidate(
            direction="bullish",
            barrier_price=barrier,
            zone_low=float(np.min(lows[near_range])),
            zone_high=float(np.max(highs[near_range])),
            candle_index=r_idx,
            timestamp_ms=int(timestamps[r_idx]) if timestamps[r_idx] != 0 else 0,
            swing_index=bi,
        )

    def _check_bearish_ftr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        timestamps: np.ndarray,
        swing: _SwingPoint,
        n: int,
    ) -> Optional[_FTRCandidate]:
        """Check for bearish FTR: support becomes resistance."""
        barrier = swing.price
        bi = swing.index

        # Step 2: Find first candle that breaks below the barrier
        bi_break: Optional[int] = None
        search_end = min(bi + 30, n)
        for j in range(bi + 1, search_end):
            if lows[j] < barrier:
                bi_break = j
                break
        if bi_break is None:
            return None

        # Step 3: After the break, price must retrace but STAY BELOW barrier
        post_end = min(bi_break + 20, n)
        post_range = slice(bi_break + 1, post_end)
        if post_end - (bi_break + 1) < 3:
            return None

        retrace_high = float(np.max(highs[post_range]))
        if retrace_high >= barrier * 1.001:
            return None

        # Find retrace candle index
        post_highs = highs[post_range]
        r_idx = bi_break + 1 + int(np.argmax(post_highs))

        # Step 4 & 5: Continuation below retrace low
        cont_end = min(r_idx + 15, n)
        cont_range = slice(r_idx + 1, cont_end)
        if cont_end - (r_idx + 1) == 0:
            return None

        cont_min = float(np.min(lows[cont_range]))
        retrace_low_at_idx = lows[r_idx]
        if cont_min >= min(retrace_high, retrace_low_at_idx):
            return None

        # Define zone around the retrace area
        near_start = max(0, r_idx - 2)
        near_end = min(n, r_idx + 5)
        near_range = slice(near_start, near_end)

        return _FTRCandidate(
            direction="bearish",
            barrier_price=barrier,
            zone_low=float(np.min(lows[near_range])),
            zone_high=float(np.max(highs[near_range])),
            candle_index=r_idx,
            timestamp_ms=int(timestamps[r_idx]) if timestamps[r_idx] != 0 else 0,
            swing_index=bi,
        )

    # ── Flag Limit Detection ────────────────────────────────────────

    def _detect_fl(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        timestamps: np.ndarray,
        tf: str,
    ) -> list[_FLCandidate]:
        """Detect Flag Limit patterns: base candle → impulse move.

        A base candle has a small body relative to its range. An impulse
        is a 1–3 candle move that exceeds the threshold percentage.
        """
        candidates: list[_FLCandidate] = []
        n = len(closes)
        base_threshold = BASE_BODY_THRESHOLDS.get(tf, 1.0)

        for i in range(self.lookback, n - 3):
            c_open = opens[i]
            c_close = closes[i]
            c_high = highs[i]
            c_low = lows[i]

            if c_open <= 0:
                continue

            # Body ratio check — base candles have small bodies
            body = abs(c_close - c_open)
            total_range = c_high - c_low
            body_ratio = body / total_range if total_range > 0 else 1.0

            # Also check body % of price for higher TFs
            body_pct = body / c_open * 100

            # Must pass either body_ratio OR body_pct threshold
            if body_ratio >= BASE_BODY_RATIO and body_pct > base_threshold:
                continue

            # Check for impulse in the next 1–3 candles
            for look_ahead in range(1, min(4, n - i)):
                future_idx = i + look_ahead
                if closes[future_idx] <= 0 or closes[i] <= 0:
                    continue

                move_pct = (
                    (closes[future_idx] - closes[i]) / closes[i] * 100
                )

                if abs(move_pct) >= self.impulse_threshold_pct:
                    if move_pct > 0:
                        # Bullish impulse → base is DEMAND zone
                        candidates.append(
                            _FLCandidate(
                                direction="bullish",
                                base_low=c_low,
                                base_high=max(c_open, c_close),
                                candle_index=i,
                                timestamp_ms=int(timestamps[i]) if timestamps[i] != 0 else 0,
                            )
                        )
                    else:
                        # Bearish impulse → base is SUPPLY zone
                        candidates.append(
                            _FLCandidate(
                                direction="bearish",
                                base_low=min(c_open, c_close),
                                base_high=c_high,
                                candle_index=i,
                                timestamp_ms=int(timestamps[i]) if timestamps[i] != 0 else 0,
                            )
                        )
                    break  # Found impulse, move to next candle

        return candidates

    # ── Quasimodo Detection ─────────────────────────────────────────

    def _detect_qm(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        timestamps: np.ndarray,
        swings: list[_SwingPoint],
        tf: str,
    ) -> list[_QMCandidate]:
        """Detect Quasimodo patterns from swing sequence.

        Bearish QM: SH → SL → SH (higher) → break below SL
            (Higher high traps buyers, then price reverses through prior low.)

        Bullish QM: SL → SH → SL (lower) → break above SH
            (Lower low traps sellers, then price reverses through prior high.)
        """
        candidates: list[_QMCandidate] = []
        if len(swings) < 4:
            return candidates

        for i in range(2, len(swings) - 1):
            s0 = swings[i - 2]
            s1 = swings[i - 1]
            s2 = swings[i]
            s3 = swings[i + 1]

            # ── Bearish QM ──
            # SH1 → SL1 → SH2 (higher than SH1) → Break below SL1
            if (
                s0.sw_type == "high"
                and s1.sw_type == "low"
                and s2.sw_type == "high"
                and s2.price > s0.price
                and s3.sw_type == "low"
                and s3.price < s1.price
            ):
                # Entry zone: between SH1 and SL1
                candidates.append(
                    _QMCandidate(
                        direction="bearish",
                        entry_low=min(s0.price, s1.price),
                        entry_high=max(s0.price, s1.price),
                        candle_index=s3.index,
                        timestamp_ms=int(timestamps[s3.index]) if timestamps[s3.index] != 0 else 0,
                        sequence=[
                            f"SH1=${s0.price:.0f}",
                            f"SL1=${s1.price:.0f}",
                            f"SH2=${s2.price:.0f}",
                            f"BreakSL1=${s3.price:.0f}",
                        ],
                    )
                )

            # ── Bullish QM ──
            # SL1 → SH1 → SL2 (lower than SL1) → Break above SH1
            if (
                s0.sw_type == "low"
                and s1.sw_type == "high"
                and s2.sw_type == "low"
                and s2.price < s0.price
                and s3.sw_type == "high"
                and s3.price > s1.price
            ):
                candidates.append(
                    _QMCandidate(
                        direction="bullish",
                        entry_low=min(s0.price, s1.price),
                        entry_high=max(s0.price, s1.price),
                        candle_index=s3.index,
                        timestamp_ms=int(timestamps[s3.index]) if timestamps[s3.index] != 0 else 0,
                        sequence=[
                            f"SL1=${s0.price:.0f}",
                            f"SH1=${s1.price:.0f}",
                            f"SL2=${s2.price:.0f}",
                            f"BreakSH1=${s3.price:.0f}",
                        ],
                    )
                )

        return candidates

    # ── Candidate → Zone Conversion ─────────────────────────────────

    def _ftr_candidates_to_zones(
        self, candidates: list[_FTRCandidate], tf: str
    ) -> list[Zone]:
        """Convert FTR candidates to Zone objects with base scores."""
        zones: list[Zone] = []
        for c in candidates:
            midpoint = (c.zone_low + c.zone_high) / 2
            zone_type = ZoneType.DEMAND if c.direction == "bullish" else ZoneType.SUPPLY
            base_score = TF_WEIGHTS.get(tf, 1) * PATTERN_SCORE_BONUS[PatternType.FTR]

            zones.append(
                Zone(
                    zone_type=zone_type,
                    pattern=PatternType.FTR,
                    timeframe=tf,
                    price_low=c.zone_low,
                    price_high=c.zone_high,
                    midpoint=midpoint,
                    score=base_score,
                    freshness=Freshness.UNTOUCHED,
                    touch_count=0,
                    confluence_count=1,
                    candle_index=c.candle_index,
                    timestamp_ms=c.timestamp_ms,
                    barrier_price=c.barrier_price,
                )
            )
        return zones

    def _fl_candidates_to_zones(
        self, candidates: list[_FLCandidate], tf: str
    ) -> list[Zone]:
        """Convert FL candidates to Zone objects with base scores."""
        zones: list[Zone] = []
        for c in candidates:
            midpoint = (c.base_low + c.base_high) / 2
            zone_type = ZoneType.DEMAND if c.direction == "bullish" else ZoneType.SUPPLY
            base_score = TF_WEIGHTS.get(tf, 1) * PATTERN_SCORE_BONUS[PatternType.FL]

            zones.append(
                Zone(
                    zone_type=zone_type,
                    pattern=PatternType.FL,
                    timeframe=tf,
                    price_low=c.base_low,
                    price_high=c.base_high,
                    midpoint=midpoint,
                    score=base_score,
                    freshness=Freshness.UNTOUCHED,
                    touch_count=0,
                    confluence_count=1,
                    candle_index=c.candle_index,
                    timestamp_ms=c.timestamp_ms,
                )
            )
        return zones

    def _qm_candidates_to_zones(
        self, candidates: list[_QMCandidate], tf: str
    ) -> list[Zone]:
        """Convert QM candidates to Zone objects with base scores."""
        zones: list[Zone] = []
        for c in candidates:
            midpoint = (c.entry_low + c.entry_high) / 2
            # Note: bearish QM → short → SUPPLY zone at the break
            #       bullish QM → long → DEMAND zone at the break
            zone_type = ZoneType.SUPPLY if c.direction == "bearish" else ZoneType.DEMAND
            base_score = TF_WEIGHTS.get(tf, 1) * PATTERN_SCORE_BONUS[PatternType.QM]

            zones.append(
                Zone(
                    zone_type=zone_type,
                    pattern=PatternType.QM,
                    timeframe=tf,
                    price_low=c.entry_low,
                    price_high=c.entry_high,
                    midpoint=midpoint,
                    score=base_score,
                    freshness=Freshness.UNTOUCHED,
                    touch_count=0,
                    confluence_count=1,
                    candle_index=c.candle_index,
                    timestamp_ms=c.timestamp_ms,
                )
            )
        return zones

    # ── Confluence Scoring ──────────────────────────────────────────

    def _apply_confluence(self, zones: list[Zone]) -> list[Zone]:
        """Find zones that overlap across different timeframes and apply
        confluence bonus multipliers.

        Two zones from different TFs are considered overlapping if their
        midpoints are within ``confluence_overlap_pct`` % of each other.
        """
        if not zones:
            return zones

        overlap_tol = self.confluence_overlap_pct / 100.0

        for i, z1 in enumerate(zones):
            confluence_tfs: set[str] = {z1.timeframe}
            for j, z2 in enumerate(zones):
                if i == j:
                    continue
                if z1.zone_type != z2.zone_type:
                    continue
                if z1.timeframe == z2.timeframe:
                    continue
                if z1.midpoint <= 0 or z2.midpoint <= 0:
                    continue
                # Check midpoint overlap within tolerance
                diff_pct = abs(z1.midpoint - z2.midpoint) / z1.midpoint
                if diff_pct <= overlap_tol:
                    confluence_tfs.add(z2.timeframe)

            n_tfs = len(confluence_tfs)
            z1.confluence_count = n_tfs
            bonus = CONFLUENCE_BONUS.get(
                min(n_tfs, max(CONFLUENCE_BONUS.keys())), 2.0
            )
            z1.score *= bonus

        return zones

    # ── Freshness Scoring ───────────────────────────────────────────

    def _update_freshness(self, zones: list[Zone], current_price: float) -> list[Zone]:
        """Update freshness based on how close current price is to each zone.

        Zones near the current price are considered touched. Distant zones
        remain fresh. Full implementation would track every candle's visit
        to each zone; this is a simplified approximation.
        """
        if current_price <= 0:
            return zones

        for z in zones:
            dist = z.distance_to_price(current_price)

            if dist < 0.2:
                z.freshness = Freshness.TOUCHED_TWICE
                z.touch_count = 2
            elif dist < 0.5:
                z.freshness = Freshness.TOUCHED_ONCE
                z.touch_count = 1
            # else: stays UNTOUCHED

            # Apply freshness multiplier
            z.score *= FRESHNESS_MULTIPLIER[z.freshness]

        return zones

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _latest_price(dfs: dict[str, pd.DataFrame]) -> Optional[float]:
        """Get the latest close price from the highest-resolution TF available."""
        for tf in ["15m", "1h", "4h", "1d", "1w"]:
            if tf in dfs and dfs[tf] is not None and len(dfs[tf]) > 0:
                df = dfs[tf]
                if "close" in df.columns:
                    return float(df["close"].iloc[-1])
        return None
