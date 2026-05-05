"""
Oda Signal Engine — MTF Engulf Entry Confirmation
===================================================

The critical open question from the Hinano-Mori RTM synthesis:
  "Can multi-timeframe engulf confirmation convert the 78% zone hold rate
   into profitable trades?"

When a higher-timeframe zone (1h, 4h) is near price, this engine scans
lower-timeframe candles (15m) for engulf/pinbar confirmation at the zone
boundary before generating an entry signal.

Signal generation:
  1. Identify nearby zones (within distance_pct of current price)
  2. For each zone, scan 15m candles for engulf confirmation
  3. Bullish engulf at demand zone → LONG signal
  4. Bearish engulf at supply zone → SHORT signal
  5. Calculate entry price (engulf candle close), stop loss (zone boundary),
     take profit (2R from entry), and confidence score

Adapted from:
  - Hinano-Mori RTM synthesis (entry timing = the fault line)
  - CryptoMecha01 sweep_detector.py (FVG detection pattern)
  - RTM MTF engulf methodology (IF Myante)

Usage:
    from oda.signals import SignalEngine
    engine = SignalEngine()
    signals = engine.scan(price, zones_1h, df_15m)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from oda.zones import Zone, ZoneType

logger = logging.getLogger("oda.signals")


# ── Enums & Data Classes ────────────────────────────────────────────


class SignalDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class EntrySignal:
    """A confirmed entry signal with entry, stop, and target prices."""
    direction: SignalDirection
    zone: Zone                         # The zone that triggered this signal
    entry_price: float                 # Limit order price
    stop_price: float                  # Stop loss price
    take_profit: float                 # Take profit (2R from entry)
    confidence: float                  # 0.0–1.0 confidence score
    risk_distance_pct: float           # Distance from entry to stop as %
    timestamp_ms: int                  # Candle timestamp
    candle_index: int                  # Index in the 15m DataFrame
    reason: str = ""                   # Human-readable signal reason

    @property
    def risk_reward_ratio(self) -> float:
        """Reward-to-risk ratio (R)."""
        if self.risk_distance_pct == 0 or self.entry_price == 0:
            return 0.0
        reward = abs(self.take_profit - self.entry_price)
        risk = abs(self.entry_price - self.stop_price)
        return reward / risk if risk > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"{self.direction.value} @ ${self.entry_price:,.2f} | "
            f"Stop: ${self.stop_price:,.2f} | TP: ${self.take_profit:,.2f} | "
            f"Conf: {self.confidence:.0%} | {self.reason}"
        )


@dataclass
class PipelineAudit:
    """Instrumentation counters for signal pipeline diagnosis.
    
    Populated during scan() when an audit instance is passed in.
    Tracks how many zones/signals pass/fail each pipeline stage.
    """
    # Zone-level counters
    zones_total: int = 0
    zones_near_price: int = 0
    zones_far_from_price: int = 0

    # Engulf check counters (entered _check_zone for a nearby zone)
    demand_zones_checked: int = 0
    supply_zones_checked: int = 0

    # Bullish engulf rejection reasons
    bullish_price_not_at_zone: int = 0    # l3 > zone.price_low * 1.005
    bullish_breakdown: int = 0             # c3v < zone.price_low
    bullish_low_confidence: int = 0        # confidence < min_confidence

    # Bearish engulf rejection reasons
    bearish_price_not_at_zone: int = 0    # h3 < zone.price_high * 0.995
    bearish_breakout: int = 0              # c3v > zone.price_high
    bearish_low_confidence: int = 0        # confidence < min_confidence

    # Signal pipeline stages
    engulf_matched: int = 0          # _check_engulf returned a signal object (confidence passed)
    confidence_passed: int = 0       # signal.confidence >= min_confidence (in scan, after engulf returned)
    rr_passed: int = 0               # signal.risk_reward_ratio >= min_rr_ratio
    signals_generated: int = 0       # final signals returned by scan()

    # Confidence component contribution counters
    # (incremented when the component contributed to confidence for ANY engulf check,
    #  regardless of whether the total confidence passed the threshold)
    comp_close_direction: int = 0      # is_bullish / is_bearish
    comp_engulfs_prev: int = 0         # c3 body engulfs prior candle
    comp_strong_body: int = 0          # body_ratio >= engulf_body_ratio
    comp_wick_rejection: int = 0       # wick_rejection > 0.3
    comp_vol_surge: int = 0            # vol_c3 / median_vol >= engulf_volume_mult
    comp_zone_score: int = 0           # zone.score >= 5.0

    def merge(self, other: "PipelineAudit") -> None:
        """Merge another audit into this one (for accumulating per-bar audits)."""
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name,
                    getattr(self, field_name) + getattr(other, field_name))


# ── Signal Engine ────────────────────────────────────────────────────


class SignalEngine:
    """Scans zones against 15m candles for MTF engulf entry signals.

    When a higher-TF zone (1h+) is near price, scans 15m candles for
    engulf confirmation at the zone boundary.

    Parameters:
        engulf_body_ratio: Minimum body ratio for engulf candle (default 0.6).
        engulf_volume_mult: Volume multiplier vs 20-period median (default 1.2).
        zone_distance_pct: Max distance from price to zone as % (default 2.0).
        min_confidence: Minimum confidence to generate signal (default 0.3).
    """

    def __init__(
        self,
        engulf_body_ratio: float = 0.6,
        engulf_volume_mult: float = 1.5,
        zone_distance_pct: float = 1.5,
        min_confidence: float = 0.50,
        min_rr_ratio: float = 1.5,
    ) -> None:
        self.engulf_body_ratio = engulf_body_ratio
        self.engulf_volume_mult = engulf_volume_mult
        self.zone_distance_pct = zone_distance_pct
        self.min_confidence = min_confidence
        self.min_rr_ratio = min_rr_ratio

    def scan(
        self,
        current_price: float,
        zones: list[Zone],
        df_15m: pd.DataFrame,
        current_idx: Optional[int] = None,
        audit: Optional[PipelineAudit] = None,
    ) -> list[EntrySignal]:
        """Scan for entry signals at the current price point.

        Args:
            current_price: Current BTC price.
            zones: Active zones (typically from 1h+ timeframes).
            df_15m: 15-minute OHLCV DataFrame.
            current_idx: Current candle index (default: last candle).
            audit: Optional PipelineAudit to populate with counters.

        Returns:
            List of EntrySignal objects, sorted by confidence descending.
        """
        if current_idx is None:
            current_idx = len(df_15m) - 1

        if current_idx < 3:
            return []

        # Filter zones near current price
        nearby = [
            z for z in zones
            if z.distance_to_price(current_price) <= self.zone_distance_pct
        ]

        if audit is not None:
            audit.zones_total = len(zones)
            audit.zones_near_price = len(nearby)
            audit.zones_far_from_price = len(zones) - len(nearby)

        signals: list[EntrySignal] = []
        for zone in nearby:
            signal = self._check_zone(zone, df_15m, current_idx, current_price, audit=audit)

            if audit is not None:
                if signal is not None:
                    audit.engulf_matched += 1
                    audit.confidence_passed += 1

            if signal and signal.confidence >= self.min_confidence:
                if audit is not None and signal is not None:
                    pass  # confidence_passed already counted above
                if signal.risk_reward_ratio >= self.min_rr_ratio:
                    if audit is not None:
                        audit.rr_passed += 1
                        audit.signals_generated += 1
                    signals.append(signal)

        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals

    def _check_zone(
        self,
        zone: Zone,
        df: pd.DataFrame,
        idx: int,
        current_price: float,
        audit: Optional[PipelineAudit] = None,
    ) -> Optional[EntrySignal]:
        """Check if a zone has engulf confirmation on 15m candles."""
        if idx < 3:
            return None

        # Get the latest 3 candles for engulf pattern detection
        c3 = df.iloc[idx]       # Current candle
        c2 = df.iloc[idx - 1]   # Previous
        c1 = df.iloc[idx - 2]   # Two back

        ts = int(c3["timestamp"]) if "timestamp" in df.columns else int(idx)

        if zone.zone_type == ZoneType.DEMAND:
            if audit is not None:
                audit.demand_zones_checked += 1
            return self._check_bullish_engulf(zone, c1, c2, c3, ts, idx, df, audit=audit)
        elif zone.zone_type == ZoneType.SUPPLY:
            if audit is not None:
                audit.supply_zones_checked += 1
            return self._check_bearish_engulf(zone, c1, c2, c3, ts, idx, df, audit=audit)

        return None

    def _check_bullish_engulf(
        self,
        zone: Zone,
        c1: pd.Series,
        c2: pd.Series,
        c3: pd.Series,
        ts: int,
        idx: int,
        df: pd.DataFrame,
        audit: Optional[PipelineAudit] = None,
    ) -> Optional[EntrySignal]:
        """Check for bullish engulf at a demand zone.

        Pattern: c1 is bearish (close < open), c2-c3 engulf bullish.
        The zone boundary (price_low) acts as support.
        Price should be near or at the zone boundary.
        """
        o1, c1v = float(c1["open"]), float(c1["close"])
        o2, c2v = float(c2["open"]), float(c2["close"])
        o3, c3v = float(c3["open"]), float(c3["close"])
        l3 = float(c3["low"])
        h3 = float(c3["high"])

        # Condition 1: Candle 3 low touches or pierces the demand zone
        if l3 > zone.price_low * 1.005:
            if audit is not None:
                audit.bullish_price_not_at_zone += 1
            return None  # Price hasn't reached the zone

        # Condition 2: Candle 3 closes back above the zone (rejection)
        if c3v < zone.price_low:
            if audit is not None:
                audit.bullish_breakdown += 1
            return None  # Breakdown, not a rejection

        # Condition 3: Bullish engulf — c3 body engulfs c1/c2 or shows
        # strong bullish rejection (pinbar with long lower wick)
        body_c3 = abs(c3v - o3)
        range_c3 = h3 - l3 if h3 > l3 else 0.0001
        body_ratio_c3 = body_c3 / range_c3

        # Check for bullish engulf: closed higher than previous candle high
        is_bullish = c3v > o3  # Green candle
        engulfs_prev = c3v > max(o2, c2v)  # Close above prior candle

        # Lower wick = rejection of zone
        lower_wick = min(o3, c3v) - l3
        wick_rejection = lower_wick / range_c3 if range_c3 > 0 else 0

        # Confidence scoring
        confidence = 0.0
        reasons: list[str] = []

        if is_bullish:
            confidence += 0.20
            reasons.append("bullish_close")

        if engulfs_prev:
            confidence += 0.25
            reasons.append("engulfs_prev")

        if body_ratio_c3 >= self.engulf_body_ratio:
            confidence += 0.15
            reasons.append(f"strong_body({body_ratio_c3:.2f})")

        if wick_rejection > 0.3:
            confidence += 0.20
            reasons.append(f"wick_rejection({wick_rejection:.2f})")

        # Volume check
        vol_c3 = float(c3.get("volume", 0))
        if idx >= 20:
            past_vol = df["volume"].iloc[idx - 20 : idx]
            median_vol = past_vol.median() if hasattr(past_vol, "median") else past_vol.median()
            if median_vol > 0 and vol_c3 / median_vol >= self.engulf_volume_mult:
                confidence += 0.10
                reasons.append(f"vol_surge({vol_c3/median_vol:.1f}x)")

        # Zone quality bonus
        confidence += min(zone.score / 20.0, 0.10)
        if zone.score >= 5.0:
            reasons.append(f"zone_score({zone.score:.1f})")

        if audit is not None:
            # Track confidence component contributions
            if is_bullish:
                audit.comp_close_direction += 1
            if engulfs_prev:
                audit.comp_engulfs_prev += 1
            if body_ratio_c3 >= self.engulf_body_ratio:
                audit.comp_strong_body += 1
            if wick_rejection > 0.3:
                audit.comp_wick_rejection += 1
            if idx >= 20:
                past_vol = df["volume"].iloc[idx - 20 : idx]
                median_vol = past_vol.median() if hasattr(past_vol, "median") else past_vol.median()
                if median_vol > 0 and vol_c3 / median_vol >= self.engulf_volume_mult:
                    audit.comp_vol_surge += 1
            if zone.score >= 5.0:
                audit.comp_zone_score += 1

        if confidence < self.min_confidence:
            if audit is not None:
                audit.bullish_low_confidence += 1
            return None

        # Entry at close of confirmation candle
        entry_price = c3v
        # Stop loss below the zone boundary or the candle low, whichever is tighter
        stop_price = min(zone.price_low * 0.998, l3 * 0.998)
        # Take profit at 2R
        risk = entry_price - stop_price
        take_profit = entry_price + (risk * 2.0)

        return EntrySignal(
            direction=SignalDirection.LONG,
            zone=zone,
            entry_price=entry_price,
            stop_price=stop_price,
            take_profit=take_profit,
            confidence=min(confidence, 1.0),
            risk_distance_pct=risk / entry_price * 100 if entry_price > 0 else 0,
            timestamp_ms=ts,
            candle_index=idx,
            reason="+".join(reasons),
        )

    def _check_bearish_engulf(
        self,
        zone: Zone,
        c1: pd.Series,
        c2: pd.Series,
        c3: pd.Series,
        ts: int,
        idx: int,
        df: pd.DataFrame,
        audit: Optional[PipelineAudit] = None,
    ) -> Optional[EntrySignal]:
        """Check for bearish engulf at a supply zone.

        Pattern: c1 is bullish, c2-c3 engulf bearish.
        The zone boundary (price_high) acts as resistance.
        """
        o3, c3v = float(c3["open"]), float(c3["close"])
        o2, c2v = float(c2["open"]), float(c2["close"])
        l3 = float(c3["low"])
        h3 = float(c3["high"])

        # Condition 1: Candle 3 high touches or pierces the supply zone
        if h3 < zone.price_high * 0.995:
            if audit is not None:
                audit.bearish_price_not_at_zone += 1
            return None

        # Condition 2: Candle 3 closes back below the zone (rejection)
        if c3v > zone.price_high:
            if audit is not None:
                audit.bearish_breakout += 1
            return None  # Breakout, not a rejection

        body_c3 = abs(c3v - o3)
        range_c3 = h3 - l3 if h3 > l3 else 0.0001
        body_ratio_c3 = body_c3 / range_c3

        is_bearish = c3v < o3
        engulfs_prev = c3v < min(o2, c2v)

        # Upper wick = rejection of zone
        upper_wick = h3 - max(o3, c3v)
        wick_rejection = upper_wick / range_c3 if range_c3 > 0 else 0

        confidence = 0.0
        reasons: list[str] = []

        if is_bearish:
            confidence += 0.20
            reasons.append("bearish_close")

        if engulfs_prev:
            confidence += 0.25
            reasons.append("engulfs_prev")

        if body_ratio_c3 >= self.engulf_body_ratio:
            confidence += 0.15
            reasons.append(f"strong_body({body_ratio_c3:.2f})")

        if wick_rejection > 0.3:
            confidence += 0.20
            reasons.append(f"wick_rejection({wick_rejection:.2f})")

        vol_c3 = float(c3.get("volume", 0))
        if idx >= 20:
            past_vol = df["volume"].iloc[idx - 20 : idx]
            median_vol = float(np.median(past_vol.values)) if hasattr(past_vol, "values") else float(past_vol.median())
            if median_vol > 0 and vol_c3 / median_vol >= self.engulf_volume_mult:
                confidence += 0.10
                reasons.append(f"vol_surge({vol_c3/median_vol:.1f}x)")

        confidence += min(zone.score / 20.0, 0.10)
        if zone.score >= 5.0:
            reasons.append(f"zone_score({zone.score:.1f})")

        if audit is not None:
            # Track confidence component contributions
            if is_bearish:
                audit.comp_close_direction += 1
            if engulfs_prev:
                audit.comp_engulfs_prev += 1
            if body_ratio_c3 >= self.engulf_body_ratio:
                audit.comp_strong_body += 1
            if wick_rejection > 0.3:
                audit.comp_wick_rejection += 1
            if idx >= 20:
                past_vol = df["volume"].iloc[idx - 20 : idx]
                median_vol = float(np.median(past_vol.values)) if hasattr(past_vol, "values") else float(past_vol.median())
                if median_vol > 0 and vol_c3 / median_vol >= self.engulf_volume_mult:
                    audit.comp_vol_surge += 1
            if zone.score >= 5.0:
                audit.comp_zone_score += 1

        if confidence < self.min_confidence:
            if audit is not None:
                audit.bearish_low_confidence += 1
            return None

        entry_price = c3v
        stop_price = max(zone.price_high * 1.002, h3 * 1.002)
        risk = stop_price - entry_price
        take_profit = entry_price - (risk * 2.0)

        return EntrySignal(
            direction=SignalDirection.SHORT,
            zone=zone,
            entry_price=entry_price,
            stop_price=stop_price,
            take_profit=take_profit,
            confidence=min(confidence, 1.0),
            risk_distance_pct=risk / entry_price * 100 if entry_price > 0 else 0,
            timestamp_ms=ts,
            candle_index=idx,
            reason="+".join(reasons),
        )
