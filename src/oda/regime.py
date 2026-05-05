"""
Oda Regime Detector — Market regime classification as signal filter
===================================================================

Identifies the current market regime using higher-timeframe data (1h):
- Trend direction via 200-period EMA
- Trend strength via ADX(14)
- Volatility regime via ATR(14) vs its SMA(50)

The regime acts as a gate on the signal pipeline: trades that go against
the dominant trend are filtered out. This prevents the RTM engulf strategy
from taking counter-trend positions in strongly trending markets, and
reduces trade frequency in choppy/ranging conditions.

Regime classification:
  BULLISH   — ADX > 20, price > 200 EMA, +DI > -DI  → only LONG signals
  BEARISH   — ADX > 20, price < 200 EMA, -DI > +DI  → only SHORT signals
  RANGING   — ADX < 20                                → no signals (or filtered)
  TRANSITION — everything else                        → allow both directions

Usage:
    from oda.regime import RegimeDetector, RegimeConfig

    config = RegimeConfig.from_env()
    detector = RegimeDetector(config)
    regime = detector.update(high, low, close)  # numpy arrays from 1h data
    if detector.allows_direction(signal_direction):
        # take trade
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("oda.regime")


# ── Enums ─────────────────────────────────────────────────────────────


class RegimeType(Enum):
    """Market regime classification."""
    BULLISH = "BULLISH"       # Trending up — only LONG signals
    BEARISH = "BEARISH"       # Trending down — only SHORT signals
    RANGING = "RANGING"       # Choppy/low ADX — filter all or high bar
    TRANSITION = "TRANSITION" # Between regimes — allow both


class VolRegime(Enum):
    """Volatility regime from ATR."""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


# ── Config ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RegimeConfig:
    """Regime detection parameters, all overridable via env vars.

    Environment variables:
        ODA_REGIME_ENABLED: Enable regime filter (default "true").
        ODA_REGIME_EMA_PERIOD: EMA period for trend detection (default 200).
        ODA_REGIME_ADX_PERIOD: ADX period (default 14).
        ODA_REGIME_ADX_TRENDING: ADX threshold for trending regime (default 20).
        ODA_REGIME_ALLOW_RANGING: Allow trades in ranging regime (default "false").
        ODA_REGIME_CONFIDENCE_BOOST: Confidence multiplier for regime-aligned trades (default 1.1).
        ODA_REGIME_ATR_PERIOD: ATR period for volatility regime (default 14).
        ODA_REGIME_ATR_SMA_PERIOD: ATR SMA period (default 50).
        ODA_REGIME_ATR_HIGH_MULT: ATR/SMA multiplier for high vol threshold (default 1.5).
        ODA_REGIME_ATR_LOW_MULT: ATR/SMA multiplier for low vol threshold (default 0.7).
    """
    enabled: bool = True
    ema_period: int = 200
    adx_period: int = 14
    adx_trending_threshold: float = 20.0
    allow_ranging: bool = False
    confidence_boost: float = 1.1        # Confidence multiplier for regime-aligned trades
    atr_period: int = 14
    atr_sma_period: int = 50
    atr_high_mult: float = 1.5           # ATR / ATR_SMA above this = HIGH vol regime
    atr_low_mult: float = 0.7            # ATR / ATR_SMA below this = LOW vol regime

    @classmethod
    def from_env(cls) -> "RegimeConfig":
        import os
        return cls(
            enabled=_env_bool("ODA_REGIME_ENABLED", True),
            ema_period=int(os.environ.get("ODA_REGIME_EMA_PERIOD", "200")),
            adx_period=int(os.environ.get("ODA_REGIME_ADX_PERIOD", "14")),
            adx_trending_threshold=float(os.environ.get("ODA_REGIME_ADX_TRENDING", "20.0")),
            allow_ranging=_env_bool("ODA_REGIME_ALLOW_RANGING", False),
            confidence_boost=float(os.environ.get("ODA_REGIME_CONFIDENCE_BOOST", "1.1")),
            atr_period=int(os.environ.get("ODA_REGIME_ATR_PERIOD", "14")),
            atr_sma_period=int(os.environ.get("ODA_REGIME_ATR_SMA_PERIOD", "50")),
            atr_high_mult=float(os.environ.get("ODA_REGIME_ATR_HIGH_MULT", "1.5")),
            atr_low_mult=float(os.environ.get("ODA_REGIME_ATR_LOW_MULT", "0.7")),
        )


def _env_bool(key: str, default: bool) -> bool:
    import os
    v = os.environ.get(key)
    if v is None:
        return default
    return v.lower() in ("true", "1", "yes")


# ── Regime Detector ───────────────────────────────────────────────────


class RegimeDetector:
    """Classifies market regime from higher-TF OHLCV data (typically 1h).

    Call ``update()`` with 1h price arrays to refresh regime state.
    Then use ``allows_direction()`` to filter signals.
    """

    def __init__(self, config: Optional[RegimeConfig] = None) -> None:
        self.config = config or RegimeConfig()

        # Current regime state
        self.regime: RegimeType = RegimeType.RANGING
        self.vol_regime: VolRegime = VolRegime.NORMAL
        self.adx: float = 0.0
        self.plus_di: float = 0.0
        self.minus_di: float = 0.0
        self.ema_200: float = 0.0
        self.price_vs_ema_pct: float = 0.0  # +% = above EMA, -% = below
        self.atr_value: float = 0.0
        self.atr_ratio: float = 0.0  # ATR / ATR_SMA

        # Cache for ADX/ATR computation (avoid recompute on unchanged data)
        self._cache_n: int = 0
        self._cache_adx: float = 0.0
        self._cache_plus_di: float = 0.0
        self._cache_minus_di: float = 0.0
        self._cache_atr: float = 0.0
        self._cache_atr_sma: float = 0.0

        logger.debug("RegimeDetector initialized (EMA=%d, ADX=%d, threshold=%.1f)",
                      self.config.ema_period, self.config.adx_period,
                      self.config.adx_trending_threshold)

    # ── Public API ─────────────────────────────────────────────────────

    def update(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> RegimeType:
        """Update regime classification from OHLCV arrays.

        Args:
            high: Array of high prices (1h timeframe).
            low: Array of low prices (1h timeframe).
            close: Array of close prices (1h timeframe).

        Returns:
            Current RegimeType.
        """
        n = len(close)
        if n < max(self.config.ema_period, self.config.adx_period * 2):
            logger.warning("Insufficient data for regime detection (%d bars)", n)
            self.regime = RegimeType.RANGING
            return self.regime

        if not self.config.enabled:
            self.regime = RegimeType.TRANSITION
            return self.regime

        # Compute indicators (cached if data length unchanged)
        if n != self._cache_n:
            self._compute_indicators(high, low, close)
            self._cache_n = n

        self.adx = self._cache_adx
        self.plus_di = self._cache_plus_di
        self.minus_di = self._cache_minus_di
        self.atr_value = self._cache_atr
        self.atr_ratio = self._cache_atr / self._cache_atr_sma if self._cache_atr_sma > 0 else 1.0

        # EMA on close
        s = pd.Series(close)
        self.ema_200 = float(s.ewm(span=self.config.ema_period, adjust=False).mean().iloc[-1])
        self.price_vs_ema_pct = (close[-1] - self.ema_200) / self.ema_200 * 100

        # Volatility regime
        if self.atr_ratio >= self.config.atr_high_mult:
            self.vol_regime = VolRegime.HIGH
        elif self.atr_ratio <= self.config.atr_low_mult:
            self.vol_regime = VolRegime.LOW
        else:
            self.vol_regime = VolRegime.NORMAL

        # Classify regime
        self.regime = self._classify()

        logger.debug("Regime: %s | ADX=%.1f +DI=%.1f -DI=%.1f | "
                     "EMA200=$%.0f price_vs_ema=%+.2f%% | ATR=%.2f ratio=%.2f vol=%s",
                     self.regime.value, self.adx, self.plus_di, self.minus_di,
                     self.ema_200, self.price_vs_ema_pct,
                     self.atr_value, self.atr_ratio, self.vol_regime.value)

        return self.regime

    def allows_direction(self, direction_str: str) -> bool:
        """Check if the current regime allows a trade in *direction*.

        Args:
            direction_str: "LONG" or "SHORT".

        Returns:
            True if the trade direction is compatible with the current regime.
        """
        if not self.config.enabled:
            return True

        if self.regime == RegimeType.RANGING:
            return self.config.allow_ranging

        if self.regime == RegimeType.BULLISH:
            return direction_str == "LONG"

        if self.regime == RegimeType.BEARISH:
            return direction_str == "SHORT"

        # TRANSITION — allow both
        return True

    def get_confidence_multiplier(self, direction_str: str) -> float:
        """Return a confidence multiplier for a trade in *direction*.

        Regime-aligned trades get a boost. Counter-regime trades get a penalty.
        Ranging regime heavily penalizes all trades.
        """
        if not self.config.enabled:
            return 1.0

        if self.regime == RegimeType.RANGING:
            return 0.5  # Heavy penalty in ranging markets

        if self.regime == RegimeType.TRANSITION:
            return 0.9  # Slight penalty in transition

        # Trending — aligned get boost, opposite get penalty
        aligned = (
            (self.regime == RegimeType.BULLISH and direction_str == "LONG") or
            (self.regime == RegimeType.BEARISH and direction_str == "SHORT")
        )
        if aligned:
            return self.config.confidence_boost
        else:
            return 0.6  # Penalty for counter-trend

    def summary(self) -> dict:
        """Return current regime as a dict for logging/reporting."""
        return {
            "regime": self.regime.value,
            "adx": round(self.adx, 1),
            "plus_di": round(self.plus_di, 1),
            "minus_di": round(self.minus_di, 1),
            "ema_200": round(self.ema_200, 2),
            "price_vs_ema_pct": round(self.price_vs_ema_pct, 2),
            "vol_regime": self.vol_regime.value,
            "atr_ratio": round(self.atr_ratio, 2),
        }

    # ── Internal Computation ─────────────────────────────────────────

    def _compute_indicators(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> None:
        """Compute cached ADX, +DI, -DI, ATR, and ATR SMA."""
        n = len(close)
        period = self.config.adx_period

        # ── True Range ────────────────────────────────────────────────
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # ── +DM and -DM ───────────────────────────────────────────────
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # ── Wilder's Smoothing ────────────────────────────────────────
        # First smoothed values: simple sum over first `period` values
        smoothed_tr = np.zeros(n)
        smoothed_plus = np.zeros(n)
        smoothed_minus = np.zeros(n)

        alpha = 1.0 / period

        # Initial SMA for first period
        smoothed_tr[period - 1] = np.sum(tr[1:period])  # tr[1] to tr[period-1] = period-1 values
        # Actually let's use EWMA which is cleaner
        # Use pandas for Wilder's smoothing

        # Better approach: use pandas EWM with alpha=1/period (Wilder's)
        tr_s = pd.Series(tr)
        plus_dm_s = pd.Series(plus_dm)
        minus_dm_s = pd.Series(minus_dm)

        # Wilder's smoothing = EMA with alpha = 1/period
        smoothed_tr_s = tr_s.ewm(alpha=alpha, adjust=False).mean()
        smoothed_plus_s = plus_dm_s.ewm(alpha=alpha, adjust=False).mean()
        smoothed_minus_s = minus_dm_s.ewm(alpha=alpha, adjust=False).mean()

        # ── +DI and -DI ───────────────────────────────────────────────
        atr_arr = smoothed_tr_s.values
        plus_di_arr = np.zeros(n)
        minus_di_arr = np.zeros(n)
        for i in range(period, n):
            if atr_arr[i] > 0:
                plus_di_arr[i] = 100.0 * smoothed_plus_s.iloc[i] / atr_arr[i]
                minus_di_arr[i] = 100.0 * smoothed_minus_s.iloc[i] / atr_arr[i]

        # ── DX and ADX ────────────────────────────────────────────────
        dx = np.zeros(n)
        for i in range(period, n):
            sum_di = plus_di_arr[i] + minus_di_arr[i]
            if sum_di > 0:
                dx[i] = 100.0 * abs(plus_di_arr[i] - minus_di_arr[i]) / sum_di

        dx_s = pd.Series(dx)
        adx_arr = dx_s.ewm(alpha=alpha, adjust=False).mean().values

        # Store cached values
        self._cache_adx = float(adx_arr[-1])
        self._cache_plus_di = float(plus_di_arr[-1])
        self._cache_minus_di = float(minus_di_arr[-1])
        self._cache_atr = float(atr_arr[-1])

        # ATR SMA
        if n >= self.config.atr_sma_period:
            atr_sma = np.mean(atr_arr[-self.config.atr_sma_period:])
        else:
            atr_sma = float(np.mean(atr_arr))
        self._cache_atr_sma = atr_sma

    def _classify(self) -> RegimeType:
        """Classify regime from computed indicators."""
        cfg = self.config

        # Ranging: ADX below threshold
        if self.adx < cfg.adx_trending_threshold:
            return RegimeType.RANGING

        # Trending: ADX above threshold
        price_above_ema = self.price_vs_ema_pct > 0
        di_bullish = self.plus_di > self.minus_di

        if price_above_ema and di_bullish:
            return RegimeType.BULLISH

        if not price_above_ema and not di_bullish:
            return RegimeType.BEARISH

        # Mixed signals — transition
        return RegimeType.TRANSITION
