#!/usr/bin/env python3
"""
Oda Signal Pipeline Audit — Phase 2 Task 001
=============================================
Instruments the signal pipeline with per-bar and per-month counters
to diagnose the post-May-2021 signal flatline.

Usage:
    python scripts/audit_pipeline.py

Output: prints monthly summary table + writes to ~/oda/ph2-audit-results.txt
"""

from __future__ import annotations

import copy
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

# Ensure oda is importable
sys.path.insert(0, os.path.expanduser("~/oda/src"))

from oda.config import Settings
from oda.data import fetch_all
from oda.zones import Zone, ZoneDetector, ZoneType
from oda.signals import EntrySignal, SignalDirection, SignalEngine
from oda.backtest import BacktestEngine, BacktestResult, BacktestTrade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("oda.audit")

# Suppress noisy loggers
logging.getLogger("oda.zones").setLevel(logging.WARNING)
logging.getLogger("oda.backtest").setLevel(logging.WARNING)

# ── Audit state ───────────────────────────────────────────────────────

class AuditState:
    """Thread-safe audit state shared between wrapped methods."""
    def __init__(self):
        # Per-call audit (resets each scan())
        self.per_bar: dict[str, int] = defaultdict(int)
        
        # Per-month accumulators
        self.monthly: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Zone rebuild stats
        self.zone_rebuilds: list[dict[str, Any]] = []
        
        # Current month tracking
        self._current_month: str = ""
        
        # Confidence component accumulators
        self.conf_components: dict[str, int] = defaultdict(int)
        
        # Engulf rejection reason accumulators
        self.engulf_rejections: dict[str, int] = defaultdict(int)
        
        # Track attempted bars (price near a zone)
        self.bars_with_near_zones: int = 0
        
        # Which layer rejected (for primary cause detection)
        self.layer_rejections_2022plus: dict[str, int] = defaultdict(int)
        
    def set_month(self, ts_ms: int):
        if ts_ms > 1e10:
            ts_ms = ts_ms / 1000
        month = datetime.fromtimestamp(ts_ms, tz=timezone.utc).strftime("%Y-%m")
        self._current_month = month
        
    def inc(self, key: str, month_key: str | None = None):
        self.per_bar[key] += 1
        m = month_key or self._current_month
        if m:
            self.monthly[m][key] += 1
            
    def commit_bar(self, ts_ms: int):
        """Flush per-bar counters to monthly aggregates and reset."""
        if not self.per_bar:
            return
        m = self._current_month
        for k, v in self.per_bar.items():
            self.monthly[m][k] += v
        self.per_bar.clear()
        
    def get_month(self, ts_ms: int) -> str:
        if ts_ms > 1e10:
            ts_ms = ts_ms / 1000
        return datetime.fromtimestamp(ts_ms, tz=timezone.utc).strftime("%Y-%m")


# Singleton audit state
AUDIT = AuditState()

# ── Wrapped SignalEngine.scan() ───────────────────────────────────────

_original_check_bullish = SignalEngine._check_bullish_engulf
_original_check_bearish = SignalEngine._check_bearish_engulf
_original_check_zone = SignalEngine._check_zone

def _audit_check_bullish(self, zone, c1, c2, c3, ts, idx, df, audit=None):
    """Wrapped _check_bullish_engulf with per-bar rejection logging."""
    
    o1, c1v = float(c1["open"]), float(c1["close"])
    o2, c2v = float(c2["open"]), float(c2["close"])
    o3, c3v = float(c3["open"]), float(c3["close"])
    l3 = float(c3["low"])
    h3 = float(c3["high"])
    
    # Condition 1: Price hasn't reached zone
    if l3 > zone.price_low * 1.005:
        if audit is not None:
            audit['engulf_reject_zone_not_reached'] += 1
            AUDIT.engulf_rejections['price_not_at_zone'] += 1
        return None
    
    # Condition 2: Close below zone (breakdown)
    if c3v < zone.price_low:
        if audit is not None:
            audit['engulf_reject_breakdown'] += 1
            AUDIT.engulf_rejections['close_breakdown'] += 1
        return None
    
    body_c3 = abs(c3v - o3)
    range_c3 = h3 - l3 if h3 > l3 else 0.0001
    body_ratio_c3 = body_c3 / range_c3
    
    is_bullish = c3v > o3
    engulfs_prev = c3v > max(o2, c2v)
    
    lower_wick = min(o3, c3v) - l3
    wick_rejection = lower_wick / range_c3 if range_c3 > 0 else 0
    
    confidence = 0.0
    comps = []
    
    if is_bullish:
        confidence += 0.20
        comps.append('bullish_close')
    
    if engulfs_prev:
        confidence += 0.25
        comps.append('engulfs_prev')
    
    if body_ratio_c3 >= self.engulf_body_ratio:
        confidence += 0.15
        comps.append('strong_body')
    
    if wick_rejection > 0.3:
        confidence += 0.20
        comps.append('wick_rejection')
    
    vol_c3 = float(c3.get("volume", 0))
    if idx >= 20:
        past_vol = df["volume"].iloc[idx - 20 : idx]
        median_vol = past_vol.median()
        if median_vol > 0 and vol_c3 / median_vol >= self.engulf_volume_mult:
            confidence += 0.10
            comps.append('vol_surge')
    
    confidence += min(zone.score / 20.0, 0.10)
    if zone.score >= 5.0:
        comps.append('zone_score')
    
    # Track confidence components even when below threshold
    if audit is not None:
        for comp in comps:
            AUDIT.conf_components[comp] += 1
    
    if confidence < self.min_confidence:
        if audit is not None:
            audit['engulf_reject_low_confidence'] += 1
            AUDIT.engulf_rejections['low_confidence'] += 1
            audit['confidence_value'] = confidence
            audit['confidence_comps'] = comps
        return None
    
    # Track that engulf matched (confidence passes)
    if audit is not None:
        audit['engulf_matched'] += 1
    
    entry_price = c3v
    stop_price = min(zone.price_low * 0.998, l3 * 0.998)
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
        reason="+".join(comps),
    )

def _audit_check_bearish(self, zone, c1, c2, c3, ts, idx, df, audit=None):
    """Wrapped _check_bearish_engulf with per-bar rejection logging."""
    
    o3, c3v = float(c3["open"]), float(c3["close"])
    o2, c2v = float(c2["open"]), float(c2["close"])
    l3 = float(c3["low"])
    h3 = float(c3["high"])
    
    # Condition 1: Price hasn't reached zone
    if h3 < zone.price_high * 0.995:
        if audit is not None:
            audit['engulf_reject_zone_not_reached'] += 1
            AUDIT.engulf_rejections['price_not_at_zone'] += 1
        return None
    
    # Condition 2: Close above zone (breakout)
    if c3v > zone.price_high:
        if audit is not None:
            audit['engulf_reject_breakout'] += 1
            AUDIT.engulf_rejections['close_breakout'] += 1
        return None
    
    body_c3 = abs(c3v - o3)
    range_c3 = h3 - l3 if h3 > l3 else 0.0001
    body_ratio_c3 = body_c3 / range_c3
    
    is_bearish = c3v < o3
    engulfs_prev = c3v < min(o2, c2v)
    
    upper_wick = h3 - max(o3, c3v)
    wick_rejection = upper_wick / range_c3 if range_c3 > 0 else 0
    
    confidence = 0.0
    comps = []
    
    if is_bearish:
        confidence += 0.20
        comps.append('bearish_close')
    
    if engulfs_prev:
        confidence += 0.25
        comps.append('engulfs_prev')
    
    if body_ratio_c3 >= self.engulf_body_ratio:
        confidence += 0.15
        comps.append('strong_body')
    
    if wick_rejection > 0.3:
        confidence += 0.20
        comps.append('wick_rejection')
    
    vol_c3 = float(c3.get("volume", 0))
    if idx >= 20:
        past_vol = df["volume"].iloc[idx - 20 : idx]
        median_vol = float(np.median(past_vol.values))
        if median_vol > 0 and vol_c3 / median_vol >= self.engulf_volume_mult:
            confidence += 0.10
            comps.append('vol_surge')
    
    confidence += min(zone.score / 20.0, 0.10)
    if zone.score >= 5.0:
        comps.append('zone_score')
    
    if audit is not None:
        for comp in comps:
            AUDIT.conf_components[comp] += 1
    
    if confidence < self.min_confidence:
        if audit is not None:
            audit['engulf_reject_low_confidence'] += 1
            AUDIT.engulf_rejections['low_confidence'] += 1
            audit['confidence_value'] = confidence
            audit['confidence_comps'] = comps
        return None
    
    if audit is not None:
        audit['engulf_matched'] += 1
    
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
        reason="+".join(comps),
    )

def _audit_check_zone(self, zone, df, idx, current_price, audit=None):
    """Wrapped _check_zone."""
    if idx < 3:
        return None

    c3 = df.iloc[idx]
    c2 = df.iloc[idx - 1]
    c1 = df.iloc[idx - 2]
    ts = int(c3["timestamp"]) if "timestamp" in df.columns else int(idx)

    if zone.zone_type == ZoneType.DEMAND:
        return _audit_check_bullish(self, zone, c1, c2, c3, ts, idx, df, audit=audit)
    elif zone.zone_type == ZoneType.SUPPLY:
        return _audit_check_bearish(self, zone, c1, c2, c3, ts, idx, df, audit=audit)

    return None


def _audit_scan(self, current_price, zones, df_15m, current_idx=None):
    """Wrapped scan() with audit counters."""
    if current_idx is None:
        current_idx = len(df_15m) - 1
    if current_idx < 3:
        return []

    # Get bar timestamp for monthly tracking
    bar = df_15m.iloc[current_idx]
    bar_ts = int(bar["timestamp"]) if "timestamp" in df_15m.columns else current_idx
    AUDIT.set_month(bar_ts)

    # Filter zones near price
    nearby = [
        z for z in zones
        if z.distance_to_price(current_price) <= self.zone_distance_pct
    ]

    audit = defaultdict(int)
    audit['total_zones'] = len(zones)
    audit['zones_near_price'] = len(nearby)

    if not nearby:
        # No zones near price — track this as a rejection at zone proximity layer
        AUDIT.inc('bar_no_near_zones')
        AUDIT.layer_rejections_2022plus['zone_proximity'] += 1
        return []

    AUDIT.inc('bar_has_near_zones')
    AUDIT.bars_with_near_zones += 1

    signals: list[EntrySignal] = []
    for zone in nearby:
        audit['engulf_checked'] += 1
        signal = _audit_check_zone(self, zone, df_15m, current_idx, current_price, audit=audit)
        
        if signal is None:
            # Engulf didn't match — track the rejection reason
            if audit.get('engulf_reject_zone_not_reached', 0) > 0:
                AUDIT.layer_rejections_2022plus['engulf_zone_not_reached'] += 1
            elif audit.get('engulf_reject_breakdown', 0) > 0 or audit.get('engulf_reject_breakout', 0) > 0:
                AUDIT.layer_rejections_2022plus['engulf_break_violation'] += 1
            elif audit.get('engulf_reject_low_confidence', 0) > 0:
                AUDIT.layer_rejections_2022plus['confidence_threshold'] += 1
            continue
        
        if signal.confidence >= self.min_confidence:
            if signal.risk_reward_ratio >= self.min_rr_ratio:
                signals.append(signal)
                audit['signals_generated'] += 1
            else:
                audit['rr_failed'] += 1
                AUDIT.layer_rejections_2022plus['rr_filter'] += 1
        else:
            audit['confidence_failed'] += 1
            AUDIT.layer_rejections_2022plus['confidence_threshold'] += 1

    signals.sort(key=lambda s: s.confidence, reverse=True)

    # Flush audit counters to monthly
    for k, v in audit.items():
        if isinstance(v, (int, float)) and k in ('total_zones', 'zones_near_price', 'engulf_checked',
                                                  'engulf_matched', 'confidence_failed', 'rr_failed',
                                                  'signals_generated', 'engulf_reject_zone_not_reached',
                                                  'engulf_reject_breakdown', 'engulf_reject_breakout',
                                                  'engulf_reject_low_confidence'):
            AUDIT.monthly[AUDIT._current_month][k] += int(v)

    return signals


# ── Wrapped BacktestEngine.run() ─────────────────────────────────────

_original_backtest_run = BacktestEngine.run

def _audit_backtest_run(self, data, start_date=None, end_date=None, burn_in_days=90):
    """Wrapped run() with monthly zone tracking."""
    
    df_15m = data.get("15m")
    if df_15m is None:
        logger.error("No 15m data available.")
        return BacktestResult()
    
    if "timestamp" not in df_15m.columns and "time" in df_15m.columns:
        df_15m["timestamp"] = df_15m["time"].astype("int64") // 10**6
    for tf in data:
        if "timestamp" not in data[tf].columns and "time" in data[tf].columns:
            data[tf]["timestamp"] = data[tf]["time"].astype("int64") // 10**6
    
    if df_15m is None or len(df_15m) < 100:
        logger.error("Need at least 100 candles.")
        return BacktestResult()
    
    # Date filter
    if start_date or end_date:
        if "timestamp" in df_15m.columns:
            ts_col = df_15m["timestamp"]
            if ts_col.iloc[0] > 1e10:
                ts_col = ts_col / 1000
            mask = pd.Series(True, index=df_15m.index)
            if start_date:
                start_ts = pd.Timestamp(start_date, tz="UTC").timestamp()
                mask &= ts_col >= start_ts
            if end_date:
                end_ts = pd.Timestamp(end_date, tz="UTC").timestamp()
                mask &= ts_col <= end_ts
            df_15m = df_15m.loc[mask]
    
    df_15m = df_15m.reset_index(drop=True)
    n_bars = len(df_15m)
    
    burn_in_bars = burn_in_days * 96
    first_live_bar = min(burn_in_bars, n_bars)
    
    equity = self.settings.risk.initial_capital
    equity_curve = [equity]
    open_trades = []
    closed_trades = []
    trade_counter = 0
    zone_cooldowns = {}
    cooldown_bars = self.settings.trading.cooldown_bars
    
    peak_equity = equity
    max_dd_threshold = self.settings.risk.max_drawdown_pct
    halted = False
    halt_date = ""
    
    current_zones = []
    last_zone_rebuild = -999
    
    start_ts_val = df_15m["timestamp"].iloc[0]
    end_ts_val = df_15m["timestamp"].iloc[-1]
    to_dt = lambda ts: datetime.fromtimestamp(
        ts / 1000 if ts > 1e10 else ts, tz=timezone.utc
    )
    
    logger.info(f"Audit backtest: {n_bars} bars, "
                f"{to_dt(start_ts_val/1000 if start_ts_val>1e10 else start_ts_val).date()} → "
                f"{to_dt(end_ts_val/1000 if end_ts_val>1e10 else end_ts_val).date()}, "
                f"burn-in: {burn_in_days}d")
    
    t_start = time.time()
    last_print = t_start
    
    # Track signals that passed scan() but were blocked by backtest filters
    signals_blocked_cooldown = 0
    signals_blocked_overlap = 0
    signals_blocked_validation = 0
    signals_blocked_sizing = 0
    signals_blocked_risk = 0
    
    for idx in range(first_live_bar, n_bars):
        bar = df_15m.iloc[idx]
        bar_ts = int(bar["timestamp"]) if "timestamp" in df_15m.columns else idx
        bar_close = float(bar["close"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        
        month_key = AUDIT.get_month(bar_ts)
        
        # ── Rebuild zones periodically (original logic) ──
        if idx - last_zone_rebuild >= self.zone_rebuild_interval:
            sliced_data = {tf: df.iloc[:idx+1] for tf, df in data.items()}
            current_zones = self.zone_detector.detect_all(sliced_data)
            last_zone_rebuild = idx
            
            # AUDIT: log zone stats
            tf_counts = defaultdict(int)
            for z in current_zones:
                tf_counts[z.timeframe] += 1
            avg_score = np.mean([z.score for z in current_zones]) if current_zones else 0
            
            AUDIT.monthly[month_key]['zone_rebuild_count'] += 1
            AUDIT.monthly[month_key]['zone_total'] += len(current_zones)
            AUDIT.monthly[month_key]['zone_avg_score'] += avg_score
        
        # ── Manage open trades (original logic, copied) ──
        still_open = []
        for trade in open_trades:
            exit_now = False
            exit_price = bar_close
            exit_reason = ""
            
            if trade.direction == "LONG":
                if bar_low <= trade.stop_price:
                    exit_now = True
                    exit_price = trade.stop_price
                    exit_reason = "stop_loss"
                elif bar_high >= trade.take_profit:
                    exit_now = True
                    exit_price = trade.take_profit
                    exit_reason = "take_profit"
            elif trade.direction == "SHORT":
                if bar_high >= trade.stop_price:
                    exit_now = True
                    exit_price = trade.stop_price
                    exit_reason = "stop_loss"
                elif bar_low <= trade.take_profit:
                    exit_now = True
                    exit_price = trade.take_profit
                    exit_reason = "take_profit"
            
            # Trailing stops (simplified — enough for audit)
            if not exit_now and trade.entry_price != trade.stop_price:
                r_value = abs(trade.entry_price - trade.stop_price)
                if r_value > 0:
                    if trade.direction == "LONG":
                        unrealized_r = (bar_high - trade.entry_price) / r_value
                        if not trade.breakeven_active and unrealized_r >= 1.0:
                            trade.breakeven_active = True
                            trade.stop_price = trade.entry_price
                        if unrealized_r >= 1.5:
                            trade.trailing_active = True
                            best_trail = trade.stop_price
                            for z in current_zones:
                                if z.zone_type.value == "DEMAND" and z.score >= 2.0:
                                    if best_trail < z.midpoint < bar_high:
                                        best_trail = z.midpoint
                            if best_trail > trade.stop_price:
                                trade.stop_price = best_trail
                    elif trade.direction == "SHORT":
                        unrealized_r = (trade.entry_price - bar_low) / r_value
                        if not trade.breakeven_active and unrealized_r >= 1.0:
                            trade.breakeven_active = True
                            trade.stop_price = trade.entry_price
                        if unrealized_r >= 1.5:
                            trade.trailing_active = True
                            best_trail = trade.stop_price
                            for z in current_zones:
                                if z.zone_type.value == "SUPPLY" and z.score >= 2.0:
                                    if best_trail > z.midpoint > bar_low:
                                        best_trail = z.midpoint
                            if best_trail < trade.stop_price:
                                trade.stop_price = best_trail
            
            if exit_now:
                trade.exit_time = to_dt(bar_ts)
                trade.exit_price = exit_price
                if exit_reason == "stop_loss" and trade.trailing_active:
                    exit_reason = "trailing_stop"
                trade.exit_reason = exit_reason
                trade.status = "won" if (
                    (trade.direction == "LONG" and exit_price > trade.entry_price) or
                    (trade.direction == "SHORT" and exit_price < trade.entry_price)
                ) else "lost"
                
                if trade.direction == "LONG":
                    pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
                else:
                    pnl_pct = (trade.entry_price - exit_price) / trade.entry_price
                
                trade.pnl = trade.risk_dollars * (pnl_pct / trade.initial_risk_pct) if trade.initial_risk_pct > 0 else 0.0
                trade.pnl_r = trade.pnl / trade.risk_dollars if trade.risk_dollars > 0 else 0
                
                equity += trade.pnl
                equity_curve.append(equity)
                closed_trades.append(trade)
                zone_cooldowns[trade.zone_info] = idx
            else:
                still_open.append(trade)
        
        open_trades = still_open
        
        # ── Drawdown circuit breaker ──
        if equity > peak_equity:
            peak_equity = equity
        if peak_equity > 0:
            current_dd = (peak_equity - equity) / peak_equity * 100
            if current_dd >= max_dd_threshold and not halted:
                halted = True
                halt_date = str(to_dt(bar_ts).date())
                logger.warning(f"HALTED: Drawdown {current_dd:.1f}% on {halt_date}")
        
        # ── Generate entry signals ──
        if not halted:
            htf_zones = [z for z in current_zones if z.timeframe in ("1h", "4h", "1d", "1w")]
            if not htf_zones and current_zones:
                htf_zones = current_zones
            
            # Track zone availability per month
            AUDIT.monthly[month_key]['bar_total_zones_htf'] += len(htf_zones)
            
            # Signal engine scan (uses audited version)
            signals = self.signal_engine.scan(bar_close, htf_zones, df_15m, idx)
            
            # Track how many signals came from the engine
            AUDIT.monthly[month_key]['signals_from_engine'] += len(signals)
            
            for sig in signals:
                # Cooldown check
                zone_key = f"{sig.zone.zone_type.value}_{sig.zone.timeframe}_{sig.zone.midpoint:.0f}"
                last_exit = zone_cooldowns.get(zone_key, -cooldown_bars - 1)
                if idx - last_exit < cooldown_bars:
                    signals_blocked_cooldown += 1
                    AUDIT.monthly[month_key]['blocked_cooldown'] += 1
                    AUDIT.layer_rejections_2022plus['cooldown'] += 1
                    continue
                
                # Position overlap prevention
                open_dirs = {t.direction for t in open_trades}
                if sig.direction.value in open_dirs:
                    signals_blocked_overlap += 1
                    AUDIT.monthly[month_key]['blocked_overlap'] += 1
                    AUDIT.layer_rejections_2022plus['overlap'] += 1
                    continue
                
                # Validation
                if sig.entry_price <= 0 or sig.stop_price <= 0:
                    signals_blocked_validation += 1
                    continue
                if sig.direction == SignalDirection.LONG and sig.stop_price >= sig.entry_price:
                    signals_blocked_validation += 1
                    continue
                if sig.direction == SignalDirection.SHORT and sig.stop_price <= sig.entry_price:
                    signals_blocked_validation += 1
                    continue
                
                # Position sizing
                leverage = float(self.settings.trading.max_leverage)
                win_prob = 0.40 + (sig.confidence * 0.15)
                risk_dollars, qty_btc = self.risk_manager.calculate_size(
                    win_prob=win_prob,
                    reward_risk=sig.risk_reward_ratio,
                    entry=sig.entry_price,
                    stop=sig.stop_price,
                    leverage=leverage,
                )
                
                if risk_dollars <= 0 or qty_btc <= 0:
                    signals_blocked_sizing += 1
                    continue
                
                if not self.risk_manager.can_open(risk_dollars):
                    signals_blocked_risk += 1
                    continue
                
                # Create trade
                trade_counter += 1
                trade = BacktestTrade(
                    trade_id=trade_counter,
                    direction=sig.direction.value,
                    entry_time=to_dt(sig.timestamp_ms),
                    entry_price=sig.entry_price,
                    stop_price=sig.stop_price,
                    take_profit=sig.take_profit,
                    zone_info=zone_key,
                    quantity_btc=qty_btc,
                    risk_dollars=risk_dollars,
                    confidence=sig.confidence,
                    initial_risk_pct=abs(sig.entry_price - sig.stop_price) / sig.entry_price
                        if sig.entry_price > 0 else 0.0,
                )
                open_trades.append(trade)
                zone_cooldowns[zone_key] = idx
                
                AUDIT.monthly[month_key]['trades_taken'] += 1
        
        # Progress logging
        if time.time() - last_print > 30.0:
            pct = (idx - first_live_bar) / (n_bars - first_live_bar) * 100
            dt_str = to_dt(bar_ts).strftime("%Y-%m-%d")
            logger.info(f"  [{pct:5.1f}%] {dt_str} | "
                        f"Equity: ${equity:,.2f} | Open: {len(open_trades)} | "
                        f"Closed: {len(closed_trades)}")
            last_print = time.time()
    
    # Close remaining trades
    if open_trades and len(df_15m) > 0:
        last_close = float(df_15m["close"].iloc[-1])
        last_ts = int(df_15m["timestamp"].iloc[-1])
        for trade in open_trades:
            trade.exit_time = to_dt(last_ts)
            trade.exit_price = last_close
            trade.exit_reason = "end_of_data"
            trade.status = "won" if (
                (trade.direction == "LONG" and last_close > trade.entry_price) or
                (trade.direction == "SHORT" and last_close < trade.entry_price)
            ) else "lost"
            pnl_pct = abs(last_close - trade.entry_price) / trade.entry_price
            if trade.direction == "LONG":
                pnl_pct = (last_close - trade.entry_price) / trade.entry_price
            else:
                pnl_pct = (trade.entry_price - last_close) / trade.entry_price
            trade.pnl = trade.risk_dollars * (pnl_pct / (abs(trade.entry_price - trade.stop_price) / trade.entry_price))
            trade.pnl_r = trade.pnl / trade.risk_dollars if trade.risk_dollars > 0 else 0
            equity += trade.pnl
            closed_trades.append(trade)
    
    # Compute metrics
    all_trades = closed_trades
    wins = [t for t in all_trades if t.status == "won"]
    losses = [t for t in all_trades if t.status == "lost"]
    expired = [t for t in all_trades if t.status == "expired"]
    
    gross_profit = sum(t.pnl for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1.0
    true_pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
    total_pnl = gross_profit - gross_loss if losses else gross_profit
    final_equity_val = self.settings.risk.initial_capital + total_pnl
    total_return = (final_equity_val / self.settings.risk.initial_capital - 1) * 100
    win_rate = len(wins) / len(all_trades) if all_trades else 0
    
    # Sharpe (simplified)
    if len(equity_curve) > 1:
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = float(returns.mean() / returns.std() * np.sqrt(365)) if returns.std() > 0 else 0
    else:
        sharpe = 0
    
    result = BacktestResult(
        total_trades=len(all_trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        expired_trades=len(expired),
        win_rate=win_rate,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        total_pnl=total_pnl,
        true_profit_factor=true_pf,
        sharpe_ratio=sharpe,
        max_drawdown_pct=current_dd if halted else 0,
        final_equity=final_equity_val,
        total_return_pct=total_return,
        equity_curve=equity_curve,
        trades=all_trades,
        start_date=str(to_dt(start_ts_val).date()),
        end_date=str(to_dt(end_ts_val).date()),
        duration_days=n_bars // 96,
        candles_processed=n_bars - first_live_bar,
        halted=halted,
        halt_reason=f"Drawdown {current_dd:.1f}%" if halted else "",
        halt_date=halt_date,
    )
    
    # Store audit data on result
    result.audit_monthly = dict(AUDIT.monthly)
    result.audit_conf_components = dict(AUDIT.conf_components)
    result.audit_engulf_rejections = dict(AUDIT.engulf_rejections)
    result.audit_layer_rejections = dict(AUDIT.layer_rejections_2022plus)
    result.audit_bars_with_zones = AUDIT.bars_with_near_zones
    result.signals_blocked_cooldown = signals_blocked_cooldown
    result.signals_blocked_overlap = signals_blocked_overlap
    result.signals_blocked_validation = signals_blocked_validation
    result.signals_blocked_sizing = signals_blocked_sizing
    result.signals_blocked_risk = signals_blocked_risk
    
    return result


# ── Apply patches ─────────────────────────────────────────────────────

SignalEngine.scan = _audit_scan

# ── Run ───────────────────────────────────────────────────────────────

def load_cached_data():
    """Load CSV cache directly — avoids Binance API fetch."""
    import glob
    
    cache_dir = os.path.expanduser("~/oda/data/cache")
    tf_map = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"}
    
    data = {}
    for tf_label, tf_key in tf_map.items():
        pattern = os.path.join(cache_dir, f"BTCUSDT_{tf_label}.csv")
        files = glob.glob(pattern)
        if files:
            df = pd.read_csv(files[0])
            # Standardize column names
            df.columns = [c.lower().strip() for c in df.columns]
            # Parse time column to ms timestamp
            if "time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["time"]).astype("int64") // 10**6
            elif "timestamp" in df.columns:
                if df["timestamp"].dtype == object:
                    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**6
            # Sort by timestamp
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp").reset_index(drop=True)
            data[tf_key] = df
            logger.info(f"  Loaded {tf_label}: {len(df):>6,} candles from cache")
        else:
            logger.warning(f"  No cache found for {tf_label}")
    
    return data


def run_audit():
    """Run the instrumented backtest and produce the monthly analysis."""
    
    logger.info("=" * 60)
    logger.info("ODA SIGNAL PIPELINE AUDIT")
    logger.info("=" * 60)
    
    settings = Settings.load()
    
    # Override settings for this run
    os.environ["ODA_RISK_PCT"] = "2"
    os.environ["ODA_MAX_LEVERAGE"] = "3"
    os.environ["ODA_CAPITAL"] = "1000"
    settings = Settings.load()
    
    logger.info("Loading cached CSVs...")
    data = load_cached_data()
    logger.info(f"Timeframes loaded: {list(data.keys())}")
    for tf, df in data.items():
        logger.info(f"  {tf:4s}: {len(df):>6,} candles")
    
    logger.info("\nRunning instrumented backtest...")
    engine = BacktestEngine(settings, zone_rebuild_interval=96)
    
    t0 = time.time()
    result = engine.run(
        data,
        start_date="2021-01-01",
        end_date=None,  # Use all available data
        burn_in_days=90,
    )
    elapsed = time.time() - t0
    
    print()
    print(result.summary())
    print()
    
    # ── Produce monthly summary ──
    print("=" * 60)
    print("PIPELINE AUDIT — MONTHLY BREAKDOWN")
    print("=" * 60)
    
    months = sorted(result.audit_monthly.keys())
    
    header = f"{'Month':<10} {'Z/Rb':>6} {'ZoneN':>6} {'Engulf':>7} {'Conf≥':>6} {'RR≥':>5} {'Coold':>5} {'Signal':>6} {'Trade':>6}"
    sep = "-" * len(header)
    
    lines = [sep, "Pipeline Stage:  Z=Rebuild  NearPrice  Engulf OK  Conf≥0.5  RR≥1.5  Cooldown  Engine  Trades", sep, header, sep]
    
    totals = defaultdict(int)
    
    for month in months:
        m = result.audit_monthly[month]
        
        # Zone rebuild averages
        z_rebuilds = m.get('zone_rebuild_count', 0) or 1
        z_total_avg = m.get('zone_total', 0) / z_rebuilds if z_rebuilds > 0 else 0
        z_score_avg = m.get('zone_avg_score', 0) / z_rebuilds if z_rebuilds > 0 else 0
        
        zones_near = m.get('zones_near_price', 0)
        engulf_matched = m.get('engulf_matched', 0)
        conf_passed = engulf_matched - m.get('engulf_reject_low_confidence', 0)
        rr_passed = m.get('signals_generated', 0)
        signals_from_engine = m.get('signals_from_engine', 0)
        blocked_cooldown = m.get('blocked_cooldown', 0)
        trades_taken = m.get('trades_taken', 0)
        
        line = (
            f"{month:<10} {z_total_avg:>6.1f} {zones_near:>6} {engulf_matched:>7} "
            f"{conf_passed:>6} {rr_passed:>5} {blocked_cooldown:>5} "
            f"{signals_from_engine:>6} {trades_taken:>6}"
        )
        lines.append(line)
        
        totals['z_total_avg'] += z_total_avg
        totals['zones_near'] += zones_near
        totals['engulf_matched'] += engulf_matched
        totals['conf_passed'] += conf_passed
        totals['rr_passed'] += rr_passed
        totals['signals_from_engine'] += signals_from_engine
        totals['trades_taken'] += trades_taken
    
    lines.append(sep)
    n_months = len(months) or 1
    lines.append(
        f"{'AVG':<10} {totals['z_total_avg']/n_months:>6.1f} {totals['zones_near']:>6} "
        f"{totals['engulf_matched']:>7} {totals['conf_passed']:>6} {totals['rr_passed']:>5} "
        f"{'—':>5} {totals['signals_from_engine']:>6} {totals['trades_taken']:>6}"
    )
    lines.append(sep)
    
    print("\n".join(lines))
    print()
    
    # ── Engulf rejection reasons ──
    print("=" * 60)
    print("ENGULF REJECTION REASONS (all months)")
    print("=" * 60)
    for reason, count in sorted(result.audit_engulf_rejections.items(), key=lambda x: -x[1]):
        print(f"  {reason:<30s} {count:>6}")
    print()
    
    # ── Confidence component breakdown ──
    print("=" * 60)
    print("CONFIDENCE COMPONENT BREAKDOWN")
    print("=" * 60)
    comp_order = ['bullish_close', 'bearish_close', 'engulfs_prev', 'strong_body', 
                  'wick_rejection', 'vol_surge', 'zone_score']
    for comp in comp_order:
        count = result.audit_conf_components.get(comp, 0)
        if count > 0:
            print(f"  {comp:<25s} {count:>6}")
    print()
    
    # ── Backend filter blocks ──
    print("=" * 60)
    print("SIGNALS BLOCKED BY BACKTEST FILTERS")
    print("=" * 60)
    print(f"  Cooldown block:      {result.signals_blocked_cooldown:>6}")
    print(f"  Direction overlap:   {result.signals_blocked_overlap:>6}")
    print(f"  Validation fail:     {result.signals_blocked_validation:>6}")
    print(f"  Sizing fail:         {result.signals_blocked_sizing:>6}")
    print(f"  Risk check fail:     {result.signals_blocked_risk:>6}")
    print()
    
    # ── Diagnosis ──
    print("=" * 60)
    print("DIAGNOSIS — PRIMARY CAUSE OF POST-MAY-2021 FLATLINE")
    print("=" * 60)
    
    # Analyze which layer caused the most rejections
    lr = result.audit_layer_rejections if hasattr(result, 'audit_layer_rejections') else {}
    
    # Only look at months after May 2021 for the flatline analysis
    post_may_months = [m for m in months if m > "2021-05"]
    pre_june_months = [m for m in months if m <= "2021-05"]
    
    # Compare zone counts pre vs post
    pre_zones = []
    post_zones = []
    pre_signals = []
    post_signals = []
    pre_near = []
    post_near = []
    
    for month in months:
        m = result.audit_monthly[month]
        z_rebuilds = m.get('zone_rebuild_count', 0) or 1
        z_avg = m.get('zone_total', 0) / z_rebuilds if z_rebuilds > 0 else 0
        near_count = m.get('zones_near_price', 0)
        signals = m.get('signals_from_engine', 0)
        
        if month <= "2021-05":
            pre_zones.append(z_avg)
            pre_near.append(near_count)
            pre_signals.append(signals)
        elif month > "2021-05":
            post_zones.append(z_avg)
            post_near.append(near_count)
            post_signals.append(signals)
    
    avg_pre_zones = np.mean(pre_zones) if pre_zones else 0
    avg_post_zones = np.mean(post_zones) if post_zones else 0
    avg_pre_near = np.mean(pre_near) if pre_near else 0
    avg_post_near = np.mean(post_near) if post_near else 0
    avg_pre_signals = np.mean(pre_signals) if pre_signals else 0
    avg_post_signals = np.mean(post_signals) if post_signals else 0
    
    print(f"  Periods compared: {len(pre_june_months)} pre-June-2021 months vs {len(post_may_months)} post-May-2021 months")
    print(f"  Avg zones/rebuild (pre):  {avg_pre_zones:.1f}")
    print(f"  Avg zones/rebuild (post): {avg_post_zones:.1f}")
    print(f"  Avg near-price checks/mo (pre):  {avg_pre_near:.1f}")
    print(f"  Avg near-price checks/mo (post): {avg_post_near:.1f}")
    print(f"  Avg signals/mo (pre):  {avg_pre_signals:.1f}")
    print(f"  Avg signals/mo (post): {avg_post_signals:.1f}")
    print()
    
    # Determine primary cause
    diagnoses = []
    
    if avg_post_zones < avg_pre_zones * 0.5:
        diagnoses.append(("Zone Detection", f"Zone count dropped from {avg_pre_zones:.1f} to {avg_post_zones:.1f} per rebuild ({(1-avg_post_zones/avg_pre_zones)*100:.0f}% reduction)"))
    elif avg_post_near < avg_pre_near * 0.3:
        diagnoses.append(("Zone Proximity", f"Near-price checks dropped from {avg_pre_near:.0f} to {avg_post_near:.0f} per month"))
    
    total_engulf_rejections = sum(result.audit_engulf_rejections.values())
    if total_engulf_rejections > 0:
        low_conf = result.audit_engulf_rejections.get('low_confidence', 0)
        pct_low_conf = low_conf / total_engulf_rejections * 100
        if pct_low_conf > 60:
            diagnoses.append(("Confidence Threshold", f"{pct_low_conf:.0f}% of engulf rejections are due to low confidence"))
    
    if avg_post_signals == 0 and avg_pre_signals > 0:
        diagnoses.append(("Signal Engine (combined)", f"Zero signals generated post-May-2021 vs {avg_pre_signals:.1f}/mo pre-June"))
    
    if not diagnoses:
        diagnoses.append(("Mixed/Inconclusive", "No single dominant layer — see monthly breakdown for detail"))
    
    print("  Primary cause ranking:")
    for i, (layer, reason) in enumerate(diagnoses, 1):
        print(f"    {i}. {layer} — {reason}")
    print()
    
    # Write to file
    output_path = os.path.expanduser("~/oda/ph2-audit-results.txt")
    with open(output_path, "w") as f:
        f.write("=== PIPELINE AUDIT RESULTS ===\n")
        f.write(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"Symbol: BTCUSDT | Capital: $1,000 | Leverage: 3x | Risk: 2%\n\n")
        f.write("\n".join(lines))
        f.write("\n\n")
        f.write("ENGULF REJECTION REASONS:\n")
        for reason, count in sorted(result.audit_engulf_rejections.items(), key=lambda x: -x[1]):
            f.write(f"  {reason:<30s} {count:>6}\n")
        f.write("\n")
        f.write("CONFIDENCE COMPONENT BREAKDOWN:\n")
        for comp in comp_order:
            count = result.audit_conf_components.get(comp, 0)
            if count > 0:
                f.write(f"  {comp:<25s} {count:>6}\n")
        f.write("\n")
        f.write("DIAGNOSIS:\n")
        for layer, reason in diagnoses:
            f.write(f"  Primary: {layer}\n")
            f.write(f"  Evidence: {reason}\n")
        f.write(f"\nElapsed: {elapsed:.0f}s\n")
    
    logger.info(f"\nAudit results written to ~/oda/ph2-audit-results.txt")
    logger.info(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    run_audit()
