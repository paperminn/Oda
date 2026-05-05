"""
Oda Backtest Engine — Walk-Forward Simulation with True PF
===========================================================

Walk-forward backtest against cached BTC/USDT OHLCV data. Simulates
real-time trading decisions: zone detection → signal generation →
position sizing → trade management → equity tracking.

Key metrics: True PF, Sortino Ratio, Sharpe Ratio, Maximum Drawdown.

Methodology adapted from CryptoMecha01 master.py (Z's prior work).
The Proxy PF formula is permanently banned — only True PF is reported.

True PF = sum(positive_pnl) / abs(sum(negative_pnl))
  where pnl = actual per-trade P&L in dollars

Usage:
    from oda.config import Settings
    from oda.backtest import BacktestEngine

    settings = Settings.load()
    engine = BacktestEngine(settings)
    results = engine.run(data)  # data from oda.data.fetch_all()
    print(results.summary())
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from oda.config import Settings
from oda.data import fetch_all
from oda.zones import Zone, ZoneDetector, ZoneType
from oda.signals import EntrySignal, PipelineAudit, SignalDirection, SignalEngine
from oda.risk import Position, RiskManager
from oda.regime import RegimeDetector, RegimeType

logger = logging.getLogger("oda.backtest")


# ── Data Classes ─────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """A trade tracked during backtest."""
    trade_id: int
    direction: str       # "LONG" or "SHORT"
    entry_time: datetime
    entry_price: float
    stop_price: float
    take_profit: float
    zone_info: str       # Zone description
    quantity_btc: float
    risk_dollars: float
    confidence: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_r: float = 0.0
    status: str = "open"  # open, won, lost, expired
    exit_reason: str = ""
    # Nunchi trailing stops (CryptoMecha01)
    breakeven_active: bool = False
    trailing_active: bool = False
    initial_risk_pct: float = 0.0  # abs(entry - stop) / entry at trade open


@dataclass
class BacktestResult:
    """Complete backtest results with metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    expired_trades: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_pnl: float = 0.0
    true_profit_factor: float = 0.0
    sortino_ratio: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_r_per_trade: float = 0.0
    avg_r_win: float = 0.0
    avg_r_loss: float = 0.0
    final_equity: float = 0.0
    total_return_pct: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    duration_days: int = 0
    candles_processed: int = 0
    halted: bool = False
    halt_reason: str = ""
    halt_date: str = ""

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            "=" * 60,
            "ODA BACKTEST RESULTS",
            "=" * 60,
            f"Period: {self.start_date} → {self.end_date} ({self.duration_days} days)",
            f"Candles processed: {self.candles_processed:,}",
            "",
            f"Total Trades:    {self.total_trades:>6}",
            f"Winning Trades:  {self.winning_trades:>6}  ({self.win_rate:.1%})",
            f"Losing Trades:   {self.losing_trades:>6}",
            f"Expired Trades:  {self.expired_trades:>6}",
            "",
            f"Gross Profit:    ${self.gross_profit:>10,.2f}",
            f"Gross Loss:      ${self.gross_loss:>10,.2f}",
            f"Total P&L:       ${self.total_pnl:>10,.2f}",
            f"Final Equity:    ${self.final_equity:>10,.2f}",
            f"Total Return:    {self.total_return_pct:>10.2f}%",
            "",
            f"True PF:         {self.true_profit_factor:>10.3f}",
            f"Sortino Ratio:   {self.sortino_ratio:>10.3f}",
            f"Sharpe Ratio:    {self.sharpe_ratio:>10.3f}",
            f"Max Drawdown:    {self.max_drawdown_pct:>10.2f}%",
            "",
            f"Avg R/Trade:     {self.avg_r_per_trade:>10.3f}",
            f"Avg R Win:       {self.avg_r_win:>10.3f}",
            f"Avg R Loss:      {self.avg_r_loss:>10.3f}",
            "",
        ]
        if self.halted:
            lines.extend([
                f"⚠️  HALTED:         {self.halt_reason}",
                f"   Halt Date:     {self.halt_date}",
                "",
            ])
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Backtest Engine ──────────────────────────────────────────────────

class BacktestEngine:
    """Walk-forward backtest engine.

    Simulates trading by walking through 15m candles, detecting zones,
    generating entry signals, sizing positions, and managing trades.

    Parameters:
        settings: Oda Settings (config, risk, trading params).
        zone_rebuild_interval: Bars between zone rebuilds (default 96 = 1 day on 15m).
    """

    def __init__(
        self,
        settings: Settings,
        zone_rebuild_interval: int = 96,
        zone_window_days: int = 90,
    ) -> None:
        self.settings = settings
        self.zone_rebuild_interval = zone_rebuild_interval
        self.zone_window_days = zone_window_days

        # Modules
        self.zone_detector = ZoneDetector(
            max_zones_per_tf=20,
        )

        # Env-driven min_confidence — lets cron sweep parameter without code edits
        _min_conf = float(os.environ.get("ODA_MIN_CONFIDENCE", "0.50"))
        self.signal_engine = SignalEngine(
            min_confidence=_min_conf,
        )
        self.risk_manager = RiskManager(settings.risk)
        self.regime_detector = RegimeDetector(settings.regime) if hasattr(settings, 'regime') else RegimeDetector()

    def run(
        self,
        data: dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        burn_in_days: int = 90,
    ) -> BacktestResult:
        """Run the walk-forward backtest.

        Args:
            data: Multi-TF OHLCV DataFrames from oda.data.fetch_all().
            start_date: ISO date string to start (default: earliest in data).
            end_date: ISO date string to end (default: latest in data).
            burn_in_days: Days of burn-in (no trades, just zone learning).

        Returns:
            BacktestResult with all metrics and trade history.
        """
        df_15m = data.get("15m")
        if df_15m is None:
            logger.error("No 15m data available.")
            return BacktestResult()

        # Ensure timestamp column exists (data.py uses 'time')
        if "timestamp" not in df_15m.columns and "time" in df_15m.columns:
            df_15m["timestamp"] = df_15m["time"].astype("int64") // 10**6  # ns → ms
        for tf in data:
            if "timestamp" not in data[tf].columns and "time" in data[tf].columns:
                data[tf]["timestamp"] = data[tf]["time"].astype("int64") // 10**6
        if df_15m is None or len(df_15m) < 100:
            logger.error("Need at least 100 candles of 15m data.")
            return BacktestResult()

        # Filter by date range
        if start_date or end_date:
            if "timestamp" in df_15m.columns:
                ts_col = df_15m["timestamp"]
                if ts_col.iloc[0] > 1e10:  # milliseconds
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

        # Determine burn-in bar index
        burn_in_bars = burn_in_days * 96  # 96 15m bars per day
        first_live_bar = min(burn_in_bars, n_bars)

        # State
        equity = self.settings.risk.initial_capital
        equity_curve: list[float] = [equity]
        open_trades: list[BacktestTrade] = []
        closed_trades: list[BacktestTrade] = []
        trade_counter = 0
        zone_cooldowns: dict[str, int] = {}  # zone_key → last_exit_bar
        cooldown_bars = self.settings.trading.cooldown_bars

        # Drawdown circuit breaker
        peak_equity = equity
        max_dd_threshold = self.settings.risk.max_drawdown_pct
        halted = False
        halt_date = ""

        # Zone cache
        current_zones: list[Zone] = []
        last_zone_rebuild = -999

        # Tracking
        start_ts = df_15m["timestamp"].iloc[0]
        end_ts = df_15m["timestamp"].iloc[-1]
        to_dt = lambda ts: datetime.fromtimestamp(
            ts / 1000 if ts > 1e10 else ts, tz=timezone.utc
        )

        logger.info(
            f"Backtest: {n_bars} bars, "
            f"{to_dt(start_ts).date()} → {to_dt(end_ts).date()}, "
            f"burn-in: {burn_in_days}d"
        )

        t_start = time.time()
        last_print = t_start

        # ── Pipeline audit (monthly aggregation) ──
        monthly_audit: dict[str, PipelineAudit] = {}
        monthly_rebuild_zones: dict[str, int] = {}
        monthly_rebuild_count: dict[str, int] = {}
        monthly_zones_near_bar: dict[str, int] = {}
        monthly_trades_taken: dict[str, int] = {}

        for idx in range(first_live_bar, n_bars):
            bar = df_15m.iloc[idx]
            bar_ts = int(bar["timestamp"]) if "timestamp" in df_15m.columns else idx
            bar_close = float(bar["close"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])

            # ── Rebuild zones periodically ──
            if idx - last_zone_rebuild >= self.zone_rebuild_interval:
                if self.zone_window_days > 0:
                    # Sliding window: only data from the last N days
                    bar_ts = int(bar["timestamp"]) if "timestamp" in df_15m.columns else idx
                    # Normalize to ms timestamp
                    if bar_ts < 1e10:
                        bar_ts = bar_ts * 1000  # seconds → ms
                    window_ms = self.zone_window_days * 86400 * 1000
                    cutoff_ms = bar_ts - window_ms

                    sliced_data = {}
                    for tf, df in data.items():
                        ts_col = df["timestamp"]
                        # ts_col might be in ms or seconds — normalize to same unit
                        ts_in_seconds = ts_col.iloc[0] < 1e10
                        if ts_in_seconds:
                            cutoff = cutoff_ms // 1000  # seconds
                            upper = bar_ts // 1000     # seconds
                        else:
                            cutoff = cutoff_ms  # ms
                            upper = bar_ts       # ms
                        # Window: [cutoff, upper] — no look-ahead
                        mask = (ts_col >= cutoff) & (ts_col <= upper)
                        sliced_data[tf] = df.loc[mask].copy()
                else:
                    # Original behavior: all data up to current index
                    sliced_data = {tf: df.iloc[:idx+1] for tf, df in data.items()}
                current_zones = self.zone_detector.detect_all(sliced_data)
                last_zone_rebuild = idx

                # ── Update regime detection on each zone rebuild ──
                _df_1h = sliced_data.get("1h")
                if _df_1h is not None and len(_df_1h) > 50:
                    _h = _df_1h["high"].to_numpy(dtype=np.float64)
                    _l = _df_1h["low"].to_numpy(dtype=np.float64)
                    _c = _df_1h["close"].to_numpy(dtype=np.float64)
                    self.regime_detector.update(_h, _l, _c)

                month_key = to_dt(bar_ts).strftime("%Y-%m")
                monthly_rebuild_zones[month_key] = monthly_rebuild_zones.get(month_key, 0) + len(current_zones)
                monthly_rebuild_count[month_key] = monthly_rebuild_count.get(month_key, 0) + 1

            # ── Manage open trades ──
            still_open: list[BacktestTrade] = []
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

                # ── Nunchi trailing stops (CryptoMecha01) ──
                if not exit_now and trade.entry_price != trade.stop_price:
                    r_value = abs(trade.entry_price - trade.stop_price)
                    if r_value > 0:
                        if trade.direction == "LONG":
                            unrealized_r = (bar_high - trade.entry_price) / r_value
                            # +1R Breakeven: move SL to entry
                            if not trade.breakeven_active and unrealized_r >= 1.0:
                                trade.breakeven_active = True
                                trade.stop_price = trade.entry_price
                            # +1.5R Trailing: trail to nearest demand zone above current SL
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

                    # P&L calculation
                    if trade.direction == "LONG":
                        pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
                    else:
                        pnl_pct = (trade.entry_price - exit_price) / trade.entry_price

                    trade.pnl = trade.risk_dollars * (pnl_pct / trade.initial_risk_pct) if trade.initial_risk_pct > 0 else 0.0
                    trade.pnl_r = trade.pnl / trade.risk_dollars if trade.risk_dollars > 0 else 0

                    equity += trade.pnl
                    self.risk_manager.record_trade_result(trade.pnl, to_dt(bar_ts).strftime("%Y-%m-%d"))
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
                    logger.warning(
                        f"HALTED: Drawdown {current_dd:.1f}% >= {max_dd_threshold}% "
                        f"on {halt_date} at bar {idx}"
                    )

            # ── Generate entry signals ──
            month_key = to_dt(bar_ts).strftime("%Y-%m")
            bar_audit = PipelineAudit()

            if not halted:
                # Only use higher-TF zones (1h+) for entry
                htf_zones = [z for z in current_zones if z.timeframe in ("1h", "4h", "1d", "1w")]
                if not htf_zones and current_zones:
                    htf_zones = current_zones  # Fallback: use all zones

                # Track whether bar close is near any zone
                any_near = any(
                    z.distance_to_price(bar_close) <= self.signal_engine.zone_distance_pct
                    for z in htf_zones
                )
                if any_near:
                    monthly_zones_near_bar[month_key] = monthly_zones_near_bar.get(month_key, 0) + 1

                signals = self.signal_engine.scan(bar_close, htf_zones, df_15m, idx, audit=bar_audit)

                for sig in signals:
                    # Cooldown check
                    zone_key = f"{sig.zone.zone_type.value}_{sig.zone.timeframe}_{sig.zone.midpoint:.0f}"
                    last_exit = zone_cooldowns.get(zone_key, -cooldown_bars - 1)
                    if idx - last_exit < cooldown_bars:
                        continue

                    # Position overlap prevention: only one trade per direction
                    open_dirs = {t.direction for t in open_trades}
                    if sig.direction.value in open_dirs:
                        continue

                    # Validate signal
                    if sig.entry_price <= 0 or sig.stop_price <= 0:
                        continue
                    if sig.direction == SignalDirection.LONG and sig.stop_price >= sig.entry_price:
                        continue
                    if sig.direction == SignalDirection.SHORT and sig.stop_price <= sig.entry_price:
                        continue

                    # ── Regime filter ──
                    if not self.regime_detector.allows_direction(sig.direction.value):
                        logger.debug(
                            "Regime filter: %s signal filtered (regime=%s)",
                            sig.direction.value, self.regime_detector.regime.value,
                        )
                        continue
                    # Apply regime confidence multiplier to win_prob estimate
                    regime_mult = self.regime_detector.get_confidence_multiplier(sig.direction.value)

                    # Position sizing
                    leverage = float(self.settings.trading.max_leverage)
                    win_prob = (0.40 + (sig.confidence * 0.15)) * regime_mult  # 0.40–0.55 range × regime mult
                    # Cap win_prob to reasonable range
                    win_prob = min(max(win_prob, 0.10), 0.90)
                    risk_dollars, qty_btc = self.risk_manager.calculate_size(
                        win_prob=win_prob,
                        reward_risk=sig.risk_reward_ratio,
                        entry=sig.entry_price,
                        stop=sig.stop_price,
                        leverage=leverage,
                        atr_ratio=self.regime_detector.atr_ratio,
                    )

                    if risk_dollars <= 0 or qty_btc <= 0:
                        continue

                    if not self.risk_manager.can_open(risk_dollars):
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

                    # Track trades taken per month
                    monthly_trades_taken[month_key] = monthly_trades_taken.get(month_key, 0) + 1

            # Aggregate bar audit into monthly
            if bar_audit.zones_total > 0:
                if month_key not in monthly_audit:
                    monthly_audit[month_key] = PipelineAudit()
                monthly_audit[month_key].merge(bar_audit)

            # Track equity curve even when no trade exits
            if idx % 96 == 0 and equity_curve[-1] != equity:
                pass  # Equity already tracked on close

            # Progress logging
            if time.time() - last_print > 10.0:
                pct = (idx - first_live_bar) / (n_bars - first_live_bar) * 100
                dt_str = to_dt(bar_ts).strftime("%Y-%m-%d")
                logger.info(
                    f"  [{pct:5.1f}%] {dt_str} | "
                    f"Equity: ${equity:,.2f} | "
                    f"Open: {len(open_trades)} | "
                    f"Closed: {len(closed_trades)}"
                )
                last_print = time.time()

        # Close any remaining open trades at last price
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
                self.risk_manager.record_trade_result(trade.pnl, to_dt(last_ts).strftime("%Y-%m-%d"))
                closed_trades.append(trade)

        # ── Compute Metrics ──
        all_trades = closed_trades
        wins = [t for t in all_trades if t.status == "won"]
        losses = [t for t in all_trades if t.status == "lost"]
        expired = [t for t in all_trades if t.status == "expired"]

        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1.0

        true_pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
        total_pnl = gross_profit - gross_loss if losses else gross_profit
        final_equity = self.settings.risk.initial_capital + total_pnl
        total_return = (final_equity / self.settings.risk.initial_capital - 1) * 100
        win_rate = len(wins) / len(all_trades) if all_trades else 0

        # Sortino & Sharpe
        returns_list = [t.pnl_r for t in all_trades]  # Returns in R
        sortino = self._sortino(returns_list)
        sharpe = self._sharpe(returns_list)

        # Max drawdown from equity curve
        peaks = np.maximum.accumulate(equity_curve) if equity_curve else np.array([equity])
        drawdowns = (peaks - equity_curve) / peaks * 100
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Average R
        avg_r = np.mean(returns_list) if returns_list else 0.0
        avg_r_win = np.mean([t.pnl_r for t in wins]) if wins else 0.0
        avg_r_loss = np.mean([t.pnl_r for t in losses]) if losses else 0.0

        duration_days = (to_dt(end_ts) - to_dt(start_ts)).days

        elapsed = time.time() - t_start
        logger.info(f"Backtest complete in {elapsed:.1f}s. {len(all_trades)} trades, True PF: {true_pf:.3f}")

        # ── Print pipeline audit results ──
        print()
        print("=" * 80)
        print("PIPELINE AUDIT RESULTS")
        print("=" * 80)

        # Collect all months and sort
        all_months = sorted(set(list(monthly_audit.keys()) + list(monthly_rebuild_zones.keys()) +
                                list(monthly_zones_near_bar.keys()) + list(monthly_trades_taken.keys())))

        # Column headers
        print(f"{'Month':<10} {'Z/Rebld':>8} {'NearPx':>7} {'Engulf':>7} {'PassRR':>7} {'Signal':>7} {'Trade':>7}")
        print("-" * 60)

        for month in all_months:
            audit = monthly_audit.get(month, None)
            rebuild_zones = monthly_rebuild_zones.get(month, 0)
            rebuild_count = monthly_rebuild_count.get(month, 0)
            zones_per = round(rebuild_zones / rebuild_count, 1) if rebuild_count > 0 else 0
            near_bar = monthly_zones_near_bar.get(month, 0)
            engulf = audit.zones_near_price if audit else 0  # zones_near_price = zones that entered the engulf pipeline
            rr = audit.rr_passed if audit else 0
            sig = audit.signals_generated if audit else 0
            trades = monthly_trades_taken.get(month, 0)

            print(f"{month:<10} {zones_per:>8} {near_bar:>7} {engulf:>7} {rr:>7} {sig:>7} {trades:>7}")

        print("-" * 60)
        print()

        # ── Engulf rejection reasons ──
        # Aggregate across all months
        total_bullish_not_at_zone = sum(v.bullish_price_not_at_zone for v in monthly_audit.values())
        total_bullish_breakdown = sum(v.bullish_breakdown for v in monthly_audit.values())
        total_bullish_low_conf = sum(v.bullish_low_confidence for v in monthly_audit.values())
        total_bearish_not_at_zone = sum(v.bearish_price_not_at_zone for v in monthly_audit.values())
        total_bearish_breakout = sum(v.bearish_breakout for v in monthly_audit.values())
        total_bearish_low_conf = sum(v.bearish_low_confidence for v in monthly_audit.values())
        total_engulf_matched = sum(v.engulf_matched for v in monthly_audit.values())
        total_demand_checked = sum(v.demand_zones_checked for v in monthly_audit.values())
        total_supply_checked = sum(v.supply_zones_checked for v in monthly_audit.values())

        # Confidence component totals
        total_comp_close = sum(v.comp_close_direction for v in monthly_audit.values())
        total_comp_engulfs = sum(v.comp_engulfs_prev for v in monthly_audit.values())
        total_comp_body = sum(v.comp_strong_body for v in monthly_audit.values())
        total_comp_wick = sum(v.comp_wick_rejection for v in monthly_audit.values())
        total_comp_vol = sum(v.comp_vol_surge for v in monthly_audit.values())
        total_comp_zone = sum(v.comp_zone_score for v in monthly_audit.values())

        print("Engulf Rejection Reasons (all months):")
        print(f"  Demand zones checked (near price): {total_demand_checked}")
        print(f"    -> price_not_at_zone:     {total_bullish_not_at_zone}")
        print(f"    -> breakdown:             {total_bullish_breakdown}")
        print(f"    -> low_confidence:        {total_bullish_low_conf}")
        print(f"  Supply zones checked (near price): {total_supply_checked}")
        print(f"    -> price_not_at_zone:     {total_bearish_not_at_zone}")
        print(f"    -> breakout:              {total_bearish_breakout}")
        print(f"    -> low_confidence:        {total_bearish_low_conf}")
        print(f"  Total engulf matched (signal returned): {total_engulf_matched}")
        print()

        print("Confidence Component Contributions (all months):")
        print(f"  close_direction contributed:  {total_comp_close:>6} times")
        print(f"  engulfs_prev contributed:     {total_comp_engulfs:>6} times")
        print(f"  strong_body contributed:      {total_comp_body:>6} times")
        print(f"  wick_rejection contributed:   {total_comp_wick:>6} times")
        print(f"  vol_surge contributed:        {total_comp_vol:>6} times")
        print(f"  zone_score contributed:       {total_comp_zone:>6} times")
        print(f"  (Total engulf checks: {total_demand_checked + total_supply_checked})")
        print()

        # ── Regime summary ──
        regime = self.regime_detector
        print("=== REGIME DETECTOR ===")
        print(f"  Final regime:      {regime.regime.value}")
        print(f"  ADX:               {regime.adx:.1f} (+DI: {regime.plus_di:.1f}, -DI: {regime.minus_di:.1f})")
        print(f"  Price vs EMA200:   {regime.price_vs_ema_pct:+.2f}%")
        print(f"  Vol regime:        {regime.vol_regime.value} (ATR ratio: {regime.atr_ratio:.2f})")
        print()

        # ── Primary cause analysis ──
        # Zone detection: if zones_per_rebuild drops significantly post-2021
        post_may = [m for m in all_months if m >= "2021-06"]
        pre_jun = [m for m in all_months if m < "2021-06"]
        post_avg_zones = sum(monthly_rebuild_zones.get(m, 0) for m in post_may) / max(sum(monthly_rebuild_count.get(m, 0) for m in post_may), 1)
        pre_avg_zones = sum(monthly_rebuild_zones.get(m, 0) for m in pre_jun) / max(sum(monthly_rebuild_count.get(m, 0) for m in pre_jun), 1)

        # Near-price ratio
        post_near = sum(monthly_zones_near_bar.get(m, 0) for m in post_may)
        pre_near = sum(monthly_zones_near_bar.get(m, 0) for m in pre_jun)

        print("=== DIAGNOSIS ===")
        print()
        # Determine primary cause
        has_zones = post_avg_zones > 0
        has_near_price = post_near > 0
        post_engulf_matched = sum(v.engulf_matched for v in monthly_audit.values() if any(k == v for k in monthly_audit) or True)  # all post-may audit
        post_engulf = sum(monthly_audit[m].engulf_matched for m in post_may if m in monthly_audit)
        post_rr = sum(monthly_audit[m].rr_passed for m in post_may if m in monthly_audit)
        post_signals = sum(monthly_audit[m].signals_generated for m in post_may if m in monthly_audit)
        post_trades = sum(monthly_trades_taken.get(m, 0) for m in post_may)

        if not has_zones:
            print("PRIMARY CAUSE: Zone Detection — Zero zones detected after May 2021.")
            print()
        elif not has_near_price:
            print("PRIMARY CAUSE: Zone Detection — Zones exist but none are near price after May 2021.")
            print(f"  Pre-Jun avg zones/rebuild: {pre_avg_zones:.1f}")
            print(f"  Post-May avg zones/rebuild: {post_avg_zones:.1f}")
            print(f"  Post-May bars with nearby zones: {post_near}")
            print()
        elif post_engulf == 0:
            print("PRIMARY CAUSE: Engulf Matching — Zones near price but no engulf pattern detected after May 2021.")
            print()
        elif post_rr == 0:
            print("PRIMARY CAUSE: RR Filter — Engulf patterns detected but none passed the R:R threshold after May 2021.")
            print()
        elif post_signals > 0 and post_trades == 0:
            print("PRIMARY CAUSE: Post-RR Filters — Signals generated but none became trades (cooldown/overlap/sizing).")
            print()
        else:
            print(f"Post-May 2021 trades found: {post_trades}")
            print(f"Pre-Jun 2021 trades found: {sum(monthly_trades_taken.get(m, 0) for m in pre_jun)}")
            print()

        print(f"Pre-Jun 2021 avg zones/rebuild: {pre_avg_zones:.1f}")
        print(f"Post-May 2021 avg zones/rebuild: {post_avg_zones:.1f}")
        print(f"Post-May 2021 engulf matched: {post_engulf}")
        print(f"Post-May 2021 RR passed: {post_rr}")
        print(f"Post-May 2021 signals generated: {post_signals}")
        print(f"Post-May 2021 trades taken: {post_trades}")
        print()

        return BacktestResult(
            total_trades=len(all_trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            expired_trades=len(expired),
            win_rate=win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            total_pnl=total_pnl,
            true_profit_factor=true_pf,
            sortino_ratio=sortino,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            avg_r_per_trade=avg_r,
            avg_r_win=avg_r_win,
            avg_r_loss=avg_r_loss,
            final_equity=final_equity,
            total_return_pct=total_return,
            equity_curve=equity_curve,
            trades=all_trades,
            start_date=str(to_dt(start_ts).date()),
            end_date=str(to_dt(end_ts).date()),
            duration_days=duration_days,
            candles_processed=n_bars,
            halted=halted,
            halt_reason=f"MDD >= {max_dd_threshold}%" if halted else "",
            halt_date=halt_date,
        )

    @staticmethod
    def _sortino(returns: list[float], target: float = 0.0) -> float:
        """Sortino ratio: mean return / std of negative returns only."""
        if not returns:
            return 0.0
        arr = np.array(returns)
        mu = np.mean(arr)
        downside = arr[arr < target]
        if len(downside) < 2:
            return 0.0
        downside_std = np.std(downside, ddof=1)
        return float((mu - target) / downside_std) if downside_std > 0 else 0.0

    @staticmethod
    def _sharpe(returns: list[float], risk_free: float = 0.0) -> float:
        """Sharpe ratio: (mean return - rf) / std of all returns."""
        if len(returns) < 2:
            return 0.0
        arr = np.array(returns)
        mu = np.mean(arr)
        std = np.std(arr, ddof=1)
        return float((mu - risk_free) / std) if std > 0 else 0.0
