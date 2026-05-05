"""
Oda Risk Manager — Half-Kelly position sizing and exposure control.

Adapted from CryptoMecha01 strategy/risk_manager.py (Z's prior work).
Implements fractional Kelly sizing, global exposure caps, maximum drawdown
hard stops, and trade logging with win-rate tracking.

Usage:
    from oda.config import Settings
    from oda.risk import RiskManager, Position

    settings = Settings.load()
    rm = RiskManager(settings.risk)

    # Calculate position size
    risk_dollars, qty_btc = rm.calculate_size(
        win_prob=0.55, reward_risk=2.0, entry=90000.0, stop=89500.0, leverage=3.0
    )

    if rm.can_open(risk_dollars):
        pos = Position(
            symbol="BTCUSDT", direction="LONG",
            entry_price=entry, stop_price=stop,
            quantity_btc=qty_btc, risk_dollars=risk_dollars, leverage=3.0,
        )
        rm.open_position(pos)

    pnl = rm.close_position("BTCUSDT", exit_price=91000.0, reason="target")
    stats = rm.get_stats()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from oda.config import RiskConfig

logger = logging.getLogger("oda.risk")


# ── Data Structures ─────────────────────────────────────────────────

@dataclass
class Position:
    """An open trading position with risk metadata."""

    symbol: str
    direction: str          # 'LONG' or 'SHORT'
    entry_price: float
    stop_price: float
    quantity_btc: float     # base (margin) quantity; effective size = qty * leverage
    risk_dollars: float     # maximum loss if stopped out
    leverage: float


@dataclass
class TradeRecord:
    """A completed (closed) trade for performance tracking."""

    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    stop_price: float
    quantity_btc: float
    risk_dollars: float
    leverage: float
    pnl: float              # realised P&L in dollars
    pnl_r: float            # P&L in units of R (pnl / risk_dollars)
    reason: str             # exit reason: target, stop, signal, manual, etc.


@dataclass
class RiskState:
    """Snapshot of risk manager state for reporting."""

    equity: float
    peak_equity: float
    max_drawdown_pct: float
    open_positions: List[Position]
    total_trades: int
    winning_trades: int
    losing_trades: int


# ── Risk Manager ────────────────────────────────────────────────────

class RiskManager:
    """
    Portfolio risk manager with Half-Kelly position sizing.

    Core responsibilities:
    - Kelly-based position sizing: f* = (p*b - q) / b, halved by kelly_fraction
    - Per-trade risk cap (default 2 % of equity)
    - Global exposure cap (default 5 % of equity across all positions)
    - Maximum drawdown hard stop (default 10 %)
    - Trade log with win rate and average-R tracking
    """

    def __init__(
        self,
        config: RiskConfig,
        initial_equity: Optional[float] = None,
    ) -> None:
        """
        Args:
            config:         Frozen RiskConfig from oda.config.Settings.
            initial_equity: Override the config's initial_capital (useful for
                            resuming after a previous session).
        """
        self._config = config

        equity = (
            initial_equity
            if initial_equity is not None
            else config.initial_capital
        )
        self._equity: float = equity
        self._peak_equity: float = equity
        self._max_drawdown_pct: float = 0.0

        self._open_positions: List[Position] = []
        self._trade_log: List[TradeRecord] = []

        self._total_trades: int = 0
        self._winning_trades: int = 0
        self._losing_trades: int = 0

        # Daily loss tracking
        self._current_date: Optional[str] = None
        self._daily_pnl: float = 0.0
        self._daily_limit_logged: bool = False  # Suppress repeated daily-loss warnings

    # ── Position Sizing ─────────────────────────────────────────────

    def calculate_size(
        self,
        win_prob: float,
        reward_risk: float,
        entry: float,
        stop: float,
        leverage: float = 1.0,
        atr_ratio: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calculate Half-Kelly position size with optional volatility scaling.

        Kelly criterion:  f* = (p * b - q) / b

        where  p = win_prob,  b = reward_risk,  q = 1 - p.

        The result is multiplied by ``kelly_fraction`` (default 0.5 for
        Half-Kelly) and then capped at ``max_risk_per_trade_pct`` of equity.

        When ``vol_scale_enabled`` is True and ``atr_ratio`` is provided,
        the position size is scaled inversely to the ATR ratio. This means
        position sizes shrink when volatility is elevated (atr_ratio > 1.0)
        and increase when volatility is suppressed (atr_ratio < 1.0), within
        configurable limits (vol_scale_min / vol_scale_max).

        Args:
            win_prob:    Estimated win probability (0.0 – 1.0).
            reward_risk: Reward-to-risk ratio (e.g. 2.0 = 2:1).
            entry:       Entry price.
            stop:        Stop-loss price.
            leverage:    Account leverage multiplier (≥ 1.0).
            atr_ratio:   Current ATR / ATR_SMA ratio (from regime detector).
                         Used for inverse-volatility scaling. Ignored when
                         vol_scale_enabled is False or None.

        Returns:
            (risk_dollars, quantity_btc) tuple.  Both are 0.0 when the Kelly
            fraction is non-positive or parameters are invalid.
        """
        # ── guard invalid inputs ────────────────────────────────────
        if not (0.0 < win_prob < 1.0 and reward_risk > 0.0
                and entry > 0.0 and stop > 0.0 and leverage > 0.0):
            logger.warning(
                "calculate_size: invalid parameters "
                f"(wp={win_prob}, rr={reward_risk}, e={entry}, s={stop}, l={leverage})"
            )
            return 0.0, 0.0

        p = win_prob
        b = reward_risk
        q = 1.0 - p

        # Kelly fraction: f* = (p*b - q) / b
        kelly_f = (p * b - q) / b

        if kelly_f <= 0.0:
            logger.info(
                f"Kelly f* = {kelly_f:.4f} — no positive edge, skipping."
            )
            return 0.0, 0.0

        # Min EV threshold: expected value in R = p*b - q
        ev_r = p * b - q
        if ev_r < self._config.min_ev_threshold:
            logger.info(
                f"EV in R = {ev_r:.4f} < min_ev_threshold "
                f"({self._config.min_ev_threshold}) — skipping."
            )
            return 0.0, 0.0

        # Apply fractional multiplier (Half-Kelly by default)
        suggested_risk_pct = kelly_f * self._config.kelly_fraction

        # ── Volatility scaling (inverse ATR) — applied BEFORE the per-trade cap ──
        # When vol is high (atr_ratio > 1), shrink position size.
        # When vol is low (atr_ratio < 1), grow position size (capped).
        # Applying scaling before the per-trade cap ensures that can_open()
        # correctly validates the final size without rejecting vol-scaled trades.
        if self._config.vol_scale_enabled and atr_ratio is not None and atr_ratio > 0.0:
            vol_scale = 1.0 / atr_ratio
            vol_scale = max(vol_scale, self._config.vol_scale_min)
            vol_scale = min(vol_scale, self._config.vol_scale_max)
            suggested_risk_pct *= vol_scale
            logger.debug(
                "Vol scaling: atr_ratio=%.3f  scale=%.3f  adjusted_suggested_pct=%.4f%%",
                atr_ratio, vol_scale, suggested_risk_pct * 100,
            )

        # Cap at per-trade maximum (applied AFTER vol scaling)
        final_risk_pct = min(
            suggested_risk_pct, self._config.max_risk_per_trade_pct
        )

        risk_dollars = self._equity * final_risk_pct

        # quantity_btc is the *margin* quantity; effective position = qty * leverage.
        # At the stop price we lose exactly risk_dollars:
        #   loss = qty * leverage * |entry - stop| = risk_dollars
        #   → qty = risk_dollars / |entry - stop| / leverage
        stop_distance = abs(entry - stop)
        if stop_distance == 0.0:
            logger.warning("Stop distance is zero — cannot compute quantity.")
            return risk_dollars, 0.0

        quantity_btc = risk_dollars / stop_distance / leverage

        logger.debug(
            "Size calc: wp=%.2f  rr=%.1f  kelly=%.4f  "
            "risk_pct=%.2f%%  risk=$%.2f  qty=%.6f",
            p, b, kelly_f, final_risk_pct * 100, risk_dollars, quantity_btc,
        )

        return risk_dollars, quantity_btc

    # ── Exposure Checks ─────────────────────────────────────────────

    def can_open(self, risk_dollars: float) -> bool:
        """
        Return True if a new position can be opened without breaching limits.

        Checks, in order:
        1. Drawdown hard stop (≥ configured max_drawdown_pct).
        2. Per-trade risk cap.
        3. Global exposure cap.
        """
        # 1. Drawdown hard stop
        max_dd = self._config.max_drawdown_pct / 100.0
        if self._max_drawdown_pct >= max_dd:
            logger.warning(
                "can_open: drawdown hard stop (%s). "
                "Current MDD: %.2f%% ≥ %.1f%%",
                "rejected",
                self._max_drawdown_pct * 100,
                self._config.max_drawdown_pct,
            )
            return False

        # 2. Per-trade risk cap
        max_per_trade = self._equity * self._config.max_risk_per_trade_pct
        if risk_dollars > max_per_trade:
            logger.warning(
                "can_open: per-trade cap exceeded "
                "($%.2f > $%.2f)", risk_dollars, max_per_trade,
            )
            return False

        # 3. Global exposure cap
        current_exposure = sum(p.risk_dollars for p in self._open_positions)
        max_global = self._equity * self._config.global_exposure_cap_pct

        if (current_exposure + risk_dollars) > max_global:
            logger.warning(
                "can_open: global exposure cap exceeded "
                "($%.2f + $%.2f = $%.2f > $%.2f)",
                current_exposure, risk_dollars,
                current_exposure + risk_dollars, max_global,
            )
            return False

        # 4. Daily loss limit
        daily_pct = self._config.daily_loss_pct / 100.0
        if daily_pct > 0.0 and self._daily_pnl <= -self._equity * daily_pct:
            if not self._daily_limit_logged:
                logger.warning(
                    "can_open: daily loss limit reached "
                    "(daily PnL: $%.2f, limit: $%.2f)",
                    self._daily_pnl, self._equity * daily_pct,
                )
                self._daily_limit_logged = True
            return False

        return True

    # ── Position Lifecycle ──────────────────────────────────────────

    def open_position(self, position: Position) -> None:
        """
        Register a newly opened position.

        The caller should already have called :meth:`can_open` to validate.
        """
        self._open_positions.append(position)
        logger.info(
            "Position opened: %s %s  entry=%.2f  stop=%.2f  "
            "qty=%.6f  risk=$%.2f",
            position.symbol,
            position.direction,
            position.entry_price,
            position.stop_price,
            position.quantity_btc,
            position.risk_dollars,
        )

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "manual",
    ) -> float:
        """
        Close an open position and record the trade outcome.

        Args:
            symbol:     Symbol to close (must match exactly one open position).
            exit_price: Price at which the position was closed.
            reason:     Reason for closing (target, stop, signal, manual, …).

        Returns:
            Realised P&L in dollars.  Returns 0.0 if the symbol is not found.

        Raises:
            ValueError: If multiple positions exist for the same symbol.
        """
        matches = [p for p in self._open_positions if p.symbol == symbol]

        if not matches:
            logger.warning("close_position: no open position for %s", symbol)
            return 0.0

        if len(matches) > 1:
            raise ValueError(
                f"Multiple open positions for {symbol!r} — "
                f"use a more specific identifier."
            )

        pos = matches[0]
        self._open_positions.remove(pos)

        # ── Calculate P&L ───────────────────────────────────────────
        if pos.direction == "LONG":
            pnl = pos.quantity_btc * pos.leverage * (exit_price - pos.entry_price)
        elif pos.direction == "SHORT":
            pnl = pos.quantity_btc * pos.leverage * (pos.entry_price - exit_price)
        else:
            raise ValueError(
                f"Unknown direction {pos.direction!r} — expected LONG or SHORT."
            )

        pnl_r = pnl / pos.risk_dollars if pos.risk_dollars != 0.0 else 0.0

        # ── Record the trade ────────────────────────────────────────
        record = TradeRecord(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            stop_price=pos.stop_price,
            quantity_btc=pos.quantity_btc,
            risk_dollars=pos.risk_dollars,
            leverage=pos.leverage,
            pnl=pnl,
            pnl_r=pnl_r,
            reason=reason,
        )
        self._trade_log.append(record)

        # ── Update statistics ───────────────────────────────────────
        self._total_trades += 1
        if pnl > 0.0:
            self._winning_trades += 1
        elif pnl < 0.0:
            self._losing_trades += 1
        # pnl == 0.0 is a scratch trade — counted but neither win nor loss.

        # Update equity and drawdown tracking
        self._update_equity(pnl)

        logger.info(
            "Position closed: %s %s  entry=%.2f  exit=%.2f  "
            "pnl=$%.2f (%+.2fR)  reason=%s",
            symbol, pos.direction, pos.entry_price, exit_price,
            pnl, pnl_r, reason,
        )

        return pnl

    # ── Statistics ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Return a dictionary of risk and performance statistics.

        Keys
        ----
        equity : float
            Current realised equity.
        peak_equity : float
            Highest equity ever reached.
        max_drawdown_pct : float
            Maximum peak-to-trough drawdown (percentage, 0–100).
        current_drawdown_pct : float
            Current drawdown from peak (percentage).
        open_positions : int
            Number of currently open positions.
        total_trades : int
            Total closed trades.
        winning_trades : int
            Number of winning trades.
        losing_trades : int
            Number of losing trades.
        win_rate : float
            Win rate (0.0 – 1.0).
        avg_r_per_trade : float
            Average P&L per trade in units of R.
        total_pnl : float
            Cumulative realised P&L in dollars.
        current_exposure_pct : float
            Current total risk as percentage of equity.
        """
        total_pnl = sum(t.pnl for t in self._trade_log)

        win_rate = (
            self._winning_trades / self._total_trades
            if self._total_trades > 0
            else 0.0
        )

        avg_r = 0.0
        if self._total_trades > 0:
            total_r = sum(t.pnl_r for t in self._trade_log)
            avg_r = total_r / self._total_trades

        current_drawdown = 0.0
        if self._peak_equity > 0.0:
            current_drawdown = (
                (self._peak_equity - self._equity) / self._peak_equity
            )

        current_exposure = sum(p.risk_dollars for p in self._open_positions)
        current_exposure_pct = (
            (current_exposure / self._equity) * 100.0
            if self._equity > 0.0
            else 0.0
        )

        return {
            "equity": self._equity,
            "peak_equity": self._peak_equity,
            "max_drawdown_pct": self._max_drawdown_pct * 100.0,
            "current_drawdown_pct": current_drawdown * 100.0,
            "open_positions": len(self._open_positions),
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "losing_trades": self._losing_trades,
            "win_rate": win_rate,
            "avg_r_per_trade": avg_r,
            "total_pnl": total_pnl,
            "current_exposure_pct": current_exposure_pct,
        }

    # ── Daily Loss Tracking ─────────────────────────────────────────

    def set_current_date(self, date_str: str) -> None:
        """Update the current trading date for daily loss tracking.

        Resets daily P&L when the date changes.

        Args:
            date_str: Date string in YYYY-MM-DD format.
        """
        if self._current_date is None:
            self._current_date = date_str
        elif date_str != self._current_date:
            logger.debug(
                "Daily P&L reset: $%.2f → $0.00 (date: %s → %s)",
                self._daily_pnl, self._current_date, date_str,
            )
            self._current_date = date_str
            self._daily_pnl = 0.0
            self._daily_limit_logged = False

    def record_trade_result(self, pnl: float, date_str: Optional[str] = None) -> None:
        """Record a trade's P&L for daily loss tracking.

        Args:
            pnl: Realised P&L in dollars.
            date_str: Optional date override. If not provided, uses current date.
        """
        if date_str is not None:
            self.set_current_date(date_str)
        self._daily_pnl += pnl
        logger.debug(
            "Daily P&L: $%.2f (after trade $%+.2f)",
            self._daily_pnl, pnl,
        )

    # ── Properties ──────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        """Current realised equity."""
        return self._equity

    @property
    def peak_equity(self) -> float:
        """Highest equity ever reached."""
        return self._peak_equity

    @property
    def max_drawdown_pct(self) -> float:
        """Maximum drawdown as a fraction (0.0 – 1.0)."""
        return self._max_drawdown_pct

    @property
    def open_positions(self) -> List[Position]:
        """Shallow copy of open positions list."""
        return list(self._open_positions)

    @property
    def trade_log(self) -> List[TradeRecord]:
        """Shallow copy of the trade log."""
        return list(self._trade_log)

    # ── Internal Helpers ────────────────────────────────────────────

    def _update_equity(self, realized_pnl: float) -> None:
        """Update equity after a trade closes and track peak / max drawdown."""
        self._equity += realized_pnl

        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        if self._peak_equity > 0.0:
            current_dd = (
                (self._peak_equity - self._equity) / self._peak_equity
            )
            if current_dd > self._max_drawdown_pct:
                self._max_drawdown_pct = current_dd

        logger.debug(
            "Equity: $%,.2f  |  Peak: $%,.2f  |  MDD: %.2f%%",
            self._equity,
            self._peak_equity,
            self._max_drawdown_pct * 100,
        )
