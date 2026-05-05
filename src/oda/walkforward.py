"""
Oda Walk-Forward Validation Framework
======================================

Walk-forward analysis (WFA) is the gold standard for evaluating
trading strategy robustness. It simulates how a strategy would have
performed if deployed live across sequential market regimes.

Methodology
-----------
1. Split historical data into N sequential folds.
2. For each fold i:
   a. In-Sample (IS): All data from start to fold_i_train_end (expanding window)
   b. Out-Of-Sample (OOS): Data from fold_i_train_end to fold_i_end
   c. Optimize parameters on IS (confidence threshold sweep)
   d. Test best parameter on OOS (unseen data)
3. Aggregate OOS metrics across all folds for realistic performance estimate.

The key insight: if the strategy's optimal parameters are stable across
folds and OOS performance is consistent with IS, the strategy is robust.
If optimal parameters jump around or OOS is much worse than IS, the
strategy is overfit to specific market conditions.

Usage:
    python -m oda backtest --symbol BTCUSDT --walk-forward --folds 3

References:
    - Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies"
    - Walk-forward analysis is the closest simulation to live trading without
      actually trading live.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from oda.config import Settings
from oda.backtest import BacktestEngine, BacktestResult

# Add stdout handler for visible progress
import logging as _logging
_handler = _logging.StreamHandler(sys.stdout)
_handler.setLevel(_logging.INFO)
_handler.setFormatter(_logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s'))
# Apply to oda loggers
for _name in ('oda.backtest', 'oda.walkforward', 'oda.risk', 'oda.signals', 'oda.regime', 'oda.zones'):
    _l = _logging.getLogger(_name)
    _l.setLevel(_logging.INFO)
    _l.addHandler(_handler)


# ── Data Classes ─────────────────────────────────────────────────────


@dataclass
class WalkForwardFold:
    """Results for one walk-forward fold."""

    fold_id: int
    # Date ranges as ISO strings
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str

    # IS (in-sample) results per confidence value
    is_confidence_sweep: dict[float, BacktestResult] = field(default_factory=dict)
    is_best_confidence: float = 0.0
    is_best_pf: float = 0.0

    # OOS (out-of-sample) results — the metric that matters
    oos_result: Optional[BacktestResult] = None
    oos_confidence_used: float = 0.0

    @property
    def oos_pf(self) -> float:
        return self.oos_result.true_profit_factor if self.oos_result else 0.0

    @property
    def oos_win_rate(self) -> float:
        return self.oos_result.win_rate if self.oos_result else 0.0

    @property
    def oos_return_pct(self) -> float:
        return self.oos_result.total_return_pct if self.oos_result else 0.0

    @property
    def oos_trades(self) -> int:
        return self.oos_result.total_trades if self.oos_result else 0

    def summary_line(self) -> str:
        """One-line summary of this fold."""
        return (
            f"Fold {self.fold_id}: "
            f"IS=({self.is_start}..{self.is_end}) "
            f"OOS=({self.oos_start}..{self.oos_end}) "
            f"best_conf={self.is_best_confidence:.2f} "
            f"OOS: {self.oos_trades}t, WR={self.oos_win_rate:.1%}, "
            f"PF={self.oos_pf:.3f}, ret={self.oos_return_pct:+.2f}%"
        )


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward results across all folds."""

    folds: list[WalkForwardFold] = field(default_factory=list)
    total_folds: int = 0
    total_duration_days: int = 0

    # Aggregate OOS metrics
    oos_total_trades: int = 0
    oos_total_wins: int = 0
    oos_total_losses: int = 0
    oos_aggregate_return_pct: float = 0.0
    oos_avg_win_rate: float = 0.0
    oos_avg_pf: float = 0.0

    # Parameter stability
    confidence_values: list[float] = field(default_factory=list)
    confidence_std: float = 0.0
    confidence_unique: int = 0

    # Walk-forward efficiency
    avg_is_pf: float = 0.0
    wf_efficiency: float = 0.0  # OOS avg PF / IS avg PF

    # Gross aggregate across all folds
    @property
    def oos_weighted_pf(self) -> float:
        """Profit factor from aggregated gross P&L across all folds."""
        total_gp = sum(
            f.oos_result.gross_profit for f in self.folds if f.oos_result
        )
        total_gl = sum(
            f.oos_result.gross_loss for f in self.folds if f.oos_result
        )
        return total_gp / total_gl if total_gl > 0 else 0.0

    def summary(self) -> str:
        """Full walk-forward report."""
        lines = [
            "=" * 70,
            "WALK-FORWARD VALIDATION RESULTS",
            "=" * 70,
            f"Folds: {self.total_folds}  |  "
            f"OOS Duration: {self.total_duration_days} days",
            "",
            "─" * 70,
            "Per-Fold OOS Results",
            "─" * 70,
        ]

        for fold in self.folds:
            pf = fold.oos_pf
            wr = fold.oos_win_rate
            ret = fold.oos_return_pct
            trades = fold.oos_trades
            conf = fold.oos_confidence_used
            lines.append(
                f"  Fold {fold.fold_id}: conf={conf:.2f}  "
                f"{trades:>5}t  WR={wr:>6.1%}  PF={pf:>7.3f}  "
                f"Ret={ret:>+8.2f}%  "
                f"[{fold.oos_start} → {fold.oos_end}]"
            )

        lines.extend([
            "",
            "─" * 70,
            "Aggregate OOS Metrics (across all folds)",
            "─" * 70,
            f"  Total Trades:              {self.oos_total_trades:>6}",
            f"  Total Wins:                {self.oos_total_wins:>6}  "
            f"({self.oos_avg_win_rate:.1%})",
            f"  Total Losses:              {self.oos_total_losses:>6}",
            f"  Aggregate PF (gross):      {self.oos_weighted_pf:>8.3f}",
            f"  Average PF (per-fold):     {self.oos_avg_pf:>8.3f}",
            f"  Aggregate Return (sum):    {self.oos_aggregate_return_pct:>+8.2f}%",
            "",
        ])

        # Interpretation
        lines.extend([
            "─" * 70,
            "Interpretation",
            "─" * 70,
        ])

        if self.oos_avg_pf > 1.0:
            lines.append(
                "  ✅ Avg OOS PF > 1.0: Strategy shows positive edge OOS."
            )
        elif self.oos_avg_pf > 0.9:
            lines.append(
                "  ⚠️  Avg OOS PF 0.9-1.0: Near breakeven OOS."
            )
        else:
            lines.append(
                "  ❌ Avg OOS PF <= 0.9: Strategy does NOT have edge OOS."
            )

        if self.confidence_unique == 1:
            lines.append(
                "  ✅ Optimal confidence is stable across folds (always "
                f"{self.confidence_values[0]:.2f})."
            )
        elif self.confidence_std <= 0.05:
            lines.append(
                "  ✅ Confidence parameter is stable "
                f"(std={self.confidence_std:.3f})."
            )
        else:
            lines.append(
                "  ⚠️  Confidence varies across folds "
                f"(std={self.confidence_std:.3f}, "
                f"values={self.confidence_values})."
            )

        lines.extend([
            f"  Walk-Forward Efficiency (OOS PF / IS PF avg): "
            f"{self.wf_efficiency:.3f}",
        ])
        if self.wf_efficiency >= 0.8:
            lines.append(
                "  ✅ Efficiency >= 0.8: Parameters transfer well OOS."
            )
        elif self.wf_efficiency >= 0.5:
            lines.append(
                "  ⚠️  Efficiency 0.5-0.8: Moderate overfit risk."
            )
        else:
            lines.append(
                "  ❌ Efficiency < 0.5: High overfit risk."
            )

        lines.append("=" * 70)
        return "\n".join(lines)


# ── Walk-Forward Validator ──────────────────────────────────────────


class WalkForwardValidator:
    """Walk-forward validation framework.

    Splits data into sequential training (IS) and testing (OOS) windows,
    optimizes confidence threshold on IS, tests on OOS, and reports
    aggregate out-of-sample metrics.

    Fold structure (anchored/expanding IS):

        Fold 0:  IS=[start ....... is_end_0]  OOS=[oos_start_0 ... fold_end_0]
        Fold 1:  IS=[start ....................... oos_start_1]  OOS=[... fold_end_1]
        Fold 2:  IS=[start ......................................... oos_start_2]
        ...

    Each fold's IS includes all data before its OOS. This simulates
    retraining on all available data before testing the next period.
    """

    def __init__(
        self,
        settings: Settings,
        num_folds: int = 3,
        train_pct: float = 0.70,
        confidence_values: Optional[list[float]] = None,
        regime_enabled: bool = True,
    ) -> None:
        self.settings = settings
        self.num_folds = num_folds
        self.train_pct = train_pct
        self.confidence_values = confidence_values or [0.65, 0.75]
        self.regime_enabled = regime_enabled

    # ── Fold Construction ────────────────────────────────────────────

    def build_folds(self, data: dict[str, pd.DataFrame]) -> list[WalkForwardFold]:
        """Build sequential walk-forward folds from multi-TF data.

        Divides the full timeline into ``num_folds`` equal segments.
        For fold 0:
            IS = first ``train_pct`` of segment 0 (training portion)
            OOS = remaining ``(1-train_pct)`` of segment 0
        For fold i > 0:
            IS = all data from start to segment i start (expanding window)
            OOS = segment i (the full segment)

        This ensures:
        - The first fold has sufficient IS data for training
        - Each subsequent fold uses ALL available past data as IS
        - OOS periods are non-overlapping and sequential

        Returns
        -------
        list[WalkForwardFold]
            Fold definitions with date ranges but no results yet.
        """
        df_15m = data.get("15m")
        if df_15m is None or len(df_15m) < 1500:
            raise ValueError(
                f"Need >= 1500 bars of 15m data for walk-forward "
                f"(got {len(df_15m) if df_15m is not None else 0})"
            )

        # Use timestamp column for splitting
        if "timestamp" in df_15m.columns:
            ts_col = df_15m["timestamp"]
            start_ts = int(ts_col.iloc[0])
            end_ts = int(ts_col.iloc[-1])
        else:
            # Fallback: need to parse datetime to ms
            from datetime import timezone
            start_ts = int(df_15m["time"].iloc[0].timestamp() * 1000)
            end_ts = int(df_15m["time"].iloc[-1].timestamp() * 1000)

        to_dt = lambda ts: datetime.fromtimestamp(
            ts / 1000 if ts > 1e10 else ts, tz=timezone.utc
        )

        # Normalize to seconds
        start_s = start_ts / 1000 if start_ts > 1e10 else float(start_ts)
        end_s = end_ts / 1000 if end_ts > 1e10 else float(end_ts)

        total_span_s = end_s - start_s
        fold_span_s = total_span_s / self.num_folds

        folds: list[WalkForwardFold] = []
        for i in range(self.num_folds):
            # OOS = segment i
            oos_start_s = start_s + i * fold_span_s
            oos_end_s = start_s + (i + 1) * fold_span_s

            if i == 0:
                # Fold 0: first train_pct of segment 0 is IS, rest is OOS
                segment_duration = fold_span_s
                is_end_s = oos_start_s + segment_duration * self.train_pct
                # OOS starts after IS training portion
                actual_oos_start_s = is_end_s
            else:
                # Fold i > 0: IS = all data from start to OOS start
                is_end_s = oos_start_s
                actual_oos_start_s = oos_start_s

            fold = WalkForwardFold(
                fold_id=i,
                is_start=str(to_dt(start_s * 1000 if start_ts > 1e10 else start_s).date()),
                is_end=str(to_dt(is_end_s * 1000 if start_ts > 1e10 else is_end_s).date()),
                oos_start=str(to_dt(actual_oos_start_s * 1000 if start_ts > 1e10 else actual_oos_start_s).date()),
                oos_end=str(to_dt(oos_end_s * 1000 if start_ts > 1e10 else oos_end_s).date()),
            )
            folds.append(fold)

        return folds

    # ── Running ──────────────────────────────────────────────────────

    def run(
        self,
        data: dict[str, pd.DataFrame],
    ) -> WalkForwardResult:
        """Run complete walk-forward validation.

        For each fold sequentially:
        1. Run IS confidence sweep across confidence_values
        2. Select best confidence by PF
        3. Run OOS backtest with best confidence

        Returns aggregated WalkForwardResult.
        """
        folds = self.build_folds(data)
        logger.info(
            f"Walk-forward: {self.num_folds} folds, "
            f"conf sweep={self.confidence_values}, "
            f"regime={self.regime_enabled}"
        )

        total_start = time.time()

        for fold in folds:
            print(f"\n{'='*60}", flush=True)
            print(f"FOLD {fold.fold_id}/{self.num_folds-1}", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"  IS:   {fold.is_start} → {fold.is_end}", flush=True)
            print(f"  OOS:  {fold.oos_start} → {fold.oos_end}", flush=True)
            sys.stdout.flush()

            # ── Step 1: IS Confidence Sweep ──
            is_results: dict[float, BacktestResult] = {}
            for conf_val in self.confidence_values:
                print(f"\n  [IS sweep] confidence={conf_val:.2f}  (starting backtest...)", flush=True)
                result = self._run_backtest(
                    data,
                    start_date=fold.is_start,
                    end_date=fold.is_end,
                    confidence=conf_val,
                )
                is_results[conf_val] = result
                print(
                    f"  [IS sweep] conf={conf_val:.2f} → {result.total_trades}t, "
                    f"PF={result.true_profit_factor:.3f}, "
                    f"WR={result.win_rate:.1%}",
                    flush=True,
                )

            # Select best confidence by PF
            best_conf = max(
                is_results,
                key=lambda c: is_results[c].true_profit_factor,
            )
            best_pf = is_results[best_conf].true_profit_factor

            fold.is_confidence_sweep = is_results
            fold.is_best_confidence = best_conf
            fold.is_best_pf = best_pf

            print(f"\n  [IS winner] conf={best_conf:.2f} (PF={best_pf:.3f})", flush=True)

            # ── Step 2: OOS Backtest with best parameter ──
            print(f"\n  [OOS test] confidence={best_conf:.2f}  (starting backtest...)", flush=True)
            oos_result = self._run_backtest(
                data,
                start_date=fold.oos_start,
                end_date=fold.oos_end,
                confidence=best_conf,
            )
            fold.oos_result = oos_result
            fold.oos_confidence_used = best_conf
            print(
                f"  [OOS result] {oos_result.total_trades}t, "
                f"PF={oos_result.true_profit_factor:.3f}, "
                f"WR={oos_result.win_rate:.1%}, "
                f"Ret={oos_result.total_return_pct:+.2f}%",
                flush=True,
            )

        total_elapsed = time.time() - total_start
        print(
            f"\nWalk-forward complete in {total_elapsed:.0f}s "
            f"({total_elapsed/60:.1f} min)",
            flush=True,
        )

        # ── Aggregate Results ──
        return self._aggregate(folds)

    def _run_backtest(
        self,
        data: dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        confidence: float = 0.75,
    ) -> BacktestResult:
        """Run a single backtest with given parameters.

        Sets environment variables for the backtest engine to pick up,
        reloads Settings to reflect current env, then runs.
        """
        # Override env vars for this run
        if confidence:
            os.environ["ODA_MIN_CONFIDENCE"] = str(confidence)
        os.environ["ODA_REGIME_ENABLED"] = str(self.regime_enabled).lower()
        os.environ["ODA_MAX_DD_PCT"] = "100.0"  # No MDD halt during walk-forward
        os.environ["ODA_DAILY_LOSS_PCT"] = "100.0"

        # Reload settings to pick up env var overrides
        from oda.config import Settings as ReloadedSettings
        settings = ReloadedSettings.load()

        engine = BacktestEngine(settings)
        result = engine.run(
            data,
            start_date=start_date,
            end_date=end_date,
            burn_in_days=30,  # Shorter burn-in for windowed tests
        )
        return result

    def _aggregate(self, folds: list[WalkForwardFold]) -> WalkForwardResult:
        """Aggregate results across all folds."""
        oos_total_trades = 0
        oos_total_wins = 0
        oos_total_losses = 0
        oos_total_return = 0.0
        oos_pf_list: list[float] = []
        confidence_vals: list[float] = []
        is_pf_list: list[float] = []

        for fold in folds:
            if fold.oos_result:
                oos_total_trades += fold.oos_result.total_trades
                oos_total_wins += fold.oos_result.winning_trades
                oos_total_losses += fold.oos_result.losing_trades
                oos_total_return += fold.oos_result.total_return_pct
                oos_pf_list.append(fold.oos_result.true_profit_factor)

            confidence_vals.append(fold.is_best_confidence)
            is_pf_list.append(fold.is_best_pf)

        # Duration from first fold OOS start to last fold OOS end
        if folds and folds[0].oos_result:
            try:
                start_dt = datetime.fromisoformat(folds[0].oos_start)
                end_dt = datetime.fromisoformat(folds[-1].oos_end)
                total_days = (end_dt - start_dt).days
            except (ValueError, TypeError):
                total_days = 0
        else:
            total_days = 0

        n = len(folds)
        avg_win_rate = oos_total_wins / oos_total_trades if oos_total_trades > 0 else 0.0
        avg_pf = float(np.mean(oos_pf_list)) if oos_pf_list else 0.0
        confidence_std = float(np.std(confidence_vals)) if n > 1 else 0.0
        confidence_unique = len(set(confidence_vals))
        avg_is_pf = float(np.mean(is_pf_list)) if is_pf_list else 0.0
        wf_efficiency = avg_pf / avg_is_pf if avg_is_pf > 0 else 0.0

        return WalkForwardResult(
            folds=folds,
            total_folds=n,
            total_duration_days=total_days,
            oos_total_trades=oos_total_trades,
            oos_total_wins=oos_total_wins,
            oos_total_losses=oos_total_losses,
            oos_aggregate_return_pct=oos_total_return,
            oos_avg_win_rate=avg_win_rate,
            oos_avg_pf=avg_pf,
            confidence_values=confidence_vals,
            confidence_std=confidence_std,
            confidence_unique=confidence_unique,
            avg_is_pf=avg_is_pf,
            wf_efficiency=wf_efficiency,
        )
