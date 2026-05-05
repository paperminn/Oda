"""
Oda CLI — Entry point for the crypto trading bot.

Usage:
    python -m oda backtest --symbol BTCUSDT
    python -m oda fetch --symbol BTCUSDT
    python -m oda zones --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import logging
import sys

from oda.config import Settings
from oda.data import fetch_all
from oda.zones import ZoneDetector
from oda.backtest import BacktestEngine
from oda.walkforward import WalkForwardValidator


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run a walk-forward backtest or walk-forward validation."""
    # Override settings from CLI
    import os
    if args.risk_pct:
        os.environ["ODA_RISK_PCT"] = str(args.risk_pct)
    if args.leverage:
        os.environ["ODA_MAX_LEVERAGE"] = str(args.leverage)
    if args.capital:
        os.environ["ODA_CAPITAL"] = str(args.capital)

    settings = Settings.load()

    # Fetch data
    print("Fetching data from Binance...")
    data = fetch_all(settings, historical=args.historical, start_date="2021-01-01" if args.historical else None)
    print(f"  Timeframes: {list(data.keys())}")
    for tf, df in data.items():
        print(f"  {tf:4s}: {len(df):>6,} candles")

    if args.walk_forward:
        cmd_walk_forward(args, settings, data)
    else:
        cmd_single_backtest(args, settings, data)


def cmd_single_backtest(args: argparse.Namespace, settings: Settings, data: dict) -> None:
    """Run a single walk-forward backtest."""
    print("\nRunning backtest...")
    engine = BacktestEngine(settings)
    result = engine.run(
        data,
        start_date=args.start,
        end_date=args.end,
        burn_in_days=args.burn_in,
    )

    print()
    print(result.summary())

    if args.trades:
        print("\nTrade History:")
        for t in result.trades[-20:]:
            print(
                f"  [{t.exit_time}] {t.direction} "
                f"Entry: ${t.entry_price:,.2f} → Exit: ${t.exit_price:,.2f} "
                f"PnL: ${t.pnl:,.2f} ({t.pnl_r:+.2f}R) [{t.exit_reason}]"
            )


def cmd_walk_forward(args: argparse.Namespace, settings: Settings, data: dict) -> None:
    """Run walk-forward validation."""
    print(f"\n{'='*60}")
    print("WALK-FORWARD VALIDATION")
    print(f"{'='*60}")
    print(f"Folds: {args.folds}  |  Confidence sweep: {args.conf_sweep}")

    validator = WalkForwardValidator(
        settings=settings,
        num_folds=args.folds,
        confidence_values=[float(c) for c in args.conf_sweep.split(",")] if args.conf_sweep else [0.65, 0.75],
        regime_enabled=not args.no_regime,
    )

    result = validator.run(data)

    print()
    print(result.summary())

    # Show per-fold details
    if args.verbose:
        print("\nPer-Fold Detail:\n")
        for fold in result.folds:
            print(f"── Fold {fold.fold_id} ──")
            print(f"  IS confidence sweep:")
            for conf, bt_result in sorted(fold.is_confidence_sweep.items()):
                print(f"    conf={conf:.2f}: {bt_result.total_trades}t "
                      f"PF={bt_result.true_profit_factor:.3f} "
                      f"WR={bt_result.win_rate:.1%}")
            print(f"  IS winner: conf={fold.is_best_confidence:.2f} "
                  f"(PF={fold.is_best_pf:.3f})")
            if fold.oos_result:
                oos = fold.oos_result
                print(f"  OOS ({fold.oos_start} → {fold.oos_end}): "
                      f"{oos.total_trades}t WR={oos.win_rate:.1%} "
                      f"PF={oos.true_profit_factor:.3f} "
                      f"Ret={oos.total_return_pct:+.2f}%")
            print()


def cmd_fetch(args: argparse.Namespace) -> None:
    """Fetch and cache OHLCV data."""
    settings = Settings.load()
    print(f"Fetching {args.symbol} data from Binance...")
    data = fetch_all(settings)
    for tf, df in data.items():
        print(f"  {tf:4s}: {len(df):>6,} candles → {args.symbol}_{tf}.csv")


def cmd_zones(args: argparse.Namespace) -> None:
    """Detect RTM zones and display them."""
    settings = Settings.load()
    print(f"Fetching data for {args.symbol}...")
    data = fetch_all(settings)

    detector = ZoneDetector()
    zones = detector.detect_all(data)
    print(f"\nZones detected: {len(zones)}")
    print()

    # Show top 20 zones
    for z in sorted(zones, key=lambda z: z.score, reverse=True)[:20]:
        print(
            f"  {z.zone_type.value:6s} {z.pattern.value:3s} "
            f"[{z.timeframe:3s}] ${z.price_low:,.0f}–${z.price_high:,.0f} "
            f"score={z.score:.1f} fresh={z.freshness.value} "
            f"conf={z.confluence_count}TF"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oda — Crypto Futures Trading Bot"
    )
    sub = parser.add_subparsers(dest="command")

    # backtest
    bt = sub.add_parser("backtest", help="Run walk-forward backtest or walk-forward validation")
    bt.add_argument("--symbol", default="BTCUSDT")
    bt.add_argument("--risk-pct", type=float, default=2.0)
    bt.add_argument("--leverage", type=int, default=3)
    bt.add_argument("--capital", type=float, default=1000.0)
    bt.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    bt.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    bt.add_argument("--burn-in", type=int, default=90, help="Burn-in days")
    bt.add_argument("--trades", action="store_true", help="Show trade history")
    bt.add_argument("--historical", action="store_true", help="Fetch full history (2021-2026), not just 1000 candles")
    bt.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation instead of single backtest")
    bt.add_argument("--folds", type=int, default=3, help="Number of walk-forward folds (default: 3)")
    bt.add_argument("--conf-sweep", default="0.65,0.75", help="Comma-separated confidence values to sweep on IS (default: 0.65,0.75)")
    bt.add_argument("--no-regime", action="store_true", help="Disable regime filter in walk-forward")
    bt.add_argument("--verbose", action="store_true", help="Show per-fold details")

    # fetch
    ft = sub.add_parser("fetch", help="Fetch and cache OHLCV data")
    ft.add_argument("--symbol", default="BTCUSDT")

    # zones
    zn = sub.add_parser("zones", help="Detect RTM zones")
    zn.add_argument("--symbol", default="BTCUSDT")

    args = parser.parse_args()

    if args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "zones":
        cmd_zones(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
