"""
Oda Baseline Phase 1 — Full backtest from cache, no API calls.
Sets a reasonable zone rebuild interval to complete in sane time.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from oda.config import Settings
from oda.data import fetch_all
from oda.backtest import BacktestEngine

# ── 1. Full 2021-2026 backtest ──
print("=" * 60)
print("ODA BASELINE — Full Period (cached data)")
print("=" * 60)

settings = Settings.load()
print(f"\nLoading data from cache (no API calls)...")
t0 = time.time()
data = fetch_all(settings)  # No --historical = uses fresh cache
t1 = time.time()
print(f"Data loaded in {t1-t0:.1f}s")
for tf, df in data.items():
    print(f"  {tf:4s}: {len(df):>7,} candles  [{df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()}]")

# Backtest with longer zone rebuild (672 bars = 7 days on 15m)
engine = BacktestEngine(settings, zone_rebuild_interval=672)
print(f"\nRunning backtest...")
t2 = time.time()
result = engine.run(data, burn_in_days=90)
t3 = time.time()
print(f"Backtest completed in {t3-t2:.1f}s")

print(f"\n{result.summary()}")

# ── 2. 2024-only comparative backtest ──
print("\n" + "=" * 60)
print("ODA BASELINE — 2024 Only (comparison with prior 39-trade run)")
print("=" * 60)

engine2 = BacktestEngine(settings, zone_rebuild_interval=672)
result_2024 = engine2.run(data, start_date="2024-01-01", end_date="2024-12-31", burn_in_days=90)
t4 = time.time()
print(f"\n2024 backtest completed in {t4-t3:.1f}s")
print(f"\n{result_2024.summary()}")

# ── 3. Save results ──
output_dir = os.path.expanduser("~/z-brain/projects/oda")
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "baseline-ph1-results.txt"), "w") as f:
    f.write(result.summary())
    f.write("\n\nTRADE LOG\n")
    f.write("-" * 60 + "\n")
    for t in result.trades:
        f.write(
            f"[{t.exit_time.date()}] {t.direction:5s} "
            f"Entry: ${t.entry_price:>8,.2f} → Exit: ${t.exit_price:>8,.2f} "
            f"PnL: ${t.pnl:>+7,.2f} ({t.pnl_r:+.2f}R) [{t.exit_reason}]\n"
        )

with open(os.path.join(output_dir, "baseline-ph1-results-2024-only.txt"), "w") as f:
    f.write(result_2024.summary())
    f.write("\n\nTRADE LOG\n")
    f.write("-" * 60 + "\n")
    for t in result_2024.trades:
        f.write(
            f"[{t.exit_time.date()}] {t.direction:5s} "
            f"Entry: ${t.entry_price:>8,.2f} → Exit: ${t.exit_price:>8,.2f} "
            f"PnL: ${t.pnl:>+7,.2f} ({t.pnl_r:+.2f}R) [{t.exit_reason}]\n"
        )

print(f"\nResults saved to {output_dir}/")
print("Done.")
