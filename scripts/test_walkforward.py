"""Test walk-forward validation with a single fold on a limited period."""
import os, sys, time
os.environ['ODA_MAX_DD_PCT'] = '100.0'
os.environ['ODA_DAILY_LOSS_PCT'] = '100.0'

from oda.config import Settings
from oda.data import fetch_all
from oda.walkforward import WalkForwardValidator

print("Loading settings...", flush=True)
s = Settings.load()
print("Fetching data...", flush=True)
d = fetch_all(s)
print(f"Data: 15m={len(d['15m'])} bars", flush=True)

# Build folds and show them
v = WalkForwardValidator(s, num_folds=2, confidence_values=[0.65, 0.75])
folds = v.build_folds(d)
for f in folds:
    df = d['15m']
    is_mask = (df['time'] >= f.is_start) & (df['time'] <= f.is_end)
    oos_mask = (df['time'] >= f.oos_start) & (df['time'] <= f.oos_end)
    print(f"Fold {f.fold_id}: IS={is_mask.sum()} bars ({f.is_start}..{f.is_end}), "
          f"OOS={oos_mask.sum()} bars ({f.oos_start}..{f.oos_end})", flush=True)

# Run only fold 0 to verify the mechanism
# For fold 0: IS is ~685 days, OOS is ~294 days  
print("\n--- Running Fold 0 IS sweep ---", flush=True)
t0 = time.time()

# Test confidence=0.65 on IS
os.environ['ODA_MIN_CONFIDENCE'] = '0.65'
os.environ['ODA_REGIME_ENABLED'] = 'false'
from oda.config import Settings as Reloaded
s2 = Reloaded.load()
from oda.backtest import BacktestEngine
e1 = BacktestEngine(s2)
r1 = e1.run(d, start_date=folds[0].is_start, end_date=folds[0].is_end, burn_in_days=30)
print(f"IS conf=0.65: {r1.total_trades}t, PF={r1.true_profit_factor:.3f}, WR={r1.win_rate:.1%}", flush=True)

# Test confidence=0.75 on IS
os.environ['ODA_MIN_CONFIDENCE'] = '0.75'
s3 = Reloaded.load()
e2 = BacktestEngine(s3)
r2 = e2.run(d, start_date=folds[0].is_start, end_date=folds[0].is_end, burn_in_days=30)
print(f"IS conf=0.75: {r2.total_trades}t, PF={r2.true_profit_factor:.3f}, WR={r2.win_rate:.1%}", flush=True)

# Pick best
best_conf = 0.75 if r2.true_profit_factor >= r1.true_profit_factor else 0.65
best_pf = max(r1.true_profit_factor, r2.true_profit_factor)
print(f"\nIS winner: conf={best_conf:.2f} (PF={best_pf:.3f})", flush=True)

# Run OOS with best confidence
os.environ['ODA_MIN_CONFIDENCE'] = str(best_conf)
s4 = Reloaded.load()
e3 = BacktestEngine(s4)
r3 = e3.run(d, start_date=folds[0].oos_start, end_date=folds[0].oos_end, burn_in_days=30)
print(f"\nOOS result ({folds[0].oos_start}..{folds[0].oos_end}): "
      f"{r3.total_trades}t, PF={r3.true_profit_factor:.3f}, "
      f"WR={r3.win_rate:.1%}, Ret={r3.total_return_pct:+.2f}%", flush=True)

t1 = time.time()
print(f"\nTotal time: {t1-t0:.0f}s ({t1-t0:.1f} min)", flush=True)
print("\n=== Walk-Forward Single Fold Demo Complete ===", flush=True)
