"""Run walk-forward fold 0 only, with progress output."""
import os, sys
os.environ['ODA_MAX_DD_PCT'] = '100.0'
os.environ['ODA_DAILY_LOSS_PCT'] = '100.0'
os.environ['ODA_REGIME_ENABLED'] = 'false'

from oda.config import Settings
from oda.data import fetch_all
from oda.walkforward import WalkForwardValidator

print("Loading data...", flush=True)
s = Settings.load()
d = fetch_all(s)

# Build folds manually for fold 0
v = WalkForwardValidator(s, num_folds=2, confidence_values=[0.65, 0.75])
folds = v.build_folds(d)

for f in folds:
    print(f"Fold {f.fold_id}: IS={f.is_start}..{f.is_end}  OOS={f.oos_start}..{f.oos_end}", flush=True)

# Run only fold 0 
f0 = folds[0]
print(f"\n{'='*60}", flush=True)
print(f"RUNNING FOLD 0 ONLY", flush=True)
print(f"{'='*60}", flush=True)

import time
t0 = time.time()

# IS sweep: test conf=0.65
os.environ['ODA_MIN_CONFIDENCE'] = '0.65'
from oda.config import Settings as RS
s2 = RS.load()
from oda.backtest import BacktestEngine
print(f"\n[IS sweep] confidence=0.65  [{f0.is_start} → {f0.is_end}]", flush=True)
e1 = BacktestEngine(s2)
r1 = e1.run(d, start_date=f0.is_start, end_date=f0.is_end, burn_in_days=30)
print(f"  → {r1.total_trades}t, PF={r1.true_profit_factor:.3f}, WR={r1.win_rate:.1%}", flush=True)

# IS sweep: test conf=0.75
os.environ['ODA_MIN_CONFIDENCE'] = '0.75'
s3 = RS.load()
print(f"\n[IS sweep] confidence=0.75  [{f0.is_start} → {f0.is_end}]", flush=True)
e2 = BacktestEngine(s3)
r2 = e2.run(d, start_date=f0.is_start, end_date=f0.is_end, burn_in_days=30)
print(f"  → {r2.total_trades}t, PF={r2.true_profit_factor:.3f}, WR={r2.win_rate:.1%}", flush=True)

# Select best
best_conf = 0.75 if r2.true_profit_factor >= r1.true_profit_factor else 0.65
best_pf = max(r1.true_profit_factor, r2.true_profit_factor)
print(f"\n  Best: conf={best_conf:.2f} (PF={best_pf:.3f})", flush=True)

# OOS test
os.environ['ODA_MIN_CONFIDENCE'] = str(best_conf)
s4 = RS.load()
print(f"\n[OOS test] confidence={best_conf:.2f}  [{f0.oos_start} → {f0.oos_end}]", flush=True)
e3 = BacktestEngine(s4)
r3 = e3.run(d, start_date=f0.oos_start, end_date=f0.oos_end, burn_in_days=30)
print(f"  → {r3.total_trades}t, PF={r3.true_profit_factor:.3f}, WR={r3.win_rate:.1%}, Ret={r3.total_return_pct:+.2f}%", flush=True)

t1 = time.time()
print(f"\nDone in {t1-t0:.0f}s ({t1-t0:.1f} min)", flush=True)

# Final summary
print(f"\n{'='*60}", flush=True)
print("WALK-FORWARD FOLD 0 RESULTS", flush=True)
print(f"{'='*60}", flush=True)
print(f"IS period:     {f0.is_start} → {f0.is_end}", flush=True)
print(f"OOS period:    {f0.oos_start} → {f0.oos_end}", flush=True)
print(f"IS conf=0.65:  {r1.total_trades}t  PF={r1.true_profit_factor:.3f}  WR={r1.win_rate:.1%}", flush=True)
print(f"IS conf=0.75:  {r2.total_trades}t  PF={r2.true_profit_factor:.3f}  WR={r2.win_rate:.1%}", flush=True)
print(f"Best conf:     {best_conf:.2f} (PF={best_pf:.3f})", flush=True)
print(f"OOS result:    {r3.total_trades}t  PF={r3.true_profit_factor:.3f}  WR={r3.win_rate:.1%}  Ret={r3.total_return_pct:+.2f}%", flush=True)
print(f"\nWalk-Forward Efficiency: OOS PF / IS best PF = {r3.true_profit_factor/best_pf:.3f}", flush=True)
