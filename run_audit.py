#!/usr/bin/env python3
"""Minimal Oda audit runner — loads CSVs directly, no Binance fetch."""
import sys, os, time, glob, logging

sys.path.insert(0, os.path.expanduser("~/oda/src"))
os.environ["ODA_RISK_PCT"] = "2"
os.environ["ODA_MAX_LEVERAGE"] = "3"
os.environ["ODA_CAPITAL"] = "1000"
os.environ["ODA_COOLDOWN_BARS"] = "48"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logging.getLogger("oda.zones").setLevel(logging.WARNING)
logging.getLogger("oda.risk").setLevel(logging.WARNING)

import pandas as pd
import numpy as np
from oda.config import Settings
from oda.backtest import BacktestEngine

print("=" * 60)
print("ODA SIGNAL PIPELINE AUDIT")
print("=" * 60)

t0 = time.time()

cache = os.path.expanduser("~/oda/data/cache")
tf_map = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"}
data = {}

for label, key in tf_map.items():
    files = glob.glob(os.path.join(cache, f"BTCUSDT_{label}.csv"))
    if files:
        df = pd.read_csv(files[0])
        df.columns = [c.lower().strip() for c in df.columns]
        if "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["time"]).astype("int64") // 10**6
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)
        data[key] = df
        print(f"  Loaded {label:4s}: {len(df):>6,} candles")
    else:
        print(f"  WARNING: No cache for {label}")

settings = Settings.load()
engine = BacktestEngine(settings, zone_rebuild_interval=672)
print(f"\nCapital: ${settings.risk.initial_capital} | Leverage: {settings.trading.max_leverage}x")
print(f"Running backtest 2021-01-01 to 2026-05-04...\n")

result = engine.run(data, start_date="2021-01-01", burn_in_days=90)
elapsed = time.time() - t0

print()
print(result.summary())
print(f"\nElapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
