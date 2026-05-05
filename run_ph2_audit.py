#!/usr/bin/env python3
"""Run Oda backtest with pipeline audit, loading cached data directly."""
import logging
import time
import sys
import os

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)

sys.path.insert(0, os.path.expanduser("~/oda/src"))
from oda.config import Settings
from oda.backtest import BacktestEngine

CACHE_DIR = os.path.expanduser("~/oda/data/cache")
SYMBOL = "BTCUSDT"

# Load cached data directly
data = {}
for tf in ["15m", "1h", "4h", "1d", "1w"]:
    path = os.path.join(CACHE_DIR, f"{SYMBOL}_{tf}.csv")
    print(f"Loading {path}...")
    df = pd.read_csv(path, parse_dates=["time"])
    # Ensure timestamp column
    if "time" in df.columns and "timestamp" not in df.columns:
        df["timestamp"] = df["time"].astype("int64") // 10**6
    data[tf] = df
    print(f"  {tf}: {len(df)} candles ({df['time'].iloc[0]} → {df['time'].iloc[-1]})")

# Load settings
settings = Settings.load()

# Run backtest
print("\nRunning backtest with pipeline audit...")
t0 = time.time()
engine = BacktestEngine(settings)
result = engine.run(
    data,
    start_date="2021-01-01",
    end_date=None,
    burn_in_days=90,
)
elapsed = time.time() - t0
print(f"\nBacktest completed in {elapsed:.1f}s\n")

# Print summary
print(result.summary())
