"""Quick import and smoke check for oda.data."""
import sys
sys.path.insert(0, "/home/memento/oda/src")

from oda.data import fetch_all, fetch_ohlcv, OHLCV_COLUMNS, _cache_path
print("✓ All imports successful")
print(f"  OHLCV_COLUMNS = {OHLCV_COLUMNS}")
