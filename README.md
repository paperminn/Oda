# Oda — Crypto Futures Trading Bot ⏸️ FROZEN

> **FROZEN 2026-05-06** — This project is permanently terminated.
> **Reason:** Structural negative edge on RTM zones + MTF engulf entry for BTC/USDT. PF 0.934 max across 15+ configurations, 3 parameter sweeps, 5 years of data.
> **Successor:** [Nagare](https://github.com/paperminn/nagare) — momentum-based v2.
> **Carryover:** `z-brain/projects/nagare/reflection-carryover.md`
> **Post-mortem:** `oda/post-mortem.md`

This repository is a **research archive only.** The code is preserved for reference — the walk-forward validator, risk controls, data pipeline, and test framework are reusable. The strategy logic (RTM zone detection + MTF engulf) is proven unprofitable on BTC/USDT.

Architecture, results, and configuration documentation remain below as historical record.

---

## Architecture (Historical Reference)

```
oda/
├── src/oda/
│   ├── config.py      # Frozen dataclass settings, .env loader
│   ├── data.py        # Binance OHLCV fetcher, multi-TF caching
│   ├── zones.py       # RTM zone detection (FTR, FL, QM patterns) — 969 LOC
│   ├── signals.py     # MTF engulf entry confirmation
│   ├── risk.py        # Half-Kelly position sizing — 572 LOC (reusable)
│   ├── backtest.py    # Walk-forward backtest engine — 790 LOC
│   ├── walkforward.py # Walk-forward validator — 542 LOC (reusable)
│   ├── regime.py      # Market regime classifier
│   └── cli.py         # CLI entry point
├── tests/             # 39 tests (2 failing — see post-mortem)
├── data/cache/        # Cached OHLCV data (gitignored)
└── config/.env        # Environment variables (gitignored)
```

## Results (Historical Record)

### Full Backtest (2021-01-01 → 2026-05-04)

| Metric | Best Value |
|--------|-----------|
| True PF | **0.934** (confidence 0.75 + regime filter) |
| Trades | 1,474 |
| Win Rate | 26.1% |
| Max DD | ~100% (capital exhausted) |
| Survival | ~4 years before halt |

### What Failed

1. **Zone freshness** — RTM zones are structurally stale after regime shifts. 100 zones/rebuild, zero within 1.5% of price after May 2021.
2. **Win rate cap** — 26.1% WR requires min 2.85R avg win to break even. Actual avg win was ~1.8R. Structural PF cap ~0.963.
3. **No combination of parameters fixes this** — 3 sweeps (confidence thresholds, regime filter, volatility scaling) confirmed the edge is not tunable.

For full analysis: `oda/post-mortem.md`

## Reproducibility

All experiments are fully deterministic with seeded random states:

```bash
cd ~/oda
python -m oda backtest --conf-sweep 0.50,0.55,0.60,0.65,0.70,0.75
```

Data cached at `oda/data/cache/BTCUSDT_*.csv` (24-hour TTL).

## Key Lessons (Carried to Nagare)

1. **Cross-DataFrame timestamp comparison must use timestamps, not positional indices** — the flatline bug was a one-character fix that restored 6,031 signals.
2. **True PF only** — `sum(positive_pnl) / abs(sum(negative_pnl))`. Proxy PF permanently banned.
3. **15m + limit orders is minimum viable timeframe** — 5m + taker fees is mathematically unviable.
4. **Isotonic calibration under n<50 produces erratic probabilities** — use Platt (sigmoid) scaling for small training folds.
5. **Multi-instrument expansion ≠ diversification** — tested with XAUUSDT, SOL, LINK, PAXG. All degraded portfolio PF.

Private — Z's project.
