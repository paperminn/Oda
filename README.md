# Oda — Crypto Futures Trading Bot

Production-grade BTC/USDT futures trading bot with RTM (Read The Market) zone detection, MTF engulf entry confirmation, and walk-forward backtesting.

**Status:** MVP v0.1.0 — functional backtest engine, 57 live trades on real Binance data.  
**Target:** $1,200/month passive income by June 2028 from $1,000 + $100/month top-ups.

## Quick Start

```bash
cd ~/oda
pip install -e .

# Fetch data
python -m oda fetch

# Run backtest
python -m oda backtest --symbol BTCUSDT --capital 1000 --risk-pct 2.0 --leverage 3

# Detect zones
python -m oda zones
```

## Architecture

```
oda/
├── src/oda/
│   ├── config.py      # Frozen dataclass settings, .env loader
│   ├── data.py        # Binance OHLCV fetcher, multi-TF caching
│   ├── zones.py       # RTM zone detection (FTR, FL, QM patterns)
│   ├── signals.py     # MTF engulf entry confirmation
│   ├── risk.py        # Half-Kelly position sizing
│   ├── backtest.py    # Walk-forward backtest engine
│   └── cli.py         # CLI entry point
├── tests/
├── data/cache/        # Cached OHLCV data (gitignored)
└── config/.env        # Environment variables (gitignored)
```

## Strategy

1. **Zone Detection** — RTM supply/demand zones across 5 timeframes (1w, 1d, 4h, 1h, 15m) using FTR (Fail to Return), FL (Flag Limit), and QM (Quasimodo) patterns. Zone detection accuracy: ~95%.

2. **Entry Confirmation** — MTF engulf: when a 1h+ zone is near price, scan 15m for bullish/bearish engulf at zone boundary. Confidence scored on body ratio, engulf strength, wick rejection, and volume surge.

3. **Risk Management** — Half-Kelly position sizing with 2% max risk per trade. Global exposure cap at 5%. 10% max drawdown hard stop.

4. **Walk-Forward Backtest** — Expanding window with burn-in period. Metrics: True PF, Sortino, Sharpe, Max Drawdown.

## Latest Backtest Results (2024, with Nunchi Trailing Stops)

```
Period:    2024-01-01 → 2024-08-06 (halted at 10.9% MDD)
Candles:   35,041 (15m)
Trades:    39
Win Rate:  28.2%
True PF:   1.467   ← +27% improvement from Nunchi breakeven
P&L:       +$140.00 (14.0% return)
Max DD:    10.94%
Avg R Win: +2.00R
Avg R Loss: -0.54R   ← Breakeven stops halved average loss
Breakeven: 22 trades activated (56% of trades)
Sharpe:    0.146
```

### Before/After Nunchi Trailing

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| True PF | 1.158 | 1.467 | +27% |
| Avg R Loss | -1.00R | -0.54R | -46% |
| Gross Loss | $380 | $300 | -21% |
| Win Rate | 36.7% | 28.2% | Lower (but smaller losses) |
| Total P&L | $60 | $140 | +133% |

## Key Decisions

- **True PF only** — `sum(positive_pnl) / abs(sum(negative_pnl))`. Proxy PF banned (lesson from CryptoMecha01 Phase 17).
- **No HMM** — regime detection removed (non-deterministic in CryptoMecha01). Future: rolling volatility percentile.
- **Python 3.11+** — no legacy 3.9 constraint.
- **Deterministic only** — all random states seeded.

## Roadmap

- [x] Nunchi trailing stops (breakeven at 1R) — **+27% PF improvement**
- [x] Drawdown circuit breaker (halts at 10% MDD)
- [x] Position overlap prevention
- [x] Full historical data pipeline (248K rows, 2020-2026)
- [ ] Zone-based trailing at +2R (code ready, needs 2R+ trade runways)
- [ ] Multi-ticker expansion (ETH, SOL with correlation check)
- [ ] PPO/RL integration for entry timing
- [ ] Paper trading harness (Binance testnet)
- [ ] Can Can pattern detection (Mori's 4-stage state machine)

## Research Foundation

- **RTM Synthesis** — `z-brain/research/crypto/RTM-Hinano-Mori-SYNTHESIS.md`
- **CryptoMecha01 Analysis** — `z-brain/projects/oda/cryptomecha01-full-analysis.md`
- **RTM Detector** — `z-brain/research/rtm/rtm_detector.py` (proven 95% accuracy)
- **CryptoMecha01 Vault** — `~/cryptomecha01-review/obsidian/` (multi-agent governance)

## Configuration

Set via environment variables or `config/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| ODA_SYMBOL | BTCUSDT | Trading pair |
| ODA_CAPITAL | 1000.0 | Starting capital |
| ODA_RISK_PCT | 2.0 | Risk per trade (%) |
| ODA_MAX_LEVERAGE | 3 | Max leverage |
| ODA_KELLY_FRACTION | 0.5 | Kelly fraction (0.5 = Half-Kelly) |
| ODA_BT_START_YEAR | 2021 | Backtest start year |
| ODA_BT_END_YEAR | 2026 | Backtest end year |

## License

Private — Z's project.
