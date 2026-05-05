# AGENT_STATE.md — Oda Current State
# Last updated: 2026-05-05 by Hinano

## Project Status
**Phase:** 2.5 — Signal Quality Tuning
**Deploy Target:** Fall 2026

## Last Backtest (2026-05-05, regime detection)
- **Winning param: ODA_MIN_CONFIDENCE=0.75, ODA_REGIME_ENABLED=true**
- Period: 2020-12-22 → 2026-05-04 (1959 days)
- 1,474 trades, 26.1% WR, PF 0.934, -101.92% return
- Halted Feb 2025 (MDD 101.56%)
- Regime filter does not improve PF but extends survival +1 year

## Confidence Threshold Sweep Results
| Conf | Trades | WR | PF | Return | Halt |
|------|--------|----|----|--------|------|
| 0.50 (baseline) | 843 | 24.7% | 0.885 | -100.4% | early |
| 0.55 | 1,073 | 25.5% | 0.913 | -96.85% | 2022-10 |
| 0.60 | 1,129 | 25.7% | 0.913 | -101.85% | 2022-12 |
| 0.65 | 1,061 | 25.4% | 0.907 | -101.07% | 2022-10 |
| 0.70 | 1,119 | 25.8% | 0.916 | -97.01% | 2023-01 |
| **0.75** | **1,454** | **26.1%** | **0.932** | **-101.54%** | **2024-02** |

## Regime Filter Results
| Config | Trades | WR | PF | Return | Halt |
|--------|--------|----|----|--------|------|
| Regime OFF (baseline) | 1,454 | 26.1% | 0.932 | -101.54% | 2024-02 |
| **Regime ON** | **1,474** | **26.1%** | **0.934** | **-101.92%** | **2025-02** |

## Active Issues
1. Win rate (26.1%) still below breakeven (33%)
2. PF at 0.934 — unprofitable in all configurations
3. MDD circuit breaker triggers at 10% — but even with MDD disabled, strategy loses -100%
4. Regime filter helps survival time but not edge
5. Strategy consistently loses in both trending and ranging regimes

## Recent Changes
- `AGENT_STATE.md`: Updated with TASK-005 regime detection results
- `AGENT_LOG_Hinano.md`: TASK-005, TASK-009, and TASK-007 entries added
- `AGENT_DIRECTIVES.md`: TASK-005, TASK-009, and TASK-007 marked complete
- `oda/src/oda/regime.py`: Created — new regime detection module
- `oda/src/oda/config.py`: Added RegimeConfig with env-var overrides
- `oda/src/oda/backtest.py`: Wired regime detector into signal pipeline
- TASK-007 verified: volatility-scaled position sizing already implemented end-to-end

## Infrastructure
- Data: Cached CSV, 2020-12 → 2026-05, 188K 15m bars
- Backtest: 672-bar rebuild, 90-day sliding window
- Memory: Hindsight (MiniMax M2.7, hermes-oda bank TBD)
- Default min_confidence: 0.75
- Regime detection: ON (can toggle via ODA_REGIME_ENABLED)

## Next
- [ ] TASK-008: Walk-forward validation framework
- [ ] TASK-010: PPO baseline (Stable-Baselines3)
- [ ] TASK-011: VectorBT backtesting comparison
- [ ] TASK-012: Binance Futures connector for live paper trading
