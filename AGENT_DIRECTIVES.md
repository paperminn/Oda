# AGENT_DIRECTIVES.md — Oda Phase 2 Tasks
# Created: 2026-05-04 by Hinano

## Phase 2: Signal Pipeline Fix & Audit
- [x] TASK-001: Reproduce signal flatline (confirmed: 0 trades post-May 2021)
- [x] TASK-002: Fix sliding window index bug (cross-DF comparison)
- [x] TASK-003: Re-run backtest with fix (843 trades, PF 0.885)
- [x] TASK-004: Verify post-May signals firing (6,031 signals confirmed)

## Phase 2.5: Signal Quality (next)
- [x] TASK-005: Regime detection as signal filter (PF 0.934, no edge improvement)
- [x] TASK-006: Confidence threshold tuning (winner: 0.75, PF 0.932)
- [x] TASK-007: Volatility-scaled position sizing (already implemented, verified with backtest)
- [ ] TASK-008: Walk-forward validation framework
- [x] TASK-009: Programmatic risk controls (sprint conviction 78 — already implemented)

## Phase 3: RL Integration (later)
- [ ] TASK-010: PPO baseline (Stable-Baselines3)
- [ ] TASK-011: VectorBT backtesting comparison
- [ ] TASK-012: Binance Futures connector for live paper trading

## Sprint Findings to Apply
| Finding | Conviction | Task |
|---------|-----------|------|
| Programmatic risk controls FIRST | 78 | TASK-009 ✅ done |
| Regime detection as filter | 68-69 | TASK-005 ✅ done |
| Backtest-to-live decay (68%→51%) | 77.5 | Design note |
| Walk-forward as differentiator | — | TASK-008 |
| PPO-only before DQN | 69 | TASK-010 |
