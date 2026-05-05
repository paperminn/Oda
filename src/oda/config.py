"""
Oda Configuration — Frozen dataclass settings loaded from environment.
All settings are immutable at runtime. Load once, pass everywhere.

Adapted from CryptoMecha01 config/settings.py (Z's prior work).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Runtime import needed for Settings.load()
from oda.regime import RegimeConfig


def _get_env(key: str, default: str = "", required: bool = False) -> str:
    val = os.environ.get(key, default)
    if required and not val:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Check config/.env and make sure it is loaded."
        )
    return val


# ── Data Config ────────────────────────────────────────────────────

@dataclass(frozen=True)
class DataConfig:
    """Binance API and data settings."""
    symbol: str = "BTCUSDT"
    intervals: tuple[str, ...] = ("15m", "1h", "4h", "1d", "1w")
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    binance_base_url: str = "https://api.binance.com"
    rate_limit_rps: float = 1.0  # requests per second

    @classmethod
    def from_env(cls) -> "DataConfig":
        return cls(
            symbol=_get_env("ODA_SYMBOL", "BTCUSDT"),
            cache_dir=Path(_get_env("ODA_CACHE_DIR", "data/cache")),
        )


# ── Trading Config ─────────────────────────────────────────────────

@dataclass(frozen=True)
class TradingConfig:
    """Trading parameters."""
    base_risk_pct: float = 2.0       # % of equity risked per trade
    max_leverage: int = 3            # Max allowed leverage
    min_zone_score: float = 2.0      # Minimum RTM zone score
    zone_lookback_candles: int = 100  # Candles for zone detection
    fvg_min_width_pct: float = 0.05  # Min FVG width for engulf
    cooldown_bars: int = 32          # 32 x 15m = 8h cooldown
    max_trade_duration_hours: int = 48
    trailing_activation_r: float = 1.5
    trailing_callback_pct: float = 0.8

    @classmethod
    def from_env(cls) -> "TradingConfig":
        return cls(
            base_risk_pct=float(_get_env("ODA_RISK_PCT", "2.0")),
            max_leverage=int(_get_env("ODA_MAX_LEVERAGE", "3")),
            min_zone_score=float(_get_env("ODA_MIN_ZONE_SCORE", "2.0")),
        )


# ── Risk Config ────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters. Half-Kelly framework from CryptoMecha01."""
    initial_capital: float = 1000.0
    kelly_fraction: float = 0.5       # Half-Kelly
    max_risk_per_trade_pct: float = 0.02  # 2% cap
    global_exposure_cap_pct: float = 0.05  # 5% total
    max_drawdown_pct: float = 10.0    # Hard stop
    min_ev_threshold: float = 0.10    # Min expected value in R
    daily_loss_pct: float = 5.0       # Halt trading for the day if losses exceed this % of equity

    # Volatility-scaled position sizing
    vol_scale_enabled: bool = False   # Enable inverse-ATR position scaling
    vol_scale_min: float = 0.25       # Minimum scale factor (cap reduction at 75%)
    vol_scale_max: float = 1.50       # Maximum scale factor (cap increase at 50%)

    @classmethod
    def from_env(cls) -> "RiskConfig":
        return cls(
            initial_capital=float(_get_env("ODA_CAPITAL", "1000.0")),
            kelly_fraction=float(_get_env("ODA_KELLY_FRACTION", "0.5")),
            max_drawdown_pct=float(_get_env("ODA_MAX_DD_PCT", "10.0")),
            min_ev_threshold=float(_get_env("ODA_MIN_EV_R", "0.10")),
            daily_loss_pct=float(_get_env("ODA_DAILY_LOSS_PCT", "5.0")),
            vol_scale_enabled=_get_env("ODA_VOL_SCALE_ENABLED", "false").lower() in ("true", "1", "yes"),
            vol_scale_min=float(_get_env("ODA_VOL_SCALE_MIN", "0.25")),
            vol_scale_max=float(_get_env("ODA_VOL_SCALE_MAX", "1.50")),
        )


# ── Backtest Config ────────────────────────────────────────────────

@dataclass(frozen=True)
class BacktestConfig:
    """Backtest parameters."""
    start_year: int = 2021
    end_year: int = 2026
    burn_in_days: int = 90
    train_min_samples: int = 50
    retrain_interval_samples: int = 25
    walk_forward_folds: int = 5
    maker_fee: float = 0.0002       # 0.02%
    taker_fee: float = 0.0004       # 0.04%
    limit_fill_rate: float = 0.70   # 70% fill probability

    @classmethod
    def from_env(cls) -> "BacktestConfig":
        return cls(
            start_year=int(_get_env("ODA_BT_START_YEAR", "2021")),
            end_year=int(_get_env("ODA_BT_END_YEAR", "2026")),
        )


# ── Master Settings ────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    """Master configuration — all sub-configs frozen at load time."""
    data: DataConfig
    trading: TradingConfig
    risk: RiskConfig
    backtest: BacktestConfig
    regime: "RegimeConfig"
    project_root: Path

    @property
    def symbol(self) -> str:
        return self.data.symbol

    @classmethod
    def load(cls, env_file: Optional[str] = None) -> "Settings":
        _load_dotenv(env_file)
        project_root = Path(__file__).parent.parent.parent.resolve()

        return cls(
            data=DataConfig.from_env(),
            trading=TradingConfig.from_env(),
            risk=RiskConfig.from_env(),
            backtest=BacktestConfig.from_env(),
            regime=RegimeConfig.from_env(),
            project_root=project_root,
        )


def _load_dotenv(env_file: Optional[str] = None) -> None:
    """Minimal .env loader — no external dependency."""
    if env_file:
        path = Path(env_file)
    else:
        path = Path(__file__).parent.parent.parent / "config" / ".env"

    if not path.exists():
        return

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key not in os.environ:
                os.environ[key] = value
