from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when the project configuration is invalid."""


@dataclass(slots=True)
class AppConfig:
    name: str = "hybrid-quant-framework"
    stage: str = "research"
    timezone: str = "UTC"
    log_level: str = "INFO"


@dataclass(slots=True)
class MarketConfig:
    symbol: str = "BTCUSDT"
    venue: str = "binance"
    execution_timeframe: str = "5m"
    filter_timeframe: str = "1H"
    style: str = "intraday"


@dataclass(slots=True)
class StorageConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"


@dataclass(slots=True)
class DataConfig:
    provider: str = "binance"
    execution_timeframe: str = "5m"
    filter_timeframe: str = "1H"
    warmup_bars: int = 2000
    timezone: str = "UTC"
    historical_api_url: str = "https://api.binance.com"
    request_timeout_seconds: int = 30
    request_limit: int = 1000
    default_start: str = "2024-01-01T00:00:00+00:00"
    default_end: str | None = None
    parquet_engine: str | None = None
    parquet_compression: str = "snappy"
    allow_gaps: bool = False
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    include_funding: bool = False
    include_open_interest: bool = False
    provider_params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FeatureConfig:
    enabled: list[str] = field(default_factory=list)
    lookback_window: int = 96
    regime_window: int = 288
    normalize: bool = True


@dataclass(slots=True)
class StrategyConfig:
    name: str = "mean_reversion_trend_regime"
    mode: str = "rule_based"
    signal_cooldown_bars: int = 3
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5
    trend_filter: str = "ema_200_1h"
    regime_filter: str = "adx_1h"
    mean_reversion_anchor: str = "vwap"
    adx_threshold: float = 25.0
    atr_multiple_stop: float = 1.0
    atr_multiple_target: float = 1.0
    time_stop_bars: int = 12
    session_close_hour_utc: int = 23
    session_close_minute_utc: int = 55
    no_entry_minutes_before_close: int = 30


@dataclass(slots=True)
class RiskConfig:
    max_risk_per_trade: float = 0.0025
    max_daily_loss: float = 0.03
    max_open_positions: int = 1
    max_leverage: float = 2.0
    prop_firm_mode: bool = True


@dataclass(slots=True)
class BacktestConfig:
    initial_capital: float = 100000.0
    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    latency_ms: int = 250


@dataclass(slots=True)
class EnvConfig:
    max_steps: int = 3000
    reward_mode: str = "risk_adjusted"
    observation_window: int = 64
    action_space: str = "target_position"


@dataclass(slots=True)
class RLConfig:
    enabled: bool = False
    algorithm: str = "PPO"
    checkpoint_dir: str = "artifacts/rl"
    total_timesteps: int = 1000000


@dataclass(slots=True)
class ValidationConfig:
    walk_forward_splits: int = 4
    min_trades: int = 50
    max_drawdown_limit: float = 0.08
    sharpe_floor: float = 1.0


@dataclass(slots=True)
class PaperConfig:
    enabled: bool = False
    dry_run: bool = True
    broker: str = "simulator"
    heartbeat_seconds: int = 30


@dataclass(slots=True)
class Settings:
    app: AppConfig = field(default_factory=AppConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Settings":
        return cls(
            app=AppConfig(**payload.get("app", {})),
            market=MarketConfig(**payload.get("market", {})),
            storage=StorageConfig(**payload.get("storage", {})),
            data=DataConfig(**payload.get("data", {})),
            features=FeatureConfig(**payload.get("features", {})),
            strategy=StrategyConfig(**payload.get("strategy", {})),
            risk=RiskConfig(**payload.get("risk", {})),
            backtest=BacktestConfig(**payload.get("backtest", {})),
            env=EnvConfig(**payload.get("env", {})),
            rl=RLConfig(**payload.get("rl", {})),
            validation=ValidationConfig(**payload.get("validation", {})),
            paper=PaperConfig(**payload.get("paper", {})),
        )


def load_settings(config_dir: str | Path) -> Settings:
    config_path = Path(config_dir)
    if not config_path.exists():
        raise ConfigError(f"Config directory not found: {config_path}")

    merged: dict[str, Any] = {}
    ordered_files = [
        "base.yaml",
        "data.yaml",
        "features.yaml",
        "strategy.yaml",
        "risk.yaml",
        "backtest.yaml",
        "env.yaml",
        "rl.yaml",
        "validation.yaml",
        "paper.yaml",
    ]

    for filename in ordered_files:
        file_path = config_path / filename
        if not file_path.exists():
            continue

        with file_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}

        if not isinstance(loaded, dict):
            raise ConfigError(f"Invalid YAML structure in {file_path}")

        merged = _deep_merge(merged, loaded)

    return Settings.from_dict(merged)


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in incoming.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result
