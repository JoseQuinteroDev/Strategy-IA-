from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    variant_name: str = "baseline_v1"
    family: str = "mean_reversion"
    mode: str = "rule_based"
    signal_cooldown_bars: int = 3
    entry_zscore: float = 2.0
    exit_zscore: float | None = 0.5
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
    blocked_hours_utc: list[int] = field(default_factory=list)
    allowed_hours_utc: list[int] = field(default_factory=list)
    allowed_weekdays: list[int] = field(default_factory=list)
    exclude_weekends: bool = False
    minimum_anchor_distance_atr: float = 0.0
    minimum_expected_move_bps: float = 0.0
    minimum_target_to_cost_ratio: float = 0.0
    estimated_round_trip_cost_bps: float = 0.0
    breakout_lookback_bars: int = 20
    breakout_buffer_atr: float = 0.0
    minimum_breakout_range_atr: float = 0.0
    momentum_lookback_bars: int = 20
    minimum_momentum_abs: float = 0.0
    minimum_candle_range_atr: float = 0.0


@dataclass(slots=True)
class RiskConfig:
    max_risk_per_trade: float = 0.0025
    max_daily_loss: float = 0.03
    max_total_drawdown: float = 0.1
    daily_kill_switch: bool = True
    max_trades_per_day: int = 6
    max_open_positions: int = 1
    max_leverage: float = 2.0
    block_outside_session: bool = True
    session_start_hour_utc: int = 0
    session_start_minute_utc: int = 0
    session_end_hour_utc: int = 23
    session_end_minute_utc: int = 55
    require_stop_loss: bool = True
    prop_firm_mode: bool = True


@dataclass(slots=True)
class BacktestConfig:
    initial_capital: float = 100000.0
    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    latency_ms: int = 250
    intrabar_exit_policy: str = "conservative"


@dataclass(slots=True)
class EnvConfig:
    max_steps: int = 3000
    reward_mode: str = "risk_adjusted"
    state_context_bars: int = 64
    observation_window: int | None = None
    action_space: str = "candidate_trade_discrete"

    @property
    def effective_state_context_bars(self) -> int:
        raw_value = self.observation_window if self.observation_window is not None else self.state_context_bars
        return max(1, int(raw_value))


@dataclass(slots=True)
class RLConfig:
    enabled: bool = False
    algorithm: str = "PPO"
    checkpoint_dir: str = "artifacts/rl"
    total_timesteps: int = 1000000
    seeds: list[int] = field(default_factory=lambda: [7, 11])
    policy: str = "MlpPolicy"
    learning_rate: float = 0.0003
    n_steps: int = 128
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    clip_range: float = 0.2
    eval_freq: int = 5000
    checkpoint_freq: int = 5000
    n_eval_episodes: int = 1
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str = "test"
    device: str = "auto"
    verbose: int = 1
    tensorboard_log_dir: str = "artifacts/rl/tensorboard"


@dataclass(slots=True)
class ValidationCostScenarioConfig:
    name: str
    fee_multiplier: float = 1.0
    slippage_multiplier: float = 1.0


@dataclass(slots=True)
class ValidationDecisionConfig:
    min_total_test_trades: int = 10
    min_positive_test_window_ratio_go: float = 0.60
    min_positive_test_window_ratio_caution: float = 0.33
    min_positive_temporal_block_ratio_go: float = 0.60
    min_positive_temporal_block_ratio_caution: float = 0.33
    min_aggregated_test_total_return_go: float = 0.0
    min_aggregated_test_total_return_caution: float = -0.01
    max_test_drawdown_go: float = 0.05
    max_test_drawdown_caution: float = 0.10
    min_test_sharpe_go: float = 0.0
    min_test_sharpe_caution: float = -0.50
    max_monte_carlo_drawdown_p95_go: float = 0.05
    max_monte_carlo_drawdown_p95_caution: float = 0.10
    min_cost_survival_ratio_go: float = 0.50
    min_cost_survival_ratio_caution: float = 0.15


@dataclass(slots=True)
class ValidationConfig:
    walk_forward_splits: int = 4
    walk_forward_train_ratio: float = 0.60
    walk_forward_validation_ratio: float = 0.20
    walk_forward_test_ratio: float = 0.20
    temporal_block_frequency: str = "monthly"
    monte_carlo_simulations: int = 500
    monte_carlo_seed: int = 11
    min_trades: int = 50
    max_drawdown_limit: float = 0.08
    sharpe_floor: float = 1.0
    cost_scenarios: list[ValidationCostScenarioConfig] = field(
        default_factory=lambda: [
            ValidationCostScenarioConfig(name="base", fee_multiplier=1.0, slippage_multiplier=1.0),
            ValidationCostScenarioConfig(name="fees_x1_5", fee_multiplier=1.5, slippage_multiplier=1.0),
            ValidationCostScenarioConfig(name="fees_x2", fee_multiplier=2.0, slippage_multiplier=1.0),
            ValidationCostScenarioConfig(name="slippage_x1", fee_multiplier=1.0, slippage_multiplier=1.0),
            ValidationCostScenarioConfig(name="slippage_x2", fee_multiplier=1.0, slippage_multiplier=2.0),
            ValidationCostScenarioConfig(name="slippage_x3", fee_multiplier=1.0, slippage_multiplier=3.0),
        ]
    )
    decision: ValidationDecisionConfig = field(default_factory=ValidationDecisionConfig)


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
            validation=_validation_config_from_dict(payload.get("validation", {})),
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


def apply_settings_overrides(settings: Settings, overrides: dict[str, Any]) -> Settings:
    payload = asdict(settings)
    merged = _deep_merge(payload, overrides)
    return Settings.from_dict(merged)


def _validation_config_from_dict(payload: dict[str, Any]) -> ValidationConfig:
    validation_payload = dict(payload or {})
    raw_cost_scenarios = validation_payload.pop("cost_scenarios", None)
    raw_decision = validation_payload.pop("decision", None)

    validation = ValidationConfig(**validation_payload)
    if raw_cost_scenarios is not None:
        validation.cost_scenarios = [
            ValidationCostScenarioConfig(**item) for item in raw_cost_scenarios
        ]
    if raw_decision is not None:
        validation.decision = ValidationDecisionConfig(**raw_decision)
    return validation


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in incoming.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result
