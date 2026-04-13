from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Sequence


class SignalSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass(slots=True)
class MarketBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class MarketDataBatch:
    symbol: str
    timeframe: str
    bars: Sequence[MarketBar]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FeatureSnapshot:
    timestamp: datetime
    values: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyContext:
    symbol: str
    execution_timeframe: str
    filter_timeframe: str
    bars: Sequence[MarketBar]
    features: Sequence[FeatureSnapshot]
    regime: str = "unknown"


@dataclass(slots=True)
class StrategySignal:
    symbol: str
    timestamp: datetime
    side: SignalSide
    strength: float
    rationale: str
    entry_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    time_stop_bars: int | None = None
    close_on_session_end: bool = True
    entry_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PortfolioState:
    equity: float = 100000.0
    cash: float = 100000.0
    daily_pnl_pct: float = 0.0
    open_positions: int = 0
    gross_exposure: float = 0.0
    peak_equity: float = 100000.0
    total_drawdown_pct: float = 0.0
    trades_today: int = 0
    consecutive_losses_today: int = 0
    daily_kill_switch_active: bool = False
    session_allowed: bool = True
    timestamp: datetime | None = None


@dataclass(slots=True)
class RiskDecision:
    approved: bool
    size_fraction: float
    max_leverage: float
    rationale: str
    reason_code: str | None = None
    blocked_by: Sequence[str] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestRequest:
    bars: Sequence[MarketBar]
    features: Sequence[FeatureSnapshot]
    signal: StrategySignal | None = None
    signals: Sequence[StrategySignal] = field(default_factory=tuple)
    initial_capital: float = 100000.0
    risk_per_trade_fraction: float = 0.0025
    max_leverage: float = 2.0
    signal_cooldown_bars: int = 0
    exit_zscore_threshold: float | None = None
    session_close_hour_utc: int = 23
    session_close_minute_utc: int = 55
    session_close_timezone: str = "UTC"
    session_close_windows: Sequence[str] = field(default_factory=tuple)
    intrabar_exit_policy: str | None = None
    gap_exit_policy: str | None = None


@dataclass(slots=True)
class ExecutedTrade:
    symbol: str
    side: SignalSide
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    net_pnl: float
    fees_paid: float
    return_pct: float
    bars_held: int
    exit_reason: str
    entry_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestResult:
    start: datetime | None
    end: datetime | None
    trades: int
    total_return: float
    max_drawdown: float
    win_rate: float = 0.0
    payoff: float = 0.0
    expectancy: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    pnl_net: float = 0.0
    equity_final: float = 0.0
    trade_records: Sequence[ExecutedTrade] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EnvObservation:
    timestamp: datetime | None
    features: dict[str, float]
    position: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EnvTransition:
    observation: EnvObservation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingArtifact:
    algorithm: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationReport:
    passed: bool
    checks: dict[str, bool]
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PaperOrder:
    symbol: str
    side: SignalSide
    size_fraction: float
    order_type: str = "market"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PaperExecution:
    accepted: bool
    order: PaperOrder | None
    venue: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
