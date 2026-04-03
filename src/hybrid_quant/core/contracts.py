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


@dataclass(slots=True)
class RiskDecision:
    approved: bool
    size_fraction: float
    max_leverage: float
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestRequest:
    bars: Sequence[MarketBar]
    features: Sequence[FeatureSnapshot]
    signal: StrategySignal | None = None
    initial_capital: float = 100000.0


@dataclass(slots=True)
class BacktestResult:
    start: datetime | None
    end: datetime | None
    trades: int
    total_return: float
    max_drawdown: float
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
