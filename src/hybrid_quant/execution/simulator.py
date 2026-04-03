from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
import math

import pandas as pd

from hybrid_quant.core import ExecutedTrade, FeatureSnapshot, MarketBar, SignalSide, StrategySignal


@dataclass(slots=True)
class PendingEntry:
    signal: StrategySignal
    generated_index: int
    stop_distance: float
    target_distance: float
    size_fraction: float
    max_leverage: float


@dataclass(slots=True)
class OpenPosition:
    symbol: str
    side: SignalSide
    entry_timestamp: pd.Timestamp
    entry_index: int
    entry_price: float
    stop_price: float
    target_price: float
    quantity: float
    entry_fee: float
    time_stop_bars: int | None
    close_on_session_end: bool
    entry_reason: str | None
    signal_metadata: dict[str, object]


def resolve_intrabar_policy(policy: str | None, default: str = "conservative") -> str:
    resolved = policy or default
    allowed = {"stop_first", "target_first", "conservative"}
    if resolved not in allowed:
        raise ValueError(f"Unsupported intrabar exit policy: {resolved}")
    return resolved


def apply_entry_slippage(side: SignalSide, price: float, slippage_bps: float) -> float:
    slip = slippage_bps / 10000.0
    if side == SignalSide.LONG:
        return price * (1.0 + slip)
    return price * (1.0 - slip)


def apply_exit_slippage(side: SignalSide, price: float, slippage_bps: float) -> float:
    slip = slippage_bps / 10000.0
    if side == SignalSide.LONG:
        return price * (1.0 - slip)
    return price * (1.0 + slip)


def intrabar_exit_flags(position: OpenPosition, bar: MarketBar) -> tuple[bool, bool]:
    if position.side == SignalSide.LONG:
        return bar.low <= position.stop_price, bar.high >= position.target_price
    return bar.high >= position.stop_price, bar.low <= position.target_price


def resolve_intrabar_exit(
    *,
    position: OpenPosition,
    stop_hit: bool,
    target_hit: bool,
    intrabar_policy: str,
) -> tuple[str, float] | None:
    if stop_hit and target_hit:
        if intrabar_policy == "stop_first":
            return "stop_loss", position.stop_price
        if intrabar_policy == "target_first":
            return "take_profit", position.target_price
        return _conservative_intrabar_exit(position)

    if stop_hit:
        return "stop_loss", position.stop_price
    if target_hit:
        return "take_profit", position.target_price
    return None


def compute_equity(cash: float, position: OpenPosition | None, close_price: float) -> float:
    if position is None:
        return cash
    direction = 1.0 if position.side == SignalSide.LONG else -1.0
    unrealized = direction * (close_price - position.entry_price) * position.quantity
    return cash + unrealized


def is_session_close(timestamp: object, hour: int, minute: int) -> bool:
    normalized = pd.Timestamp(timestamp)
    normalized = normalized.tz_localize("UTC") if normalized.tzinfo is None else normalized.tz_convert("UTC")
    session_close = time(hour, minute)
    return normalized.time() >= session_close


def is_within_session(
    timestamp: object,
    *,
    start_hour_utc: int,
    start_minute_utc: int,
    end_hour_utc: int,
    end_minute_utc: int,
) -> bool:
    normalized = pd.Timestamp(timestamp)
    normalized = normalized.tz_localize("UTC") if normalized.tzinfo is None else normalized.tz_convert("UTC")
    current_time = normalized.time()
    session_start = time(start_hour_utc, start_minute_utc)
    session_end = time(end_hour_utc, end_minute_utc)
    if session_start <= session_end:
        return session_start <= current_time <= session_end
    return current_time >= session_start or current_time <= session_end


def signal_has_executable_levels(signal: StrategySignal) -> bool:
    if signal.side not in {SignalSide.LONG, SignalSide.SHORT}:
        return False
    if signal.entry_price is None or signal.stop_price is None or signal.target_price is None:
        return False
    stop_distance = abs(float(signal.entry_price) - float(signal.stop_price))
    target_distance = abs(float(signal.target_price) - float(signal.entry_price))
    return stop_distance > 0.0 and target_distance > 0.0


def build_pending_entry(
    *,
    signal: StrategySignal,
    generated_index: int,
    size_fraction: float,
    max_leverage: float,
) -> PendingEntry | None:
    if not signal_has_executable_levels(signal):
        return None
    return PendingEntry(
        signal=signal,
        generated_index=generated_index,
        stop_distance=abs(float(signal.entry_price) - float(signal.stop_price)),
        target_distance=abs(float(signal.target_price) - float(signal.entry_price)),
        size_fraction=size_fraction,
        max_leverage=max_leverage,
    )


def open_position(
    *,
    pending: PendingEntry,
    entry_bar: MarketBar,
    index: int,
    cash: float,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[OpenPosition | None, float]:
    raw_entry_price = apply_entry_slippage(pending.signal.side, entry_bar.open, slippage_bps)
    fee_rate = fee_bps / 10000.0
    risk_budget = cash * pending.size_fraction
    quantity_from_risk = risk_budget / pending.stop_distance if pending.stop_distance > 0.0 else 0.0
    quantity_from_leverage = (cash * pending.max_leverage) / raw_entry_price if raw_entry_price > 0.0 else 0.0
    quantity = min(quantity_from_risk, quantity_from_leverage)
    if quantity <= 0.0 or not math.isfinite(quantity):
        return None, cash

    entry_fee = raw_entry_price * quantity * fee_rate
    cash_after_entry_fee = cash - entry_fee
    if pending.signal.side == SignalSide.LONG:
        stop_price = raw_entry_price - pending.stop_distance
        target_price = raw_entry_price + pending.target_distance
    else:
        stop_price = raw_entry_price + pending.stop_distance
        target_price = raw_entry_price - pending.target_distance

    position = OpenPosition(
        symbol=pending.signal.symbol,
        side=pending.signal.side,
        entry_timestamp=pd.Timestamp(entry_bar.timestamp),
        entry_index=index,
        entry_price=raw_entry_price,
        stop_price=stop_price,
        target_price=target_price,
        quantity=quantity,
        entry_fee=entry_fee,
        time_stop_bars=pending.signal.time_stop_bars,
        close_on_session_end=pending.signal.close_on_session_end,
        entry_reason=pending.signal.entry_reason,
        signal_metadata=dict(pending.signal.metadata),
    )
    return position, cash_after_entry_fee


def close_position(
    *,
    position: OpenPosition,
    exit_bar: MarketBar,
    exit_price: float,
    exit_reason: str,
    cash: float,
    fee_bps: float,
    slippage_bps: float,
    index: int,
) -> tuple[ExecutedTrade, float]:
    slipped_exit_price = apply_exit_slippage(position.side, exit_price, slippage_bps)
    fee_rate = fee_bps / 10000.0
    direction = 1.0 if position.side == SignalSide.LONG else -1.0
    gross_pnl = direction * (slipped_exit_price - position.entry_price) * position.quantity
    exit_fee = slipped_exit_price * position.quantity * fee_rate
    cash_after_exit = cash + gross_pnl - exit_fee
    total_fees = position.entry_fee + exit_fee
    net_pnl = gross_pnl - total_fees

    trade = ExecutedTrade(
        symbol=position.symbol,
        side=position.side,
        entry_timestamp=position.entry_timestamp.to_pydatetime(),
        exit_timestamp=exit_bar.timestamp,
        entry_price=position.entry_price,
        exit_price=slipped_exit_price,
        quantity=position.quantity,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        fees_paid=total_fees,
        return_pct=net_pnl / max(position.entry_price * position.quantity, 1e-12),
        bars_held=(index - position.entry_index) + 1,
        exit_reason=exit_reason,
        entry_reason=position.entry_reason,
        metadata=dict(position.signal_metadata),
    )
    return trade, cash_after_exit


def evaluate_open_position(
    *,
    position: OpenPosition,
    bar: MarketBar,
    feature: FeatureSnapshot,
    index: int,
    cash: float,
    fee_bps: float,
    slippage_bps: float,
    exit_zscore_threshold: float | None,
    session_close_hour_utc: int,
    session_close_minute_utc: int,
    intrabar_policy: str,
) -> tuple[ExecutedTrade | None, float, OpenPosition | None]:
    stop_hit, target_hit = intrabar_exit_flags(position, bar)
    exit_decision = resolve_intrabar_exit(
        position=position,
        stop_hit=stop_hit,
        target_hit=target_hit,
        intrabar_policy=intrabar_policy,
    )
    if exit_decision is not None:
        exit_reason, exit_price = exit_decision
        trade, updated_cash = close_position(
            position=position,
            exit_bar=bar,
            exit_price=exit_price,
            exit_reason=exit_reason,
            cash=cash,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            index=index,
        )
        return trade, updated_cash, None

    if position.close_on_session_end and is_session_close(
        bar.timestamp,
        session_close_hour_utc,
        session_close_minute_utc,
    ):
        trade, updated_cash = close_position(
            position=position,
            exit_bar=bar,
            exit_price=bar.close,
            exit_reason="session_close",
            cash=cash,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            index=index,
        )
        return trade, updated_cash, None

    bars_held = (index - position.entry_index) + 1
    if position.time_stop_bars is not None and bars_held >= position.time_stop_bars:
        trade, updated_cash = close_position(
            position=position,
            exit_bar=bar,
            exit_price=bar.close,
            exit_reason="time_stop",
            cash=cash,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            index=index,
        )
        return trade, updated_cash, None

    if exit_zscore_threshold is not None:
        zscore = feature.values.get("zscore_distance_to_mean")
        if zscore is not None and math.isfinite(float(zscore)) and abs(float(zscore)) <= exit_zscore_threshold:
            trade, updated_cash = close_position(
                position=position,
                exit_bar=bar,
                exit_price=bar.close,
                exit_reason="mean_reversion_exit",
                cash=cash,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                index=index,
            )
            return trade, updated_cash, None

    return None, cash, position


@dataclass(slots=True)
class PortfolioSimulator:
    initial_capital: float
    fee_bps: float
    slippage_bps: float
    intrabar_exit_policy: str = "conservative"
    cash: float = field(init=False)
    pending_entry: PendingEntry | None = field(init=False, default=None)
    position: OpenPosition | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.intrabar_exit_policy = resolve_intrabar_policy(self.intrabar_exit_policy)
        self.cash = self.initial_capital

    def queue_signal(
        self,
        *,
        signal: StrategySignal,
        index: int,
        size_fraction: float,
        max_leverage: float,
    ) -> bool:
        if self.pending_entry is not None or self.position is not None:
            return False
        pending_entry = build_pending_entry(
            signal=signal,
            generated_index=index,
            size_fraction=size_fraction,
            max_leverage=max_leverage,
        )
        if pending_entry is None:
            return False
        self.pending_entry = pending_entry
        return True

    def step(
        self,
        *,
        index: int,
        bar: MarketBar,
        feature: FeatureSnapshot,
        exit_zscore_threshold: float | None,
        session_close_hour_utc: int,
        session_close_minute_utc: int,
    ) -> ExecutedTrade | None:
        if self.pending_entry is not None and index > self.pending_entry.generated_index and self.position is None:
            self.position, self.cash = open_position(
                pending=self.pending_entry,
                entry_bar=bar,
                index=index,
                cash=self.cash,
                fee_bps=self.fee_bps,
                slippage_bps=self.slippage_bps,
            )
            self.pending_entry = None

        if self.position is None:
            return None

        trade, self.cash, self.position = evaluate_open_position(
            position=self.position,
            bar=bar,
            feature=feature,
            index=index,
            cash=self.cash,
            fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps,
            exit_zscore_threshold=exit_zscore_threshold,
            session_close_hour_utc=session_close_hour_utc,
            session_close_minute_utc=session_close_minute_utc,
            intrabar_policy=self.intrabar_exit_policy,
        )
        return trade

    def equity(self, close_price: float) -> float:
        return compute_equity(self.cash, self.position, close_price)

    def force_close(self, *, bar: MarketBar, index: int, exit_reason: str = "end_of_data") -> ExecutedTrade | None:
        if self.position is None:
            return None
        trade, self.cash = close_position(
            position=self.position,
            exit_bar=bar,
            exit_price=bar.close,
            exit_reason=exit_reason,
            cash=self.cash,
            fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps,
            index=index,
        )
        self.position = None
        return trade


def _conservative_intrabar_exit(position: OpenPosition) -> tuple[str, float]:
    if position.side == SignalSide.LONG:
        stop_pnl = position.stop_price - position.entry_price
        target_pnl = position.target_price - position.entry_price
    else:
        stop_pnl = position.entry_price - position.stop_price
        target_pnl = position.entry_price - position.target_price

    if stop_pnl <= target_pnl:
        return "stop_loss", position.stop_price
    return "take_profit", position.target_price
