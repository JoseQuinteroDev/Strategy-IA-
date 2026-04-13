from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
import math
from typing import Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
    point_value: float
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


def resolve_gap_exit_policy(policy: str | None, default: str = "level") -> str:
    resolved = policy or default
    allowed = {"level", "open"}
    if resolved not in allowed:
        raise ValueError(f"Unsupported gap exit policy: {resolved}")
    return resolved


def _resolve_timezone(timezone: str | None) -> ZoneInfo:
    try:
        return ZoneInfo(timezone or "UTC")
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unsupported timezone: {timezone}") from exc


def _slippage_amount(price: float, slippage_bps: float, slippage_points: float) -> float:
    return (price * (slippage_bps / 10000.0)) + slippage_points


def apply_entry_slippage(
    side: SignalSide,
    price: float,
    slippage_bps: float,
    slippage_points: float = 0.0,
) -> float:
    slip = _slippage_amount(price, slippage_bps, slippage_points)
    if side == SignalSide.LONG:
        return price + slip
    return price - slip


def apply_exit_slippage(
    side: SignalSide,
    price: float,
    slippage_bps: float,
    slippage_points: float = 0.0,
) -> float:
    slip = _slippage_amount(price, slippage_bps, slippage_points)
    if side == SignalSide.LONG:
        return price - slip
    return price + slip


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
    unrealized = direction * (close_price - position.entry_price) * position.quantity * position.point_value
    return cash + unrealized


def is_session_close(
    timestamp: object,
    hour: int,
    minute: int,
    timezone: str = "UTC",
    session_windows: Sequence[str] | None = None,
) -> bool:
    normalized = pd.Timestamp(timestamp)
    normalized = normalized.tz_localize("UTC") if normalized.tzinfo is None else normalized.tz_convert(timezone)
    windows = _parse_session_windows(session_windows or ())
    if windows:
        current_time = normalized.time()
        if any(_time_in_window(current_time, start, end, end_inclusive=False) for start, end in windows):
            return False
        return any(_time_reached_window_end(current_time, start, end) for start, end in windows)
    session_close = time(hour, minute)
    return normalized.time() >= session_close


def is_within_session(
    timestamp: object,
    *,
    start_hour_utc: int,
    start_minute_utc: int,
    end_hour_utc: int,
    end_minute_utc: int,
    timezone: str = "UTC",
    session_windows: Sequence[str] | None = None,
) -> bool:
    normalized = pd.Timestamp(timestamp)
    normalized = normalized.tz_localize("UTC") if normalized.tzinfo is None else normalized.tz_convert(timezone)
    current_time = normalized.time()
    windows = _parse_session_windows(session_windows or ())
    if windows:
        return any(_time_in_window(current_time, start, end, end_inclusive=True) for start, end in windows)
    session_start = time(start_hour_utc, start_minute_utc)
    session_end = time(end_hour_utc, end_minute_utc)
    if session_start <= session_end:
        return session_start <= current_time <= session_end
    return current_time >= session_start or current_time <= session_end


def _parse_session_windows(raw_windows: Sequence[str]) -> list[tuple[time, time]]:
    windows: list[tuple[time, time]] = []
    for raw_window in raw_windows:
        if not str(raw_window).strip():
            continue
        try:
            start_raw, end_raw = str(raw_window).split("-", maxsplit=1)
            start_hour, start_minute = (int(part) for part in start_raw.strip().split(":", maxsplit=1))
            end_hour, end_minute = (int(part) for part in end_raw.strip().split(":", maxsplit=1))
        except ValueError as exc:
            raise ValueError(f"Invalid session window '{raw_window}'. Expected HH:MM-HH:MM.") from exc
        windows.append((time(start_hour, start_minute), time(end_hour, end_minute)))
    return windows


def _time_in_window(current: time, start: time, end: time, *, end_inclusive: bool) -> bool:
    if start <= end:
        return start <= current <= end if end_inclusive else start <= current < end
    if end_inclusive:
        return current >= start or current <= end
    return current >= start or current < end


def _time_reached_window_end(current: time, start: time, end: time) -> bool:
    if start <= end:
        return current >= end
    return end <= current < start


def _round_contract_quantity(
    quantity: float,
    *,
    contract_step: float,
    min_contracts: float,
    max_contracts: float | None,
) -> float:
    if not math.isfinite(quantity) or quantity <= 0.0:
        return 0.0
    adjusted = quantity
    if max_contracts is not None and max_contracts > 0.0:
        adjusted = min(adjusted, max_contracts)
    if contract_step > 0.0:
        adjusted = math.floor(adjusted / contract_step) * contract_step
    if min_contracts > 0.0 and adjusted < min_contracts:
        return 0.0
    return adjusted


def _fee_amount(
    *,
    price: float,
    quantity: float,
    point_value: float,
    fee_bps: float,
    fee_per_contract_per_side: float,
) -> float:
    notional_fee = price * quantity * point_value * (fee_bps / 10000.0)
    fixed_fee = abs(quantity) * fee_per_contract_per_side
    return notional_fee + fixed_fee


def resolve_gap_exit(position: OpenPosition, bar: MarketBar, *, gap_exit_policy: str) -> tuple[str, float] | None:
    if gap_exit_policy != "open":
        return None
    if position.side == SignalSide.LONG:
        if bar.open <= position.stop_price:
            return "gap_stop_loss", bar.open
        if bar.open >= position.target_price:
            return "gap_take_profit", bar.open
    else:
        if bar.open >= position.stop_price:
            return "gap_stop_loss", bar.open
        if bar.open <= position.target_price:
            return "gap_take_profit", bar.open
    return None


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
    point_value: float = 1.0,
    contract_step: float = 0.0,
    min_contracts: float = 0.0,
    max_contracts: float | None = None,
    fee_per_contract_per_side: float = 0.0,
    slippage_points: float = 0.0,
) -> tuple[OpenPosition | None, float]:
    raw_entry_price = apply_entry_slippage(
        pending.signal.side,
        entry_bar.open,
        slippage_bps,
        slippage_points,
    )
    risk_budget = cash * pending.size_fraction
    risk_per_contract = pending.stop_distance * point_value
    quantity_from_risk = risk_budget / risk_per_contract if risk_per_contract > 0.0 else 0.0
    quantity_from_leverage = (
        (cash * pending.max_leverage) / (raw_entry_price * point_value)
        if raw_entry_price > 0.0 and point_value > 0.0
        else 0.0
    )
    quantity = _round_contract_quantity(
        min(quantity_from_risk, quantity_from_leverage),
        contract_step=contract_step,
        min_contracts=min_contracts,
        max_contracts=max_contracts,
    )
    if quantity <= 0.0 or not math.isfinite(quantity):
        return None, cash

    entry_fee = _fee_amount(
        price=raw_entry_price,
        quantity=quantity,
        point_value=point_value,
        fee_bps=fee_bps,
        fee_per_contract_per_side=fee_per_contract_per_side,
    )
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
        point_value=point_value,
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
    fee_per_contract_per_side: float = 0.0,
    slippage_points: float = 0.0,
    index: int,
) -> tuple[ExecutedTrade, float]:
    slipped_exit_price = apply_exit_slippage(position.side, exit_price, slippage_bps, slippage_points)
    direction = 1.0 if position.side == SignalSide.LONG else -1.0
    gross_pnl = direction * (slipped_exit_price - position.entry_price) * position.quantity * position.point_value
    exit_fee = _fee_amount(
        price=slipped_exit_price,
        quantity=position.quantity,
        point_value=position.point_value,
        fee_bps=fee_bps,
        fee_per_contract_per_side=fee_per_contract_per_side,
    )
    cash_after_exit = cash + gross_pnl - exit_fee
    total_fees = position.entry_fee + exit_fee
    net_pnl = gross_pnl - total_fees

    metadata = dict(position.signal_metadata)
    metadata.update(
        {
            "point_value": position.point_value,
            "entry_fee": position.entry_fee,
            "exit_fee": exit_fee,
            "contract_quantity": position.quantity,
        }
    )
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
        return_pct=net_pnl / max(position.entry_price * position.quantity * position.point_value, 1e-12),
        bars_held=(index - position.entry_index) + 1,
        exit_reason=exit_reason,
        entry_reason=position.entry_reason,
        metadata=metadata,
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
    fee_per_contract_per_side: float = 0.0,
    slippage_points: float = 0.0,
    exit_zscore_threshold: float | None = None,
    session_close_hour_utc: int = 23,
    session_close_minute_utc: int = 55,
    session_close_timezone: str = "UTC",
    session_close_windows: Sequence[str] | None = None,
    intrabar_policy: str = "conservative",
    gap_exit_policy: str = "level",
) -> tuple[ExecutedTrade | None, float, OpenPosition | None]:
    if index > position.entry_index:
        gap_decision = resolve_gap_exit(position, bar, gap_exit_policy=gap_exit_policy)
        if gap_decision is not None:
            exit_reason, exit_price = gap_decision
            trade, updated_cash = close_position(
                position=position,
                exit_bar=bar,
                exit_price=exit_price,
                exit_reason=exit_reason,
                cash=cash,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                fee_per_contract_per_side=fee_per_contract_per_side,
                slippage_points=slippage_points,
                index=index,
            )
            return trade, updated_cash, None

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
            fee_per_contract_per_side=fee_per_contract_per_side,
            slippage_points=slippage_points,
            index=index,
        )
        return trade, updated_cash, None

    if position.close_on_session_end and is_session_close(
        bar.timestamp,
        session_close_hour_utc,
        session_close_minute_utc,
        session_close_timezone,
        session_close_windows,
    ):
        trade, updated_cash = close_position(
            position=position,
            exit_bar=bar,
            exit_price=bar.close,
            exit_reason="session_close",
            cash=cash,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            fee_per_contract_per_side=fee_per_contract_per_side,
            slippage_points=slippage_points,
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
            fee_per_contract_per_side=fee_per_contract_per_side,
            slippage_points=slippage_points,
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
                fee_per_contract_per_side=fee_per_contract_per_side,
                slippage_points=slippage_points,
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
    gap_exit_policy: str = "level"
    point_value: float = 1.0
    contract_step: float = 0.0
    min_contracts: float = 0.0
    max_contracts: float | None = None
    fee_per_contract_per_side: float = 0.0
    slippage_points: float = 0.0
    cash: float = field(init=False)
    pending_entry: PendingEntry | None = field(init=False, default=None)
    position: OpenPosition | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.intrabar_exit_policy = resolve_intrabar_policy(self.intrabar_exit_policy)
        self.gap_exit_policy = resolve_gap_exit_policy(self.gap_exit_policy)
        if self.point_value <= 0.0:
            raise ValueError("point_value must be positive.")
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
        session_close_timezone: str = "UTC",
        session_close_windows: Sequence[str] | None = None,
    ) -> ExecutedTrade | None:
        if self.pending_entry is not None and index > self.pending_entry.generated_index and self.position is None:
            self.position, self.cash = open_position(
                pending=self.pending_entry,
                entry_bar=bar,
                index=index,
                cash=self.cash,
                fee_bps=self.fee_bps,
                slippage_bps=self.slippage_bps,
                point_value=self.point_value,
                contract_step=self.contract_step,
                min_contracts=self.min_contracts,
                max_contracts=self.max_contracts,
                fee_per_contract_per_side=self.fee_per_contract_per_side,
                slippage_points=self.slippage_points,
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
            fee_per_contract_per_side=self.fee_per_contract_per_side,
            slippage_points=self.slippage_points,
            exit_zscore_threshold=exit_zscore_threshold,
            session_close_hour_utc=session_close_hour_utc,
            session_close_minute_utc=session_close_minute_utc,
            session_close_timezone=session_close_timezone,
            session_close_windows=session_close_windows,
            intrabar_policy=self.intrabar_exit_policy,
            gap_exit_policy=self.gap_exit_policy,
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
            fee_per_contract_per_side=self.fee_per_contract_per_side,
            slippage_points=self.slippage_points,
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
