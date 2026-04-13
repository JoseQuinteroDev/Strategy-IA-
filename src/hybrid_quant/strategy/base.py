from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime, time, timedelta
import math
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from hybrid_quant.core import SignalSide, StrategyContext, StrategySignal


class Strategy(ABC):
    @abstractmethod
    def generate(self, context: StrategyContext) -> StrategySignal:
        """Produce a signal from market context."""


class IntradayStrategySupport:
    execution_timeframe: str
    filter_timeframe: str
    atr_multiple_stop: float
    atr_multiple_target: float
    time_stop_bars: int
    close_on_session_end: bool
    session_close_hour_utc: int
    session_close_minute_utc: int
    no_entry_minutes_before_close: int
    enforce_entry_session: bool
    entry_session_start_hour_utc: int
    entry_session_start_minute_utc: int
    entry_session_end_hour_utc: int
    entry_session_end_minute_utc: int
    entry_session_timezone: str
    entry_session_windows: list[str] | None
    blocked_hours_utc: list[int] | None
    allowed_hours_utc: list[int] | None
    allowed_hours_long_utc: list[int] | None
    allowed_hours_short_utc: list[int] | None
    allowed_weekdays: list[int] | None
    allowed_sides: list[str] | None
    exclude_weekends: bool

    def _resolve_timestamp(self, context: StrategyContext) -> datetime:
        if context.features:
            return context.features[-1].timestamp
        if context.bars:
            return context.bars[-1].timestamp
        return datetime.now(UTC)

    def _get_feature_value(self, values: dict[str, float], name: str) -> float | None:
        raw_value = values.get(name)
        if raw_value is None:
            return None
        if not math.isfinite(float(raw_value)):
            return None
        return float(raw_value)

    def _trend_bias(self, close_price: float, anchor_value: float) -> SignalSide | None:
        if close_price > anchor_value:
            return SignalSide.LONG
        if close_price < anchor_value:
            return SignalSide.SHORT
        return None

    def _direction_gate_reason(self, side: SignalSide) -> str | None:
        allowed_sides = {value.strip().lower() for value in (self.allowed_sides or []) if value.strip()}
        if allowed_sides and side.value not in allowed_sides:
            return "Directional filter blocked the setup for this baseline variant."
        return None

    def _side_hour_gate_reason(self, *, side: SignalSide, timestamp: datetime) -> str | None:
        normalized = timestamp.astimezone(UTC)
        if side == SignalSide.LONG:
            allowed_hours = set(self.allowed_hours_long_utc or [])
            if allowed_hours and normalized.hour not in allowed_hours:
                return "Long-hour whitelist blocked the setup for this baseline variant."
        if side == SignalSide.SHORT:
            allowed_hours = set(self.allowed_hours_short_utc or [])
            if allowed_hours and normalized.hour not in allowed_hours:
                return "Short-hour whitelist blocked the setup for this baseline variant."
        return None

    def _entry_signal(
        self,
        *,
        symbol: str,
        timestamp: datetime,
        side: SignalSide,
        entry_price: float,
        atr: float,
        entry_reason: str,
        metadata: dict[str, Any],
    ) -> StrategySignal:
        if side == SignalSide.LONG:
            stop_price = entry_price - (atr * self.atr_multiple_stop)
            target_price = entry_price + (atr * self.atr_multiple_target)
        else:
            stop_price = entry_price + (atr * self.atr_multiple_stop)
            target_price = entry_price - (atr * self.atr_multiple_target)

        return StrategySignal(
            symbol=symbol,
            timestamp=timestamp,
            side=side,
            strength=1.0,
            rationale=entry_reason,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            time_stop_bars=self.time_stop_bars,
            close_on_session_end=getattr(self, "close_on_session_end", True),
            entry_reason=entry_reason,
            metadata=metadata,
        )

    def _flat_signal(
        self,
        *,
        symbol: str,
        timestamp: datetime,
        rationale: str,
        metadata: dict[str, Any],
    ) -> StrategySignal:
        return StrategySignal(
            symbol=symbol,
            timestamp=timestamp,
            side=SignalSide.FLAT,
            strength=0.0,
            rationale=rationale,
            entry_price=None,
            stop_price=None,
            target_price=None,
            time_stop_bars=self.time_stop_bars,
            close_on_session_end=getattr(self, "close_on_session_end", True),
            entry_reason=None,
            metadata=metadata,
        )

    def _session_gate_reason(self, timestamp: datetime) -> str | None:
        normalized = timestamp.astimezone(UTC)
        if getattr(self, "enforce_entry_session", False) and not self._is_inside_entry_session(timestamp):
            return (
                "outside_session: do not trade. The timestamp is outside the configured "
                f"entry window {self._entry_session_window_label()}."
            )

        if self.exclude_weekends and normalized.weekday() >= 5:
            return "Weekend trading is disabled for the selective baseline variant."

        allowed_weekdays = set(self.allowed_weekdays or [])
        if allowed_weekdays and normalized.weekday() not in allowed_weekdays:
            return "Weekday filter blocked the setup for the selective baseline variant."

        allowed_hours = set(self.allowed_hours_utc or [])
        if allowed_hours and normalized.hour not in allowed_hours:
            return "Hour whitelist blocked the setup for the selective baseline variant."

        blocked_hours = set(self.blocked_hours_utc or [])
        if normalized.hour in blocked_hours:
            return "Blocked trading hour for the selective baseline variant."

        return None

    def _is_inside_entry_session(self, timestamp: datetime) -> bool:
        normalized = self._to_entry_session_timezone(timestamp)
        current_time = normalized.time()
        windows = self._entry_session_windows()
        if windows:
            return any(_time_in_window(current_time, start, end, end_inclusive=False) for start, end in windows)
        session_start = time(
            getattr(self, "entry_session_start_hour_utc", 0),
            getattr(self, "entry_session_start_minute_utc", 0),
        )
        session_end = time(
            getattr(self, "entry_session_end_hour_utc", 23),
            getattr(self, "entry_session_end_minute_utc", 55),
        )
        if session_start <= session_end:
            return session_start <= current_time < session_end
        return current_time >= session_start or current_time < session_end

    def _entry_session_window_label(self) -> str:
        timezone = getattr(self, "entry_session_timezone", "UTC")
        windows = self._entry_session_windows()
        if windows:
            rendered = ",".join(f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}" for start, end in windows)
            return f"{rendered} {timezone}"
        return (
            f"{getattr(self, 'entry_session_start_hour_utc', 0):02d}:"
            f"{getattr(self, 'entry_session_start_minute_utc', 0):02d}-"
            f"{getattr(self, 'entry_session_end_hour_utc', 23):02d}:"
            f"{getattr(self, 'entry_session_end_minute_utc', 55):02d} {timezone}"
        )

    def _entry_session_metadata(self, timestamp: datetime) -> dict[str, Any]:
        inside = self._is_inside_entry_session(timestamp)
        return {
            "entry_session_window_utc": self._entry_session_window_label(),
            "entry_session_timezone": getattr(self, "entry_session_timezone", "UTC"),
            "entry_session_start_utc": (
                f"{getattr(self, 'entry_session_start_hour_utc', 0):02d}:"
                f"{getattr(self, 'entry_session_start_minute_utc', 0):02d}"
            ),
            "entry_session_end_utc": (
                f"{getattr(self, 'entry_session_end_hour_utc', 23):02d}:"
                f"{getattr(self, 'entry_session_end_minute_utc', 55):02d}"
            ),
            "inside_entry_session": inside,
            "outside_session": not inside,
        }

    def _to_entry_session_timezone(self, timestamp: datetime) -> datetime:
        timezone = getattr(self, "entry_session_timezone", "UTC") or "UTC"
        try:
            return timestamp.astimezone(ZoneInfo(timezone))
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unsupported entry session timezone: {timezone}") from exc

    def _entry_session_windows(self) -> list[tuple[time, time]]:
        raw_windows = getattr(self, "entry_session_windows", None) or []
        return [_parse_session_window(raw_window) for raw_window in raw_windows if str(raw_window).strip()]

    def _minutes_to_session_close(self, timestamp: datetime) -> int:
        normalized = self._to_session_close_timezone(timestamp)
        session_close = normalized.replace(
            hour=self.session_close_hour_utc,
            minute=self.session_close_minute_utc,
            second=0,
            microsecond=0,
        )
        remaining = session_close - normalized
        if remaining <= timedelta(0):
            return 0
        return int(remaining.total_seconds() // 60)

    def _is_session_close(self, timestamp: datetime) -> bool:
        normalized = self._to_session_close_timezone(timestamp)
        session_close = time(self.session_close_hour_utc, self.session_close_minute_utc)
        return normalized.time() >= session_close

    def _to_session_close_timezone(self, timestamp: datetime) -> datetime:
        timezone = getattr(self, "session_close_timezone", "UTC") or "UTC"
        try:
            return timestamp.astimezone(ZoneInfo(timezone))
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unsupported session close timezone: {timezone}") from exc


def _parse_session_window(raw_window: str) -> tuple[time, time]:
    try:
        start_raw, end_raw = str(raw_window).split("-", maxsplit=1)
        start_hour, start_minute = (int(part) for part in start_raw.strip().split(":", maxsplit=1))
        end_hour, end_minute = (int(part) for part in end_raw.strip().split(":", maxsplit=1))
    except ValueError as exc:
        raise ValueError(f"Invalid session window '{raw_window}'. Expected HH:MM-HH:MM.") from exc
    return time(start_hour, start_minute), time(end_hour, end_minute)


def _time_in_window(current: time, start: time, end: time, *, end_inclusive: bool) -> bool:
    if start <= end:
        return start <= current <= end if end_inclusive else start <= current < end
    if end_inclusive:
        return current >= start or current <= end
    return current >= start or current < end
