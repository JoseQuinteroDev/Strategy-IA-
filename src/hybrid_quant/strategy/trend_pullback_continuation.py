from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hybrid_quant.core import SignalSide, StrategyContext, StrategySignal

from .base import IntradayStrategySupport, Strategy


@dataclass(slots=True)
class TrendPullbackContinuationStrategy(Strategy, IntradayStrategySupport):
    """Trend pullback continuation baseline using M15 bias, M5 setup, and optional M1 trigger."""

    name: str
    variant_name: str
    execution_timeframe: str
    filter_timeframe: str
    entry_mode: str = "core_v1"
    atr_multiple_target: float = 2.0
    time_stop_bars: int = 60
    close_on_session_end: bool = True
    session_close_hour_utc: int = 16
    session_close_minute_utc: int = 30
    session_close_timezone: str = "Europe/Madrid"
    no_entry_minutes_before_close: int = 5
    enforce_entry_session: bool = True
    entry_session_start_hour_utc: int = 9
    entry_session_start_minute_utc: int = 0
    entry_session_end_hour_utc: int = 16
    entry_session_end_minute_utc: int = 30
    entry_session_timezone: str = "Europe/Madrid"
    entry_session_windows: list[str] | None = None
    blocked_hours_utc: list[int] | None = None
    allowed_hours_utc: list[int] | None = None
    allowed_hours_long_utc: list[int] | None = None
    allowed_hours_short_utc: list[int] | None = None
    allowed_weekdays: list[int] | None = None
    allowed_sides: list[str] | None = None
    exclude_weekends: bool = True
    stop_buffer_atr: float = 0.10
    minimum_stop_atr: float = 0.60
    maximum_stop_atr: float = 1.50
    maximum_vwap_distance_atr: float = 0.80
    default_spread_points: float = 0.10
    maximum_spread_points: float = 0.30
    maximum_spread_to_stop_ratio: float = 0.12
    use_macd_confirmation: bool = False
    use_m1_trigger: bool = True
    news_filter_enabled: bool = False
    news_block_minutes_before: int = 15
    news_block_minutes_after: int = 15

    def generate(self, context: StrategyContext) -> StrategySignal:
        timestamp = self._resolve_timestamp(context)
        metadata = self._base_metadata()
        if not context.bars or not context.features:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Insufficient context for trend pullback continuation.",
                metadata=metadata,
            )

        latest_bar = context.bars[-1]
        values = context.features[-1].values
        metadata = {**metadata, **self._entry_session_metadata(timestamp)}

        if self._is_session_close(timestamp):
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Session close: trend pullback baseline does not hold beyond the configured window.",
                metadata={**metadata, "session_close_exit": True},
            )

        session_gate_reason = self._session_gate_reason(timestamp)
        if session_gate_reason is not None:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale=session_gate_reason,
                metadata={**metadata, "blocked_by_filter": "outside_session", "candidate_status": "blocked_by_time_filter"},
            )

        if self._minutes_to_session_close(timestamp) <= self.no_entry_minutes_before_close:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="No new trend pullback entries are allowed inside the close buffer.",
                metadata={**metadata, "blocked_by_filter": "close_buffer"},
            )

        if self.news_filter_enabled:
            metadata["news_filter_status"] = "configured_but_no_news_provider_attached"
        else:
            metadata["news_filter_status"] = "disabled_no_news_provider"

        setup = self._resolve_setup(values)
        if setup is None:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="No valid trend pullback setup or trigger is active.",
                metadata={**metadata, **self._feature_metadata(values)},
            )

        side = setup["side"]
        direction_gate_reason = self._direction_gate_reason(side)
        if direction_gate_reason is not None:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale=direction_gate_reason,
                metadata={**metadata, **setup, "direction_gate": True},
            )

        stop_price = self._stop_price(values, side=side)
        atr_m5 = self._get_feature_value(values, "trend_pullback_atr_m5")
        if stop_price is None or atr_m5 is None or atr_m5 <= 0.0:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Missing stop or ATR context for trend pullback setup.",
                metadata={**metadata, **setup},
            )

        entry_price = latest_bar.close
        stop_distance = abs(entry_price - stop_price)
        min_distance = self.minimum_stop_atr * atr_m5
        max_distance = self.maximum_stop_atr * atr_m5
        if stop_distance < min_distance:
            stop_distance = min_distance
            stop_price = entry_price - stop_distance if side == SignalSide.LONG else entry_price + stop_distance
        if stop_distance > max_distance:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Trend pullback stop is too wide relative to M5 ATR.",
                metadata={**metadata, **setup, "stop_distance_atr": stop_distance / atr_m5},
            )

        spread = self._spread_points(values)
        if self.maximum_spread_points > 0.0 and spread > self.maximum_spread_points:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Spread filter blocked the trend pullback setup.",
                metadata={**metadata, **setup, "spread_points": spread},
            )
        if stop_distance > 0.0 and spread > self.maximum_spread_to_stop_ratio * stop_distance:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Spread-to-stop filter blocked the trend pullback setup.",
                metadata={**metadata, **setup, "spread_to_stop": spread / stop_distance},
            )

        target_distance = stop_distance * self.atr_multiple_target
        target_price = entry_price + target_distance if side == SignalSide.LONG else entry_price - target_distance
        return StrategySignal(
            symbol=context.symbol,
            timestamp=timestamp,
            side=side,
            strength=1.0,
            rationale=setup["entry_reason"],
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            time_stop_bars=self.time_stop_bars,
            close_on_session_end=self.close_on_session_end,
            entry_reason=setup["entry_reason"],
            metadata={
                **metadata,
                **setup,
                **self._feature_metadata(values),
                "stop_distance": stop_distance,
                "stop_distance_atr": stop_distance / atr_m5,
                "target_r_multiple": self.atr_multiple_target,
                "spread_points": spread,
                "spread_to_stop": spread / stop_distance if stop_distance > 0.0 else None,
            },
        )

    def _resolve_setup(self, values: dict[str, float]) -> dict[str, Any] | None:
        use_macd = self.use_macd_confirmation or self.entry_mode == "core_v1_macd"
        use_m1 = self.use_m1_trigger and self.entry_mode != "core_v1_no_m1"
        suffix = "macd" if use_macd else "core"
        long_column = f"trend_pullback_trigger_long_{suffix}" if use_m1 else f"trend_pullback_setup_long_{suffix}"
        short_column = f"trend_pullback_trigger_short_{suffix}" if use_m1 else f"trend_pullback_setup_short_{suffix}"
        if self._get_feature_value(values, long_column) == 1.0:
            return {
                "side": SignalSide.LONG,
                "entry_reason": f"trend_pullback_{self.entry_mode}_long",
                "setup_family": "baseline_trend_pullback_v1",
                "setup_variant": self.variant_name,
                "uses_macd": use_macd,
                "uses_m1_trigger": use_m1,
            }
        if self._get_feature_value(values, short_column) == 1.0:
            return {
                "side": SignalSide.SHORT,
                "entry_reason": f"trend_pullback_{self.entry_mode}_short",
                "setup_family": "baseline_trend_pullback_v1",
                "setup_variant": self.variant_name,
                "uses_macd": use_macd,
                "uses_m1_trigger": use_m1,
            }
        return None

    def _stop_price(self, values: dict[str, float], *, side: SignalSide) -> float | None:
        column = "trend_pullback_stop_long" if side == SignalSide.LONG else "trend_pullback_stop_short"
        return self._get_feature_value(values, column)

    def _spread_points(self, values: dict[str, float]) -> float:
        spread = self._get_feature_value(values, "spread")
        if spread is None:
            spread = self._get_feature_value(values, "trend_pullback_spread_points")
        return float(spread if spread is not None and spread > 0.0 else self.default_spread_points)

    def _base_metadata(self) -> dict[str, Any]:
        return {
            "strategy_family": "baseline_trend_pullback_v1",
            "variant_name": self.variant_name,
            "entry_mode": self.entry_mode,
            "timezone": self.entry_session_timezone,
            "entry_session_windows": ",".join(self.entry_session_windows or []),
            "news_filter_enabled": self.news_filter_enabled,
        }

    def _feature_metadata(self, values: dict[str, float]) -> dict[str, Any]:
        keys = [
            "trend_pullback_bias_m15",
            "trend_pullback_rsi_m5",
            "trend_pullback_rsi_m1",
            "trend_pullback_atr_m5",
            "trend_pullback_vwap_distance_atr_m5",
            "trend_pullback_macd_hist_m5",
        ]
        return {key: self._get_feature_value(values, key) for key in keys}
