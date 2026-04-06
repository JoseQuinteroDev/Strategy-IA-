from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hybrid_quant.core import SignalSide, StrategyContext, StrategySignal

from .base import IntradayStrategySupport, Strategy


@dataclass(slots=True)
class OpeningRangeBreakoutStrategy(Strategy, IntradayStrategySupport):
    name: str
    variant_name: str
    trend_filter: str
    execution_timeframe: str
    filter_timeframe: str
    entry_mode: str = "breakout_close_entry"
    opening_range_minutes: int = 30
    retest_max_bars: int = 3
    atr_multiple_stop: float = 1.0
    atr_multiple_target: float = 2.0
    time_stop_bars: int = 18
    session_close_hour_utc: int = 20
    session_close_minute_utc: int = 55
    no_entry_minutes_before_close: int = 20
    blocked_hours_utc: list[int] | None = None
    allowed_hours_utc: list[int] | None = None
    allowed_hours_long_utc: list[int] | None = None
    allowed_hours_short_utc: list[int] | None = None
    allowed_weekdays: list[int] | None = None
    allowed_sides: list[str] | None = None
    exclude_weekends: bool = True
    minimum_expected_move_bps: float = 0.0
    minimum_target_to_cost_ratio: float = 0.0
    estimated_round_trip_cost_bps: float = 0.0
    momentum_lookback_bars: int = 20
    minimum_momentum_abs: float = 0.0
    minimum_candle_range_atr: float = 0.0
    use_ema_200_1h_slope: bool = True
    minimum_opening_range_width_atr: float = 0.0
    maximum_opening_range_width_atr: float = 0.0
    minimum_relative_volume: float = 0.0
    max_breakout_distance_atr: float = 0.0
    max_breakouts_per_day: int = 1

    def generate(self, context: StrategyContext) -> StrategySignal:
        timestamp = self._resolve_timestamp(context)
        base_metadata = {
            "strategy": self.name,
            "variant_name": self.variant_name,
            "strategy_family": "opening_range_breakout",
            "execution_timeframe": self.execution_timeframe,
            "filter_timeframe": self.filter_timeframe,
            "trend_filter": self.trend_filter,
            "entry_mode": self.entry_mode,
            "opening_range_minutes": self.opening_range_minutes,
            "retest_max_bars": self.retest_max_bars,
            "time_stop_bars": self.time_stop_bars,
            "close_on_session_end": True,
            "blocked_hours_utc": list(self.blocked_hours_utc or []),
            "allowed_hours_utc": list(self.allowed_hours_utc or []),
            "allowed_hours_long_utc": list(self.allowed_hours_long_utc or []),
            "allowed_hours_short_utc": list(self.allowed_hours_short_utc or []),
            "allowed_weekdays": list(self.allowed_weekdays or []),
            "allowed_sides": list(self.allowed_sides or []),
            "exclude_weekends": self.exclude_weekends,
            "minimum_momentum_abs": self.minimum_momentum_abs,
            "momentum_lookback_bars": self.momentum_lookback_bars,
            "minimum_candle_range_atr": self.minimum_candle_range_atr,
            "use_ema_200_1h_slope": self.use_ema_200_1h_slope,
            "minimum_opening_range_width_atr": self.minimum_opening_range_width_atr,
            "maximum_opening_range_width_atr": self.maximum_opening_range_width_atr,
            "minimum_relative_volume": self.minimum_relative_volume,
            "max_breakout_distance_atr": self.max_breakout_distance_atr,
            "minimum_expected_move_bps": self.minimum_expected_move_bps,
            "minimum_target_to_cost_ratio": self.minimum_target_to_cost_ratio,
            "estimated_round_trip_cost_bps": self.estimated_round_trip_cost_bps,
            "max_breakouts_per_day": self.max_breakouts_per_day,
        }

        if not context.bars or not context.features:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Insufficient market context to evaluate the opening range breakout strategy.",
                metadata=base_metadata,
            )

        latest_bar = context.bars[-1]
        latest_features = context.features[-1].values

        if self._is_session_close(timestamp):
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Session close: flatten intraday exposure and avoid overnight risk.",
                metadata={**base_metadata, "session_close_exit": True},
            )

        minutes_to_close = self._minutes_to_session_close(timestamp)
        if minutes_to_close <= self.no_entry_minutes_before_close:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="No new ORB entries are allowed inside the session-close buffer.",
                metadata={**base_metadata, "minutes_to_close": minutes_to_close},
            )

        session_gate_reason = self._session_gate_reason(timestamp)
        if session_gate_reason is not None:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale=session_gate_reason,
                metadata={**base_metadata, "session_gate": True},
            )

        close_price = latest_bar.close
        atr = self._get_feature_value(latest_features, "atr_14")
        ema_200_1h = self._get_feature_value(latest_features, "ema_200_1h")
        ema_200_1h_slope = self._get_feature_value(latest_features, "ema_200_1h_slope")
        opening_range_high = self._get_feature_value(latest_features, "opening_range_high")
        opening_range_low = self._get_feature_value(latest_features, "opening_range_low")
        opening_range_width = self._get_feature_value(latest_features, "opening_range_width")
        opening_range_width_atr = self._get_feature_value(latest_features, "opening_range_width_atr")
        opening_range_ready = self._get_feature_value(latest_features, "opening_range_ready")
        momentum = self._get_feature_value(
            latest_features,
            f"momentum_{self.momentum_lookback_bars}",
        )
        if momentum is None:
            momentum = self._get_feature_value(latest_features, "momentum_20")
        candle_range_atr = self._get_feature_value(latest_features, "candle_range_atr")
        relative_volume = self._get_feature_value(latest_features, "relative_volume")
        long_breakout_entry = self._get_feature_value(latest_features, "opening_range_long_breakout_entry")
        short_breakout_entry = self._get_feature_value(latest_features, "opening_range_short_breakout_entry")
        long_retest_entry = self._get_feature_value(latest_features, "opening_range_long_retest_entry")
        short_retest_entry = self._get_feature_value(latest_features, "opening_range_short_retest_entry")
        breakout_count_today = self._get_feature_value(latest_features, "opening_range_breakout_count_today")
        first_breakout_of_day = self._get_feature_value(latest_features, "opening_range_first_breakout_of_day")

        required_values = {
            "atr_14": atr,
            "ema_200_1h": ema_200_1h,
            "ema_200_1h_slope": ema_200_1h_slope,
            "opening_range_high": opening_range_high,
            "opening_range_low": opening_range_low,
            "opening_range_width": opening_range_width,
            "opening_range_width_atr": opening_range_width_atr,
            "opening_range_ready": opening_range_ready,
        }
        missing = [name for name, value in required_values.items() if value is None]
        if missing:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale=f"Missing or invalid ORB features for signal generation: {missing}.",
                metadata=base_metadata,
            )

        if opening_range_ready <= 0.0:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Opening range is not complete yet, so the ORB baseline stays flat.",
                metadata={
                    **base_metadata,
                    "opening_range_high": opening_range_high,
                    "opening_range_low": opening_range_low,
                    "opening_range_width": opening_range_width,
                    "opening_range_width_atr": opening_range_width_atr,
                },
            )

        side, trigger_name = self._resolve_entry_side(
            long_breakout_entry=long_breakout_entry,
            short_breakout_entry=short_breakout_entry,
            long_retest_entry=long_retest_entry,
            short_retest_entry=short_retest_entry,
        )
        if side is None:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="No valid opening range breakout trigger is active on this bar.",
                metadata={
                    **base_metadata,
                    "opening_range_high": opening_range_high,
                    "opening_range_low": opening_range_low,
                    "opening_range_width": opening_range_width,
                    "opening_range_width_atr": opening_range_width_atr,
                },
            )

        direction_gate_reason = self._direction_gate_reason(side)
        if direction_gate_reason is not None:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale=direction_gate_reason,
                metadata={**base_metadata, "direction_gate": True},
            )

        side_hour_gate_reason = self._side_hour_gate_reason(side=side, timestamp=timestamp)
        if side_hour_gate_reason is not None:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale=side_hour_gate_reason,
                metadata={**base_metadata, "directional_hour_gate": True},
            )

        breakout_count = int(breakout_count_today) if breakout_count_today is not None else 0
        if self.max_breakouts_per_day > 0 and breakout_count > self.max_breakouts_per_day:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="The ORB baseline already used the maximum number of daily breakout opportunities.",
                metadata={**base_metadata, "opening_range_breakout_count_today": breakout_count},
            )

        breakout_level = opening_range_high if side == SignalSide.LONG else opening_range_low
        breakout_distance = (
            close_price - breakout_level if side == SignalSide.LONG else breakout_level - close_price
        )
        breakout_distance_atr = breakout_distance / atr if atr > 0.0 else float("nan")
        expected_move = atr * self.atr_multiple_target
        expected_move_bps = (expected_move / close_price) * 10000.0 if close_price > 0.0 else float("nan")
        target_to_cost_ratio = (
            expected_move_bps / self.estimated_round_trip_cost_bps
            if self.estimated_round_trip_cost_bps > 0.0 and expected_move_bps == expected_move_bps
            else float("inf")
        )

        quality_metadata = {
            "opening_range_high": opening_range_high,
            "opening_range_low": opening_range_low,
            "opening_range_width": opening_range_width,
            "opening_range_width_atr": opening_range_width_atr,
            "ema_200_1h": ema_200_1h,
            "ema_200_1h_slope": ema_200_1h_slope,
            "momentum": momentum,
            "abs_momentum": abs(momentum) if momentum is not None else None,
            "candle_range_atr": candle_range_atr,
            "relative_volume": relative_volume,
            "breakout_level": breakout_level,
            "breakout_distance": breakout_distance,
            "breakout_distance_atr": breakout_distance_atr,
            "opening_range_breakout_count_today": breakout_count,
            "first_breakout_of_day": bool(first_breakout_of_day == 1.0),
            "entry_trigger": trigger_name,
            "target_to_cost_ratio": target_to_cost_ratio,
            "expected_move_bps": expected_move_bps,
        }
        quality_reason = self._quality_gate_failure(
            side=side,
            close_price=close_price,
            ema_200_1h=ema_200_1h,
            ema_200_1h_slope=ema_200_1h_slope,
            opening_range_width_atr=opening_range_width_atr,
            momentum=momentum,
            candle_range_atr=candle_range_atr,
            relative_volume=relative_volume,
            breakout_distance_atr=breakout_distance_atr,
            expected_move_bps=expected_move_bps,
            target_to_cost_ratio=target_to_cost_ratio,
        )
        if quality_reason is not None:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale=quality_reason,
                metadata={**base_metadata, **quality_metadata},
            )

        if side == SignalSide.LONG:
            entry_reason = (
                "Long ORB: opening range high was broken with the 1H EMA200 and its slope aligned upward."
                if self.entry_mode == "breakout_close_entry"
                else "Long ORB retest: a valid retest of the broken opening-range high confirmed the entry."
            )
        else:
            entry_reason = (
                "Short ORB: opening range low was broken with the 1H EMA200 and its slope aligned downward."
                if self.entry_mode == "breakout_close_entry"
                else "Short ORB retest: a valid retest of the broken opening-range low confirmed the entry."
            )

        return self._entry_signal(
            symbol=context.symbol,
            timestamp=timestamp,
            side=side,
            entry_price=close_price,
            atr=atr,
            entry_reason=entry_reason,
            metadata={**base_metadata, **quality_metadata},
        )

    def _resolve_entry_side(
        self,
        *,
        long_breakout_entry: float | None,
        short_breakout_entry: float | None,
        long_retest_entry: float | None,
        short_retest_entry: float | None,
    ) -> tuple[SignalSide | None, str | None]:
        if self.entry_mode == "breakout_close_entry":
            if long_breakout_entry == 1.0:
                return SignalSide.LONG, "breakout_close_entry"
            if short_breakout_entry == 1.0:
                return SignalSide.SHORT, "breakout_close_entry"
            return None, None

        if self.entry_mode == "breakout_retest_entry":
            if long_retest_entry == 1.0:
                return SignalSide.LONG, "breakout_retest_entry"
            if short_retest_entry == 1.0:
                return SignalSide.SHORT, "breakout_retest_entry"
            return None, None

        raise ValueError(f"Unsupported ORB entry mode: {self.entry_mode}")

    def _quality_gate_failure(
        self,
        *,
        side: SignalSide,
        close_price: float,
        ema_200_1h: float,
        ema_200_1h_slope: float,
        opening_range_width_atr: float,
        momentum: float | None,
        candle_range_atr: float | None,
        relative_volume: float | None,
        breakout_distance_atr: float,
        expected_move_bps: float,
        target_to_cost_ratio: float,
    ) -> str | None:
        if side == SignalSide.LONG and close_price <= ema_200_1h:
            return "Long ORB rejected because price is not above the 1H EMA200 trend filter."
        if side == SignalSide.SHORT and close_price >= ema_200_1h:
            return "Short ORB rejected because price is not below the 1H EMA200 trend filter."

        if self.use_ema_200_1h_slope:
            if side == SignalSide.LONG and ema_200_1h_slope <= 0.0:
                return "Long ORB rejected because the 1H EMA200 slope is not positive."
            if side == SignalSide.SHORT and ema_200_1h_slope >= 0.0:
                return "Short ORB rejected because the 1H EMA200 slope is not negative."

        if (
            self.minimum_opening_range_width_atr > 0.0
            and opening_range_width_atr < self.minimum_opening_range_width_atr
        ):
            return "ORB rejected because the opening range is too narrow relative to ATR."

        if (
            self.maximum_opening_range_width_atr > 0.0
            and opening_range_width_atr > self.maximum_opening_range_width_atr
        ):
            return "ORB rejected because the opening range is too wide relative to ATR."

        if momentum is None:
            return "ORB rejected because momentum is missing."
        if side == SignalSide.LONG and momentum <= 0.0:
            return "Long ORB rejected because momentum is not aligned upward."
        if side == SignalSide.SHORT and momentum >= 0.0:
            return "Short ORB rejected because momentum is not aligned downward."
        if self.minimum_momentum_abs > 0.0 and abs(momentum) < self.minimum_momentum_abs:
            return "ORB rejected because momentum is too weak for a quality breakout."

        if candle_range_atr is None:
            return "ORB rejected because candle-range expansion is missing."
        if self.minimum_candle_range_atr > 0.0 and candle_range_atr < self.minimum_candle_range_atr:
            return "ORB rejected because the breakout candle lacks volatility expansion."

        if relative_volume is not None and self.minimum_relative_volume > 0.0 and relative_volume < self.minimum_relative_volume:
            return "ORB rejected because relative volume is too weak."

        if self.max_breakout_distance_atr > 0.0 and breakout_distance_atr > self.max_breakout_distance_atr:
            return "ORB rejected because the breakout candle is already too extended away from the broken level."

        if self.minimum_expected_move_bps > 0.0 and expected_move_bps < self.minimum_expected_move_bps:
            return "ORB rejected because the projected move is too small in basis points."

        if self.minimum_target_to_cost_ratio > 0.0 and target_to_cost_ratio < self.minimum_target_to_cost_ratio:
            return "ORB rejected because the target-to-cost ratio is too weak."

        return None
