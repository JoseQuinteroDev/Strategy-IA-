from __future__ import annotations

from dataclasses import dataclass

from hybrid_quant.core import SignalSide, StrategyContext, StrategySignal

from .base import IntradayStrategySupport, Strategy


@dataclass(slots=True)
class IntradayNasdaqContextualStrategy(Strategy, IntradayStrategySupport):
    name: str
    variant_name: str
    trend_filter: str
    execution_timeframe: str
    filter_timeframe: str
    entry_mode: str = "context_pullback_continuation"
    opening_range_minutes: int = 30
    retest_max_bars: int = 6
    atr_multiple_stop: float = 1.0
    atr_multiple_target: float = 1.8
    time_stop_bars: int = 14
    session_close_hour_utc: int = 20
    session_close_minute_utc: int = 55
    no_entry_minutes_before_close: int = 15
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
    use_ema_200_1h_trend_filter: bool = True
    use_ema_200_1h_slope: bool = True
    ema_200_1h_slope_tolerance: float = 0.0
    minimum_opening_range_width_atr: float = 0.0
    maximum_opening_range_width_atr: float = 0.0
    minimum_relative_volume: float = 0.0
    max_breakout_distance_atr: float = 0.0
    max_breakouts_per_day: int = 4
    use_intraday_vwap_filter: bool = True
    use_intraday_ema20_filter: bool = True
    use_intraday_ema50_alignment: bool = True
    use_opening_range_mid_filter: bool = True

    def generate(self, context: StrategyContext) -> StrategySignal:
        timestamp = self._resolve_timestamp(context)
        base_metadata = {
            "strategy": self.name,
            "variant_name": self.variant_name,
            "strategy_family": "intraday_nasdaq_contextual",
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
            "use_ema_200_1h_trend_filter": self.use_ema_200_1h_trend_filter,
            "use_ema_200_1h_slope": self.use_ema_200_1h_slope,
            "ema_200_1h_slope_tolerance": self.ema_200_1h_slope_tolerance,
            "minimum_opening_range_width_atr": self.minimum_opening_range_width_atr,
            "maximum_opening_range_width_atr": self.maximum_opening_range_width_atr,
            "minimum_relative_volume": self.minimum_relative_volume,
            "max_breakout_distance_atr": self.max_breakout_distance_atr,
            "minimum_expected_move_bps": self.minimum_expected_move_bps,
            "minimum_target_to_cost_ratio": self.minimum_target_to_cost_ratio,
            "estimated_round_trip_cost_bps": self.estimated_round_trip_cost_bps,
            "max_breakouts_per_day": self.max_breakouts_per_day,
            "use_intraday_vwap_filter": self.use_intraday_vwap_filter,
            "use_intraday_ema20_filter": self.use_intraday_ema20_filter,
            "use_intraday_ema50_alignment": self.use_intraday_ema50_alignment,
            "use_opening_range_mid_filter": self.use_opening_range_mid_filter,
        }

        if not context.bars or not context.features:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Insufficient market context to evaluate the intraday contextual Nasdaq strategy.",
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
                rationale="No new intraday contextual entries are allowed inside the session-close buffer.",
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
        intraday_vwap = self._get_feature_value(latest_features, "intraday_vwap")
        ema_20 = self._get_feature_value(latest_features, "ema_20")
        ema_50 = self._get_feature_value(latest_features, "ema_50")
        opening_range_high = self._get_feature_value(latest_features, "opening_range_high")
        opening_range_low = self._get_feature_value(latest_features, "opening_range_low")
        opening_range_mid = self._get_feature_value(latest_features, "opening_range_mid")
        opening_range_width = self._get_feature_value(latest_features, "opening_range_width")
        opening_range_width_atr = self._get_feature_value(latest_features, "opening_range_width_atr")
        opening_range_ready = self._get_feature_value(latest_features, "opening_range_ready")
        context_bias_side = self._get_feature_value(latest_features, "context_bias_side")
        context_support_level = self._get_feature_value(latest_features, "context_support_level")
        context_trigger_level = self._get_feature_value(latest_features, "context_trigger_level")
        context_support_distance_atr = self._get_feature_value(latest_features, "context_support_distance_atr")
        context_trigger_distance_atr = self._get_feature_value(latest_features, "context_trigger_distance_atr")
        context_session_range_width_atr = self._get_feature_value(
            latest_features,
            "context_session_range_width_atr",
        )
        setup_count_today = self._get_feature_value(latest_features, "context_setup_count_today")
        first_setup_of_day = self._get_feature_value(latest_features, "context_first_setup_of_day")
        momentum = self._get_feature_value(
            latest_features,
            f"momentum_{self.momentum_lookback_bars}",
        )
        if momentum is None:
            momentum = self._get_feature_value(latest_features, "momentum_20")
        candle_range_atr = self._get_feature_value(latest_features, "candle_range_atr")
        relative_volume = self._get_feature_value(latest_features, "relative_volume")

        long_pullback_entry = self._get_feature_value(latest_features, "context_long_pullback_entry")
        short_pullback_entry = self._get_feature_value(latest_features, "context_short_pullback_entry")
        long_reclaim_entry = self._get_feature_value(latest_features, "context_long_reclaim_entry")
        short_reclaim_entry = self._get_feature_value(latest_features, "context_short_reclaim_entry")
        long_session_trend_entry = self._get_feature_value(
            latest_features,
            "context_long_session_trend_entry",
        )
        short_session_trend_entry = self._get_feature_value(
            latest_features,
            "context_short_session_trend_entry",
        )

        required_values = {
            "atr_14": atr,
            "ema_200_1h": ema_200_1h,
            "ema_200_1h_slope": ema_200_1h_slope,
            "intraday_vwap": intraday_vwap,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "opening_range_high": opening_range_high,
            "opening_range_low": opening_range_low,
            "opening_range_mid": opening_range_mid,
            "opening_range_width": opening_range_width,
            "opening_range_width_atr": opening_range_width_atr,
            "opening_range_ready": opening_range_ready,
        }
        missing = [name for name, value in required_values.items() if value is None]
        if missing:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale=f"Missing or invalid contextual intraday features for signal generation: {missing}.",
                metadata=base_metadata,
            )

        if opening_range_ready <= 0.0:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Opening range is not complete yet, so the contextual intraday baseline stays flat.",
                metadata={
                    **base_metadata,
                    "opening_range_high": opening_range_high,
                    "opening_range_low": opening_range_low,
                    "opening_range_mid": opening_range_mid,
                    "opening_range_width": opening_range_width,
                    "opening_range_width_atr": opening_range_width_atr,
                },
            )

        side, trigger_name = self._resolve_entry_side(
            long_pullback_entry=long_pullback_entry,
            short_pullback_entry=short_pullback_entry,
            long_reclaim_entry=long_reclaim_entry,
            short_reclaim_entry=short_reclaim_entry,
            long_session_trend_entry=long_session_trend_entry,
            short_session_trend_entry=short_session_trend_entry,
        )
        if side is None:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="No contextual intraday setup is active on this bar.",
                metadata={
                    **base_metadata,
                    "opening_range_high": opening_range_high,
                    "opening_range_low": opening_range_low,
                    "opening_range_mid": opening_range_mid,
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

        context_setup_count = int(setup_count_today) if setup_count_today is not None else 0
        if self.max_breakouts_per_day > 0 and context_setup_count > self.max_breakouts_per_day:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="The contextual intraday baseline already used the maximum number of daily setup opportunities.",
                metadata={**base_metadata, "context_setup_count_today": context_setup_count},
            )

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
            "opening_range_mid": opening_range_mid,
            "opening_range_width": opening_range_width,
            "opening_range_width_atr": opening_range_width_atr,
            "ema_200_1h": ema_200_1h,
            "ema_200_1h_slope": ema_200_1h_slope,
            "intraday_vwap": intraday_vwap,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "context_bias_side": context_bias_side,
            "context_support_level": context_support_level,
            "context_trigger_level": context_trigger_level,
            "context_support_distance_atr": context_support_distance_atr,
            "context_trigger_distance_atr": context_trigger_distance_atr,
            "context_session_range_width_atr": context_session_range_width_atr,
            "momentum": momentum,
            "abs_momentum": abs(momentum) if momentum is not None else None,
            "candle_range_atr": candle_range_atr,
            "relative_volume": relative_volume,
            "breakout_level": context_trigger_level,
            "breakout_distance_atr": context_trigger_distance_atr,
            "breakout_distance": (
                (context_trigger_distance_atr * atr)
                if context_trigger_distance_atr is not None and atr is not None
                else None
            ),
            "context_setup_count_today": context_setup_count,
            "first_setup_of_day": bool(first_setup_of_day == 1.0),
            "entry_trigger": trigger_name,
            "target_to_cost_ratio": target_to_cost_ratio,
            "expected_move_bps": expected_move_bps,
        }
        quality_reason = self._quality_gate_failure(
            side=side,
            close_price=close_price,
            ema_200_1h=ema_200_1h,
            ema_200_1h_slope=ema_200_1h_slope,
            intraday_vwap=intraday_vwap,
            ema_20=ema_20,
            ema_50=ema_50,
            opening_range_mid=opening_range_mid,
            opening_range_width_atr=opening_range_width_atr,
            context_bias_side=context_bias_side,
            context_trigger_distance_atr=context_trigger_distance_atr,
            momentum=momentum,
            candle_range_atr=candle_range_atr,
            relative_volume=relative_volume,
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

        return self._entry_signal(
            symbol=context.symbol,
            timestamp=timestamp,
            side=side,
            entry_price=close_price,
            atr=atr,
            entry_reason=self._entry_reason(side=side, trigger_name=trigger_name),
            metadata={**base_metadata, **quality_metadata},
        )

    def _resolve_entry_side(
        self,
        *,
        long_pullback_entry: float | None,
        short_pullback_entry: float | None,
        long_reclaim_entry: float | None,
        short_reclaim_entry: float | None,
        long_session_trend_entry: float | None,
        short_session_trend_entry: float | None,
    ) -> tuple[SignalSide | None, str | None]:
        if self.entry_mode == "context_pullback_continuation":
            if long_pullback_entry == 1.0:
                return SignalSide.LONG, "context_pullback_continuation"
            if short_pullback_entry == 1.0:
                return SignalSide.SHORT, "context_pullback_continuation"
            return None, None

        if self.entry_mode == "vwap_reclaim_acceptance":
            if long_reclaim_entry == 1.0:
                return SignalSide.LONG, "vwap_reclaim_acceptance"
            if short_reclaim_entry == 1.0:
                return SignalSide.SHORT, "vwap_reclaim_acceptance"
            return None, None

        if self.entry_mode == "session_trend_continuation":
            if long_session_trend_entry == 1.0:
                return SignalSide.LONG, "session_trend_continuation"
            if short_session_trend_entry == 1.0:
                return SignalSide.SHORT, "session_trend_continuation"
            return None, None

        raise ValueError(f"Unsupported contextual intraday entry mode: {self.entry_mode}")

    def _quality_gate_failure(
        self,
        *,
        side: SignalSide,
        close_price: float,
        ema_200_1h: float,
        ema_200_1h_slope: float,
        intraday_vwap: float,
        ema_20: float,
        ema_50: float,
        opening_range_mid: float,
        opening_range_width_atr: float,
        context_bias_side: float | None,
        context_trigger_distance_atr: float | None,
        momentum: float | None,
        candle_range_atr: float | None,
        relative_volume: float | None,
        expected_move_bps: float,
        target_to_cost_ratio: float,
    ) -> str | None:
        if self.use_ema_200_1h_trend_filter:
            if side == SignalSide.LONG and close_price <= ema_200_1h:
                return "Long contextual intraday setup rejected because price is not above the 1H EMA200 trend filter."
            if side == SignalSide.SHORT and close_price >= ema_200_1h:
                return "Short contextual intraday setup rejected because price is not below the 1H EMA200 trend filter."

        if self.use_ema_200_1h_slope:
            slope_tolerance = abs(self.ema_200_1h_slope_tolerance)
            if side == SignalSide.LONG and ema_200_1h_slope <= -slope_tolerance:
                return "Long contextual intraday setup rejected because the 1H EMA200 slope is not positive."
            if side == SignalSide.SHORT and ema_200_1h_slope >= slope_tolerance:
                return "Short contextual intraday setup rejected because the 1H EMA200 slope is not negative."

        if self.use_intraday_ema50_alignment:
            if side == SignalSide.LONG and ema_20 < ema_50:
                return "Long contextual intraday setup rejected because the fast intraday trend is not aligned."
            if side == SignalSide.SHORT and ema_20 > ema_50:
                return "Short contextual intraday setup rejected because the fast intraday trend is not aligned."

        if self.use_intraday_vwap_filter:
            if side == SignalSide.LONG and close_price <= intraday_vwap:
                return "Long contextual intraday setup rejected because price is not accepted above VWAP."
            if side == SignalSide.SHORT and close_price >= intraday_vwap:
                return "Short contextual intraday setup rejected because price is not accepted below VWAP."

        if self.use_intraday_ema20_filter:
            if side == SignalSide.LONG and close_price <= ema_20:
                return "Long contextual intraday setup rejected because price is not above the intraday EMA20."
            if side == SignalSide.SHORT and close_price >= ema_20:
                return "Short contextual intraday setup rejected because price is not below the intraday EMA20."

        if self.use_opening_range_mid_filter:
            if side == SignalSide.LONG and close_price <= opening_range_mid:
                return "Long contextual intraday setup rejected because price is not above the opening-range midpoint."
            if side == SignalSide.SHORT and close_price >= opening_range_mid:
                return "Short contextual intraday setup rejected because price is not below the opening-range midpoint."

        if context_bias_side is not None:
            if side == SignalSide.LONG and context_bias_side < 0.0:
                return "Long contextual intraday setup rejected because the intraday session bias is still short."
            if side == SignalSide.SHORT and context_bias_side > 0.0:
                return "Short contextual intraday setup rejected because the intraday session bias is still long."

        if (
            self.minimum_opening_range_width_atr > 0.0
            and opening_range_width_atr < self.minimum_opening_range_width_atr
        ):
            return "Contextual intraday setup rejected because the opening range is too narrow relative to ATR."

        if (
            self.maximum_opening_range_width_atr > 0.0
            and opening_range_width_atr > self.maximum_opening_range_width_atr
        ):
            return "Contextual intraday setup rejected because the opening range is too wide relative to ATR."

        if momentum is None:
            return "Contextual intraday setup rejected because momentum is missing."
        if side == SignalSide.LONG and momentum <= 0.0:
            return "Long contextual intraday setup rejected because momentum is not aligned upward."
        if side == SignalSide.SHORT and momentum >= 0.0:
            return "Short contextual intraday setup rejected because momentum is not aligned downward."
        if self.minimum_momentum_abs > 0.0 and abs(momentum) < self.minimum_momentum_abs:
            return "Contextual intraday setup rejected because momentum is too weak."

        if candle_range_atr is None:
            return "Contextual intraday setup rejected because candle-range expansion is missing."
        if self.minimum_candle_range_atr > 0.0 and candle_range_atr < self.minimum_candle_range_atr:
            return "Contextual intraday setup rejected because the trigger candle lacks volatility expansion."

        if relative_volume is not None and self.minimum_relative_volume > 0.0 and relative_volume < self.minimum_relative_volume:
            return "Contextual intraday setup rejected because relative volume is too weak."

        if (
            context_trigger_distance_atr is not None
            and self.max_breakout_distance_atr > 0.0
            and context_trigger_distance_atr > self.max_breakout_distance_atr
        ):
            return "Contextual intraday setup rejected because the entry is too extended away from its intraday reference level."

        if self.minimum_expected_move_bps > 0.0 and expected_move_bps < self.minimum_expected_move_bps:
            return "Contextual intraday setup rejected because the projected move is too small in basis points."

        if self.minimum_target_to_cost_ratio > 0.0 and target_to_cost_ratio < self.minimum_target_to_cost_ratio:
            return "Contextual intraday setup rejected because the target-to-cost ratio is too weak."

        return None

    def _entry_reason(self, *, side: SignalSide, trigger_name: str) -> str:
        side_label = "Long" if side == SignalSide.LONG else "Short"
        if trigger_name == "context_pullback_continuation":
            return (
                f"{side_label} contextual intraday pullback: price pulled back into intraday support "
                "and resumed in the session trend."
            )
        if trigger_name == "vwap_reclaim_acceptance":
            return (
                f"{side_label} contextual intraday reclaim: price reclaimed the VWAP / opening-range context "
                "and showed renewed acceptance."
            )
        return (
            f"{side_label} contextual intraday continuation: price broke a local session structure in the "
            "direction of the dominant intraday context."
        )
