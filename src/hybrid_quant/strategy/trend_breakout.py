from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from hybrid_quant.core import SignalSide, StrategyContext, StrategySignal

from .base import IntradayStrategySupport, Strategy


@dataclass(slots=True)
class TrendBreakoutStrategy(Strategy, IntradayStrategySupport):
    name: str
    variant_name: str
    trend_filter: str
    regime_filter: str
    execution_timeframe: str
    filter_timeframe: str
    adx_threshold: float = 20.0
    atr_multiple_stop: float = 1.0
    atr_multiple_target: float = 2.0
    time_stop_bars: int = 18
    session_close_hour_utc: int = 20
    session_close_minute_utc: int = 55
    no_entry_minutes_before_close: int = 20
    blocked_hours_utc: list[int] | None = None
    allowed_hours_utc: list[int] | None = None
    allowed_weekdays: list[int] | None = None
    exclude_weekends: bool = True
    minimum_expected_move_bps: float = 0.0
    minimum_target_to_cost_ratio: float = 0.0
    estimated_round_trip_cost_bps: float = 0.0
    breakout_lookback_bars: int = 20
    breakout_buffer_atr: float = 0.0
    minimum_breakout_range_atr: float = 0.0
    momentum_lookback_bars: int = 20
    minimum_momentum_abs: float = 0.0
    minimum_candle_range_atr: float = 0.0

    def generate(self, context: StrategyContext) -> StrategySignal:
        timestamp = self._resolve_timestamp(context)
        base_metadata = {
            "strategy": self.name,
            "variant_name": self.variant_name,
            "strategy_family": "trend_breakout",
            "execution_timeframe": self.execution_timeframe,
            "filter_timeframe": self.filter_timeframe,
            "trend_filter": self.trend_filter,
            "regime_filter": self.regime_filter,
            "time_stop_bars": self.time_stop_bars,
            "close_on_session_end": True,
            "regime": context.regime,
            "breakout_lookback_bars": self.breakout_lookback_bars,
            "breakout_buffer_atr": self.breakout_buffer_atr,
            "minimum_breakout_range_atr": self.minimum_breakout_range_atr,
            "momentum_lookback_bars": self.momentum_lookback_bars,
            "minimum_momentum_abs": self.minimum_momentum_abs,
            "minimum_candle_range_atr": self.minimum_candle_range_atr,
            "minimum_expected_move_bps": self.minimum_expected_move_bps,
            "minimum_target_to_cost_ratio": self.minimum_target_to_cost_ratio,
            "estimated_round_trip_cost_bps": self.estimated_round_trip_cost_bps,
            "blocked_hours_utc": list(self.blocked_hours_utc or []),
            "allowed_hours_utc": list(self.allowed_hours_utc or []),
            "allowed_weekdays": list(self.allowed_weekdays or []),
            "exclude_weekends": self.exclude_weekends,
        }

        if not context.bars or not context.features:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Insufficient market context to evaluate the trend breakout strategy.",
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
                rationale="No new breakouts are allowed inside the session-close buffer.",
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

        ema_200_1h = self._get_feature_value(latest_features, "ema_200_1h")
        adx_1h = self._get_feature_value(latest_features, "adx_1h")
        atr = self._get_feature_value(latest_features, "atr_14")
        breakout_high = self._get_feature_value(latest_features, f"breakout_high_{self.breakout_lookback_bars}")
        breakout_low = self._get_feature_value(latest_features, f"breakout_low_{self.breakout_lookback_bars}")
        breakout_range_width_atr = self._get_feature_value(
            latest_features,
            f"breakout_range_width_atr_{self.breakout_lookback_bars}",
        )
        momentum = self._get_feature_value(latest_features, f"momentum_{self.momentum_lookback_bars}")
        candle_range_atr = self._get_feature_value(latest_features, "candle_range_atr")

        required_values = {
            "ema_200_1h": ema_200_1h,
            "adx_1h": adx_1h,
            "atr_14": atr,
            f"breakout_high_{self.breakout_lookback_bars}": breakout_high,
            f"breakout_low_{self.breakout_lookback_bars}": breakout_low,
            f"breakout_range_width_atr_{self.breakout_lookback_bars}": breakout_range_width_atr,
            f"momentum_{self.momentum_lookback_bars}": momentum,
            "candle_range_atr": candle_range_atr,
        }
        missing = [name for name, value in required_values.items() if value is None]
        if missing:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale=f"Missing or invalid features for signal generation: {missing}.",
                metadata=base_metadata,
            )

        close_price = latest_bar.close
        trend_bias = self._trend_bias(close_price, ema_200_1h)
        if trend_bias is None:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Trend filter is neutral because price is sitting on the 1H EMA200.",
                metadata={**base_metadata, "ema_200_1h": ema_200_1h},
            )

        if adx_1h < self.adx_threshold:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale=(
                    f"Trend regime filter blocked the trade: ADX {adx_1h:.2f} is below "
                    f"the breakout threshold {self.adx_threshold:.2f}."
                ),
                metadata={**base_metadata, "ema_200_1h": ema_200_1h, "adx_1h": adx_1h},
            )

        if trend_bias == SignalSide.LONG:
            breakout_level = breakout_high
            breakout_trigger = breakout_level + (atr * self.breakout_buffer_atr)
            if close_price < breakout_trigger:
                return self._flat_signal(
                    symbol=context.symbol,
                    timestamp=timestamp,
                    rationale="Long breakout setup is not confirmed above the prior range high.",
                    metadata={
                        **base_metadata,
                        "ema_200_1h": ema_200_1h,
                        "adx_1h": adx_1h,
                        "breakout_level": breakout_level,
                        "breakout_trigger": breakout_trigger,
                    },
                )

            quality_reason, quality_metadata = self._quality_gate_failure(
                side=SignalSide.LONG,
                close_price=close_price,
                atr=atr,
                breakout_level=breakout_level,
                breakout_range_width_atr=breakout_range_width_atr,
                momentum=momentum,
                candle_range_atr=candle_range_atr,
            )
            if quality_reason is not None:
                return self._flat_signal(
                    symbol=context.symbol,
                    timestamp=timestamp,
                    rationale=quality_reason,
                    metadata={
                        **base_metadata,
                        "ema_200_1h": ema_200_1h,
                        "adx_1h": adx_1h,
                        "breakout_level": breakout_level,
                        "breakout_trigger": breakout_trigger,
                        **quality_metadata,
                    },
                )

            entry_reason = (
                f"Long breakout: price is above 1H EMA200, ADX confirms trend, and close cleared the prior "
                f"{self.breakout_lookback_bars}-bar range high."
            )
            return self._entry_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                side=SignalSide.LONG,
                entry_price=close_price,
                atr=atr,
                entry_reason=entry_reason,
                metadata={
                    **base_metadata,
                    "ema_200_1h": ema_200_1h,
                    "adx_1h": adx_1h,
                    "breakout_level": breakout_level,
                    "breakout_trigger": breakout_trigger,
                    **quality_metadata,
                },
            )

        breakout_level = breakout_low
        breakout_trigger = breakout_level - (atr * self.breakout_buffer_atr)
        if close_price > breakout_trigger:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Short breakout setup is not confirmed below the prior range low.",
                metadata={
                    **base_metadata,
                    "ema_200_1h": ema_200_1h,
                    "adx_1h": adx_1h,
                    "breakout_level": breakout_level,
                    "breakout_trigger": breakout_trigger,
                },
            )

        quality_reason, quality_metadata = self._quality_gate_failure(
            side=SignalSide.SHORT,
            close_price=close_price,
            atr=atr,
            breakout_level=breakout_level,
            breakout_range_width_atr=breakout_range_width_atr,
            momentum=momentum,
            candle_range_atr=candle_range_atr,
        )
        if quality_reason is not None:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale=quality_reason,
                metadata={
                    **base_metadata,
                    "ema_200_1h": ema_200_1h,
                    "adx_1h": adx_1h,
                    "breakout_level": breakout_level,
                    "breakout_trigger": breakout_trigger,
                    **quality_metadata,
                },
            )

        entry_reason = (
            f"Short breakout: price is below 1H EMA200, ADX confirms trend, and close cleared the prior "
            f"{self.breakout_lookback_bars}-bar range low."
        )
        return self._entry_signal(
            symbol=context.symbol,
            timestamp=timestamp,
            side=SignalSide.SHORT,
            entry_price=close_price,
            atr=atr,
            entry_reason=entry_reason,
            metadata={
                **base_metadata,
                "ema_200_1h": ema_200_1h,
                "adx_1h": adx_1h,
                "breakout_level": breakout_level,
                "breakout_trigger": breakout_trigger,
                **quality_metadata,
            },
        )

    def _quality_gate_failure(
        self,
        *,
        side: SignalSide,
        close_price: float,
        atr: float,
        breakout_level: float,
        breakout_range_width_atr: float,
        momentum: float,
        candle_range_atr: float,
    ) -> tuple[str | None, dict[str, Any]]:
        breakout_distance = (
            close_price - breakout_level if side == SignalSide.LONG else breakout_level - close_price
        )
        breakout_distance_atr = breakout_distance / atr if atr > 0.0 else float("nan")
        expected_move = atr * self.atr_multiple_target
        expected_move_bps = (expected_move / close_price) * 10000.0 if close_price > 0.0 else float("nan")
        target_to_cost_ratio = (
            expected_move_bps / self.estimated_round_trip_cost_bps
            if self.estimated_round_trip_cost_bps > 0.0 and math.isfinite(expected_move_bps)
            else float("inf")
        )
        metadata = {
            "entry_side": side.value,
            "breakout_level": breakout_level,
            "breakout_distance": breakout_distance,
            "breakout_distance_atr": breakout_distance_atr,
            "breakout_range_width_atr": breakout_range_width_atr,
            "momentum": momentum,
            "abs_momentum": abs(momentum),
            "candle_range_atr": candle_range_atr,
            "expected_move_bps": expected_move_bps,
            "target_to_cost_ratio": target_to_cost_ratio,
        }

        if side == SignalSide.LONG and momentum <= 0.0:
            return "Setup rejected because momentum is not aligned with the long breakout.", metadata
        if side == SignalSide.SHORT and momentum >= 0.0:
            return "Setup rejected because momentum is not aligned with the short breakout.", metadata

        if self.minimum_momentum_abs > 0.0 and abs(momentum) < self.minimum_momentum_abs:
            return "Setup rejected because the momentum filter is too weak for a trend breakout.", metadata

        if self.minimum_breakout_range_atr > 0.0 and breakout_range_width_atr < self.minimum_breakout_range_atr:
            return "Setup rejected because the prior range is too narrow relative to ATR.", metadata

        if self.minimum_candle_range_atr > 0.0 and candle_range_atr < self.minimum_candle_range_atr:
            return "Setup rejected because the breakout candle lacks volatility expansion.", metadata

        if self.minimum_expected_move_bps > 0.0 and (
            not math.isfinite(expected_move_bps) or expected_move_bps < self.minimum_expected_move_bps
        ):
            return "Setup rejected because the projected move is too small in basis points.", metadata

        if self.minimum_target_to_cost_ratio > 0.0 and (
            not math.isfinite(target_to_cost_ratio) or target_to_cost_ratio < self.minimum_target_to_cost_ratio
        ):
            return (
                "Setup rejected because the target-to-cost ratio is too weak for a cost-aware breakout baseline.",
                metadata,
            )

        return None, metadata
