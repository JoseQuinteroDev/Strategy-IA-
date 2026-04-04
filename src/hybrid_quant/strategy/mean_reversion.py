from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from hybrid_quant.core import SignalSide, StrategyContext, StrategySignal

from .base import IntradayStrategySupport, Strategy


@dataclass(slots=True)
class MeanReversionStrategy(Strategy, IntradayStrategySupport):
    name: str
    variant_name: str
    entry_zscore: float
    exit_zscore: float | None
    trend_filter: str
    regime_filter: str
    execution_timeframe: str
    filter_timeframe: str
    mean_reversion_anchor: str = "vwap"
    adx_threshold: float = 25.0
    atr_multiple_stop: float = 1.0
    atr_multiple_target: float = 1.0
    time_stop_bars: int = 12
    session_close_hour_utc: int = 23
    session_close_minute_utc: int = 55
    no_entry_minutes_before_close: int = 30
    blocked_hours_utc: list[int] | None = None
    allowed_hours_utc: list[int] | None = None
    allowed_weekdays: list[int] | None = None
    exclude_weekends: bool = False
    minimum_anchor_distance_atr: float = 0.0
    minimum_expected_move_bps: float = 0.0
    minimum_target_to_cost_ratio: float = 0.0
    estimated_round_trip_cost_bps: float = 0.0

    def generate(self, context: StrategyContext) -> StrategySignal:
        timestamp = self._resolve_timestamp(context)
        base_metadata = {
            "strategy": self.name,
            "variant_name": self.variant_name,
            "strategy_family": "mean_reversion",
            "execution_timeframe": self.execution_timeframe,
            "filter_timeframe": self.filter_timeframe,
            "trend_filter": self.trend_filter,
            "regime_filter": self.regime_filter,
            "mean_reversion_anchor": self.mean_reversion_anchor,
            "time_stop_bars": self.time_stop_bars,
            "close_on_session_end": True,
            "regime": context.regime,
            "blocked_hours_utc": list(self.blocked_hours_utc or []),
            "allowed_hours_utc": list(self.allowed_hours_utc or []),
            "allowed_weekdays": list(self.allowed_weekdays or []),
            "exclude_weekends": self.exclude_weekends,
            "minimum_anchor_distance_atr": self.minimum_anchor_distance_atr,
            "minimum_expected_move_bps": self.minimum_expected_move_bps,
            "minimum_target_to_cost_ratio": self.minimum_target_to_cost_ratio,
            "estimated_round_trip_cost_bps": self.estimated_round_trip_cost_bps,
        }

        if not context.bars or not context.features:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Insufficient market context to evaluate the intraday strategy.",
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
                rationale="No new entries inside the session-close buffer.",
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
        zscore = self._get_feature_value(latest_features, "zscore_distance_to_mean")
        mean_anchor_name = self._resolve_anchor_name(latest_features)
        mean_anchor_value = self._get_feature_value(latest_features, mean_anchor_name)

        required_values = {
            "ema_200_1h": ema_200_1h,
            "adx_1h": adx_1h,
            "atr_14": atr,
            "zscore_distance_to_mean": zscore,
            mean_anchor_name: mean_anchor_value,
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

        if adx_1h > self.adx_threshold:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale=(
                    f"ADX regime filter blocked the trade: {adx_1h:.2f} is above the "
                    f"mean-reversion threshold {self.adx_threshold:.2f}."
                ),
                metadata={**base_metadata, "adx_1h": adx_1h},
            )

        if trend_bias == SignalSide.LONG and close_price < mean_anchor_value and zscore <= -self.entry_zscore:
            reason = (
                f"Long mean reversion: price is above 1H EMA200, ADX is contained, and "
                f"close is stretched below {mean_anchor_name}."
            )
            quality_reason, quality_metadata = self._setup_quality_failure(
                side=SignalSide.LONG,
                close_price=close_price,
                atr=atr,
                zscore=zscore,
                mean_anchor_name=mean_anchor_name,
                mean_anchor_value=mean_anchor_value,
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
                        "anchor_name": mean_anchor_name,
                        "anchor_value": mean_anchor_value,
                        "zscore_distance_to_mean": zscore,
                        **quality_metadata,
                    },
                )
            return self._entry_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                side=SignalSide.LONG,
                entry_price=close_price,
                atr=atr,
                entry_reason=reason,
                metadata={
                    **base_metadata,
                    "ema_200_1h": ema_200_1h,
                    "adx_1h": adx_1h,
                    "anchor_name": mean_anchor_name,
                    "anchor_value": mean_anchor_value,
                    "zscore_distance_to_mean": zscore,
                    **quality_metadata,
                },
            )

        if trend_bias == SignalSide.SHORT and close_price > mean_anchor_value and zscore >= self.entry_zscore:
            reason = (
                f"Short mean reversion: price is below 1H EMA200, ADX is contained, and "
                f"close is stretched above {mean_anchor_name}."
            )
            quality_reason, quality_metadata = self._setup_quality_failure(
                side=SignalSide.SHORT,
                close_price=close_price,
                atr=atr,
                zscore=zscore,
                mean_anchor_name=mean_anchor_name,
                mean_anchor_value=mean_anchor_value,
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
                        "anchor_name": mean_anchor_name,
                        "anchor_value": mean_anchor_value,
                        "zscore_distance_to_mean": zscore,
                        **quality_metadata,
                    },
                )
            return self._entry_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                side=SignalSide.SHORT,
                entry_price=close_price,
                atr=atr,
                entry_reason=reason,
                metadata={
                    **base_metadata,
                    "ema_200_1h": ema_200_1h,
                    "adx_1h": adx_1h,
                    "anchor_name": mean_anchor_name,
                    "anchor_value": mean_anchor_value,
                    "zscore_distance_to_mean": zscore,
                    **quality_metadata,
                },
            )

        return self._flat_signal(
            symbol=context.symbol,
            timestamp=timestamp,
            rationale="Setup does not meet trend, regime, and stretch requirements simultaneously.",
            metadata={
                **base_metadata,
                "ema_200_1h": ema_200_1h,
                "adx_1h": adx_1h,
                "anchor_name": mean_anchor_name,
                "anchor_value": mean_anchor_value,
                "zscore_distance_to_mean": zscore,
            },
        )

    def _resolve_anchor_name(self, features: dict[str, float]) -> str:
        if self.mean_reversion_anchor == "ema50":
            return "ema_50"
        if self.mean_reversion_anchor == "vwap":
            return "intraday_vwap"
        if "intraday_vwap" in features and self._get_feature_value(features, "intraday_vwap") is not None:
            return "intraday_vwap"
        return "ema_50"

    def _setup_quality_failure(
        self,
        *,
        side: SignalSide,
        close_price: float,
        atr: float,
        zscore: float,
        mean_anchor_name: str,
        mean_anchor_value: float,
    ) -> tuple[str | None, dict[str, Any]]:
        anchor_distance = abs(close_price - mean_anchor_value)
        anchor_distance_atr = anchor_distance / atr if atr > 0.0 else float("nan")
        expected_move = atr * self.atr_multiple_target
        expected_move_bps = (expected_move / close_price) * 10000.0 if close_price > 0.0 else float("nan")
        target_to_cost_ratio = (
            expected_move_bps / self.estimated_round_trip_cost_bps
            if self.estimated_round_trip_cost_bps > 0.0 and math.isfinite(expected_move_bps)
            else float("inf")
        )
        metadata = {
            "entry_side": side.value,
            "anchor_distance": anchor_distance,
            "anchor_distance_atr": anchor_distance_atr,
            "expected_move_bps": expected_move_bps,
            "target_to_cost_ratio": target_to_cost_ratio,
            "abs_entry_zscore": abs(zscore),
            "anchor_name": mean_anchor_name,
        }

        if self.minimum_anchor_distance_atr > 0.0 and (
            not math.isfinite(anchor_distance_atr) or anchor_distance_atr < self.minimum_anchor_distance_atr
        ):
            return (
                "Setup rejected because the stretch versus the mean anchor is too small relative to ATR.",
                metadata,
            )

        if self.minimum_expected_move_bps > 0.0 and (
            not math.isfinite(expected_move_bps) or expected_move_bps < self.minimum_expected_move_bps
        ):
            return (
                "Setup rejected because the projected move is too small in basis points.",
                metadata,
            )

        if self.minimum_target_to_cost_ratio > 0.0 and (
            not math.isfinite(target_to_cost_ratio) or target_to_cost_ratio < self.minimum_target_to_cost_ratio
        ):
            return (
                "Setup rejected because the target-to-cost ratio is too weak for a cost-aware baseline.",
                metadata,
            )

        return None, metadata
