from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
import math

from hybrid_quant.core import SignalSide, StrategyContext, StrategySignal


class Strategy(ABC):
    @abstractmethod
    def generate(self, context: StrategyContext) -> StrategySignal:
        """Produce a signal from market context."""


@dataclass(slots=True)
class MeanReversionStrategy(Strategy):
    name: str
    entry_zscore: float
    exit_zscore: float
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

    def generate(self, context: StrategyContext) -> StrategySignal:
        timestamp = self._resolve_timestamp(context)
        base_metadata = {
            "strategy": self.name,
            "execution_timeframe": self.execution_timeframe,
            "filter_timeframe": self.filter_timeframe,
            "trend_filter": self.trend_filter,
            "regime_filter": self.regime_filter,
            "mean_reversion_anchor": self.mean_reversion_anchor,
            "time_stop_bars": self.time_stop_bars,
            "close_on_session_end": True,
            "regime": context.regime,
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
                },
            )

        if trend_bias == SignalSide.SHORT and close_price > mean_anchor_value and zscore >= self.entry_zscore:
            reason = (
                f"Short mean reversion: price is below 1H EMA200, ADX is contained, and "
                f"close is stretched above {mean_anchor_name}."
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

    def _resolve_timestamp(self, context: StrategyContext) -> datetime:
        if context.features:
            return context.features[-1].timestamp
        if context.bars:
            return context.bars[-1].timestamp
        return datetime.now(UTC)

    def _resolve_anchor_name(self, features: dict[str, float]) -> str:
        if self.mean_reversion_anchor == "ema50":
            return "ema_50"
        if self.mean_reversion_anchor == "vwap":
            return "intraday_vwap"
        if "intraday_vwap" in features and self._get_feature_value(features, "intraday_vwap") is not None:
            return "intraday_vwap"
        return "ema_50"

    def _get_feature_value(self, values: dict[str, float], name: str) -> float | None:
        raw_value = values.get(name)
        if raw_value is None:
            return None
        if not math.isfinite(float(raw_value)):
            return None
        return float(raw_value)

    def _trend_bias(self, close_price: float, ema_200_1h: float) -> SignalSide | None:
        if close_price > ema_200_1h:
            return SignalSide.LONG
        if close_price < ema_200_1h:
            return SignalSide.SHORT
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
        metadata: dict[str, float | int | str | bool],
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
            close_on_session_end=True,
            entry_reason=entry_reason,
            metadata=metadata,
        )

    def _flat_signal(
        self,
        *,
        symbol: str,
        timestamp: datetime,
        rationale: str,
        metadata: dict[str, float | int | str | bool],
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
            close_on_session_end=True,
            entry_reason=None,
            metadata=metadata,
        )

    def _minutes_to_session_close(self, timestamp: datetime) -> int:
        normalized = timestamp.astimezone(UTC)
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
        normalized = timestamp.astimezone(UTC)
        session_close = time(self.session_close_hour_utc, self.session_close_minute_utc)
        return normalized.time() >= session_close
