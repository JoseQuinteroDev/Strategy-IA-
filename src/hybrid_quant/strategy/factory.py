from __future__ import annotations

from hybrid_quant.core import Settings

from .base import Strategy
from .mean_reversion import MeanReversionStrategy
from .trend_breakout import TrendBreakoutStrategy


def build_strategy(settings: Settings, *, estimated_round_trip_cost_bps: float) -> Strategy:
    family = settings.strategy.family
    if family == "mean_reversion":
        return MeanReversionStrategy(
            name=settings.strategy.name,
            variant_name=settings.strategy.variant_name,
            entry_zscore=settings.strategy.entry_zscore,
            exit_zscore=settings.strategy.exit_zscore,
            trend_filter=settings.strategy.trend_filter,
            regime_filter=settings.strategy.regime_filter,
            execution_timeframe=settings.market.execution_timeframe,
            filter_timeframe=settings.market.filter_timeframe,
            mean_reversion_anchor=settings.strategy.mean_reversion_anchor,
            adx_threshold=settings.strategy.adx_threshold,
            atr_multiple_stop=settings.strategy.atr_multiple_stop,
            atr_multiple_target=settings.strategy.atr_multiple_target,
            time_stop_bars=settings.strategy.time_stop_bars,
            session_close_hour_utc=settings.strategy.session_close_hour_utc,
            session_close_minute_utc=settings.strategy.session_close_minute_utc,
            no_entry_minutes_before_close=settings.strategy.no_entry_minutes_before_close,
            blocked_hours_utc=settings.strategy.blocked_hours_utc,
            allowed_hours_utc=settings.strategy.allowed_hours_utc,
            allowed_weekdays=settings.strategy.allowed_weekdays,
            exclude_weekends=settings.strategy.exclude_weekends,
            minimum_anchor_distance_atr=settings.strategy.minimum_anchor_distance_atr,
            minimum_expected_move_bps=settings.strategy.minimum_expected_move_bps,
            minimum_target_to_cost_ratio=settings.strategy.minimum_target_to_cost_ratio,
            estimated_round_trip_cost_bps=estimated_round_trip_cost_bps,
        )

    if family == "trend_breakout":
        return TrendBreakoutStrategy(
            name=settings.strategy.name,
            variant_name=settings.strategy.variant_name,
            trend_filter=settings.strategy.trend_filter,
            regime_filter=settings.strategy.regime_filter,
            execution_timeframe=settings.market.execution_timeframe,
            filter_timeframe=settings.market.filter_timeframe,
            adx_threshold=settings.strategy.adx_threshold,
            atr_multiple_stop=settings.strategy.atr_multiple_stop,
            atr_multiple_target=settings.strategy.atr_multiple_target,
            time_stop_bars=settings.strategy.time_stop_bars,
            session_close_hour_utc=settings.strategy.session_close_hour_utc,
            session_close_minute_utc=settings.strategy.session_close_minute_utc,
            no_entry_minutes_before_close=settings.strategy.no_entry_minutes_before_close,
            blocked_hours_utc=settings.strategy.blocked_hours_utc,
            allowed_hours_utc=settings.strategy.allowed_hours_utc,
            allowed_weekdays=settings.strategy.allowed_weekdays,
            exclude_weekends=settings.strategy.exclude_weekends,
            minimum_expected_move_bps=settings.strategy.minimum_expected_move_bps,
            minimum_target_to_cost_ratio=settings.strategy.minimum_target_to_cost_ratio,
            estimated_round_trip_cost_bps=estimated_round_trip_cost_bps,
            breakout_lookback_bars=settings.strategy.breakout_lookback_bars,
            breakout_buffer_atr=settings.strategy.breakout_buffer_atr,
            minimum_breakout_range_atr=settings.strategy.minimum_breakout_range_atr,
            momentum_lookback_bars=settings.strategy.momentum_lookback_bars,
            minimum_momentum_abs=settings.strategy.minimum_momentum_abs,
            minimum_candle_range_atr=settings.strategy.minimum_candle_range_atr,
        )

    raise ValueError(f"Unsupported strategy family: {family}")
