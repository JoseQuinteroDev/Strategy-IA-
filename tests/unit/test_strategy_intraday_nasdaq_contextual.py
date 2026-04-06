from __future__ import annotations

import unittest
from datetime import UTC, datetime

from hybrid_quant.core import FeatureSnapshot, MarketBar, SignalSide, StrategyContext
from hybrid_quant.strategy.intraday_nasdaq_contextual import IntradayNasdaqContextualStrategy


def _bar(timestamp: datetime, *, close: float = 106.0) -> MarketBar:
    return MarketBar(
        timestamp=timestamp,
        open=close - 0.4,
        high=close + 0.8,
        low=close - 0.8,
        close=close,
        volume=180.0,
    )


def _feature_snapshot(timestamp: datetime, **overrides: float) -> FeatureSnapshot:
    values = {
        "atr_14": 2.0,
        "ema_200_1h": 100.0,
        "ema_200_1h_slope": 0.5,
        "intraday_vwap": 104.0,
        "ema_20": 104.8,
        "ema_50": 104.1,
        "opening_range_high": 105.0,
        "opening_range_low": 99.0,
        "opening_range_mid": 102.0,
        "opening_range_width": 6.0,
        "opening_range_width_atr": 1.5,
        "opening_range_ready": 1.0,
        "context_bias_side": 1.0,
        "context_long_pullback_entry": 0.0,
        "context_short_pullback_entry": 0.0,
        "context_long_reclaim_entry": 0.0,
        "context_short_reclaim_entry": 0.0,
        "context_long_session_trend_entry": 0.0,
        "context_short_session_trend_entry": 0.0,
        "context_support_level": 104.8,
        "context_trigger_level": 104.8,
        "context_support_distance_atr": 0.35,
        "context_trigger_distance_atr": 0.35,
        "context_session_range_width_atr": 2.4,
        "context_setup_count_today": 1.0,
        "context_first_setup_of_day": 1.0,
        "momentum_20": 0.008,
        "candle_range_atr": 0.9,
        "relative_volume": 1.2,
    }
    values.update(overrides)
    return FeatureSnapshot(timestamp=timestamp, values=values, metadata={})


def _strategy(**overrides: object) -> IntradayNasdaqContextualStrategy:
    params = {
        "name": "intraday_nasdaq_contextual_nq",
        "variant_name": "baseline_nq_intraday_contextual",
        "trend_filter": "ema_200_1h",
        "execution_timeframe": "5m",
        "filter_timeframe": "1H",
        "entry_mode": "context_pullback_continuation",
        "opening_range_minutes": 30,
        "retest_max_bars": 6,
        "atr_multiple_stop": 1.0,
        "atr_multiple_target": 1.8,
        "time_stop_bars": 14,
        "session_close_hour_utc": 20,
        "session_close_minute_utc": 55,
        "no_entry_minutes_before_close": 15,
        "allowed_weekdays": [0, 1, 2, 3, 4],
        "exclude_weekends": True,
        "minimum_expected_move_bps": 6.0,
        "minimum_target_to_cost_ratio": 1.2,
        "estimated_round_trip_cost_bps": 1.0,
        "momentum_lookback_bars": 20,
        "minimum_momentum_abs": 0.0010,
        "minimum_candle_range_atr": 0.30,
        "use_ema_200_1h_trend_filter": True,
        "minimum_opening_range_width_atr": 0.20,
        "maximum_opening_range_width_atr": 3.50,
        "minimum_relative_volume": 0.70,
        "max_breakout_distance_atr": 0.80,
        "max_breakouts_per_day": 4,
        "use_intraday_vwap_filter": True,
        "use_intraday_ema20_filter": True,
        "use_intraday_ema50_alignment": True,
        "use_opening_range_mid_filter": True,
    }
    params.update(overrides)
    return IntradayNasdaqContextualStrategy(**params)


class IntradayNasdaqContextualStrategyTests(unittest.TestCase):
    def test_generates_long_context_pullback_entry(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 10, tzinfo=UTC)
        signal = _strategy().generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=105.6)],
                features=[_feature_snapshot(timestamp, context_long_pullback_entry=1.0)],
            )
        )

        self.assertEqual(signal.side, SignalSide.LONG)
        self.assertEqual(signal.metadata["entry_trigger"], "context_pullback_continuation")

    def test_generates_short_reclaim_entry(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 20, tzinfo=UTC)
        signal = _strategy(entry_mode="vwap_reclaim_acceptance").generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=97.5)],
                features=[
                    _feature_snapshot(
                        timestamp,
                        ema_200_1h=100.0,
                        ema_200_1h_slope=-0.4,
                        intraday_vwap=98.6,
                        ema_20=98.2,
                        ema_50=98.7,
                        opening_range_high=106.0,
                        opening_range_low=99.0,
                        opening_range_mid=102.5,
                        context_bias_side=-1.0,
                        context_short_reclaim_entry=1.0,
                        context_trigger_level=98.6,
                        context_trigger_distance_atr=0.35,
                        momentum_20=-0.007,
                    )
                ],
            )
        )

        self.assertEqual(signal.side, SignalSide.SHORT)
        self.assertEqual(signal.metadata["entry_trigger"], "vwap_reclaim_acceptance")

    def test_blocks_long_when_fast_intraday_trend_is_not_aligned(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 25, tzinfo=UTC)
        signal = _strategy().generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=105.0)],
                features=[
                    _feature_snapshot(
                        timestamp,
                        context_long_pullback_entry=1.0,
                        ema_20=103.8,
                        ema_50=104.4,
                    )
                ],
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)
        self.assertIn("fast intraday trend", signal.rationale)

    def test_blocks_entry_when_too_extended_from_reference(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
        signal = _strategy(
            entry_mode="session_trend_continuation",
            max_breakout_distance_atr=0.25,
        ).generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=106.2)],
                features=[
                    _feature_snapshot(
                        timestamp,
                        context_long_session_trend_entry=1.0,
                        context_trigger_distance_atr=0.50,
                    )
                ],
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)
        self.assertIn("too extended", signal.rationale)

    def test_disabling_slope_filter_allows_entry_with_negative_slope(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 35, tzinfo=UTC)
        signal = _strategy(
            entry_mode="session_trend_continuation",
            use_ema_200_1h_slope=False,
        ).generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=106.2)],
                features=[
                    _feature_snapshot(
                        timestamp,
                        ema_200_1h_slope=-2.0,
                        context_long_session_trend_entry=1.0,
                    )
                ],
            )
        )

        self.assertEqual(signal.side, SignalSide.LONG)

    def test_disabling_opening_range_mid_filter_allows_setup_below_mid(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 40, tzinfo=UTC)
        signal = _strategy(
            entry_mode="session_trend_continuation",
            use_opening_range_mid_filter=False,
        ).generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=101.5)],
                features=[
                    _feature_snapshot(
                        timestamp,
                        opening_range_mid=102.0,
                        intraday_vwap=100.8,
                        ema_20=101.1,
                        ema_50=100.9,
                        context_long_session_trend_entry=1.0,
                        context_bias_side=1.0,
                        context_trigger_distance_atr=0.25,
                        momentum_20=0.006,
                    )
                ],
            )
        )

        self.assertEqual(signal.side, SignalSide.LONG)
