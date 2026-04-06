from __future__ import annotations

import unittest
from datetime import UTC, datetime

from hybrid_quant.core import FeatureSnapshot, MarketBar, SignalSide, StrategyContext
from hybrid_quant.strategy.opening_range_breakout import OpeningRangeBreakoutStrategy


def _bar(timestamp: datetime, *, close: float = 101.0) -> MarketBar:
    return MarketBar(
        timestamp=timestamp,
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=100.0,
    )


def _feature_snapshot(
    timestamp: datetime,
    **overrides: float,
) -> FeatureSnapshot:
    values = {
        "atr_14": 2.0,
        "ema_200_1h": 100.0,
        "ema_200_1h_slope": 0.5,
        "opening_range_high": 101.0,
        "opening_range_low": 95.0,
        "opening_range_width": 6.0,
        "opening_range_width_atr": 1.5,
        "opening_range_ready": 1.0,
        "opening_range_long_breakout_entry": 0.0,
        "opening_range_short_breakout_entry": 0.0,
        "opening_range_long_retest_entry": 0.0,
        "opening_range_short_retest_entry": 0.0,
        "opening_range_breakout_count_today": 1.0,
        "opening_range_first_breakout_of_day": 1.0,
        "momentum_20": 0.01,
        "candle_range_atr": 1.1,
        "relative_volume": 1.5,
    }
    values.update(overrides)
    return FeatureSnapshot(timestamp=timestamp, values=values, metadata={})


def _strategy(**overrides: object) -> OpeningRangeBreakoutStrategy:
    params = {
        "name": "opening_range_breakout_nq",
        "variant_name": "baseline_nq_orb",
        "trend_filter": "ema_200_1h",
        "execution_timeframe": "5m",
        "filter_timeframe": "1H",
        "entry_mode": "breakout_close_entry",
        "opening_range_minutes": 30,
        "retest_max_bars": 3,
        "atr_multiple_stop": 1.0,
        "atr_multiple_target": 2.0,
        "time_stop_bars": 18,
        "session_close_hour_utc": 20,
        "session_close_minute_utc": 55,
        "no_entry_minutes_before_close": 20,
        "allowed_weekdays": [0, 1, 2, 3, 4],
        "exclude_weekends": True,
        "minimum_expected_move_bps": 12.0,
        "minimum_target_to_cost_ratio": 2.5,
        "estimated_round_trip_cost_bps": 1.0,
        "momentum_lookback_bars": 20,
        "minimum_momentum_abs": 0.0025,
        "minimum_candle_range_atr": 0.75,
        "minimum_opening_range_width_atr": 0.5,
        "maximum_opening_range_width_atr": 2.5,
        "minimum_relative_volume": 1.1,
        "max_breakout_distance_atr": 0.35,
        "max_breakouts_per_day": 1,
    }
    params.update(overrides)
    return OpeningRangeBreakoutStrategy(**params)


class OpeningRangeBreakoutStrategyTests(unittest.TestCase):
    def test_generates_long_breakout_close_entry(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 5, tzinfo=UTC)
        strategy = _strategy()
        bar = _bar(timestamp, close=101.4)
        feature = _feature_snapshot(
            timestamp,
            opening_range_long_breakout_entry=1.0,
            breakout_distance_atr=0.2,
        )

        signal = strategy.generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[bar],
                features=[feature],
            )
        )

        self.assertEqual(signal.side, SignalSide.LONG)
        self.assertEqual(signal.metadata["entry_mode"], "breakout_close_entry")
        self.assertTrue(signal.metadata["first_breakout_of_day"])

    def test_generates_short_breakout_retest_entry(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 10, tzinfo=UTC)
        strategy = _strategy(entry_mode="breakout_retest_entry")
        bar = _bar(timestamp, close=94.4)
        feature = _feature_snapshot(
            timestamp,
            ema_200_1h=100.0,
            ema_200_1h_slope=-0.5,
            opening_range_short_retest_entry=1.0,
            opening_range_high=105.0,
            opening_range_low=95.0,
            momentum_20=-0.01,
            relative_volume=1.6,
        )

        signal = strategy.generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[bar],
                features=[feature],
            )
        )

        self.assertEqual(signal.side, SignalSide.SHORT)
        self.assertEqual(signal.metadata["entry_trigger"], "breakout_retest_entry")

    def test_blocks_trade_without_htf_trend_bias(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 5, tzinfo=UTC)
        strategy = _strategy()
        bar = _bar(timestamp, close=99.8)
        feature = _feature_snapshot(timestamp, opening_range_long_breakout_entry=1.0, ema_200_1h=100.0)

        signal = strategy.generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[bar],
                features=[feature],
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)
        self.assertIn("EMA200", signal.rationale)

    def test_blocks_trade_when_breakout_is_too_extended(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 5, tzinfo=UTC)
        strategy = _strategy(max_breakout_distance_atr=0.10)
        bar = _bar(timestamp, close=102.0)
        feature = _feature_snapshot(timestamp, opening_range_long_breakout_entry=1.0)

        signal = strategy.generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[bar],
                features=[feature],
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)
        self.assertIn("too extended", signal.rationale)

    def test_can_disable_ema_slope_filter_for_ablation(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 5, tzinfo=UTC)
        strategy = _strategy(use_ema_200_1h_slope=False)
        bar = _bar(timestamp, close=101.4)
        feature = _feature_snapshot(
            timestamp,
            opening_range_long_breakout_entry=1.0,
            ema_200_1h_slope=-0.5,
        )

        signal = strategy.generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[bar],
                features=[feature],
            )
        )

        self.assertEqual(signal.side, SignalSide.LONG)
        self.assertFalse(signal.metadata["use_ema_200_1h_slope"])

    def test_blocks_trade_outside_session(self) -> None:
        timestamp = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
        strategy = _strategy()
        bar = _bar(timestamp, close=101.4)
        feature = _feature_snapshot(timestamp, opening_range_long_breakout_entry=1.0)

        signal = strategy.generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[bar],
                features=[feature],
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)
        self.assertIn("Session close", signal.rationale)
