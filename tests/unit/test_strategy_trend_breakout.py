from __future__ import annotations

import unittest
from datetime import UTC, datetime

from hybrid_quant.core import FeatureSnapshot, MarketBar, SignalSide, StrategyContext
from hybrid_quant.strategy import TrendBreakoutStrategy


def _bar(timestamp: datetime, close: float, *, spread: float = 1.2) -> MarketBar:
    return MarketBar(
        timestamp=timestamp,
        open=close - 0.4,
        high=close + spread,
        low=close - spread,
        close=close,
        volume=100.0,
    )


def _feature(timestamp: datetime, **values: float) -> FeatureSnapshot:
    return FeatureSnapshot(timestamp=timestamp, values=values, metadata={})


def _strategy(**overrides: object) -> TrendBreakoutStrategy:
    params: dict[str, object] = {
        "name": "trend_breakout_nasdaq",
        "variant_name": "baseline_trend_nasdaq",
        "trend_filter": "ema_200_1h",
        "regime_filter": "adx_1h",
        "execution_timeframe": "5m",
        "filter_timeframe": "1H",
        "adx_threshold": 22.0,
        "atr_multiple_stop": 1.0,
        "atr_multiple_target": 2.0,
        "time_stop_bars": 18,
        "session_close_hour_utc": 20,
        "session_close_minute_utc": 55,
        "no_entry_minutes_before_close": 20,
        "allowed_hours_utc": [13, 14, 15, 16, 17, 18, 19, 20],
        "allowed_weekdays": [0, 1, 2, 3, 4],
        "exclude_weekends": True,
        "minimum_expected_move_bps": 12.0,
        "minimum_target_to_cost_ratio": 2.5,
        "estimated_round_trip_cost_bps": 3.0,
        "breakout_lookback_bars": 20,
        "breakout_buffer_atr": 0.1,
        "minimum_breakout_range_atr": 1.2,
        "momentum_lookback_bars": 20,
        "minimum_momentum_abs": 0.0025,
        "minimum_candle_range_atr": 0.75,
    }
    params.update(overrides)
    return TrendBreakoutStrategy(**params)


class TrendBreakoutStrategyTests(unittest.TestCase):
    def test_generates_long_breakout_signal(self) -> None:
        timestamp = datetime(2024, 1, 2, 15, 0, tzinfo=UTC)
        strategy = _strategy()
        signal = strategy.generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=121.0)],
                features=[
                    _feature(
                        timestamp,
                        ema_200_1h=118.0,
                        adx_1h=28.0,
                        atr_14=1.5,
                        breakout_high_20=120.0,
                        breakout_low_20=114.0,
                        breakout_range_width_atr_20=4.0,
                        momentum_20=0.009,
                        candle_range_atr=1.1,
                    )
                ],
                regime="trend",
            )
        )

        self.assertEqual(signal.side, SignalSide.LONG)
        self.assertAlmostEqual(signal.entry_price or 0.0, 121.0)
        self.assertAlmostEqual(signal.stop_price or 0.0, 119.5)
        self.assertAlmostEqual(signal.target_price or 0.0, 124.0)
        self.assertIn("Long breakout", signal.entry_reason or "")
        self.assertEqual(signal.metadata["strategy_family"], "trend_breakout")

    def test_generates_short_breakout_signal(self) -> None:
        timestamp = datetime(2024, 1, 2, 16, 0, tzinfo=UTC)
        strategy = _strategy()
        signal = strategy.generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=109.0)],
                features=[
                    _feature(
                        timestamp,
                        ema_200_1h=112.0,
                        adx_1h=29.0,
                        atr_14=1.5,
                        breakout_high_20=116.0,
                        breakout_low_20=110.0,
                        breakout_range_width_atr_20=4.0,
                        momentum_20=-0.010,
                        candle_range_atr=1.0,
                    )
                ],
                regime="trend",
            )
        )

        self.assertEqual(signal.side, SignalSide.SHORT)
        self.assertAlmostEqual(signal.entry_price or 0.0, 109.0)
        self.assertAlmostEqual(signal.stop_price or 0.0, 110.5)
        self.assertAlmostEqual(signal.target_price or 0.0, 106.0)
        self.assertIn("Short breakout", signal.entry_reason or "")

    def test_blocks_trade_when_breakout_quality_is_too_weak(self) -> None:
        timestamp = datetime(2024, 1, 2, 15, 0, tzinfo=UTC)
        strategy = _strategy()
        signal = strategy.generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=120.3, spread=0.5)],
                features=[
                    _feature(
                        timestamp,
                        ema_200_1h=118.0,
                        adx_1h=28.0,
                        atr_14=1.5,
                        breakout_high_20=120.0,
                        breakout_low_20=114.0,
                        breakout_range_width_atr_20=0.8,
                        momentum_20=0.001,
                        candle_range_atr=0.4,
                    )
                ],
                regime="trend",
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)
        self.assertIn(signal.rationale, {
            "Setup rejected because the prior range is too narrow relative to ATR.",
            "Setup rejected because the momentum filter is too weak for a trend breakout.",
            "Setup rejected because the breakout candle lacks volatility expansion.",
        })

    def test_blocks_trade_outside_allowed_hours(self) -> None:
        timestamp = datetime(2024, 1, 2, 10, 0, tzinfo=UTC)
        strategy = _strategy()
        signal = strategy.generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=121.0)],
                features=[
                    _feature(
                        timestamp,
                        ema_200_1h=118.0,
                        adx_1h=28.0,
                        atr_14=1.5,
                        breakout_high_20=120.0,
                        breakout_low_20=114.0,
                        breakout_range_width_atr_20=4.0,
                        momentum_20=0.009,
                        candle_range_atr=1.1,
                    )
                ],
                regime="trend",
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)
        self.assertTrue(signal.metadata["session_gate"])
        self.assertIn("Hour whitelist blocked", signal.rationale)
