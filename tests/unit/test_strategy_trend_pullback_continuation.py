from __future__ import annotations

import unittest
from datetime import UTC, datetime

from hybrid_quant.core import FeatureSnapshot, MarketBar, SignalSide, StrategyContext
from hybrid_quant.strategy.trend_pullback_continuation import TrendPullbackContinuationStrategy


def _bar(timestamp: datetime, *, close: float = 100.0) -> MarketBar:
    return MarketBar(
        timestamp=timestamp,
        open=close - 0.1,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=100.0,
    )


def _feature(timestamp: datetime, **overrides: float) -> FeatureSnapshot:
    values = {
        "trend_pullback_trigger_long_core": 0.0,
        "trend_pullback_trigger_short_core": 0.0,
        "trend_pullback_trigger_long_macd": 0.0,
        "trend_pullback_trigger_short_macd": 0.0,
        "trend_pullback_setup_long_core": 0.0,
        "trend_pullback_setup_short_core": 0.0,
        "trend_pullback_setup_long_macd": 0.0,
        "trend_pullback_setup_short_macd": 0.0,
        "trend_pullback_stop_long": 99.0,
        "trend_pullback_stop_short": 101.0,
        "trend_pullback_atr_m5": 2.0,
        "trend_pullback_rsi_m5": 52.0,
        "trend_pullback_rsi_m1": 56.0,
        "trend_pullback_vwap_distance_atr_m5": 0.4,
        "trend_pullback_macd_hist_m5": 0.1,
        "trend_pullback_spread_points": 0.05,
    }
    values.update(overrides)
    return FeatureSnapshot(timestamp=timestamp, values=values, metadata={})


def _strategy(**overrides: object) -> TrendPullbackContinuationStrategy:
    params = {
        "name": "trend_pullback_continuation_xauusd",
        "variant_name": "core_v1",
        "execution_timeframe": "1m",
        "filter_timeframe": "15m",
        "entry_mode": "core_v1",
        "atr_multiple_target": 2.0,
        "time_stop_bars": 60,
        "close_on_session_end": True,
        "session_close_timezone": "Europe/Madrid",
        "session_close_hour_utc": 16,
        "session_close_minute_utc": 30,
        "no_entry_minutes_before_close": 5,
        "enforce_entry_session": True,
        "entry_session_timezone": "Europe/Madrid",
        "entry_session_windows": ["09:00-11:00", "14:00-16:30"],
        "allowed_weekdays": [0, 1, 2, 3, 4],
        "exclude_weekends": True,
        "minimum_stop_atr": 0.6,
        "maximum_stop_atr": 1.5,
        "maximum_spread_points": 0.30,
        "maximum_spread_to_stop_ratio": 0.12,
        "default_spread_points": 0.05,
    }
    params.update(overrides)
    return TrendPullbackContinuationStrategy(**params)


class TrendPullbackContinuationStrategyTests(unittest.TestCase):
    def test_generates_core_long_inside_madrid_session(self) -> None:
        timestamp = datetime(2024, 1, 2, 8, 30, tzinfo=UTC)
        signal = _strategy().generate(
            StrategyContext(
                symbol="XAUUSD",
                execution_timeframe="1m",
                filter_timeframe="15m",
                bars=[_bar(timestamp)],
                features=[_feature(timestamp, trend_pullback_trigger_long_core=1.0)],
            )
        )

        self.assertEqual(signal.side, SignalSide.LONG)
        self.assertEqual(signal.entry_reason, "trend_pullback_core_v1_long")
        self.assertAlmostEqual(signal.stop_price or 0.0, 98.8)
        self.assertAlmostEqual(signal.target_price or 0.0, 102.4)
        self.assertFalse(signal.metadata["outside_session"])

    def test_blocks_entries_outside_required_session_windows(self) -> None:
        timestamp = datetime(2024, 1, 2, 7, 30, tzinfo=UTC)
        signal = _strategy().generate(
            StrategyContext(
                symbol="XAUUSD",
                execution_timeframe="1m",
                filter_timeframe="15m",
                bars=[_bar(timestamp)],
                features=[_feature(timestamp, trend_pullback_trigger_long_core=1.0)],
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)
        self.assertEqual(signal.metadata["blocked_by_filter"], "outside_session")

    def test_macd_variant_requires_macd_trigger_columns(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 15, tzinfo=UTC)
        signal = _strategy(entry_mode="core_v1_macd", use_macd_confirmation=True).generate(
            StrategyContext(
                symbol="XAUUSD",
                execution_timeframe="1m",
                filter_timeframe="15m",
                bars=[_bar(timestamp)],
                features=[_feature(timestamp, trend_pullback_trigger_long_core=1.0)],
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)

        confirmed = _strategy(entry_mode="core_v1_macd", use_macd_confirmation=True).generate(
            StrategyContext(
                symbol="XAUUSD",
                execution_timeframe="1m",
                filter_timeframe="15m",
                bars=[_bar(timestamp)],
                features=[_feature(timestamp, trend_pullback_trigger_long_macd=1.0)],
            )
        )
        self.assertEqual(confirmed.side, SignalSide.LONG)

    def test_no_m1_variant_uses_m5_setup_columns(self) -> None:
        timestamp = datetime(2024, 1, 2, 8, 45, tzinfo=UTC)
        signal = _strategy(entry_mode="core_v1_no_m1", use_m1_trigger=False).generate(
            StrategyContext(
                symbol="XAUUSD",
                execution_timeframe="1m",
                filter_timeframe="15m",
                bars=[_bar(timestamp)],
                features=[_feature(timestamp, trend_pullback_setup_short_core=1.0)],
            )
        )

        self.assertEqual(signal.side, SignalSide.SHORT)
        self.assertEqual(signal.entry_reason, "trend_pullback_core_v1_no_m1_short")

    def test_blocks_spread_too_large_relative_to_stop(self) -> None:
        timestamp = datetime(2024, 1, 2, 8, 30, tzinfo=UTC)
        signal = _strategy().generate(
            StrategyContext(
                symbol="XAUUSD",
                execution_timeframe="1m",
                filter_timeframe="15m",
                bars=[_bar(timestamp)],
                features=[_feature(timestamp, trend_pullback_trigger_long_core=1.0, trend_pullback_spread_points=0.50)],
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)
        self.assertIn("Spread", signal.rationale)


if __name__ == "__main__":
    unittest.main()
