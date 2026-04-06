from __future__ import annotations

import unittest
from datetime import UTC, datetime

from hybrid_quant.core import FeatureSnapshot, MarketBar, SignalSide, StrategyContext
from hybrid_quant.strategy.orb_intraday_active import IntradayActiveOrbStrategy


def _bar(timestamp: datetime, *, close: float = 106.0) -> MarketBar:
    return MarketBar(
        timestamp=timestamp,
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=150.0,
    )


def _feature_snapshot(timestamp: datetime, **overrides: float) -> FeatureSnapshot:
    values = {
        "atr_14": 2.0,
        "ema_200_1h": 100.0,
        "ema_200_1h_slope": 0.5,
        "intraday_vwap": 104.0,
        "ema_20": 104.5,
        "opening_range_high": 105.0,
        "opening_range_low": 99.0,
        "opening_range_width": 6.0,
        "opening_range_width_atr": 1.5,
        "opening_range_ready": 1.0,
        "opening_range_long_continuation_entry": 0.0,
        "opening_range_short_continuation_entry": 0.0,
        "opening_range_long_pullback_entry": 0.0,
        "opening_range_short_pullback_entry": 0.0,
        "opening_range_long_reclaim_entry": 0.0,
        "opening_range_short_reclaim_entry": 0.0,
        "opening_range_acceptance_bars": 2.0,
        "opening_range_pullback_depth_atr": 0.25,
        "opening_range_reclaim_distance_atr": 0.20,
        "opening_range_breakout_age_bars": 2.0,
        "opening_range_intraday_setup_count_today": 1.0,
        "opening_range_intraday_first_setup_of_day": 1.0,
        "momentum_20": 0.008,
        "candle_range_atr": 0.8,
        "relative_volume": 1.2,
    }
    values.update(overrides)
    return FeatureSnapshot(timestamp=timestamp, values=values, metadata={})


def _strategy(**overrides: object) -> IntradayActiveOrbStrategy:
    params = {
        "name": "orb_intraday_active_nq",
        "variant_name": "baseline_nq_intraday_orb_active",
        "trend_filter": "ema_200_1h",
        "execution_timeframe": "5m",
        "filter_timeframe": "1H",
        "entry_mode": "breakout_continuation",
        "opening_range_minutes": 30,
        "retest_max_bars": 8,
        "atr_multiple_stop": 1.0,
        "atr_multiple_target": 1.5,
        "time_stop_bars": 12,
        "session_close_hour_utc": 20,
        "session_close_minute_utc": 55,
        "no_entry_minutes_before_close": 15,
        "allowed_weekdays": [0, 1, 2, 3, 4],
        "exclude_weekends": True,
        "minimum_expected_move_bps": 6.0,
        "minimum_target_to_cost_ratio": 1.4,
        "estimated_round_trip_cost_bps": 1.0,
        "momentum_lookback_bars": 20,
        "minimum_momentum_abs": 0.0010,
        "minimum_candle_range_atr": 0.40,
        "minimum_opening_range_width_atr": 0.25,
        "maximum_opening_range_width_atr": 3.25,
        "minimum_relative_volume": 0.85,
        "max_breakout_distance_atr": 0.75,
        "max_breakouts_per_day": 4,
        "minimum_acceptance_bars": 1,
        "maximum_pullback_depth_atr": 0.80,
        "use_intraday_vwap_filter": True,
        "use_intraday_ema20_filter": False,
    }
    params.update(overrides)
    return IntradayActiveOrbStrategy(**params)


class IntradayActiveOrbStrategyTests(unittest.TestCase):
    def test_generates_long_continuation_entry(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 5, tzinfo=UTC)
        signal = _strategy().generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=106.2)],
                features=[_feature_snapshot(timestamp, opening_range_long_continuation_entry=1.0)],
            )
        )

        self.assertEqual(signal.side, SignalSide.LONG)
        self.assertEqual(signal.metadata["entry_trigger"], "breakout_continuation")

    def test_generates_short_pullback_entry(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 15, tzinfo=UTC)
        signal = _strategy(entry_mode="first_pullback_after_breakout").generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=97.8)],
                features=[
                    _feature_snapshot(
                        timestamp,
                        ema_200_1h=100.0,
                        ema_200_1h_slope=-0.5,
                        intraday_vwap=98.8,
                        ema_20=98.5,
                        opening_range_high=106.0,
                        opening_range_low=99.0,
                        opening_range_short_pullback_entry=1.0,
                        opening_range_acceptance_bars=2.0,
                        opening_range_pullback_depth_atr=0.35,
                        momentum_20=-0.007,
                    )
                ],
            )
        )

        self.assertEqual(signal.side, SignalSide.SHORT)
        self.assertEqual(signal.metadata["entry_trigger"], "first_pullback_after_breakout")

    def test_blocks_pullback_when_too_deep(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 20, tzinfo=UTC)
        signal = _strategy(
            entry_mode="first_pullback_after_breakout",
            maximum_pullback_depth_atr=0.30,
        ).generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=106.0)],
                features=[
                    _feature_snapshot(
                        timestamp,
                        opening_range_long_pullback_entry=1.0,
                        opening_range_pullback_depth_atr=0.50,
                    )
                ],
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)
        self.assertIn("too deep", signal.rationale)

    def test_reclaim_requires_vwap_acceptance_when_enabled(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 25, tzinfo=UTC)
        signal = _strategy(entry_mode="reclaim_acceptance").generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=105.2)],
                features=[
                    _feature_snapshot(
                        timestamp,
                        opening_range_long_reclaim_entry=1.0,
                        intraday_vwap=105.5,
                    )
                ],
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)
        self.assertIn("VWAP", signal.rationale)
