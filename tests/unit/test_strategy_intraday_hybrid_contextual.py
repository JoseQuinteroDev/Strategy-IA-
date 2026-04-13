from __future__ import annotations

import unittest
from datetime import UTC, datetime

from hybrid_quant.core import FeatureSnapshot, MarketBar, SignalSide, StrategyContext
from hybrid_quant.strategy.intraday_hybrid_contextual import IntradayHybridContextualStrategy


def _bar(timestamp: datetime, *, open_price: float = 105.2, close: float = 106.0) -> MarketBar:
    return MarketBar(
        timestamp=timestamp,
        open=open_price,
        high=max(open_price, close) + 0.8,
        low=min(open_price, close) - 0.8,
        close=close,
        volume=220.0,
    )


def _feature_snapshot(timestamp: datetime, **overrides: float) -> FeatureSnapshot:
    values = {
        "atr_14": 2.0,
        "adx_1h": 24.0,
        "ema_200_1h": 100.0,
        "ema_200_1h_slope": 0.5,
        "price_vs_ema_200_1h_pct": 0.04,
        "intraday_vwap": 104.0,
        "ema_20": 104.8,
        "ema_50": 104.2,
        "distance_to_mean": 2.0,
        "zscore_distance_to_mean": 1.1,
        "momentum_20": 0.008,
        "candle_range_atr": 0.8,
        "relative_volume": 1.2,
        "context_long_pullback_entry": 0.0,
        "context_short_pullback_entry": 0.0,
        "context_long_session_trend_entry": 0.0,
        "context_short_session_trend_entry": 0.0,
        "context_support_level": 104.8,
        "context_trigger_level": 104.8,
        "context_trigger_distance_atr": 0.35,
        "context_session_range_width_atr": 1.2,
        "context_setup_count_today": 1.0,
        "opening_range_high": 105.0,
        "opening_range_low": 99.0,
        "opening_range_mid": 102.0,
        "opening_range_width_atr": 1.5,
    }
    values.update(overrides)
    return FeatureSnapshot(timestamp=timestamp, values=values, metadata={})


def _strategy(**overrides: object) -> IntradayHybridContextualStrategy:
    params = {
        "name": "intraday_hybrid_contextual_nq",
        "variant_name": "baseline_intraday_hybrid",
        "trend_filter": "ema_200_1h",
        "regime_filter": "macro_context",
        "execution_timeframe": "5m",
        "filter_timeframe": "1H",
        "entry_mode": "macro_pullback_continuation",
        "atr_multiple_stop": 1.0,
        "atr_multiple_target": 1.5,
        "time_stop_bars": 12,
        "close_on_session_end": True,
        "session_close_hour_utc": 20,
        "session_close_minute_utc": 55,
        "no_entry_minutes_before_close": 15,
        "enforce_entry_session": True,
        "entry_session_start_hour_utc": 14,
        "entry_session_start_minute_utc": 0,
        "entry_session_end_hour_utc": 19,
        "entry_session_end_minute_utc": 0,
        "allowed_hours_utc": [14, 15, 16, 17, 18],
        "allowed_weekdays": [0, 1, 2, 3, 4],
        "exclude_weekends": True,
        "entry_zscore": 2.0,
        "mean_reversion_anchor": "vwap",
        "adx_threshold": 22.0,
        "minimum_anchor_distance_atr": 0.45,
        "minimum_expected_move_bps": 1.0,
        "minimum_target_to_cost_ratio": 1.0,
        "estimated_round_trip_cost_bps": 1.0,
        "momentum_lookback_bars": 20,
        "minimum_momentum_abs": 0.0005,
        "minimum_candle_range_atr": 0.25,
        "use_ema_200_1h_trend_filter": True,
        "use_ema_200_1h_slope": True,
        "use_macro_bias_filter": True,
        "minimum_relative_volume": 0.50,
        "max_breakout_distance_atr": 0.90,
        "max_breakouts_per_day": 4,
        "maximum_pullback_depth_atr": 0.90,
        "use_intraday_vwap_filter": True,
        "use_intraday_ema20_filter": True,
        "use_intraday_ema50_alignment": True,
    }
    params.update(overrides)
    return IntradayHybridContextualStrategy(**params)


class IntradayHybridContextualStrategyTests(unittest.TestCase):
    def test_generates_macro_pullback_long(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 10, tzinfo=UTC)
        signal = _strategy().generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp)],
                features=[_feature_snapshot(timestamp, context_long_pullback_entry=1.0)],
            )
        )

        self.assertEqual(signal.side, SignalSide.LONG)
        self.assertEqual(signal.metadata["setup_label"], "pullback_with_macro_bias")
        self.assertEqual(signal.metadata["macro_bias_side"], "long")

    def test_blocks_pullback_when_macro_bias_conflicts(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 15, tzinfo=UTC)
        signal = _strategy().generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp)],
                features=[_feature_snapshot(timestamp, ema_200_1h_slope=-0.5, context_long_pullback_entry=1.0)],
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)

    def test_can_disable_macro_bias_filter_for_htf_ablation(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 15, tzinfo=UTC)
        signal = _strategy(use_macro_bias_filter=False).generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp)],
                features=[_feature_snapshot(timestamp, ema_200_1h_slope=-0.5, context_long_pullback_entry=1.0)],
            )
        )

        self.assertEqual(signal.side, SignalSide.LONG)
        self.assertFalse(signal.metadata["outside_session"])

    def test_generates_controlled_mean_reversion_long_in_range_regime(self) -> None:
        timestamp = datetime(2024, 1, 2, 15, 0, tzinfo=UTC)
        signal = _strategy(
            entry_mode="controlled_mean_reversion",
            use_ema_200_1h_trend_filter=False,
            use_ema_200_1h_slope=False,
            minimum_momentum_abs=0.0,
            minimum_relative_volume=0.0,
            max_breakout_distance_atr=2.5,
        ).generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, open_price=100.4, close=101.0)],
                features=[
                    _feature_snapshot(
                        timestamp,
                        adx_1h=12.0,
                        intraday_vwap=103.0,
                        zscore_distance_to_mean=-2.4,
                        candle_range_atr=0.6,
                    )
                ],
            )
        )

        self.assertEqual(signal.side, SignalSide.LONG)
        self.assertEqual(signal.metadata["entry_trigger"], "controlled_mean_reversion")

    def test_blocks_mean_reversion_in_strong_trend(self) -> None:
        timestamp = datetime(2024, 1, 2, 15, 5, tzinfo=UTC)
        signal = _strategy(entry_mode="controlled_mean_reversion", use_ema_200_1h_trend_filter=False).generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, open_price=100.4, close=101.0)],
                features=[_feature_snapshot(timestamp, adx_1h=35.0, zscore_distance_to_mean=-2.4)],
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)

    def test_generates_compression_expansion_short(self) -> None:
        timestamp = datetime(2024, 1, 2, 16, 0, tzinfo=UTC)
        signal = _strategy(entry_mode="compression_expansion_continuation").generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, open_price=97.8, close=97.0)],
                features=[
                    _feature_snapshot(
                        timestamp,
                        ema_200_1h=100.0,
                        ema_200_1h_slope=-0.6,
                        price_vs_ema_200_1h_pct=-0.03,
                        context_short_session_trend_entry=1.0,
                        context_trigger_level=98.0,
                        context_trigger_distance_atr=0.5,
                        momentum_20=-0.009,
                    )
                ],
            )
        )

        self.assertEqual(signal.side, SignalSide.SHORT)
        self.assertEqual(signal.metadata["setup_label"], "compression_expansion_continuation")

    def test_blocks_entries_outside_mandatory_entry_window(self) -> None:
        timestamp = datetime(2024, 1, 2, 19, 5, tzinfo=UTC)
        signal = _strategy().generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp)],
                features=[_feature_snapshot(timestamp, context_long_pullback_entry=1.0)],
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)
        self.assertIn("outside_session", signal.rationale)
        self.assertTrue(signal.metadata["outside_session"])
        self.assertEqual(signal.metadata["blocked_by_filter"], "outside_session")
        self.assertEqual(signal.metadata["candidate_status"], "blocked_by_time_filter")

    def test_allows_entries_inside_mandatory_entry_window_boundary(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 0, tzinfo=UTC)
        signal = _strategy().generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp)],
                features=[_feature_snapshot(timestamp, context_long_pullback_entry=1.0)],
            )
        )

        self.assertEqual(signal.side, SignalSide.LONG)
        self.assertFalse(signal.metadata["outside_session"])
        self.assertEqual(signal.metadata["entry_session_window_utc"], "14:00-19:00 UTC")

    def test_blocks_exact_session_end_as_exclusive_for_next_bar_execution(self) -> None:
        timestamp = datetime(2024, 1, 2, 19, 0, tzinfo=UTC)
        signal = _strategy().generate(
            StrategyContext(
                symbol="NQ",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp)],
                features=[_feature_snapshot(timestamp, context_long_pullback_entry=1.0)],
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)
        self.assertTrue(signal.metadata["outside_session"])


if __name__ == "__main__":
    unittest.main()
