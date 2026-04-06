from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hybrid_quant.features import DeterministicFeatureConfig, build_features


def _sample_ohlcv(periods: int = 5 * 24 * 12) -> pd.DataFrame:
    index = pd.date_range("2024-01-01T00:00:00Z", periods=periods, freq="5min", tz="UTC")
    index.name = "open_time"
    step = np.arange(periods, dtype=float)
    close = 100.0 + (step * 0.02) + np.sin(step / 8.0)
    frame = pd.DataFrame(index=index)
    frame["open"] = close - 0.15
    frame["high"] = close + 0.45 + (step % 3) * 0.01
    frame["low"] = close - 0.55 - (step % 3) * 0.01
    frame["close"] = close
    frame["volume"] = 50.0 + (step % 24)
    return frame


class DeterministicFeatureTests(unittest.TestCase):
    def test_build_features_returns_expected_columns_and_index(self) -> None:
        df = _sample_ohlcv()

        features = build_features(df)

        expected_columns = {
            "log_return",
            "atr_14",
            "ema_200_1h",
            "ema_200_1h_slope",
            "adx_1h",
            "intraday_vwap",
            "ema_50",
            "realized_volatility_20",
            "candle_range",
            "candle_range_pct",
            "candle_range_atr",
            "relative_volume",
            "distance_to_mean",
            "zscore_distance_to_mean",
            "breakout_high_20",
            "breakout_low_20",
            "breakout_range_width_20",
            "breakout_range_width_atr_20",
            "momentum_20",
            "opening_range_high",
            "opening_range_low",
            "opening_range_mid",
            "opening_range_width",
            "opening_range_width_atr",
            "opening_range_ready",
            "opening_range_long_breakout_entry",
            "opening_range_short_breakout_entry",
            "opening_range_long_retest_entry",
            "opening_range_short_retest_entry",
            "opening_range_breakout_count_today",
            "opening_range_first_breakout_of_day",
            "context_bias_side",
            "context_long_pullback_entry",
            "context_short_pullback_entry",
            "context_long_reclaim_entry",
            "context_short_reclaim_entry",
            "context_long_session_trend_entry",
            "context_short_session_trend_entry",
            "context_trigger_distance_atr",
            "price_vs_ema_200_1h_pct",
            "hour_utc",
            "day_of_week",
            "session_asia",
            "session_europe",
            "session_us",
        }

        self.assertTrue(expected_columns.issubset(features.columns))
        pd.testing.assert_index_equal(features.index, df.index)
        self.assertAlmostEqual(
            features["log_return"].iloc[1],
            np.log(df["close"].iloc[1] / df["close"].iloc[0]),
        )
        self.assertAlmostEqual(
            features["candle_range"].iloc[25],
            df["high"].iloc[25] - df["low"].iloc[25],
        )
        self.assertAlmostEqual(
            features["candle_range_pct"].iloc[25],
            (df["high"].iloc[25] - df["low"].iloc[25]) / df["close"].iloc[25],
        )

    def test_build_features_creates_temporal_and_higher_timeframe_features(self) -> None:
        df = _sample_ohlcv()

        features = build_features(df)

        self.assertEqual(features.loc[df.index[0], "hour_utc"], 0.0)
        self.assertEqual(features.loc[df.index[0], "day_of_week"], 0.0)
        self.assertEqual(features.loc[df.index[0], "session_asia"], 1.0)
        self.assertEqual(features.loc[pd.Timestamp("2024-01-01T08:00:00Z"), "session_europe"], 1.0)
        self.assertEqual(features.loc[pd.Timestamp("2024-01-01T16:00:00Z"), "session_us"], 1.0)
        self.assertTrue(
            (features[["session_asia", "session_europe", "session_us"]].sum(axis=1) == 1.0).all()
        )
        self.assertTrue(pd.isna(features.loc[pd.Timestamp("2024-01-01T00:55:00Z"), "ema_200_1h"]))
        self.assertFalse(pd.isna(features.loc[pd.Timestamp("2024-01-01T01:00:00Z"), "ema_200_1h"]))
        self.assertFalse(features["adx_1h"].dropna().empty)
        self.assertGreaterEqual(features["adx_1h"].dropna().iloc[-1], 0.0)
        self.assertFalse(features["zscore_distance_to_mean"].dropna().empty)
        self.assertFalse(features["momentum_20"].dropna().empty)
        self.assertFalse(features["breakout_high_20"].dropna().empty)

    def test_breakout_features_are_causal_and_shifted(self) -> None:
        df = _sample_ohlcv(periods=120)

        features = build_features(df)
        probe_index = df.index[40]
        probe_position = df.index.get_loc(probe_index)
        previous_high = df["high"].iloc[probe_position - 20 : probe_position].max()
        previous_low = df["low"].iloc[probe_position - 20 : probe_position].min()

        self.assertAlmostEqual(features.loc[probe_index, "breakout_high_20"], previous_high)
        self.assertAlmostEqual(features.loc[probe_index, "breakout_low_20"], previous_low)
        self.assertAlmostEqual(
            features.loc[probe_index, "breakout_range_width_20"],
            previous_high - previous_low,
        )
        self.assertAlmostEqual(
            features.loc[probe_index, "momentum_20"],
            (df["close"].iloc[probe_position] / df["close"].iloc[probe_position - 20]) - 1.0,
        )

    def test_build_features_validates_required_columns(self) -> None:
        df = _sample_ohlcv().drop(columns=["volume"])

        with self.assertRaises(ValueError):
            build_features(df)

    def test_opening_range_features_are_built_after_the_range_completes(self) -> None:
        index = pd.date_range("2024-01-02T13:30:00Z", periods=14, freq="5min", tz="UTC")
        frame = pd.DataFrame(index=index)
        frame["open"] = [100, 101, 102, 101, 103, 102, 104, 107, 106, 108, 107, 109, 108, 110]
        frame["high"] = [101, 103, 104, 103, 105, 104, 108, 109, 108, 110, 109, 111, 110, 112]
        frame["low"] = [99, 100, 101, 100, 102, 101, 103, 105, 104, 106, 105, 107, 106, 108]
        frame["close"] = [100, 102, 103, 102, 104, 103, 107, 106, 107, 109, 108, 110, 109, 111]
        frame["volume"] = np.linspace(100, 200, len(frame))

        features = build_features(
            frame,
            config=DeterministicFeatureConfig(opening_range_minutes=30, retest_max_bars=3),
        )

        self.assertTrue(pd.isna(features.loc[pd.Timestamp("2024-01-02T13:55:00Z"), "opening_range_high"]))
        self.assertEqual(features.loc[pd.Timestamp("2024-01-02T14:00:00Z"), "opening_range_ready"], 1.0)
        self.assertAlmostEqual(features.loc[pd.Timestamp("2024-01-02T14:00:00Z"), "opening_range_high"], 105.0)
        self.assertAlmostEqual(features.loc[pd.Timestamp("2024-01-02T14:00:00Z"), "opening_range_low"], 99.0)
        self.assertAlmostEqual(features.loc[pd.Timestamp("2024-01-02T14:00:00Z"), "opening_range_mid"], 102.0)
        self.assertAlmostEqual(features.loc[pd.Timestamp("2024-01-02T14:00:00Z"), "opening_range_width"], 6.0)

    def test_opening_range_features_mark_breakout_and_retest_entries_causally(self) -> None:
        index = pd.date_range("2024-01-02T13:30:00Z", periods=10, freq="5min", tz="UTC")
        frame = pd.DataFrame(index=index)
        frame["open"] = [100, 101, 102, 101, 103, 102, 105, 104, 106, 107]
        frame["high"] = [101, 103, 104, 103, 105, 104, 108, 106, 108, 109]
        frame["low"] = [99, 100, 101, 100, 102, 101, 103, 104, 105, 106]
        frame["close"] = [100, 102, 103, 102, 104, 103, 107, 105, 107, 108]
        frame["volume"] = np.linspace(100, 220, len(frame))

        features = build_features(
            frame,
            config=DeterministicFeatureConfig(opening_range_minutes=30, retest_max_bars=2),
        )

        breakout_bar = pd.Timestamp("2024-01-02T14:00:00Z")
        retest_bar = pd.Timestamp("2024-01-02T14:05:00Z")
        self.assertEqual(features.loc[breakout_bar, "opening_range_long_breakout_entry"], 1.0)
        self.assertEqual(features.loc[breakout_bar, "opening_range_breakout_count_today"], 1.0)
        self.assertEqual(features.loc[breakout_bar, "opening_range_first_breakout_of_day"], 1.0)
        self.assertEqual(features.loc[retest_bar, "opening_range_long_retest_entry"], 1.0)
        self.assertEqual(features.loc[retest_bar, "opening_range_breakout_count_today"], 1.0)
        self.assertEqual(features.loc[retest_bar, "opening_range_short_breakout_entry"], 0.0)

    def test_opening_range_intraday_active_features_mark_continuation_pullback_and_reclaim(self) -> None:
        index = pd.date_range("2024-01-02T13:30:00Z", periods=12, freq="5min", tz="UTC")
        frame = pd.DataFrame(index=index)
        frame["open"] = [100, 101, 102, 101, 103, 102, 106, 107, 106, 104, 106, 108]
        frame["high"] = [101, 103, 104, 103, 105, 104, 108, 110, 111, 106, 108, 110]
        frame["low"] = [99, 100, 101, 100, 102, 101, 104, 106, 105, 103, 104, 107]
        frame["close"] = [100, 102, 103, 102, 104, 103, 107, 109, 110, 104, 107, 109]
        frame["volume"] = np.linspace(100, 260, len(frame))

        features = build_features(
            frame,
            config=DeterministicFeatureConfig(
                opening_range_minutes=30,
                retest_max_bars=8,
                opening_range_breakout_buffer_atr=0.0,
            ),
        )

        self.assertEqual(
            features.loc[pd.Timestamp("2024-01-02T14:05:00Z"), "opening_range_long_continuation_entry"],
            1.0,
        )
        self.assertEqual(
            features.loc[pd.Timestamp("2024-01-02T14:10:00Z"), "opening_range_long_pullback_entry"],
            1.0,
        )
        self.assertEqual(
            features.loc[pd.Timestamp("2024-01-02T14:20:00Z"), "opening_range_long_reclaim_entry"],
            1.0,
        )
        self.assertEqual(
            features.loc[pd.Timestamp("2024-01-02T14:20:00Z"), "opening_range_intraday_setup_count_today"],
            3.0,
        )

    def test_intraday_contextual_features_mark_pullback_and_reclaim_entries(self) -> None:
        index = pd.date_range("2024-01-02T13:30:00Z", periods=13, freq="5min", tz="UTC")
        frame = pd.DataFrame(index=index)
        frame["open"] = [100, 101, 102, 101, 103, 102, 104, 105, 104.6, 104.3, 103.9, 105.7, 105.8]
        frame["high"] = [101, 103, 104, 103, 105, 104, 105.5, 106.2, 105.2, 104.9, 106.4, 107.2, 108.0]
        frame["low"] = [99, 100, 101, 100, 102, 101, 103.5, 104.7, 103.1, 103.2, 103.7, 105.4, 105.3]
        frame["close"] = [100, 102, 103, 102, 104, 103, 105.0, 105.8, 104.8, 103.3, 106.1, 105.9, 107.4]
        frame["volume"] = [100, 110, 120, 130, 140, 150, 220, 240, 180, 190, 230, 250, 260]

        features = build_features(
            frame,
            config=DeterministicFeatureConfig(
                opening_range_minutes=30,
                opening_range_breakout_buffer_atr=0.05,
            ),
        )

        self.assertEqual(features.loc[pd.Timestamp("2024-01-02T14:10:00Z"), "context_long_pullback_entry"], 1.0)
        self.assertEqual(features.loc[pd.Timestamp("2024-01-02T14:20:00Z"), "context_long_reclaim_entry"], 1.0)
        self.assertEqual(features.loc[pd.Timestamp("2024-01-02T14:20:00Z"), "context_bias_side"], 1.0)

    def test_intraday_contextual_session_trend_can_disable_or_mid_structure(self) -> None:
        index = pd.date_range("2024-01-02T13:30:00Z", periods=13, freq="5min", tz="UTC")
        frame = pd.DataFrame(index=index)
        frame["open"] = [102.0, 102.2, 102.4, 102.3, 102.5, 102.4, 103.2, 103.4, 103.6, 103.8, 104.0, 104.1, 104.4]
        frame["high"] = [110.0, 103.0, 103.2, 103.1, 103.3, 103.2, 103.6, 103.8, 104.0, 104.2, 104.4, 104.5, 104.9]
        frame["low"] = [100.0, 101.9, 102.0, 102.0, 102.1, 102.0, 103.0, 103.2, 103.4, 103.6, 103.8, 103.9, 104.2]
        frame["close"] = [102.1, 102.3, 102.5, 102.4, 102.6, 102.5, 103.4, 103.6, 103.8, 104.0, 104.2, 104.3, 104.8]
        frame["volume"] = [40, 45, 50, 55, 60, 65, 220, 240, 250, 260, 270, 280, 290]

        default_features = build_features(
            frame,
            config=DeterministicFeatureConfig(
                opening_range_minutes=30,
                opening_range_breakout_buffer_atr=0.0,
            ),
        )
        relaxed_features = build_features(
            frame,
            config=DeterministicFeatureConfig(
                opening_range_minutes=30,
                opening_range_breakout_buffer_atr=0.0,
                use_opening_range_mid_filter=False,
                require_context_or_mid_structure=False,
            ),
        )

        probe_bar = pd.Timestamp("2024-01-02T14:30:00Z")
        self.assertEqual(default_features.loc[probe_bar, "context_long_session_trend_entry"], 0.0)
        self.assertEqual(relaxed_features.loc[probe_bar, "context_long_session_trend_entry"], 1.0)
