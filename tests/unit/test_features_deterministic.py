from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hybrid_quant.features import build_features


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
            "adx_1h",
            "intraday_vwap",
            "ema_50",
            "realized_volatility_20",
            "candle_range",
            "candle_range_pct",
            "distance_to_mean",
            "zscore_distance_to_mean",
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

    def test_build_features_validates_required_columns(self) -> None:
        df = _sample_ohlcv().drop(columns=["volume"])

        with self.assertRaises(ValueError):
            build_features(df)
