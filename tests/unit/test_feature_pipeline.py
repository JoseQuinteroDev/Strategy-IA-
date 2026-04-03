from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from hybrid_quant.core import MarketBar, MarketDataBatch
from hybrid_quant.features import FeaturePipeline


class FeaturePipelineTests(unittest.TestCase):
    def test_pipeline_transforms_market_batch_into_feature_snapshots(self) -> None:
        bars = [
            MarketBar(
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=UTC) + timedelta(minutes=5 * index),
                open=100.0 + index,
                high=100.8 + index,
                low=99.2 + index,
                close=100.4 + index,
                volume=50.0 + index,
            )
            for index in range(30)
        ]

        pipeline = FeaturePipeline(
            feature_names=["log_return", "candle_range", "hour_utc"],
            lookback_window=50,
            regime_window=200,
            normalize=True,
        )

        snapshots = pipeline.transform(MarketDataBatch(symbol="BTCUSDT", timeframe="5m", bars=bars))

        self.assertEqual(len(snapshots), len(bars))
        self.assertEqual(set(snapshots[-1].values.keys()), {"log_return", "candle_range", "hour_utc"})
        self.assertEqual(snapshots[0].metadata["source"], "deterministic")
        self.assertAlmostEqual(snapshots[0].values["candle_range"], 1.6)

