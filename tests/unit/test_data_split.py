from __future__ import annotations

import unittest

import pandas as pd

from hybrid_quant.data import TemporalDatasetSplitter, TemporalSplitConfig


class TemporalSplitTests(unittest.TestCase):
    def test_splitter_preserves_chronological_order(self) -> None:
        index = pd.date_range("2024-01-01T00:00:00Z", periods=10, freq="5min", tz="UTC")
        frame = pd.DataFrame({"close": list(range(10))}, index=index)

        splits = TemporalDatasetSplitter().split(
            frame,
            TemporalSplitConfig(train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2),
        )

        self.assertEqual(splits["train"].row_count, 6)
        self.assertEqual(splits["validation"].row_count, 2)
        self.assertEqual(splits["test"].row_count, 2)
        self.assertLess(splits["train"].end, splits["validation"].start)
        self.assertLess(splits["validation"].end, splits["test"].start)

