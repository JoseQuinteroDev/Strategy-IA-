from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from hybrid_quant.data import ParquetDatasetStore


class ParquetStorageTests(unittest.TestCase):
    def test_export_frame_calls_to_parquet_with_target_path(self) -> None:
        frame = pd.DataFrame({"close": [100.0]})
        store = ParquetDatasetStore(compression="snappy", engine=None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "dataset.parquet"
            with patch.object(pd.DataFrame, "to_parquet", autospec=True) as mocked_to_parquet:
                exported = store.export_frame(frame, target)

        self.assertEqual(exported, target)
        mocked_to_parquet.assert_called_once()
        self.assertEqual(mocked_to_parquet.call_args.args[1], target)
        self.assertEqual(mocked_to_parquet.call_args.kwargs["compression"], "snappy")

