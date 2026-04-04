from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hybrid_quant.data import ExternalOHLCVImportError, ExternalOHLCVImporter, read_ohlcv_frame


class ExternalOHLCVImporterTests(unittest.TestCase):
    def test_importer_maps_common_columns_sorts_rows_and_removes_duplicates(self) -> None:
        importer = ExternalOHLCVImporter()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "external.csv"
            output_path = tmp_path / "normalized.csv"
            pd.DataFrame(
                {
                    "Timestamp": [
                        "2024-01-01 00:05:00+00:00",
                        "2024-01-01 00:00:00+00:00",
                        "2024-01-01 00:05:00+00:00",
                    ],
                    "Open": [101.0, 100.0, 102.0],
                    "High": [102.0, 101.0, 103.0],
                    "Low": [99.5, 99.0, 100.5],
                    "Close": [101.5, 100.5, 102.5],
                    "Volume": [10, 11, 12],
                }
            ).to_csv(input_path, index=False)

            result = importer.import_file(
                input_path=input_path,
                output_path=output_path,
                interval="5m",
            )

            self.assertEqual(result.rows_in, 3)
            self.assertEqual(result.cleaning_report.duplicates_removed, 1)
            self.assertEqual(result.dominant_interval, "5m")
            self.assertEqual(result.column_mapping["open_time"], "Timestamp")

            frame = read_ohlcv_frame(output_path)
            self.assertEqual(list(frame.columns), ["open", "high", "low", "close", "volume"])
            self.assertEqual(len(frame), 2)
            self.assertTrue(frame.index.is_monotonic_increasing)
            self.assertEqual(frame.index.name, "open_time")

    def test_importer_combines_separate_date_and_time_columns(self) -> None:
        importer = ExternalOHLCVImporter()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "mt.csv"
            output_path = tmp_path / "normalized.csv"
            pd.DataFrame(
                {
                    "<DATE>": ["2024-01-01", "2024-01-01"],
                    "<TIME>": ["00:00:00", "00:05:00"],
                    "<OPEN>": [100.0, 101.0],
                    "<HIGH>": [101.0, 102.0],
                    "<LOW>": [99.0, 100.0],
                    "<CLOSE>": [100.5, 101.5],
                    "<TICK_VOLUME>": [20, 21],
                }
            ).to_csv(input_path, index=False)

            result = importer.import_file(
                input_path=input_path,
                output_path=output_path,
                interval="5m",
            )

            self.assertEqual(result.column_mapping["open_time"], "<DATE>+<TIME>")
            frame = read_ohlcv_frame(output_path)
            self.assertEqual(frame.index[0].isoformat(), "2024-01-01T00:00:00+00:00")
            self.assertEqual(frame.iloc[1]["volume"], 21.0)

    def test_importer_raises_clear_error_when_required_columns_are_missing(self) -> None:
        importer = ExternalOHLCVImporter()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "missing-volume.csv"
            output_path = tmp_path / "normalized.csv"
            pd.DataFrame(
                {
                    "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z"],
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.0, 100.0],
                    "close": [100.5, 101.5],
                }
            ).to_csv(input_path, index=False)

            with self.assertRaises(ExternalOHLCVImportError) as context:
                importer.import_file(
                    input_path=input_path,
                    output_path=output_path,
                    interval="5m",
                )

        message = str(context.exception)
        self.assertIn("volume", message)
        self.assertIn("Accepted aliases", message)

    def test_importer_raises_clear_error_when_timestamps_are_invalid(self) -> None:
        importer = ExternalOHLCVImporter()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "bad-time.csv"
            output_path = tmp_path / "normalized.csv"
            pd.DataFrame(
                {
                    "timestamp": ["not-a-time", "2024-01-01T00:05:00Z"],
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.0, 100.0],
                    "close": [100.5, 101.5],
                    "volume": [10, 11],
                }
            ).to_csv(input_path, index=False)

            with self.assertRaises(ExternalOHLCVImportError) as context:
                importer.import_file(
                    input_path=input_path,
                    output_path=output_path,
                    interval="5m",
                )

        self.assertIn("Could not parse all timestamps", str(context.exception))

    def test_importer_raises_clear_error_on_wrong_bar_cadence(self) -> None:
        importer = ExternalOHLCVImporter()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "one-minute.csv"
            output_path = tmp_path / "normalized.csv"
            pd.DataFrame(
                {
                    "timestamp": [
                        "2024-01-01T00:00:00Z",
                        "2024-01-01T00:01:00Z",
                        "2024-01-01T00:02:00Z",
                    ],
                    "open": [100.0, 101.0, 102.0],
                    "high": [101.0, 102.0, 103.0],
                    "low": [99.0, 100.0, 101.0],
                    "close": [100.5, 101.5, 102.5],
                    "volume": [10, 11, 12],
                }
            ).to_csv(input_path, index=False)

            with self.assertRaises(ExternalOHLCVImportError) as context:
                importer.import_file(
                    input_path=input_path,
                    output_path=output_path,
                    interval="5m",
                    allow_gaps=True,
                )

        self.assertIn("dominant interval appears to be 1m", str(context.exception))
