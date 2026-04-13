from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from hybrid_quant.azir import classify_columns, inspect_csv


class AzirCsvInspectionTests(unittest.TestCase):
    def test_classifies_azir_event_log_columns(self) -> None:
        columns = ["timestamp", "event_id", "event_type", "symbol", "magic", "net_pnl"]

        self.assertEqual(classify_columns(columns), "azir_event_log")

    def test_classifies_ohlcv_columns(self) -> None:
        columns = ["open_time", "open", "high", "low", "close", "volume"]

        self.assertEqual(classify_columns(columns), "ohlcv")

    def test_inspect_csv_reports_event_log_coverage_and_suitability(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
            path = Path(tmp_dir) / "azir_events.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["timestamp", "event_id", "event_type", "symbol", "magic"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "timestamp": "2021.01.04 16:30:00",
                        "event_id": "2021.01.04_XAUUSD-STD_123456321",
                        "event_type": "opportunity",
                        "symbol": "XAUUSD-STD",
                        "magic": "123456321",
                    }
                )
                writer.writerow(
                    {
                        "timestamp": "2021.01.05 16:30:00",
                        "event_id": "2021.01.05_XAUUSD-STD_123456321",
                        "event_type": "blocked_friday",
                        "symbol": "XAUUSD-STD",
                        "magic": "123456321",
                    }
                )

            inspection = inspect_csv(path)

        self.assertEqual(inspection.dataset_type, "azir_event_log")
        self.assertEqual(inspection.rows, 2)
        self.assertEqual(inspection.first_timestamp, "2021-01-04 16:30:00")
        self.assertEqual(inspection.last_timestamp, "2021-01-05 16:30:00")
        self.assertEqual(inspection.suitability["opportunity_parity"], "yes")


if __name__ == "__main__":
    unittest.main()
