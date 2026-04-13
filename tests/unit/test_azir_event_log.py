from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from hybrid_quant.azir import AZIR_EVENT_COLUMNS, validate_event_row, write_event_log


class AzirEventLogSchemaTests(unittest.TestCase):
    def test_validate_event_row_accepts_minimal_supported_event(self) -> None:
        row = {
            "timestamp": "2025.12.31 16:30:00",
            "event_id": "2025.12.31_XAUUSD_123456321",
            "event_type": "opportunity",
            "symbol": "XAUUSD",
            "magic": "123456321",
        }

        self.assertEqual(validate_event_row(row), [])

    def test_validate_event_row_reports_missing_and_unknown_fields(self) -> None:
        row = {
            "timestamp": "",
            "event_type": "not_a_real_event",
            "symbol": "XAUUSD",
            "magic": "123456321",
            "extra": "unexpected",
        }

        errors = validate_event_row(row)

        self.assertIn("missing required columns: event_id", errors)
        self.assertIn("unknown columns: extra", errors)
        self.assertIn("empty required column: timestamp", errors)
        self.assertIn("empty required column: event_id", errors)
        self.assertIn("unsupported event_type: not_a_real_event", errors)

    def test_write_event_log_uses_canonical_column_order(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
            output_path = write_event_log(
                [
                    {
                        "timestamp": "2025.12.31 16:30:00",
                        "event_id": "2025.12.31_XAUUSD_123456321",
                        "event_type": "blocked_friday",
                        "symbol": "XAUUSD",
                        "magic": "123456321",
                        "notes": "NoTradeFridays blocked the daily opportunity.",
                    }
                ],
                Path(tmp_dir) / "azir_events.csv",
            )

            with output_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

        self.assertEqual(tuple(reader.fieldnames or ()), AZIR_EVENT_COLUMNS)
        self.assertEqual(rows[0]["event_type"], "blocked_friday")
        self.assertEqual(rows[0]["notes"], "NoTradeFridays blocked the daily opportunity.")


if __name__ == "__main__":
    unittest.main()
