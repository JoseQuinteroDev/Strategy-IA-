from __future__ import annotations

import unittest

import pandas as pd

from hybrid_quant.data import OHLCVCleaner, TimeIndexValidationError, TimeIndexValidator


class DataCleaningTests(unittest.TestCase):
    def test_cleaner_sorts_index_and_removes_duplicate_timestamps(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [102.0, 100.0, 101.0],
                "close": [102.5, 100.5, 101.5],
            },
            index=pd.to_datetime(
                [
                    "2024-01-01T00:10:00Z",
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:10:00Z",
                ],
                utc=True,
            ),
        )

        cleaned, report = OHLCVCleaner().clean(frame)

        self.assertEqual(report.rows_in, 3)
        self.assertEqual(report.rows_out, 2)
        self.assertEqual(report.duplicates_removed, 1)
        self.assertTrue(cleaned.index.is_monotonic_increasing)
        self.assertTrue(cleaned.index.is_unique)
        self.assertEqual(cleaned.index.name, "open_time")

    def test_validator_detects_gaps_when_strict(self) -> None:
        frame = pd.DataFrame(
            {"close": [100.0, 101.0, 102.0]},
            index=pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:05:00Z",
                    "2024-01-01T00:15:00Z",
                ],
                utc=True,
            ),
        )

        validator = TimeIndexValidator()

        with self.assertRaises(TimeIndexValidationError):
            validator.validate(frame, "5m", allow_gaps=False)

        report = validator.validate(frame, "5m", allow_gaps=True)
        self.assertEqual(report.gap_count, 1)
        self.assertTrue(report.is_sorted)
        self.assertTrue(report.is_unique)

