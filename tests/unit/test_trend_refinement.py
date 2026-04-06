from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hybrid_quant.baseline.trend_refinement import main


def _trend_frame() -> pd.DataFrame:
    start = pd.Timestamp("2024-01-02T13:30:00Z")
    index = pd.date_range(start, periods=12 * 24 * 8, freq="5min", tz="UTC")
    rows = []
    for i, timestamp in enumerate(index):
        base = 17000.0 + (0.08 * i) + (8.0 * math.sin(i / 12.0))
        breakout_push = 0.0
        if timestamp.hour in {14, 15, 16} and timestamp.minute == 0:
            breakout_push -= 18.0
        if timestamp.hour in {17, 19} and timestamp.minute == 0:
            breakout_push += 10.0
        close = base + breakout_push
        open_ = close - 1.5
        high = close + 4.0
        low = close - 4.0
        rows.append(
            {
                "open_time": timestamp,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 250.0,
            }
        )
    frame = pd.DataFrame(rows).set_index("open_time")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


class TrendRefinementRunnerTests(unittest.TestCase):
    def test_trend_refinement_runner_generates_expected_alias_artifacts(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "nq.csv"
            output_dir = tmp_path / "refinement"
            _trend_frame().to_csv(input_path)

            exit_code = main(
                [
                    "--config-dir",
                    str(config_dir),
                    "--input-path",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--allow-gaps",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "trend_nasdaq_refinement_comparison.json").exists())
            self.assertTrue((output_dir / "trend_nasdaq_refinement_summary.md").exists())
            self.assertTrue((output_dir / "baseline_trend_nasdaq_report.json").exists())
            self.assertTrue((output_dir / "baseline_trend_nasdaq_v2_report.json").exists())
            self.assertTrue((output_dir / "baseline_trend_nasdaq_v2_hourly_breakdown.csv").exists())
            self.assertTrue((output_dir / "baseline_trend_nasdaq_v2_side_breakdown.csv").exists())

            payload = json.loads(
                (output_dir / "trend_nasdaq_refinement_comparison.json").read_text(encoding="utf-8")
            )
            self.assertIn("baseline_trend_nasdaq", payload["variants"])
            self.assertIn("baseline_trend_nasdaq_v2", payload["variants"])
            self.assertIn("baseline_trend_nasdaq_v2_long_only", payload["variants"])
            self.assertIn("baseline_trend_nasdaq_v2_short_only", payload["variants"])
