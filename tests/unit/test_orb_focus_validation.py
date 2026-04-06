from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hybrid_quant.baseline.orb_focus_validation import (
    OrbFocusValidationRunner,
    load_orb_focus_validation_config,
)


def _orb_frame() -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    session_days = pd.date_range("2024-01-02", periods=5, freq="D", tz="UTC")
    for day_index, day in enumerate(session_days):
        session_start = day + pd.Timedelta(hours=13, minutes=30)
        session_index = pd.date_range(session_start, periods=90, freq="5min", tz="UTC")
        session_base = 100.0 + (day_index * 1.2)
        for bar_index, timestamp in enumerate(session_index):
            if bar_index < 6:
                close = session_base + (0.15 * math.sin(bar_index))
                high = session_base + 1.0 + (bar_index % 2) * 0.1
                low = session_base - 1.0 - (bar_index % 2) * 0.1
                volume = 100 + bar_index
            elif day_index in {2, 4} and bar_index in {6, 9}:
                direction = 1.0 if day_index == 4 else -1.0
                close = session_base + (1.75 * direction)
                high = close + 0.9
                low = close - 0.8
                volume = 320.0
            else:
                drift = 0.18 if day_index == 4 else 0.05
                offset = 1.25 if day_index == 4 else 0.7
                close = session_base + offset + (drift * (bar_index - 6))
                high = close + 0.9
                low = close - 0.7
                volume = 180 + (bar_index % 5) * 8

            rows.append(
                {
                    "open_time": timestamp,
                    "open": close - 0.3,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

    frame = pd.DataFrame(rows).set_index("open_time")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


class OrbFocusValidationRunnerTests(unittest.TestCase):
    def test_runner_generates_expected_artifacts_for_small_matrix(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        experiment_yaml = """
name: orb_focus_validation_test
base_variant: orb30_close_multi_no_slope_no_rvol
summary_thresholds:
  minimum_trades_total: 1
  minimum_trades_per_year: 0.1
variants:
  - name: reference
    label: Reference
    candidate: true
    overrides: {}
  - name: tighter_extension
    label: Tighter extension
    candidate: true
    overrides:
      strategy:
        max_breakout_distance_atr: 0.25
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            experiment_path = tmp_path / "orb_focus_validation_test.yaml"
            experiment_path.write_text(experiment_yaml, encoding="utf-8")

            runner = OrbFocusValidationRunner(
                config_dir,
                load_orb_focus_validation_config(experiment_path),
            )
            artifacts = runner.run(
                input_frame=_orb_frame(),
                output_dir=tmp_path / "orb_focus_validation",
                allow_gaps=True,
            )

            self.assertTrue(artifacts.comparison_path.exists())
            self.assertTrue(artifacts.results_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.temporal_blocks_path.exists())
            self.assertEqual(len(artifacts.variant_artifacts), 2)
            self.assertTrue((artifacts.output_dir / "yearly_variant_summary.csv").exists())
            self.assertTrue((artifacts.output_dir / "quarterly_variant_summary.csv").exists())

            payload = json.loads(artifacts.comparison_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment_name"], "orb_focus_validation_test")
            self.assertEqual(len(payload["variants"]), 2)
            self.assertIn("best_stability_variant", payload["conclusion"])
