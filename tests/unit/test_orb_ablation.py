from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hybrid_quant.baseline.orb_ablation import OrbAblationRunner, load_orb_ablation_config


def _orb_frame() -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    session_days = pd.date_range("2024-01-02", periods=3, freq="D", tz="UTC")
    for day_index, day in enumerate(session_days):
        session_start = day + pd.Timedelta(hours=13, minutes=30)
        session_index = pd.date_range(session_start, periods=60, freq="5min", tz="UTC")
        session_base = 100.0 + (day_index * 1.0)
        for bar_index, timestamp in enumerate(session_index):
            if bar_index < 6:
                close = session_base + (0.12 * math.sin(bar_index))
                high = session_base + 0.9
                low = session_base - 0.9
                volume = 100 + bar_index
            elif day_index == len(session_days) - 1 and bar_index == 6:
                close = session_base + 1.6
                high = close + 0.8
                low = close - 0.6
                volume = 320.0
            else:
                close = session_base + 0.7 + (0.08 * (bar_index - 6))
                high = close + 0.7
                low = close - 0.6
                volume = 150 + (bar_index % 6) * 5

            rows.append(
                {
                    "open_time": timestamp,
                    "open": close - 0.2,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

    frame = pd.DataFrame(rows).set_index("open_time")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


class OrbAblationRunnerTests(unittest.TestCase):
    def test_runner_generates_expected_artifacts_for_small_matrix(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        experiment_yaml = """
name: orb_ablation_test
base_variant: baseline_nq_orb
dimensions:
  - key: opening_range_minutes
    options:
      - key: orb15
        label: OR 15m
        value: 15
        overrides:
          strategy:
            opening_range_minutes: 15
      - key: orb30
        label: OR 30m
        value: 30
        overrides:
          strategy:
            opening_range_minutes: 30
  - key: entry_mode
    options:
      - key: close
        label: Close
        value: breakout_close_entry
        overrides:
          strategy:
            entry_mode: breakout_close_entry
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            experiment_path = tmp_path / "orb_ablation_test.yaml"
            experiment_path.write_text(experiment_yaml, encoding="utf-8")

            runner = OrbAblationRunner(config_dir, load_orb_ablation_config(experiment_path))
            artifacts = runner.run(
                input_frame=_orb_frame(),
                output_dir=tmp_path / "orb_ablation",
                allow_gaps=True,
            )

            self.assertTrue(artifacts.comparison_path.exists())
            self.assertTrue(artifacts.results_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertEqual(len(artifacts.variant_artifacts), 2)
            self.assertTrue((artifacts.output_dir / "opening_range_summary.csv").exists())
            self.assertTrue((artifacts.output_dir / "entry_mode_summary.csv").exists())

            payload = json.loads(artifacts.comparison_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment_name"], "orb_ablation_test")
            self.assertEqual(len(payload["variants"]), 2)
            self.assertIn("best_robustness_variant", payload["conclusion"])
