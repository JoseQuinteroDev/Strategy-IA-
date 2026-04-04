from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hybrid_quant.baseline.comparison import BaselineComparisonRunner


def _synthetic_frame() -> pd.DataFrame:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    index = pd.date_range(start, periods=12 * 24 * 8, freq="5min", tz="UTC")
    rows = []
    for i, timestamp in enumerate(index):
        base = 100.0 + (0.02 * i) + (1.2 * math.sin(i / 18.0))
        shock = 0.0
        if timestamp.hour == 13 and timestamp.minute == 0:
            shock -= 4.0
        if timestamp.hour == 12 and timestamp.minute == 0:
            shock -= 3.0
        if timestamp.dayofweek >= 5 and timestamp.hour == 12 and timestamp.minute == 0:
            shock -= 2.5

        close = base + shock
        open_ = close - 0.2
        high = max(open_, close) + 0.5
        low = min(open_, close) - 0.5
        rows.append(
            {
                "open_time": timestamp,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000.0,
            }
        )

    frame = pd.DataFrame(rows).set_index("open_time")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


class BaselineComparisonRunnerTests(unittest.TestCase):
    def test_comparison_runner_generates_expected_artifacts(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        runner = BaselineComparisonRunner(config_dir)
        frame = _synthetic_frame()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "comparison"
            artifacts = runner.run(
                output_dir=output_dir,
                input_frame=frame,
                variants=("baseline_v1", "baseline_v2", "baseline_v3"),
                oos_start=frame.index[-288].to_pydatetime(),
                oos_end=frame.index[-1].to_pydatetime(),
            )

            self.assertTrue(artifacts.comparison_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue((output_dir / "baseline_v1_v2_v3_comparison.json").exists())
            self.assertTrue((output_dir / "baseline_v1_v2_v3_summary.md").exists())
            self.assertTrue((output_dir / "baseline_v1_report.json").exists())
            self.assertTrue((output_dir / "baseline_v2_report.json").exists())
            self.assertTrue((output_dir / "baseline_v3_report.json").exists())
            self.assertTrue((output_dir / "baseline_v2_monthly_breakdown.csv").exists())
            self.assertTrue((output_dir / "baseline_v2_hourly_breakdown.csv").exists())
            self.assertTrue((output_dir / "baseline_v2_exit_reason_breakdown.csv").exists())
            self.assertTrue((output_dir / "baseline_v2_side_breakdown.csv").exists())
            self.assertTrue((output_dir / "baseline_v3_monthly_breakdown.csv").exists())
            self.assertTrue((output_dir / "baseline_v3_hourly_breakdown.csv").exists())
            self.assertIsNotNone(artifacts.oos_comparison_path)
            self.assertTrue(artifacts.oos_comparison_path.exists())
            self.assertTrue(artifacts.oos_summary_path.exists())

            payload = json.loads(artifacts.comparison_path.read_text(encoding="utf-8"))
            self.assertIn("baseline_v1", payload["variants"])
            self.assertIn("baseline_v2", payload["variants"])
            self.assertIn("baseline_v3", payload["variants"])

            baseline_v1 = payload["variants"]["baseline_v1"]
            baseline_v2 = payload["variants"]["baseline_v2"]
            baseline_v3 = payload["variants"]["baseline_v3"]
            self.assertGreater(baseline_v1["number_of_trades"], 0)
            self.assertLessEqual(baseline_v2["number_of_trades"], baseline_v1["number_of_trades"])
            self.assertLessEqual(baseline_v3["number_of_trades"], baseline_v2["number_of_trades"])

            required_keys = {
                "number_of_trades",
                "win_rate",
                "average_win",
                "average_loss",
                "payoff",
                "expectancy",
                "gross_pnl",
                "net_pnl",
                "fees_paid",
                "slippage_impact_estimate",
                "max_drawdown",
                "sharpe",
                "sortino",
                "calmar",
                "profitable_months_pct",
                "max_consecutive_losses",
            }
            self.assertTrue(required_keys.issubset(set(baseline_v1)))
            self.assertTrue(required_keys.issubset(set(baseline_v2)))
            self.assertTrue(required_keys.issubset(set(baseline_v3)))
            self.assertIn("baseline_v3_vs_baseline_v2", payload["pair_deltas"])

            summary = artifacts.summary_path.read_text(encoding="utf-8")
            self.assertIn("baseline_v3_vs_baseline_v2", summary)

    def test_comparison_runner_supports_generic_trend_variant(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        runner = BaselineComparisonRunner(config_dir)
        frame = _synthetic_frame()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "comparison_generic"
            artifacts = runner.run(
                output_dir=output_dir,
                input_frame=frame,
                variants=("baseline_v3", "baseline_trend_nasdaq"),
                allow_gaps=True,
            )

            payload = json.loads(artifacts.comparison_path.read_text(encoding="utf-8"))
            self.assertIn("baseline_v3", payload["variants"])
            self.assertIn("baseline_trend_nasdaq", payload["variants"])
            self.assertIn("baseline_trend_nasdaq_vs_baseline_v3", payload["pair_deltas"])
            self.assertEqual(payload["period"]["symbol"], "mixed_variants")

            summary = artifacts.summary_path.read_text(encoding="utf-8")
            self.assertIn("baseline_trend_nasdaq", summary)
            self.assertIn("Key Takeaways", summary)
