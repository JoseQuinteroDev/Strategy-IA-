from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hybrid_quant.baseline.analyze import main


def _trend_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-02T13:30:00Z", periods=78 * 4, freq="5min", tz="UTC")
    step = pd.Series(range(len(index)), dtype=float)
    close = 17000.0 + (step * 0.9)
    frame = pd.DataFrame(index=index)
    frame.index.name = "open_time"
    frame["open"] = close - 0.4
    frame["high"] = close + 1.7
    frame["low"] = close - 1.0
    frame["close"] = close
    frame["volume"] = 200.0
    return frame


class BaselineAnalyzeRunnerTests(unittest.TestCase):
    def test_analyze_cli_runs_trend_variant_and_writes_diagnostics(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "nq.csv"
            output_dir = tmp_path / "analysis"
            _trend_frame().to_csv(input_path)

            exit_code = main(
                [
                    "--config-dir",
                    str(config_dir),
                    "--variant",
                    "baseline_trend_nasdaq",
                    "--input-path",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--allow-gaps",
                ]
            )

            self.assertEqual(exit_code, 0)
            diagnostics_path = output_dir / "diagnostics" / "diagnostics.json"
            self.assertTrue((output_dir / "baseline" / "report.json").exists())
            self.assertTrue(diagnostics_path.exists())
            self.assertTrue((output_dir / "diagnostics" / "diagnostics_summary.md").exists())

            payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["baseline_metrics"]["strategy_family"], "trend_breakout")
            self.assertIn("momentum", payload["breakdowns"])
            self.assertIn("breakout_distance", payload["breakdowns"])
            self.assertIn("target_to_cost", payload["breakdowns"])
            self.assertIn("validation_verdict", payload["automatic_conclusion"])
