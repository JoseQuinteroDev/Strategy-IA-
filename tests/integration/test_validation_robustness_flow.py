from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from hybrid_quant.validation.robustness import main


def _synthetic_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01T00:00:00Z", periods=12 * 24 * 21, freq="5min", tz="UTC")
    rows = []
    for idx, timestamp in enumerate(index):
        base = 100.0 + (0.008 * idx)
        shock = 0.0
        if timestamp.hour in {12, 16, 19, 22} and timestamp.minute == 0:
            shock = -3.0 if timestamp.day % 2 == 0 else 3.0
        close = base + shock
        open_ = close - 0.15
        high = max(open_, close) + 0.45
        low = min(open_, close) - 0.45
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


class RobustnessValidationFlowTests(unittest.TestCase):
    def test_cli_runner_writes_expected_robustness_artifacts(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "ohlcv.csv"
            output_dir = tmp_path / "robustness"
            _synthetic_frame().to_csv(input_path)

            exit_code = main(
                [
                    "--config-dir",
                    str(config_dir),
                    "--input-path",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--variant",
                    "baseline_v3",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "robustness_report.json").exists())
            self.assertTrue((output_dir / "robustness_summary.md").exists())
            self.assertTrue((output_dir / "walk_forward_results.csv").exists())
            self.assertTrue((output_dir / "temporal_block_results.csv").exists())
            self.assertTrue((output_dir / "monte_carlo_summary.json").exists())
            self.assertTrue((output_dir / "cost_sensitivity.csv").exists())

            report = json.loads((output_dir / "robustness_report.json").read_text(encoding="utf-8"))
            self.assertIn("decision", report)
            self.assertIn(report["decision"]["classification"], {"GO", "GO WITH CAUTION", "NO-GO"})
            self.assertIn("failed_caution_checks", report["decision"])

    def test_cli_runner_supports_request_mode_without_input_path(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"

        class _StubRunner:
            def run(self, **kwargs):  # noqa: ANN003
                self.kwargs = kwargs
                output_dir = Path(kwargs["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                report_path = output_dir / "robustness_report.json"
                summary_path = output_dir / "robustness_summary.md"
                report_path.write_text('{"decision":{"classification":"NO-GO"}}', encoding="utf-8")
                summary_path.write_text("# stub\n", encoding="utf-8")
                return type(
                    "Artifacts",
                    (),
                    {
                        "report": {"decision": {"classification": "NO-GO"}},
                        "report_path": report_path,
                        "summary_path": summary_path,
                    },
                )()

        stub_runner = _StubRunner()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "robustness-request"
            with patch("hybrid_quant.validation.robustness.RobustnessValidationRunner.from_config", return_value=stub_runner):
                exit_code = main(
                    [
                        "--config-dir",
                        str(config_dir),
                        "--output-dir",
                        str(output_dir),
                        "--variant",
                        "baseline_v3",
                        "--start",
                        "2024-01-01T00:00:00+00:00",
                        "--end",
                        "2024-03-31T23:55:00+00:00",
                        "--allow-gaps",
                        "--artifact-suffix",
                        "extended",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("request", stub_runner.kwargs)
            self.assertIsNone(stub_runner.kwargs.get("input_frame"))
            self.assertEqual(stub_runner.kwargs["artifact_suffix"], "extended")
