from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hybrid_quant.core import apply_settings_overrides, load_settings
from hybrid_quant.validation.robustness import RobustnessValidationRunner


def _synthetic_frame(days: int = 14) -> pd.DataFrame:
    index = pd.date_range("2024-01-01T00:00:00Z", periods=12 * 24 * days, freq="5min", tz="UTC")
    rows = []
    for idx, timestamp in enumerate(index):
        base = 100.0 + (0.01 * idx)
        intraday_wave = 1.2 * ((idx % 36) / 36.0 - 0.5)
        shock = 0.0
        if timestamp.hour in {12, 16, 19, 22} and timestamp.minute == 0:
            shock = -3.5 if timestamp.day % 2 == 0 else 3.5
        close = base + intraday_wave + shock
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


def _runner() -> RobustnessValidationRunner:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    settings = load_settings(config_dir)
    overridden = apply_settings_overrides(
        settings,
        {
            "validation": {
                "walk_forward_splits": 2,
                "walk_forward_train_ratio": 0.5,
                "walk_forward_validation_ratio": 0.25,
                "walk_forward_test_ratio": 0.25,
                "monte_carlo_simulations": 20,
                "monte_carlo_seed": 17,
                "cost_scenarios": [
                    {"name": "base", "fee_multiplier": 1.0, "slippage_multiplier": 1.0},
                    {"name": "fees_x2", "fee_multiplier": 2.0, "slippage_multiplier": 1.0},
                    {"name": "slippage_x2", "fee_multiplier": 1.0, "slippage_multiplier": 2.0},
                ],
            }
        },
    )
    return RobustnessValidationRunner(config_dir, variant_name="baseline_v3", settings=overridden)


class RobustnessValidationRunnerTests(unittest.TestCase):
    def test_monte_carlo_is_reproducible_with_fixed_seed(self) -> None:
        runner = _runner()
        trade_frame = pd.DataFrame({"net_pnl": [100.0, -40.0, 80.0, -20.0], "gross_pnl": [100.0, -40.0, 80.0, -20.0], "fees_paid": [0.0, 0.0, 0.0, 0.0]})

        first = runner._run_monte_carlo(trade_frame)
        second = runner._run_monte_carlo(trade_frame)

        self.assertEqual(first, second)
        self.assertEqual(first["simulations"], 20)
        self.assertIn("p95", first["max_drawdown"])

    def test_cost_sensitivity_reflects_higher_fee_stress(self) -> None:
        runner = _runner()
        frame = _synthetic_frame()

        results, summary = runner._run_cost_sensitivity(frame)

        self.assertEqual(set(results["scenario"]), {"base", "fees_x2", "slippage_x2"})
        base_row = results.loc[results["scenario"] == "base"].iloc[0]
        fees_row = results.loc[results["scenario"] == "fees_x2"].iloc[0]
        self.assertGreaterEqual(fees_row["fees_paid"], base_row["fees_paid"])
        self.assertIn("survival_ratio", summary)

    def test_runner_generates_robustness_artifacts(self) -> None:
        runner = _runner()
        frame = _synthetic_frame(days=21)

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = runner.run(
                output_dir=Path(tmp_dir) / "robustness",
                input_frame=frame,
                artifact_suffix="extended",
            )

            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.walk_forward_results_path.exists())
            self.assertTrue(artifacts.temporal_block_results_path.exists())
            self.assertTrue(artifacts.monte_carlo_summary_path.exists())
            self.assertTrue(artifacts.cost_sensitivity_path.exists())
            self.assertTrue((Path(tmp_dir) / "robustness" / "robustness_report_extended.json").exists())
            self.assertTrue((Path(tmp_dir) / "robustness" / "walk_forward_results_extended.csv").exists())

            report = artifacts.report
            self.assertIn("walk_forward", report)
            self.assertIn("monte_carlo", report)
            self.assertIn("cost_sensitivity", report)
            self.assertIn(report["decision"]["classification"], {"GO", "GO WITH CAUTION", "NO-GO"})
            self.assertIn("failed_go_checks", report["decision"])
            self.assertIn("note", report["cost_sensitivity"]["summary"])
