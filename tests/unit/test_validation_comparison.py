from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hybrid_quant.validation.comparison import RobustnessComparisonRunner


def _report(
    *,
    start: str,
    end: str,
    bars: int,
    classification: str,
    full_trades: int,
    full_net_pnl: float,
    walk_forward_test_trades: int,
    walk_forward_positive_ratio: float,
    walk_forward_mean_test_sharpe: float,
    temporal_positive_ratio: float,
    failed_caution_checks: list[str],
    failed_go_checks: list[str],
) -> dict:
    return {
        "variant": "baseline_v3",
        "dataset": {
            "symbol": "BTCUSDT",
            "execution_timeframe": "5m",
            "filter_timeframe": "1H",
            "start": start,
            "end": end,
            "bars": bars,
        },
        "limitations": [],
        "full_dataset": {
            "metrics": {
                "number_of_trades": full_trades,
                "gross_pnl": full_net_pnl + 100.0,
                "net_pnl": full_net_pnl,
                "fees_paid": 100.0,
                "sharpe": 0.25,
                "max_drawdown": 0.02,
            }
        },
        "walk_forward": {
            "summary": {
                "test_windows": 3,
                "positive_test_windows": int(round(3 * walk_forward_positive_ratio)),
                "positive_test_window_ratio": walk_forward_positive_ratio,
                "total_test_trades": walk_forward_test_trades,
                "total_test_net_pnl": full_net_pnl / 2.0,
                "mean_test_sharpe": walk_forward_mean_test_sharpe,
            }
        },
        "temporal_blocks": {
            "summary": {
                "blocks": 6,
                "positive_blocks": int(round(6 * temporal_positive_ratio)),
                "positive_block_ratio": temporal_positive_ratio,
            }
        },
        "monte_carlo": {
            "max_drawdown": {"p95": 0.03},
        },
        "cost_sensitivity": {
            "summary": {
                "survival_ratio": 0.5,
            }
        },
        "decision": {
            "classification": classification,
            "failed_caution_checks": failed_caution_checks,
            "failed_go_checks": failed_go_checks,
        },
    }


class RobustnessComparisonRunnerTests(unittest.TestCase):
    def test_compare_reports_detects_strengthened_evidence(self) -> None:
        runner = RobustnessComparisonRunner()
        q1_report = _report(
            start="2024-01-01T00:00:00+00:00",
            end="2024-03-31T23:55:00+00:00",
            bars=26208,
            classification="NO-GO",
            full_trades=14,
            full_net_pnl=466.13,
            walk_forward_test_trades=8,
            walk_forward_positive_ratio=0.33,
            walk_forward_mean_test_sharpe=-0.66,
            temporal_positive_ratio=0.33,
            failed_caution_checks=["total_test_trades", "mean_test_sharpe"],
            failed_go_checks=["total_test_trades"],
        )
        extended_report = _report(
            start="2023-01-01T00:00:00+00:00",
            end="2024-12-31T23:55:00+00:00",
            bars=210240,
            classification="GO WITH CAUTION",
            full_trades=84,
            full_net_pnl=2100.0,
            walk_forward_test_trades=36,
            walk_forward_positive_ratio=0.50,
            walk_forward_mean_test_sharpe=0.10,
            temporal_positive_ratio=0.50,
            failed_caution_checks=[],
            failed_go_checks=["positive_test_window_ratio"],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            baseline_path = tmp_path / "q1.json"
            extended_path = tmp_path / "extended.json"
            baseline_path.write_text(json.dumps(q1_report), encoding="utf-8")
            extended_path.write_text(json.dumps(extended_report), encoding="utf-8")

            artifacts = runner.compare(
                baseline_report_path=baseline_path,
                extended_report_path=extended_path,
                output_dir=tmp_path / "comparison",
            )

            self.assertTrue(artifacts.comparison_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertEqual(
                artifacts.comparison["conclusion"]["recommended_status"],
                "GO WITH CAUTION",
            )
            self.assertTrue(artifacts.comparison["conclusion"]["evidence_strengthened"])
