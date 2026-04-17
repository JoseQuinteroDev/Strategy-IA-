import csv
import json
import unittest
from pathlib import Path

from hybrid_quant.azir.event_log import AZIR_EVENT_COLUMNS
from hybrid_quant.azir.fractal_full_lifecycle_export import (
    FULL_LIFECYCLE_LOG_NAME,
    inspect_event_log,
    run_fractal_full_lifecycle_export_assessment,
)


def _write_event_log(path: Path, event_types: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AZIR_EVENT_COLUMNS)
        writer.writeheader()
        for event_type in event_types:
            row = {column: "" for column in AZIR_EVENT_COLUMNS}
            row.update(
                {
                    "timestamp": "2025.01.06 16:30:00",
                    "event_id": "2025-01-06_XAUUSD-STD_123456321",
                    "event_type": event_type,
                    "symbol": "XAUUSD-STD",
                    "magic": "123456321",
                }
            )
            writer.writerow(row)


class AzirFractalFullLifecycleExportTests(unittest.TestCase):
    def test_inspect_event_log_marks_fill_exit_lifecycle(self) -> None:
        root = Path("artifacts") / "unit-test-fractal-full-lifecycle"
        log_path = root / FULL_LIFECYCLE_LOG_NAME
        _write_event_log(log_path, ["opportunity", "fill", "trailing_modified", "exit"])

        inspection = inspect_event_log(log_path, "XAUUSD-STD")

        self.assertTrue(inspection["schema_compatible"])
        self.assertEqual(inspection["event_counts"]["fill"], 1)
        self.assertEqual(inspection["event_counts"]["exit"], 1)

    def test_runner_reports_exporter_ready_when_lifecycle_log_is_missing(self) -> None:
        root = Path("artifacts") / "unit-test-fractal-full-lifecycle-runner"
        current_log = root / "todos_los_ticks.csv"
        candidate_log = root / "fractal_candidate_event_log.csv"
        m5_path = root / "xauusd_m5.csv"
        m1_path = root / "xauusd_m1.csv"
        tick_path = root / "tick_level.csv"
        forced_close = root / "forced_close_revaluation_report.json"
        protected = root / "fractal_protected_economic_report.json"
        tick_report = root / "fractal_tick_replay_report.json"
        ea_path = root / "AzirFractalCandidateFullLifecycleExport.mq5"
        logger_path = root / "AzirEventLogger.mqh"
        output_dir = root / "out"

        _write_event_log(current_log, ["opportunity", "fill", "exit"])
        _write_event_log(candidate_log, ["opportunity"])
        for path in (m5_path, m1_path, tick_path, ea_path, logger_path):
            path.write_text("placeholder\n", encoding="utf-8")
        forced_close.write_text("{}", encoding="utf-8")
        protected.write_text("{}", encoding="utf-8")
        tick_report.write_text(
            json.dumps(
                {
                    "coverage": {
                        "closed_trades_priced": 866,
                        "tick_priced_trades": 342,
                        "m1_fallback_trades": 103,
                        "m5_fallback_trades": 421,
                        "unpriced_trades": 0,
                    },
                    "candidate_tick_metrics": {"net_pnl": 5248.7},
                }
            ),
            encoding="utf-8",
        )

        report = run_fractal_full_lifecycle_export_assessment(
            current_log_path=current_log,
            candidate_setup_log_path=candidate_log,
            m5_input_path=m5_path,
            m1_input_path=m1_path,
            tick_input_path=tick_path,
            forced_close_report_path=forced_close,
            fractal_protected_report_path=protected,
            fractal_tick_replay_report_path=tick_report,
            mql5_ea_path=ea_path,
            mql5_logger_path=logger_path,
            output_dir=output_dir,
            symbol="XAUUSD-STD",
        )

        self.assertEqual(report["readiness"]["status"], "exporter_ready_waiting_for_mt5_run")
        self.assertFalse(report["decision"]["candidate_promoted"])
        self.assertTrue((output_dir / "fractal_full_lifecycle_export_report.json").exists())
        self.assertTrue((output_dir / "fractal_tick_gap_closure_report.csv").exists())
        self.assertTrue((output_dir / "fractal_lifecycle_schema.md").exists())


if __name__ == "__main__":
    unittest.main()
