import unittest
from pathlib import Path

from hybrid_quant.azir.fractal_candidate_export import (
    CANDIDATE_NAME,
    build_candidate_export_parity_rows,
    run_fractal_candidate_export_validation,
)
from hybrid_quant.azir.replica import AzirPythonReplica, AzirReplicaConfig
from tests.unit.test_azir_best_setup_candidate import _write_minimal_mt5_log
from tests.unit.test_azir_setup_research import _bars, _write_ohlcv, _write_protected_report


class AzirFractalCandidateExportTests(unittest.TestCase):
    def test_runner_generates_equivalent_export_when_mt5_candidate_log_is_missing(self) -> None:
        root = Path("artifacts") / "unit-test-azir-fractal-candidate-export"
        root.mkdir(parents=True, exist_ok=True)
        mt5_path = root / "mt5_current.csv"
        m5_path = root / "m5.csv"
        protected_path = root / "protected.json"
        output_dir = root / "out_missing_candidate"
        _write_minimal_mt5_log(mt5_path)
        _write_ohlcv(m5_path, _bars())
        _write_protected_report(protected_path)

        report = run_fractal_candidate_export_validation(
            mt5_log_path=mt5_path,
            m5_input_path=m5_path,
            output_dir=output_dir,
            protected_report_path=protected_path,
            symbol="XAUUSD-STD",
        )

        self.assertEqual(report["candidate_name"], CANDIDATE_NAME)
        self.assertFalse(report["candidate_evidence"]["is_real_mt5_candidate_log"])
        self.assertFalse(report["readiness"]["may_replace_azir_now"])
        self.assertTrue((output_dir / "fractal_candidate_event_log.csv").exists())
        self.assertTrue((output_dir / "fractal_candidate_mt5_report.json").exists())
        self.assertTrue((output_dir / "azir_vs_fractal_mt5_day_by_day.csv").exists())
        self.assertTrue((output_dir / "fractal_candidate_readiness_assessment.md").exists())

    def test_runner_marks_real_candidate_log_when_provided(self) -> None:
        root = Path("artifacts") / "unit-test-azir-fractal-candidate-export"
        root.mkdir(parents=True, exist_ok=True)
        mt5_path = root / "mt5_current_real_candidate.csv"
        candidate_path = root / "mt5_candidate.csv"
        m5_path = root / "m5_real_candidate.csv"
        output_dir = root / "out_real_candidate"
        _write_minimal_mt5_log(mt5_path)
        _write_minimal_mt5_log(candidate_path)
        _write_ohlcv(m5_path, _bars())

        report = run_fractal_candidate_export_validation(
            mt5_log_path=mt5_path,
            m5_input_path=m5_path,
            output_dir=output_dir,
            candidate_log_path=candidate_path,
            symbol="XAUUSD-STD",
        )

        self.assertTrue(report["candidate_evidence"]["is_real_mt5_candidate_log"])
        self.assertEqual(report["candidate_evidence"]["source_type"], "real_mt5_candidate_event_log")

    def test_candidate_export_parity_detects_matching_python_export(self) -> None:
        events = AzirPythonReplica(
            _bars(),
            AzirReplicaConfig(
                symbol="XAUUSD-STD",
                swing_bars=10,
                swing_definition="fractal",
                fractal_side_bars=2,
            ),
        ).run()

        rows = build_candidate_export_parity_rows(events, events)

        self.assertTrue(rows)
        self.assertTrue(all(float(row["field_match_pct"]) == 100.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
