import json
import unittest
from pathlib import Path

from hybrid_quant.azir.best_setup_candidate import (
    CANDIDATE_NAME,
    candidate_definition,
    run_best_setup_candidate_validation,
)
from hybrid_quant.azir.event_log import AZIR_EVENT_COLUMNS
from tests.unit.test_azir_setup_research import _bars, _write_ohlcv, _write_protected_report


def _write_minimal_mt5_log(path: Path) -> None:
    header = list(AZIR_EVENT_COLUMNS)
    row = {key: "" for key in header}
    row.update(
        {
            "timestamp": "2025.01.06 16:30:00",
            "event_id": "2025-01-06_XAUUSD-STD_123456321",
            "event_type": "opportunity",
            "symbol": "XAUUSD-STD",
            "magic": "123456321",
            "day_of_week": "1",
            "is_friday": "false",
            "server_time": "2025.01.06 16:30:00",
            "timeframe": "M5",
            "ny_open_hour": "16",
            "ny_open_minute": "30",
            "close_hour": "22",
            "swing_bars": "10",
            "lot_size": "0.1",
            "sl_points": "500",
            "tp_points": "500",
            "trailing_start_points": "90",
            "trailing_step_points": "50",
            "swing_high": "101.0",
            "swing_low": "99.0",
            "buy_entry": "101.05",
            "sell_entry": "98.95",
            "pending_distance_points": "210",
            "ema20": "100",
            "prev_close": "100.5",
            "prev_close_above_ema20": "true",
            "atr": "1.0",
            "atr_points": "100",
            "atr_filter_enabled": "true",
            "atr_minimum": "100",
            "atr_filter_passed": "true",
            "rsi_gate_enabled": "true",
            "rsi_gate_required": "false",
            "rsi_gate_passed": "true",
            "allow_buys": "true",
            "allow_sells": "true",
            "trend_filter_enabled": "true",
            "buy_order_placed": "true",
            "sell_order_placed": "false",
        }
    )
    path.write_text(",".join(header) + "\n" + ",".join(str(row[key]) for key in header), encoding="utf-8")


class AzirBestSetupCandidateTests(unittest.TestCase):
    def test_candidate_definition_names_formal_candidate(self) -> None:
        definition = candidate_definition()

        self.assertEqual(definition["name"], CANDIDATE_NAME)
        self.assertIn("last 10", definition["swing_window"].lower())

    def test_runner_writes_candidate_artifacts(self) -> None:
        root = Path("artifacts") / "unit-test-azir-best-setup-candidate"
        root.mkdir(parents=True, exist_ok=True)
        mt5_path = root / "mt5.csv"
        m5_path = root / "m5.csv"
        protected_path = root / "protected.json"
        research_path = root / "research.json"
        output_dir = root / "out"
        _write_minimal_mt5_log(mt5_path)
        _write_ohlcv(m5_path, _bars())
        _write_protected_report(protected_path)
        research_path.write_text(json.dumps({"candidate_ranking": []}), encoding="utf-8")

        report = run_best_setup_candidate_validation(
            mt5_log_path=mt5_path,
            m5_input_path=m5_path,
            protected_report_path=protected_path,
            setup_research_report_path=research_path,
            output_dir=output_dir,
            symbol="XAUUSD-STD",
        )

        self.assertEqual(report["candidate_name"], CANDIDATE_NAME)
        self.assertFalse(report["readiness"]["may_replace_frozen_benchmark_now"])
        self.assertTrue((output_dir / "best_setup_candidate_report.json").exists())
        self.assertTrue((output_dir / "azir_vs_fractal_day_by_day.csv").exists())
        self.assertTrue((output_dir / "candidate_readiness_assessment.md").exists())


if __name__ == "__main__":
    unittest.main()
