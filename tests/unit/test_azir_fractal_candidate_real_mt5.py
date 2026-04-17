import csv
import unittest
from pathlib import Path

from hybrid_quant.azir.event_log import AZIR_EVENT_COLUMNS
from hybrid_quant.azir.fractal_candidate_real_mt5 import (
    inspect_event_log,
    read_canonical_event_log,
    run_real_mt5_fractal_candidate_comparison,
)
from tests.unit.test_azir_best_setup_candidate import _write_minimal_mt5_log
from tests.unit.test_azir_setup_research import _bars, _write_ohlcv


def _write_candidate_setup_only_log(path: Path) -> None:
    row = {column: "" for column in AZIR_EVENT_COLUMNS}
    row.update(
        {
            "timestamp": "2025.01.06 16:30:00",
            "event_id": "2025-01-06_XAUUSD_123456321",
            "event_type": "opportunity",
            "symbol": "XAUUSD",
            "magic": "123456321",
            "day_of_week": "1",
            "is_friday": "false",
            "server_time": "2025.01.06 16:30:00",
            "timeframe": "M5",
            "ny_open_hour": "16",
            "ny_open_minute": "30",
            "close_hour": "22",
            "swing_bars": "10",
            "swing_high": "100.5",
            "swing_low": "99.0",
            "buy_entry": "100.55",
            "sell_entry": "98.95",
            "pending_distance_points": "160",
            "ema20": "100",
            "prev_close": "100.2",
            "prev_close_above_ema20": "true",
            "atr": "1.0",
            "atr_points": "100",
            "atr_filter_enabled": "true",
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AZIR_EVENT_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


class AzirFractalCandidateRealMT5Tests(unittest.TestCase):
    def test_schema_inspection_marks_setup_only_log_as_not_economic(self) -> None:
        root = Path("artifacts") / "unit-test-azir-fractal-real-mt5"
        root.mkdir(parents=True, exist_ok=True)
        candidate_path = root / "candidate.csv"
        _write_candidate_setup_only_log(candidate_path)

        rows = read_canonical_event_log(candidate_path, symbol_filter="XAUUSD")
        schema = inspect_event_log(candidate_path, rows)

        self.assertTrue(schema["schema_compatible"])
        self.assertTrue(schema["usable_for_setup_parity"])
        self.assertFalse(schema["usable_for_economic_audit"])
        self.assertEqual(schema["symbols"], {"XAUUSD": 1})

    def test_runner_compares_different_current_and_candidate_symbols(self) -> None:
        root = Path("artifacts") / "unit-test-azir-fractal-real-mt5"
        root.mkdir(parents=True, exist_ok=True)
        current_path = root / "current.csv"
        candidate_path = root / "candidate_runner.csv"
        m5_path = root / "m5.csv"
        output_dir = root / "out"
        _write_minimal_mt5_log(current_path)
        _write_candidate_setup_only_log(candidate_path)
        _write_ohlcv(m5_path, _bars())

        report = run_real_mt5_fractal_candidate_comparison(
            current_log_path=current_path,
            candidate_log_path=candidate_path,
            m5_input_path=m5_path,
            output_dir=output_dir,
            current_symbol="XAUUSD-STD",
            candidate_symbol="XAUUSD",
        )

        self.assertEqual(report["sources"]["candidate_symbol"], "XAUUSD")
        self.assertFalse(report["economics"]["candidate_fractal_observed"]["available"])
        self.assertFalse(report["readiness"]["strict_symbol_match"])
        self.assertFalse(report["readiness"]["may_become_baseline_azir_economic_candidate_fractal_v1"])
        self.assertFalse(report["readiness"]["may_freeze_protected_candidate_benchmark_now"])
        self.assertTrue((output_dir / "fractal_candidate_real_mt5_report.json").exists())
        self.assertTrue((output_dir / "azir_vs_fractal_real_day_by_day.csv").exists())


if __name__ == "__main__":
    unittest.main()
