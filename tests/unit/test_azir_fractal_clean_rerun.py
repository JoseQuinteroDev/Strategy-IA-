import csv
import json
import unittest
from pathlib import Path

from hybrid_quant.azir.event_log import AZIR_EVENT_COLUMNS
from hybrid_quant.azir.fractal_clean_rerun import run_clean_rerun_gate, validate_clean_csv


def _row(event_type: str, timestamp: str, *, lot: str = "0.01", net: str = "1.00") -> dict[str, str]:
    day = timestamp[:10].replace(".", "-")
    row = {column: "" for column in AZIR_EVENT_COLUMNS}
    row.update(
        {
            "timestamp": timestamp,
            "event_id": f"{day}_XAUUSD-STD_123456321",
            "event_type": event_type,
            "symbol": "XAUUSD-STD",
            "magic": "123456321",
            "timeframe": "M5",
            "lot_size": lot if event_type == "opportunity" else "",
            "buy_order_placed": "true" if event_type == "opportunity" else "",
            "sell_order_placed": "false" if event_type == "opportunity" else "",
            "buy_entry": "2000.00" if event_type == "opportunity" else "",
            "fill_side": "buy" if event_type in {"fill", "exit"} else "",
            "fill_price": "2000.00" if event_type in {"fill", "exit"} else "",
            "exit_reason": "take_profit" if event_type == "exit" else "",
            "gross_pnl": net if event_type == "exit" else "",
            "net_pnl": net if event_type == "exit" else "",
        }
    )
    return row


def _write_log(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AZIR_EVENT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class AzirFractalCleanRerunTests(unittest.TestCase):
    def test_validate_clean_csv_accepts_single_segment_lot_001_lifecycle(self) -> None:
        root = Path("artifacts") / "unit-test-fractal-clean-rerun"
        clean = root / "fractal_candidate_full_lifecycle_event_log_clean_001.csv"
        _write_log(
            clean,
            [
                _row("opportunity", "2021.01.04 16:30:00"),
                _row("fill", "2021.01.04 16:31:00"),
                _row("trailing_modified", "2021.01.04 16:31:30"),
                _row("exit", "2025.12.30 16:30:45"),
            ],
        )

        report = validate_clean_csv(clean, "XAUUSD-STD")

        self.assertTrue(report["accepted"])
        self.assertEqual(report["segments_detected"], 1)
        self.assertEqual(report["lot_counts"], {"0.01": 1})

    def test_runner_writes_readiness_artifacts_when_clean_csv_is_accepted(self) -> None:
        root = Path("artifacts") / "unit-test-fractal-clean-rerun-runner"
        current = root / "todos_los_ticks.csv"
        clean = root / "fractal_candidate_full_lifecycle_event_log_clean_001.csv"
        setup = root / "fractal_candidate_event_log.csv"
        forced = root / "forced_close_revaluation_report.json"
        out = root / "out"
        rows = [
            _row("opportunity", "2021.01.04 16:30:00", net="4.00"),
            _row("fill", "2021.01.04 16:31:00", net="4.00"),
            _row("trailing_modified", "2021.01.04 16:31:30", net="4.00"),
            _row("exit", "2025.12.30 16:30:45", net="4.00"),
        ]
        _write_log(current, rows)
        _write_log(clean, rows)
        _write_log(setup, [_row("opportunity", "2021.01.04 16:30:00")])
        forced.write_text(
            json.dumps(
                {
                    "metrics": {
                        "azir_with_risk_engine_v1_forced_closes_revalued": {
                            "closed_trades": 1,
                            "net_pnl": 4.0,
                            "profit_factor": None,
                            "expectancy": 4.0,
                            "win_rate": 100.0,
                            "average_win": 4.0,
                            "average_loss": None,
                            "payoff": None,
                            "max_drawdown_abs": 0.0,
                            "max_consecutive_losses": 0,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        for name in ("xauusd_m5.csv", "xauusd_m1.csv", "tick_level.csv", "fractal_protected.json", "previous_final.json"):
            (root / name).write_text("placeholder\n", encoding="utf-8")

        report = run_clean_rerun_gate(
            current_log_path=current,
            clean_candidate_log_path=clean,
            candidate_setup_log_path=setup,
            m5_input_path=root / "xauusd_m5.csv",
            m1_input_path=root / "xauusd_m1.csv",
            tick_input_path=root / "tick_level.csv",
            forced_close_report_path=forced,
            fractal_protected_report_path=root / "fractal_protected.json",
            previous_final_report_path=root / "previous_final.json",
            output_dir=out,
            symbol="XAUUSD-STD",
        )

        self.assertTrue(report["validation"]["accepted"])
        self.assertTrue((out / "fractal_clean_rerun_requirements.md").exists())
        self.assertTrue((out / "fractal_clean_csv_validation_report.json").exists())
        self.assertTrue((out / "fractal_clean_vs_baseline_template.csv").exists())


if __name__ == "__main__":
    unittest.main()
