import csv
import json
import unittest
from pathlib import Path

from hybrid_quant.azir.event_log import AZIR_EVENT_COLUMNS
from hybrid_quant.azir.fractal_lifecycle_final_comparison import (
    read_segmented_event_log,
    run_fractal_lifecycle_final_comparison,
    select_primary_segment,
)


def _row(event_type: str, timestamp: str, *, lot: str = "0.01", net: str = "", side: str = "buy") -> dict[str, str]:
    row = {column: "" for column in AZIR_EVENT_COLUMNS}
    day = timestamp[:10].replace(".", "-")
    row.update(
        {
            "timestamp": timestamp,
            "event_id": f"{day}_XAUUSD-STD_123456321",
            "event_type": event_type,
            "symbol": "XAUUSD-STD",
            "magic": "123456321",
            "day_of_week": "1",
            "is_friday": "false",
            "timeframe": "M5",
            "lot_size": lot if event_type == "opportunity" else "",
            "sl_points": "500" if event_type == "opportunity" else "",
            "tp_points": "500" if event_type == "opportunity" else "",
            "trailing_start_points": "90" if event_type == "opportunity" else "",
            "trailing_step_points": "50" if event_type == "opportunity" else "",
            "buy_order_placed": "true" if event_type == "opportunity" else "",
            "sell_order_placed": "false" if event_type == "opportunity" else "",
            "buy_entry": "2000.00" if event_type == "opportunity" else "",
            "sell_entry": "1900.00" if event_type == "opportunity" else "",
            "pending_distance_points": "10000" if event_type == "opportunity" else "",
            "fill_side": side if event_type in {"fill", "exit"} else "",
            "fill_price": "2000.00" if event_type in {"fill", "exit"} else "",
            "duration_to_fill_seconds": "60" if event_type == "fill" else "",
            "exit_reason": "take_profit" if event_type == "exit" else "",
            "gross_pnl": net if event_type == "exit" else "",
            "net_pnl": net if event_type == "exit" else "",
            "commission": "0.00" if event_type == "exit" else "",
            "swap": "0.00" if event_type == "exit" else "",
        }
    )
    return row


def _write_log(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AZIR_EVENT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class AzirFractalLifecycleFinalComparisonTests(unittest.TestCase):
    def test_segment_reader_detects_tester_timestamp_reset_and_selects_full_segment(self) -> None:
        root = Path("artifacts") / "unit-test-fractal-final-comparison-segments"
        log_path = root / "candidate.csv"
        _write_log(
            log_path,
            [
                _row("opportunity", "2025.01.02 16:30:00", lot="0.10"),
                _row("fill", "2025.01.02 16:31:00"),
                _row("exit", "2025.01.02 16:32:00", net="10.00"),
                _row("opportunity", "2024.01.02 16:30:00", lot="0.05"),
                _row("fill", "2024.01.02 16:31:00"),
                _row("exit", "2024.01.02 16:32:00", net="5.00"),
                _row("opportunity", "2024.01.03 16:30:00", lot="0.05"),
                _row("fill", "2024.01.03 16:31:00"),
                _row("exit", "2024.01.03 16:32:00", net="5.00"),
            ],
        )

        segments = read_segmented_event_log(log_path, "XAUUSD-STD")
        selected = select_primary_segment(segments)

        self.assertEqual(len(segments), 2)
        self.assertEqual(selected["segment_index"], 2)
        self.assertEqual(selected["primary_lot"], 0.05)

    def test_runner_keeps_baseline_when_normalized_candidate_net_is_lower(self) -> None:
        root = Path("artifacts") / "unit-test-fractal-final-comparison-runner"
        current = root / "todos_los_ticks.csv"
        candidate = root / "fractal_candidate_full_lifecycle_event_log.csv"
        setup = root / "fractal_candidate_event_log.csv"
        forced = root / "forced_close_revaluation_report.json"
        out = root / "out"

        _write_log(
            current,
            [
                _row("opportunity", "2025.01.02 16:30:00", lot="0.01"),
                _row("fill", "2025.01.02 16:31:00"),
                _row("exit", "2025.01.02 16:32:00", net="4.00"),
            ],
        )
        _write_log(
            candidate,
            [
                _row("opportunity", "2025.01.02 16:30:00", lot="0.05"),
                _row("fill", "2025.01.02 16:31:00"),
                _row("exit", "2025.01.02 16:32:00", net="15.00"),
            ],
        )
        _write_log(setup, [_row("opportunity", "2025.01.02 16:30:00", lot="0.05")])
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
        for name in ("xauusd_m5.csv", "xauusd_m1.csv", "tick_level.csv", "fractal_protected.json", "fractal_tick.json"):
            (root / name).write_text("placeholder\n", encoding="utf-8")

        report = run_fractal_lifecycle_final_comparison(
            current_log_path=current,
            candidate_full_lifecycle_log_path=candidate,
            candidate_setup_log_path=setup,
            m5_input_path=root / "xauusd_m5.csv",
            m1_input_path=root / "xauusd_m1.csv",
            tick_input_path=root / "tick_level.csv",
            forced_close_report_path=forced,
            fractal_protected_report_path=root / "fractal_protected.json",
            fractal_tick_replay_report_path=root / "fractal_tick.json",
            output_dir=out,
            symbol="XAUUSD-STD",
        )

        self.assertFalse(report["decision"]["promote_candidate"])
        self.assertEqual(report["metrics"]["candidate_protected_normalized_to_baseline_lot"]["net_pnl"], 3.0)
        self.assertTrue((out / "baseline_vs_fractal_protected_comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
