import csv
import json
import unittest
from pathlib import Path

from hybrid_quant.azir.event_log import AZIR_EVENT_COLUMNS
from hybrid_quant.azir.fractal_tick_replay import run_fractal_tick_replay


def _setup_row() -> dict[str, str]:
    row = {column: "" for column in AZIR_EVENT_COLUMNS}
    row.update(
        {
            "timestamp": "2025.01.06 16:30:00",
            "event_id": "2025-01-06_XAUUSD-STD_123456321",
            "event_type": "opportunity",
            "symbol": "XAUUSD-STD",
            "day_of_week": "1",
            "is_friday": "false",
            "server_time": "2025.01.06 16:30:00",
            "timeframe": "M5",
            "close_hour": "22",
            "swing_high": "100.00",
            "swing_low": "95.00",
            "buy_entry": "100.05",
            "sell_entry": "94.95",
            "pending_distance_points": "510",
            "atr_filter_passed": "true",
            "rsi_gate_required": "false",
            "rsi_gate_passed": "true",
            "buy_order_placed": "true",
            "sell_order_placed": "false",
        }
    )
    return row


def _write_event_log(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AZIR_EVENT_COLUMNS)
        writer.writeheader()
        writer.writerow(_setup_row())


def _write_ohlcv(path: Path) -> None:
    rows = [
        ["open_time", "open", "high", "low", "close", "volume"],
        ["2025-01-06 16:30:00", "99.90", "100.10", "99.80", "100.00", "1"],
        ["2025-01-06 16:31:00", "100.05", "105.20", "100.00", "105.00", "1"],
        ["2025-01-06 22:00:00", "104.00", "104.00", "104.00", "104.00", "1"],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _write_tick_csv(path: Path) -> None:
    rows = [
        ["time", "time_msc", "bid", "ask", "last", "volume", "volume_real", "flags"],
        ["2025-01-06 16:30:00", "1736181000000", "99.90", "100.00", "0.00", "0", "0.00000000", "4"],
        ["2025-01-06 16:30:10", "1736181010000", "99.95", "100.06", "0.00", "0", "0.00000000", "4"],
        ["2025-01-06 16:31:00", "1736181060000", "105.10", "105.20", "0.00", "0", "0.00000000", "4"],
        ["2025-01-06 22:00:00", "1736200800000", "104.00", "104.10", "0.00", "0", "0.00000000", "4"],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


class AzirFractalTickReplayTests(unittest.TestCase):
    def test_runner_prices_candidate_with_tick_data(self) -> None:
        root = Path("artifacts") / "unit-test-fractal-tick-replay"
        root.mkdir(parents=True, exist_ok=True)
        current_log = root / "current.csv"
        candidate_log = root / "candidate.csv"
        tick_path = root / "ticks.csv"
        m1_path = root / "m1.csv"
        m5_path = root / "m5.csv"
        protected_report = root / "protected.json"
        output_dir = root / "out"

        _write_event_log(current_log)
        _write_event_log(candidate_log)
        _write_tick_csv(tick_path)
        _write_ohlcv(m1_path)
        _write_ohlcv(m5_path)
        protected_report.write_text(
            json.dumps(
                {
                    "benchmark_name": "baseline_azir_protected_economic_v1",
                    "metrics": {
                        "azir_with_risk_engine_v1_forced_closes_revalued": {
                            "closed_trades": 1,
                            "net_pnl": 10.0,
                            "win_rate": 100.0,
                            "average_win": 10.0,
                            "average_loss": None,
                            "payoff": None,
                            "profit_factor": None,
                            "expectancy": 10.0,
                            "max_drawdown_abs": 0.0,
                            "max_consecutive_losses": 0,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        report = run_fractal_tick_replay(
            current_log_path=current_log,
            candidate_log_path=candidate_log,
            tick_input_path=tick_path,
            m1_input_path=m1_path,
            m5_input_path=m5_path,
            protected_report_path=protected_report,
            output_dir=output_dir,
            symbol="XAUUSD-STD",
        )

        self.assertEqual(report["coverage"]["tick_priced_trades"], 1)
        self.assertEqual(report["coverage"]["m1_fallback_trades"], 0)
        self.assertEqual(report["candidate_tick_metrics"]["closed_trades"], 1)
        self.assertTrue((output_dir / "fractal_tick_replay_report.json").exists())
        self.assertTrue((output_dir / "fractal_tick_priced_trades.csv").exists())


if __name__ == "__main__":
    unittest.main()
