import csv
import json
import unittest
from datetime import datetime
from pathlib import Path

from hybrid_quant.azir.event_log import AZIR_EVENT_COLUMNS
from hybrid_quant.azir.fractal_protected_economic import (
    run_fractal_protected_economic_candidate,
    simulate_setup_from_bars,
)
from hybrid_quant.azir.replica import AzirReplicaConfig, OhlcvBar
from hybrid_quant.risk.azir_state import AzirRiskConfig


def _candidate_row(timestamp: str = "2025.01.06 16:30:00") -> dict[str, str]:
    row = {column: "" for column in AZIR_EVENT_COLUMNS}
    row.update(
        {
            "timestamp": timestamp,
            "event_id": "2025-01-06_XAUUSD-STD_123456321",
            "event_type": "opportunity",
            "symbol": "XAUUSD-STD",
            "day_of_week": "1",
            "is_friday": "false",
            "server_time": timestamp,
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


def _write_log(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AZIR_EVENT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


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


class AzirFractalProtectedEconomicTests(unittest.TestCase):
    def test_simulate_setup_from_bars_prices_buy_take_profit(self) -> None:
        bars = [
            OhlcvBar(open_time=datetime(2025, 1, 6, 16, 30), open=99.9, high=100.1, low=99.8, close=100.0, volume=1),
            OhlcvBar(open_time=datetime(2025, 1, 6, 16, 31), open=100.05, high=105.2, low=100.0, close=105.0, volume=1),
            OhlcvBar(open_time=datetime(2025, 1, 6, 22, 0), open=104.0, high=104.0, low=104.0, close=104.0, volume=1),
        ]

        result = simulate_setup_from_bars(
            setup_row=_candidate_row(),
            bars=bars,
            pricing_source="m1_replay",
            replay_config=AzirReplicaConfig(symbol="XAUUSD-STD"),
            risk_config=AzirRiskConfig(),
        )

        self.assertTrue(result["has_exit"])
        self.assertEqual(result["fill_side"], "buy")
        self.assertEqual(result["exit_reason"], "take_profit")
        self.assertAlmostEqual(float(result["net_pnl"]), 50.0)

    def test_runner_writes_protected_candidate_artifacts(self) -> None:
        root = Path("artifacts") / "unit-test-fractal-protected-economic"
        root.mkdir(parents=True, exist_ok=True)
        current_log = root / "current.csv"
        candidate_log = root / "candidate.csv"
        m1_path = root / "m1.csv"
        m5_path = root / "m5.csv"
        protected_report = root / "protected.json"
        output_dir = root / "out"

        row = _candidate_row()
        _write_log(current_log, [row])
        _write_log(candidate_log, [row])
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

        report = run_fractal_protected_economic_candidate(
            current_log_path=current_log,
            candidate_log_path=candidate_log,
            m5_input_path=m5_path,
            m1_input_path=m1_path,
            protected_report_path=protected_report,
            output_dir=output_dir,
            symbol="XAUUSD-STD",
        )

        self.assertEqual(report["coverage"]["closed_trades_priced"], 1)
        self.assertEqual(report["coverage"]["m1_priced_trades"], 1)
        self.assertTrue((output_dir / "fractal_protected_economic_report.json").exists())
        self.assertTrue((output_dir / "current_vs_fractal_protected_comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
