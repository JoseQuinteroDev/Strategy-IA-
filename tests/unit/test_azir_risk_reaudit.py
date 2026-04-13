from __future__ import annotations

import unittest
from datetime import datetime

from hybrid_quant.azir.economic_audit import reconstruct_lifecycles
from hybrid_quant.azir.risk_reaudit import apply_risk_engine_to_lifecycle
from hybrid_quant.risk.azir_state import AzirRiskConfig


def _row(timestamp: str, event_type: str, **kwargs: object) -> dict[str, object]:
    row = {
        "timestamp": timestamp,
        "event_type": event_type,
        "symbol": "XAUUSD-STD",
        "event_id": kwargs.pop("event_id", ""),
        "_raw_event_id_blank": kwargs.pop("_raw_event_id_blank", True),
        "_event_day": timestamp.split(" ")[0].replace(".", "-"),
        "_timestamp_dt": datetime.fromisoformat(timestamp.replace(".", "-")),
        "_event_id_day": "",
        "spread_points": kwargs.pop("spread_points", "27"),
    }
    row.update(kwargs)
    return row


def _reconstruct(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    return reconstruct_lifecycles(rows, session_start_hour=16, session_end_hour=21, close_hour=22)


class AzirRiskReauditTests(unittest.TestCase):
    def test_out_of_window_fill_is_prevented_by_close_cancel(self) -> None:
        rows = [
            _row(
                "2025.01.20 16:30:00",
                "opportunity",
                event_id="2025.01.20_XAUUSD-STD_123",
                _raw_event_id_blank=False,
                _event_id_day="2025-01-20",
                buy_order_placed="true",
                sell_order_placed="false",
            ),
            _row("2025.01.21 03:27:29", "fill", fill_side="buy", fill_price="2712.37"),
            _row("2025.01.21 04:00:00", "exit", exit_reason="take_profit", net_pnl="3.0", gross_pnl="3.0"),
        ]
        reconstructed = _reconstruct(rows)

        result = apply_risk_engine_to_lifecycle(
            rows=rows,
            lifecycle_rows=reconstructed["lifecycles"],
            trade_rows=reconstructed["trades"],
            config=AzirRiskConfig(),
        )

        self.assertEqual(result["trade_decisions"][0]["risk_status"], "prevented")
        self.assertIn("hard_cancel_all_pendings_at_close", result["trade_decisions"][0]["risk_rules"])
        self.assertEqual(len(result["protected_trades"]), 0)
        self.assertFalse(result["lifecycle_after"][0]["cleanup_issue_after"])

    def test_second_fill_from_same_setup_is_removed_after_first_fill(self) -> None:
        rows = [
            _row(
                "2025.01.20 16:30:00",
                "opportunity",
                event_id="2025.01.20_XAUUSD-STD_123",
                _raw_event_id_blank=False,
                _event_id_day="2025-01-20",
                buy_order_placed="true",
                sell_order_placed="true",
            ),
            _row("2025.01.20 16:35:00", "fill", fill_side="buy", fill_price="2700.00"),
            _row("2025.01.20 16:40:00", "exit", exit_reason="take_profit", net_pnl="2.0", gross_pnl="2.0"),
            _row("2025.01.20 17:00:00", "fill", fill_side="sell", fill_price="2690.00"),
            _row("2025.01.20 17:10:00", "exit", exit_reason="stop_loss", net_pnl="-3.0", gross_pnl="-3.0"),
        ]
        reconstructed = _reconstruct(rows)

        result = apply_risk_engine_to_lifecycle(
            rows=rows,
            lifecycle_rows=reconstructed["lifecycles"],
            trade_rows=reconstructed["trades"],
            config=AzirRiskConfig(),
        )

        self.assertEqual([row["risk_status"] for row in result["trade_decisions"]], ["kept_observed_exit", "prevented"])
        self.assertEqual(len(result["protected_trades"]), 1)
        self.assertEqual(result["protected_trades"][0]["net_pnl"], "2.0")

    def test_next_day_exit_is_marked_unpriced_forced_close(self) -> None:
        rows = [
            _row(
                "2025.01.20 16:30:00",
                "opportunity",
                event_id="2025.01.20_XAUUSD-STD_123",
                _raw_event_id_blank=False,
                _event_id_day="2025-01-20",
                buy_order_placed="true",
                sell_order_placed="false",
            ),
            _row("2025.01.20 21:00:00", "fill", fill_side="buy", fill_price="2700.00"),
            _row("2025.01.21 09:00:00", "exit", exit_reason="take_profit", net_pnl="4.0", gross_pnl="4.0"),
        ]
        reconstructed = _reconstruct(rows)

        result = apply_risk_engine_to_lifecycle(
            rows=rows,
            lifecycle_rows=reconstructed["lifecycles"],
            trade_rows=reconstructed["trades"],
            config=AzirRiskConfig(),
        )

        self.assertEqual(result["trade_decisions"][0]["risk_status"], "forced_close_unpriced")
        self.assertFalse(result["trade_decisions"][0]["pnl_priced_after_risk"])
        self.assertEqual(len(result["protected_trades"]), 0)
        self.assertEqual(result["lifecycle_after"][0]["forced_close_unpriced_count"], 1)


if __name__ == "__main__":
    unittest.main()
