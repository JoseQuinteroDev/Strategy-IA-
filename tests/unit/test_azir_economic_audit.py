from __future__ import annotations

import unittest
from datetime import datetime

from hybrid_quant.azir.economic_audit import (
    build_anomaly_reports,
    build_trailing_report,
    reconstruct_lifecycles,
)


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
    }
    row.update(kwargs)
    return row


class AzirEconomicAuditTests(unittest.TestCase):
    def test_blank_fill_is_assigned_to_prior_gtc_lifecycle(self) -> None:
        rows = [
            _row(
                "2025.01.06 16:30:00",
                "opportunity",
                event_id="2025.01.06_XAUUSD-STD_123",
                _raw_event_id_blank=False,
                _event_id_day="2025-01-06",
                buy_order_placed="true",
                sell_order_placed="false",
                is_friday="false",
            ),
            _row("2025.01.07 03:00:00", "fill", fill_side="buy", fill_price="1.0"),
            _row("2025.01.07 03:02:00", "exit", exit_reason="take_profit", net_pnl="1.0", gross_pnl="1.0"),
        ]

        result = reconstruct_lifecycles(rows, session_start_hour=16, session_end_hour=21, close_hour=22)

        self.assertEqual(result["trades"][0]["setup_day"], "2025-01-06")
        self.assertTrue(result["trades"][0]["assigned_from_blank_event_id"])
        self.assertTrue(result["lifecycles"][0]["survived_change_of_day"])
        self.assertEqual(result["lifecycles"][0]["out_of_window_fill_count"], 1)

    def test_anomaly_reports_include_cleanup_survival_and_friday_exit(self) -> None:
        rows = [
            _row(
                "2025.06.19 16:30:00",
                "opportunity",
                event_id="2025.06.19_XAUUSD-STD_123",
                _raw_event_id_blank=False,
                _event_id_day="2025-06-19",
                buy_order_placed="false",
                sell_order_placed="true",
                is_friday="false",
            ),
            _row("2025.06.20 04:11:41", "fill", fill_side="sell", fill_price="1.0"),
            _row("2025.06.20 04:59:38", "exit", exit_reason="take_profit", net_pnl="5.0", gross_pnl="5.0"),
        ]
        result = reconstruct_lifecycles(rows, session_start_hour=16, session_end_hour=21, close_hour=22)

        anomalies = build_anomaly_reports(rows, result["lifecycles"], result["trades"], 16, 21)

        self.assertEqual(len(anomalies["out_of_window_fills"]), 1)
        self.assertEqual(len(anomalies["friday_exit_events"]), 1)
        self.assertEqual(len(anomalies["open_order_cleanup_issues"]), 1)

    def test_trailing_report_is_observational_not_causal(self) -> None:
        trades = [
            {"has_exit": True, "net_pnl": "1.0", "trailing_activated": True, "time_to_first_trailing_seconds": "60"},
            {"has_exit": True, "net_pnl": "-5.0", "trailing_activated": False},
        ]

        report = build_trailing_report(trades)

        self.assertEqual(report["summary"]["causality"], "not_proven_observational_only")
        self.assertEqual(report["summary"]["trailing_activated_trades"], 1)


if __name__ == "__main__":
    unittest.main()
