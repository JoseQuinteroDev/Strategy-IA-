from __future__ import annotations

import unittest

from hybrid_quant.azir.audit import build_daily_audit_records, summarize_records


class AzirAuditTests(unittest.TestCase):
    def test_canonical_setup_prefers_last_placed_opportunity(self) -> None:
        rows = [
            {
                "timestamp": "2025.01.06 16:30:00",
                "event_type": "opportunity",
                "symbol": "XAUUSD-STD",
                "magic": "123456321",
                "buy_order_placed": "false",
                "sell_order_placed": "false",
                "is_friday": "false",
            },
            {
                "timestamp": "2025.01.06 16:30:11",
                "event_type": "opportunity",
                "symbol": "XAUUSD-STD",
                "magic": "123456321",
                "buy_order_placed": "true",
                "sell_order_placed": "false",
                "is_friday": "false",
            },
        ]

        records = build_daily_audit_records(rows)

        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["buy_order_placed"])
        self.assertEqual(records[0]["mt5_opportunity_rows"], 2)

    def test_summarize_counts_multiple_exits_same_day_without_double_counting_setup(self) -> None:
        rows = [
            {
                "timestamp": "2025.01.06 16:30:00",
                "event_type": "opportunity",
                "symbol": "XAUUSD-STD",
                "magic": "123456321",
                "buy_order_placed": "true",
                "sell_order_placed": "false",
                "is_friday": "false",
            },
            {
                "timestamp": "2025.01.06 16:35:00",
                "event_type": "fill",
                "symbol": "XAUUSD-STD",
                "magic": "123456321",
                "fill_side": "buy",
            },
            {
                "timestamp": "2025.01.06 16:40:00",
                "event_type": "exit",
                "symbol": "XAUUSD-STD",
                "magic": "123456321",
                "net_pnl": "2.0",
                "gross_pnl": "2.0",
                "exit_reason": "take_profit",
            },
            {
                "timestamp": "2025.01.06 17:10:00",
                "event_type": "fill",
                "symbol": "XAUUSD-STD",
                "magic": "123456321",
                "fill_side": "sell",
            },
            {
                "timestamp": "2025.01.06 17:25:00",
                "event_type": "exit",
                "symbol": "XAUUSD-STD",
                "magic": "123456321",
                "net_pnl": "-1.0",
                "gross_pnl": "-1.0",
                "exit_reason": "stop_loss_or_trailing_stop",
            },
        ]

        records = build_daily_audit_records(rows)
        summary = summarize_records(records)

        self.assertEqual(len(records), 2)
        self.assertEqual(summary["setup_days"], 1)
        self.assertEqual(summary["filled_trades"], 2)
        self.assertEqual(summary["net_pnl"], 1.0)
        self.assertEqual(summary["tp_exits"], 1)
        self.assertEqual(summary["sl_or_trailing_exits"], 1)


if __name__ == "__main__":
    unittest.main()
