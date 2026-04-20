import unittest
from datetime import datetime

from hybrid_quant.azir.management_replay_v2 import TickRecord
from hybrid_quant.azir.replica import OhlcvBar
from hybrid_quant.azir.tick_level_trade_protection_label_replay import (
    TickProtectionSnapshotConfig,
    build_tick_level_snapshot_rows,
)
from hybrid_quant.azir.trade_protection_research import PriceSeries
from hybrid_quant.env.azir_management_env import AzirManagementEvent


def _event() -> AzirManagementEvent:
    return AzirManagementEvent(
        setup_day="2025-01-06",
        fill_timestamp=datetime(2025, 1, 6, 16, 30, 0),
        setup={"lot_size": "0.10", "atr_points": "200", "spread_points": "20"},
        trade={
            "fill_side": "buy",
            "fill_price": "100.0",
            "exit_timestamp": "2025.01.06 16:35:00",
            "exit_reason": "observed",
            "net_pnl": "1.0",
        },
        lifecycle={},
    )


class TickLevelTradeProtectionLabelReplayTests(unittest.TestCase):
    def test_early_snapshot_uses_ticks_when_available(self) -> None:
        event = _event()
        key = "2025-01-06|2025-01-06 16:30:00|buy"
        ticks = [
            TickRecord(datetime(2025, 1, 6, 16, 30, 5), 1, 100.10, 100.12),
            TickRecord(datetime(2025, 1, 6, 16, 30, 30), 2, 100.25, 100.28),
        ]

        rows = build_tick_level_snapshot_rows(
            [event],
            {key: ticks},
            PriceSeries.from_bars("m1_fallback", 1, []),
            TickProtectionSnapshotConfig(snapshot_seconds=(30,)),
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["data_source"], "tick")
        self.assertEqual(rows[0]["snapshot_seconds_after_fill"], 30)
        self.assertGreater(float(rows[0]["unrealized_pnl_so_far"]), 0.0)

    def test_m1_fallback_requires_closed_bar_before_snapshot(self) -> None:
        event = _event()
        bars = [
            OhlcvBar(datetime(2025, 1, 6, 16, 30), 100.0, 100.2, 99.9, 100.1, 10),
            OhlcvBar(datetime(2025, 1, 6, 16, 31), 100.1, 100.3, 100.0, 100.2, 10),
        ]

        rows = build_tick_level_snapshot_rows(
            [event],
            {},
            PriceSeries.from_bars("m1_fallback", 1, bars),
            TickProtectionSnapshotConfig(snapshot_seconds=(30, 60, 120)),
        )

        self.assertEqual([row["snapshot_seconds_after_fill"] for row in rows], [60, 120])
        self.assertTrue(all(row["data_source"] == "m1_fallback" for row in rows))


if __name__ == "__main__":
    unittest.main()
