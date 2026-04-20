import unittest
from datetime import datetime

from hybrid_quant.azir.replica import OhlcvBar
from hybrid_quant.azir.trade_protection_research import (
    PriceSeries,
    TradeProtectionConfig,
    build_post_entry_dataset,
    compare_protection_results,
    price_protection_heuristics,
)
from hybrid_quant.env.azir_management_env import AzirManagementEvent


def _event(*, side: str = "buy", pnl: float = -5.0) -> AzirManagementEvent:
    return AzirManagementEvent(
        setup_day="2025-01-06",
        fill_timestamp=datetime(2025, 1, 6, 16, 30, 30),
        setup={
            "day_of_week": "1",
            "lot_size": "0.10",
            "sl_points": "500",
            "tp_points": "500",
            "trailing_start_points": "90",
            "trailing_step_points": "50",
            "atr_points": "200",
            "spread_points": "20",
        },
        trade={
            "fill_side": side,
            "fill_price": "100.0",
            "exit_timestamp": "2025.01.06 16:37:00",
            "exit_reason": "observed_exit",
            "net_pnl": str(pnl),
            "mfe_points": "120",
            "mae_points": "80",
        },
        lifecycle={},
    )


def _bars() -> list[OhlcvBar]:
    return [
        OhlcvBar(datetime(2025, 1, 6, 16, 30), 100.0, 100.1, 99.9, 100.0, 1),
        OhlcvBar(datetime(2025, 1, 6, 16, 31), 100.0, 101.2, 100.0, 100.9, 2),
        OhlcvBar(datetime(2025, 1, 6, 16, 32), 100.9, 101.0, 100.0, 100.0, 3),
        OhlcvBar(datetime(2025, 1, 6, 16, 33), 100.0, 100.2, 99.8, 99.9, 4),
        OhlcvBar(datetime(2025, 1, 6, 16, 34), 99.9, 100.0, 99.7, 99.8, 5),
        OhlcvBar(datetime(2025, 1, 6, 16, 35), 99.8, 99.9, 99.6, 99.7, 6),
        OhlcvBar(datetime(2025, 1, 6, 16, 36), 99.7, 99.8, 99.5, 99.6, 7),
    ]


class AzirTradeProtectionResearchTests(unittest.TestCase):
    def test_post_entry_dataset_is_causal_snapshot_without_final_outcome(self) -> None:
        event = _event()
        rows = build_post_entry_dataset(
            [event],
            [PriceSeries.from_bars("m1", 1, _bars())],
            TradeProtectionConfig(snapshot_minutes=(5,)),
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["future_outcome_excluded"], True)
        self.assertNotIn("net_pnl", rows[0])
        self.assertNotIn("exit_reason", rows[0])
        self.assertEqual(rows[0]["data_source"], "m1")

    def test_break_even_heuristic_can_clip_later_loss(self) -> None:
        event = _event(pnl=-5.0)
        rows = price_protection_heuristics(
            [event],
            [PriceSeries.from_bars("m1", 1, _bars())],
            TradeProtectionConfig(),
        )

        be = next(row for row in rows if row.heuristic == "move_to_be_after_90_points")
        self.assertEqual(be.exit_reason, "break_even_stop_hit_after_threshold")
        self.assertAlmostEqual(be.net_pnl or 0.0, 0.0)

    def test_comparison_uses_same_coverage_delta(self) -> None:
        event = _event(pnl=-5.0)
        rows = price_protection_heuristics(
            [event],
            [PriceSeries.from_bars("m1", 1, _bars())],
            TradeProtectionConfig(),
        )
        comparison, exits, risk = compare_protection_results([event], rows)

        be = next(row for row in comparison if row["heuristic"] == "move_to_be_after_90_points")
        self.assertEqual(be["priced_trades"], 1)
        self.assertAlmostEqual(be["delta_net_pnl_vs_base_same_coverage"], 5.0)
        self.assertTrue(exits)
        self.assertTrue(risk)


if __name__ == "__main__":
    unittest.main()
