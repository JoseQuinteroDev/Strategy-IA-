import unittest
from datetime import datetime

from hybrid_quant.azir.management_replay import HeuristicSpec, ManagementReplayConfig
from hybrid_quant.azir.management_replay_v2 import (
    TickRecord,
    evaluate_management_heuristics_v2,
    price_all_management_actions_v2,
    price_management_action_v2,
)
from hybrid_quant.azir.replica import OhlcvBar
from hybrid_quant.env.azir_management_env import (
    ACTION_CLOSE_EARLY,
    ACTION_TRAILING_CONSERVATIVE,
    AzirManagementEvent,
)


def _event(*, pnl: float = -1.0, mfe: float = 120.0) -> AzirManagementEvent:
    return AzirManagementEvent(
        setup_day="2025-01-06",
        fill_timestamp=datetime(2025, 1, 6, 16, 30, 0),
        setup={"lot_size": "0.01"},
        trade={
            "fill_side": "buy",
            "fill_price": "100.0",
            "exit_timestamp": "2025.01.06 16:35:00",
            "exit_reason": "observed_exit",
            "net_pnl": str(pnl),
            "mfe_points": str(mfe),
            "mae_points": "50",
        },
        lifecycle={},
    )


def _ticks() -> list[TickRecord]:
    return [
        TickRecord(datetime(2025, 1, 6, 16, 30, 10), 1, 100.2, 100.4),
        TickRecord(datetime(2025, 1, 6, 16, 31, 0), 2, 101.3, 101.5),
        TickRecord(datetime(2025, 1, 6, 16, 31, 30), 3, 100.6, 100.8),
        TickRecord(datetime(2025, 1, 6, 16, 32, 0), 4, 100.5, 100.7),
    ]


def _m1_bars() -> list[OhlcvBar]:
    return [
        OhlcvBar(datetime(2025, 1, 6, 16, 30), 100.0, 100.2, 99.9, 100.1, 1),
        OhlcvBar(datetime(2025, 1, 6, 16, 31), 100.1, 100.2, 99.8, 100.0, 1),
    ]


class AzirManagementReplayV2Tests(unittest.TestCase):
    def test_close_early_uses_tick_bid_for_buy_exit(self) -> None:
        result = price_management_action_v2(
            _event(),
            _ticks(),
            _m1_bars(),
            ACTION_CLOSE_EARLY,
            ManagementReplayConfig(),
        )

        self.assertEqual(result.pricing_source, "tick")
        self.assertEqual(result.exit_reason, "close_early_tick_checkpoint")
        self.assertAlmostEqual(result.net_pnl or 0.0, 1.3)

    def test_trailing_uses_chronological_tick_stop(self) -> None:
        result = price_management_action_v2(
            _event(),
            _ticks(),
            _m1_bars(),
            ACTION_TRAILING_CONSERVATIVE,
            ManagementReplayConfig(),
        )

        self.assertEqual(result.pricing_source, "tick")
        self.assertEqual(result.exit_reason, "tick_management_stop_hit")
        self.assertAlmostEqual(result.net_pnl or 0.0, 0.6)

    def test_v2_falls_back_to_m1_when_ticks_are_missing(self) -> None:
        result = price_management_action_v2(
            _event(),
            [],
            _m1_bars(),
            ACTION_CLOSE_EARLY,
            ManagementReplayConfig(),
        )

        self.assertEqual(result.pricing_source, "m1_fallback")

    def test_heuristic_rows_report_tick_and_m1_source_counts(self) -> None:
        event = _event()
        action_rows = price_all_management_actions_v2(
            [event],
            {f"{event.setup_day}|{event.fill_timestamp.isoformat(sep=' ')}|{event.side}": _ticks()},
            _m1_bars(),
            ManagementReplayConfig(),
        )
        rows, same_coverage, exits = evaluate_management_heuristics_v2(
            [event],
            action_rows,
            [HeuristicSpec("close", "always", "close_early")],
        )

        self.assertEqual(rows[0]["tick_priced_trades"], 1)
        self.assertEqual(same_coverage[0]["tick_priced_trades"], 1)
        self.assertTrue(exits)


if __name__ == "__main__":
    unittest.main()
