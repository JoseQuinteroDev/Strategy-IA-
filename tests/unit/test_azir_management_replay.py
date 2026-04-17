import unittest
from datetime import datetime
from pathlib import Path

from hybrid_quant.azir.management_replay import (
    HeuristicSpec,
    ManagementReplayConfig,
    evaluate_management_heuristics,
    load_config,
    price_all_management_actions,
    price_management_action,
)
from hybrid_quant.env.azir_management_env import (
    ACTION_CLOSE_EARLY,
    ACTION_MOVE_TO_BREAK_EVEN,
    ACTION_TRAILING_CONSERVATIVE,
    AzirManagementEvent,
)
from hybrid_quant.azir.replica import OhlcvBar


def _event(*, side: str = "buy", pnl: float = -1.0, mfe: float = 120.0) -> AzirManagementEvent:
    return AzirManagementEvent(
        setup_day="2025-01-06",
        fill_timestamp=datetime(2025, 1, 6, 16, 30, 30),
        setup={
            "lot_size": "0.01",
            "atr_points": "200",
            "atr": "2.0",
            "swing_high": "100.0",
            "swing_low": "95.0",
            "buy_entry": "100.0",
            "sell_entry": "95.0",
        },
        trade={
            "fill_side": side,
            "fill_price": "100.0",
            "exit_timestamp": "2025.01.06 16:36:00",
            "exit_reason": "observed_exit",
            "net_pnl": str(pnl),
            "mfe_points": str(mfe),
            "mae_points": "50",
        },
        lifecycle={},
    )


def _bars() -> list[OhlcvBar]:
    return [
        OhlcvBar(datetime(2025, 1, 6, 16, 30), 100.0, 100.3, 99.9, 100.2, 1),
        OhlcvBar(datetime(2025, 1, 6, 16, 31), 100.2, 101.3, 100.1, 100.9, 1),
        OhlcvBar(datetime(2025, 1, 6, 16, 32), 100.9, 101.1, 100.0, 100.0, 1),
        OhlcvBar(datetime(2025, 1, 6, 16, 33), 100.0, 100.4, 99.6, 100.1, 1),
        OhlcvBar(datetime(2025, 1, 6, 16, 34), 100.1, 100.2, 99.4, 99.8, 1),
        OhlcvBar(datetime(2025, 1, 6, 16, 35), 99.8, 100.0, 99.5, 99.7, 1),
    ]


class AzirManagementReplayTests(unittest.TestCase):
    def test_default_config_loads_core_management_heuristics(self) -> None:
        config, heuristics = load_config(Path("configs/experiments/azir_management_price_replay_v1.yaml"))

        self.assertEqual(config.point, 0.01)
        self.assertIn("always_base_management", {item.name for item in heuristics})
        self.assertIn("trailing_conservative", {item.action for item in heuristics})

    def test_close_early_uses_first_closed_m1_after_fill(self) -> None:
        result = price_management_action(_event(), _bars(), ACTION_CLOSE_EARLY, ManagementReplayConfig())

        self.assertEqual(result.status, "priced_with_m1_proxy")
        self.assertEqual(result.exit_timestamp, "2025-01-06 16:31:00")
        self.assertAlmostEqual(result.net_pnl or 0.0, 0.2)

    def test_break_even_can_clip_later_loss_after_activation(self) -> None:
        result = price_management_action(_event(), _bars(), ACTION_MOVE_TO_BREAK_EVEN, ManagementReplayConfig())

        self.assertEqual(result.exit_reason, "break_even_or_original_stop_hit_m1")
        self.assertAlmostEqual(result.net_pnl or 0.0, 0.0)

    def test_trailing_conservative_ratchets_stop_from_m1_path(self) -> None:
        result = price_management_action(_event(), _bars(), ACTION_TRAILING_CONSERVATIVE, ManagementReplayConfig())

        self.assertEqual(result.exit_reason, "trailing_stop_hit_m1")
        self.assertAlmostEqual(result.net_pnl or 0.0, 0.6)

    def test_management_action_keeps_base_when_m1_threshold_never_activates(self) -> None:
        config = ManagementReplayConfig(trailing_conservative_activation_points=300.0)
        result = price_management_action(_event(pnl=2.5), _bars(), ACTION_TRAILING_CONSERVATIVE, config)

        self.assertEqual(result.exit_reason, "base_management_no_m1_activation")
        self.assertAlmostEqual(result.net_pnl or 0.0, 2.5)

    def test_heuristic_comparison_uses_same_coverage_delta(self) -> None:
        event = _event(pnl=-1.0, mfe=120.0)
        action_rows = price_all_management_actions([event], _bars(), ManagementReplayConfig())
        rows, exits = evaluate_management_heuristics(
            [event],
            action_rows,
            [
                HeuristicSpec("always_base_management", "always", "base_management"),
                HeuristicSpec("be", "mfe_threshold", "move_to_break_even", threshold_points=90.0),
            ],
        )

        be = next(row for row in rows if row["heuristic"] == "be")
        self.assertEqual(be["priced_trades"], 1)
        self.assertAlmostEqual(be["delta_net_pnl_vs_base_same_coverage"], 1.0)
        self.assertTrue(exits)


if __name__ == "__main__":
    unittest.main()
