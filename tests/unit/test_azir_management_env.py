from __future__ import annotations

import math
import unittest
from datetime import datetime

from hybrid_quant.env.azir_management_env import (
    ACTION_BASE_MANAGEMENT,
    ACTION_CLOSE_EARLY,
    ACTION_MOVE_TO_BREAK_EVEN,
    ACTION_TRAILING_AGGRESSIVE,
    ACTION_TRAILING_CONSERVATIVE,
    FORBIDDEN_MANAGEMENT_OBSERVATION_FIELDS,
    MANAGEMENT_ACTIONS,
    MANAGEMENT_OBSERVATION_FIELDS,
    AzirManagementEvent,
    AzirManagementReplayEnvironment,
)


def _management_event(*, pnl: float = -5.0, mfe_points: float = 100.0, mae_points: float = 50.0) -> AzirManagementEvent:
    return AzirManagementEvent(
        setup_day="2025-01-06",
        fill_timestamp=datetime.fromisoformat("2025-01-06 17:15:00"),
        setup={
            "timestamp": "2025-01-06 16:30:00",
            "event_type": "opportunity",
            "day_of_week": "1",
            "buy_order_placed": "true",
            "sell_order_placed": "false",
            "buy_allowed_by_trend": "true",
            "sell_allowed_by_trend": "false",
            "swing_high": "2000.0",
            "swing_low": "1990.0",
            "buy_entry": "2000.05",
            "sell_entry": "1989.95",
            "pending_distance_points": "1010",
            "spread_points": "20",
            "ema20": "1998.0",
            "prev_close_vs_ema20_points": "25",
            "atr": "2.0",
            "atr_points": "200",
            "rsi": "55",
            "trend_filter_enabled": "true",
            "atr_filter_passed": "true",
            "rsi_gate_required": "false",
            "trailing_start_points": "90",
            "trailing_step_points": "50",
        },
        trade={
            "setup_day": "2025-01-06",
            "fill_timestamp": "2025.01.06 17:15:00",
            "fill_side": "buy",
            "fill_price": "2000.05",
            "duration_to_fill_seconds": "2700",
            "mfe_points": str(mfe_points),
            "mae_points": str(mae_points),
            "net_pnl": pnl,
            "gross_pnl": pnl,
            "exit_reason": "stop_loss_or_trailing_stop",
        },
        lifecycle={"setup_day": "2025-01-06", "protected_lifecycle_status": "filled_observed_exit_kept"},
    )


class AzirManagementReplayEnvironmentTests(unittest.TestCase):
    def test_action_contract_is_small_and_discrete(self) -> None:
        env = AzirManagementReplayEnvironment([_management_event()])

        self.assertEqual(env.action_space.n, 5)
        self.assertEqual(MANAGEMENT_ACTIONS[ACTION_BASE_MANAGEMENT], "base_management")
        self.assertEqual(MANAGEMENT_ACTIONS[ACTION_CLOSE_EARLY], "close_early")
        self.assertEqual(MANAGEMENT_ACTIONS[ACTION_MOVE_TO_BREAK_EVEN], "move_to_break_even")
        self.assertEqual(MANAGEMENT_ACTIONS[ACTION_TRAILING_CONSERVATIVE], "trailing_conservative")
        self.assertEqual(MANAGEMENT_ACTIONS[ACTION_TRAILING_AGGRESSIVE], "trailing_aggressive")

    def test_observation_excludes_future_outcome_fields(self) -> None:
        env = AzirManagementReplayEnvironment([_management_event()])

        obs, info = env.reset(seed=123)

        self.assertEqual(obs.shape, (len(MANAGEMENT_OBSERVATION_FIELDS),))
        self.assertEqual(set(MANAGEMENT_OBSERVATION_FIELDS) & set(FORBIDDEN_MANAGEMENT_OBSERVATION_FIELDS), set())
        self.assertTrue(all(math.isfinite(float(value)) for value in obs))
        self.assertNotIn("mfe_points", info["observation_schema"])
        self.assertNotIn("net_pnl", info["observation_schema"])

    def test_base_management_reward_uses_protected_pnl(self) -> None:
        env = AzirManagementReplayEnvironment([_management_event(pnl=7.5)])
        env.reset()

        _, reward, terminated, _, info = env.step(ACTION_BASE_MANAGEMENT)

        self.assertTrue(terminated)
        self.assertEqual(reward, 7.5)
        self.assertEqual(info["reward_breakdown"]["pricing_confidence"], "observed")
        self.assertEqual(info["reward_breakdown"]["base_protected_net_pnl"], 7.5)

    def test_break_even_proxy_is_explicitly_marked_as_proxy(self) -> None:
        env = AzirManagementReplayEnvironment([_management_event(pnl=-5.0, mfe_points=120.0)])
        env.reset()

        _, reward, _, _, info = env.step(ACTION_MOVE_TO_BREAK_EVEN)

        self.assertEqual(reward, 0.0)
        self.assertEqual(info["reward_breakdown"]["management_proxy_net_pnl"], 0.0)
        self.assertEqual(info["reward_breakdown"]["pricing_confidence"], "proxy_requires_price_replay")

    def test_zero_information_trade_only_allows_base_management(self) -> None:
        env = AzirManagementReplayEnvironment([_management_event(pnl=0.0, mfe_points=0.0, mae_points=0.0)])
        _, info = env.reset()

        self.assertEqual(info["valid_actions"], (ACTION_BASE_MANAGEMENT,))


if __name__ == "__main__":
    unittest.main()
