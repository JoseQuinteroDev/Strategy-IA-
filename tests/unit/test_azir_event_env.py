from __future__ import annotations

import unittest
from datetime import datetime

from hybrid_quant.env.azir_event_env import (
    ACTION_SKIP,
    ACTION_TAKE,
    FORBIDDEN_OBSERVATION_FIELDS,
    OBSERVATION_FIELDS,
    AzirEventReplayEnvironment,
    AzirReplayEvent,
)
from hybrid_quant.risk.azir_state import AzirRiskConfig


def _event(
    *,
    day: str = "2025-01-06",
    spread_points: str = "20",
    order: bool = True,
    pnl: float = 1.5,
) -> AzirReplayEvent:
    return AzirReplayEvent(
        setup_day=day,
        timestamp=datetime.fromisoformat(f"{day} 16:30:00"),
        setup={
            "timestamp": f"{day} 16:30:00",
            "event_type": "opportunity",
            "day_of_week": "1",
            "is_friday": "false",
            "buy_order_placed": "true" if order else "false",
            "sell_order_placed": "false",
            "buy_allowed_by_trend": "true",
            "sell_allowed_by_trend": "false",
            "swing_high": "2000.0",
            "swing_low": "1990.0",
            "buy_entry": "2000.05",
            "sell_entry": "1989.95",
            "pending_distance_points": "1010",
            "spread_points": spread_points,
            "ema20": "1998.0",
            "prev_close_vs_ema20_points": "25",
            "atr": "2.0",
            "atr_points": "200",
            "rsi": "55",
            "trend_filter_enabled": "true",
            "atr_filter_enabled": "true",
            "atr_filter_passed": "true",
            "rsi_gate_enabled": "true",
            "rsi_gate_required": "false",
        },
        outcome={
            "protected_net_pnl": pnl,
            "protected_gross_pnl": pnl,
            "has_protected_fill": True,
        },
        lifecycle={
            "order_placed": order,
            "cleanup_count": 1,
            "lifecycle_status": "filled" if order else "no_order",
            "survived_change_of_day": False,
            "out_of_window_fill_count": 0,
        },
    )


class AzirEventReplayEnvironmentTests(unittest.TestCase):
    def test_observation_schema_excludes_future_outcomes(self) -> None:
        leaked = set(OBSERVATION_FIELDS) & set(FORBIDDEN_OBSERVATION_FIELDS)

        self.assertEqual(leaked, set())

    def test_reset_builds_reproducible_observation(self) -> None:
        env = AzirEventReplayEnvironment([_event()])

        obs_a, info_a = env.reset(seed=123)
        obs_b, info_b = env.reset(seed=123)

        self.assertEqual(obs_a.shape, (len(OBSERVATION_FIELDS),))
        self.assertEqual(list(obs_a), list(obs_b))
        self.assertEqual(info_a["valid_actions"], info_b["valid_actions"])

    def test_action_space_is_skip_take_only(self) -> None:
        env = AzirEventReplayEnvironment([_event()])

        self.assertTrue(env.action_space.contains(ACTION_SKIP))
        self.assertTrue(env.action_space.contains(ACTION_TAKE))
        self.assertFalse(env.action_space.contains(2))

    def test_take_valid_setup_uses_protected_pnl_reward(self) -> None:
        env = AzirEventReplayEnvironment([_event(pnl=2.0)])
        env.reset()

        _, reward, terminated, _, info = env.step(ACTION_TAKE)

        self.assertTrue(terminated)
        self.assertGreater(reward, 0.0)
        self.assertEqual(info["action_effect"], "take")
        self.assertEqual(info["reward_breakdown"]["protected_net_pnl"], 2.0)

    def test_risk_blocked_take_is_transformed_to_skip(self) -> None:
        env = AzirEventReplayEnvironment([_event(spread_points="999")], risk_config=AzirRiskConfig(max_spread_points=50))
        env.reset()

        _, reward, _, _, info = env.step(ACTION_TAKE)

        self.assertLess(reward, 0.0)
        self.assertEqual(info["action_effect"], "risk_blocked_take_transformed_to_skip")
        self.assertFalse(info["risk_approved"])
        self.assertEqual(info["valid_actions"], (ACTION_SKIP,))

    def test_take_without_azir_order_is_invalid_noop(self) -> None:
        env = AzirEventReplayEnvironment([_event(order=False)])
        env.reset()

        _, reward, _, _, info = env.step(ACTION_TAKE)

        self.assertLess(reward, 0.0)
        self.assertEqual(info["action_effect"], "invalid_take_no_azir_order_transformed_to_skip")


if __name__ == "__main__":
    unittest.main()
