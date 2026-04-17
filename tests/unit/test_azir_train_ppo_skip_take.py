from __future__ import annotations

import random
import unittest
from datetime import datetime
from pathlib import Path

from hybrid_quant.azir.train_ppo_skip_take import (
    AzirPPOConfig,
    evaluate_event_policy,
    load_config,
    make_env,
    random_valid_policy,
    split_events,
    take_all_valid_policy,
    take_only_sell_valid_policy,
    trade_metrics,
)
from hybrid_quant.env.azir_event_env import ACTION_SKIP, ACTION_TAKE, AzirEventReplayEnvironment, AzirReplayEvent


def _event(day: str, *, buy: bool = True, sell: bool = False, pnl: float = 1.0, order: bool = True) -> AzirReplayEvent:
    return AzirReplayEvent(
        setup_day=day,
        timestamp=datetime.fromisoformat(f"{day} 16:30:00"),
        setup={
            "timestamp": f"{day} 16:30:00",
            "event_type": "opportunity",
            "day_of_week": "1",
            "is_friday": "false",
            "buy_order_placed": "true" if buy and order else "false",
            "sell_order_placed": "true" if sell and order else "false",
            "buy_allowed_by_trend": "true",
            "sell_allowed_by_trend": "true",
            "spread_points": "20",
        },
        outcome={"protected_net_pnl": pnl, "protected_gross_pnl": pnl, "has_protected_fill": order},
        lifecycle={"order_placed": order, "cleanup_count": 1, "lifecycle_status": "filled"},
    )


class AzirTrainPPOSkipTakeTests(unittest.TestCase):
    def test_split_events_is_temporal_and_non_empty(self) -> None:
        events = [_event(f"2025-01-{day:02d}") for day in range(1, 11)]

        splits = split_events(events, AzirPPOConfig(train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2))

        self.assertEqual(len(splits["train"]), 6)
        self.assertEqual(len(splits["validation"]), 2)
        self.assertEqual(len(splits["test"]), 2)
        self.assertEqual(splits["train"][0].setup_day, "2025-01-01")
        self.assertEqual(splits["test"][-1].setup_day, "2025-01-10")

    def test_take_all_valid_collects_all_valid_trade_pnl(self) -> None:
        events = [_event("2025-01-01", pnl=1.0), _event("2025-01-02", pnl=-2.0)]

        result = evaluate_event_policy("take_all_valid", events, take_all_valid_policy, seed=1)

        self.assertEqual(result["trades_taken"], 2)
        self.assertEqual(result["net_pnl"], -1.0)
        self.assertEqual(result["take_attempts"], 2)

    def test_take_all_valid_uses_current_event_valid_actions(self) -> None:
        events = [
            _event("2025-01-01", pnl=1.0, order=False),
            _event("2025-01-02", pnl=2.0, order=True),
        ]

        result = evaluate_event_policy("take_all_valid", events, take_all_valid_policy, seed=1)

        self.assertEqual(result["trades_taken"], 1)
        self.assertEqual(result["net_pnl"], 2.0)
        self.assertEqual(result["invalid_attempts"], 0)

    def test_take_only_sell_valid_skips_buy_setups(self) -> None:
        events = [
            _event("2025-01-01", buy=True, sell=False, pnl=10.0),
            _event("2025-01-02", buy=False, sell=True, pnl=2.0),
        ]

        result = evaluate_event_policy("take_only_sell_valid", events, take_only_sell_valid_policy, seed=1)

        self.assertEqual(result["trades_taken"], 1)
        self.assertEqual(result["sell_takes"], 1)
        self.assertEqual(result["net_pnl"], 2.0)

    def test_random_valid_policy_is_seed_reproducible(self) -> None:
        env = AzirEventReplayEnvironment([_event("2025-01-01")])
        _, info = env.reset(seed=123)

        action_a = random_valid_policy(env, info, random.Random(7))
        action_b = random_valid_policy(env, info, random.Random(7))

        self.assertEqual(action_a, action_b)
        self.assertIn(action_a, {ACTION_SKIP, ACTION_TAKE})

    def test_config_can_enable_v2_observation_and_reward(self) -> None:
        config_path = Path("configs/experiments/azir_ppo_skip_take_adjusted_v1.yaml")

        config = load_config(config_path)
        env = make_env([_event("2025-01-01")], config)

        self.assertEqual(config.observation_version, "v2")
        self.assertEqual(config.reward_mode, "protected_net_pnl_scaled_v2")
        self.assertEqual(env.observation_version, "v2")
        self.assertIn("pending_distance_atr", env.observation_fields)

    def test_trade_metrics_reports_profit_factor_and_drawdown(self) -> None:
        metrics = trade_metrics([1.0, -2.0, 3.0])

        self.assertEqual(metrics["net_pnl"], 2.0)
        self.assertEqual(metrics["profit_factor"], 2.0)
        self.assertEqual(metrics["max_drawdown"], 2.0)


if __name__ == "__main__":
    unittest.main()
