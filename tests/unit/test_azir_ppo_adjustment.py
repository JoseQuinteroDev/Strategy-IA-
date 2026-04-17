from __future__ import annotations

import unittest
from datetime import datetime

from hybrid_quant.azir.ppo_adjustment import observation_adjustment_report, reward_component_adjustment_report
from hybrid_quant.azir.train_ppo_skip_take import AzirPPOConfig
from hybrid_quant.env.azir_event_env import AzirReplayEvent


def _event(day: str, pnl: float = 5.0) -> AzirReplayEvent:
    return AzirReplayEvent(
        setup_day=day,
        timestamp=datetime.fromisoformat(f"{day} 16:30:00"),
        setup={
            "timestamp": f"{day} 16:30:00",
            "event_type": "opportunity",
            "day_of_week": "1",
            "is_friday": "false",
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
            "atr_filter_enabled": "true",
            "atr_filter_passed": "true",
            "rsi_gate_enabled": "true",
            "rsi_gate_required": "false",
        },
        outcome={"protected_net_pnl": pnl, "protected_gross_pnl": pnl, "has_protected_fill": True},
        lifecycle={"order_placed": True, "cleanup_count": 1, "lifecycle_status": "filled"},
    )


class AzirPPOAdjustmentTests(unittest.TestCase):
    def test_observation_adjustment_reports_raw_v1_and_relative_v2(self) -> None:
        events = [_event("2025-01-01"), _event("2025-01-02")]
        splits = {"train": events, "validation": events, "test": events}

        rows = observation_adjustment_report(splits, AzirPPOConfig(observation_version="v2"))

        v1_buy_entry = next(row for row in rows if row["observation_version"] == "v1" and row["feature"] == "buy_entry")
        v2_pending_ratio = next(row for row in rows if row["observation_version"] == "v2" and row["feature"] == "pending_distance_atr")
        self.assertTrue(v1_buy_entry["raw_absolute_price_or_money"])
        self.assertEqual(v2_pending_ratio["scale_flag"], "unit_or_small_scale")

    def test_reward_adjustment_makes_skip_cost_visible(self) -> None:
        events = [_event("2025-01-01", pnl=10.0), _event("2025-01-02", pnl=10.0)]
        splits = {"train": events, "validation": events, "test": events}
        config = AzirPPOConfig(
            observation_version="v2",
            reward_mode="protected_net_pnl_scaled_v2",
            skip_opportunity_cost_weight=0.1,
            skip_opportunity_cost_cap=1.0,
        )

        rows = reward_component_adjustment_report(splits, config)

        before = next(row for row in rows if row["scenario"] == "before_v1_reward" and row["policy"] == "skip_all" and row["split"] == "test")
        after = next(row for row in rows if row["scenario"] == "after_v2_reward" and row["policy"] == "skip_all" and row["split"] == "test")
        self.assertEqual(before["sum_skip_opportunity_cost"], 0.0)
        self.assertGreater(after["sum_skip_opportunity_cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
