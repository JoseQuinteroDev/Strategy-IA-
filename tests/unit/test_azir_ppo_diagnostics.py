from __future__ import annotations

import unittest
from datetime import datetime

from hybrid_quant.azir.ppo_diagnostics import (
    atr_high_threshold,
    feature_scale_report,
    observation_diagnostics,
    reward_component_analysis,
    take_sell_valid_atr_high_policy,
    trace_policy,
)
from hybrid_quant.env.azir_event_env import ACTION_SKIP, ACTION_TAKE, AzirEventReplayEnvironment, AzirReplayEvent


def _event(
    day: str,
    *,
    buy: bool = False,
    sell: bool = True,
    pnl: float = 1.0,
    atr_points: float = 100.0,
    order: bool = True,
) -> AzirReplayEvent:
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
            "swing_high": "2350.0",
            "swing_low": "2320.0",
            "buy_entry": "2350.5",
            "sell_entry": "2319.5",
            "pending_distance_points": "310",
            "spread_points": "20",
            "ema20": "2338.0",
            "prev_close_vs_ema20_points": "40",
            "atr": str(atr_points / 100.0),
            "atr_points": str(atr_points),
            "rsi": "52",
            "trend_filter_enabled": "true",
            "atr_filter_enabled": "true",
            "atr_filter_passed": "true",
            "rsi_gate_enabled": "true",
            "rsi_gate_required": "false",
        },
        outcome={"protected_net_pnl": pnl, "protected_gross_pnl": pnl, "has_protected_fill": order},
        lifecycle={"order_placed": order, "cleanup_count": 1, "lifecycle_status": "filled"},
    )


class AzirPPODiagnosticsTests(unittest.TestCase):
    def test_atr_high_threshold_uses_train_sell_median(self) -> None:
        events = [
            _event("2025-01-01", atr_points=90.0),
            _event("2025-01-02", atr_points=110.0),
            _event("2025-01-03", buy=True, sell=False, atr_points=500.0),
        ]

        self.assertEqual(atr_high_threshold(events), 100.0)

    def test_sell_atr_high_policy_blocks_low_atr_sell(self) -> None:
        env = AzirEventReplayEnvironment([_event("2025-01-01", atr_points=80.0)])
        _, info = env.reset(seed=1)

        self.assertEqual(take_sell_valid_atr_high_policy(env, info, 100.0), ACTION_SKIP)

    def test_trace_policy_flags_skip_collapse(self) -> None:
        events = [_event(f"2025-01-{day:02d}") for day in range(1, 12)]

        row, trace = trace_policy(
            "skip_all_diagnostic",
            events,
            lambda env, info: ACTION_SKIP,
            seed=7,
            split="test",
        )

        self.assertTrue(row["collapsed_to_skip"])
        self.assertEqual(row["trades_taken"], 0)
        self.assertEqual(len(trace), len(events))

    def test_reward_component_analysis_reports_invalid_take_penalties(self) -> None:
        splits = {
            "train": [_event("2025-01-01")],
            "validation": [_event("2025-01-02", order=False)],
            "test": [_event("2025-01-03", order=False)],
        }

        rows = reward_component_analysis(splits, ppo_output_dir=__import__("pathlib").Path("missing"), seeds=(), atr_threshold=0.0)
        diagnostic = next(row for row in rows if row["split"] == "test" and row["policy"] == "take_every_event_diagnostic")

        self.assertGreater(diagnostic["invalid_action_penalty_sum"], 0.0)

    def test_observation_scale_report_flags_raw_price_features(self) -> None:
        splits = {"train": [_event("2025-01-01")], "validation": [_event("2025-01-02")], "test": [_event("2025-01-03")]}

        obs_rows = observation_diagnostics(splits)
        scale_rows = feature_scale_report(obs_rows)
        swing_high = next(row for row in scale_rows if row["feature"] == "swing_high")

        self.assertEqual(swing_high["recommended_action"], "normalize_or_express_relative_to_price")


if __name__ == "__main__":
    unittest.main()
