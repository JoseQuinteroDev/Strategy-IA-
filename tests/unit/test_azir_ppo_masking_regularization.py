from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

from hybrid_quant.azir.ppo_masking_regularization import (
    eligible_only_events,
    load_masking_config,
    masking_report_rows,
)
from hybrid_quant.azir.train_ppo_skip_take import AzirPPOConfig
from hybrid_quant.env.azir_event_env import AzirReplayEvent


def _event(day: str, *, order: bool = True, spread_points: str = "20") -> AzirReplayEvent:
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
            "spread_points": spread_points,
        },
        outcome={"protected_net_pnl": 1.0, "protected_gross_pnl": 1.0, "has_protected_fill": order},
        lifecycle={"order_placed": order, "cleanup_count": 1, "lifecycle_status": "filled" if order else "no_order"},
    )


class AzirPPOMaskingRegularizationTests(unittest.TestCase):
    def test_eligible_only_events_removes_no_order_and_risk_blocked_events(self) -> None:
        events = [
            _event("2025-01-01", order=True, spread_points="20"),
            _event("2025-01-02", order=False, spread_points="20"),
            _event("2025-01-03", order=True, spread_points="999"),
        ]

        eligible = eligible_only_events(events, AzirPPOConfig(observation_version="v2"))

        self.assertEqual([event.setup_day for event in eligible], ["2025-01-01"])

    def test_masking_report_counts_removed_events(self) -> None:
        unmasked = {"test": [_event("2025-01-01"), _event("2025-01-02", order=False)]}
        masked = {"test": [_event("2025-01-01")]}

        rows = masking_report_rows(unmasked, masked)

        self.assertEqual(rows[0]["unmasked_events"], 2)
        self.assertEqual(rows[0]["masked_eligible_events"], 1)
        self.assertEqual(rows[0]["removed_events"], 1)

    def test_masking_config_loads_regularization_variants(self) -> None:
        config = load_masking_config(Path("configs/experiments/azir_ppo_masking_regularization_v1.yaml"))

        self.assertEqual(config.masking_mode, "eligible_only")
        self.assertEqual(config.ppo_config.observation_version, "v2")
        self.assertGreaterEqual(len(config.regularization_variants), 3)


if __name__ == "__main__":
    unittest.main()
