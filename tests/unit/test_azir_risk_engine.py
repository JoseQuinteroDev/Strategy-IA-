from __future__ import annotations

import unittest
from datetime import datetime

from hybrid_quant.risk import AzirRiskConfig, AzirRiskEngine, AzirRiskState, evaluate_anomalies


def _state(**kwargs: object) -> AzirRiskState:
    defaults = {
        "timestamp": datetime(2025, 1, 6, 16, 30),
        "pending_orders": 0,
        "open_positions": 0,
        "daily_realized_pnl": 0.0,
        "trades_today": 0,
        "consecutive_losses_today": 0,
        "reconciled": True,
    }
    defaults.update(kwargs)
    return AzirRiskState(**defaults)


class AzirRiskEngineTests(unittest.TestCase):
    def test_force_cancel_at_close(self) -> None:
        engine = AzirRiskEngine()

        decision = engine.evaluate(
            _state(timestamp=datetime(2025, 1, 6, 22, 0), pending_orders=1),
            context="close_check",
        )

        self.assertTrue(decision.approved)
        self.assertIn("cancel_all_pendings", decision.actions)

    def test_blocks_new_setup_if_prior_exposure_exists(self) -> None:
        engine = AzirRiskEngine()

        decision = engine.evaluate(_state(pending_orders=1), context="setup_attempt")

        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason_code, "block_new_setups_if_any_position_or_pending_exists")
        self.assertIn("cancel_all_pendings", decision.actions)

    def test_reconciliation_blocks_dirty_state_before_setup(self) -> None:
        engine = AzirRiskEngine()

        decision = engine.evaluate(
            _state(reconciled=False, reconciliation_errors=("orphan_pending",)),
            context="before_setup",
        )

        self.assertFalse(decision.approved)
        self.assertIn("force_reconcile_orders_positions_before_setup", decision.blocked_by)
        self.assertIn("reconcile_broker_state", decision.actions)

    def test_friday_policy_blocks_and_cleans_prior_exposure(self) -> None:
        engine = AzirRiskEngine()

        decision = engine.evaluate(
            _state(timestamp=datetime(2025, 6, 20, 16, 30), pending_orders=1),
            context="setup_attempt",
        )

        self.assertFalse(decision.approved)
        self.assertIn("friday_no_new_trade_plus_close_or_cancel_prior_exposure", decision.blocked_by)
        self.assertIn("cancel_all_pendings", decision.actions)

    def test_daily_loss_and_consecutive_loss_kill_switches(self) -> None:
        engine = AzirRiskEngine(AzirRiskConfig(max_daily_loss=10.0, max_consecutive_losses=2))

        loss_decision = engine.evaluate(_state(daily_realized_pnl=-10.0), context="setup_attempt")
        streak_decision = engine.evaluate(_state(consecutive_losses_today=2), context="setup_attempt")

        self.assertIn("daily_max_loss_guard", loss_decision.blocked_by)
        self.assertIn("consecutive_losses_kill_switch", streak_decision.blocked_by)

    def test_max_trades_and_spread_guard(self) -> None:
        engine = AzirRiskEngine(AzirRiskConfig(max_trades_per_day=1, max_spread_points=30.0))

        trades_decision = engine.evaluate(_state(trades_today=1), context="setup_attempt")
        spread_decision = engine.evaluate(_state(spread_points=31.0), context="setup_attempt")

        self.assertIn("max_trades_per_day", trades_decision.blocked_by)
        self.assertIn("spread_guard_if_available", spread_decision.blocked_by)

    def test_after_fill_cancels_remaining_pendings(self) -> None:
        engine = AzirRiskEngine()

        decision = engine.evaluate(_state(open_positions=1, pending_orders=1), context="after_fill")

        self.assertIn("cancel_remaining_pendings", decision.actions)

    def test_anomaly_evaluation_maps_known_groups_to_risk_rules(self) -> None:
        anomalies = {
            "out_of_window_fills": [{"event_day": "2025-01-21", "setup_day": "2025-01-20"}],
            "friday_exit_events": [{"event_day": "2025-06-20", "setup_day": "2025-06-19"}],
            "multi_exit_days": [{"event_day": "2022-06-21", "assigned_setup_days": "2022-06-21"}],
            "open_order_cleanup_issues": [{"setup_day": "2022-06-20", "status": "missing_cleanup_or_unresolved"}],
        }

        rows = evaluate_anomalies(anomalies, AzirRiskConfig())

        self.assertEqual(len(rows), 4)
        self.assertTrue(any("hard_cancel_all_pendings_at_close" in row["rules"] for row in rows))
        self.assertTrue(any(row["impact"] == "would_mitigate" for row in rows))


if __name__ == "__main__":
    unittest.main()
