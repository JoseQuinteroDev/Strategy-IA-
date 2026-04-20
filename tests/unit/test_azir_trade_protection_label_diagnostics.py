import unittest
from datetime import datetime

from hybrid_quant.azir.trade_protection_label_diagnostics import (
    LabelDiagnosticsConfig,
    build_label_distribution,
    build_separability_summary,
    diagnose_features,
    label_post_entry_rows,
)
from hybrid_quant.env.azir_management_env import AzirManagementEvent


def _event(*, pnl: float) -> AzirManagementEvent:
    return AzirManagementEvent(
        setup_day="2025-01-06",
        fill_timestamp=datetime(2025, 1, 6, 16, 30, 0),
        setup={"lot_size": "0.10"},
        trade={
            "fill_side": "buy",
            "fill_price": "100.0",
            "exit_timestamp": "2025.01.06 16:45:00",
            "exit_reason": "observed_exit",
            "net_pnl": str(pnl),
            "mfe_points": "100",
            "mae_points": "50",
        },
        lifecycle={},
    )


def _snapshot(*, pnl_now: float, direction: float = -0.5) -> dict[str, object]:
    return {
        "event_key": "2025-01-06|2025-01-06 16:30:00|buy",
        "setup_day": "2025-01-06",
        "fill_timestamp": "2025-01-06 16:30:00",
        "snapshot_timestamp": "2025-01-06 16:35:00",
        "snapshot_minutes_after_fill": 5,
        "data_source": "m1",
        "side": "buy",
        "entry_price": 100.0,
        "distance_to_initial_sl_points": 400.0,
        "distance_to_initial_tp_points": 500.0,
        "mfe_points_so_far": 10.0,
        "mae_points_so_far": 120.0,
        "unrealized_pnl_so_far": pnl_now,
        "time_to_session_close_minutes": 300.0,
        "atr_points_setup": 200.0,
        "atr_relative_mfe": 0.05,
        "atr_relative_mae": 0.60,
        "post_entry_speed_points_per_min": 2.0,
        "m1_m5_close_direction_proxy": direction,
        "volume_proxy_sum": 100.0,
        "spread_to_atr": 0.1,
        "fill_hour": 16,
        "day_of_week": 1,
    }


class AzirTradeProtectionLabelDiagnosticsTests(unittest.TestCase):
    def test_labels_identify_helpful_and_harmful_early_close(self) -> None:
        helpful = label_post_entry_rows([_snapshot(pnl_now=-1.0)], [_event(pnl=-5.0)], LabelDiagnosticsConfig())
        harmful = label_post_entry_rows([_snapshot(pnl_now=-1.0)], [_event(pnl=2.0)], LabelDiagnosticsConfig())

        self.assertEqual(helpful[0]["label_early_close_helpful"], 1)
        self.assertEqual(helpful[0]["label_momentum_break_true"], 1)
        self.assertEqual(harmful[0]["label_early_close_harmful"], 1)
        self.assertEqual(harmful[0]["label_false_deterioration"], 1)

    def test_feature_diagnostics_detect_simple_separation(self) -> None:
        rows = []
        for index in range(10):
            row = _snapshot(pnl_now=-float(index), direction=-float(index))
            row.update(
                {
                    "event_key": f"2025-01-06|2025-01-06 16:3{index % 10}:00|buy",
                    "label_deteriorated_trade": int(index >= 5),
                    "label_early_close_helpful": int(index >= 5),
                    "label_early_close_harmful": int(index < 5),
                    "label_recoverable_trade": 0,
                    "label_momentum_break_true": int(index >= 5),
                    "label_false_deterioration": 0,
                    "label_base_final_net_pnl": -5.0 if index >= 5 else 2.0,
                    "label_early_close_proxy_pnl": -float(index),
                    "label_early_close_delta_vs_base": 0.0,
                    "label_final_winner": int(index < 5),
                    "label_final_loser": int(index >= 5),
                }
            )
            rows.append(row)

        features = diagnose_features(rows)
        distribution = build_label_distribution(rows)
        separability = build_separability_summary(features, distribution, LabelDiagnosticsConfig(min_labeled_snapshots=1))

        self.assertGreaterEqual(separability["label_early_close_helpful"]["best_auc_edge_abs"], 0.4)


if __name__ == "__main__":
    unittest.main()
