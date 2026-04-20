import unittest

from hybrid_quant.azir.trade_protection_threshold_refinement import (
    MomentumCandidate,
    ThresholdRefinementConfig,
    candidate_triggers,
    evaluate_early_close_helpful_thresholds,
    evaluate_momentum_candidate,
)


def _row(
    *,
    year: str,
    side: str,
    delta: float,
    direction: float = -0.30,
    unrealized: float = -0.10,
    mae_atr: float = 0.30,
    speed: float = 10.0,
    source: str = "tick",
    horizon: int = 60,
) -> dict[str, object]:
    return {
        "event_key": f"{year}-{side}-{delta}-{horizon}",
        "setup_day": f"{year}-01-02",
        "side": side,
        "data_source": source,
        "snapshot_seconds_after_fill": horizon,
        "m1_m5_close_direction_proxy": direction,
        "unrealized_pnl_so_far": unrealized,
        "atr_relative_mae": mae_atr,
        "post_entry_speed_points_per_min": speed,
        "label_early_close_delta_vs_base": delta,
        "mae_points_so_far": mae_atr * 100.0,
        "mfe_points_so_far": max(delta, 0.0),
        "spread_to_atr": 0.05,
        "fill_hour": 16,
    }


class TradeProtectionThresholdRefinementTests(unittest.TestCase):
    def test_candidate_triggers_respects_all_thresholds(self) -> None:
        candidate = MomentumCandidate("balanced", direction_max=-0.25, unrealized_max=-0.25, mae_atr_min=0.25)

        self.assertTrue(candidate_triggers(candidate, _row(year="2024", side="buy", delta=1.0, unrealized=-0.30)))
        self.assertFalse(candidate_triggers(candidate, _row(year="2024", side="buy", delta=1.0, unrealized=-0.10)))
        self.assertFalse(candidate_triggers(candidate, _row(year="2024", side="buy", delta=1.0, direction=-0.10, unrealized=-0.30)))
        self.assertFalse(candidate_triggers(candidate, _row(year="2024", side="buy", delta=1.0, unrealized=-0.30, mae_atr=0.10)))

    def test_momentum_candidate_can_reach_watchlist_assessment(self) -> None:
        rows: list[dict[str, object]] = []
        for year in ("2024", "2025"):
            for side in ("buy", "sell"):
                rows.extend(_row(year=year, side=side, delta=1.0) for _ in range(8))
                rows.extend(_row(year=year, side=side, delta=-1.0) for _ in range(2))
                rows.extend(_row(year=year, side=side, delta=0.0) for _ in range(5))

        candidate = MomentumCandidate("candidate", direction_max=-0.25, unrealized_max=0.0)
        result = evaluate_momentum_candidate(candidate, rows, ThresholdRefinementConfig(min_triggered_snapshots=20))

        self.assertEqual(result["triggered_snapshots"], 60)
        self.assertGreaterEqual(result["helpful_precision_pct"], 35.0)
        self.assertEqual(result["assessment"], "candidate_watchlist")

    def test_early_close_helpful_thresholds_report_auc_features(self) -> None:
        rows = [
            _row(year="2024", side="buy", delta=1.0, mae_atr=0.9),
            _row(year="2024", side="buy", delta=1.0, mae_atr=0.8),
            _row(year="2024", side="sell", delta=-1.0, mae_atr=0.1),
            _row(year="2024", side="sell", delta=-1.0, mae_atr=0.2),
        ]

        result = evaluate_early_close_helpful_thresholds(rows)

        self.assertEqual(result[1]["label_variant"], "early_close_helpful_delta_ge_0.50")
        self.assertEqual(result[1]["positive_rows"], 2)
        self.assertTrue(result[1]["best_feature"])


if __name__ == "__main__":
    unittest.main()
