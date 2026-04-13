from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from hybrid_quant.baseline.intraday_hybrid_realism import (
    _audit_dataset,
    _period_activity_metrics,
    load_intraday_hybrid_realism_config,
)


class IntradayHybridRealismTests(unittest.TestCase):
    def test_dataset_audit_distinguishes_schema_from_statistical_normalization(self) -> None:
        index = pd.date_range("2024-01-01T14:00:00Z", periods=4, freq="5min", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [17000.0, 17005.0, 17008.0, 17010.0],
                "high": [17008.0, 17010.0, 17012.0, 17016.0],
                "low": [16995.0, 17001.0, 17004.0, 17008.0],
                "close": [17005.0, 17008.0, 17010.0, 17014.0],
                "volume": [100.0, 120.0, 140.0, 130.0],
            },
            index=index,
        )

        audit = _audit_dataset(frame=frame, input_path="synthetic.csv")

        self.assertTrue(audit["schema_normalized_to_internal_ohlcv"])
        self.assertTrue(audit["dominant_interval_is_5m"])
        self.assertFalse(audit["statistical_price_normalization_detected"])
        self.assertIn("Unknown", audit["raw_back_adjusted_or_stitched_status"])

    def test_period_activity_metrics_reports_weekly_frequency(self) -> None:
        trades = pd.DataFrame(
            {
                "entry_timestamp": pd.to_datetime(
                    ["2024-01-01T14:05:00Z", "2024-01-08T14:05:00Z"],
                    utc=True,
                ),
                "net_pnl": [10.0, -5.0],
            }
        )

        metrics = _period_activity_metrics(
            trades=trades,
            start=pd.Timestamp("2024-01-01T00:00:00Z"),
            end=pd.Timestamp("2024-01-15T00:00:00Z"),
        )

        self.assertAlmostEqual(metrics["trades_per_week_avg"], 1.0)
        self.assertGreater(metrics["percentage_of_weeks_with_trade"], 0.0)

    def test_loads_default_realism_config(self) -> None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "intraday_hybrid_realism.yaml"
        experiment = load_intraday_hybrid_realism_config(config_path)

        self.assertEqual(experiment.instrument_scenarios[1].name, "mnq_realistic_base")
        self.assertEqual(experiment.cost_scenarios[-1].name, "costs_x3")


if __name__ == "__main__":
    unittest.main()
