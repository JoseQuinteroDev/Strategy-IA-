from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hybrid_quant.baseline.intraday_hybrid_research import (
    IntradayHybridResearchRunner,
    load_intraday_hybrid_research_config,
)


def _hybrid_frame() -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    session_days = pd.date_range("2024-01-02", periods=12, freq="D", tz="UTC")
    for day_index, day in enumerate(session_days):
        session_start = day + pd.Timedelta(hours=13, minutes=30)
        session_index = pd.date_range(session_start, periods=24, freq="5min", tz="UTC")
        base = 100.0 + (day_index * 1.1)
        is_range_day = day_index % 3 == 1
        is_short_day = day_index % 3 == 2
        for bar_index, timestamp in enumerate(session_index):
            if bar_index < 6:
                close = base + (0.12 * ((bar_index % 3) - 1))
            elif is_range_day:
                close = base + (1.8 if bar_index in {8, 9} else 0.2 * ((bar_index % 4) - 1))
            elif is_short_day:
                close = base - (0.42 * (bar_index - 5)) + (0.2 if bar_index in {9, 13} else -0.1)
            else:
                close = base + (0.42 * (bar_index - 5)) + (0.2 if bar_index in {9, 13} else -0.1)
            rows.append(
                {
                    "open_time": timestamp,
                    "open": close - 0.20,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 220 + (bar_index * 8) + day_index,
                }
            )
    frame = pd.DataFrame(rows).set_index("open_time")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


class IntradayHybridResearchFlowTests(unittest.TestCase):
    def test_hybrid_research_runner_generates_artifacts(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        config_dir = repo_root / "configs"
        experiment = load_intraday_hybrid_research_config(
            config_dir / "experiments" / "intraday_hybrid_research.yaml"
        )
        runner = IntradayHybridResearchRunner(config_dir, experiment)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "intraday_hybrid"
            artifacts = runner.run(
                input_frame=_hybrid_frame(),
                output_dir=output_dir,
                allow_gaps=True,
                selected_variants=(
                    "legacy_orb_control",
                    "hybrid_pullback_value",
                    "hybrid_mean_reversion_controlled",
                ),
            )

            payload = json.loads(artifacts.comparison_path.read_text(encoding="utf-8"))
            self.assertTrue(artifacts.comparison_path.exists())
            self.assertTrue(artifacts.results_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.ranking_path.exists())
            self.assertTrue(artifacts.split_results_path.exists())
            self.assertTrue(artifacts.cost_sensitivity_path.exists())
            self.assertEqual(len(payload["variants"]), 3)
            self.assertIn("final_baseline_variant", payload["conclusion"])

            signals = pd.read_csv(artifacts.variant_artifacts["hybrid_pullback_value"].artifact_dir / "signals.csv")
            self.assertIn("outside_session", signals.columns)
            self.assertIn("blocked_by_filter", signals.columns)
            self.assertIn("entry_session_window_utc", signals.columns)


if __name__ == "__main__":
    unittest.main()
