from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hybrid_quant.baseline.intraday_nasdaq_contextual_research import (
    IntradayContextualResearchRunner,
    load_intraday_contextual_research_config,
)


def _contextual_frame() -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    session_days = pd.date_range("2024-01-02", periods=10, freq="D", tz="UTC")
    for day_index, day in enumerate(session_days):
        session_start = day + pd.Timedelta(hours=13, minutes=30)
        session_index = pd.date_range(session_start, periods=18, freq="5min", tz="UTC")
        base = 100.0 + (day_index * 1.2)
        short_day = day_index % 3 == 1
        mean_revert_day = day_index % 3 == 2
        for bar_index, timestamp in enumerate(session_index):
            if bar_index < 6:
                close = base + (0.12 * ((bar_index % 3) - 1))
                high = close + 0.9
                low = close - 0.9
                volume = 100 + day_index * 4 + bar_index
            elif short_day:
                drift = -0.45 * (bar_index - 5)
                close = base + drift + (0.2 if bar_index in {8, 11} else -0.1)
                high = close + (1.0 if bar_index != 10 else 1.4)
                low = close - (1.2 if bar_index != 10 else 1.8)
                volume = 210 + (bar_index * 12)
            elif mean_revert_day:
                drift = 0.15 * (bar_index - 5)
                close = base + drift + (0.7 if bar_index in {9, 13} else -0.3)
                high = close + 1.1
                low = close - 1.0
                volume = 180 + (bar_index * 10)
            else:
                drift = 0.42 * (bar_index - 5)
                close = base + drift + (0.25 if bar_index in {8, 12} else -0.05)
                high = close + (1.1 if bar_index != 10 else 1.5)
                low = close - (1.0 if bar_index != 10 else 1.6)
                volume = 220 + (bar_index * 12)

            rows.append(
                {
                    "open_time": timestamp,
                    "open": close - 0.20,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

    frame = pd.DataFrame(rows).set_index("open_time")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


class IntradayNasdaqContextualResearchFlowTests(unittest.TestCase):
    def test_contextual_research_runner_generates_artifacts(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        config_dir = repo_root / "configs"
        experiment = load_intraday_contextual_research_config(
            config_dir / "experiments" / "intraday_nasdaq_contextual_research.yaml"
        )
        runner = IntradayContextualResearchRunner(config_dir, experiment)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "contextual_intraday"
            artifacts = runner.run(
                input_frame=_contextual_frame(),
                output_dir=output_dir,
                allow_gaps=True,
                selected_variants=(
                    "active_orb_reclaim_30m_control",
                    "context_pullback_30m",
                    "context_reclaim_30m",
                ),
            )

            payload = json.loads(artifacts.comparison_path.read_text(encoding="utf-8"))
            self.assertTrue(artifacts.comparison_path.exists())
            self.assertTrue(artifacts.results_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.activity_summary_path.exists())
            self.assertTrue(artifacts.yearly_path.exists())
            self.assertTrue(artifacts.quarterly_path.exists())
            self.assertTrue(artifacts.ranking_path.exists())
            self.assertEqual(len(payload["variants"]), 3)
            self.assertIn("best_profitability_variant", payload["conclusion"])
