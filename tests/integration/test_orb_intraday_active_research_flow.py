from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hybrid_quant.baseline.orb_intraday_active_research import (
    OrbIntradayActiveResearchRunner,
    load_orb_intraday_active_research_config,
)


def _active_orb_frame() -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    session_days = pd.date_range("2024-01-02", periods=8, freq="D", tz="UTC")
    for day_index, day in enumerate(session_days):
        session_start = day + pd.Timedelta(hours=13, minutes=30)
        session_index = pd.date_range(session_start, periods=14, freq="5min", tz="UTC")
        base = 100.0 + (day_index * 1.5)
        short_day = day_index % 2 == 1
        for bar_index, timestamp in enumerate(session_index):
            if bar_index < 6:
                close = base + (0.15 * ((bar_index % 3) - 1))
                high = base + 1.0 + (bar_index % 2) * 0.1
                low = base - 1.0 - (bar_index % 2) * 0.1
                volume = 100 + day_index * 5 + bar_index
            elif short_day:
                sequence = [
                    (base - 1.8, base - 1.0, base - 3.2),
                    (base - 2.4, base - 1.5, base - 3.4),
                    (base - 2.1, base - 1.6, base - 3.2),
                    (base + 0.2, base - 0.8, base - 1.0),
                    (base - 2.0, base - 1.2, base - 3.1),
                    (base - 2.6, base - 1.8, base - 3.5),
                    (base - 2.3, base - 1.6, base - 3.2),
                    (base - 2.7, base - 1.9, base - 3.6),
                ][bar_index - 6]
                close, high, low = sequence
                volume = 240 + (bar_index * 10)
            else:
                sequence = [
                    (base + 1.8, base + 3.2, base + 1.0),
                    (base + 2.4, base + 3.4, base + 1.5),
                    (base + 2.1, base + 3.2, base + 1.0),
                    (base - 0.2, base + 1.0, base - 1.0),
                    (base + 2.0, base + 3.1, base + 1.1),
                    (base + 2.6, base + 3.5, base + 1.8),
                    (base + 2.3, base + 3.2, base + 1.6),
                    (base + 2.7, base + 3.6, base + 1.9),
                ][bar_index - 6]
                close, high, low = sequence
                volume = 240 + (bar_index * 10)

            rows.append(
                {
                    "open_time": timestamp,
                    "open": close - 0.25,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

    frame = pd.DataFrame(rows).set_index("open_time")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


class OrbIntradayActiveResearchFlowTests(unittest.TestCase):
    def test_research_runner_generates_artifacts_for_new_family(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        config_dir = repo_root / "configs"
        experiment = load_orb_intraday_active_research_config(
            config_dir / "experiments" / "orb_intraday_active_research.yaml"
        )
        runner = OrbIntradayActiveResearchRunner(config_dir, experiment)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "orb_intraday_active"
            artifacts = runner.run(
                input_frame=_active_orb_frame(),
                output_dir=output_dir,
                allow_gaps=True,
                selected_variants=(
                    "legacy_orb_control",
                    "continuation_30m",
                    "pullback_30m",
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
            self.assertIn("best_balance_variant", payload["conclusion"])
