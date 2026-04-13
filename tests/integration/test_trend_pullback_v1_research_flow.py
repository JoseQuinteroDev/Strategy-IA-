from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hybrid_quant.baseline.trend_pullback_v1_research import (
    TrendPullbackV1ResearchRunner,
    load_trend_pullback_v1_research_config,
)


class TrendPullbackV1ResearchFlowTests(unittest.TestCase):
    def test_research_runner_generates_expected_artifacts_for_single_variant(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        experiment = load_trend_pullback_v1_research_config(
            config_dir / "experiments" / "trend_pullback_v1_research.yaml"
        )
        index = pd.date_range("2024-01-02T00:00:00Z", periods=720, freq="1min")
        base = pd.Series(range(len(index)), index=index, dtype=float) * 0.01 + 2000.0
        frame = pd.DataFrame(
            {
                "open": base,
                "high": base + 0.20,
                "low": base - 0.20,
                "close": base + 0.05,
                "volume": 100.0,
            },
            index=index,
        )
        frame.index.name = "open_time"

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            artifacts = TrendPullbackV1ResearchRunner(config_dir, experiment).run(
                input_frame=frame,
                output_dir=output_dir,
                allow_gaps=True,
                selected_variants=("core_v1",),
            )

            self.assertTrue(artifacts["comparison"].exists())
            self.assertTrue(artifacts["results"].exists())
            self.assertTrue(artifacts["summary"].exists())
            self.assertTrue(artifacts["cost_sensitivity"].exists())
            self.assertTrue(artifacts["walk_forward"].exists())
            self.assertTrue((output_dir / "variants" / "core_v1" / "baseline" / "trades.csv").exists())
            self.assertTrue((output_dir / "variants" / "core_v1" / "diagnostics" / "diagnostics.json").exists())


if __name__ == "__main__":
    unittest.main()
