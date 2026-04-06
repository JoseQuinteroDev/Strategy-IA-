from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hybrid_quant.baseline.diagnostics import BaselineDiagnosticsRunner
from hybrid_quant.baseline.runner import BaselineRunner


def _orb_frame() -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    session_days = pd.date_range("2024-01-02", periods=4, freq="D", tz="UTC")
    for day_index, day in enumerate(session_days):
        session_start = day + pd.Timedelta(hours=13, minutes=30)
        session_index = pd.date_range(session_start, periods=90, freq="5min", tz="UTC")
        session_base = 100.0 + (day_index * 1.2)
        for bar_index, timestamp in enumerate(session_index):
            if bar_index < 6:
                close = session_base + (0.15 * math.sin(bar_index))
                high = session_base + 1.0 + (bar_index % 2) * 0.1
                low = session_base - 1.0 - (bar_index % 2) * 0.1
                volume = 100 + bar_index
            elif day_index == len(session_days) - 1 and bar_index == 6:
                close = session_base + 1.75
                high = close + 1.0
                low = close - 0.8
                volume = 320.0
            else:
                drift = 0.25 if day_index == len(session_days) - 1 else 0.03
                offset = 1.75 if day_index == len(session_days) - 1 else 0.8
                close = session_base + offset + (drift * (bar_index - 6))
                high = close + 0.9
                low = close - 0.7
                volume = 180 + (bar_index % 5) * 8

            rows.append(
                {
                    "open_time": timestamp,
                    "open": close - 0.3,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

    frame = pd.DataFrame(rows).set_index("open_time")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


class BaselineNQOrbFlowTests(unittest.TestCase):
    def test_orb_baseline_runner_and_diagnostics_generate_artifacts(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        runner = BaselineRunner.from_config(config_dir, variant_name="baseline_nq_orb")
        runner.application.strategy.minimum_momentum_abs = 0.0
        runner.application.strategy.minimum_relative_volume = 0.0
        runner.application.strategy.max_breakout_distance_atr = 0.5

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "baseline_nq_orb"
            diagnostics_dir = Path(tmp_dir) / "baseline_nq_orb_diagnostics"

            artifacts = runner.run(
                output_dir=output_dir,
                input_frame=_orb_frame(),
                allow_gaps=True,
            )
            diagnostics = BaselineDiagnosticsRunner(runner.application).run(
                artifact_dir=artifacts.output_dir,
                output_dir=diagnostics_dir,
            )

            report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(diagnostics.diagnostics_path.exists())
            self.assertTrue((diagnostics.output_dir / "opening_range_width_breakdown.csv").exists())
            self.assertTrue((diagnostics.output_dir / "first_breakout_breakdown.csv").exists())
            self.assertTrue((diagnostics.output_dir / "quarterly_breakdown.csv").exists())
            self.assertTrue((diagnostics.output_dir / "yearly_equity_curve.csv").exists())
            self.assertEqual(report["symbol"], "NQ")
            self.assertGreaterEqual(report["number_of_trades"], 1)

            entry_mode_breakdown = pd.read_csv(diagnostics.output_dir / "entry_mode_breakdown.csv")
            breakout_distance_breakdown = pd.read_csv(diagnostics.output_dir / "breakout_distance_breakdown.csv")
            opening_range_width_breakdown = pd.read_csv(diagnostics.output_dir / "opening_range_width_breakdown.csv")
            enriched_trades = pd.read_csv(diagnostics.output_dir / "enriched_trades.csv")

            self.assertIn("breakout_close_entry", set(entry_mode_breakdown["entry_mode"].dropna().astype(str)))
            self.assertFalse(
                (
                    (breakout_distance_breakdown["trades"] > 0)
                    & breakout_distance_breakdown["breakout_distance_bucket"].isna()
                ).any()
            )
            self.assertFalse(
                (
                    (opening_range_width_breakdown["trades"] > 0)
                    & opening_range_width_breakdown["opening_range_width_bucket"].isna()
                ).any()
            )
            self.assertTrue(enriched_trades["entry_mode"].notna().all())
