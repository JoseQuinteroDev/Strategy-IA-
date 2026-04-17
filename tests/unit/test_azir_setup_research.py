import json
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from hybrid_quant.azir.replica import AzirPythonReplica, AzirReplicaConfig, OhlcvBar
from hybrid_quant.azir.setup_research import run_setup_research


def _bars() -> list[OhlcvBar]:
    start = datetime(2025, 1, 6, 14, 0)
    rows: list[OhlcvBar] = []
    for index in range(97):
        ts = start + timedelta(minutes=5 * index)
        drift = index * 0.01
        open_price = 100.0 + drift
        high = open_price + 0.20
        low = open_price - 0.20
        close = open_price + 0.05
        if ts == datetime(2025, 1, 6, 16, 30):
            high = 101.0
            low = 100.0
            close = 100.8
        rows.append(OhlcvBar(ts, open_price, high, low, close, 1.0))
    return rows


def _write_ohlcv(path: Path, bars: list[OhlcvBar]) -> None:
    lines = ["open_time,open,high,low,close,volume"]
    for bar in bars:
        lines.append(
            f"{bar.open_time:%Y-%m-%d %H:%M:%S},{bar.open},{bar.high},{bar.low},{bar.close},{bar.volume}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_protected_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "benchmark_name": "baseline_azir_protected_economic_v1",
                "metrics": {
                    "azir_with_risk_engine_v1_forced_closes_revalued": {
                        "closed_trades": 1,
                        "net_pnl": 1.0,
                        "profit_factor": 1.2,
                        "expectancy": 1.0,
                        "max_drawdown_abs": 0.0,
                        "max_consecutive_losses": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


class AzirSetupResearchTests(unittest.TestCase):
    def test_entry_offset_override_changes_pending_level(self) -> None:
        replica = AzirPythonReplica(
            _bars(),
            AzirReplicaConfig(
                allow_trend_filter=False,
                allow_atr_filter=False,
                no_trade_fridays=False,
                entry_offset_points=8.0,
            ),
        )

        opportunity = next(row for row in replica.run() if row["event_type"] == "opportunity")

        self.assertAlmostEqual(
            float(opportunity["buy_entry"]) - float(opportunity["swing_high"]),
            0.08,
        )

    def test_runner_writes_setup_research_artifacts(self) -> None:
        root = Path("artifacts") / "unit-test-azir-setup-research"
        root.mkdir(parents=True, exist_ok=True)
        m5_path = root / "m5.csv"
        protected_path = root / "protected.json"
        config_path = root / "config.yaml"
        output_dir = root / "out"
        _write_ohlcv(m5_path, _bars())
        _write_protected_report(protected_path)
        config_path.write_text(
            """
variants:
  - name: azir_current_python_proxy
    family: baseline_control
    replica:
      allow_trend_filter: false
      allow_atr_filter: false
      no_trade_fridays: false
  - name: offset_8_points
    family: entry_offset
    replica:
      allow_trend_filter: false
      allow_atr_filter: false
      no_trade_fridays: false
      entry_offset_points: 8.0
""",
            encoding="utf-8",
        )

        report = run_setup_research(
            m5_input_path=m5_path,
            protected_report_path=protected_path,
            output_dir=output_dir,
            config_path=config_path,
            symbol="XAUUSD-STD",
        )

        self.assertEqual(report["sprint"], "setup_base_research_for_azir_v1")
        self.assertTrue((output_dir / "candidate_variants.csv").exists())
        self.assertTrue((output_dir / "setup_research_summary.md").exists())
        self.assertTrue((output_dir / "offset_comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
