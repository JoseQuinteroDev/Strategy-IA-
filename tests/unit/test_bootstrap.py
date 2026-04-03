from __future__ import annotations

import unittest
from pathlib import Path

from hybrid_quant import build_application


class BootstrapTests(unittest.TestCase):
    def test_build_application_wires_main_components(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        application = build_application(config_dir)
        summary = application.summary()

        self.assertEqual(summary["symbol"], "BTCUSDT")
        self.assertEqual(summary["execution_timeframe"], "5m")
        self.assertEqual(summary["filter_timeframe"], "1H")
        self.assertEqual(summary["strategy"], "mean_reversion_trend_regime")
        self.assertEqual(application.rl_trainer.algorithm, "PPO")
        self.assertEqual(application.data_source.provider_name, "binance")

