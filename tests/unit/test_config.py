from __future__ import annotations

import unittest
from pathlib import Path

from hybrid_quant.core import load_settings


class ConfigLoadingTests(unittest.TestCase):
    def test_loads_expected_mvp_settings(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        settings = load_settings(config_dir)

        self.assertEqual(settings.market.symbol, "BTCUSDT")
        self.assertEqual(settings.market.execution_timeframe, "5m")
        self.assertEqual(settings.market.filter_timeframe, "1H")
        self.assertEqual(settings.strategy.name, "mean_reversion_trend_regime")
        self.assertFalse(settings.rl.enabled)
        self.assertGreaterEqual(len(settings.rl.seeds), 1)
        self.assertEqual(settings.rl.train_split, "train")
        self.assertTrue(settings.risk.prop_firm_mode)
        self.assertEqual(settings.backtest.intrabar_exit_policy, "conservative")
