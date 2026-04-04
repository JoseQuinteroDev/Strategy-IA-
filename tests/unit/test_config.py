from __future__ import annotations

import unittest
from pathlib import Path

from hybrid_quant.core import load_settings
from hybrid_quant.baseline.variants import load_variant_settings


class ConfigLoadingTests(unittest.TestCase):
    def test_loads_expected_mvp_settings(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        settings = load_settings(config_dir)

        self.assertEqual(settings.market.symbol, "BTCUSDT")
        self.assertEqual(settings.market.execution_timeframe, "5m")
        self.assertEqual(settings.market.filter_timeframe, "1H")
        self.assertEqual(settings.strategy.name, "mean_reversion_trend_regime")
        self.assertEqual(settings.strategy.family, "mean_reversion")
        self.assertFalse(settings.rl.enabled)
        self.assertGreaterEqual(len(settings.rl.seeds), 1)
        self.assertEqual(settings.rl.train_split, "train")
        self.assertEqual(settings.env.action_space, "candidate_trade_discrete")
        self.assertEqual(settings.env.effective_state_context_bars, 64)
        self.assertTrue(settings.risk.prop_firm_mode)
        self.assertEqual(settings.backtest.intrabar_exit_policy, "conservative")
        self.assertEqual(settings.validation.walk_forward_train_ratio, 0.60)
        self.assertEqual(settings.validation.monte_carlo_simulations, 500)
        self.assertGreaterEqual(len(settings.validation.cost_scenarios), 3)

    def test_loads_baseline_v2_variant_overrides(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        settings = load_variant_settings(config_dir, "baseline_v2")

        self.assertEqual(settings.strategy.variant_name, "baseline_v2")
        self.assertEqual(settings.strategy.signal_cooldown_bars, 6)
        self.assertEqual(settings.strategy.atr_multiple_target, 1.5)
        self.assertEqual(settings.strategy.adx_threshold, 18.0)
        self.assertIn(13, settings.strategy.blocked_hours_utc)
        self.assertTrue(settings.strategy.exclude_weekends)
        self.assertEqual(settings.strategy.minimum_target_to_cost_ratio, 3.0)

    def test_loads_baseline_v3_variant_overrides(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        settings = load_variant_settings(config_dir, "baseline_v3")

        self.assertEqual(settings.strategy.variant_name, "baseline_v3")
        self.assertEqual(settings.strategy.signal_cooldown_bars, 6)
        self.assertEqual(settings.strategy.atr_multiple_target, 1.5)
        self.assertEqual(settings.strategy.adx_threshold, 18.0)
        self.assertTrue(settings.strategy.exclude_weekends)
        self.assertEqual(settings.strategy.minimum_target_to_cost_ratio, 3.0)
        self.assertTrue({14, 15, 20, 21}.issubset(set(settings.strategy.blocked_hours_utc)))

    def test_loads_baseline_trend_nasdaq_variant_overrides(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        settings = load_variant_settings(config_dir, "baseline_trend_nasdaq")

        self.assertEqual(settings.market.symbol, "NQ")
        self.assertEqual(settings.market.venue, "cme")
        self.assertEqual(settings.strategy.family, "trend_breakout")
        self.assertEqual(settings.strategy.variant_name, "baseline_trend_nasdaq")
        self.assertIsNone(settings.strategy.exit_zscore)
        self.assertEqual(settings.strategy.breakout_lookback_bars, 20)
        self.assertEqual(settings.strategy.momentum_lookback_bars, 20)
        self.assertTrue(settings.data.allow_gaps)
        self.assertIn(13, settings.strategy.allowed_hours_utc)
