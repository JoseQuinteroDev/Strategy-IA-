from __future__ import annotations

import unittest
from pathlib import Path

from hybrid_quant.core import load_settings
from hybrid_quant.baseline.variants import load_variant_settings
from hybrid_quant.baseline.orb_ablation import load_orb_ablation_config
from hybrid_quant.baseline.orb_frequency_expansion import load_orb_frequency_expansion_config
from hybrid_quant.baseline.orb_frequency_push import load_orb_frequency_push_config
from hybrid_quant.baseline.orb_intraday_active_research import (
    load_orb_intraday_active_research_config,
)
from hybrid_quant.baseline.intraday_nasdaq_contextual_research import (
    load_intraday_contextual_research_config,
)
from hybrid_quant.baseline.intraday_hybrid_research import (
    load_intraday_hybrid_research_config,
)
from hybrid_quant.baseline.intraday_hybrid_realism import (
    load_intraday_hybrid_realism_config,
)
from hybrid_quant.baseline.trend_pullback_v1_research import (
    load_trend_pullback_v1_research_config,
)
from hybrid_quant.baseline.session_trend_30m_zoom import load_session_trend_30m_zoom_config


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

    def test_loads_baseline_nq_orb_variant_overrides(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        settings = load_variant_settings(config_dir, "baseline_nq_orb")

        self.assertEqual(settings.market.symbol, "NQ")
        self.assertEqual(settings.strategy.family, "opening_range_breakout")
        self.assertEqual(settings.strategy.variant_name, "baseline_nq_orb")
        self.assertEqual(settings.strategy.entry_mode, "breakout_close_entry")
        self.assertEqual(settings.strategy.opening_range_minutes, 30)
        self.assertEqual(settings.strategy.max_breakouts_per_day, 1)
        self.assertTrue(settings.strategy.use_ema_200_1h_slope)
        self.assertGreater(settings.strategy.minimum_relative_volume, 0.0)

    def test_loads_width_wider_orb_reference_variant(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        settings = load_variant_settings(config_dir, "orb30_close_multi_no_slope_no_rvol_width_wider")

        self.assertEqual(settings.market.symbol, "NQ")
        self.assertEqual(settings.strategy.family, "opening_range_breakout")
        self.assertEqual(settings.strategy.variant_name, "orb30_close_multi_no_slope_no_rvol_width_wider")
        self.assertEqual(settings.strategy.opening_range_minutes, 30)
        self.assertFalse(settings.strategy.use_ema_200_1h_slope)
        self.assertEqual(settings.strategy.minimum_relative_volume, 0.0)
        self.assertEqual(settings.strategy.max_breakouts_per_day, 3)
        self.assertEqual(settings.risk.max_trades_per_day, 3)
        self.assertEqual(settings.strategy.minimum_opening_range_width_atr, 0.40)
        self.assertEqual(settings.strategy.maximum_opening_range_width_atr, 2.80)

    def test_loads_extension_laxer_and_width_laxer_extension_laxer_variants(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        extension_laxer = load_variant_settings(config_dir, "orb30_close_multi_no_slope_no_rvol_extension_laxer")
        width_laxer = load_variant_settings(
            config_dir,
            "orb30_close_multi_no_slope_no_rvol_width_laxer_extension_laxer",
        )

        self.assertEqual(extension_laxer.strategy.max_breakout_distance_atr, 0.45)
        self.assertEqual(extension_laxer.strategy.minimum_opening_range_width_atr, 0.50)
        self.assertEqual(width_laxer.strategy.max_breakout_distance_atr, 0.45)
        self.assertEqual(width_laxer.strategy.minimum_opening_range_width_atr, 0.30)
        self.assertEqual(width_laxer.strategy.maximum_opening_range_width_atr, 3.00)
        self.assertFalse(width_laxer.strategy.use_ema_200_1h_slope)

    def test_loads_orb_ablation_matrix_config(self) -> None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "orb_ablation.yaml"
        experiment = load_orb_ablation_config(config_path)

        self.assertEqual(experiment.base_variant, "baseline_nq_orb")
        self.assertEqual(len(experiment.dimensions), 5)
        self.assertEqual(experiment.dimensions[0].key, "opening_range_minutes")
        self.assertEqual(len(experiment.dimensions[0].options), 2)

    def test_loads_orb_frequency_expansion_config(self) -> None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "orb_frequency_expansion.yaml"
        experiment = load_orb_frequency_expansion_config(config_path)

        self.assertEqual(experiment.base_variant, "orb30_close_multi_no_slope_no_rvol_width_wider")
        self.assertGreaterEqual(experiment.summary_thresholds.minimum_frequency_uplift_ratio, 1.0)
        self.assertGreaterEqual(len(experiment.variants), 8)
        self.assertEqual(experiment.variants[0].name, "reference")

    def test_loads_orb_frequency_push_config(self) -> None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "orb_frequency_push.yaml"
        experiment = load_orb_frequency_push_config(config_path)

        self.assertEqual(experiment.base_variant, "orb30_close_multi_no_slope_no_rvol_width_laxer_extension_laxer")
        self.assertGreaterEqual(experiment.summary_thresholds.target_trades_per_week_avg, 1.0)
        self.assertGreaterEqual(len(experiment.variants), 10)
        self.assertEqual(experiment.variants[0].name, "width_wider_control")

    def test_loads_baseline_nq_intraday_orb_active_variant(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        settings = load_variant_settings(config_dir, "baseline_nq_intraday_orb_active")

        self.assertEqual(settings.market.symbol, "NQ")
        self.assertEqual(settings.strategy.family, "orb_intraday_active")
        self.assertEqual(settings.strategy.variant_name, "baseline_nq_intraday_orb_active")
        self.assertEqual(settings.strategy.entry_mode, "breakout_continuation")
        self.assertEqual(settings.strategy.max_breakouts_per_day, 4)
        self.assertTrue(settings.strategy.use_intraday_vwap_filter)
        self.assertAlmostEqual(settings.strategy.maximum_pullback_depth_atr, 0.80)

    def test_loads_orb_intraday_active_research_config(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[2] / "configs" / "experiments" / "orb_intraday_active_research.yaml"
        )
        experiment = load_orb_intraday_active_research_config(config_path)

        self.assertEqual(experiment.base_variant, "baseline_nq_intraday_orb_active")
        self.assertGreaterEqual(experiment.summary_thresholds.target_trades_per_week_avg, 2.0)
        self.assertGreaterEqual(len(experiment.variants), 6)
        self.assertEqual(experiment.variants[0].name, "legacy_orb_control")

    def test_loads_baseline_nq_intraday_contextual_variant(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        settings = load_variant_settings(config_dir, "baseline_nq_intraday_contextual")

        self.assertEqual(settings.market.symbol, "NQ")
        self.assertEqual(settings.strategy.family, "intraday_nasdaq_contextual")
        self.assertEqual(settings.strategy.variant_name, "baseline_nq_intraday_contextual")
        self.assertEqual(settings.strategy.entry_mode, "context_pullback_continuation")
        self.assertEqual(settings.strategy.opening_range_minutes, 30)
        self.assertEqual(settings.strategy.max_breakouts_per_day, 4)
        self.assertTrue(settings.strategy.use_intraday_vwap_filter)
        self.assertTrue(settings.strategy.use_intraday_ema20_filter)

    def test_loads_baseline_intraday_hybrid_variant(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        settings = load_variant_settings(config_dir, "baseline_intraday_hybrid")

        self.assertEqual(settings.market.symbol, "MNQ")
        self.assertEqual(settings.strategy.family, "intraday_hybrid_contextual")
        self.assertEqual(settings.strategy.variant_name, "baseline_intraday_hybrid")
        self.assertEqual(settings.strategy.entry_mode, "macro_pullback_continuation")
        self.assertTrue(settings.strategy.use_ema_200_1h_trend_filter)
        self.assertTrue(settings.strategy.use_ema_200_1h_slope)
        self.assertTrue(settings.strategy.use_macro_bias_filter)
        self.assertTrue(settings.strategy.enforce_entry_session)
        self.assertEqual(settings.strategy.entry_session_start_hour_utc, 14)
        self.assertEqual(settings.strategy.entry_session_start_minute_utc, 0)
        self.assertEqual(settings.strategy.entry_session_end_hour_utc, 19)
        self.assertEqual(settings.strategy.entry_session_end_minute_utc, 0)
        self.assertEqual(settings.strategy.allowed_hours_utc, [14, 15, 16, 17, 18])
        self.assertTrue(settings.strategy.close_on_session_end)
        self.assertTrue(settings.risk.block_outside_session)
        self.assertEqual(settings.risk.session_start_hour_utc, 14)
        self.assertEqual(settings.risk.session_end_hour_utc, 19)
        self.assertEqual(settings.risk.max_daily_loss, 0.025)
        self.assertEqual(settings.backtest.point_value, 2.0)
        self.assertEqual(settings.backtest.contract_step, 1.0)
        self.assertEqual(settings.backtest.gap_exit_policy, "open")

    def test_loads_baseline_trend_pullback_v1_gold_variant(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        settings = load_variant_settings(config_dir, "baseline_trend_pullback_v1")

        self.assertEqual(settings.market.symbol, "XAUUSD")
        self.assertEqual(settings.market.execution_timeframe, "1m")
        self.assertEqual(settings.strategy.family, "baseline_trend_pullback_v1")
        self.assertEqual(settings.strategy.entry_mode, "core_v1")
        self.assertEqual(settings.strategy.entry_session_timezone, "Europe/Madrid")
        self.assertEqual(settings.strategy.entry_session_windows, ["09:00-11:00", "14:00-16:30"])
        self.assertEqual(settings.risk.session_windows, ["09:00-11:00", "14:00-16:30"])
        self.assertEqual(settings.risk.max_consecutive_losses_per_day, 2)
        self.assertAlmostEqual(settings.risk.max_risk_per_trade, 0.005)
        self.assertEqual(settings.backtest.point_value, 100.0)
        self.assertEqual(settings.backtest.gap_exit_policy, "open")

    def test_loads_trend_pullback_v1_research_config(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[2] / "configs" / "experiments" / "trend_pullback_v1_research.yaml"
        )
        experiment = load_trend_pullback_v1_research_config(config_path)

        self.assertEqual(experiment.base_variant, "baseline_trend_pullback_v1")
        self.assertEqual(len(experiment.variants), 5)
        self.assertEqual(experiment.variants[0].name, "core_v1")
        self.assertEqual(experiment.variants[1].name, "core_v1_macd")
        self.assertEqual(experiment.variants[2].name, "core_v1_no_m1")

    def test_loads_session_trend_30m_variant(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        settings = load_variant_settings(config_dir, "session_trend_30m")

        self.assertEqual(settings.market.symbol, "NQ")
        self.assertEqual(settings.strategy.family, "intraday_nasdaq_contextual")
        self.assertEqual(settings.strategy.variant_name, "session_trend_30m")
        self.assertEqual(settings.strategy.entry_mode, "session_trend_continuation")
        self.assertEqual(settings.strategy.opening_range_minutes, 30)
        self.assertTrue(settings.strategy.use_ema_200_1h_trend_filter)
        self.assertTrue(settings.strategy.use_ema_200_1h_slope)
        self.assertTrue(settings.strategy.use_intraday_vwap_filter)
        self.assertTrue(settings.strategy.use_intraday_ema50_alignment)
        self.assertTrue(settings.strategy.use_opening_range_mid_filter)

    def test_loads_shorts_strict_clean_hours_and_long_only_clean_hours_variants(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        shorts = load_variant_settings(config_dir, "shorts_strict_clean_hours")
        longs = load_variant_settings(config_dir, "long_only_clean_hours")

        self.assertEqual(shorts.strategy.variant_name, "shorts_strict_clean_hours")
        self.assertEqual(shorts.strategy.allowed_hours_short_utc, [14, 16, 18, 19])
        self.assertEqual(longs.strategy.variant_name, "long_only_clean_hours")
        self.assertEqual(longs.strategy.allowed_sides, ["long"])
        self.assertEqual(longs.strategy.allowed_hours_utc, [14, 16, 17, 18, 19])

    def test_loads_intraday_contextual_research_config(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "experiments"
            / "intraday_nasdaq_contextual_research.yaml"
        )
        experiment = load_intraday_contextual_research_config(config_path)

        self.assertEqual(experiment.base_variant, "baseline_nq_intraday_contextual")
        self.assertGreaterEqual(experiment.summary_thresholds.minimum_profit_factor, 1.0)
        self.assertGreaterEqual(len(experiment.variants), 6)
        self.assertEqual(experiment.variants[0].name, "active_orb_reclaim_30m_control")

    def test_loads_intraday_hybrid_research_config(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "experiments"
            / "intraday_hybrid_research.yaml"
        )
        experiment = load_intraday_hybrid_research_config(config_path)

        self.assertEqual(experiment.base_variant, "baseline_intraday_hybrid")
        self.assertGreaterEqual(experiment.summary_thresholds.minimum_profit_factor, 1.10)
        self.assertGreaterEqual(experiment.summary_thresholds.minimum_trades, 40)
        self.assertGreaterEqual(len(experiment.variants), 5)
        self.assertEqual(experiment.variants[0].name, "legacy_orb_control")

    def test_loads_intraday_hybrid_realism_config(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "experiments"
            / "intraday_hybrid_realism.yaml"
        )
        experiment = load_intraday_hybrid_realism_config(config_path)

        self.assertEqual(experiment.base_variant, "baseline_intraday_hybrid")
        self.assertGreaterEqual(len(experiment.instrument_scenarios), 3)
        self.assertGreaterEqual(len(experiment.timezone_scenarios), 3)
        self.assertGreaterEqual(len(experiment.cost_scenarios), 4)
        self.assertTrue(experiment.walk_forward.enabled)
        self.assertIn("hybrid_pullback_value", experiment.variant_comparison.selected_variants)

    def test_loads_session_trend_30m_zoom_config(self) -> None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "session_trend_30m_zoom.yaml"
        experiment = load_session_trend_30m_zoom_config(config_path)

        self.assertEqual(experiment.base_variant, "session_trend_30m")
        self.assertGreaterEqual(experiment.summary_thresholds.minimum_profit_factor, 1.05)
        self.assertGreaterEqual(len(experiment.variants), 15)
        self.assertEqual(experiment.variants[0].name, "active_orb_reclaim_30m_control")

    def test_loads_shorts_strict_clean_hours_extended_config(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "experiments"
            / "shorts_strict_clean_hours_extended.yaml"
        )
        experiment = load_session_trend_30m_zoom_config(config_path)

        self.assertEqual(experiment.base_variant, "shorts_strict_clean_hours")
        self.assertGreaterEqual(experiment.summary_thresholds.minimum_profit_factor, 1.10)
        self.assertGreaterEqual(len(experiment.variants), 12)
        self.assertEqual(experiment.variants[0].name, "session_trend_30m_original")

    def test_loads_baseline_trend_nasdaq_v2_variant_overrides(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        settings = load_variant_settings(config_dir, "baseline_trend_nasdaq_v2")

        self.assertEqual(settings.strategy.variant_name, "baseline_trend_nasdaq_v2")
        self.assertEqual(settings.strategy.allowed_hours_utc, [13, 17, 19])
        self.assertEqual(settings.strategy.breakout_buffer_atr, 0.20)
        self.assertEqual(settings.strategy.minimum_momentum_abs, 0.0040)
        self.assertEqual(settings.strategy.minimum_candle_range_atr, 0.90)

    def test_loads_baseline_trend_nasdaq_v2_directional_variants(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        long_only = load_variant_settings(config_dir, "baseline_trend_nasdaq_v2_long_only")
        short_only = load_variant_settings(config_dir, "baseline_trend_nasdaq_v2_short_only")

        self.assertEqual(long_only.strategy.allowed_sides, ["long"])
        self.assertEqual(short_only.strategy.allowed_sides, ["short"])
        self.assertEqual(long_only.market.symbol, "NQ")
        self.assertEqual(short_only.market.symbol, "NQ")
