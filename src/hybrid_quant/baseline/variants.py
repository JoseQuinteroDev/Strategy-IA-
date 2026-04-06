from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from hybrid_quant.bootstrap import TradingApplication, build_application_from_settings
from hybrid_quant.core import Settings, apply_settings_overrides, load_settings


BASELINE_VARIANTS = {
    "baseline_v1": None,
    "baseline_v2": Path("variants") / "baseline_v2.yaml",
    "baseline_v3": Path("variants") / "baseline_v3.yaml",
    "baseline_nq_orb": Path("variants") / "baseline_nq_orb.yaml",
    "baseline_nq_intraday_orb_active": Path("variants") / "baseline_nq_intraday_orb_active.yaml",
    "baseline_nq_intraday_contextual": Path("variants") / "baseline_nq_intraday_contextual.yaml",
    "session_trend_30m": Path("variants") / "session_trend_30m.yaml",
    "shorts_strict_clean_hours": Path("variants") / "shorts_strict_clean_hours.yaml",
    "long_only_clean_hours": Path("variants") / "long_only_clean_hours.yaml",
    "orb30_close_multi_no_slope_no_rvol": Path("variants") / "orb30_close_multi_no_slope_no_rvol.yaml",
    "orb30_close_multi_no_slope_no_rvol_extension_laxer": (
        Path("variants") / "orb30_close_multi_no_slope_no_rvol_extension_laxer.yaml"
    ),
    "orb30_close_multi_no_slope_no_rvol_width_laxer_extension_laxer": (
        Path("variants") / "orb30_close_multi_no_slope_no_rvol_width_laxer_extension_laxer.yaml"
    ),
    "orb30_close_multi_no_slope_no_rvol_width_wider": (
        Path("variants") / "orb30_close_multi_no_slope_no_rvol_width_wider.yaml"
    ),
    "baseline_trend_nasdaq": Path("variants") / "baseline_trend_nasdaq.yaml",
    "baseline_trend_nasdaq_v2": Path("variants") / "baseline_trend_nasdaq_v2.yaml",
    "baseline_trend_nasdaq_v2_long_only": Path("variants") / "baseline_trend_nasdaq_v2_long_only.yaml",
    "baseline_trend_nasdaq_v2_short_only": Path("variants") / "baseline_trend_nasdaq_v2_short_only.yaml",
    "baseline_trend_nasdaq_v3_short_bias": Path("variants") / "baseline_trend_nasdaq_v3_short_bias.yaml",
}


def load_variant_settings(config_dir: str | Path, variant_name: str) -> Settings:
    if variant_name not in BASELINE_VARIANTS:
        raise ValueError(f"Unknown baseline variant: {variant_name}")

    settings = load_settings(config_dir)
    relative_override_path = BASELINE_VARIANTS[variant_name]
    if relative_override_path is None:
        return settings

    override_path = Path(config_dir) / relative_override_path
    if not override_path.exists():
        raise FileNotFoundError(f"Variant override file not found: {override_path}")

    with override_path.open("r", encoding="utf-8") as handle:
        overrides = yaml.safe_load(handle) or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"Variant overrides must be a mapping: {override_path}")
    return apply_settings_overrides(settings, overrides)


def build_variant_application(config_dir: str | Path, variant_name: str) -> TradingApplication:
    settings = load_variant_settings(config_dir, variant_name)
    return build_application_from_settings(settings)


def variant_override_path(config_dir: str | Path, variant_name: str) -> Path | None:
    relative_path = BASELINE_VARIANTS.get(variant_name)
    if relative_path is None:
        return None
    return Path(config_dir) / relative_path


def variant_summary_payload(settings: Settings) -> dict[str, Any]:
    payload = {
        "name": settings.strategy.name,
        "variant_name": settings.strategy.variant_name,
        "family": settings.strategy.family,
        "symbol": settings.market.symbol,
        "execution_timeframe": settings.market.execution_timeframe,
        "filter_timeframe": settings.market.filter_timeframe,
        "entry_zscore": settings.strategy.entry_zscore,
        "exit_zscore": settings.strategy.exit_zscore,
        "signal_cooldown_bars": settings.strategy.signal_cooldown_bars,
        "atr_multiple_stop": settings.strategy.atr_multiple_stop,
        "atr_multiple_target": settings.strategy.atr_multiple_target,
        "adx_threshold": settings.strategy.adx_threshold,
        "blocked_hours_utc": list(settings.strategy.blocked_hours_utc),
        "allowed_hours_utc": list(settings.strategy.allowed_hours_utc),
        "allowed_hours_long_utc": list(settings.strategy.allowed_hours_long_utc),
        "allowed_hours_short_utc": list(settings.strategy.allowed_hours_short_utc),
        "allowed_weekdays": list(settings.strategy.allowed_weekdays),
        "allowed_sides": list(settings.strategy.allowed_sides),
        "exclude_weekends": settings.strategy.exclude_weekends,
        "entry_mode": settings.strategy.entry_mode,
        "opening_range_minutes": settings.strategy.opening_range_minutes,
        "retest_max_bars": settings.strategy.retest_max_bars,
        "minimum_anchor_distance_atr": settings.strategy.minimum_anchor_distance_atr,
        "minimum_expected_move_bps": settings.strategy.minimum_expected_move_bps,
        "minimum_target_to_cost_ratio": settings.strategy.minimum_target_to_cost_ratio,
        "estimated_round_trip_cost_bps": settings.strategy.estimated_round_trip_cost_bps,
    }
    if settings.strategy.family == "trend_breakout":
        payload.update(
            {
                "breakout_lookback_bars": settings.strategy.breakout_lookback_bars,
                "breakout_buffer_atr": settings.strategy.breakout_buffer_atr,
                "minimum_breakout_range_atr": settings.strategy.minimum_breakout_range_atr,
                "momentum_lookback_bars": settings.strategy.momentum_lookback_bars,
                "minimum_momentum_abs": settings.strategy.minimum_momentum_abs,
                "minimum_candle_range_atr": settings.strategy.minimum_candle_range_atr,
            }
        )
    if settings.strategy.family == "opening_range_breakout":
        payload.update(
            {
                "momentum_lookback_bars": settings.strategy.momentum_lookback_bars,
                "minimum_momentum_abs": settings.strategy.minimum_momentum_abs,
                "minimum_candle_range_atr": settings.strategy.minimum_candle_range_atr,
                "use_ema_200_1h_slope": settings.strategy.use_ema_200_1h_slope,
                "minimum_opening_range_width_atr": settings.strategy.minimum_opening_range_width_atr,
                "maximum_opening_range_width_atr": settings.strategy.maximum_opening_range_width_atr,
                "minimum_relative_volume": settings.strategy.minimum_relative_volume,
                "max_breakout_distance_atr": settings.strategy.max_breakout_distance_atr,
                "max_breakouts_per_day": settings.strategy.max_breakouts_per_day,
            }
        )
    if settings.strategy.family == "orb_intraday_active":
        payload.update(
            {
                "momentum_lookback_bars": settings.strategy.momentum_lookback_bars,
                "minimum_momentum_abs": settings.strategy.minimum_momentum_abs,
                "minimum_candle_range_atr": settings.strategy.minimum_candle_range_atr,
                "use_ema_200_1h_slope": settings.strategy.use_ema_200_1h_slope,
                "minimum_opening_range_width_atr": settings.strategy.minimum_opening_range_width_atr,
                "maximum_opening_range_width_atr": settings.strategy.maximum_opening_range_width_atr,
                "minimum_relative_volume": settings.strategy.minimum_relative_volume,
                "max_breakout_distance_atr": settings.strategy.max_breakout_distance_atr,
                "max_breakouts_per_day": settings.strategy.max_breakouts_per_day,
                "minimum_acceptance_bars": settings.strategy.minimum_acceptance_bars,
                "maximum_pullback_depth_atr": settings.strategy.maximum_pullback_depth_atr,
                "use_intraday_vwap_filter": settings.strategy.use_intraday_vwap_filter,
                "use_intraday_ema20_filter": settings.strategy.use_intraday_ema20_filter,
                "breakout_buffer_atr": settings.strategy.breakout_buffer_atr,
            }
        )
    if settings.strategy.family == "intraday_nasdaq_contextual":
        payload.update(
            {
                "momentum_lookback_bars": settings.strategy.momentum_lookback_bars,
                "minimum_momentum_abs": settings.strategy.minimum_momentum_abs,
                "minimum_candle_range_atr": settings.strategy.minimum_candle_range_atr,
                "use_ema_200_1h_trend_filter": settings.strategy.use_ema_200_1h_trend_filter,
                "use_ema_200_1h_slope": settings.strategy.use_ema_200_1h_slope,
                "ema_200_1h_slope_tolerance": settings.strategy.ema_200_1h_slope_tolerance,
                "minimum_opening_range_width_atr": settings.strategy.minimum_opening_range_width_atr,
                "maximum_opening_range_width_atr": settings.strategy.maximum_opening_range_width_atr,
                "minimum_relative_volume": settings.strategy.minimum_relative_volume,
                "max_breakout_distance_atr": settings.strategy.max_breakout_distance_atr,
                "max_breakouts_per_day": settings.strategy.max_breakouts_per_day,
                "use_intraday_vwap_filter": settings.strategy.use_intraday_vwap_filter,
                "use_intraday_ema20_filter": settings.strategy.use_intraday_ema20_filter,
                "use_intraday_ema50_alignment": settings.strategy.use_intraday_ema50_alignment,
                "use_opening_range_mid_filter": settings.strategy.use_opening_range_mid_filter,
                "session_trend_structure_lookback_bars": settings.strategy.session_trend_structure_lookback_bars,
                "maximum_context_compression_width_atr": settings.strategy.maximum_context_compression_width_atr,
                "require_context_vwap_structure": settings.strategy.require_context_vwap_structure,
                "require_context_or_mid_structure": settings.strategy.require_context_or_mid_structure,
                "breakout_buffer_atr": settings.strategy.breakout_buffer_atr,
            }
        )
    return payload
