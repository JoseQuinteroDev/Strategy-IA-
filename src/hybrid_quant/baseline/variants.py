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
    "baseline_trend_nasdaq": Path("variants") / "baseline_trend_nasdaq.yaml",
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
        "allowed_weekdays": list(settings.strategy.allowed_weekdays),
        "exclude_weekends": settings.strategy.exclude_weekends,
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
    return payload
