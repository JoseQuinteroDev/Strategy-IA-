from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hybrid_quant.core import SignalSide, StrategyContext, StrategySignal

from .base import IntradayStrategySupport, Strategy
from .context import MacroContext, build_macro_context, finite_or_none
from .rules import (
    absolute_distance_atr,
    candle_confirms_side,
    distance_atr,
    expected_move_bps,
    momentum_aligned,
    target_to_cost_ratio,
)


@dataclass(slots=True)
class IntradayHybridContextualStrategy(Strategy, IntradayStrategySupport):
    """Primary intraday baseline: HTF context first, concrete intraday setup second."""

    name: str
    variant_name: str
    trend_filter: str
    regime_filter: str | None
    execution_timeframe: str
    filter_timeframe: str
    entry_mode: str = "macro_pullback_continuation"
    atr_multiple_stop: float = 1.0
    atr_multiple_target: float = 1.5
    time_stop_bars: int = 12
    close_on_session_end: bool = True
    session_close_hour_utc: int = 20
    session_close_minute_utc: int = 55
    session_close_timezone: str = "UTC"
    no_entry_minutes_before_close: int = 15
    enforce_entry_session: bool = True
    entry_session_start_hour_utc: int = 14
    entry_session_start_minute_utc: int = 0
    entry_session_end_hour_utc: int = 19
    entry_session_end_minute_utc: int = 0
    entry_session_timezone: str = "UTC"
    blocked_hours_utc: list[int] | None = None
    allowed_hours_utc: list[int] | None = None
    allowed_hours_long_utc: list[int] | None = None
    allowed_hours_short_utc: list[int] | None = None
    allowed_weekdays: list[int] | None = None
    allowed_sides: list[str] | None = None
    exclude_weekends: bool = True
    entry_zscore: float = 2.0
    mean_reversion_anchor: str = "vwap"
    adx_threshold: float = 22.0
    minimum_anchor_distance_atr: float = 0.50
    minimum_expected_move_bps: float = 0.0
    minimum_target_to_cost_ratio: float = 0.0
    estimated_round_trip_cost_bps: float = 0.0
    momentum_lookback_bars: int = 20
    minimum_momentum_abs: float = 0.0
    minimum_candle_range_atr: float = 0.0
    use_ema_200_1h_trend_filter: bool = True
    use_ema_200_1h_slope: bool = True
    use_macro_bias_filter: bool = True
    ema_200_1h_slope_tolerance: float = 0.0
    minimum_relative_volume: float = 0.0
    max_breakout_distance_atr: float = 0.0
    max_breakouts_per_day: int = 4
    maximum_pullback_depth_atr: float = 0.0
    use_intraday_vwap_filter: bool = True
    use_intraday_ema20_filter: bool = True
    use_intraday_ema50_alignment: bool = True
    use_opening_range_mid_filter: bool = False

    def generate(self, context: StrategyContext) -> StrategySignal:
        timestamp = self._resolve_timestamp(context)
        base_metadata = self._base_metadata()

        if not context.bars or not context.features:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Insufficient market context for the intraday hybrid contextual baseline.",
                metadata=base_metadata,
            )

        latest_bar = context.bars[-1]
        values = context.features[-1].values
        close_price = latest_bar.close

        if self._is_session_close(timestamp):
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Session close: the hybrid intraday baseline does not hold overnight.",
                metadata={**base_metadata, "session_close_exit": True},
            )

        session_metadata = self._entry_session_metadata(timestamp)
        session_gate_reason = self._session_gate_reason(timestamp)
        if session_gate_reason is not None:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale=session_gate_reason,
                metadata={
                    **base_metadata,
                    **session_metadata,
                    "session_gate": True,
                    "blocked_by_filter": "outside_session" if session_metadata["outside_session"] else "session_filter",
                    "candidate_status": "blocked_by_time_filter",
                },
            )

        minutes_to_close = self._minutes_to_session_close(timestamp)
        if minutes_to_close <= self.no_entry_minutes_before_close:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="No new hybrid intraday entries are allowed inside the close buffer.",
                metadata={**base_metadata, **session_metadata, "minutes_to_close": minutes_to_close},
            )

        features = self._read_features(values)
        missing = [name for name in ("atr_14", "ema_200_1h", "ema_200_1h_slope", "adx_1h") if features[name] is None]
        if missing:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale=f"Missing HTF/intraday features for hybrid baseline: {missing}.",
                metadata=base_metadata,
            )

        macro = build_macro_context(
            close_price=close_price,
            ema_200_1h=features["ema_200_1h"],
            ema_200_1h_slope=features["ema_200_1h_slope"],
            adx_1h=features["adx_1h"],
            price_vs_ema_200_1h_pct=features["price_vs_ema_200_1h_pct"],
            adx_threshold=self.adx_threshold,
            use_trend_filter=self.use_ema_200_1h_trend_filter,
            use_slope_filter=self.use_ema_200_1h_slope,
            slope_tolerance=self.ema_200_1h_slope_tolerance,
        )
        setup = self._resolve_setup(
            latest_bar=latest_bar,
            features=features,
            macro=macro,
        )
        metadata = {
            **base_metadata,
            **session_metadata,
            **self._macro_metadata(macro),
            **self._feature_metadata(features, close_price=close_price),
        }

        if setup is None:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="No hybrid intraday setup is active after macro and execution filters.",
                metadata=metadata,
            )

        side = setup["side"]
        direction_gate_reason = self._direction_gate_reason(side)
        if direction_gate_reason is not None:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale=direction_gate_reason,
                metadata={**metadata, **setup, "direction_gate": True},
            )

        side_hour_gate_reason = self._side_hour_gate_reason(side=side, timestamp=timestamp)
        if side_hour_gate_reason is not None:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale=side_hour_gate_reason,
                metadata={**metadata, **setup, "directional_hour_gate": True},
            )

        setup_count = features["context_setup_count_today"]
        if (
            self.max_breakouts_per_day > 0
            and setup_count is not None
            and int(setup_count) > self.max_breakouts_per_day
        ):
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale="Hybrid intraday setup budget for the day has already been used.",
                metadata={**metadata, **setup, "context_setup_count_today": int(setup_count)},
            )

        quality_reason = self._quality_gate_failure(
            side=side,
            setup=setup,
            features=features,
            macro=macro,
            close_price=close_price,
        )
        if quality_reason is not None:
            return self._flat_signal(
                symbol=context.symbol,
                timestamp=timestamp,
                rationale=quality_reason,
                metadata={**metadata, **setup},
            )

        return self._entry_signal(
            symbol=context.symbol,
            timestamp=timestamp,
            side=side,
            entry_price=close_price,
            atr=features["atr_14"],
            entry_reason=setup["entry_reason"],
            metadata={**metadata, **setup},
        )

    def _resolve_setup(
        self,
        *,
        latest_bar,
        features: dict[str, float | None],
        macro: MacroContext,
    ) -> dict[str, Any] | None:
        if self.entry_mode == "macro_pullback_continuation":
            return self._pullback_setup(latest_bar=latest_bar, features=features, macro=macro)
        if self.entry_mode == "controlled_mean_reversion":
            return self._mean_reversion_setup(latest_bar=latest_bar, features=features, macro=macro)
        if self.entry_mode == "compression_expansion_continuation":
            return self._compression_expansion_setup(latest_bar=latest_bar, features=features, macro=macro)
        raise ValueError(f"Unsupported hybrid intraday entry mode: {self.entry_mode}")

    def _pullback_setup(
        self,
        *,
        latest_bar,
        features: dict[str, float | None],
        macro: MacroContext,
    ) -> dict[str, Any] | None:
        side: SignalSide | None = None
        if features["context_long_pullback_entry"] == 1.0:
            side = SignalSide.LONG
        elif features["context_short_pullback_entry"] == 1.0:
            side = SignalSide.SHORT
        if side is None:
            return None
        if self.use_macro_bias_filter and macro.bias_side != side:
            return None

        anchor = features["context_support_level"] or self._anchor_for_side(features, side)
        anchor_distance = distance_atr(
            price=latest_bar.close,
            anchor=anchor,
            atr=features["atr_14"],
            side=side,
        )
        return {
            "side": side,
            "setup_label": "pullback_with_macro_bias",
            "entry_trigger": "macro_pullback_continuation",
            "entry_reason": (
                "Hybrid intraday pullback: HTF bias is aligned and price resumed from VWAP/EMA value."
            ),
            "anchor_name": "context_support",
            "anchor_value": anchor,
            "anchor_distance_atr": anchor_distance,
            "breakout_level": features["context_trigger_level"],
            "breakout_distance_atr": features["context_trigger_distance_atr"],
        }

    def _mean_reversion_setup(
        self,
        *,
        latest_bar,
        features: dict[str, float | None],
        macro: MacroContext,
    ) -> dict[str, Any] | None:
        zscore = features["zscore_distance_to_mean"]
        atr = features["atr_14"]
        anchor = self._mean_reversion_anchor_value(features)
        anchor_distance = absolute_distance_atr(price=latest_bar.close, anchor=anchor, atr=atr)
        if zscore is None or anchor is None or atr is None:
            return None
        if macro.regime == "trend":
            return None

        side: SignalSide | None = None
        threshold = abs(self.entry_zscore)
        if zscore <= -threshold:
            side = SignalSide.LONG
        elif zscore >= threshold:
            side = SignalSide.SHORT
        if side is None:
            return None
        if not candle_confirms_side(side, open_price=latest_bar.open, close_price=latest_bar.close):
            return None

        return {
            "side": side,
            "setup_label": "controlled_mean_reversion",
            "entry_trigger": "controlled_mean_reversion",
            "entry_reason": (
                "Hybrid controlled mean reversion: HTF regime is not strongly trending and price is extended from value."
            ),
            "anchor_name": self.mean_reversion_anchor,
            "anchor_value": anchor,
            "anchor_distance_atr": anchor_distance,
            "zscore_distance_to_mean": zscore,
            "breakout_level": anchor,
            "breakout_distance_atr": anchor_distance,
        }

    def _compression_expansion_setup(
        self,
        *,
        latest_bar,
        features: dict[str, float | None],
        macro: MacroContext,
    ) -> dict[str, Any] | None:
        side: SignalSide | None = None
        if features["context_long_session_trend_entry"] == 1.0:
            side = SignalSide.LONG
        elif features["context_short_session_trend_entry"] == 1.0:
            side = SignalSide.SHORT
        if side is None:
            return None
        if self.use_macro_bias_filter and macro.bias_side != side:
            return None
        if macro.regime not in {"trend", "transition"}:
            return None
        if not candle_confirms_side(side, open_price=latest_bar.open, close_price=latest_bar.close):
            return None

        anchor = features["context_trigger_level"]
        trigger_distance = distance_atr(
            price=latest_bar.close,
            anchor=anchor,
            atr=features["atr_14"],
            side=side,
        )
        return {
            "side": side,
            "setup_label": "compression_expansion_continuation",
            "entry_trigger": "compression_expansion_continuation",
            "entry_reason": (
                "Hybrid compression/expansion: HTF bias is aligned and a compact intraday structure expanded."
            ),
            "anchor_name": "recent_session_structure",
            "anchor_value": anchor,
            "anchor_distance_atr": trigger_distance,
            "breakout_level": anchor,
            "breakout_distance_atr": trigger_distance,
            "compression_width_atr": features["context_session_range_width_atr"],
        }

    def _quality_gate_failure(
        self,
        *,
        side: SignalSide,
        setup: dict[str, Any],
        features: dict[str, float | None],
        macro: MacroContext,
        close_price: float,
    ) -> str | None:
        if self.use_macro_bias_filter and setup["entry_trigger"] != "controlled_mean_reversion" and not macro.matches(side):
            return "Hybrid setup rejected because the macro bias does not support the candidate direction."

        if setup["entry_trigger"] != "controlled_mean_reversion":
            momentum = features["momentum"]
            if momentum is None:
                return "Hybrid setup rejected because momentum is missing."
            if not momentum_aligned(side, momentum):
                return "Hybrid setup rejected because momentum is not aligned with the entry direction."
            if self.minimum_momentum_abs > 0.0 and abs(momentum) < self.minimum_momentum_abs:
                return "Hybrid setup rejected because momentum is below the configured minimum."

        candle_range_atr = features["candle_range_atr"]
        if candle_range_atr is None:
            return "Hybrid setup rejected because candle expansion is missing."
        if self.minimum_candle_range_atr > 0.0 and candle_range_atr < self.minimum_candle_range_atr:
            return "Hybrid setup rejected because the trigger candle lacks enough range expansion."

        relative_volume = features["relative_volume"]
        if relative_volume is not None and self.minimum_relative_volume > 0.0 and relative_volume < self.minimum_relative_volume:
            return "Hybrid setup rejected because relative volume is below the configured floor."

        anchor_distance = setup.get("anchor_distance_atr")
        if (
            self.minimum_anchor_distance_atr > 0.0
            and setup["entry_trigger"] == "controlled_mean_reversion"
            and anchor_distance is not None
            and anchor_distance < self.minimum_anchor_distance_atr
        ):
            return "Hybrid mean-reversion setup rejected because extension from value is too small."

        if (
            self.maximum_pullback_depth_atr > 0.0
            and setup["entry_trigger"] == "macro_pullback_continuation"
            and anchor_distance is not None
            and anchor_distance > self.maximum_pullback_depth_atr
        ):
            return "Hybrid pullback rejected because the entry is too far from the value anchor."

        breakout_distance = setup.get("breakout_distance_atr")
        if (
            self.max_breakout_distance_atr > 0.0
            and breakout_distance is not None
            and breakout_distance > self.max_breakout_distance_atr
        ):
            return "Hybrid setup rejected because the entry is too extended from the trigger level."

        move_bps = expected_move_bps(
            close_price=close_price,
            atr=features["atr_14"],
            atr_multiple_target=self.atr_multiple_target,
        )
        ratio = target_to_cost_ratio(
            expected_move_bps_value=move_bps,
            estimated_round_trip_cost_bps=self.estimated_round_trip_cost_bps,
        )
        setup["expected_move_bps"] = move_bps
        setup["target_to_cost_ratio"] = ratio
        if self.minimum_expected_move_bps > 0.0 and move_bps < self.minimum_expected_move_bps:
            return "Hybrid setup rejected because projected ATR target is too small versus price."
        if self.minimum_target_to_cost_ratio > 0.0 and ratio < self.minimum_target_to_cost_ratio:
            return "Hybrid setup rejected because target-to-cost ratio is too weak."
        return None

    def _read_features(self, values: dict[str, float]) -> dict[str, float | None]:
        momentum = finite_or_none(values.get(f"momentum_{self.momentum_lookback_bars}"))
        if momentum is None:
            momentum = finite_or_none(values.get("momentum_20"))
        return {
            "atr_14": finite_or_none(values.get("atr_14")),
            "adx_1h": finite_or_none(values.get("adx_1h")),
            "ema_200_1h": finite_or_none(values.get("ema_200_1h")),
            "ema_200_1h_slope": finite_or_none(values.get("ema_200_1h_slope")),
            "price_vs_ema_200_1h_pct": finite_or_none(values.get("price_vs_ema_200_1h_pct")),
            "intraday_vwap": finite_or_none(values.get("intraday_vwap")),
            "ema_20": finite_or_none(values.get("ema_20")),
            "ema_50": finite_or_none(values.get("ema_50")),
            "distance_to_mean": finite_or_none(values.get("distance_to_mean")),
            "zscore_distance_to_mean": finite_or_none(values.get("zscore_distance_to_mean")),
            "momentum": momentum,
            "candle_range_atr": finite_or_none(values.get("candle_range_atr")),
            "relative_volume": finite_or_none(values.get("relative_volume")),
            "context_long_pullback_entry": finite_or_none(values.get("context_long_pullback_entry")),
            "context_short_pullback_entry": finite_or_none(values.get("context_short_pullback_entry")),
            "context_long_session_trend_entry": finite_or_none(values.get("context_long_session_trend_entry")),
            "context_short_session_trend_entry": finite_or_none(values.get("context_short_session_trend_entry")),
            "context_support_level": finite_or_none(values.get("context_support_level")),
            "context_trigger_level": finite_or_none(values.get("context_trigger_level")),
            "context_trigger_distance_atr": finite_or_none(values.get("context_trigger_distance_atr")),
            "context_session_range_width_atr": finite_or_none(values.get("context_session_range_width_atr")),
            "context_setup_count_today": finite_or_none(values.get("context_setup_count_today")),
            "opening_range_high": finite_or_none(values.get("opening_range_high")),
            "opening_range_low": finite_or_none(values.get("opening_range_low")),
            "opening_range_mid": finite_or_none(values.get("opening_range_mid")),
            "opening_range_width_atr": finite_or_none(values.get("opening_range_width_atr")),
        }

    def _mean_reversion_anchor_value(self, features: dict[str, float | None]) -> float | None:
        anchor = self.mean_reversion_anchor.lower()
        if anchor in {"vwap", "intraday_vwap"}:
            return features["intraday_vwap"]
        if anchor == "ema20":
            return features["ema_20"]
        if anchor == "ema50":
            return features["ema_50"]
        return features["intraday_vwap"] or features["ema_50"]

    def _anchor_for_side(self, features: dict[str, float | None], side: SignalSide) -> float | None:
        candidates = [features["ema_20"]]
        if self.use_intraday_vwap_filter:
            candidates.append(features["intraday_vwap"])
        if self.use_intraday_ema50_alignment:
            candidates.append(features["ema_50"])
        clean = [value for value in candidates if value is not None]
        if not clean:
            return None
        return max(clean) if side == SignalSide.LONG else min(clean)

    def _base_metadata(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "variant_name": self.variant_name,
            "strategy_family": "intraday_hybrid_contextual",
            "execution_timeframe": self.execution_timeframe,
            "filter_timeframe": self.filter_timeframe,
            "trend_filter": self.trend_filter,
            "regime_filter": self.regime_filter,
            "entry_mode": self.entry_mode,
            "time_stop_bars": self.time_stop_bars,
            "close_on_session_end": self.close_on_session_end,
            "enforce_entry_session": self.enforce_entry_session,
            "use_macro_bias_filter": self.use_macro_bias_filter,
            "entry_session_window_utc": self._entry_session_window_label(),
            "entry_session_start_utc": (
                f"{self.entry_session_start_hour_utc:02d}:{self.entry_session_start_minute_utc:02d}"
            ),
            "entry_session_end_utc": (
                f"{self.entry_session_end_hour_utc:02d}:{self.entry_session_end_minute_utc:02d}"
            ),
            "allowed_hours_utc": list(self.allowed_hours_utc or []),
            "allowed_hours_long_utc": list(self.allowed_hours_long_utc or []),
            "allowed_hours_short_utc": list(self.allowed_hours_short_utc or []),
            "allowed_sides": list(self.allowed_sides or []),
            "minimum_target_to_cost_ratio": self.minimum_target_to_cost_ratio,
            "estimated_round_trip_cost_bps": self.estimated_round_trip_cost_bps,
        }

    def _macro_metadata(self, macro: MacroContext) -> dict[str, Any]:
        return {
            "macro_regime": macro.regime,
            "macro_bias_side": macro.bias_side.value if macro.bias_side is not None else None,
            "macro_bias_label": macro.bias_label,
            "macro_context_reason": macro.reason,
            "regime": macro.regime,
            "macro_adx_1h": macro.adx_1h,
            "macro_ema_200_1h": macro.ema_200_1h,
            "ema_200_1h_slope": macro.ema_200_1h_slope,
            "macro_price_vs_ema_200_1h_pct": macro.price_vs_ema_200_1h_pct,
        }

    def _feature_metadata(
        self,
        features: dict[str, float | None],
        *,
        close_price: float,
    ) -> dict[str, Any]:
        return {
            "intraday_vwap": features["intraday_vwap"],
            "ema_20": features["ema_20"],
            "ema_50": features["ema_50"],
            "momentum": features["momentum"],
            "abs_momentum": abs(features["momentum"]) if features["momentum"] is not None else None,
            "candle_range_atr": features["candle_range_atr"],
            "relative_volume": features["relative_volume"],
            "zscore_distance_to_mean": features["zscore_distance_to_mean"],
            "abs_entry_zscore": (
                abs(features["zscore_distance_to_mean"]) if features["zscore_distance_to_mean"] is not None else None
            ),
            "context_session_range_width_atr": features["context_session_range_width_atr"],
            "context_setup_count_today": features["context_setup_count_today"],
            "opening_range_high": features["opening_range_high"],
            "opening_range_low": features["opening_range_low"],
            "opening_range_mid": features["opening_range_mid"],
            "opening_range_width_atr": features["opening_range_width_atr"],
            "entry_price_reference": close_price,
        }
