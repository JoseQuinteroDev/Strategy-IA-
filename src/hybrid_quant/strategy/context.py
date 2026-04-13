from __future__ import annotations

from dataclasses import dataclass
import math

from hybrid_quant.core import SignalSide


@dataclass(frozen=True, slots=True)
class MacroContext:
    """Higher-timeframe state used to decide whether intraday setups are worth taking."""

    bias_side: SignalSide | None
    bias_label: str
    regime: str
    adx_1h: float | None
    ema_200_1h: float | None
    ema_200_1h_slope: float | None
    price_vs_ema_200_1h_pct: float | None
    reason: str

    def matches(self, side: SignalSide) -> bool:
        return self.bias_side is None or self.bias_side == side


def build_macro_context(
    *,
    close_price: float,
    ema_200_1h: float | None,
    ema_200_1h_slope: float | None,
    adx_1h: float | None,
    price_vs_ema_200_1h_pct: float | None,
    adx_threshold: float,
    use_trend_filter: bool,
    use_slope_filter: bool,
    slope_tolerance: float,
) -> MacroContext:
    """Build a compact HTF context without introducing look-ahead state."""

    clean_slope_tolerance = abs(float(slope_tolerance))
    trend_side: SignalSide | None = None

    if use_trend_filter:
        if ema_200_1h is None:
            return MacroContext(
                bias_side=None,
                bias_label="unknown",
                regime="unknown",
                adx_1h=adx_1h,
                ema_200_1h=ema_200_1h,
                ema_200_1h_slope=ema_200_1h_slope,
                price_vs_ema_200_1h_pct=price_vs_ema_200_1h_pct,
                reason="Missing EMA200 1H, macro context cannot be trusted.",
            )
        if close_price > ema_200_1h:
            trend_side = SignalSide.LONG
        elif close_price < ema_200_1h:
            trend_side = SignalSide.SHORT
    else:
        if price_vs_ema_200_1h_pct is not None and price_vs_ema_200_1h_pct > 0.0:
            trend_side = SignalSide.LONG
        elif price_vs_ema_200_1h_pct is not None and price_vs_ema_200_1h_pct < 0.0:
            trend_side = SignalSide.SHORT

    slope_side: SignalSide | None = None
    if use_slope_filter and ema_200_1h_slope is not None:
        if ema_200_1h_slope > clean_slope_tolerance:
            slope_side = SignalSide.LONG
        elif ema_200_1h_slope < -clean_slope_tolerance:
            slope_side = SignalSide.SHORT

    if trend_side is not None and slope_side is not None and trend_side != slope_side:
        bias_side = None
        bias_label = "conflicted"
    else:
        bias_side = slope_side or trend_side
        bias_label = bias_side.value if bias_side is not None else "neutral"

    trend_strength = 0.0
    if adx_1h is not None:
        trend_strength = max(trend_strength, adx_1h / max(adx_threshold, 1e-12))
    if ema_200_1h_slope is not None and ema_200_1h is not None and ema_200_1h > 0.0:
        trend_strength = max(trend_strength, abs(ema_200_1h_slope / ema_200_1h) * 1000.0)

    strong_trend = bool(
        (adx_1h is not None and adx_1h >= adx_threshold)
        or (use_slope_filter and slope_side is not None)
    )
    if bias_label == "conflicted":
        regime = "conflicted"
    elif strong_trend and bias_side is not None:
        regime = "trend"
    elif trend_strength <= 1.0:
        regime = "range"
    else:
        regime = "transition"

    reason = (
        f"macro={regime}, bias={bias_label}, adx={_fmt(adx_1h)}, "
        f"ema200_slope={_fmt(ema_200_1h_slope)}"
    )
    return MacroContext(
        bias_side=bias_side,
        bias_label=bias_label,
        regime=regime,
        adx_1h=adx_1h,
        ema_200_1h=ema_200_1h,
        ema_200_1h_slope=ema_200_1h_slope,
        price_vs_ema_200_1h_pct=price_vs_ema_200_1h_pct,
        reason=reason,
    )


def finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"
