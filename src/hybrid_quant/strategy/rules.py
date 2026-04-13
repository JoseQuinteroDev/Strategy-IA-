from __future__ import annotations

import math

from hybrid_quant.core import SignalSide


def momentum_aligned(side: SignalSide, momentum: float | None) -> bool:
    if momentum is None:
        return False
    if side == SignalSide.LONG:
        return momentum > 0.0
    if side == SignalSide.SHORT:
        return momentum < 0.0
    return False


def candle_confirms_side(side: SignalSide, *, open_price: float, close_price: float) -> bool:
    if side == SignalSide.LONG:
        return close_price > open_price
    if side == SignalSide.SHORT:
        return close_price < open_price
    return False


def expected_move_bps(*, close_price: float, atr: float, atr_multiple_target: float) -> float:
    if close_price <= 0.0 or atr <= 0.0:
        return float("nan")
    return ((atr * atr_multiple_target) / close_price) * 10000.0


def target_to_cost_ratio(*, expected_move_bps_value: float, estimated_round_trip_cost_bps: float) -> float:
    if estimated_round_trip_cost_bps <= 0.0:
        return float("inf")
    if not math.isfinite(expected_move_bps_value):
        return float("nan")
    return expected_move_bps_value / estimated_round_trip_cost_bps


def distance_atr(*, price: float, anchor: float | None, atr: float | None, side: SignalSide) -> float | None:
    if anchor is None or atr is None or atr <= 0.0:
        return None
    if side == SignalSide.LONG:
        return max(0.0, (price - anchor) / atr)
    if side == SignalSide.SHORT:
        return max(0.0, (anchor - price) / atr)
    return None


def absolute_distance_atr(*, price: float, anchor: float | None, atr: float | None) -> float | None:
    if anchor is None or atr is None or atr <= 0.0:
        return None
    return abs(price - anchor) / atr
