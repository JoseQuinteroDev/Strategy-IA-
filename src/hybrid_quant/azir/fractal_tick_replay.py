"""Tick-first replay for the Azir fractal setup candidate."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any

from hybrid_quant.risk.azir_state import AzirRiskConfig

from .economic_audit import _round, _to_float, _write_csv, summarize_trades
from .fractal_candidate_export import CANDIDATE_NAME, CANDIDATE_VARIANT
from .fractal_candidate_real_mt5 import read_canonical_event_log
from .fractal_protected_economic import (
    OUTPUT_BENCHMARK_NAME as PROTECTED_CANDIDATE_NAME,
    _bar_range,
    _bars_by_day,
    _base_lifecycle_row,
    _blocked_row,
    _close_index,
    _float,
    _format_dt,
    _is_true,
    _no_order_row,
    _parse_dt,
    _pct,
    _pnl,
    _pre_trade_risk_guard,
    _prevented_row,
    _resolve_fill_side,
    _select_replay_bars,
    _timestamp,
    _unpriced_setup_row,
    build_anomaly_summary,
    build_comparison_rows,
    build_decision,
    inspect_tick_csv_fast,
    load_current_protected_reference,
    replay_candidate_lifecycle,
)
from .management_replay_v2 import _tick_column_index
from .replica import AzirReplicaConfig, OhlcvBar, load_ohlcv_csv


SPRINT_NAME = "tick_level_fractal_candidate_replay_v1"
TICK_CANDIDATE_NAME = "baseline_azir_tick_replay_candidate_fractal_v1"
DEFAULT_SYMBOL = "XAUUSD-STD"


@dataclass
class TickReplayState:
    label: str
    setup_row: dict[str, Any]
    setup_dt: datetime
    close_dt: datetime
    replay_config: AzirReplicaConfig
    setup_text: str
    close_text: str
    buy_placed: bool
    sell_placed: bool
    buy_entry: float | None
    sell_entry: float | None
    ticks_seen: int = 0
    first_tick: str = ""
    last_tick: str = ""
    last_bid: float | None = None
    last_ask: float | None = None
    filled: bool = False
    closed: bool = False
    fill_side: str = ""
    fill_timestamp: datetime | None = None
    fill_price: float | None = None
    stop: float | None = None
    target: float | None = None
    mfe_points: float = 0.0
    mae_points: float = 0.0
    trailing_activated: bool = False
    trailing_modifications: int = 0
    fill_ambiguity: bool = False
    exit_timestamp: datetime | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    force_close: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tick-first replay for Azir current and fractal candidate setups.")
    parser.add_argument("--current-log-path", required=True)
    parser.add_argument("--candidate-log-path", required=True)
    parser.add_argument("--tick-input-path", required=True)
    parser.add_argument("--m1-input-path", required=True)
    parser.add_argument("--m5-input-path", required=True)
    parser.add_argument("--protected-report-path", required=True)
    parser.add_argument("--previous-fractal-report-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--lot-size", type=float, default=0.10)
    parser.add_argument("--contract-size", type=float, default=100.0)
    parser.add_argument("--config-name", default="risk_engine_azir_v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_fractal_tick_replay(
        current_log_path=Path(args.current_log_path),
        candidate_log_path=Path(args.candidate_log_path),
        tick_input_path=Path(args.tick_input_path),
        m1_input_path=Path(args.m1_input_path),
        m5_input_path=Path(args.m5_input_path),
        protected_report_path=Path(args.protected_report_path),
        previous_fractal_report_path=Path(args.previous_fractal_report_path) if args.previous_fractal_report_path else None,
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
        lot_size=args.lot_size,
        contract_size=args.contract_size,
        risk_config=AzirRiskConfig(name=args.config_name),
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_fractal_tick_replay(
    *,
    current_log_path: Path,
    candidate_log_path: Path,
    tick_input_path: Path,
    m1_input_path: Path,
    m5_input_path: Path,
    protected_report_path: Path,
    output_dir: Path,
    previous_fractal_report_path: Path | None = None,
    symbol: str = DEFAULT_SYMBOL,
    lot_size: float = 0.10,
    contract_size: float = 100.0,
    risk_config: AzirRiskConfig | None = None,
) -> dict[str, Any]:
    risk_config = risk_config or AzirRiskConfig()
    current_rows = read_canonical_event_log(current_log_path, symbol_filter=symbol)
    candidate_rows = read_canonical_event_log(candidate_log_path, symbol_filter=symbol)
    if not current_rows:
        raise ValueError(f"No current Azir rows found for symbol {symbol}.")
    if not candidate_rows:
        raise ValueError(f"No fractal candidate rows found for symbol {symbol}.")
    tick_inspection = inspect_tick_csv_fast(tick_input_path)
    if not tick_inspection.get("available"):
        raise FileNotFoundError(f"Tick CSV does not exist: {tick_input_path}")

    m1_bars = load_ohlcv_csv(m1_input_path)
    m5_bars = load_ohlcv_csv(m5_input_path)
    replay_config = AzirReplicaConfig(symbol=symbol, lot_size=lot_size, contract_size=contract_size)
    m1_by_day = _bars_by_day(m1_bars)
    m5_by_day = _bars_by_day(m5_bars)
    current_tick = replay_rows_tick_first(
        label="current",
        setup_rows=current_rows,
        tick_input_path=tick_input_path,
        tick_inspection=tick_inspection,
        m1_by_day=m1_by_day,
        m5_by_day=m5_by_day,
        replay_config=replay_config,
        risk_config=risk_config,
    )
    candidate_tick = replay_rows_tick_first(
        label="fractal",
        setup_rows=candidate_rows,
        tick_input_path=tick_input_path,
        tick_inspection=tick_inspection,
        m1_by_day=m1_by_day,
        m5_by_day=m5_by_day,
        replay_config=replay_config,
        risk_config=risk_config,
    )
    current_trades = [row for row in current_tick if row.get("has_exit")]
    candidate_trades = [row for row in candidate_tick if row.get("has_exit")]
    current_metrics = summarize_trades(current_trades)
    candidate_metrics = summarize_trades(candidate_trades)
    protected_reference = load_current_protected_reference(protected_report_path)
    previous_fractal = _load_previous_fractal(previous_fractal_report_path)
    coverage = build_tick_coverage_report(candidate_tick, candidate_rows, tick_inspection, m1_bars, m5_bars)
    current_coverage = build_tick_coverage_report(current_tick, current_rows, tick_inspection, m1_bars, m5_bars)
    anomalies = build_anomaly_summary(candidate_tick)
    decision = build_tick_decision(candidate_metrics, current_metrics, protected_reference["metrics"], coverage, previous_fractal)
    same_replay_rows = _tick_comparison_rows(current_metrics, candidate_metrics, "azir_current_tick_same_methodology")
    protected_rows = _tick_comparison_rows(protected_reference["metrics"], candidate_metrics, "baseline_azir_protected_economic_v1")
    exit_distribution = build_exit_distribution(candidate_tick, current_tick)

    report = {
        "sprint": SPRINT_NAME,
        "candidate_name": CANDIDATE_NAME,
        "candidate_variant": CANDIDATE_VARIANT,
        "tick_candidate_name": TICK_CANDIDATE_NAME,
        "previous_protected_candidate_name": PROTECTED_CANDIDATE_NAME,
        "sources": {
            "current_log_path": str(current_log_path),
            "candidate_log_path": str(candidate_log_path),
            "tick_input_path": str(tick_input_path),
            "m1_input_path": str(m1_input_path),
            "m5_input_path": str(m5_input_path),
            "protected_report_path": str(protected_report_path),
            "previous_fractal_report_path": str(previous_fractal_report_path) if previous_fractal_report_path else "",
            "symbol": symbol,
        },
        "risk_engine": asdict(risk_config),
        "pricing_methodology": {
            "tick_fill": "Buy stops trigger on ask >= entry; sell stops trigger on bid <= entry.",
            "tick_fill_price": "Conservative: buy fills at max(entry, ask), sell fills at min(entry, bid).",
            "tick_exit": "Long exits use bid; short exits use ask. Stop is checked before target.",
            "tick_trailing": "Trailing is updated tick-by-tick after favorable movement crosses Azir trailing_start_points.",
            "fallback": "If a full setup-to-close tick window is unavailable, replay falls back to M1, then M5.",
        },
        "tick_inspection": tick_inspection,
        "bar_coverage": {"m1": _bar_range(m1_bars), "m5": _bar_range(m5_bars)},
        "current_protected_reference": protected_reference,
        "previous_fractal_protected_candidate": previous_fractal,
        "current_tick_same_methodology_metrics": current_metrics,
        "candidate_tick_metrics": candidate_metrics,
        "coverage": coverage,
        "current_coverage": current_coverage,
        "anomaly_summary": anomalies,
        "same_methodology_comparison": same_replay_rows,
        "protected_reference_comparison": protected_rows,
        "exit_distribution": exit_distribution,
        "decision": decision,
        "limitations": [
            "Tick replay is still a reconstruction from exported bid/ask ticks, not a broker order-queue simulation.",
            "Rows outside full tick coverage use documented M1 or M5 fallback.",
            "M5 fallback remains materially weaker than tick or M1 for stop/target ordering.",
            "No Azir entry logic, Risk Engine rule, or trailing parameter was changed.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(same_replay_rows, output_dir / "current_vs_fractal_tick_replay.csv")
    _write_csv(protected_rows, output_dir / "current_protected_vs_fractal_tick_replay.csv")
    _write_csv([coverage], output_dir / "fractal_tick_coverage_report.csv")
    _write_csv(candidate_trades, output_dir / "fractal_tick_priced_trades.csv")
    _write_csv(current_trades, output_dir / "current_tick_priced_trades.csv")
    _write_csv(exit_distribution, output_dir / "fractal_tick_exit_distribution.csv")
    (output_dir / "fractal_tick_replay_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "fractal_tick_replay_summary.md").write_text(summary_markdown(report), encoding="utf-8")
    (output_dir / "fractal_promotion_decision.md").write_text(promotion_markdown(report), encoding="utf-8")
    return report


def replay_rows_tick_first(
    *,
    label: str,
    setup_rows: list[dict[str, Any]],
    tick_input_path: Path,
    tick_inspection: dict[str, Any],
    m1_by_day: dict[Any, list[OhlcvBar]],
    m5_by_day: dict[Any, list[OhlcvBar]],
    replay_config: AzirReplicaConfig,
    risk_config: AzirRiskConfig,
) -> list[dict[str, Any]]:
    tick_first = _parse_dt(tick_inspection.get("first_time"))
    tick_last = _parse_dt(tick_inspection.get("last_time"))
    direct_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    states: list[TickReplayState] = []

    for row in sorted(_daily_setup_rows(setup_rows), key=_timestamp):
        setup_dt = _timestamp(row)
        close_dt = datetime.combine(setup_dt.date(), time(risk_config.close_hour, 0))
        if row.get("event_type") == "blocked_friday":
            direct_rows.append(_blocked_row(row, risk_status="blocked_friday"))
            continue
        guard = _pre_trade_risk_guard(
            setup_day=setup_dt.date().isoformat(),
            setup_row=row,
            daily_realized=defaultdict(float),
            daily_loss_streak=defaultdict(int),
            daily_trades=defaultdict(int),
            risk_config=risk_config,
        )
        if guard:
            direct_rows.append(_prevented_row(row, guard))
            continue
        if not _is_true(row.get("buy_order_placed")) and not _is_true(row.get("sell_order_placed")):
            bars, source = _select_replay_bars(setup_dt, m1_by_day, m5_by_day, risk_config)
            direct_rows.append(_no_order_row(row, source if bars else "not_applicable"))
            continue
        if tick_first <= setup_dt and close_dt <= tick_last:
            states.append(_state_from_setup(label, row, setup_dt, close_dt, replay_config))
        else:
            fallback_rows.append(_fallback_replay_row(row, m1_by_day, m5_by_day, replay_config, risk_config, "outside_full_tick_coverage"))

    tick_rows, needs_fallback = replay_states_from_tick_file(tick_input_path, states)
    fallback_rows.extend(
        _fallback_replay_row(state.setup_row, m1_by_day, m5_by_day, replay_config, risk_config, "tick_window_had_no_usable_bid_ask")
        for state in needs_fallback
    )
    rows = direct_rows + fallback_rows + tick_rows
    rows.sort(key=lambda item: item.get("setup_timestamp", ""))
    return rows


def replay_states_from_tick_file(
    tick_input_path: Path,
    states: list[TickReplayState],
) -> tuple[list[dict[str, Any]], list[TickReplayState]]:
    if not states:
        return [], []
    states_sorted = sorted(states, key=lambda item: item.setup_text)
    active: list[TickReplayState] = []
    next_index = 0
    completed: list[TickReplayState] = []
    needs_fallback: list[TickReplayState] = []

    with tick_input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError(f"Tick CSV is empty: {tick_input_path}")
        columns = [item.strip() for item in header_line.strip().split(",")]
        index = _tick_column_index(columns)
        for line in handle:
            parts = line.rstrip("\r\n").split(",")
            if len(parts) < len(columns):
                continue
            tick_time = parts[index["time"]]
            if next_index >= len(states_sorted) and not active:
                break
            while next_index < len(states_sorted) and states_sorted[next_index].setup_text <= tick_time:
                active.append(states_sorted[next_index])
                next_index += 1
            if active:
                still_active: list[TickReplayState] = []
                for state in active:
                    if state.close_text < tick_time:
                        _finalize_expired_state(state)
                        completed.append(state)
                    else:
                        still_active.append(state)
                active = still_active
            if not active:
                continue
            bid_text = parts[index["bid"]]
            ask_text = parts[index["ask"]]
            if not bid_text or not ask_text or bid_text == "0.00" or ask_text == "0.00":
                continue
            bid = float(bid_text)
            ask = float(ask_text)
            tick_dt = _parse_dt(tick_time)
            for state in active:
                if state.setup_text <= tick_time <= state.close_text and not state.closed:
                    _update_state_with_tick(state, tick_dt, tick_time, bid, ask)

    for state in active + states_sorted[next_index:]:
        _finalize_expired_state(state)
        completed.append(state)
    rows = []
    for state in completed:
        if state.ticks_seen == 0:
            needs_fallback.append(state)
        else:
            rows.append(_row_from_state(state))
    return rows, needs_fallback


def _update_state_with_tick(state: TickReplayState, tick_dt: datetime, tick_text: str, bid: float, ask: float) -> None:
    state.ticks_seen += 1
    state.first_tick = state.first_tick or tick_text
    state.last_tick = tick_text
    state.last_bid = bid
    state.last_ask = ask
    if not state.filled:
        hit_buy = bool(state.buy_placed and state.buy_entry is not None and ask >= state.buy_entry)
        hit_sell = bool(state.sell_placed and state.sell_entry is not None and bid <= state.sell_entry)
        if not hit_buy and not hit_sell:
            return
        state.fill_ambiguity = hit_buy and hit_sell
        side = _resolve_fill_side_from_tick(state, bid, ask, hit_buy, hit_sell)
        entry = state.buy_entry if side == "buy" else state.sell_entry
        if entry is None:
            return
        state.filled = True
        state.fill_side = side
        state.fill_timestamp = tick_dt
        state.fill_price = max(entry, ask) if side == "buy" else min(entry, bid)
        state.stop = state.fill_price - state.replay_config.sl_points * state.replay_config.point if side == "buy" else state.fill_price + state.replay_config.sl_points * state.replay_config.point
        state.target = state.fill_price + state.replay_config.tp_points * state.replay_config.point if side == "buy" else state.fill_price - state.replay_config.tp_points * state.replay_config.point
        return
    _update_open_position_with_tick(state, tick_dt, bid, ask)


def _update_open_position_with_tick(state: TickReplayState, tick_dt: datetime, bid: float, ask: float) -> None:
    if state.closed or state.fill_price is None or state.stop is None or state.target is None:
        return
    price = bid if state.fill_side == "buy" else ask
    favorable = (price - state.fill_price) / state.replay_config.point if state.fill_side == "buy" else (state.fill_price - price) / state.replay_config.point
    adverse = (state.fill_price - price) / state.replay_config.point if state.fill_side == "buy" else (price - state.fill_price) / state.replay_config.point
    state.mfe_points = max(state.mfe_points, favorable)
    state.mae_points = max(state.mae_points, adverse)
    if _stop_hit(state.fill_side, price, state.stop):
        state.exit_timestamp = tick_dt
        state.exit_price = _conservative_stop_fill(state.fill_side, price, state.stop)
        state.exit_reason = "tick_stop_or_trailing_stop"
        state.closed = True
        return
    if _target_hit(state.fill_side, price, state.target):
        state.exit_timestamp = tick_dt
        state.exit_price = state.target
        state.exit_reason = "tick_take_profit"
        state.closed = True
        return
    if favorable >= state.replay_config.trailing_start_points:
        if state.fill_side == "buy":
            new_stop = price - state.replay_config.trailing_step_points * state.replay_config.point
            if new_stop > state.stop:
                state.stop = new_stop
                state.trailing_activated = True
                state.trailing_modifications += 1
        else:
            new_stop = price + state.replay_config.trailing_step_points * state.replay_config.point
            if new_stop < state.stop:
                state.stop = new_stop
                state.trailing_activated = True
                state.trailing_modifications += 1


def _state_from_setup(
    label: str,
    row: dict[str, Any],
    setup_dt: datetime,
    close_dt: datetime,
    replay_config: AzirReplicaConfig,
) -> TickReplayState:
    return TickReplayState(
        label=label,
        setup_row=row,
        setup_dt=setup_dt,
        close_dt=close_dt,
        replay_config=replay_config,
        setup_text=setup_dt.strftime("%Y-%m-%d %H:%M:%S"),
        close_text=close_dt.strftime("%Y-%m-%d %H:%M:%S"),
        buy_placed=_is_true(row.get("buy_order_placed")),
        sell_placed=_is_true(row.get("sell_order_placed")),
        buy_entry=_to_float(row.get("buy_entry")),
        sell_entry=_to_float(row.get("sell_entry")),
    )


def _finalize_expired_state(state: TickReplayState) -> None:
    if state.closed:
        return
    if not state.filled:
        return
    state.closed = True
    state.force_close = True
    state.exit_reason = "tick_risk_engine_forced_session_close"
    state.exit_timestamp = state.close_dt
    if state.fill_price is None:
        state.exit_price = None
        return
    state.exit_price = _last_known_close_price(state)


def _last_known_close_price(state: TickReplayState) -> float:
    if state.fill_side == "buy" and state.last_bid is not None:
        return state.last_bid
    if state.fill_side == "sell" and state.last_ask is not None:
        return state.last_ask
    return state.fill_price or 0.0


def _row_from_state(state: TickReplayState) -> dict[str, Any]:
    if not state.filled:
        return {
            **_base_lifecycle_row(state.setup_row),
            "protected_status": "cancelled_no_fill_at_close",
            "risk_status": "cancelled_no_fill_at_close",
            "risk_reason": "Risk Engine hard close cancels live pendings at operational close.",
            "pricing_source": "tick_replay",
            "has_fill": False,
            "has_exit": False,
            "forced_close": "false",
            "fill_ambiguity": str(state.fill_ambiguity).lower(),
            "ticks_seen": state.ticks_seen,
            "first_tick": state.first_tick,
            "last_tick": state.last_tick,
        }
    if not state.closed or state.exit_price is None or state.fill_timestamp is None or state.fill_price is None:
        return _unpriced_setup_row(state.setup_row, "tick_state_missing_close_or_price")
    pnl = _pnl(state.fill_side, state.fill_price, state.exit_price, state.replay_config)
    return {
        **_base_lifecycle_row(state.setup_row),
        "protected_status": "filled_tick_replayed_exit",
        "risk_status": "kept_tick_replayed_exit",
        "risk_reason": "Candidate/current setup priced with tick-first replay and Risk Engine close guard.",
        "pricing_source": "tick_replay",
        "has_fill": True,
        "has_exit": True,
        "fill_timestamp": _format_dt(state.fill_timestamp),
        "fill_side": state.fill_side,
        "fill_price": _round(state.fill_price),
        "duration_to_fill_seconds": int((state.fill_timestamp - state.setup_dt).total_seconds()),
        "exit_timestamp": _format_dt(state.exit_timestamp or state.close_dt),
        "exit_reason": state.exit_reason,
        "exit_price": _round(state.exit_price),
        "duration_seconds": int(((state.exit_timestamp or state.close_dt) - state.fill_timestamp).total_seconds()),
        "gross_pnl": _round(pnl),
        "net_pnl": _round(pnl),
        "mfe_points": _round(state.mfe_points),
        "mae_points": _round(state.mae_points),
        "trailing_activated": str(state.trailing_activated).lower(),
        "trailing_modifications": state.trailing_modifications,
        "forced_close": str(state.force_close).lower(),
        "fill_ambiguity": str(state.fill_ambiguity).lower(),
        "ticks_seen": state.ticks_seen,
        "first_tick": state.first_tick,
        "last_tick": state.last_tick,
        "commission": 0.0,
        "swap": 0.0,
    }


def _fallback_replay_row(
    row: dict[str, Any],
    m1_by_day: dict[Any, list[OhlcvBar]],
    m5_by_day: dict[Any, list[OhlcvBar]],
    replay_config: AzirReplicaConfig,
    risk_config: AzirRiskConfig,
    reason: str,
) -> dict[str, Any]:
    replay_rows = replay_candidate_lifecycle(
        candidate_rows=[row],
        m1_by_day=m1_by_day,
        m5_by_day=m5_by_day,
        replay_config=replay_config,
        risk_config=risk_config,
    )
    if not replay_rows:
        return _unpriced_setup_row(row, f"{reason}: no fallback replay row")
    result = dict(replay_rows[0])
    result["tick_fallback_reason"] = reason
    return result


def _daily_setup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_day: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("event_type") not in {"opportunity", "blocked_friday"}:
            continue
        by_day[_timestamp(row).date().isoformat()] = row
    return [by_day[day] for day in sorted(by_day)]


def _resolve_fill_side_from_tick(state: TickReplayState, bid: float, ask: float, hit_buy: bool, hit_sell: bool) -> str:
    if hit_buy and not hit_sell:
        return "buy"
    if hit_sell and not hit_buy:
        return "sell"
    mid = (bid + ask) / 2.0
    buy_distance = abs(mid - (state.buy_entry or mid))
    sell_distance = abs(mid - (state.sell_entry or mid))
    return "buy" if buy_distance <= sell_distance else "sell"


def _stop_hit(side: str, price: float, stop: float) -> bool:
    return price <= stop if side == "buy" else price >= stop


def _target_hit(side: str, price: float, target: float) -> bool:
    return price >= target if side == "buy" else price <= target


def _conservative_stop_fill(side: str, price: float, stop: float) -> float:
    return min(price, stop) if side == "buy" else max(price, stop)


def build_tick_coverage_report(
    lifecycle_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    tick_inspection: dict[str, Any],
    m1_bars: list[OhlcvBar],
    m5_bars: list[OhlcvBar],
) -> dict[str, Any]:
    setup_rows = _daily_setup_rows(source_rows)
    priced = [row for row in lifecycle_rows if row.get("has_exit")]
    source_counts = Counter(row.get("pricing_source", "") for row in priced)
    no_fill = [row for row in lifecycle_rows if row.get("protected_status") == "cancelled_no_fill_at_close"]
    no_order = [row for row in lifecycle_rows if row.get("protected_status") == "no_order_placed"]
    blocked = [row for row in lifecycle_rows if row.get("risk_status") in {"blocked_friday", "prevented"}]
    unpriced = [row for row in lifecycle_rows if row.get("risk_status") == "unpriced"]
    return {
        "setup_rows": len(setup_rows),
        "lifecycle_rows": len(lifecycle_rows),
        "closed_trades_priced": len(priced),
        "closed_trades_priced_pct_of_setups": _pct(len(priced), len(setup_rows)),
        "tick_priced_trades": source_counts.get("tick_replay", 0),
        "tick_priced_pct_of_priced": _pct(source_counts.get("tick_replay", 0), len(priced)),
        "m1_fallback_trades": source_counts.get("m1_replay", 0),
        "m1_fallback_pct_of_priced": _pct(source_counts.get("m1_replay", 0), len(priced)),
        "m5_fallback_trades": source_counts.get("m5_fallback_proxy", 0),
        "m5_fallback_pct_of_priced": _pct(source_counts.get("m5_fallback_proxy", 0), len(priced)),
        "unpriced_trades": len(unpriced),
        "cancelled_no_fill_setups": len(no_fill),
        "no_order_setups": len(no_order),
        "blocked_or_prevented_setups": len(blocked),
        "tick_first_time": tick_inspection.get("first_time", ""),
        "tick_last_time": tick_inspection.get("last_time", ""),
        "tick_size_bytes": tick_inspection.get("size_bytes", 0),
        "tick_has_bid_ask_columns": tick_inspection.get("has_bid_ask_columns", False),
        "m1_first_bar": _bar_range(m1_bars)["first"],
        "m1_last_bar": _bar_range(m1_bars)["last"],
        "m5_first_bar": _bar_range(m5_bars)["first"],
        "m5_last_bar": _bar_range(m5_bars)["last"],
    }


def _tick_comparison_rows(left_metrics: dict[str, Any], candidate_metrics: dict[str, Any], left_label: str) -> list[dict[str, Any]]:
    rows = build_comparison_rows(left_metrics, candidate_metrics, left_label)
    for row in rows:
        if PROTECTED_CANDIDATE_NAME in row:
            row[TICK_CANDIDATE_NAME] = row.pop(PROTECTED_CANDIDATE_NAME)
    return rows


def build_exit_distribution(candidate_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, source in [("current", current_rows), ("fractal", candidate_rows)]:
        exits = [row for row in source if row.get("has_exit")]
        counts = Counter(row.get("exit_reason", "") for row in exits)
        for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "dataset": label,
                    "exit_reason": reason,
                    "trades": count,
                    "pct_of_trades": _pct(count, len(exits)),
                }
            )
    return rows


def build_tick_decision(
    candidate_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
    protected_metrics: dict[str, Any],
    coverage: dict[str, Any],
    previous_fractal: dict[str, Any],
) -> dict[str, Any]:
    candidate_net = _float(candidate_metrics.get("net_pnl"))
    current_net = _float(current_metrics.get("net_pnl"))
    candidate_pf = _float(candidate_metrics.get("profit_factor"))
    current_pf = _float(current_metrics.get("profit_factor"))
    tick_pct = _float(coverage.get("tick_priced_pct_of_priced"))
    m5_pct = _float(coverage.get("m5_fallback_pct_of_priced"))
    unpriced = int(coverage.get("unpriced_trades", 0) or 0)
    improves_same_method = candidate_net > current_net and candidate_pf >= current_pf
    improves_protected_reference = candidate_net > _float(protected_metrics.get("net_pnl"))
    enough_tick_for_promotion = tick_pct >= 35.0 and m5_pct <= 40.0 and unpriced == 0
    may_replace = improves_same_method and improves_protected_reference and enough_tick_for_promotion
    return {
        "candidate_beats_current_same_methodology": improves_same_method,
        "candidate_beats_current_protected_reference": improves_protected_reference,
        "candidate_tick_priced_pct": tick_pct,
        "candidate_m5_fallback_pct": m5_pct,
        "may_promote_to_final_replacement": may_replace,
        "may_keep_as_serious_candidate": improves_same_method and improves_protected_reference and unpriced == 0,
        "recommended_next_sprint": "mt5_fractal_candidate_full_lifecycle_export_or_tick_gap_closure_v1"
        if not may_replace
        else "freeze_fractal_protected_economic_v2_candidate",
        "reason": _decision_reason_tick(improves_same_method, enough_tick_for_promotion, tick_pct, m5_pct, unpriced),
        "previous_fractal_net_pnl": previous_fractal.get("candidate_metrics", {}).get("net_pnl", ""),
        "previous_fractal_profit_factor": previous_fractal.get("candidate_metrics", {}).get("profit_factor", ""),
    }


def _decision_reason_tick(
    improves_same_method: bool,
    enough_tick_for_promotion: bool,
    tick_pct: float,
    m5_pct: float,
    unpriced: int,
) -> str:
    if not improves_same_method:
        return "The fractal candidate no longer beats current Azir under the same tick-first replay methodology."
    if enough_tick_for_promotion:
        return "The fractal candidate beats current Azir under the same tick-first replay and has adequate tick/fallback coverage."
    return (
        "The fractal candidate still beats current Azir under the same methodology, but coverage is not strong enough "
        f"for final replacement yet: tick={tick_pct}%, M5 fallback={m5_pct}%, unpriced={unpriced}."
    )


def _load_previous_fractal(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summary_markdown(report: dict[str, Any]) -> str:
    current = report["current_tick_same_methodology_metrics"]
    candidate = report["candidate_tick_metrics"]
    coverage = report["coverage"]
    decision = report["decision"]
    return (
        "# Fractal Tick Replay v1\n\n"
        "## Executive Summary\n\n"
        f"- Current same-methodology net/PF/expectancy: {current.get('net_pnl')} / "
        f"{current.get('profit_factor')} / {current.get('expectancy')}.\n"
        f"- Fractal tick-first net/PF/expectancy: {candidate.get('net_pnl')} / "
        f"{candidate.get('profit_factor')} / {candidate.get('expectancy')}.\n"
        f"- Candidate tick priced trades: {coverage.get('tick_priced_trades')} "
        f"({coverage.get('tick_priced_pct_of_priced')}% of priced trades).\n"
        f"- Fallback: M1 {coverage.get('m1_fallback_trades')}, M5 {coverage.get('m5_fallback_trades')}, "
        f"unpriced {coverage.get('unpriced_trades')}.\n"
        f"- Final replacement now: {decision['may_promote_to_final_replacement']}.\n\n"
        "## Decision\n\n"
        f"{decision['reason']}\n"
    )


def promotion_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    coverage = report["coverage"]
    return (
        "# Fractal Promotion Decision\n\n"
        f"- Beats current same-methodology replay: {decision['candidate_beats_current_same_methodology']}.\n"
        f"- Beats current protected reference: {decision['candidate_beats_current_protected_reference']}.\n"
        f"- Keep as serious candidate: {decision['may_keep_as_serious_candidate']}.\n"
        f"- Promote to final replacement now: {decision['may_promote_to_final_replacement']}.\n"
        f"- Tick priced pct: {coverage['tick_priced_pct_of_priced']}.\n"
        f"- M5 fallback pct: {coverage['m5_fallback_pct_of_priced']}.\n"
        f"- Recommended next sprint: `{decision['recommended_next_sprint']}`.\n\n"
        f"{decision['reason']}\n"
    )


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "sprint": report["sprint"],
        "candidate_net_pnl": report["candidate_tick_metrics"].get("net_pnl"),
        "current_same_methodology_net_pnl": report["current_tick_same_methodology_metrics"].get("net_pnl"),
        "candidate_profit_factor": report["candidate_tick_metrics"].get("profit_factor"),
        "tick_priced_trades": report["coverage"].get("tick_priced_trades"),
        "m1_fallback_trades": report["coverage"].get("m1_fallback_trades"),
        "m5_fallback_trades": report["coverage"].get("m5_fallback_trades"),
        "unpriced_trades": report["coverage"].get("unpriced_trades"),
        "may_promote_to_final_replacement": report["decision"]["may_promote_to_final_replacement"],
        "recommended_next_sprint": report["decision"]["recommended_next_sprint"],
        "reason": report["decision"]["reason"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
