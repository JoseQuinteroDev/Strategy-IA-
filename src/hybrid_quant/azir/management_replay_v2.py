"""Tick-first management replay for protected Azir trades.

This module upgrades the V1 M1-only management replay by using broker-exported
tick data when available. M1 remains a documented fallback, and cases without
enough evidence stay unpriced.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from hybrid_quant.env.azir_management_env import (
    ACTION_BASE_MANAGEMENT,
    ACTION_CLOSE_EARLY,
    ACTION_MOVE_TO_BREAK_EVEN,
    ACTION_TRAILING_AGGRESSIVE,
    ACTION_TRAILING_CONSERVATIVE,
    MANAGEMENT_ACTIONS,
    AzirManagementEvent,
    build_azir_management_replay_dataset,
)
from hybrid_quant.risk.azir_state import AzirRiskConfig

from .economic_audit import _parse_timestamp, _round, _to_float, _write_csv
from .management_replay import (
    HeuristicSpec,
    ManagementReplayConfig,
    _benchmark_quality,
    _choose_heuristic_action,
    _delta,
    _event_key,
    _minutes_between,
    load_config as load_v1_config,
    price_management_action as price_m1_management_action,
)
from .replica import load_ohlcv_csv


DEFAULT_CONFIG_PATH = Path("configs/experiments/azir_management_price_replay_v1.yaml")


@dataclass(frozen=True)
class TickRecord:
    timestamp: datetime
    time_msc: int
    bid: float
    ask: float
    last: float | None = None
    flags: int | None = None


@dataclass(frozen=True)
class TickInspection:
    path: str
    size_bytes: int
    columns: list[str]
    row_count: int
    first_time: str
    last_time: str
    first_time_msc: str
    last_time_msc: str
    bid_ask_complete_rows: int
    bid_ask_complete_pct: float
    symbol_apparent: str
    timezone_apparent: str


@dataclass(frozen=True)
class TickWindow:
    key: str
    start_text: str
    end_text: str


@dataclass(frozen=True)
class V2ActionReplayResult:
    event_key: str
    setup_day: str
    fill_timestamp: str
    exit_timestamp: str
    side: str
    action: str
    action_id: int
    pricing_source: str
    status: str
    exit_reason: str
    exit_price: float | None
    net_pnl: float | None
    duration_minutes: float | None
    tick_count_used: int
    m1_bars_used: int
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "event_key": self.event_key,
            "setup_day": self.setup_day,
            "fill_timestamp": self.fill_timestamp,
            "exit_timestamp": self.exit_timestamp,
            "side": self.side,
            "action": self.action,
            "action_id": self.action_id,
            "pricing_source": self.pricing_source,
            "status": self.status,
            "exit_reason": self.exit_reason,
            "exit_price": "" if self.exit_price is None else _round(self.exit_price),
            "net_pnl": "" if self.net_pnl is None else _round(self.net_pnl),
            "duration_minutes": "" if self.duration_minutes is None else _round(self.duration_minutes),
            "tick_count_used": self.tick_count_used,
            "m1_bars_used": self.m1_bars_used,
            "notes": self.notes,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay Azir management actions using tick data first, M1 fallback second.")
    parser.add_argument("--mt5-log-path", required=True)
    parser.add_argument("--protected-report-path", required=True)
    parser.add_argument("--m1-input-path", required=True)
    parser.add_argument("--tick-input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--symbol", default="XAUUSD-STD")
    parser.add_argument("--tick-symbol-label", default="XAUUSD-STD")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    replay_config, heuristics = load_v1_config(Path(args.config_path) if args.config_path else None)
    report = run_management_price_replay_v2(
        mt5_log_path=Path(args.mt5_log_path),
        protected_report_path=Path(args.protected_report_path),
        m1_input_path=Path(args.m1_input_path),
        tick_input_path=Path(args.tick_input_path),
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
        tick_symbol_label=args.tick_symbol_label,
        replay_config=replay_config,
        heuristics=heuristics,
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_management_price_replay_v2(
    *,
    mt5_log_path: Path,
    protected_report_path: Path,
    m1_input_path: Path,
    tick_input_path: Path,
    output_dir: Path,
    symbol: str = "XAUUSD-STD",
    tick_symbol_label: str = "XAUUSD-STD",
    replay_config: ManagementReplayConfig | None = None,
    heuristics: list[HeuristicSpec] | None = None,
) -> dict[str, Any]:
    replay_config = replay_config or ManagementReplayConfig()
    heuristics = heuristics or [HeuristicSpec("always_base_management", "always", "base_management")]
    events = build_azir_management_replay_dataset(
        mt5_log_path=mt5_log_path,
        protected_report_path=protected_report_path,
        symbol=symbol,
        risk_config=AzirRiskConfig(),
    )
    m1_bars = load_ohlcv_csv(m1_input_path)
    tick_windows = _tick_windows_for_events(events)
    tick_map, inspection = extract_ticks_for_windows(tick_input_path, tick_windows, tick_symbol_label=tick_symbol_label)
    action_results = price_all_management_actions_v2(events, tick_map, m1_bars, replay_config)
    heuristic_rows, same_coverage_rows, exit_distribution = evaluate_management_heuristics_v2(events, action_results, heuristics)
    coverage_rows = build_tick_coverage_rows(events, tick_map)
    report = _build_report_v2(
        mt5_log_path=mt5_log_path,
        protected_report_path=protected_report_path,
        m1_input_path=m1_input_path,
        tick_input_path=tick_input_path,
        inspection=inspection,
        events=events,
        coverage_rows=coverage_rows,
        action_results=action_results,
        heuristic_rows=heuristic_rows,
        exit_distribution=exit_distribution,
        replay_config=replay_config,
        heuristics=heuristics,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv([row.to_row() for row in action_results], output_dir / "management_tick_replay_cases.csv")
    _write_csv(coverage_rows, output_dir / "management_tick_coverage_report.csv")
    _write_csv(heuristic_rows, output_dir / "management_heuristics_comparison_v2.csv")
    _write_csv(same_coverage_rows, output_dir / "heuristics_vs_base_same_coverage_v2.csv")
    _write_csv(exit_distribution, output_dir / "management_exit_distribution_v2.csv")
    (output_dir / "management_replay_v2_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "management_replay_v2_summary.md").write_text(_summary_markdown_v2(report), encoding="utf-8")
    (output_dir / "management_limitations_v2.md").write_text(_limitations_markdown_v2(report), encoding="utf-8")
    return report


def extract_ticks_for_windows(
    tick_input_path: Path,
    windows: list[TickWindow],
    *,
    tick_symbol_label: str,
) -> tuple[dict[str, list[TickRecord]], TickInspection]:
    if not tick_input_path.exists():
        raise FileNotFoundError(f"Tick CSV does not exist: {tick_input_path}")
    ticks_by_key: dict[str, list[TickRecord]] = {window.key: [] for window in windows}
    windows_sorted = sorted(windows, key=lambda item: item.start_text)
    active: list[TickWindow] = []
    next_index = 0
    row_count = 0
    complete_bid_ask = 0
    first_time = ""
    last_time = ""
    first_time_msc = ""
    last_time_msc = ""

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
            row_count += 1
            tick_time = parts[index["time"]]
            tick_msc_text = parts[index["time_msc"]]
            if not first_time:
                first_time = tick_time
                first_time_msc = tick_msc_text
            last_time = tick_time
            last_time_msc = tick_msc_text
            bid_text = parts[index["bid"]]
            ask_text = parts[index["ask"]]
            has_bid_ask = bool(bid_text and ask_text and bid_text != "0.00" and ask_text != "0.00")
            if has_bid_ask:
                complete_bid_ask += 1

            while next_index < len(windows_sorted) and windows_sorted[next_index].start_text <= tick_time:
                active.append(windows_sorted[next_index])
                next_index += 1
            if active:
                active = [window for window in active if window.end_text >= tick_time]
            if not active or not has_bid_ask:
                continue

            tick = TickRecord(
                timestamp=_parse_tick_timestamp(tick_time),
                time_msc=int(tick_msc_text),
                bid=float(bid_text),
                ask=float(ask_text),
                last=_optional_float(parts[index["last"]]) if "last" in index else None,
                flags=int(float(parts[index["flags"]])) if "flags" in index and parts[index["flags"]] else None,
            )
            for window in active:
                if window.start_text <= tick_time <= window.end_text:
                    ticks_by_key[window.key].append(tick)

    inspection = TickInspection(
        path=str(tick_input_path),
        size_bytes=tick_input_path.stat().st_size,
        columns=columns,
        row_count=row_count,
        first_time=first_time,
        last_time=last_time,
        first_time_msc=first_time_msc,
        last_time_msc=last_time_msc,
        bid_ask_complete_rows=complete_bid_ask,
        bid_ask_complete_pct=_round(complete_bid_ask / row_count * 100.0) if row_count else 0.0,
        symbol_apparent=tick_symbol_label,
        timezone_apparent="MT5 exported clock; no timezone column. time and time_msc are internally aligned.",
    )
    return ticks_by_key, inspection


def price_all_management_actions_v2(
    events: list[AzirManagementEvent],
    tick_map: dict[str, list[TickRecord]],
    m1_bars: list[Any],
    config: ManagementReplayConfig,
) -> list[V2ActionReplayResult]:
    rows: list[V2ActionReplayResult] = []
    for event in events:
        for action_id in MANAGEMENT_ACTIONS:
            rows.append(price_management_action_v2(event, tick_map.get(_event_key(event), []), m1_bars, action_id, config))
    return rows


def price_management_action_v2(
    event: AzirManagementEvent,
    ticks: list[TickRecord],
    m1_bars: list[Any],
    action_id: int,
    config: ManagementReplayConfig,
) -> V2ActionReplayResult:
    if action_id == ACTION_BASE_MANAGEMENT:
        return _base_result(event)
    tick_result = _price_with_ticks(event, ticks, action_id, config)
    if tick_result.net_pnl is not None:
        return tick_result
    m1_result = price_m1_management_action(event, m1_bars, action_id, config)
    if m1_result.net_pnl is not None:
        return V2ActionReplayResult(
            event_key=_event_key(event),
            setup_day=event.setup_day,
            fill_timestamp=event.fill_timestamp.isoformat(sep=" "),
            exit_timestamp=m1_result.exit_timestamp,
            side=event.side,
            action=m1_result.action,
            action_id=m1_result.action_id,
            pricing_source="m1_fallback",
            status=m1_result.status,
            exit_reason=m1_result.exit_reason,
            exit_price=m1_result.exit_price,
            net_pnl=m1_result.net_pnl,
            duration_minutes=m1_result.duration_minutes,
            tick_count_used=0,
            m1_bars_used=m1_result.m1_bars_used,
            notes="Tick coverage insufficient; used documented M1 fallback.",
        )
    return tick_result


def evaluate_management_heuristics_v2(
    events: list[AzirManagementEvent],
    action_results: list[V2ActionReplayResult],
    heuristics: list[HeuristicSpec],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    results_by_key_action = {(row.event_key, row.action): row for row in action_results}
    heuristic_rows: list[dict[str, Any]] = []
    same_coverage_rows: list[dict[str, Any]] = []
    exit_distribution: list[dict[str, Any]] = []

    for heuristic in heuristics:
        chosen: list[V2ActionReplayResult] = []
        base_same_coverage: list[V2ActionReplayResult] = []
        source_counts: Counter[str] = Counter()
        unpriced = 0
        for event in events:
            action = _choose_heuristic_action(event, heuristic)
            result = results_by_key_action.get((_event_key(event), action))
            if result is None or result.net_pnl is None:
                unpriced += 1
                continue
            chosen.append(result)
            source_counts[result.pricing_source] += 1
            base_same_coverage.append(results_by_key_action[(_event_key(event), MANAGEMENT_ACTIONS[ACTION_BASE_MANAGEMENT])])
        metrics = _v2_metrics(chosen)
        base_metrics = _v2_metrics(base_same_coverage)
        row = {
            "heuristic": heuristic.name,
            "mode": heuristic.mode,
            "action": heuristic.action,
            "side_filter": heuristic.side,
            "threshold_points": heuristic.threshold_points,
            "events": len(events),
            "priced_trades": len(chosen),
            "tick_priced_trades": source_counts.get("tick", 0),
            "m1_fallback_trades": source_counts.get("m1_fallback", 0),
            "observed_base_trades": source_counts.get("observed_protected_benchmark", 0),
            "unpriced_trades": unpriced,
            "coverage_pct": _round(len(chosen) / len(events) * 100.0) if events else 0.0,
            "tick_priced_pct": _round(source_counts.get("tick", 0) / len(chosen) * 100.0) if chosen else 0.0,
            "m1_fallback_pct": _round(source_counts.get("m1_fallback", 0) / len(chosen) * 100.0) if chosen else 0.0,
            **metrics,
            "base_same_coverage_net_pnl": base_metrics["net_pnl"],
            "base_same_coverage_profit_factor": base_metrics["profit_factor"],
            "base_same_coverage_expectancy": base_metrics["expectancy"],
            "base_same_coverage_max_drawdown": base_metrics["max_drawdown"],
            "delta_net_pnl_vs_base_same_coverage": _delta(base_metrics["net_pnl"], metrics["net_pnl"]),
            "delta_expectancy_vs_base_same_coverage": _delta(base_metrics["expectancy"], metrics["expectancy"]),
            "benchmark_quality": _benchmark_quality(len(chosen), len(events), unpriced),
        }
        heuristic_rows.append(row)
        same_coverage_rows.append(
            {
                "heuristic": heuristic.name,
                "priced_trades": len(chosen),
                "tick_priced_trades": source_counts.get("tick", 0),
                "m1_fallback_trades": source_counts.get("m1_fallback", 0),
                "heuristic_net_pnl": metrics["net_pnl"],
                "base_same_coverage_net_pnl": base_metrics["net_pnl"],
                "delta_net_pnl": row["delta_net_pnl_vs_base_same_coverage"],
                "heuristic_profit_factor": metrics["profit_factor"],
                "base_same_coverage_profit_factor": base_metrics["profit_factor"],
                "heuristic_max_drawdown": metrics["max_drawdown"],
                "base_same_coverage_max_drawdown": base_metrics["max_drawdown"],
            }
        )
        for exit_reason, count in sorted(Counter(result.exit_reason for result in chosen).items()):
            exit_distribution.append(
                {
                    "heuristic": heuristic.name,
                    "exit_reason": exit_reason,
                    "trades": count,
                    "pct_of_priced_trades": _round(count / len(chosen) * 100.0) if chosen else 0.0,
                }
            )

    heuristic_rows.sort(
        key=lambda row: (
            row["benchmark_quality"] in {"full_coverage", "usable_m1_subset"},
            _float(row["delta_net_pnl_vs_base_same_coverage"]),
            _float(row["profit_factor"]),
            -_float(row["max_drawdown"]),
        ),
        reverse=True,
    )
    return heuristic_rows, same_coverage_rows, exit_distribution


def _price_with_ticks(
    event: AzirManagementEvent,
    ticks: list[TickRecord],
    action_id: int,
    config: ManagementReplayConfig,
) -> V2ActionReplayResult:
    if not ticks:
        return _unpriced_v2(event, action_id, "unpriced_no_tick_coverage")
    fill_dt = event.fill_timestamp
    exit_dt = _parse_timestamp(event.trade.get("exit_timestamp"))
    entry = _to_float(event.trade.get("fill_price"))
    if entry is None or exit_dt.year <= 1 or event.side not in {"buy", "sell"}:
        return _unpriced_v2(event, action_id, "missing_entry_side_or_exit")

    if action_id == ACTION_CLOSE_EARLY:
        return _price_close_early_ticks(event, ticks, action_id, config, entry, fill_dt, exit_dt)

    activation = _activation_points_for_action(action_id, config)
    step_points = _step_points_for_action(action_id, config)
    if activation is None:
        return _unpriced_v2(event, action_id, "unsupported_tick_action")

    stop: float | None = None
    activated = False
    best_favorable = entry
    target = entry + config.tp_points * config.point if event.side == "buy" else entry - config.tp_points * config.point

    for tick in ticks:
        if tick.timestamp < fill_dt or tick.timestamp > exit_dt:
            continue
        price = _close_price(event.side, tick)
        if not activated:
            favorable_points = _favorable_points(event.side, entry, price, config.point)
            if favorable_points < activation:
                continue
            activated = True
            if action_id == ACTION_MOVE_TO_BREAK_EVEN:
                stop = entry
            else:
                best_favorable = price
                stop = _trailing_stop(event.side, best_favorable, step_points or 0.0, config.point)
            continue

        if action_id in {ACTION_TRAILING_CONSERVATIVE, ACTION_TRAILING_AGGRESSIVE}:
            if _is_more_favorable(event.side, price, best_favorable):
                best_favorable = price
                stop = _trailing_stop(event.side, best_favorable, step_points or 0.0, config.point)

        if stop is not None and _stop_hit(event.side, price, stop):
            return _tick_exit(event, action_id, tick, _conservative_stop_fill(event.side, price, stop), "tick_management_stop_hit", len(ticks), config)
        if _target_hit(event.side, price, target):
            return _tick_exit(event, action_id, tick, target, "tick_target_hit_after_management_activation", len(ticks), config)

    if not activated:
        return V2ActionReplayResult(
            event_key=_event_key(event),
            setup_day=event.setup_day,
            fill_timestamp=fill_dt.isoformat(sep=" "),
            exit_timestamp=exit_dt.isoformat(sep=" "),
            side=event.side,
            action=MANAGEMENT_ACTIONS[action_id],
            action_id=action_id,
            pricing_source="observed_protected_benchmark",
            status="priced_observed_protected_no_tick_activation",
            exit_reason="base_management_no_tick_activation",
            exit_price=None,
            net_pnl=event.protected_net_pnl,
            duration_minutes=_minutes_between(fill_dt, exit_dt),
            tick_count_used=len(ticks),
            m1_bars_used=0,
            notes="Tick path never activated this management rule; base protected outcome retained.",
        )

    horizon_tick = _last_tick_before_or_at(ticks, exit_dt)
    if horizon_tick is None:
        return _unpriced_v2(event, action_id, "unpriced_no_tick_at_horizon_after_activation")
    return _tick_exit(event, action_id, horizon_tick, _close_price(event.side, horizon_tick), "tick_horizon_close_after_management_activation", len(ticks), config)


def _price_close_early_ticks(
    event: AzirManagementEvent,
    ticks: list[TickRecord],
    action_id: int,
    config: ManagementReplayConfig,
    entry: float,
    fill_dt: datetime,
    exit_dt: datetime,
) -> V2ActionReplayResult:
    decision_time = fill_dt + timedelta(seconds=60)
    tick = next((item for item in ticks if decision_time <= item.timestamp <= exit_dt), None)
    if tick is None:
        return _unpriced_v2(event, action_id, "unpriced_no_tick_at_close_early_checkpoint")
    price = _close_price(event.side, tick)
    pnl = _pnl(event.side, entry, price, _lot_size(event, config), config.contract_size)
    return V2ActionReplayResult(
        event_key=_event_key(event),
        setup_day=event.setup_day,
        fill_timestamp=fill_dt.isoformat(sep=" "),
        exit_timestamp=tick.timestamp.isoformat(sep=" "),
        side=event.side,
        action=MANAGEMENT_ACTIONS[action_id],
        action_id=action_id,
        pricing_source="tick",
        status="priced_with_tick",
        exit_reason="close_early_tick_checkpoint",
        exit_price=price,
        net_pnl=pnl,
        duration_minutes=_minutes_between(fill_dt, tick.timestamp),
        tick_count_used=len(ticks),
        m1_bars_used=0,
        notes="Market close at first tick at or after fill+60s, using bid for buy and ask for sell.",
    )


def _tick_exit(
    event: AzirManagementEvent,
    action_id: int,
    tick: TickRecord,
    exit_price: float,
    reason: str,
    tick_count: int,
    config: ManagementReplayConfig,
) -> V2ActionReplayResult:
    entry = _to_float(event.trade.get("fill_price")) or 0.0
    pnl = _pnl(event.side, entry, exit_price, _lot_size(event, config), config.contract_size)
    return V2ActionReplayResult(
        event_key=_event_key(event),
        setup_day=event.setup_day,
        fill_timestamp=event.fill_timestamp.isoformat(sep=" "),
        exit_timestamp=tick.timestamp.isoformat(sep=" "),
        side=event.side,
        action=MANAGEMENT_ACTIONS[action_id],
        action_id=action_id,
        pricing_source="tick",
        status="priced_with_tick",
        exit_reason=reason,
        exit_price=exit_price,
        net_pnl=pnl,
        duration_minutes=_minutes_between(event.fill_timestamp, tick.timestamp),
        tick_count_used=tick_count,
        m1_bars_used=0,
        notes="Tick replay uses bid for long exits and ask for short exits.",
    )


def build_tick_coverage_rows(events: list[AzirManagementEvent], tick_map: dict[str, list[TickRecord]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        ticks = tick_map.get(_event_key(event), [])
        exit_dt = _parse_timestamp(event.trade.get("exit_timestamp"))
        duration = _minutes_between(event.fill_timestamp, exit_dt) if exit_dt.year > 1 else None
        rows.append(
            {
                "event_key": _event_key(event),
                "setup_day": event.setup_day,
                "fill_timestamp": event.fill_timestamp.isoformat(sep=" "),
                "exit_timestamp": exit_dt.isoformat(sep=" ") if exit_dt.year > 1 else "",
                "side": event.side,
                "duration_minutes": "" if duration is None else _round(duration),
                "tick_count": len(ticks),
                "first_tick": ticks[0].timestamp.isoformat(sep=" ") if ticks else "",
                "last_tick": ticks[-1].timestamp.isoformat(sep=" ") if ticks else "",
                "has_tick_coverage": bool(ticks),
                "tick_source_usable_for_management": bool(ticks and exit_dt.year > 1),
            }
        )
    return rows


def _build_report_v2(
    *,
    mt5_log_path: Path,
    protected_report_path: Path,
    m1_input_path: Path,
    tick_input_path: Path,
    inspection: TickInspection,
    events: list[AzirManagementEvent],
    coverage_rows: list[dict[str, Any]],
    action_results: list[V2ActionReplayResult],
    heuristic_rows: list[dict[str, Any]],
    exit_distribution: list[dict[str, Any]],
    replay_config: ManagementReplayConfig,
    heuristics: list[HeuristicSpec],
) -> dict[str, Any]:
    priced_by_action = {
        label: len([row for row in action_results if row.action == label and row.net_pnl is not None])
        for label in MANAGEMENT_ACTIONS.values()
    }
    source_counts = Counter(row.pricing_source for row in action_results if row.net_pnl is not None)
    status_counts = Counter(row.status for row in action_results)
    tick_events = len([row for row in coverage_rows if row["has_tick_coverage"]])
    base = next((row for row in heuristic_rows if row["heuristic"] == "always_base_management"), {})
    candidates = [
        row
        for row in heuristic_rows
        if row["heuristic"] != "always_base_management"
        and row["benchmark_quality"] in {"full_coverage", "usable_m1_subset"}
        and _float(row["delta_net_pnl_vs_base_same_coverage"]) > 0.0
    ]
    best_candidate = max(
        candidates,
        key=lambda row: (_float(row["delta_net_pnl_vs_base_same_coverage"]), _float(row["tick_priced_pct"]), _float(row["profit_factor"])),
        default=None,
    )
    strong_candidate = bool(
        best_candidate
        and _float(best_candidate["tick_priced_pct"]) >= 50.0
        and _float(best_candidate["coverage_pct"]) >= 50.0
        and _float(best_candidate["profit_factor"]) >= 1.10
        and _float(best_candidate["delta_net_pnl_vs_base_same_coverage"]) >= 10.0
    )
    return {
        "sprint": "management_price_replay_v2_with_tick_or_broker_execution",
        "mt5_log_path": str(mt5_log_path),
        "protected_report_path": str(protected_report_path),
        "m1_input_path": str(m1_input_path),
        "tick_input_path": str(tick_input_path),
        "tick_inspection": asdict(inspection),
        "events": len(events),
        "events_with_tick_coverage": tick_events,
        "events_with_tick_coverage_pct": _round(tick_events / len(events) * 100.0) if events else 0.0,
        "replay_config": asdict(replay_config),
        "heuristics": [asdict(heuristic) for heuristic in heuristics],
        "pricing_methodology": {
            "base_management": "Observed/revalued protected benchmark net_pnl.",
            "close_early": "First real tick at or after fill+60s; long exits use bid, short exits use ask.",
            "move_to_break_even": "Tick path activates BE after favorable bid/ask movement reaches threshold; no activation keeps base benchmark.",
            "trailing": "Tick path ratchets stop after activation; no activation keeps base benchmark.",
            "fallback": "M1 fallback is used only if tick evidence cannot price the action; otherwise the case is unpriced.",
        },
        "pricing_coverage_by_action": priced_by_action,
        "pricing_source_counts": dict(source_counts),
        "status_counts": dict(status_counts),
        "heuristic_comparison": heuristic_rows,
        "exit_distribution": exit_distribution,
        "decision": {
            "management_valuation_serious_enough_for_heuristic_screening": bool(tick_events > 0),
            "management_benchmark_ready_for_ppo_training": strong_candidate,
            "best_candidate": best_candidate["heuristic"] if best_candidate else "",
            "best_candidate_delta_net_pnl_vs_base_same_coverage": best_candidate["delta_net_pnl_vs_base_same_coverage"] if best_candidate else "",
            "base_management_full_net_pnl": base.get("net_pnl", ""),
            "recommended_next_sprint": "train_management_ppo_for_azir_v1" if strong_candidate else "expand_tick_coverage_or_export_position_lifecycle_ticks_v3",
            "reason": (
                "A tick-priced management heuristic is strong enough on coverage, PF, and uplift to justify management PPO."
                if strong_candidate
                else "Tick replay improves fidelity, but no management heuristic beats base strongly enough to justify PPO yet."
                if best_candidate
                else "No tick/M1-priced management heuristic improves base management on fair same-coverage comparison."
            ),
        },
        "limitations": [
            "Tick CSV begins after many protected Azir trades, so earlier trades still rely on M1 fallback or base benchmark.",
            "The tick CSV has no explicit symbol/timezone column; symbol and clock are inferred from export context and aligned time/time_msc fields.",
            "Spread execution is more realistic with bid/ask ticks, but broker order queue/latency is still not simulated.",
            "Counterfactual management exits are not tick-perfect if the action falls back to M1.",
            "A management benchmark should not be frozen unless the useful tick-priced coverage is broad enough and the uplift is economically material.",
        ],
    }


def _summary_markdown_v2(report: dict[str, Any]) -> str:
    tick = report["tick_inspection"]
    top_lines = "\n".join(
        "- "
        f"`{row['heuristic']}`: net={row['net_pnl']}, PF={row['profit_factor']}, "
        f"exp={row['expectancy']}, DD={row['max_drawdown']}, priced={row['priced_trades']}, "
        f"tick={row['tick_priced_trades']}, m1={row['m1_fallback_trades']}, "
        f"delta_vs_base={row['delta_net_pnl_vs_base_same_coverage']}."
        for row in report["heuristic_comparison"][:6]
    )
    coverage = "\n".join(f"- `{key}`: {value}" for key, value in report["pricing_coverage_by_action"].items())
    decision = report["decision"]
    return (
        "# Azir Management Price Replay V2\n\n"
        "## Executive Summary\n\n"
        "- This sprint uses real broker tick data first, M1 fallback second, and does not train PPO.\n"
        f"- Tick file: `{tick['path']}`.\n"
        f"- Tick rows: {tick['row_count']}; range: {tick['first_time']} -> {tick['last_time']}.\n"
        f"- Bid/ask complete rows: {tick['bid_ask_complete_rows']} ({tick['bid_ask_complete_pct']}%).\n"
        f"- Protected events: {report['events']}; events with tick coverage: {report['events_with_tick_coverage']} ({report['events_with_tick_coverage_pct']}%).\n"
        f"- Base full net PnL: {decision['base_management_full_net_pnl']}.\n"
        f"- Best candidate: `{decision['best_candidate'] or 'none'}`.\n"
        f"- Ready for management PPO: {decision['management_benchmark_ready_for_ppo_training']}.\n\n"
        "## Pricing Coverage By Action\n\n"
        f"{coverage}\n\n"
        "## Top Heuristics\n\n"
        f"{top_lines}\n\n"
        "## Decision\n\n"
        f"- {decision['reason']}\n"
        f"- Recommended next sprint: `{decision['recommended_next_sprint']}`.\n"
    )


def _limitations_markdown_v2(report: dict[str, Any]) -> str:
    methodology = "\n".join(f"- `{key}`: {value}" for key, value in report["pricing_methodology"].items())
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    return (
        "# Management Replay V2 Limitations\n\n"
        "## Methodology\n\n"
        f"{methodology}\n\n"
        "## Limitations\n\n"
        f"{limitations}\n"
    )


def _base_result(event: AzirManagementEvent) -> V2ActionReplayResult:
    exit_dt = _parse_timestamp(event.trade.get("exit_timestamp"))
    return V2ActionReplayResult(
        event_key=_event_key(event),
        setup_day=event.setup_day,
        fill_timestamp=event.fill_timestamp.isoformat(sep=" "),
        exit_timestamp=exit_dt.isoformat(sep=" ") if exit_dt.year > 1 else "",
        side=event.side,
        action=MANAGEMENT_ACTIONS[ACTION_BASE_MANAGEMENT],
        action_id=ACTION_BASE_MANAGEMENT,
        pricing_source="observed_protected_benchmark",
        status="priced_observed_protected",
        exit_reason=str(event.trade.get("exit_reason", "")),
        exit_price=None,
        net_pnl=event.protected_net_pnl,
        duration_minutes=_minutes_between(event.fill_timestamp, exit_dt) if exit_dt.year > 1 else None,
        tick_count_used=0,
        m1_bars_used=0,
        notes="Frozen protected economic benchmark outcome.",
    )


def _unpriced_v2(event: AzirManagementEvent, action_id: int, reason: str) -> V2ActionReplayResult:
    return V2ActionReplayResult(
        event_key=_event_key(event),
        setup_day=event.setup_day,
        fill_timestamp=event.fill_timestamp.isoformat(sep=" "),
        exit_timestamp=str(event.trade.get("exit_timestamp", "")),
        side=event.side,
        action=MANAGEMENT_ACTIONS[action_id],
        action_id=action_id,
        pricing_source="unpriced",
        status=reason,
        exit_reason="unpriced",
        exit_price=None,
        net_pnl=None,
        duration_minutes=None,
        tick_count_used=0,
        m1_bars_used=0,
        notes=reason,
    )


def _tick_windows_for_events(events: list[AzirManagementEvent]) -> list[TickWindow]:
    windows: list[TickWindow] = []
    for event in events:
        exit_dt = _parse_timestamp(event.trade.get("exit_timestamp"))
        if exit_dt.year <= 1 or exit_dt < event.fill_timestamp:
            continue
        windows.append(
            TickWindow(
                key=_event_key(event),
                start_text=event.fill_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                end_text=exit_dt.strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
    return windows


def _tick_column_index(columns: list[str]) -> dict[str, int]:
    required = {"time", "time_msc", "bid", "ask"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"Tick CSV missing required columns: {sorted(missing)}")
    index = {name: columns.index(name) for name in columns}
    return index


def _parse_tick_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace(".", "-"))


def _optional_float(value: Any) -> float | None:
    parsed = _to_float(value)
    return parsed


def _v2_metrics(results: list[V2ActionReplayResult]) -> dict[str, Any]:
    pnl_values = [float(row.net_pnl or 0.0) for row in results]
    metrics = _management_metrics_from_pnl(pnl_values)
    durations = [row.duration_minutes for row in results if row.duration_minutes is not None]
    return {
        **metrics,
        "average_duration_minutes": _round(mean(durations)) if durations else 0.0,
    }


def _management_metrics_from_pnl(pnl_values: list[float]) -> dict[str, Any]:
    from .train_ppo_skip_take import trade_metrics

    return trade_metrics(pnl_values)


def _activation_points_for_action(action_id: int, config: ManagementReplayConfig) -> float | None:
    if action_id == ACTION_MOVE_TO_BREAK_EVEN:
        return config.break_even_activation_points
    if action_id == ACTION_TRAILING_CONSERVATIVE:
        return config.trailing_conservative_activation_points
    if action_id == ACTION_TRAILING_AGGRESSIVE:
        return config.trailing_aggressive_activation_points
    return None


def _step_points_for_action(action_id: int, config: ManagementReplayConfig) -> float | None:
    if action_id == ACTION_TRAILING_CONSERVATIVE:
        return config.trailing_conservative_step_points
    if action_id == ACTION_TRAILING_AGGRESSIVE:
        return config.trailing_aggressive_step_points
    return None


def _close_price(side: str, tick: TickRecord) -> float:
    return tick.bid if side == "buy" else tick.ask


def _favorable_points(side: str, entry: float, price: float, point: float) -> float:
    return (price - entry) / point if side == "buy" else (entry - price) / point


def _is_more_favorable(side: str, price: float, current_best: float) -> bool:
    return price > current_best if side == "buy" else price < current_best


def _trailing_stop(side: str, best_favorable: float, step_points: float, point: float) -> float:
    return best_favorable - step_points * point if side == "buy" else best_favorable + step_points * point


def _stop_hit(side: str, price: float, stop: float) -> bool:
    return price <= stop if side == "buy" else price >= stop


def _target_hit(side: str, price: float, target: float) -> bool:
    return price >= target if side == "buy" else price <= target


def _conservative_stop_fill(side: str, price: float, stop: float) -> float:
    return min(price, stop) if side == "buy" else max(price, stop)


def _last_tick_before_or_at(ticks: list[TickRecord], timestamp: datetime) -> TickRecord | None:
    candidate: TickRecord | None = None
    for tick in ticks:
        if tick.timestamp <= timestamp:
            candidate = tick
        else:
            break
    return candidate


def _pnl(side: str, entry: float, exit_price: float, lot_size: float, contract_size: float) -> float:
    direction = 1.0 if side == "buy" else -1.0
    return (exit_price - entry) * direction * lot_size * contract_size


def _lot_size(event: AzirManagementEvent, config: ManagementReplayConfig) -> float:
    parsed = _to_float(event.setup.get("lot_size"))
    return parsed if parsed is not None and parsed > 0.0 else config.default_lot_size


def _float(value: Any) -> float:
    parsed = _to_float(value)
    return 0.0 if parsed is None else float(parsed)


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "sprint": report["sprint"],
        "tick_input_path": report["tick_input_path"],
        "tick_rows": report["tick_inspection"]["row_count"],
        "tick_range": {
            "start": report["tick_inspection"]["first_time"],
            "end": report["tick_inspection"]["last_time"],
        },
        "events": report["events"],
        "events_with_tick_coverage": report["events_with_tick_coverage"],
        "pricing_coverage_by_action": report["pricing_coverage_by_action"],
        "pricing_source_counts": report["pricing_source_counts"],
        "best_candidate": report["decision"]["best_candidate"],
        "ready_for_management_ppo": report["decision"]["management_benchmark_ready_for_ppo_training"],
        "recommended_next_sprint": report["decision"]["recommended_next_sprint"],
        "reason": report["decision"]["reason"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
