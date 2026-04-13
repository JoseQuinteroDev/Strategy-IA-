"""Deep audit for the frozen Azir setup benchmark.

The MT5 EA remains the source of truth. This module audits the empirical MT5
event log and explicitly separates the setup benchmark, which is now
high-fidelity, from the execution/economic layer, which is still not a frozen
Python benchmark because trailing, MFE/MAE and PnL are tick-dependent.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .comparison import read_event_log


BENCHMARK_NAME = "baseline_azir_setup_frozen_v1"
DEFAULT_SYMBOL = "XAUUSD-STD"


@dataclass(frozen=True)
class AuditPaths:
    mt5_log_path: Path
    parity_dir: Path | None
    output_dir: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the frozen Azir setup benchmark.")
    parser.add_argument("--mt5-log-path", required=True, help="Canonical Azir MT5 event log CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory where audit artifacts will be written.")
    parser.add_argument("--parity-dir", default="", help="Optional final setup parity artifact directory.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--start", default="", help="Optional inclusive start timestamp/date in server time.")
    parser.add_argument("--end", default="", help="Optional inclusive end timestamp/date in server time.")
    parser.add_argument("--docs-output", default="", help="Optional benchmark definition path under docs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parity_dir = Path(args.parity_dir) if args.parity_dir else None
    docs_output = Path(args.docs_output) if args.docs_output else None

    report = run_audit(
        mt5_log_path=Path(args.mt5_log_path),
        output_dir=output_dir,
        parity_dir=parity_dir,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        docs_output=docs_output,
    )
    print(json.dumps(_compact_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_audit(
    *,
    mt5_log_path: Path,
    output_dir: Path,
    parity_dir: Path | None = None,
    symbol: str = DEFAULT_SYMBOL,
    start: str = "",
    end: str = "",
    docs_output: Path | None = None,
) -> dict[str, Any]:
    """Run the audit and write all requested artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    mt5_rows = _filter_events(read_event_log(mt5_log_path), start, end, symbol)
    if not mt5_rows:
        raise ValueError("No MT5 Azir event rows found for the requested filters.")

    records = build_daily_audit_records(mt5_rows)
    parity_summary = _read_parity_summary(parity_dir)
    benchmark = build_benchmark_definition(
        mt5_rows=mt5_rows,
        records=records,
        mt5_log_path=mt5_log_path,
        parity_dir=parity_dir,
        parity_summary=parity_summary,
        symbol=symbol,
    )

    aggregate = summarize_records(records)
    breakdowns = write_audit_artifacts(
        records=records,
        output_dir=output_dir,
        benchmark=benchmark,
        aggregate=aggregate,
        mt5_log_path=mt5_log_path,
        parity_summary=parity_summary,
    )

    report = {
        "benchmark": benchmark,
        "source_log": str(mt5_log_path),
        "parity_dir": str(parity_dir) if parity_dir else None,
        "row_count": len(mt5_rows),
        "event_counts": dict(Counter(row.get("event_type", "") for row in mt5_rows)),
        "aggregate_metrics": aggregate,
        "operational_warnings": _operational_warnings(records),
        "breakdowns": breakdowns,
        "decision": {
            "setup_benchmark_status": "FROZEN",
            "economic_benchmark_status": "NOT_FROZEN_APPROXIMATE",
            "recommended_next_sprint": "economic_audit_before_risk_engine",
            "reason": (
                "Setup/filtros are benchmark-grade, but execution economics still depend "
                "on MT5 tick ordering, trailing updates and broker-side order events."
            ),
        },
    }

    (output_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    benchmark_text = _benchmark_markdown(benchmark)
    (output_dir / "benchmark_definition.md").write_text(benchmark_text, encoding="utf-8")
    if docs_output is not None:
        docs_output.parent.mkdir(parents=True, exist_ok=True)
        docs_output.write_text(benchmark_text, encoding="utf-8")
    (output_dir / "audit_summary.md").write_text(_audit_summary_markdown(report), encoding="utf-8")
    return report


def build_daily_audit_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one canonical audit row per setup day."""

    setups = _canonical_setup_rows(rows)
    grouped = _group_events_by_day(rows)
    days = sorted(set(setups) | set(grouped))
    records: list[dict[str, Any]] = []
    for day in days:
        setup = setups.get(day, {})
        fills = grouped.get(day, {}).get("fill", [])
        exits = grouped.get(day, {}).get("exit", [])
        trailing = grouped.get(day, {}).get("trailing_modified", [])
        cleanup = grouped.get(day, {}).get("no_fill_close_cleanup", [])
        reference = setup or (fills[0] if fills else {}) or (exits[0] if exits else {}) or (trailing[0] if trailing else {})
        setup_dt = _parse_timestamp(reference.get("timestamp", day))
        buy_order = _is_true(setup.get("buy_order_placed"))
        sell_order = _is_true(setup.get("sell_order_placed"))
        order_placed = buy_order or sell_order
        filled = bool(fills)
        is_friday = _is_true(setup.get("is_friday")) or setup.get("event_type") == "blocked_friday"
        atr_points = _to_float(setup.get("atr_points"))
        distance_points = _to_float(setup.get("pending_distance_points"))
        base_record = {
            "day": day,
            "year": setup_dt.year,
            "quarter": f"{setup_dt.year}-Q{((setup_dt.month - 1) // 3) + 1}",
            "month": f"{setup_dt.year}-{setup_dt.month:02d}",
            "weekday": setup_dt.strftime("%A"),
            "setup_hour": setup_dt.hour,
            "event_type": setup.get("event_type", ""),
            "is_friday": is_friday,
            "swing_high": _to_float(setup.get("swing_high")),
            "swing_low": _to_float(setup.get("swing_low")),
            "buy_entry": _to_float(setup.get("buy_entry")),
            "sell_entry": _to_float(setup.get("sell_entry")),
            "pending_distance_points": distance_points,
            "pending_distance_regime": _pending_distance_regime(distance_points),
            "spread_points": _to_float(setup.get("spread_points")),
            "ema20": _to_float(setup.get("ema20")),
            "prev_close": _to_float(setup.get("prev_close")),
            "prev_close_above_ema20": _is_true(setup.get("prev_close_above_ema20")),
            "atr_points": atr_points,
            "atr_regime": _atr_regime(atr_points),
            "atr_filter_enabled": _is_true(setup.get("atr_filter_enabled")),
            "atr_filter_passed": _is_true(setup.get("atr_filter_passed")),
            "rsi": _to_float(setup.get("rsi")),
            "rsi_gate_enabled": _is_true(setup.get("rsi_gate_enabled")),
            "rsi_gate_required": _is_true(setup.get("rsi_gate_required")),
            "trend_filter_enabled": _is_true(setup.get("trend_filter_enabled")),
            "trend_bias": _trend_bias(setup),
            "buy_order_placed": buy_order,
            "sell_order_placed": sell_order,
            "order_placed": order_placed,
            "mt5_opportunity_rows": len(grouped.get(day, {}).get("opportunity", [])),
        }
        event_count = max(len(fills), len(exits), 1)
        for trade_index in range(event_count):
            fill = fills[trade_index] if trade_index < len(fills) else {}
            exit_row = exits[trade_index] if trade_index < len(exits) else {}
            fill_dt = _parse_timestamp(fill.get("timestamp", "")) if fill else None
            exit_dt = _parse_timestamp(exit_row.get("timestamp", "")) if exit_row else None
            trailing_activated = bool(trailing) or _is_true(exit_row.get("trailing_activated"))
            records.append(
                {
                    **base_record,
                    "trade_sequence": trade_index + 1 if fill or exit_row else "",
                    "fill_hour": "" if fill_dt is None else fill_dt.hour,
                    "exit_hour": "" if exit_dt is None else exit_dt.hour,
                    "fill_status": _fill_status(setup, order_placed, bool(fill), is_friday, cleanup),
                    "filled": bool(fill),
                    "fill_side": fill.get("fill_side") or exit_row.get("fill_side") or "",
                    "fill_price": _to_float(fill.get("fill_price") or exit_row.get("fill_price")),
                    "duration_to_fill_seconds": _to_float(fill.get("duration_to_fill_seconds")),
                    "exit_reason": exit_row.get("exit_reason", ""),
                    "gross_pnl": _to_float(exit_row.get("gross_pnl")) or 0.0,
                    "net_pnl": _to_float(exit_row.get("net_pnl")) or 0.0,
                    "commission": _to_float(exit_row.get("commission")) or 0.0,
                    "swap": _to_float(exit_row.get("swap")) or 0.0,
                    "mfe_points": _to_float(exit_row.get("mfe_points")),
                    "mae_points": _to_float(exit_row.get("mae_points")),
                    "trailing_activated": trailing_activated,
                    "trailing_modifications": len(trailing),
                    "has_exit": bool(exit_row),
                }
            )
    return records


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    exits = [row for row in rows if row.get("has_exit")]
    daily = _first_record_by_day(rows)
    pnl_values = [_to_float(row.get("net_pnl")) or 0.0 for row in exits]
    gross_values = [_to_float(row.get("gross_pnl")) or 0.0 for row in exits]
    wins = [value for value in pnl_values if value > 0]
    losses = [value for value in pnl_values if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    filled_trades = len(exits)
    return {
        "setup_days": len([row for row in daily if row.get("event_type") in {"opportunity", "blocked_friday"}]),
        "opportunity_days": len([row for row in daily if row.get("event_type") == "opportunity"]),
        "friday_blocked_days": len([row for row in daily if row.get("is_friday")]),
        "order_placed_days": len([row for row in daily if row.get("order_placed")]),
        "filled_trades": filled_trades,
        "no_fill_days": len([row for row in daily if row.get("fill_status") == "order_no_fill"]),
        "net_pnl": round(sum(pnl_values), 2),
        "gross_pnl": round(sum(gross_values), 2),
        "win_rate": _pct(len(wins), filled_trades),
        "average_win": _round_or_none(mean(wins) if wins else None),
        "average_loss": _round_or_none(mean(losses) if losses else None),
        "payoff": _round_or_none((mean(wins) / abs(mean(losses))) if wins and losses else None),
        "profit_factor": _round_or_none(gross_profit / gross_loss if gross_loss else None),
        "expectancy": _round_or_none(mean(pnl_values) if pnl_values else None),
        "max_drawdown_abs": _round_or_none(_max_drawdown(pnl_values)),
        "max_consecutive_losses": _max_consecutive_losses(pnl_values),
        "average_losing_streak": _round_or_none(_average_losing_streak(pnl_values)),
        "trailing_activated_trades": len([row for row in exits if row.get("trailing_activated")]),
        "tp_exits": len([row for row in exits if _normalized_exit_reason(row) == "take_profit"]),
        "sl_or_trailing_exits": len(
            [row for row in exits if _normalized_exit_reason(row) == "stop_loss_or_trailing_stop"]
        ),
        "session_close_exits": len(
            [row for row in exits if _normalized_exit_reason(row) == "expert_close_or_session_close"]
        ),
    }


def _first_record_by_day(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[Any] = set()
    daily: list[dict[str, Any]] = []
    for row in records:
        day = row.get("day")
        if day in seen:
            continue
        seen.add(day)
        daily.append(row)
    return daily


def _normalized_exit_reason(row: dict[str, Any]) -> str:
    return str(row.get("exit_reason", "")).strip().lower()


def write_audit_artifacts(
    *,
    records: list[dict[str, Any]],
    output_dir: Path,
    benchmark: dict[str, Any],
    aggregate: dict[str, Any],
    mt5_log_path: Path,
    parity_summary: dict[str, Any],
) -> dict[str, Any]:
    _write_records_csv(records, output_dir / "daily_audit_records.csv")
    breakdown_specs = {
        "yearly_breakdown.csv": ("year", "year"),
        "quarterly_breakdown.csv": ("quarter", "quarter"),
        "monthly_breakdown.csv": ("month", "month"),
        "weekday_breakdown.csv": ("weekday", "weekday"),
        "hour_breakdown.csv": ("fill_hour", "fill_hour"),
        "side_breakdown.csv": ("fill_side", "side"),
        "trailing_breakdown.csv": ("trailing_activated", "trailing_activated"),
        "atr_regime_breakdown.csv": ("atr_regime", "atr_regime"),
        "pending_distance_breakdown.csv": ("pending_distance_regime", "pending_distance_regime"),
        "trend_filter_breakdown.csv": ("trend_bias", "trend_bias"),
        "rsi_gate_breakdown.csv": ("rsi_gate_required", "rsi_gate_required"),
        "friday_filter_breakdown.csv": ("is_friday", "is_friday"),
        "fill_status_breakdown.csv": ("fill_status", "fill_status"),
        "exit_reason_breakdown.csv": ("exit_reason", "exit_reason"),
    }
    breakdowns: dict[str, Any] = {}
    for filename, (field, label) in breakdown_specs.items():
        groups = _group_records(records, field)
        rows = [_summary_row(label, group, items) for group, items in groups.items()]
        rows.sort(key=lambda row: str(row["group"]))
        _write_summary_csv(rows, output_dir / filename)
        breakdowns[filename] = rows

    filter_rows = _build_filter_ablation_summary(records)
    _write_summary_csv(filter_rows, output_dir / "filter_ablation_summary.csv")
    breakdowns["filter_ablation_summary.csv"] = filter_rows

    overview = {
        "benchmark_name": benchmark["name"],
        "mt5_log_path": str(mt5_log_path),
        "parity_summary": parity_summary,
        "aggregate_metrics": aggregate,
        "notes": [
            "Breakdowns use canonical daily setup rows plus MT5 fill/exit/trailing events.",
            "Filter ablation is observational grouping, not a counterfactual rerun.",
            "Economic metrics come from the MT5 event log and are not a frozen Python execution benchmark.",
        ],
    }
    (output_dir / "audit_overview.json").write_text(
        json.dumps(overview, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return breakdowns


def build_benchmark_definition(
    *,
    mt5_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    mt5_log_path: Path,
    parity_dir: Path | None,
    parity_summary: dict[str, Any],
    symbol: str,
) -> dict[str, Any]:
    timestamps = [_parse_timestamp(row["timestamp"]) for row in mt5_rows if row.get("timestamp")]
    first_ts = min(timestamps)
    last_ts = max(timestamps)
    setup_days = [row["day"] for row in records if row.get("event_type") in {"opportunity", "blocked_friday"}]
    return {
        "name": BENCHMARK_NAME,
        "status": "official_setup_benchmark_frozen",
        "source_of_truth": [
            r"C:\Users\joseq\Documents\Playground\Azir.mq5",
            str(mt5_log_path),
            str(parity_dir) if parity_dir else "not_provided",
        ],
        "symbol": symbol,
        "timeframe": "M5 setup, M1 RSI gate state when available in MT5",
        "server_time_dependency": "MT5 broker/server time; no timezone conversion in EA.",
        "event_log_range": {
            "first_timestamp": first_ts.isoformat(sep=" "),
            "last_timestamp": last_ts.isoformat(sep=" "),
            "first_setup_day": min(setup_days) if setup_days else None,
            "last_setup_day": max(setup_days) if setup_days else None,
        },
        "inputs": {
            "setup_time": "16:30 broker/server time",
            "close_hour": "22 broker/server time",
            "swing_bars": 10,
            "entry_offset_points": 5,
            "ema_filter": "EMA20 on M5 closed bar shift 1",
            "atr_filter": "SMA true range parity with MT5 iATR observed value; 14 closed M5 bars shift 1",
            "rsi_gate": "Required only when both pendings are placed and distance >= minimum threshold.",
            "friday_filter": "NoTradeFridays blocks setup on Friday.",
        },
        "setup_parity": {
            "setup_day_match_pct": parity_summary.get("setup_day_match_pct"),
            "setup_field_match_pct": parity_summary.get("setup_field_match_pct"),
            "atr_parity_pct": parity_summary.get("atr_parity_pct"),
            "fill_count_match_pct": parity_summary.get("fill_count_match_pct"),
            "remaining_setup_divergence_days": parity_summary.get("remaining_setup_divergence_days"),
        },
        "frozen_scope": [
            "daily setup day presence",
            "swing high/low and pending levels",
            "EMA/ATR setup filters",
            "RSI gate required flag at setup",
            "buy/sell order placement intent after canonical 16:30 deduplication",
        ],
        "not_frozen_scope": [
            "tick-level trailing modifications",
            "MFE/MAE exact path",
            "PnL equivalence in the Python replica",
            "broker order-send failures not inferable from OHLC bars",
        ],
        "limitations": [
            "MT5 logs repeated opportunities inside 16:30; the benchmark uses a canonical daily setup row.",
            "Economic results are empirical MT5 log evidence, not yet a Python execution benchmark.",
            "Trailing and exact exit path need tick replay or explicit approximation rules before Risk Engine/PPO.",
        ],
    }


def _build_filter_ablation_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = (
        ("atr_filter_passed", "atr_filter_passed"),
        ("trend_bias", "trend_bias"),
        ("rsi_gate_required", "rsi_gate_required"),
        ("is_friday", "friday_filter"),
        ("fill_status", "fill_status"),
        ("trailing_activated", "trailing_activated"),
    )
    rows: list[dict[str, Any]] = []
    for field, component in specs:
        for group, items in _group_records(records, field).items():
            row = _summary_row(component, group, items)
            row["method"] = "observational_grouping_not_counterfactual"
            row["interpretation_warning"] = (
                "This is not a rerun without the filter; it only summarizes realized MT5 log rows."
            )
            rows.append(row)
    rows.sort(key=lambda row: (row["dimension"], str(row["group"])))
    return rows


def _operational_warnings(records: list[dict[str, Any]]) -> dict[str, Any]:
    exit_counts_by_day = Counter(row["day"] for row in records if row.get("has_exit"))
    off_window = [
        row
        for row in records
        if row.get("has_exit")
        and row.get("fill_hour") != ""
        and int(row.get("fill_hour")) not in {16, 17, 18, 19, 20, 21}
    ]
    friday_with_exits = [
        row for row in records if row.get("is_friday") and row.get("has_exit")
    ]
    no_cleanup = [
        row for row in records if row.get("fill_status") == "order_no_fill_no_cleanup_event"
    ]
    return {
        "days_with_multiple_exits": len([day for day, count in exit_counts_by_day.items() if count > 1]),
        "extra_exits_on_multi_exit_days": sum(max(0, count - 1) for count in exit_counts_by_day.values()),
        "fills_outside_16_21_hour": len(off_window),
        "fills_outside_16_21_examples": [
            {"day": row.get("day"), "fill_hour": row.get("fill_hour"), "side": row.get("fill_side")}
            for row in off_window[:10]
        ],
        "friday_blocked_days_with_exit_events": len(friday_with_exits),
        "friday_exit_examples": [
            {"day": row.get("day"), "exit_reason": row.get("exit_reason"), "net_pnl": row.get("net_pnl")}
            for row in friday_with_exits[:10]
        ],
        "order_no_fill_without_cleanup_event_days": len(no_cleanup),
        "order_no_fill_without_cleanup_examples": [
            {"day": row.get("day"), "buy_order": row.get("buy_order_placed"), "sell_order": row.get("sell_order_placed")}
            for row in no_cleanup[:10]
        ],
    }


def _summary_row(dimension: str, group: Any, records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_records(records)
    return {"dimension": dimension, "group": "" if group == "" else group, **summary}


def _group_records(records: list[dict[str, Any]], field: str) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row.get(field, "")].append(row)
    return grouped


def _write_records_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    if not records:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "dimension",
        "group",
        "setup_days",
        "opportunity_days",
        "friday_blocked_days",
        "order_placed_days",
        "filled_trades",
        "no_fill_days",
        "net_pnl",
        "gross_pnl",
        "win_rate",
        "average_win",
        "average_loss",
        "payoff",
        "profit_factor",
        "expectancy",
        "max_drawdown_abs",
        "max_consecutive_losses",
        "average_losing_streak",
        "trailing_activated_trades",
        "tp_exits",
        "sl_or_trailing_exits",
        "session_close_exits",
        "method",
        "interpretation_warning",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _canonical_setup_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("event_type") in {"opportunity", "blocked_friday"}:
            grouped[_event_day(row)].append(row)
    return {day: _canonical_setup_row(day_rows) for day, day_rows in grouped.items()}


def _canonical_setup_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    opportunity_rows = [row for row in rows if row.get("event_type") == "opportunity"]
    if opportunity_rows:
        placed = [
            row
            for row in opportunity_rows
            if _is_true(row.get("buy_order_placed")) or _is_true(row.get("sell_order_placed"))
        ]
        if placed:
            return placed[-1]
        return opportunity_rows[0]
    return rows[0]


def _group_events_by_day(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[_event_day(row)][row.get("event_type", "")].append(row)
    return grouped


def _fill_status(
    setup: dict[str, Any],
    order_placed: bool,
    filled: bool,
    is_friday: bool,
    cleanup: list[dict[str, Any]],
) -> str:
    if setup.get("event_type") == "blocked_friday" or is_friday:
        return "blocked_friday"
    if filled:
        return "filled"
    if order_placed:
        return "order_no_fill" if cleanup else "order_no_fill_no_cleanup_event"
    return "no_order"


def _trend_bias(setup: dict[str, Any]) -> str:
    buy = _is_true(setup.get("buy_order_placed"))
    sell = _is_true(setup.get("sell_order_placed"))
    if buy and sell:
        return "both"
    if buy:
        return "buy_only"
    if sell:
        return "sell_only"
    if _is_true(setup.get("buy_allowed_by_trend")) and not _is_true(setup.get("sell_allowed_by_trend")):
        return "buy_allowed_no_order"
    if _is_true(setup.get("sell_allowed_by_trend")) and not _is_true(setup.get("buy_allowed_by_trend")):
        return "sell_allowed_no_order"
    return "none"


def _atr_regime(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 150:
        return "atr_lt_150"
    if value < 250:
        return "atr_150_250"
    if value < 400:
        return "atr_250_400"
    return "atr_ge_400"


def _pending_distance_regime(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 250:
        return "distance_lt_250"
    if value < 400:
        return "distance_250_400"
    if value < 600:
        return "distance_400_600"
    return "distance_ge_600"


def _read_parity_summary(parity_dir: Path | None) -> dict[str, Any]:
    if parity_dir is None:
        return {}
    parity_report_path = parity_dir / "parity_report.json"
    final_summary_path = parity_dir / "setup_fidelity_final_summary.json"
    atr_report_path = parity_dir / "atr_parity_report.json"
    summary: dict[str, Any] = {}
    if parity_report_path.exists():
        report = _read_json(parity_report_path)
        daily = report.get("daily_opportunity_parity", {})
        fill = report.get("fill_exit_coverage", {})
        summary.update(
            {
                "setup_day_match_pct": daily.get("setup_day_match_pct"),
                "setup_field_match_pct": daily.get("field_match_pct"),
                "fill_count_match_pct": fill.get("fill_count_match_pct"),
                "exit_count_match_pct": fill.get("exit_count_match_pct"),
            }
        )
    if final_summary_path.exists():
        final = _read_json(final_summary_path)
        summary["remaining_setup_divergence_days"] = final.get("remaining_setup_divergence_days")
        summary["remaining_setup_divergences_by_field"] = final.get("by_field")
    if atr_report_path.exists():
        atr = _read_json(atr_report_path)
        summary["atr_parity_pct"] = atr.get("match_pct")
    return summary


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _filter_events(rows: list[dict[str, Any]], start: str, end: str, symbol: str) -> list[dict[str, Any]]:
    start_dt = _parse_optional_datetime(start, is_end=False)
    end_dt = _parse_optional_datetime(end, is_end=True)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if symbol and row.get("symbol") and row.get("symbol") != symbol:
            continue
        timestamp = row.get("timestamp", "")
        if not timestamp:
            continue
        event_dt = _parse_timestamp(timestamp)
        if start_dt is not None and event_dt < start_dt:
            continue
        if end_dt is not None and event_dt > end_dt:
            continue
        filtered.append(row)
    return filtered


def _parse_optional_datetime(value: str, *, is_end: bool) -> datetime | None:
    if not value:
        return None
    value = value.strip().replace(".", "-")
    if len(value) == 10:
        suffix = "23:59:59" if is_end else "00:00:00"
        value = f"{value} {suffix}"
    return datetime.fromisoformat(value.replace("T", " "))


def _parse_timestamp(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.combine(date.min, datetime.min.time())
    text = text.replace(".", "-").replace("T", " ")
    if len(text) == 10:
        text = f"{text} 00:00:00"
    return datetime.fromisoformat(text)


def _event_day(row: dict[str, Any]) -> str:
    timestamp = str(row.get("timestamp", "")).strip()
    if timestamp:
        return timestamp.split(" ")[0].replace(".", "-")
    event_id = str(row.get("event_id", "")).strip()
    if event_id:
        return event_id.split("_")[0].replace(".", "-")
    return "unknown_date"


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator * 100.0, 4)


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _max_drawdown(pnl_values: list[float]) -> float:
    peak = 0.0
    equity = 0.0
    max_dd = 0.0
    for pnl in pnl_values:
        equity += pnl
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return abs(max_dd)


def _max_consecutive_losses(pnl_values: list[float]) -> int:
    best = 0
    current = 0
    for value in pnl_values:
        if value < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _average_losing_streak(pnl_values: list[float]) -> float | None:
    streaks: list[int] = []
    current = 0
    for value in pnl_values:
        if value < 0:
            current += 1
        elif current:
            streaks.append(current)
            current = 0
    if current:
        streaks.append(current)
    return mean(streaks) if streaks else None


def _compact_console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": report["benchmark"]["name"],
        "setup_benchmark_status": report["decision"]["setup_benchmark_status"],
        "economic_benchmark_status": report["decision"]["economic_benchmark_status"],
        "aggregate_metrics": report["aggregate_metrics"],
        "recommended_next_sprint": report["decision"]["recommended_next_sprint"],
    }


def _benchmark_markdown(benchmark: dict[str, Any]) -> str:
    setup_parity = benchmark["setup_parity"]
    limitations = "\n".join(f"- {item}" for item in benchmark["limitations"])
    frozen = "\n".join(f"- {item}" for item in benchmark["frozen_scope"])
    not_frozen = "\n".join(f"- {item}" for item in benchmark["not_frozen_scope"])
    sources = "\n".join(f"- `{source}`" for source in benchmark["source_of_truth"])
    return (
        f"# {benchmark['name']}\n\n"
        "Status: FROZEN for setup/filtro parity only.\n\n"
        "## Sources of Truth\n\n"
        f"{sources}\n\n"
        "## Scope\n\n"
        f"- Symbol: `{benchmark['symbol']}`\n"
        f"- Timeframe: {benchmark['timeframe']}\n"
        f"- Server time: {benchmark['server_time_dependency']}\n"
        f"- First event timestamp: `{benchmark['event_log_range']['first_timestamp']}`\n"
        f"- Last event timestamp: `{benchmark['event_log_range']['last_timestamp']}`\n"
        f"- First setup day: `{benchmark['event_log_range']['first_setup_day']}`\n"
        f"- Last setup day: `{benchmark['event_log_range']['last_setup_day']}`\n\n"
        "## Inputs Frozen\n\n"
        f"- Setup time: {benchmark['inputs']['setup_time']}\n"
        f"- Close hour: {benchmark['inputs']['close_hour']}\n"
        f"- Swing bars: {benchmark['inputs']['swing_bars']}\n"
        f"- Entry offset: {benchmark['inputs']['entry_offset_points']} points\n"
        f"- EMA filter: {benchmark['inputs']['ema_filter']}\n"
        f"- ATR filter: {benchmark['inputs']['atr_filter']}\n"
        f"- RSI gate: {benchmark['inputs']['rsi_gate']}\n"
        f"- Friday filter: {benchmark['inputs']['friday_filter']}\n\n"
        "## Parity Evidence\n\n"
        f"- Setup day match: {setup_parity.get('setup_day_match_pct')}%\n"
        f"- Setup field match: {setup_parity.get('setup_field_match_pct')}%\n"
        f"- ATR parity: {setup_parity.get('atr_parity_pct')}%\n"
        f"- Fill count match: {setup_parity.get('fill_count_match_pct')}%\n"
        f"- Remaining setup divergence days: {setup_parity.get('remaining_setup_divergence_days')}\n\n"
        "## Frozen Scope\n\n"
        f"{frozen}\n\n"
        "## Explicitly Not Frozen\n\n"
        f"{not_frozen}\n\n"
        "## Limitations\n\n"
        f"{limitations}\n"
    )


def _audit_summary_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate_metrics"]
    benchmark = report["benchmark"]
    warnings = report.get("operational_warnings", {})
    side_rows = report["breakdowns"].get("side_breakdown.csv", [])
    trailing_rows = report["breakdowns"].get("trailing_breakdown.csv", [])
    yearly_rows = report["breakdowns"].get("yearly_breakdown.csv", [])
    best_year = _best_group(yearly_rows)
    worst_year = _worst_group(yearly_rows)
    side_text = _render_key_rows(side_rows, "side")
    trailing_text = _render_key_rows(trailing_rows, "trailing")
    return (
        "# Azir Deep Audit - Frozen Setup Benchmark\n\n"
        "## Executive Summary\n\n"
        f"- Benchmark congelado: `{benchmark['name']}` para setup/filtros, no para economia Python.\n"
        f"- Rango MT5 auditado: `{benchmark['event_log_range']['first_timestamp']}` a "
        f"`{benchmark['event_log_range']['last_timestamp']}`.\n"
        f"- Setup day match: {benchmark['setup_parity'].get('setup_day_match_pct')}%; "
        f"setup field match: {benchmark['setup_parity'].get('setup_field_match_pct')}%; "
        f"ATR parity: {benchmark['setup_parity'].get('atr_parity_pct')}%.\n"
        f"- Trades cerrados MT5: {aggregate['filled_trades']}; net PnL MT5 log: {aggregate['net_pnl']}; "
        f"profit factor: {aggregate['profit_factor']}; expectancy: {aggregate['expectancy']}.\n"
        "- Estado: setup benchmark FROZEN; execution/economic benchmark NOT FROZEN.\n\n"
        "## De Donde Sale El Edge\n\n"
        "- El edge observable sale principalmente de una tasa de acierto muy alta en trades cerrados por MT5, "
        "pero con perdidas medias bastante mayores que las ganancias medias. Esto hace que el sistema dependa "
        "de mantener controlada la cola de perdidas y no solo de ganar muchas veces.\n"
        f"- Mejor anio por net PnL: {_format_group_row(best_year)}.\n"
        f"- Peor anio por net PnL: {_format_group_row(worst_year)}.\n"
        "- La lectura por filtros es observacional, no una ablacion contrafactual: todavia no sabemos cuanto "
        "ganaria/perderia Azir sin cada filtro hasta hacer reruns controlados.\n\n"
        "## Lados Y Trailing\n\n"
        f"{side_text}\n\n"
        f"{trailing_text}\n\n"
        "## Filtros\n\n"
        "- ATR: esta reproducido con paridad completa a nivel setup; los dias bloqueados por ATR son evidencia "
        "de control de regimen, pero no prueban por si solos valor economico contrafactual.\n"
        "- Trend filter: en la configuracion auditada actua como selector direccional principal; protege la "
        "estructura de setup y debe preservarse en el futuro Risk Engine.\n"
        "- RSI gate: queda resuelto a nivel de setup, pero casi siempre inactivo con la configuracion actual "
        "porque requiere ambos pendientes colocados.\n"
        "- Friday filter: bloquea oportunidades; no se debe cambiar sin una prueba contrafactual separada.\n\n"
        "## Limitaciones Vivas\n\n"
        "- PnL, trailing, MFE y MAE son evidencia real del log MT5, pero Python todavia no puede congelarlos "
        "sin tick replay o reglas explicitas de aproximacion.\n"
        "- La comparacion economica no debe usarse todavia como benchmark para PPO.\n"
        "- `filter_ablation_summary.csv` no es una ablacion real; es agrupacion observacional.\n\n"
        "## Riesgos Operativos Detectados\n\n"
        f"- Dias con mas de un exit registrado: {warnings.get('days_with_multiple_exits')} "
        f"(exits extra: {warnings.get('extra_exits_on_multi_exit_days')}).\n"
        f"- Fills fuera de horas 16-21 server time: {warnings.get('fills_outside_16_21_hour')}.\n"
        f"- Dias bloqueados por viernes pero con exit event en el log: "
        f"{warnings.get('friday_blocked_days_with_exit_events')}.\n"
        f"- Dias con orden/no fill sin cleanup explicito: "
        f"{warnings.get('order_no_fill_without_cleanup_event_days')}.\n"
        "- Esto refuerza que el siguiente trabajo debe auditar ciclo de vida de pendientes/cierre exacto antes "
        "de diseniar Risk Engine o PPO.\n\n"
        "## Decision\n\n"
        "- `baseline_azir_setup_frozen_v1` queda oficialmente congelada como benchmark de setup.\n"
        "- El siguiente sprint recomendado es auditoria economica adicional antes de Risk Engine: primero "
        "definir que economia MT5 aceptamos como referencia y como aproximarla en Python sin ticks.\n"
    )


def _render_key_rows(rows: list[dict[str, Any]], label: str) -> str:
    if not rows:
        return f"- No hay desglose disponible para {label}."
    lines = []
    for row in rows:
        if row.get("filled_trades", 0):
            lines.append(
                f"- {label} `{row.get('group')}`: trades={row.get('filled_trades')}, "
                f"net={row.get('net_pnl')}, PF={row.get('profit_factor')}, "
                f"expectancy={row.get('expectancy')}."
            )
    return "\n".join(lines) if lines else f"- No hay trades cerrados para {label}."


def _best_group(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    populated = [row for row in rows if row.get("filled_trades", 0)]
    return max(populated, key=lambda row: row.get("net_pnl", 0.0), default=None)


def _worst_group(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    populated = [row for row in rows if row.get("filled_trades", 0)]
    return min(populated, key=lambda row: row.get("net_pnl", 0.0), default=None)


def _format_group_row(row: dict[str, Any] | None) -> str:
    if not row:
        return "n/a"
    return (
        f"{row.get('group')} net={row.get('net_pnl')}, trades={row.get('filled_trades')}, "
        f"PF={row.get('profit_factor')}, DD_abs={row.get('max_drawdown_abs')}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
