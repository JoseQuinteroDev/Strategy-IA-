"""MT5/export bridge for the Azir swing_10_fractal setup candidate.

This sprint deliberately keeps Azir as the frozen source of truth.  The module
can compare a real MT5 candidate event log when one is provided; otherwise it
generates a clearly-labelled Python-equivalent candidate export so the workflow
and artifact contract are reproducible before the user runs the MT5 script.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .best_setup_candidate import (
    CANDIDATE_NAME,
    CANDIDATE_VARIANT,
    candidate_definition,
    candidate_definition_markdown,
    comparison_breakdown_rows,
)
from .comparison import _first_daily_setup_rows
from .economic_audit import (
    _read_raw_event_log,
    _round,
    _to_float,
    _write_csv,
    reconstruct_lifecycles,
    summarize_trades,
)
from .event_log import AZIR_EVENT_COLUMNS, write_event_log
from .replica import AzirPythonReplica, AzirReplicaConfig, load_ohlcv_csv
from .setup_research import compute_metrics, extract_trade_rows


SPRINT_NAME = "mt5_candidate_fractal_event_log_export_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export/compare Azir swing_10_fractal candidate evidence.")
    parser.add_argument("--mt5-log-path", required=True, help="Observed MT5 Azir current event log.")
    parser.add_argument("--m5-input-path", required=True, help="XAUUSD M5 OHLCV CSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-log-path", default="", help="Optional real MT5 fractal candidate event log.")
    parser.add_argument("--protected-report-path", default="", help="Optional protected benchmark report JSON.")
    parser.add_argument("--symbol", default="XAUUSD-STD")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_fractal_candidate_export_validation(
        mt5_log_path=Path(args.mt5_log_path),
        m5_input_path=Path(args.m5_input_path),
        output_dir=Path(args.output_dir),
        candidate_log_path=Path(args.candidate_log_path) if args.candidate_log_path else None,
        protected_report_path=Path(args.protected_report_path) if args.protected_report_path else None,
        symbol=args.symbol,
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_fractal_candidate_export_validation(
    *,
    mt5_log_path: Path,
    m5_input_path: Path,
    output_dir: Path,
    candidate_log_path: Path | None = None,
    protected_report_path: Path | None = None,
    symbol: str = "XAUUSD-STD",
) -> dict[str, Any]:
    """Generate or consume a candidate event log and compare it to current Azir."""

    output_dir.mkdir(parents=True, exist_ok=True)
    mt5_rows = _read_raw_event_log(mt5_log_path, symbol)
    if not mt5_rows:
        raise ValueError(f"No current Azir MT5 rows found for symbol {symbol}.")

    bars = load_ohlcv_csv(m5_input_path)
    if not bars:
        raise ValueError("M5 input has no bars.")

    current_python_events = AzirPythonReplica(bars, AzirReplicaConfig(symbol=symbol)).run()
    candidate_python_events = _candidate_python_events(bars, symbol)
    candidate_source = _load_or_generate_candidate_log(
        candidate_log_path=candidate_log_path,
        candidate_python_events=candidate_python_events,
        output_dir=output_dir,
        symbol=symbol,
    )
    candidate_rows = candidate_source["rows"]

    day_rows = build_mt5_candidate_day_by_day_rows(
        mt5_rows=mt5_rows,
        current_python_events=current_python_events,
        candidate_rows=candidate_rows,
        candidate_python_events=candidate_python_events,
    )
    diff_rows = [row for row in day_rows if row["candidate_changes_levels_vs_mt5"] or row["candidate_changes_order_intent_vs_mt5"]]
    candidate_export_parity_rows = build_candidate_export_parity_rows(candidate_rows, candidate_python_events)

    current_python_trades = extract_trade_rows(current_python_events)
    candidate_python_trades = extract_trade_rows(candidate_python_events)
    mt5_reconstruction = reconstruct_lifecycles(mt5_rows, session_start_hour=16, session_end_hour=21, close_hour=22)
    mt5_observed_metrics = summarize_trades(mt5_reconstruction["trades"])
    current_python_metrics = compute_metrics(current_python_trades)
    candidate_python_metrics = compute_metrics(candidate_python_trades)
    protected_reference = _load_protected_reference(protected_report_path)

    setup_comparison = summarize_mt5_candidate_setup(day_rows)
    proxy_delta = _metric_delta(candidate_python_metrics, current_python_metrics)
    readiness = build_readiness(
        candidate_source=candidate_source,
        setup_comparison=setup_comparison,
        proxy_delta=proxy_delta,
        candidate_python_metrics=candidate_python_metrics,
        current_python_metrics=current_python_metrics,
    )

    report = {
        "sprint": SPRINT_NAME,
        "candidate_name": CANDIDATE_NAME,
        "candidate_variant": CANDIDATE_VARIANT,
        "sources": {
            "current_azir_mt5_log_path": str(mt5_log_path),
            "m5_input_path": str(m5_input_path),
            "candidate_log_path": str(candidate_log_path) if candidate_log_path else "",
            "protected_report_path": str(protected_report_path) if protected_report_path else "",
        },
        "candidate_evidence": {
            "source_type": candidate_source["source_type"],
            "is_real_mt5_candidate_log": candidate_source["is_real_mt5_candidate_log"],
            "event_rows": len(candidate_rows),
            "setup_rows": len(_first_daily_setup_rows(candidate_rows)),
            "artifact_path": str(output_dir / "fractal_candidate_event_log.csv"),
        },
        "m5_range": {
            "start": bars[0].open_time.isoformat(sep=" "),
            "end": bars[-1].open_time.isoformat(sep=" "),
            "rows": len(bars),
        },
        "definition": candidate_definition(),
        "setup_comparison": setup_comparison,
        "proxy_metrics": {
            "mt5_observed_current_azir": mt5_observed_metrics,
            "python_current_azir_proxy": current_python_metrics,
            "python_swing_10_fractal_proxy": candidate_python_metrics,
            "delta_fractal_proxy_vs_current_proxy": proxy_delta,
            "protected_reference": protected_reference,
        },
        "candidate_export_parity_vs_python": summarize_candidate_export_parity(candidate_export_parity_rows),
        "readiness": readiness,
        "limitations": build_limitations(candidate_source),
    }

    yearly_rows = comparison_breakdown_rows(
        mt5_trades=mt5_reconstruction["trades"],
        current_trades=current_python_trades,
        candidate_trades=candidate_python_trades,
        key="year",
    )
    side_rows = comparison_breakdown_rows(
        mt5_trades=mt5_reconstruction["trades"],
        current_trades=current_python_trades,
        candidate_trades=candidate_python_trades,
        key="side",
    )

    _write_csv(day_rows, output_dir / "azir_vs_fractal_mt5_day_by_day.csv")
    _write_csv(diff_rows, output_dir / "azir_vs_fractal_mt5_diff.csv")
    _write_csv(candidate_export_parity_rows, output_dir / "candidate_export_vs_python_parity.csv")
    _write_csv(yearly_rows, output_dir / "azir_vs_fractal_mt5_yearly_comparison.csv")
    _write_csv(side_rows, output_dir / "azir_vs_fractal_mt5_side_comparison.csv")
    (output_dir / "fractal_candidate_mt5_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "fractal_candidate_definition.md").write_text(
        fractal_candidate_definition_markdown(),
        encoding="utf-8",
    )
    (output_dir / "fractal_candidate_readiness_assessment.md").write_text(
        readiness_markdown(report),
        encoding="utf-8",
    )
    (output_dir / "fractal_candidate_mt5_summary.md").write_text(
        summary_markdown(report),
        encoding="utf-8",
    )
    return report


def _candidate_python_events(bars: list[Any], symbol: str) -> list[dict[str, Any]]:
    return AzirPythonReplica(
        bars,
        AzirReplicaConfig(
            symbol=symbol,
            swing_bars=10,
            swing_definition="fractal",
            fractal_side_bars=2,
        ),
    ).run()


def _load_or_generate_candidate_log(
    *,
    candidate_log_path: Path | None,
    candidate_python_events: list[dict[str, Any]],
    output_dir: Path,
    symbol: str,
) -> dict[str, Any]:
    if candidate_log_path is not None and candidate_log_path.exists():
        rows = _read_raw_event_log(candidate_log_path, symbol)
        write_event_log(_canonical_event_rows(rows), output_dir / "fractal_candidate_event_log.csv")
        return {
            "rows": rows,
            "source_type": "real_mt5_candidate_event_log",
            "is_real_mt5_candidate_log": True,
            "input_path": str(candidate_log_path),
        }

    write_event_log(_canonical_event_rows(candidate_python_events), output_dir / "fractal_candidate_event_log.csv")
    return {
        "rows": candidate_python_events,
        "source_type": "python_equivalent_export_pending_mt5_run",
        "is_real_mt5_candidate_log": False,
        "input_path": "",
    }


def build_mt5_candidate_day_by_day_rows(
    *,
    mt5_rows: list[dict[str, Any]],
    current_python_events: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    candidate_python_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mt5_daily = _first_daily_setup_rows(mt5_rows)
    current_python_daily = _first_daily_setup_rows(current_python_events)
    candidate_daily = _first_daily_setup_rows(candidate_rows)
    candidate_python_daily = _first_daily_setup_rows(candidate_python_events)
    candidate_proxy_outcomes = _python_outcomes_by_day(candidate_python_events)
    current_proxy_outcomes = _python_outcomes_by_day(current_python_events)
    days = sorted(set(mt5_daily) | set(candidate_daily))
    rows: list[dict[str, Any]] = []
    for day in days:
        mt5 = mt5_daily.get(day, {})
        current_python = current_python_daily.get(day, {})
        candidate = candidate_daily.get(day, {})
        candidate_python = candidate_python_daily.get(day, {})
        current_outcome = current_proxy_outcomes.get(day, {})
        candidate_outcome = candidate_proxy_outcomes.get(day, {})
        row = {
            "day": day,
            "year": day[:4],
            "mt5_current_event_type": mt5.get("event_type", ""),
            "candidate_event_type": candidate.get("event_type", ""),
            "mt5_current_buy_order_placed": _is_true(mt5.get("buy_order_placed")),
            "mt5_current_sell_order_placed": _is_true(mt5.get("sell_order_placed")),
            "candidate_buy_order_placed": _is_true(candidate.get("buy_order_placed")),
            "candidate_sell_order_placed": _is_true(candidate.get("sell_order_placed")),
            "python_current_buy_order_placed": _is_true(current_python.get("buy_order_placed")),
            "python_current_sell_order_placed": _is_true(current_python.get("sell_order_placed")),
            "mt5_current_swing_high": _to_float(mt5.get("swing_high")),
            "mt5_current_swing_low": _to_float(mt5.get("swing_low")),
            "candidate_swing_high": _to_float(candidate.get("swing_high")),
            "candidate_swing_low": _to_float(candidate.get("swing_low")),
            "candidate_python_swing_high": _to_float(candidate_python.get("swing_high")),
            "candidate_python_swing_low": _to_float(candidate_python.get("swing_low")),
            "mt5_current_buy_entry": _to_float(mt5.get("buy_entry")),
            "mt5_current_sell_entry": _to_float(mt5.get("sell_entry")),
            "candidate_buy_entry": _to_float(candidate.get("buy_entry")),
            "candidate_sell_entry": _to_float(candidate.get("sell_entry")),
            "mt5_current_pending_distance_points": _to_float(mt5.get("pending_distance_points")),
            "candidate_pending_distance_points": _to_float(candidate.get("pending_distance_points")),
            "mt5_current_atr_filter_passed": _is_true(mt5.get("atr_filter_passed")),
            "candidate_atr_filter_passed": _is_true(candidate.get("atr_filter_passed")),
            "mt5_current_rsi_gate_required": _is_true(mt5.get("rsi_gate_required")),
            "candidate_rsi_gate_required": _is_true(candidate.get("rsi_gate_required")),
            "current_proxy_fill_side": current_outcome.get("fill_side", ""),
            "candidate_proxy_fill_side": candidate_outcome.get("fill_side", ""),
            "current_proxy_net_pnl": current_outcome.get("net_pnl", ""),
            "candidate_proxy_net_pnl": candidate_outcome.get("net_pnl", ""),
        }
        row["candidate_vs_mt5_buy_entry_diff"] = _diff(candidate.get("buy_entry"), mt5.get("buy_entry"))
        row["candidate_vs_mt5_sell_entry_diff"] = _diff(candidate.get("sell_entry"), mt5.get("sell_entry"))
        row["candidate_vs_python_candidate_buy_entry_diff"] = _diff(
            candidate.get("buy_entry"), candidate_python.get("buy_entry")
        )
        row["candidate_vs_python_candidate_sell_entry_diff"] = _diff(
            candidate.get("sell_entry"), candidate_python.get("sell_entry")
        )
        row["candidate_changes_levels_vs_mt5"] = _changed_any(
            row,
            ["candidate_vs_mt5_buy_entry_diff", "candidate_vs_mt5_sell_entry_diff"],
            tolerance=0.011,
        )
        row["candidate_changes_order_intent_vs_mt5"] = (
            row["candidate_buy_order_placed"] != row["mt5_current_buy_order_placed"]
            or row["candidate_sell_order_placed"] != row["mt5_current_sell_order_placed"]
        )
        row["candidate_export_matches_python_levels"] = not _changed_any(
            row,
            [
                "candidate_vs_python_candidate_buy_entry_diff",
                "candidate_vs_python_candidate_sell_entry_diff",
            ],
            tolerance=0.011,
        )
        row["candidate_proxy_delta_pnl_vs_current_proxy"] = _round(
            (_to_float(row["candidate_proxy_net_pnl"]) or 0.0)
            - (_to_float(row["current_proxy_net_pnl"]) or 0.0)
        )
        rows.append(row)
    return rows


def build_candidate_export_parity_rows(
    candidate_rows: list[dict[str, Any]],
    candidate_python_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_daily = _first_daily_setup_rows(candidate_rows)
    python_daily = _first_daily_setup_rows(candidate_python_events)
    fields = [
        "event_type",
        "swing_high",
        "swing_low",
        "buy_entry",
        "sell_entry",
        "pending_distance_points",
        "atr_filter_passed",
        "rsi_gate_required",
        "buy_order_placed",
        "sell_order_placed",
    ]
    rows: list[dict[str, Any]] = []
    for day in sorted(set(candidate_daily) | set(python_daily)):
        candidate = candidate_daily.get(day, {})
        python = python_daily.get(day, {})
        row: dict[str, Any] = {
            "day": day,
            "candidate_present": bool(candidate),
            "python_present": bool(python),
        }
        matches = 0
        compared = 0
        for field in fields:
            candidate_value = candidate.get(field, "")
            python_value = python.get(field, "")
            row[f"candidate_{field}"] = candidate_value
            row[f"python_{field}"] = python_value
            match = _field_matches(field, candidate_value, python_value)
            row[f"{field}_match"] = match
            if candidate or python:
                compared += 1
                matches += 1 if match else 0
        row["field_match_pct"] = _round((matches / compared) * 100.0) if compared else 0.0
        rows.append(row)
    return rows


def _canonical_event_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{column: row.get(column, "") for column in AZIR_EVENT_COLUMNS} for row in rows]


def summarize_mt5_candidate_setup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mt5_days = [row for row in rows if row["mt5_current_event_type"]]
    candidate_days = [row for row in rows if row["candidate_event_type"]]
    common_days = [row for row in rows if row["mt5_current_event_type"] and row["candidate_event_type"]]
    changed_levels = [row for row in rows if row["candidate_changes_levels_vs_mt5"]]
    changed_intent = [row for row in rows if row["candidate_changes_order_intent_vs_mt5"]]
    positive_proxy_delta = [
        row for row in rows if (_to_float(row.get("candidate_proxy_delta_pnl_vs_current_proxy")) or 0.0) > 0.0
    ]
    negative_proxy_delta = [
        row for row in rows if (_to_float(row.get("candidate_proxy_delta_pnl_vs_current_proxy")) or 0.0) < 0.0
    ]
    export_match = [row for row in common_days if row["candidate_export_matches_python_levels"]]
    return {
        "total_days_compared": len(rows),
        "mt5_current_setup_days": len(mt5_days),
        "candidate_setup_days": len(candidate_days),
        "common_setup_days": len(common_days),
        "candidate_setup_coverage_vs_mt5_pct": _pct(len(common_days), len(mt5_days)),
        "candidate_changed_level_days_vs_mt5": len(changed_levels),
        "candidate_changed_level_pct_vs_mt5": _pct(len(changed_levels), len(common_days)),
        "candidate_changed_order_intent_days_vs_mt5": len(changed_intent),
        "candidate_positive_proxy_delta_days": len(positive_proxy_delta),
        "candidate_negative_proxy_delta_days": len(negative_proxy_delta),
        "candidate_export_level_match_vs_python_pct": _pct(len(export_match), len(common_days)),
    }


def summarize_candidate_export_parity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"daily_rows": 0, "average_field_match_pct": 0.0}
    return {
        "daily_rows": len(rows),
        "average_field_match_pct": _round(
            sum(_to_float(row.get("field_match_pct")) or 0.0 for row in rows) / len(rows)
        ),
        "perfect_days": len([row for row in rows if (_to_float(row.get("field_match_pct")) or 0.0) == 100.0]),
    }


def build_readiness(
    *,
    candidate_source: dict[str, Any],
    setup_comparison: dict[str, Any],
    proxy_delta: dict[str, Any],
    candidate_python_metrics: dict[str, Any],
    current_python_metrics: dict[str, Any],
) -> dict[str, Any]:
    has_real_candidate_log = bool(candidate_source["is_real_mt5_candidate_log"])
    proxy_improves = (
        (_to_float(proxy_delta.get("net_pnl")) or 0.0) > 0.0
        and (_to_float(candidate_python_metrics.get("profit_factor")) or 0.0)
        > (_to_float(current_python_metrics.get("profit_factor")) or 0.0)
        and (_to_float(candidate_python_metrics.get("expectancy")) or 0.0) > 0.0
    )
    setup_coverage_ok = (_to_float(setup_comparison.get("candidate_setup_coverage_vs_mt5_pct")) or 0.0) >= 99.0
    can_be_economic_candidate = has_real_candidate_log and proxy_improves and setup_coverage_ok
    return {
        "candidate_name": CANDIDATE_NAME,
        "proxy_improvement_still_visible": proxy_improves,
        "real_candidate_mt5_log_available": has_real_candidate_log,
        "setup_coverage_ok": setup_coverage_ok,
        "may_become_economic_candidate": can_be_economic_candidate,
        "may_replace_azir_now": False,
        "ready_for_ppo": False,
        "recommended_next_sprint": (
            "protected_economic_candidate_fractal_v1"
            if can_be_economic_candidate
            else "run_mt5_fractal_candidate_export_and_compare_real_log_v1"
        ),
        "reason": (
            "A real candidate MT5 log exists and the proxy improvement remains visible; build protected candidate economics next."
            if can_be_economic_candidate
            else "The candidate is export-ready, but it still lacks enough real MT5 candidate evidence for economic promotion."
        ),
    }


def build_limitations(candidate_source: dict[str, Any]) -> list[str]:
    limitations = [
        "Azir current remains the official frozen baseline.",
        "This sprint does not change Azir.mq5, trailing, Risk Engine, RL, or PPO.",
        "Economic candidate promotion requires a protected replay/audit after candidate MT5 evidence exists.",
    ]
    if not candidate_source["is_real_mt5_candidate_log"]:
        limitations.insert(
            0,
            "No real MT5 candidate log was provided; fractal_candidate_event_log.csv is a Python-equivalent export contract.",
        )
    return limitations


def fractal_candidate_definition_markdown() -> str:
    return (
        candidate_definition_markdown()
        + "\n## MT5 Export Contract\n\n"
        "- The auxiliary MT5 script is an exporter, not a replacement EA.\n"
        "- It writes canonical Azir event-log columns so Python can compare it with `todos_los_ticks.csv`.\n"
        "- If no confirmed 2-left/2-right pivot exists in the last 10 closed M5 bars, it falls back to the rolling high/low.\n"
        "- Fill, trailing and PnL remain out of scope unless the candidate is run as a real EA/export with lifecycle evidence.\n"
    )


def summary_markdown(report: dict[str, Any]) -> str:
    evidence = report["candidate_evidence"]
    setup = report["setup_comparison"]
    proxy = report["proxy_metrics"]
    delta = proxy["delta_fractal_proxy_vs_current_proxy"]
    readiness = report["readiness"]
    return (
        "# Fractal Candidate MT5 Export V1\n\n"
        "## Executive Summary\n\n"
        f"- Candidate: `{report['candidate_name']}` / `{report['candidate_variant']}`.\n"
        f"- Candidate evidence source: `{evidence['source_type']}`.\n"
        f"- Real MT5 candidate log available: {evidence['is_real_mt5_candidate_log']}.\n"
        f"- Current Azir MT5 setup days: {setup['mt5_current_setup_days']}.\n"
        f"- Candidate setup days: {setup['candidate_setup_days']}.\n"
        f"- Candidate setup coverage vs current MT5: {setup['candidate_setup_coverage_vs_mt5_pct']}%.\n"
        f"- Candidate changed levels vs current MT5 on {setup['candidate_changed_level_days_vs_mt5']} days "
        f"({setup['candidate_changed_level_pct_vs_mt5']}%).\n"
        f"- Candidate changed order intent vs current MT5 on {setup['candidate_changed_order_intent_days_vs_mt5']} days.\n"
        f"- Proxy delta net/PF/expectancy/DD: {delta.get('net_pnl', '')} / "
        f"{delta.get('profit_factor', '')} / {delta.get('expectancy', '')} / "
        f"{delta.get('max_drawdown', '')}.\n"
        f"- May become economic candidate now: {readiness['may_become_economic_candidate']}.\n\n"
        "## Decision\n\n"
        f"{readiness['reason']}\n"
    )


def readiness_markdown(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    return (
        "# Fractal Candidate Readiness Assessment\n\n"
        f"- Candidate: `{readiness['candidate_name']}`.\n"
        f"- Proxy improvement still visible: {readiness['proxy_improvement_still_visible']}.\n"
        f"- Real candidate MT5 log available: {readiness['real_candidate_mt5_log_available']}.\n"
        f"- Setup coverage OK: {readiness['setup_coverage_ok']}.\n"
        f"- May become economic candidate: {readiness['may_become_economic_candidate']}.\n"
        f"- May replace Azir now: {readiness['may_replace_azir_now']}.\n"
        f"- Ready for PPO: {readiness['ready_for_ppo']}.\n"
        f"- Recommended next sprint: `{readiness['recommended_next_sprint']}`.\n\n"
        "## Limitations\n\n"
        f"{limitations}\n"
    )


def _load_protected_reference(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    metrics = report.get("metrics", {}).get("azir_with_risk_engine_v1_forced_closes_revalued", {})
    return {
        "benchmark_name": report.get("benchmark_name", "baseline_azir_protected_economic_v1"),
        "closed_trades": metrics.get("closed_trades"),
        "net_pnl": metrics.get("net_pnl"),
        "profit_factor": metrics.get("profit_factor"),
        "expectancy": metrics.get("expectancy"),
        "max_drawdown": metrics.get("max_drawdown_abs"),
        "max_consecutive_losses": metrics.get("max_consecutive_losses"),
    }


def _python_outcomes_by_day(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    outcomes: dict[str, dict[str, Any]] = {}
    fills: dict[str, dict[str, Any]] = {}
    for row in events:
        day = _event_day(row)
        if row.get("event_type") == "fill":
            fills[day] = row
        elif row.get("event_type") == "exit":
            outcomes[day] = {
                "fill_side": row.get("fill_side", fills.get(day, {}).get("fill_side", "")),
                "net_pnl": _to_float(row.get("net_pnl")) or 0.0,
                "exit_reason": row.get("exit_reason", ""),
            }
    return outcomes


def _metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    fields = ["closed_trades", "net_pnl", "profit_factor", "expectancy", "max_drawdown", "max_consecutive_losses"]
    result: dict[str, Any] = {}
    for field in fields:
        left_value = _to_float(left.get(field))
        right_value = _to_float(right.get(field))
        if left_value is None or right_value is None:
            continue
        result[field] = _round(left_value - right_value)
    return result


def _event_day(row: dict[str, Any]) -> str:
    timestamp = str(row.get("timestamp", "")).strip()
    if timestamp:
        return timestamp.split(" ")[0].replace(".", "-")
    event_id = str(row.get("event_id", "")).strip()
    if event_id:
        return event_id.split("_")[0].replace(".", "-")
    return ""


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _diff(left: Any, right: Any) -> float | str:
    left_value = _to_float(left)
    right_value = _to_float(right)
    if left_value is None or right_value is None:
        return ""
    return _round(left_value - right_value)


def _changed_any(row: dict[str, Any], fields: list[str], *, tolerance: float) -> bool:
    for field in fields:
        value = _to_float(row.get(field))
        if value is not None and abs(value) > tolerance:
            return True
    return False


def _field_matches(field: str, left: Any, right: Any) -> bool:
    numeric_tolerance = {
        "swing_high": 0.011,
        "swing_low": 0.011,
        "buy_entry": 0.011,
        "sell_entry": 0.011,
        "pending_distance_points": 1.0,
    }
    if field in numeric_tolerance:
        left_value = _to_float(left)
        right_value = _to_float(right)
        if left_value is None and right_value is None:
            return True
        if left_value is None or right_value is None:
            return False
        return abs(left_value - right_value) <= numeric_tolerance[field]
    return str(left).strip().lower() == str(right).strip().lower()


def _pct(numerator: int, denominator: int) -> float:
    return _round((numerator / denominator) * 100.0) if denominator else 0.0


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    readiness = report["readiness"]
    setup = report["setup_comparison"]
    delta = report["proxy_metrics"]["delta_fractal_proxy_vs_current_proxy"]
    return {
        "sprint": report["sprint"],
        "candidate": report["candidate_variant"],
        "candidate_evidence_source": report["candidate_evidence"]["source_type"],
        "real_mt5_candidate_log": report["candidate_evidence"]["is_real_mt5_candidate_log"],
        "setup_coverage_vs_mt5_pct": setup["candidate_setup_coverage_vs_mt5_pct"],
        "changed_level_days_vs_mt5": setup["candidate_changed_level_days_vs_mt5"],
        "delta_proxy_net_pnl": delta.get("net_pnl"),
        "may_become_economic_candidate": readiness["may_become_economic_candidate"],
        "recommended_next_sprint": readiness["recommended_next_sprint"],
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
