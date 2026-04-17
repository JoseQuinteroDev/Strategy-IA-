"""Compare real MT5 evidence for Azir current versus the fractal candidate.

This runner consumes the observed Azir log and a separately exported
`fractal_candidate_event_log.csv`.  It is intentionally narrower than the
protected economic benchmark: the candidate exporter may contain setup-only
evidence, so this module refuses to infer real candidate economics when fills
and exits are not present.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .comparison import _first_daily_setup_rows
from .economic_audit import _read_raw_event_log, _round, _to_float, _write_csv, reconstruct_lifecycles, summarize_trades
from .event_log import AZIR_EVENT_COLUMNS
from .fractal_candidate_export import CANDIDATE_NAME, CANDIDATE_VARIANT
from .replica import AzirPythonReplica, AzirReplicaConfig, load_ohlcv_csv
from .setup_research import compute_metrics, extract_trade_rows


SPRINT_NAME = "compare_real_mt5_fractal_candidate_and_decide_promotion_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare real MT5 Azir current vs fractal candidate setup logs.")
    parser.add_argument("--current-log-path", required=True, help="Observed MT5 Azir current event log.")
    parser.add_argument("--candidate-log-path", required=True, help="Real MT5 fractal candidate event log.")
    parser.add_argument("--m5-input-path", default="", help="Optional XAUUSD M5 CSV for supplementary proxy context.")
    parser.add_argument("--protected-report-path", default="", help="Optional protected benchmark report JSON.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--current-symbol", default="XAUUSD-STD")
    parser.add_argument("--candidate-symbol", default="", help="Blank means auto-detect / no symbol filter.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_real_mt5_fractal_candidate_comparison(
        current_log_path=Path(args.current_log_path),
        candidate_log_path=Path(args.candidate_log_path),
        output_dir=Path(args.output_dir),
        m5_input_path=Path(args.m5_input_path) if args.m5_input_path else None,
        protected_report_path=Path(args.protected_report_path) if args.protected_report_path else None,
        current_symbol=args.current_symbol,
        candidate_symbol=args.candidate_symbol or None,
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_real_mt5_fractal_candidate_comparison(
    *,
    current_log_path: Path,
    candidate_log_path: Path,
    output_dir: Path,
    m5_input_path: Path | None = None,
    protected_report_path: Path | None = None,
    current_symbol: str = "XAUUSD-STD",
    candidate_symbol: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    current_rows = read_canonical_event_log(current_log_path, symbol_filter=current_symbol)
    candidate_rows_all = read_canonical_event_log(candidate_log_path, symbol_filter=None)
    inferred_candidate_symbol = candidate_symbol or infer_primary_symbol(candidate_rows_all)
    candidate_rows = filter_rows_by_symbol(candidate_rows_all, inferred_candidate_symbol)
    if not current_rows:
        raise ValueError(f"No current Azir MT5 rows found for symbol {current_symbol}.")
    if not candidate_rows:
        raise ValueError(f"No candidate MT5 rows found for symbol {inferred_candidate_symbol}.")

    current_schema = inspect_event_log(current_log_path, current_rows)
    candidate_schema = inspect_event_log(candidate_log_path, candidate_rows)
    day_rows = build_real_day_by_day_rows(current_rows, candidate_rows)
    diff_rows = [
        row
        for row in day_rows
        if row["missing_current"]
        or row["missing_candidate"]
        or row["event_type_changed"]
        or row["level_changed"]
        or row["order_intent_changed"]
        or row["filter_state_changed"]
    ]
    setup_summary = summarize_real_setup_comparison(day_rows)
    anomaly_summary = summarize_candidate_anomalies(candidate_rows, day_rows)
    current_economics = build_economic_summary(current_log_path, current_symbol)
    candidate_economics = build_candidate_economic_summary(candidate_log_path, inferred_candidate_symbol, candidate_rows)
    proxy_context = build_proxy_context(m5_input_path, current_symbol) if m5_input_path else {}
    protected_reference = load_protected_reference(protected_report_path)

    readiness = build_readiness(
        current_schema=current_schema,
        candidate_schema=candidate_schema,
        setup_summary=setup_summary,
        anomaly_summary=anomaly_summary,
        candidate_economics=candidate_economics,
        proxy_context=proxy_context,
        same_symbol=current_symbol == inferred_candidate_symbol,
    )
    report = {
        "sprint": SPRINT_NAME,
        "candidate_name": CANDIDATE_NAME,
        "candidate_variant": CANDIDATE_VARIANT,
        "sources": {
            "current_log_path": str(current_log_path),
            "candidate_log_path": str(candidate_log_path),
            "m5_input_path": str(m5_input_path) if m5_input_path else "",
            "protected_report_path": str(protected_report_path) if protected_report_path else "",
            "current_symbol": current_symbol,
            "candidate_symbol": inferred_candidate_symbol,
        },
        "current_log_schema": current_schema,
        "candidate_log_schema": candidate_schema,
        "setup_comparison": setup_summary,
        "strict_comparability": {
            "same_symbol": current_symbol == inferred_candidate_symbol,
            "current_symbol": current_symbol,
            "candidate_symbol": inferred_candidate_symbol,
        },
        "candidate_anomalies": anomaly_summary,
        "economics": {
            "current_azir_observed": current_economics,
            "candidate_fractal_observed": candidate_economics,
            "protected_reference": protected_reference,
            "supplementary_python_proxy_context": proxy_context,
        },
        "readiness": readiness,
        "limitations": build_limitations(candidate_economics),
    }

    _write_csv(day_rows, output_dir / "azir_vs_fractal_real_day_by_day.csv")
    _write_csv(diff_rows, output_dir / "azir_vs_fractal_real_diff.csv")
    (output_dir / "fractal_candidate_real_schema_check.json").write_text(
        json.dumps({"current": current_schema, "candidate": candidate_schema}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "fractal_candidate_real_mt5_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "fractal_candidate_real_readiness_assessment.md").write_text(
        readiness_markdown(report),
        encoding="utf-8",
    )
    (output_dir / "fractal_candidate_real_mt5_summary.md").write_text(
        summary_markdown(report),
        encoding="utf-8",
    )
    return report


def read_canonical_event_log(path: Path, *, symbol_filter: str | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Event log does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(AZIR_EVENT_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Event log missing canonical columns: {sorted(missing)}")
        rows = []
        for row in reader:
            if symbol_filter and row.get("symbol") and row.get("symbol") != symbol_filter:
                continue
            normalized = {column: row.get(column, "") for column in AZIR_EVENT_COLUMNS}
            normalized["_event_day"] = event_day(normalized)
            normalized["_timestamp_dt"] = parse_timestamp(normalized.get("timestamp"))
            rows.append(normalized)
    rows.sort(key=lambda item: item["_timestamp_dt"])
    return rows


def filter_rows_by_symbol(rows: list[dict[str, Any]], symbol: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("symbol") == symbol]


def infer_primary_symbol(rows: list[dict[str, Any]]) -> str:
    counts = Counter(row.get("symbol", "") for row in rows if row.get("symbol"))
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def inspect_event_log(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    columns = list(AZIR_EVENT_COLUMNS)
    event_counts = Counter(row.get("event_type", "") for row in rows)
    symbols = Counter(row.get("symbol", "") for row in rows)
    timestamps = [row["_timestamp_dt"] for row in rows if row.get("timestamp")]
    setup_days = _first_daily_setup_rows(rows)
    event_types = sorted(event_counts)
    has_fills = bool(event_counts.get("fill"))
    has_exits = bool(event_counts.get("exit"))
    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
        "columns": columns,
        "missing_canonical_columns": [],
        "extra_columns": [],
        "schema_compatible": True,
        "row_count": len(rows),
        "first_timestamp": format_dt(min(timestamps)) if timestamps else "",
        "last_timestamp": format_dt(max(timestamps)) if timestamps else "",
        "event_type_counts": dict(event_counts),
        "event_types": event_types,
        "symbols": dict(symbols),
        "setup_days": len(setup_days),
        "timeframes": dict(Counter(row.get("timeframe", "") for row in rows)),
        "has_fill_events": has_fills,
        "has_exit_events": has_exits,
        "has_economic_fields": has_fills and has_exits,
        "blank_event_id_rows": len([row for row in rows if not str(row.get("event_id", "")).strip()]),
        "duplicate_setup_rows": count_duplicate_setup_rows(rows),
        "usable_for_setup_parity": bool(rows) and bool(setup_days),
        "usable_for_fill_exit_parity": has_fills and has_exits,
        "usable_for_economic_audit": has_exits,
    }


def count_duplicate_setup_rows(rows: list[dict[str, Any]]) -> int:
    counts = Counter(event_day(row) for row in rows if row.get("event_type") in {"opportunity", "blocked_friday"})
    return sum(max(0, count - 1) for count in counts.values())


def build_real_day_by_day_rows(
    current_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current_daily = _first_daily_setup_rows(current_rows)
    candidate_daily = _first_daily_setup_rows(candidate_rows)
    current_fills = event_counts_by_day(current_rows, "fill")
    candidate_fills = event_counts_by_day(candidate_rows, "fill")
    current_exits = event_counts_by_day(current_rows, "exit")
    candidate_exits = event_counts_by_day(candidate_rows, "exit")
    rows: list[dict[str, Any]] = []
    fields = [
        "swing_high",
        "swing_low",
        "buy_entry",
        "sell_entry",
        "pending_distance_points",
        "ema20",
        "prev_close",
        "prev_close_vs_ema20_points",
        "atr",
        "atr_points",
        "rsi",
    ]
    for day in sorted(set(current_daily) | set(candidate_daily)):
        current = current_daily.get(day, {})
        candidate = candidate_daily.get(day, {})
        row: dict[str, Any] = {
            "day": day,
            "year": day[:4],
            "current_present": bool(current),
            "candidate_present": bool(candidate),
            "missing_current": not bool(current),
            "missing_candidate": not bool(candidate),
            "current_event_type": current.get("event_type", ""),
            "candidate_event_type": candidate.get("event_type", ""),
            "current_is_friday": current.get("is_friday", ""),
            "candidate_is_friday": candidate.get("is_friday", ""),
            "current_buy_order_placed": bool_text(current.get("buy_order_placed")),
            "current_sell_order_placed": bool_text(current.get("sell_order_placed")),
            "candidate_buy_order_placed": bool_text(candidate.get("buy_order_placed")),
            "candidate_sell_order_placed": bool_text(candidate.get("sell_order_placed")),
            "current_atr_filter_passed": bool_text(current.get("atr_filter_passed")),
            "candidate_atr_filter_passed": bool_text(candidate.get("atr_filter_passed")),
            "current_rsi_gate_required": bool_text(current.get("rsi_gate_required")),
            "candidate_rsi_gate_required": bool_text(candidate.get("rsi_gate_required")),
            "current_fill_events": current_fills.get(day, 0),
            "candidate_fill_events": candidate_fills.get(day, 0),
            "current_exit_events": current_exits.get(day, 0),
            "candidate_exit_events": candidate_exits.get(day, 0),
        }
        for field in fields:
            current_value = _to_float(current.get(field))
            candidate_value = _to_float(candidate.get(field))
            row[f"current_{field}"] = current_value if current_value is not None else current.get(field, "")
            row[f"candidate_{field}"] = candidate_value if candidate_value is not None else candidate.get(field, "")
            row[f"{field}_diff_candidate_minus_current"] = numeric_diff(candidate.get(field), current.get(field))
        row["event_type_changed"] = current.get("event_type", "") != candidate.get("event_type", "")
        row["level_changed"] = any(
            abs(_to_float(row.get(f"{field}_diff_candidate_minus_current")) or 0.0) > tolerance
            for field, tolerance in {
                "swing_high": 0.011,
                "swing_low": 0.011,
                "buy_entry": 0.011,
                "sell_entry": 0.011,
                "pending_distance_points": 1.0,
            }.items()
        )
        row["order_intent_changed"] = (
            row["current_buy_order_placed"] != row["candidate_buy_order_placed"]
            or row["current_sell_order_placed"] != row["candidate_sell_order_placed"]
        )
        row["filter_state_changed"] = (
            row["current_atr_filter_passed"] != row["candidate_atr_filter_passed"]
            or row["current_rsi_gate_required"] != row["candidate_rsi_gate_required"]
        )
        rows.append(row)
    return rows


def event_counts_by_day(rows: list[dict[str, Any]], event_type: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("event_type") == event_type:
            counts[event_day(row)] += 1
    return counts


def summarize_real_setup_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    current_days = [row for row in rows if row["current_present"]]
    candidate_days = [row for row in rows if row["candidate_present"]]
    common_days = [row for row in rows if row["current_present"] and row["candidate_present"]]
    return {
        "total_days_compared": len(rows),
        "current_setup_days": len(current_days),
        "candidate_setup_days": len(candidate_days),
        "common_setup_days": len(common_days),
        "candidate_coverage_vs_current_pct": pct(len(common_days), len(current_days)),
        "missing_candidate_days": len([row for row in rows if row["missing_candidate"]]),
        "candidate_extra_days": len([row for row in rows if row["missing_current"]]),
        "blocked_friday_current": len([row for row in rows if row.get("current_event_type") == "blocked_friday"]),
        "blocked_friday_candidate": len([row for row in rows if row.get("candidate_event_type") == "blocked_friday"]),
        "level_changed_days": len([row for row in common_days if row["level_changed"]]),
        "level_changed_pct": pct(len([row for row in common_days if row["level_changed"]]), len(common_days)),
        "order_intent_changed_days": len([row for row in common_days if row["order_intent_changed"]]),
        "filter_state_changed_days": len([row for row in common_days if row["filter_state_changed"]]),
        "candidate_fill_events": sum(int(row["candidate_fill_events"]) for row in rows),
        "candidate_exit_events": sum(int(row["candidate_exit_events"]) for row in rows),
        "current_fill_events": sum(int(row["current_fill_events"]) for row in rows),
        "current_exit_events": sum(int(row["current_exit_events"]) for row in rows),
    }


def summarize_candidate_anomalies(candidate_rows: list[dict[str, Any]], day_rows: list[dict[str, Any]]) -> dict[str, Any]:
    event_counts = Counter(row.get("event_type", "") for row in candidate_rows)
    return {
        "unsupported_event_types": sorted(set(event_counts) - {"opportunity", "blocked_friday", "fill", "exit", "trailing_modified", "opposite_pending_cancelled", "no_fill_close_cleanup"}),
        "duplicate_setup_rows": count_duplicate_setup_rows(candidate_rows),
        "blank_event_id_rows": len([row for row in candidate_rows if not str(row.get("event_id", "")).strip()]),
        "days_missing_candidate": [row["day"] for row in day_rows if row["missing_candidate"]][:50],
        "candidate_extra_days": [row["day"] for row in day_rows if row["missing_current"]][:50],
        "days_with_candidate_fills": len([row for row in day_rows if int(row["candidate_fill_events"]) > 0]),
        "days_with_candidate_exits": len([row for row in day_rows if int(row["candidate_exit_events"]) > 0]),
    }


def build_economic_summary(path: Path, symbol: str) -> dict[str, Any]:
    rows = _read_raw_event_log(path, symbol)
    reconstruction = reconstruct_lifecycles(rows, session_start_hour=16, session_end_hour=21, close_hour=22)
    return {
        "available": True,
        "source": "observed_mt5_event_log",
        "metrics": summarize_trades(reconstruction["trades"]),
        "trade_rows": len(reconstruction["trades"]),
    }


def build_candidate_economic_summary(path: Path, symbol: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    event_counts = Counter(row.get("event_type", "") for row in rows)
    if not event_counts.get("exit"):
        return {
            "available": False,
            "source": "real_candidate_mt5_log",
            "reason": "Candidate log contains no exit events, so real MT5 PnL/PF/expectancy cannot be computed.",
            "event_type_counts": dict(event_counts),
        }
    reconstruction_rows = _read_raw_event_log(path, symbol)
    reconstruction = reconstruct_lifecycles(reconstruction_rows, session_start_hour=16, session_end_hour=21, close_hour=22)
    return {
        "available": True,
        "source": "real_candidate_mt5_event_log",
        "metrics": summarize_trades(reconstruction["trades"]),
        "trade_rows": len(reconstruction["trades"]),
        "event_type_counts": dict(event_counts),
    }


def build_proxy_context(m5_input_path: Path | None, symbol: str) -> dict[str, Any]:
    if m5_input_path is None or not m5_input_path.exists():
        return {}
    bars = load_ohlcv_csv(m5_input_path)
    current_events = AzirPythonReplica(bars, AzirReplicaConfig(symbol=symbol)).run()
    candidate_events = AzirPythonReplica(
        bars,
        AzirReplicaConfig(symbol=symbol, swing_bars=10, swing_definition="fractal", fractal_side_bars=2),
    ).run()
    current_metrics = compute_metrics(extract_trade_rows(current_events))
    candidate_metrics = compute_metrics(extract_trade_rows(candidate_events))
    return {
        "m5_path": str(m5_input_path),
        "m5_rows": len(bars),
        "m5_range": {
            "start": bars[0].open_time.isoformat(sep=" ") if bars else "",
            "end": bars[-1].open_time.isoformat(sep=" ") if bars else "",
        },
        "current_python_proxy": current_metrics,
        "candidate_fractal_python_proxy": candidate_metrics,
        "delta_candidate_minus_current": metric_delta(candidate_metrics, current_metrics),
        "interpretation": "Supplementary proxy only; not MT5-real candidate economics.",
    }


def load_protected_reference(path: Path | None) -> dict[str, Any]:
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


def build_readiness(
    *,
    current_schema: dict[str, Any],
    candidate_schema: dict[str, Any],
    setup_summary: dict[str, Any],
    anomaly_summary: dict[str, Any],
    candidate_economics: dict[str, Any],
    proxy_context: dict[str, Any],
    same_symbol: bool,
) -> dict[str, Any]:
    proxy_delta = proxy_context.get("delta_candidate_minus_current", {})
    proxy_improves = (_to_float(proxy_delta.get("net_pnl")) or 0.0) > 0.0 and (_to_float(proxy_delta.get("profit_factor")) or 0.0) > 0.0
    setup_survives = (
        candidate_schema.get("schema_compatible")
        and candidate_schema.get("usable_for_setup_parity")
        and (_to_float(setup_summary.get("candidate_coverage_vs_current_pct")) or 0.0) >= 99.0
        and not anomaly_summary.get("unsupported_event_types")
    )
    economic_available = bool(candidate_economics.get("available"))
    promote_to_economic_candidate = bool(setup_survives and proxy_improves and same_symbol)
    if not same_symbol:
        next_sprint = "rerun_fractal_candidate_mt5_export_on_xauusd_std_v1"
        reason = (
            "The real candidate log is usable as setup evidence, but it was exported on a different symbol "
            "than current Azir; strict comparability blocks promotion."
        )
    elif promote_to_economic_candidate:
        next_sprint = "protected_economic_candidate_fractal_v1"
        reason = (
            "Real MT5 setup evidence is usable and the proxy edge remains visible; the next step is protected economic valuation, not replacement."
        )
    else:
        next_sprint = "fractal_candidate_operational_validation_followup_v1"
        reason = "The real MT5 candidate log is not yet strong enough to promote the candidate."
    return {
        "swing_10_fractal_survives_real_mt5_setup_evidence": bool(setup_survives),
        "proxy_improvement_still_visible": bool(proxy_improves),
        "strict_symbol_match": bool(same_symbol),
        "real_candidate_economic_evidence_available": economic_available,
        "may_become_baseline_azir_economic_candidate_fractal_v1": promote_to_economic_candidate,
        "may_freeze_protected_candidate_benchmark_now": False,
        "may_replace_official_azir_baseline_now": False,
        "ready_for_ppo": False,
        "recommended_next_sprint": next_sprint,
        "reason": reason,
        "important_guardrail": (
            "Candidate economics are not frozen because the real candidate log has no fills/exits."
            if not economic_available
            else "Candidate has economic events, but still needs protected benchmark audit before freeze."
        ),
        "current_schema_compatible": current_schema.get("schema_compatible"),
    }


def build_limitations(candidate_economics: dict[str, Any]) -> list[str]:
    limitations = [
        "Azir current remains the official baseline.",
        "No RL/PPO/Risk Engine changes are part of this sprint.",
        "A real setup log does not equal a protected economic benchmark.",
    ]
    if not candidate_economics.get("available"):
        limitations.append("The candidate log has no fills/exits, so real candidate PnL/PF/expectancy are unavailable.")
    return limitations


def summary_markdown(report: dict[str, Any]) -> str:
    schema = report["candidate_log_schema"]
    setup = report["setup_comparison"]
    economics = report["economics"]["candidate_fractal_observed"]
    proxy_delta = report["economics"].get("supplementary_python_proxy_context", {}).get("delta_candidate_minus_current", {})
    readiness = report["readiness"]
    comparability = report["strict_comparability"]
    return (
        "# Real MT5 Fractal Candidate Comparison\n\n"
        "## Executive Summary\n\n"
        f"- Candidate log rows: {schema['row_count']}.\n"
        f"- Candidate range: {schema['first_timestamp']} -> {schema['last_timestamp']}.\n"
        f"- Candidate event types: {schema['event_type_counts']}.\n"
        f"- Candidate symbol(s): {schema['symbols']}.\n"
        f"- Strict symbol match: {comparability['same_symbol']} "
        f"({comparability['current_symbol']} vs {comparability['candidate_symbol']}).\n"
        f"- Setup coverage vs current Azir MT5: {setup['candidate_coverage_vs_current_pct']}%.\n"
        f"- Level changed days: {setup['level_changed_days']} ({setup['level_changed_pct']}%).\n"
        f"- Order intent changed days: {setup['order_intent_changed_days']}.\n"
        f"- Candidate economic evidence available: {economics['available']}.\n"
        f"- Supplementary proxy delta net/PF/expectancy/DD: {proxy_delta.get('net_pnl', '')} / "
        f"{proxy_delta.get('profit_factor', '')} / {proxy_delta.get('expectancy', '')} / "
        f"{proxy_delta.get('max_drawdown', '')}.\n"
        f"- May become economic candidate: {readiness['may_become_baseline_azir_economic_candidate_fractal_v1']}.\n"
        f"- May freeze protected benchmark now: {readiness['may_freeze_protected_candidate_benchmark_now']}.\n\n"
        "## Decision\n\n"
        f"{readiness['reason']}\n\n"
        f"Guardrail: {readiness['important_guardrail']}\n"
    )


def readiness_markdown(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    return (
        "# Real MT5 Fractal Candidate Readiness\n\n"
        f"- Survives real MT5 setup evidence: {readiness['swing_10_fractal_survives_real_mt5_setup_evidence']}.\n"
        f"- Proxy improvement still visible: {readiness['proxy_improvement_still_visible']}.\n"
        f"- Strict symbol match: {readiness['strict_symbol_match']}.\n"
        f"- Real economic evidence available: {readiness['real_candidate_economic_evidence_available']}.\n"
        f"- May become `baseline_azir_economic_candidate_fractal_v1`: "
        f"{readiness['may_become_baseline_azir_economic_candidate_fractal_v1']}.\n"
        f"- May freeze protected candidate benchmark now: {readiness['may_freeze_protected_candidate_benchmark_now']}.\n"
        f"- May replace official Azir baseline now: {readiness['may_replace_official_azir_baseline_now']}.\n"
        f"- Ready for PPO: {readiness['ready_for_ppo']}.\n"
        f"- Recommended next sprint: `{readiness['recommended_next_sprint']}`.\n\n"
        "## Limitations\n\n"
        f"{limitations}\n"
    )


def metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    keys = ["closed_trades", "net_pnl", "profit_factor", "expectancy", "max_drawdown", "max_consecutive_losses"]
    result: dict[str, Any] = {}
    for key in keys:
        left_value = _to_float(left.get(key))
        right_value = _to_float(right.get(key))
        if left_value is None or right_value is None:
            continue
        result[key] = _round(left_value - right_value)
    return result


def numeric_diff(left: Any, right: Any) -> float | str:
    left_value = _to_float(left)
    right_value = _to_float(right)
    if left_value is None or right_value is None:
        return ""
    return _round(left_value - right_value)


def bool_text(value: Any) -> str:
    return "true" if str(value).strip().lower() == "true" else "false"


def event_day(row: dict[str, Any]) -> str:
    timestamp = str(row.get("timestamp", "")).strip()
    if timestamp:
        return timestamp.split(" ")[0].replace(".", "-")
    event_id = str(row.get("event_id", "")).strip()
    if event_id:
        return event_id.split("_")[0].replace(".", "-")
    return ""


def parse_timestamp(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.min
    return datetime.fromisoformat(text.replace(".", "-").replace("T", " "))


def format_dt(value: datetime) -> str:
    if value == datetime.min:
        return ""
    return value.strftime("%Y.%m.%d %H:%M:%S")


def pct(numerator: int, denominator: int) -> float:
    return _round((numerator / denominator) * 100.0) if denominator else 0.0


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    readiness = report["readiness"]
    return {
        "sprint": report["sprint"],
        "current_symbol": report["sources"]["current_symbol"],
        "candidate_symbol": report["sources"]["candidate_symbol"],
        "candidate_rows": report["candidate_log_schema"]["row_count"],
        "candidate_event_types": report["candidate_log_schema"]["event_type_counts"],
        "setup_coverage_vs_current_pct": report["setup_comparison"]["candidate_coverage_vs_current_pct"],
        "level_changed_days": report["setup_comparison"]["level_changed_days"],
        "order_intent_changed_days": report["setup_comparison"]["order_intent_changed_days"],
        "candidate_economics_available": report["economics"]["candidate_fractal_observed"]["available"],
        "may_become_economic_candidate": readiness["may_become_baseline_azir_economic_candidate_fractal_v1"],
        "recommended_next_sprint": readiness["recommended_next_sprint"],
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
