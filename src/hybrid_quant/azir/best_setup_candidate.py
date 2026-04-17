"""Validate the best Azir setup candidate against current Azir evidence.

The candidate is not promoted here. This sprint compares the observed MT5 Azir
setup log with the current Python replica and the `swing_10_fractal` candidate,
then writes a readiness assessment for a controlled MT5 export/backtest.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .comparison import _first_daily_setup_rows
from .economic_audit import _read_raw_event_log, _round, _to_float, _write_csv, reconstruct_lifecycles, summarize_trades
from .replica import AzirPythonReplica, AzirReplicaConfig, load_ohlcv_csv
from .setup_research import compute_metrics, extract_trade_rows, max_drawdown, max_consecutive_losses


CANDIDATE_NAME = "baseline_azir_setup_candidate_fractal_v1"
CANDIDATE_VARIANT = "swing_10_fractal"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Azir current setup with swing_10_fractal candidate.")
    parser.add_argument("--mt5-log-path", required=True)
    parser.add_argument("--m5-input-path", required=True)
    parser.add_argument("--protected-report-path", required=True)
    parser.add_argument("--setup-research-report-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbol", default="XAUUSD-STD")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_best_setup_candidate_validation(
        mt5_log_path=Path(args.mt5_log_path),
        m5_input_path=Path(args.m5_input_path),
        protected_report_path=Path(args.protected_report_path),
        output_dir=Path(args.output_dir),
        setup_research_report_path=Path(args.setup_research_report_path)
        if args.setup_research_report_path
        else None,
        symbol=args.symbol,
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_best_setup_candidate_validation(
    *,
    mt5_log_path: Path,
    m5_input_path: Path,
    protected_report_path: Path,
    output_dir: Path,
    setup_research_report_path: Path | None = None,
    symbol: str = "XAUUSD-STD",
) -> dict[str, Any]:
    mt5_rows = _read_raw_event_log(mt5_log_path, symbol)
    if not mt5_rows:
        raise ValueError(f"No MT5 Azir rows found for symbol {symbol}.")

    bars = load_ohlcv_csv(m5_input_path)
    current_events = AzirPythonReplica(bars, AzirReplicaConfig(symbol=symbol)).run()
    candidate_events = AzirPythonReplica(
        bars,
        AzirReplicaConfig(
            symbol=symbol,
            swing_bars=10,
            swing_definition="fractal",
            fractal_side_bars=2,
        ),
    ).run()

    protected_reference = _load_protected_reference(protected_report_path)
    setup_research_reference = _load_setup_research_reference(setup_research_report_path)

    day_rows = build_day_by_day_rows(mt5_rows, current_events, candidate_events)
    diff_rows = [row for row in day_rows if row["candidate_changes_setup"] or row["candidate_changes_order_intent"]]
    current_trades = extract_trade_rows(current_events)
    candidate_trades = extract_trade_rows(candidate_events)
    current_metrics = compute_metrics(current_trades)
    candidate_metrics = compute_metrics(candidate_trades)
    mt5_reconstruction = reconstruct_lifecycles(mt5_rows, session_start_hour=16, session_end_hour=21, close_hour=22)
    mt5_observed_metrics = summarize_trades(mt5_reconstruction["trades"])
    yearly_rows = comparison_breakdown_rows(
        mt5_trades=mt5_reconstruction["trades"],
        current_trades=current_trades,
        candidate_trades=candidate_trades,
        key="year",
    )
    side_rows = comparison_breakdown_rows(
        mt5_trades=mt5_reconstruction["trades"],
        current_trades=current_trades,
        candidate_trades=candidate_trades,
        key="side",
    )

    setup_comparison = summarize_setup_comparison(day_rows)
    proxy_comparison = {
        "current_python_proxy": current_metrics,
        "candidate_fractal_python_proxy": candidate_metrics,
        "delta_candidate_vs_current": _metric_delta(candidate_metrics, current_metrics),
        "mt5_observed_reference": mt5_observed_metrics,
        "protected_reference": protected_reference,
    }
    readiness = build_readiness_assessment(
        setup_comparison=setup_comparison,
        current_metrics=current_metrics,
        candidate_metrics=candidate_metrics,
        protected_reference=protected_reference,
    )
    report = {
        "sprint": "mt5_forward_parity_export_for_best_setup_candidate_v1",
        "candidate_name": CANDIDATE_NAME,
        "candidate_variant": CANDIDATE_VARIANT,
        "sources": {
            "mt5_log_path": str(mt5_log_path),
            "m5_input_path": str(m5_input_path),
            "protected_report_path": str(protected_report_path),
            "setup_research_report_path": str(setup_research_report_path) if setup_research_report_path else "",
        },
        "m5_range": {
            "start": bars[0].open_time.isoformat(sep=" ") if bars else "",
            "end": bars[-1].open_time.isoformat(sep=" ") if bars else "",
            "rows": len(bars),
        },
        "definition": candidate_definition(),
        "setup_comparison": setup_comparison,
        "proxy_comparison": proxy_comparison,
        "setup_research_reference": setup_research_reference,
        "readiness": readiness,
        "limitations": [
            "No MT5 event log exists yet for the fractal candidate; candidate fills and PnL are still Python replica proxy.",
            "The MT5 current Azir side of the comparison is real empirical evidence from todos_los_ticks.csv.",
            "The candidate must be implemented/exported in MT5 and compared again before any benchmark promotion.",
            "This sprint does not change Azir.mq5, trailing, Risk Engine, RL, or PPO.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(day_rows, output_dir / "azir_vs_fractal_day_by_day.csv")
    _write_csv(diff_rows, output_dir / "azir_vs_fractal_setup_diff.csv")
    _write_csv(yearly_rows, output_dir / "azir_vs_fractal_yearly_comparison.csv")
    _write_csv(side_rows, output_dir / "azir_vs_fractal_side_comparison.csv")
    (output_dir / "best_setup_candidate_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "best_setup_candidate_definition.md").write_text(
        candidate_definition_markdown(),
        encoding="utf-8",
    )
    (output_dir / "candidate_readiness_assessment.md").write_text(
        readiness_markdown(report),
        encoding="utf-8",
    )
    (output_dir / "best_setup_candidate_summary.md").write_text(
        summary_markdown(report),
        encoding="utf-8",
    )
    return report


def build_day_by_day_rows(
    mt5_rows: list[dict[str, Any]],
    current_events: list[dict[str, Any]],
    candidate_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mt5_daily = _first_daily_setup_rows(mt5_rows)
    current_daily = _first_daily_setup_rows(current_events)
    candidate_daily = _first_daily_setup_rows(candidate_events)
    current_outcomes = _python_outcomes_by_day(current_events)
    candidate_outcomes = _python_outcomes_by_day(candidate_events)
    mt5_outcomes = _mt5_outcomes_by_day(mt5_rows)
    days = sorted(set(mt5_daily) | set(current_daily) | set(candidate_daily))
    rows: list[dict[str, Any]] = []
    for day in days:
        mt5 = mt5_daily.get(day, {})
        current = current_daily.get(day, {})
        candidate = candidate_daily.get(day, {})
        current_outcome = current_outcomes.get(day, {})
        candidate_outcome = candidate_outcomes.get(day, {})
        mt5_outcome = mt5_outcomes.get(day, {})
        row = {
            "day": day,
            "year": day[:4],
            "mt5_event_type": mt5.get("event_type", ""),
            "current_event_type": current.get("event_type", ""),
            "candidate_event_type": candidate.get("event_type", ""),
            "mt5_buy_order_placed": _is_true(mt5.get("buy_order_placed")),
            "mt5_sell_order_placed": _is_true(mt5.get("sell_order_placed")),
            "current_buy_order_placed": _is_true(current.get("buy_order_placed")),
            "current_sell_order_placed": _is_true(current.get("sell_order_placed")),
            "candidate_buy_order_placed": _is_true(candidate.get("buy_order_placed")),
            "candidate_sell_order_placed": _is_true(candidate.get("sell_order_placed")),
            "mt5_swing_high": _to_float(mt5.get("swing_high")),
            "mt5_swing_low": _to_float(mt5.get("swing_low")),
            "current_swing_high": _to_float(current.get("swing_high")),
            "current_swing_low": _to_float(current.get("swing_low")),
            "candidate_swing_high": _to_float(candidate.get("swing_high")),
            "candidate_swing_low": _to_float(candidate.get("swing_low")),
            "mt5_buy_entry": _to_float(mt5.get("buy_entry")),
            "mt5_sell_entry": _to_float(mt5.get("sell_entry")),
            "current_buy_entry": _to_float(current.get("buy_entry")),
            "current_sell_entry": _to_float(current.get("sell_entry")),
            "candidate_buy_entry": _to_float(candidate.get("buy_entry")),
            "candidate_sell_entry": _to_float(candidate.get("sell_entry")),
            "mt5_atr_filter_passed": _is_true(mt5.get("atr_filter_passed")),
            "current_atr_filter_passed": _is_true(current.get("atr_filter_passed")),
            "candidate_atr_filter_passed": _is_true(candidate.get("atr_filter_passed")),
            "mt5_rsi_gate_required": _is_true(mt5.get("rsi_gate_required")),
            "current_rsi_gate_required": _is_true(current.get("rsi_gate_required")),
            "candidate_rsi_gate_required": _is_true(candidate.get("rsi_gate_required")),
            "mt5_fill_side": mt5_outcome.get("fill_side", ""),
            "current_proxy_fill_side": current_outcome.get("fill_side", ""),
            "candidate_proxy_fill_side": candidate_outcome.get("fill_side", ""),
            "mt5_net_pnl": mt5_outcome.get("net_pnl", ""),
            "current_proxy_net_pnl": current_outcome.get("net_pnl", ""),
            "candidate_proxy_net_pnl": candidate_outcome.get("net_pnl", ""),
            "candidate_vs_current_buy_entry_diff": _diff(candidate.get("buy_entry"), current.get("buy_entry")),
            "candidate_vs_current_sell_entry_diff": _diff(candidate.get("sell_entry"), current.get("sell_entry")),
            "candidate_vs_mt5_buy_entry_diff": _diff(candidate.get("buy_entry"), mt5.get("buy_entry")),
            "candidate_vs_mt5_sell_entry_diff": _diff(candidate.get("sell_entry"), mt5.get("sell_entry")),
        }
        row["candidate_changes_setup"] = _changed_any(
            row,
            [
                "candidate_vs_current_buy_entry_diff",
                "candidate_vs_current_sell_entry_diff",
            ],
            tolerance=0.011,
        )
        row["candidate_changes_order_intent"] = (
            row["candidate_buy_order_placed"] != row["current_buy_order_placed"]
            or row["candidate_sell_order_placed"] != row["current_sell_order_placed"]
        )
        row["candidate_proxy_delta_pnl_vs_current_proxy"] = _round(
            (_to_float(row["candidate_proxy_net_pnl"]) or 0.0)
            - (_to_float(row["current_proxy_net_pnl"]) or 0.0)
        )
        rows.append(row)
    return rows


def summarize_setup_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    common = [row for row in rows if row["mt5_event_type"] and row["candidate_event_type"]]
    changed = [row for row in rows if row["candidate_changes_setup"]]
    intent_changed = [row for row in rows if row["candidate_changes_order_intent"]]
    candidate_positive_delta = [
        row for row in rows if (_to_float(row.get("candidate_proxy_delta_pnl_vs_current_proxy")) or 0.0) > 0.0
    ]
    return {
        "total_days_compared": len(rows),
        "common_mt5_candidate_setup_days": len(common),
        "candidate_setup_coverage_vs_mt5_pct": _pct(len(common), len([row for row in rows if row["mt5_event_type"]])),
        "candidate_changed_setup_days": len(changed),
        "candidate_changed_setup_pct": _pct(len(changed), len(rows)),
        "candidate_changed_order_intent_days": len(intent_changed),
        "candidate_positive_proxy_delta_days": len(candidate_positive_delta),
        "candidate_positive_proxy_delta_pct": _pct(len(candidate_positive_delta), len(rows)),
    }


def comparison_breakdown_rows(
    *,
    mt5_trades: list[dict[str, Any]],
    current_trades: list[dict[str, Any]],
    candidate_trades: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, trades in [
        ("mt5_observed_current_azir", _normalize_mt5_trades(mt5_trades)),
        ("python_current_proxy", current_trades),
        ("python_swing_10_fractal_proxy", candidate_trades),
    ]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trade in trades:
            grouped[_breakdown_key(trade, key)].append(trade)
        for group_key, group_rows in sorted(grouped.items()):
            metrics = compute_metrics(group_rows)
            rows.append({"source": source, key: group_key, **metrics})
    return rows


def candidate_definition() -> dict[str, Any]:
    return {
        "name": CANDIDATE_NAME,
        "variant": CANDIDATE_VARIANT,
        "base_logic": "Azir setup unchanged except for swing high/low selection.",
        "swing_window": "Last 10 fully closed M5 bars before the 16:30 server-time setup bar.",
        "pivot_confirmation": "A pivot high requires high greater than the 2 bars to the left and 2 bars to the right inside the closed-bar window. A pivot low mirrors this with lows.",
        "swing_high": "Most recent confirmed pivot high inside the 10-bar window; fallback to rolling max high if no confirmed pivot exists.",
        "swing_low": "Most recent confirmed pivot low inside the 10-bar window; fallback to rolling min low if no confirmed pivot exists.",
        "entry_offset": "Existing Azir hardcoded 5 points: buy_entry=swing_high+5*Point, sell_entry=swing_low-5*Point.",
        "unchanged_components": [
            "setup time",
            "EMA20 trend filter",
            "ATR filter",
            "RSI gate behavior",
            "Friday filter",
            "SL/TP/trailing",
            "Risk Engine v1",
        ],
    }


def candidate_definition_markdown() -> str:
    definition = candidate_definition()
    unchanged = "\n".join(f"- {item}" for item in definition["unchanged_components"])
    return (
        "# Best Setup Candidate Definition\n\n"
        f"- Candidate name: `{definition['name']}`.\n"
        f"- Variant: `{definition['variant']}`.\n"
        f"- Base logic: {definition['base_logic']}\n"
        f"- Swing window: {definition['swing_window']}\n"
        f"- Pivot confirmation: {definition['pivot_confirmation']}\n"
        f"- Swing high: {definition['swing_high']}\n"
        f"- Swing low: {definition['swing_low']}\n"
        f"- Entry offset: {definition['entry_offset']}\n\n"
        "## Unchanged Components\n\n"
        f"{unchanged}\n"
    )


def build_readiness_assessment(
    *,
    setup_comparison: dict[str, Any],
    current_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    protected_reference: dict[str, Any],
) -> dict[str, Any]:
    delta = _metric_delta(candidate_metrics, current_metrics)
    proxy_improves = (
        (_to_float(delta.get("net_pnl")) or 0.0) > 0.0
        and (_to_float(candidate_metrics.get("profit_factor")) or 0.0)
        > (_to_float(current_metrics.get("profit_factor")) or 0.0)
        and (_to_float(candidate_metrics.get("max_drawdown")) or 0.0)
        < (_to_float(current_metrics.get("max_drawdown")) or 0.0)
    )
    broad_coverage = (_to_float(setup_comparison.get("candidate_setup_coverage_vs_mt5_pct")) or 0.0) >= 99.0
    can_formalize = proxy_improves and broad_coverage
    return {
        "candidate_formal_name": CANDIDATE_NAME,
        "proxy_improvement_survives_day_by_day_screen": proxy_improves,
        "setup_coverage_broad_enough": broad_coverage,
        "may_become_formal_candidate": can_formalize,
        "may_replace_frozen_benchmark_now": False,
        "ready_for_ppo": False,
        "reason": (
            "The candidate improves the controlled proxy and covers the same setup calendar, but still lacks its own MT5 event log."
            if can_formalize
            else "The candidate does not yet clear the full formal-candidate gate."
        ),
        "required_before_promotion": [
            "Run/export the fractal setup logic in MT5 or an equivalent broker/tick replay.",
            "Compare candidate event log against current Azir day by day.",
            "Rebuild protected economics for the candidate under risk_engine_azir_v1.",
            "Only then consider freezing a new setup/economic benchmark.",
        ],
        "protected_reference_net_pnl": protected_reference.get("net_pnl"),
    }


def summary_markdown(report: dict[str, Any]) -> str:
    setup = report["setup_comparison"]
    current = report["proxy_comparison"]["current_python_proxy"]
    candidate = report["proxy_comparison"]["candidate_fractal_python_proxy"]
    delta = report["proxy_comparison"]["delta_candidate_vs_current"]
    readiness = report["readiness"]
    return (
        "# Azir Best Setup Candidate Validation\n\n"
        "## Executive Summary\n\n"
        "- Candidate: `swing_10_fractal`.\n"
        "- Current Azir side uses real MT5 event log evidence.\n"
        "- Candidate economics are still Python replica proxy because no MT5 candidate log exists yet.\n"
        f"- Setup coverage vs MT5 calendar: {setup['candidate_setup_coverage_vs_mt5_pct']}%.\n"
        f"- Candidate changed setup levels on {setup['candidate_changed_setup_days']} days "
        f"({setup['candidate_changed_setup_pct']}%).\n"
        f"- Current proxy net/PF/exp/DD: {current['net_pnl']} / {current['profit_factor']} / "
        f"{current['expectancy']} / {current['max_drawdown']}.\n"
        f"- Candidate proxy net/PF/exp/DD: {candidate['net_pnl']} / {candidate['profit_factor']} / "
        f"{candidate['expectancy']} / {candidate['max_drawdown']}.\n"
        f"- Delta candidate-current proxy net/PF/exp/DD: {delta.get('net_pnl', '')} / "
        f"{delta.get('profit_factor', '')} / {delta.get('expectancy', '')} / "
        f"{delta.get('max_drawdown', '')}.\n"
        f"- May become formal candidate: {readiness['may_become_formal_candidate']}.\n"
        "- May replace frozen benchmark now: False.\n\n"
        "## Decision\n\n"
        f"{readiness['reason']}\n"
    )


def readiness_markdown(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    requirements = "\n".join(f"- {item}" for item in readiness["required_before_promotion"])
    return (
        "# Candidate Readiness Assessment\n\n"
        f"- Formal candidate name: `{readiness['candidate_formal_name']}`.\n"
        f"- Proxy improvement survives screen: {readiness['proxy_improvement_survives_day_by_day_screen']}.\n"
        f"- Setup coverage broad enough: {readiness['setup_coverage_broad_enough']}.\n"
        f"- May become formal candidate: {readiness['may_become_formal_candidate']}.\n"
        f"- May replace frozen benchmark now: {readiness['may_replace_frozen_benchmark_now']}.\n"
        f"- Ready for PPO: {readiness['ready_for_ppo']}.\n\n"
        "## Required Before Promotion\n\n"
        f"{requirements}\n"
    )


def _load_protected_reference(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    metrics = report["metrics"]["azir_with_risk_engine_v1_forced_closes_revalued"]
    return {
        "benchmark_name": report.get("benchmark_name", "baseline_azir_protected_economic_v1"),
        "closed_trades": metrics.get("closed_trades"),
        "net_pnl": metrics.get("net_pnl"),
        "profit_factor": metrics.get("profit_factor"),
        "expectancy": metrics.get("expectancy"),
        "max_drawdown": metrics.get("max_drawdown_abs"),
        "max_consecutive_losses": metrics.get("max_consecutive_losses"),
    }


def _load_setup_research_reference(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    ranking = report.get("candidate_ranking", [])
    return {
        "source": str(path),
        "top_variant": ranking[0].get("variant") if ranking else "",
        "top_variant_metrics": ranking[0] if ranking else {},
    }


def _python_outcomes_by_day(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    outcomes: dict[str, dict[str, Any]] = {}
    fill_by_day: dict[str, dict[str, Any]] = {}
    for row in events:
        day = _event_day(row)
        if row.get("event_type") == "fill":
            fill_by_day[day] = row
        elif row.get("event_type") == "exit":
            outcomes[day] = {
                "fill_side": row.get("fill_side", fill_by_day.get(day, {}).get("fill_side", "")),
                "net_pnl": _to_float(row.get("net_pnl")) or 0.0,
                "exit_reason": row.get("exit_reason", ""),
            }
    return outcomes


def _mt5_outcomes_by_day(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    reconstruction = reconstruct_lifecycles(rows, session_start_hour=16, session_end_hour=21, close_hour=22)
    outcomes: dict[str, dict[str, Any]] = {}
    for trade in reconstruction["trades"]:
        day = str(trade.get("setup_day", ""))
        if not day:
            continue
        outcomes.setdefault(
            day,
            {
                "fill_side": trade.get("fill_side", ""),
                "net_pnl": _to_float(trade.get("net_pnl")) or 0.0,
                "exit_reason": trade.get("exit_reason", ""),
            },
        )
    return outcomes


def _normalize_mt5_trades(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in trades:
        rows.append(
            {
                "has_exit": bool(row.get("has_exit")),
                "setup_day": row.get("setup_day", ""),
                "year": str(row.get("setup_day", ""))[:4],
                "side": row.get("fill_side", ""),
                "net_pnl": _to_float(row.get("net_pnl")) or 0.0,
                "gross_pnl": _to_float(row.get("gross_pnl")) or 0.0,
            }
        )
    return rows


def _breakdown_key(row: dict[str, Any], key: str) -> str:
    if key == "year":
        return str(row.get("year") or str(row.get("setup_day", ""))[:4])
    if key == "side":
        return str(row.get("side") or row.get("fill_side") or "")
    return str(row.get(key, ""))


def _metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "closed_trades",
        "net_pnl",
        "profit_factor",
        "expectancy",
        "max_drawdown",
        "max_drawdown_abs",
        "max_consecutive_losses",
    ]
    result: dict[str, Any] = {}
    for key in keys:
        left_value = _to_float(left.get(key))
        right_value = _to_float(right.get(key))
        if left_value is None or right_value is None:
            continue
        out_key = "max_drawdown" if key == "max_drawdown_abs" else key
        result[out_key] = _round(left_value - right_value)
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


def _pct(numerator: int, denominator: int) -> float:
    return _round((numerator / denominator) * 100.0) if denominator else 0.0


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    readiness = report["readiness"]
    delta = report["proxy_comparison"]["delta_candidate_vs_current"]
    return {
        "sprint": report["sprint"],
        "candidate": report["candidate_variant"],
        "candidate_name": report["candidate_name"],
        "setup_coverage_vs_mt5_pct": report["setup_comparison"]["candidate_setup_coverage_vs_mt5_pct"],
        "delta_proxy_net_pnl": delta.get("net_pnl"),
        "delta_proxy_profit_factor": delta.get("profit_factor"),
        "may_become_formal_candidate": readiness["may_become_formal_candidate"],
        "may_replace_frozen_benchmark_now": readiness["may_replace_frozen_benchmark_now"],
        "ready_for_ppo": readiness["ready_for_ppo"],
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
