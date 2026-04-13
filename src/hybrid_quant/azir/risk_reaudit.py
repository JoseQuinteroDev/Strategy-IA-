"""Re-audit Azir economics after applying risk_engine_azir_v1.

This module does not rerun MT5 and does not change Azir. It applies the
external lifecycle guardrails to the empirical MT5 event log and separates
known protected economics from trades whose close would need tick/price replay.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hybrid_quant.risk.azir_state import AzirRiskConfig

from .economic_audit import (
    _event_day_from_text,
    _parse_day,
    _parse_timestamp,
    _round,
    _to_float,
    _write_csv,
    build_anomaly_reports,
    reconstruct_lifecycles,
    summarize_lifecycles,
    summarize_trades,
    _read_raw_event_log,
)


DEFAULT_SYMBOL = "XAUUSD-STD"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-audit Azir MT5 economics with risk_engine_azir_v1 applied.")
    parser.add_argument("--mt5-log-path", required=True, help="Canonical Azir MT5 event log CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for risk re-audit artifacts.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--config-name", default="risk_engine_azir_v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_risk_reaudit(
        mt5_log_path=Path(args.mt5_log_path),
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
        config=AzirRiskConfig(name=args.config_name),
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_risk_reaudit(
    *,
    mt5_log_path: Path,
    output_dir: Path,
    symbol: str = DEFAULT_SYMBOL,
    config: AzirRiskConfig | None = None,
) -> dict[str, Any]:
    config = config or AzirRiskConfig()
    rows = _read_raw_event_log(mt5_log_path, symbol)
    reconstruction = reconstruct_lifecycles(
        rows,
        session_start_hour=config.session_fill_start_hour,
        session_end_hour=config.session_fill_end_hour,
        close_hour=config.close_hour,
    )
    lifecycle_rows = reconstruction["lifecycles"]
    trade_rows = reconstruction["trades"]
    before_anomalies = build_anomaly_reports(
        rows,
        lifecycle_rows,
        trade_rows,
        config.session_fill_start_hour,
        config.session_fill_end_hour,
    )
    simulation = apply_risk_engine_to_lifecycle(
        rows=rows,
        lifecycle_rows=lifecycle_rows,
        trade_rows=trade_rows,
        config=config,
    )
    protected_trades = simulation["protected_trades"]
    decision_rows = simulation["trade_decisions"]
    lifecycle_after = simulation["lifecycle_after"]
    after_anomaly_counts = _after_anomaly_counts(protected_trades, lifecycle_after, config)
    before_counts = _before_anomaly_counts(before_anomalies)
    original_metrics = summarize_trades(trade_rows)
    protected_metrics = summarize_trades(protected_trades)
    before_after_rows = _before_after_metric_rows(original_metrics, protected_metrics)
    anomaly_rows = _anomaly_before_after_rows(before_counts, after_anomaly_counts)
    decision_summary = _decision_summary(decision_rows)
    report = {
        "source_log": str(mt5_log_path),
        "symbol": symbol,
        "risk_engine": asdict(config),
        "simulation_scope": {
            "type": "observed_lifecycle_policy_simulation",
            "mt5_rerun": False,
            "tick_replay": False,
            "known_pnl_policy": (
                "Observed MT5 net_pnl is retained only for trades that risk_engine_azir_v1 "
                "would still allow and close at the same observed exit. Forced close trades "
                "are marked unpriced instead of reusing a non-comparable later exit."
            ),
        },
        "original_lifecycle_summary": summarize_lifecycles(lifecycle_rows),
        "protected_lifecycle_summary": _summarize_lifecycle_after(lifecycle_after),
        "original_economic_metrics": original_metrics,
        "protected_known_economic_metrics": protected_metrics,
        "trade_decision_summary": decision_summary,
        "anomaly_before_after": anomaly_rows,
        "before_after_economic_comparison": before_after_rows,
        "decision": _build_decision(original_metrics, protected_metrics, decision_summary, after_anomaly_counts),
        "limitations": [
            "This is a counterfactual policy simulation over the MT5 event log, not a broker/tick-level rerun.",
            "Forced closes at the Risk Engine close time are not repriced without tick/price replay.",
            "Trailing remains observational and path-dependent; this sprint does not modify trailing.",
            "Spread guard is applied only when setup spread is available in the event log.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(before_after_rows, output_dir / "before_after_economic_comparison.csv")
    _write_csv(anomaly_rows, output_dir / "anomaly_before_after.csv")
    _write_csv(lifecycle_after, output_dir / "lifecycle_after_risk.csv")
    _write_csv(decision_rows, output_dir / "protected_trade_decisions.csv")
    (output_dir / "reaudited_economic_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "reaudited_economic_summary.md").write_text(
        _summary_markdown(report),
        encoding="utf-8",
    )
    (output_dir / "protected_benchmark_candidate.md").write_text(
        _protected_candidate_markdown(report),
        encoding="utf-8",
    )
    return report


def apply_risk_engine_to_lifecycle(
    *,
    rows: list[dict[str, Any]],
    lifecycle_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    config: AzirRiskConfig,
) -> dict[str, list[dict[str, Any]]]:
    setup_spreads = _setup_spreads_by_day(rows)
    lifecycles_by_day = {row["setup_day"]: row for row in lifecycle_rows}
    trade_decisions: list[dict[str, Any]] = []
    protected_trades: list[dict[str, Any]] = []
    kept_by_setup_day: dict[str, int] = defaultdict(int)
    kept_by_fill_day: dict[str, int] = defaultdict(int)
    daily_realized: dict[str, float] = defaultdict(float)
    daily_loss_streak: dict[str, int] = defaultdict(int)

    for trade in sorted(trade_rows, key=lambda row: row.get("fill_timestamp") or row.get("exit_timestamp") or ""):
        decision = _classify_trade(
            trade=trade,
            lifecycle=lifecycles_by_day.get(str(trade.get("setup_day", "")), {}),
            setup_spread_points=setup_spreads.get(str(trade.get("setup_day", ""))),
            kept_by_setup_day=kept_by_setup_day,
            kept_by_fill_day=kept_by_fill_day,
            daily_realized=daily_realized,
            daily_loss_streak=daily_loss_streak,
            config=config,
        )
        trade_decisions.append(decision)
        if decision["risk_status"] != "kept_observed_exit":
            continue
        protected = dict(trade)
        protected["risk_status"] = decision["risk_status"]
        protected["risk_reason"] = decision["risk_reason"]
        protected_trades.append(protected)
        setup_day = str(trade.get("setup_day", ""))
        fill_day = _event_day_from_text(trade.get("fill_timestamp"))
        pnl = _to_float(trade.get("net_pnl")) or 0.0
        kept_by_setup_day[setup_day] += 1
        kept_by_fill_day[fill_day] += 1
        daily_realized[fill_day] += pnl
        daily_loss_streak[fill_day] = daily_loss_streak[fill_day] + 1 if pnl < 0 else 0

    lifecycle_after = _build_lifecycle_after(lifecycle_rows, trade_decisions, config)
    return {
        "protected_trades": protected_trades,
        "trade_decisions": trade_decisions,
        "lifecycle_after": lifecycle_after,
    }


def _classify_trade(
    *,
    trade: dict[str, Any],
    lifecycle: dict[str, Any],
    setup_spread_points: float | None,
    kept_by_setup_day: dict[str, int],
    kept_by_fill_day: dict[str, int],
    daily_realized: dict[str, float],
    daily_loss_streak: dict[str, int],
    config: AzirRiskConfig,
) -> dict[str, Any]:
    setup_day = str(trade.get("setup_day", ""))
    fill_timestamp = str(trade.get("fill_timestamp", ""))
    exit_timestamp = str(trade.get("exit_timestamp", ""))
    fill_day = _event_day_from_text(fill_timestamp)
    exit_day = _event_day_from_text(exit_timestamp)
    fill_dt = _parse_timestamp(fill_timestamp)
    exit_dt = _parse_timestamp(exit_timestamp)
    pnl = _to_float(trade.get("net_pnl")) or 0.0

    status = "kept_observed_exit"
    reason = "Risk Engine would not change this observed lifecycle materially."
    rules: list[str] = []
    priced = True

    if config.spread_guard_enabled and setup_spread_points is not None and setup_spread_points > config.max_spread_points:
        status = "prevented"
        reason = "Setup spread exceeded configured spread guard."
        rules.append("spread_guard_if_available")
    elif config.friday_no_new_trade and _is_friday(setup_day):
        status = "prevented"
        reason = "Friday setup would be blocked by the external Risk Engine."
        rules.append("friday_no_new_trade_plus_close_or_cancel_prior_exposure")
    elif _fill_outside_window(fill_dt, config) or fill_day != setup_day:
        status = "prevented"
        reason = "Pending would have been cancelled at operational close before this fill."
        rules.extend(["hard_cancel_all_pendings_at_close", "force_reconcile_orders_positions_before_setup"])
    elif kept_by_setup_day[setup_day] >= 1:
        status = "prevented"
        reason = "A prior fill from the same setup already exists; remaining pendings would be cancelled after fill."
        rules.append("cancel_remaining_pendings_after_fill")
    elif kept_by_fill_day[fill_day] >= config.max_trades_per_day:
        status = "prevented"
        reason = "Maximum Azir trades per day reached."
        rules.append("max_trades_per_day")
    elif daily_realized[fill_day] <= -abs(config.max_daily_loss):
        status = "prevented"
        reason = "Daily max loss guard was already active before this trade."
        rules.append("daily_max_loss_guard")
    elif config.max_consecutive_losses > 0 and daily_loss_streak[fill_day] >= config.max_consecutive_losses:
        status = "prevented"
        reason = "Daily consecutive loss kill-switch was already active before this trade."
        rules.append("consecutive_losses_kill_switch")
    elif config.close_positions_at_close and exit_timestamp and _requires_unpriced_force_close(exit_dt, setup_day, config):
        status = "forced_close_unpriced"
        reason = "Risk Engine would close the position at session close; exact replacement PnL needs tick/price replay."
        rules.append("hard_cancel_all_pendings_at_close")
        priced = False

    return {
        "setup_day": setup_day,
        "fill_timestamp": fill_timestamp,
        "exit_timestamp": exit_timestamp,
        "fill_day": fill_day,
        "exit_day": exit_day,
        "fill_side": trade.get("fill_side", ""),
        "fill_price": trade.get("fill_price", ""),
        "exit_reason": trade.get("exit_reason", ""),
        "observed_net_pnl": trade.get("net_pnl", ""),
        "observed_gross_pnl": trade.get("gross_pnl", ""),
        "trade_sequence": trade.get("trade_sequence", ""),
        "original_assignment_reason": trade.get("assignment_reason", ""),
        "original_lifecycle_status": lifecycle.get("lifecycle_status", ""),
        "setup_spread_points": "" if setup_spread_points is None else setup_spread_points,
        "risk_status": status,
        "risk_reason": reason,
        "risk_rules": "|".join(rules) if rules else "none",
        "pnl_priced_after_risk": priced,
    }


def _build_lifecycle_after(
    lifecycle_rows: list[dict[str, Any]],
    trade_decisions: list[dict[str, Any]],
    config: AzirRiskConfig,
) -> list[dict[str, Any]]:
    decisions_by_setup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trade_decisions:
        decisions_by_setup[row["setup_day"]].append(row)

    rows: list[dict[str, Any]] = []
    for lifecycle in lifecycle_rows:
        setup_day = lifecycle["setup_day"]
        decisions = decisions_by_setup.get(setup_day, [])
        kept = [row for row in decisions if row["risk_status"] == "kept_observed_exit"]
        prevented = [row for row in decisions if row["risk_status"] == "prevented"]
        unpriced = [row for row in decisions if row["risk_status"] == "forced_close_unpriced"]
        had_cleanup_issue = (
            lifecycle["order_placed"]
            and lifecycle["cleanup_count"] == 0
            and (
                lifecycle["lifecycle_status"] == "missing_cleanup_or_unresolved"
                or lifecycle["survived_change_of_day"]
                or lifecycle["out_of_window_fill_count"] > 0
            )
        )
        if kept:
            protected_status = "filled_observed_exit_kept"
        elif unpriced:
            protected_status = "forced_closed_unpriced"
        elif lifecycle["order_placed"]:
            protected_status = "cancelled_or_blocked_no_protected_fill" if (prevented or had_cleanup_issue) else "cancelled_no_fill"
        else:
            protected_status = lifecycle["lifecycle_status"]
        rows.append(
            {
                "setup_day": setup_day,
                "setup_timestamp": lifecycle.get("setup_timestamp", ""),
                "original_lifecycle_status": lifecycle.get("lifecycle_status", ""),
                "protected_lifecycle_status": protected_status,
                "original_fill_count": lifecycle.get("fill_count", 0),
                "protected_fill_count": len(kept) + len(unpriced),
                "original_exit_count": lifecycle.get("exit_count", 0),
                "protected_priced_exit_count": len(kept),
                "forced_close_unpriced_count": len(unpriced),
                "prevented_trade_count": len(prevented),
                "original_cleanup_count": lifecycle.get("cleanup_count", 0),
                "cleanup_issue_before": had_cleanup_issue,
                "cleanup_issue_after": False if config.hard_cancel_all_pendings_at_close else had_cleanup_issue,
                "out_of_window_fill_count_before": lifecycle.get("out_of_window_fill_count", 0),
                "out_of_window_fill_count_after": 0,
                "survived_change_of_day_before": lifecycle.get("survived_change_of_day", False),
                "survived_change_of_day_after": False,
                "risk_actions": _lifecycle_actions(prevented, unpriced, had_cleanup_issue, config),
                "notes": _lifecycle_notes(lifecycle, prevented, unpriced),
            }
        )
    return rows


def _after_anomaly_counts(
    protected_trades: list[dict[str, Any]],
    lifecycle_after: list[dict[str, Any]],
    config: AzirRiskConfig,
) -> dict[str, int]:
    exits_by_day: dict[str, int] = defaultdict(int)
    out_of_window = 0
    friday_exits = 0
    for trade in protected_trades:
        fill_dt = _parse_timestamp(trade.get("fill_timestamp"))
        if _fill_outside_window(fill_dt, config):
            out_of_window += 1
        exit_day = _event_day_from_text(trade.get("exit_timestamp"))
        if exit_day:
            exits_by_day[exit_day] += 1
            if _parse_timestamp(trade.get("exit_timestamp")).weekday() == 4:
                friday_exits += 1
    return {
        "fills_outside_window": out_of_window,
        "friday_exit_events": friday_exits,
        "multi_exit_days": len([day for day, count in exits_by_day.items() if count > 1]),
        "cleanup_persistence_issues": len([row for row in lifecycle_after if row["cleanup_issue_after"]]),
        "residual_cross_day_exposure": len([row for row in lifecycle_after if row["survived_change_of_day_after"]]),
        "forced_close_unpriced": len([row for row in lifecycle_after if int(row["forced_close_unpriced_count"]) > 0]),
    }


def _before_anomaly_counts(anomalies: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {
        "fills_outside_window": len(anomalies["out_of_window_fills"]),
        "friday_exit_events": len(anomalies["friday_exit_events"]),
        "multi_exit_days": len(anomalies["multi_exit_days"]),
        "cleanup_persistence_issues": len(anomalies["open_order_cleanup_issues"]),
        "residual_cross_day_exposure": len(anomalies["open_order_cleanup_issues"]),
        "forced_close_unpriced": 0,
    }


def _before_after_metric_rows(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    metric_names = [
        "closed_trades",
        "net_pnl",
        "win_rate",
        "average_win",
        "average_loss",
        "payoff",
        "profit_factor",
        "expectancy",
        "max_drawdown_abs",
        "max_consecutive_losses",
    ]
    rows = []
    for metric in metric_names:
        before_value = before.get(metric)
        after_value = after.get(metric)
        rows.append(
            {
                "metric": metric,
                "azir_original_observed": before_value,
                "azir_with_risk_engine_v1_known_pnl": after_value,
                "delta": _delta(before_value, after_value),
                "notes": "Protected column excludes prevented trades and unpriced forced-close counterfactuals.",
            }
        )
    return rows


def _anomaly_before_after_rows(before: dict[str, int], after: dict[str, int]) -> list[dict[str, Any]]:
    labels = {
        "fills_outside_window": "fills fuera de ventana",
        "friday_exit_events": "exits en viernes bloqueados/prior exposure",
        "multi_exit_days": "dias con multiples exits",
        "cleanup_persistence_issues": "cleanup/persistence issues",
        "residual_cross_day_exposure": "exposicion residual entre dias",
        "forced_close_unpriced": "cierres forzados no revalorables sin replay",
    }
    rows = []
    for key, label in labels.items():
        before_count = before.get(key, 0)
        after_count = after.get(key, 0)
        rows.append(
            {
                "anomaly": key,
                "description": label,
                "before": before_count,
                "after": after_count,
                "reduced_by": before_count - after_count,
                "status": _anomaly_status(key, before_count, after_count),
            }
        )
    return rows


def _decision_summary(decision_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in decision_rows:
        counts[str(row["risk_status"])] += 1
    return dict(counts)


def _build_decision(
    before_metrics: dict[str, Any],
    after_metrics: dict[str, Any],
    decision_summary: dict[str, int],
    after_anomalies: dict[str, int],
) -> dict[str, Any]:
    forced_unpriced = decision_summary.get("forced_close_unpriced", 0)
    known_net_positive = (after_metrics.get("net_pnl") or 0) > 0
    pf_ok = (after_metrics.get("profit_factor") or 0) > 1.0
    lifecycle_clean = (
        after_anomalies.get("fills_outside_window", 0) == 0
        and after_anomalies.get("multi_exit_days", 0) == 0
        and after_anomalies.get("cleanup_persistence_issues", 0) == 0
        and after_anomalies.get("residual_cross_day_exposure", 0) == 0
    )
    return {
        "protected_economic_benchmark_can_be_frozen": bool(lifecycle_clean and known_net_positive and pf_ok and forced_unpriced == 0),
        "risk_engine_improves_lifecycle_robustness": bool(lifecycle_clean),
        "risk_engine_kills_observed_edge": not (known_net_positive and pf_ok),
        "ready_for_rl_environment_design": bool(lifecycle_clean),
        "ready_for_ppo_training": False,
        "recommended_next_sprint": (
            "price_replay_for_forced_closes_and_protected_benchmark_freeze"
            if forced_unpriced
            else "design_rl_environment_contract_with_protected_azir_benchmark"
        ),
        "reason": (
            "Lifecycle anomalies are controlled in the simulation, but protected economics are still not fully "
            "freezeable while forced-close counterfactuals remain unpriced."
            if forced_unpriced
            else "Lifecycle anomalies are controlled and all protected trades retain comparable observed exits."
        ),
        "original_net_pnl": before_metrics.get("net_pnl"),
        "protected_known_net_pnl": after_metrics.get("net_pnl"),
    }


def _summarize_lifecycle_after(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts["lifecycles"] += 1
        counts[f"status_{row['protected_lifecycle_status']}"] += 1
        counts["protected_priced_exits"] += int(row["protected_priced_exit_count"])
        counts["forced_close_unpriced"] += int(row["forced_close_unpriced_count"])
        counts["prevented_trades"] += int(row["prevented_trade_count"])
        counts["cleanup_issues_after"] += 1 if row["cleanup_issue_after"] else 0
        counts["cross_day_exposure_after"] += 1 if row["survived_change_of_day_after"] else 0
    return dict(counts)


def _setup_spreads_by_day(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    spreads: dict[str, float | None] = {}
    for row in rows:
        if row.get("event_type") not in {"opportunity", "blocked_friday"}:
            continue
        spreads[str(row["_event_day"])] = _to_float(row.get("spread_points"))
    return spreads


def _fill_outside_window(fill_dt: Any, config: AzirRiskConfig) -> bool:
    if getattr(fill_dt, "year", 1) == 1:
        return True
    return not (config.session_fill_start_hour <= fill_dt.hour <= config.session_fill_end_hour)


def _requires_unpriced_force_close(exit_dt: Any, setup_day: str, config: AzirRiskConfig) -> bool:
    if getattr(exit_dt, "year", 1) == 1:
        return False
    setup_date = _parse_day(setup_day)
    if exit_dt.date() > setup_date:
        return True
    if exit_dt.date() < setup_date:
        return False
    if exit_dt.hour > config.close_hour:
        return True
    if exit_dt.hour == config.close_hour and (exit_dt.minute > 0 or exit_dt.second > 0):
        return True
    return False


def _is_friday(day_text: str) -> bool:
    if not day_text:
        return False
    return _parse_day(day_text).weekday() == 4


def _lifecycle_actions(
    prevented: list[dict[str, Any]],
    unpriced: list[dict[str, Any]],
    had_cleanup_issue: bool,
    config: AzirRiskConfig,
) -> str:
    actions: list[str] = []
    for row in prevented:
        for rule in str(row["risk_rules"]).split("|"):
            if rule and rule != "none" and rule not in actions:
                actions.append(rule)
    if unpriced and "hard_cancel_all_pendings_at_close" not in actions:
        actions.append("hard_cancel_all_pendings_at_close")
    if had_cleanup_issue and config.hard_cancel_all_pendings_at_close and "hard_cancel_all_pendings_at_close" not in actions:
        actions.append("hard_cancel_all_pendings_at_close")
    return "|".join(actions) if actions else "none"


def _lifecycle_notes(
    lifecycle: dict[str, Any],
    prevented: list[dict[str, Any]],
    unpriced: list[dict[str, Any]],
) -> str:
    notes: list[str] = []
    if lifecycle.get("out_of_window_fill_count", 0) > 0:
        notes.append("original_out_of_window_fill_removed_or_prevented")
    if lifecycle.get("survived_change_of_day"):
        notes.append("original_cross_day_exposure_removed_or_forced_close")
    if prevented:
        notes.append("one_or_more_observed_trades_prevented")
    if unpriced:
        notes.append("forced_close_requires_price_replay")
    return "|".join(notes) if notes else "clean_or_unchanged"


def _delta(before: Any, after: Any) -> float | str:
    before_float = _to_float(before)
    after_float = _to_float(after)
    if before_float is None or after_float is None:
        return ""
    return _round(after_float - before_float)


def _anomaly_status(key: str, before_count: int, after_count: int) -> str:
    if key == "forced_close_unpriced":
        return "new_counterfactual_pricing_gap" if after_count else "none"
    if before_count and after_count == 0:
        return "resolved_by_risk_engine_simulation"
    if after_count < before_count:
        return "reduced_by_risk_engine_simulation"
    if after_count == before_count:
        return "unchanged"
    return "increased"


def _summary_markdown(report: dict[str, Any]) -> str:
    before = report["original_economic_metrics"]
    after = report["protected_known_economic_metrics"]
    decision = report["decision"]
    anomaly_lines = "\n".join(
        f"- {row['description']}: {row['before']} -> {row['after']} ({row['status']})."
        for row in report["anomaly_before_after"]
    )
    return (
        "# Azir Re-audit With risk_engine_azir_v1\n\n"
        "## Executive Summary\n\n"
        "- This is a counterfactual lifecycle simulation over the observed MT5 event log, not an MT5 rerun.\n"
        f"- Original observed closed trades: {before['closed_trades']}; protected known closed trades: {after['closed_trades']}.\n"
        f"- Original net PnL: {before['net_pnl']}; protected known net PnL: {after['net_pnl']}.\n"
        f"- Original PF: {before['profit_factor']}; protected known PF: {after['profit_factor']}.\n"
        f"- Original max DD abs: {before['max_drawdown_abs']}; protected known max DD abs: {after['max_drawdown_abs']}.\n"
        f"- Protected economic benchmark frozen: {decision['protected_economic_benchmark_can_be_frozen']}.\n"
        f"- Ready for PPO training: {decision['ready_for_ppo_training']}.\n\n"
        "## Anomalies Before vs After\n\n"
        f"{anomaly_lines}\n\n"
        "## Decision\n\n"
        f"- Risk Engine improves lifecycle robustness: {decision['risk_engine_improves_lifecycle_robustness']}.\n"
        f"- Risk Engine kills observed edge: {decision['risk_engine_kills_observed_edge']}.\n"
        f"- Ready for RL environment design: {decision['ready_for_rl_environment_design']}.\n"
        f"- Recommended next sprint: `{decision['recommended_next_sprint']}`.\n"
        f"- Reason: {decision['reason']}\n\n"
        "## Critical Limitation\n\n"
        "Protected PnL is known only for trades whose observed exit remains comparable after applying the guardrails. "
        "Any forced-close counterfactual needs price/tick replay before freezing a final economic benchmark.\n"
    )


def _protected_candidate_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    after = report["protected_known_economic_metrics"]
    return (
        "# Protected Benchmark Candidate\n\n"
        "## Candidate\n\n"
        "- Name: `baseline_azir_protected_economic_candidate_v1`.\n"
        "- Base strategy: AzirIA MT5 unchanged.\n"
        "- Protection layer: `risk_engine_azir_v1` applied externally.\n"
        "- Status: CANDIDATE, NOT FROZEN.\n\n"
        "## Known Protected Economics\n\n"
        f"- Closed trades with comparable observed exits: {after['closed_trades']}.\n"
        f"- Net PnL: {after['net_pnl']}.\n"
        f"- Profit factor: {after['profit_factor']}.\n"
        f"- Expectancy: {after['expectancy']}.\n"
        f"- Max drawdown abs: {after['max_drawdown_abs']}.\n\n"
        "## Freeze Decision\n\n"
        f"- Can freeze now: {decision['protected_economic_benchmark_can_be_frozen']}.\n"
        f"- Reason: {decision['reason']}\n\n"
        "## Required Before Freeze\n\n"
        "- Reprice any forced-close counterfactuals at the Risk Engine close timestamp.\n"
        "- Confirm that MT5/order-state cleanup logs map one-to-one to protected lifecycle state.\n"
        "- Keep trailing observational unless tick replay is added.\n"
    )


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "protected_economic_benchmark_can_be_frozen": report["decision"]["protected_economic_benchmark_can_be_frozen"],
        "risk_engine_improves_lifecycle_robustness": report["decision"]["risk_engine_improves_lifecycle_robustness"],
        "ready_for_rl_environment_design": report["decision"]["ready_for_rl_environment_design"],
        "ready_for_ppo_training": report["decision"]["ready_for_ppo_training"],
        "original_economic_metrics": report["original_economic_metrics"],
        "protected_known_economic_metrics": report["protected_known_economic_metrics"],
        "trade_decision_summary": report["trade_decision_summary"],
        "recommended_next_sprint": report["decision"]["recommended_next_sprint"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
