"""Economic and order-lifecycle audit for Azir MT5 logs.

This sprint does not change Azir. It reconstructs order/position lifecycles
from the empirical MT5 event log and marks where evidence is observational,
not causal or benchmark-grade.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from .event_log import AZIR_EVENT_COLUMNS


DEFAULT_SYMBOL = "XAUUSD-STD"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Azir economic lifecycle audit.")
    parser.add_argument("--mt5-log-path", required=True, help="Azir MT5 event log CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for economic audit artifacts.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--session-start-hour", type=int, default=16)
    parser.add_argument("--session-end-hour", type=int, default=21)
    parser.add_argument("--close-hour", type=int, default=22)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_economic_audit(
        mt5_log_path=Path(args.mt5_log_path),
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
        session_start_hour=args.session_start_hour,
        session_end_hour=args.session_end_hour,
        close_hour=args.close_hour,
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_economic_audit(
    *,
    mt5_log_path: Path,
    output_dir: Path,
    symbol: str = DEFAULT_SYMBOL,
    session_start_hour: int = 16,
    session_end_hour: int = 21,
    close_hour: int = 22,
) -> dict[str, Any]:
    rows = _read_raw_event_log(mt5_log_path, symbol)
    if not rows:
        raise ValueError("No Azir rows found for the requested symbol.")
    output_dir.mkdir(parents=True, exist_ok=True)

    reconstruction = reconstruct_lifecycles(
        rows,
        session_start_hour=session_start_hour,
        session_end_hour=session_end_hour,
        close_hour=close_hour,
    )
    lifecycle_rows = reconstruction["lifecycles"]
    trade_rows = reconstruction["trades"]
    anomalies = build_anomaly_reports(rows, lifecycle_rows, trade_rows, session_start_hour, session_end_hour)
    trailing = build_trailing_report(trade_rows)
    loss_tail = build_loss_tail_report(trade_rows)
    report = {
        "source_log": str(mt5_log_path),
        "symbol": symbol,
        "event_rows": len(rows),
        "event_counts": dict(Counter(row["event_type"] for row in rows)),
        "session_fill_hours_considered_normal": f"{session_start_hour}-{session_end_hour}",
        "lifecycle_summary": summarize_lifecycles(lifecycle_rows),
        "trade_summary": summarize_trades(trade_rows),
        "anomaly_summary": {key: len(value) for key, value in anomalies.items()},
        "trailing_summary": trailing["summary"],
        "loss_tail_summary": loss_tail["summary"],
        "decision": {
            "economic_benchmark_can_be_frozen": False,
            "ready_for_risk_engine_design": True,
            "ready_for_ppo": False,
            "recommended_next_sprint": "risk_engine_design_with_lifecycle_guards",
            "reason": (
                "The MT5 economic log is useful enough to design protective rules, "
                "but not clean enough to freeze a Python economic benchmark without "
                "explicit tick/trailing and GTC-order modeling."
            ),
        },
        "risk_engine_rules_recommended": recommended_risk_engine_rules(anomalies, trailing, loss_tail),
        "notes": [
            "Trailing analysis is observational; it is not causal evidence.",
            "Blank event_id fill/trailing/exit rows are treated as lifecycle evidence, not discarded.",
            "GTC pending survival is inferred when a fill is assigned to a previous setup day.",
        ],
    }

    _write_csv(lifecycle_rows, output_dir / "order_lifecycle_report.csv")
    _write_csv(trade_rows, output_dir / "trade_economic_records.csv")
    _write_csv(anomalies["all"], output_dir / "anomalous_days.csv")
    _write_csv(anomalies["multi_exit_days"], output_dir / "multi_exit_days.csv")
    _write_csv(anomalies["out_of_window_fills"], output_dir / "out_of_window_fills.csv")
    _write_csv(anomalies["friday_exit_events"], output_dir / "friday_exit_events.csv")
    _write_csv(anomalies["open_order_cleanup_issues"], output_dir / "open_order_cleanup_issues.csv")
    _write_csv(trailing["rows"], output_dir / "trailing_causality_report.csv")
    _write_csv(loss_tail["rows"], output_dir / "loss_tail_analysis.csv")
    (output_dir / "economic_audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "economic_audit_summary.md").write_text(
        economic_summary_markdown(report, anomalies),
        encoding="utf-8",
    )
    return report


def _read_raw_event_log(path: Path, symbol: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"MT5 event log does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(AZIR_EVENT_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Event log missing canonical columns: {sorted(missing)}")
        rows = []
        for row in reader:
            if symbol and row.get("symbol") and row.get("symbol") != symbol:
                continue
            row = dict(row)
            row["_raw_event_id_blank"] = not str(row.get("event_id", "")).strip()
            row["_event_day"] = _event_day(row)
            row["_timestamp_dt"] = _parse_timestamp(row.get("timestamp"))
            row["_event_id_day"] = _event_id_day(row.get("event_id"))
            rows.append(row)
    rows.sort(key=lambda row: row["_timestamp_dt"])
    return rows


def reconstruct_lifecycles(
    rows: list[dict[str, Any]],
    *,
    session_start_hour: int,
    session_end_hour: int,
    close_hour: int,
) -> dict[str, list[dict[str, Any]]]:
    lifecycles: dict[str, dict[str, Any]] = {}
    pending_open: list[str] = []
    active_trades: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []

    for row in rows:
        event_type = row.get("event_type", "")
        event_day = row["_event_day"]
        if event_type in {"opportunity", "blocked_friday"}:
            lifecycle = lifecycles.setdefault(event_day, _new_lifecycle(event_day, row))
            _update_lifecycle_from_setup(lifecycle, row)
            if lifecycle["order_placed"] and event_day not in pending_open:
                pending_open.append(event_day)
            continue

        setup_day = _assign_setup_day(row, lifecycles, pending_open, active_trades)
        lifecycle = lifecycles.setdefault(setup_day, _new_lifecycle(setup_day, row))

        if event_type == "no_fill_close_cleanup":
            lifecycle["cleanup_timestamp"] = row.get("timestamp", "")
            lifecycle["cleanup_count"] += 1
            if setup_day in pending_open:
                pending_open.remove(setup_day)
            continue

        if event_type == "fill":
            trade = _new_trade(row, lifecycle, len([item for item in trades if item["setup_day"] == setup_day]) + 1)
            trade["assigned_from_blank_event_id"] = bool(row["_raw_event_id_blank"])
            trade["assignment_reason"] = _assignment_reason(row, setup_day, event_day)
            trades.append(trade)
            lifecycle["fill_count"] += 1
            lifecycle["first_fill_timestamp"] = lifecycle["first_fill_timestamp"] or row.get("timestamp", "")
            lifecycle["last_fill_timestamp"] = row.get("timestamp", "")
            lifecycle["assigned_blank_fill_count"] += 1 if row["_raw_event_id_blank"] else 0
            if _parse_timestamp(row.get("timestamp")).date() > _parse_day(setup_day):
                lifecycle["survived_change_of_day"] = True
            if _is_outside_fill_window(row, session_start_hour, session_end_hour):
                lifecycle["out_of_window_fill_count"] += 1
            active_trades[setup_day] = trade
            if setup_day in pending_open:
                pending_open.remove(setup_day)
            continue

        if event_type == "trailing_modified":
            trade = active_trades.get(setup_day) or _latest_active_trade(active_trades)
            if trade is not None:
                trade["trailing_modifications"] += 1
                trade["trailing_activated"] = True
                if not trade["first_trailing_timestamp"]:
                    trade["first_trailing_timestamp"] = row.get("timestamp", "")
                    trade["time_to_first_trailing_seconds"] = _seconds_between(
                        trade["fill_timestamp"], trade["first_trailing_timestamp"]
                    )
                trade["assigned_blank_trailing_count"] += 1 if row["_raw_event_id_blank"] else 0
            lifecycle["trailing_modifications"] += 1
            continue

        if event_type == "exit":
            trade = active_trades.get(setup_day) or _latest_active_trade(active_trades)
            if trade is None:
                trade = _orphan_trade(row, lifecycle, len([item for item in trades if item["setup_day"] == setup_day]) + 1)
                trades.append(trade)
            _apply_exit_to_trade(trade, row)
            lifecycle["exit_count"] += 1
            lifecycle["last_exit_timestamp"] = row.get("timestamp", "")
            lifecycle["assigned_blank_exit_count"] += 1 if row["_raw_event_id_blank"] else 0
            if _parse_timestamp(row.get("timestamp")).date() > _parse_day(setup_day):
                lifecycle["survived_change_of_day"] = True
            active_trades.pop(setup_day, None)

    lifecycle_rows = [_finalize_lifecycle(row, close_hour) for row in lifecycles.values()]
    lifecycle_rows.sort(key=lambda row: row["setup_day"])
    trades.sort(key=lambda row: (row["fill_timestamp"] or row["exit_timestamp"] or "", row["setup_day"]))
    return {"lifecycles": lifecycle_rows, "trades": trades}


def build_anomaly_reports(
    rows: list[dict[str, Any]],
    lifecycle_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    session_start_hour: int,
    session_end_hour: int,
) -> dict[str, list[dict[str, Any]]]:
    by_event_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_event_day[row["_event_day"]].append(row)

    multi_exit_days = []
    for event_day, day_rows in sorted(by_event_day.items()):
        exits = [row for row in day_rows if row["event_type"] == "exit"]
        if len(exits) > 1:
            fills = [row for row in day_rows if row["event_type"] == "fill"]
            assigned_setups = sorted(
                {
                    trade["setup_day"]
                    for trade in trade_rows
                    if _event_day_from_text(trade.get("exit_timestamp")) == event_day
                }
            )
            multi_exit_days.append(
                {
                    "category": "multi_exit_day",
                    "event_day": event_day,
                    "exit_count": len(exits),
                    "fill_count": len(fills),
                    "assigned_setup_days": "|".join(assigned_setups),
                    "exit_timestamps": "|".join(row.get("timestamp", "") for row in exits),
                    "exit_reasons": "|".join(row.get("exit_reason", "") for row in exits),
                    "net_pnl_sum": _round(sum(_to_float(row.get("net_pnl")) or 0.0 for row in exits)),
                    "hypothesis": "previous GTC lifecycle plus same-day setup, or multiple fills from live pendings",
                }
            )

    out_of_window_fills = []
    for trade in trade_rows:
        if not trade.get("fill_timestamp"):
            continue
        fill_hour = _parse_timestamp(trade["fill_timestamp"]).hour
        if session_start_hour <= fill_hour <= session_end_hour:
            continue
        out_of_window_fills.append(
            {
                "category": "out_of_window_fill",
                "event_day": _event_day_from_text(trade.get("fill_timestamp")),
                "setup_day": trade.get("setup_day"),
                "fill_timestamp": trade.get("fill_timestamp"),
                "fill_hour": fill_hour,
                "side": trade.get("fill_side"),
                "fill_price": trade.get("fill_price"),
                "exit_timestamp": trade.get("exit_timestamp"),
                "exit_reason": trade.get("exit_reason"),
                "net_pnl": trade.get("net_pnl"),
                "assigned_from_blank_event_id": trade.get("assigned_from_blank_event_id"),
                "hypothesis": "pending GTC survived beyond intended intraday window",
            }
        )

    friday_exit_events = []
    for trade in trade_rows:
        if not trade.get("exit_timestamp"):
            continue
        exit_dt = _parse_timestamp(trade["exit_timestamp"])
        if exit_dt.weekday() == 4:
            setup = _find_lifecycle(lifecycle_rows, trade["setup_day"])
            if setup and setup.get("setup_event_type") == "blocked_friday":
                relation = "same_day_blocked_friday_setup"
            elif _event_day_from_text(trade.get("exit_timestamp")) != trade.get("setup_day"):
                relation = "previous_setup_lifecycle_closed_on_friday"
            else:
                relation = "friday_exit_after_friday_setup_or_unknown"
            friday_exit_events.append(
                {
                    "category": "friday_exit_event",
                    "event_day": _event_day_from_text(trade.get("exit_timestamp")),
                    "setup_day": trade.get("setup_day"),
                    "fill_timestamp": trade.get("fill_timestamp"),
                    "exit_timestamp": trade.get("exit_timestamp"),
                    "side": trade.get("fill_side"),
                    "exit_reason": trade.get("exit_reason"),
                    "net_pnl": trade.get("net_pnl"),
                    "relation": relation,
                    "hypothesis": "NoTradeFridays blocks new setup only; prior GTC/position can still resolve",
                }
            )

    cleanup_issues = []
    for lifecycle in lifecycle_rows:
        cleanup_risk = (
            lifecycle["order_placed"]
            and lifecycle["cleanup_count"] == 0
            and (
                lifecycle["lifecycle_status"] == "missing_cleanup_or_unresolved"
                or lifecycle["survived_change_of_day"]
                or lifecycle["out_of_window_fill_count"] > 0
            )
        )
        if cleanup_risk:
            cleanup_issues.append(
                {
                    "category": "open_order_cleanup_issue",
                    "setup_day": lifecycle["setup_day"],
                    "setup_timestamp": lifecycle["setup_timestamp"],
                    "side_placed": lifecycle["side_placed"],
                    "buy_entry": lifecycle["buy_entry"],
                    "sell_entry": lifecycle["sell_entry"],
                    "first_fill_timestamp": lifecycle["first_fill_timestamp"],
                    "last_exit_timestamp": lifecycle["last_exit_timestamp"],
                    "cleanup_count": lifecycle["cleanup_count"],
                    "survived_change_of_day": lifecycle["survived_change_of_day"],
                    "out_of_window_fill_count": lifecycle["out_of_window_fill_count"],
                    "status": lifecycle["lifecycle_status"],
                    "hypothesis": "missing cleanup log, GTC survival, or position/order state persisted past intended lifecycle",
                }
            )

    anomalous = multi_exit_days + out_of_window_fills + friday_exit_events + cleanup_issues
    anomalous.sort(key=lambda row: row.get("event_day") or row.get("setup_day") or "")
    return {
        "all": anomalous,
        "multi_exit_days": multi_exit_days,
        "out_of_window_fills": out_of_window_fills,
        "friday_exit_events": friday_exit_events,
        "open_order_cleanup_issues": cleanup_issues,
    }


def build_trailing_report(trade_rows: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [row for row in trade_rows if row.get("has_exit")]
    groups = {
        "trailing_activated": [row for row in closed if row.get("trailing_activated")],
        "trailing_not_activated": [row for row in closed if not row.get("trailing_activated")],
    }
    rows = []
    for group, trades in groups.items():
        rows.append(_trade_summary_row("trailing_status", group, trades))
    winners = [row for row in closed if (_to_float(row.get("net_pnl")) or 0.0) > 0]
    losers = [row for row in closed if (_to_float(row.get("net_pnl")) or 0.0) < 0]
    rows.append(_trade_summary_row("winner_trailing_dependency", "winners_with_trailing", [r for r in winners if r.get("trailing_activated")]))
    rows.append(_trade_summary_row("loss_tail_trailing", "losses_without_trailing", [r for r in losers if not r.get("trailing_activated")]))
    rows.append(_trade_summary_row("loss_tail_trailing", "losses_with_trailing", [r for r in losers if r.get("trailing_activated")]))
    rows.append(
        {
            "dimension": "causality_assessment",
            "group": "trailing",
            "trades": len(closed),
            "net_pnl": _round(sum(_to_float(row.get("net_pnl")) or 0.0 for row in closed)),
            "causality_level": "observational_correlation_only",
            "interpretation": (
                "The log shows strong association between trailing activation and profit, "
                "but this is path-dependent: trades activate trailing only after favorable movement."
            ),
        }
    )
    summary = {
        "closed_trades": len(closed),
        "trailing_activated_trades": len(groups["trailing_activated"]),
        "trailing_not_activated_trades": len(groups["trailing_not_activated"]),
        "winning_trades": len(winners),
        "winning_trades_with_trailing_pct": _pct(len([r for r in winners if r.get("trailing_activated")]), len(winners)),
        "losing_trades_without_trailing_pct": _pct(len([r for r in losers if not r.get("trailing_activated")]), len(losers)),
        "median_time_to_first_trailing_seconds": _safe_median(
            [_to_float(row.get("time_to_first_trailing_seconds")) for row in closed if row.get("trailing_activated")]
        ),
        "causality": "not_proven_observational_only",
    }
    return {"rows": rows, "summary": summary}


def build_loss_tail_report(trade_rows: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [row for row in trade_rows if row.get("has_exit")]
    losses = sorted(
        [row for row in closed if (_to_float(row.get("net_pnl")) or 0.0) < 0],
        key=lambda row: _to_float(row.get("net_pnl")) or 0.0,
    )
    total_loss_abs = abs(sum(_to_float(row.get("net_pnl")) or 0.0 for row in losses))
    rows = []
    cumulative = 0.0
    for rank, trade in enumerate(losses[:30], start=1):
        loss_abs = abs(_to_float(trade.get("net_pnl")) or 0.0)
        cumulative += loss_abs
        rows.append(
            {
                "rank": rank,
                "setup_day": trade.get("setup_day"),
                "fill_timestamp": trade.get("fill_timestamp"),
                "exit_timestamp": trade.get("exit_timestamp"),
                "side": trade.get("fill_side"),
                "net_pnl": trade.get("net_pnl"),
                "mfe_points": trade.get("mfe_points"),
                "mae_points": trade.get("mae_points"),
                "trailing_activated": trade.get("trailing_activated"),
                "exit_reason": trade.get("exit_reason"),
                "cumulative_loss_abs": _round(cumulative),
                "pct_of_total_loss_abs": _pct(loss_abs, total_loss_abs),
                "cumulative_pct_of_total_loss_abs": _pct(cumulative, total_loss_abs),
            }
        )
    pnl_values = [_to_float(row.get("net_pnl")) or 0.0 for row in closed]
    summary = {
        "closed_trades": len(closed),
        "loss_count": len(losses),
        "gross_loss_abs": _round(total_loss_abs),
        "worst_loss": _round(min((_to_float(row.get("net_pnl")) or 0.0 for row in losses), default=0.0)),
        "top_5_losses_abs": _round(sum(abs(_to_float(row.get("net_pnl")) or 0.0) for row in losses[:5])),
        "top_5_losses_pct_of_total_loss": _pct(
            sum(abs(_to_float(row.get("net_pnl")) or 0.0) for row in losses[:5]), total_loss_abs
        ),
        "max_drawdown_abs": _round(_max_drawdown(pnl_values)),
        "max_consecutive_losses": _max_consecutive_losses(pnl_values),
    }
    return {"rows": rows, "summary": summary}


def summarize_lifecycles(lifecycle_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "setup_lifecycles": len(lifecycle_rows),
        "order_placed_lifecycles": len([row for row in lifecycle_rows if row["order_placed"]]),
        "filled_lifecycles": len([row for row in lifecycle_rows if row["fill_count"] > 0]),
        "cleaned_no_fill_lifecycles": len([row for row in lifecycle_rows if row["lifecycle_status"] == "cancelled_no_fill"]),
        "missing_cleanup_lifecycles": len([row for row in lifecycle_rows if row["lifecycle_status"] == "missing_cleanup_or_unresolved"]),
        "overnight_survival_lifecycles": len([row for row in lifecycle_rows if row["survived_change_of_day"]]),
        "out_of_window_fill_lifecycles": len([row for row in lifecycle_rows if row["out_of_window_fill_count"] > 0]),
        "multi_fill_lifecycles": len([row for row in lifecycle_rows if row["fill_count"] > 1]),
        "multi_exit_lifecycles": len([row for row in lifecycle_rows if row["exit_count"] > 1]),
    }


def summarize_trades(trade_rows: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [row for row in trade_rows if row.get("has_exit")]
    pnl = [_to_float(row.get("net_pnl")) or 0.0 for row in closed]
    wins = [value for value in pnl if value > 0]
    losses = [value for value in pnl if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "closed_trades": len(closed),
        "net_pnl": _round(sum(pnl)),
        "win_rate": _pct(len(wins), len(closed)),
        "average_win": _round(mean(wins)) if wins else None,
        "average_loss": _round(mean(losses)) if losses else None,
        "payoff": _round((mean(wins) / abs(mean(losses))) if wins and losses else 0.0) if wins and losses else None,
        "profit_factor": _round(gross_profit / gross_loss) if gross_loss else None,
        "expectancy": _round(mean(pnl)) if pnl else None,
        "max_drawdown_abs": _round(_max_drawdown(pnl)),
        "max_consecutive_losses": _max_consecutive_losses(pnl),
    }


def recommended_risk_engine_rules(
    anomalies: dict[str, list[dict[str, Any]]],
    trailing: dict[str, Any],
    loss_tail: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "rule": "hard_cancel_all_pendings_at_close",
            "reason": "Out-of-window fills and Friday exits indicate GTC pending lifecycle risk.",
        },
        {
            "rule": "block_new_setups_if_any_position_or_pending_exists",
            "reason": "Multi-exit days show that overlapping lifecycle state can occur.",
        },
        {
            "rule": "daily_max_loss_and_consecutive_loss_kill_switch",
            "reason": "High win-rate/low-payoff profile is sensitive to isolated large losses.",
        },
        {
            "rule": "force_reconcile_orders_positions_before_setup",
            "reason": "Blank event_id events and missing cleanup days require broker-state reconciliation.",
        },
        {
            "rule": "separate_trailing_guardrails_from_entry_logic",
            "reason": "Trailing is strongly associated with profitability but remains tick/path-dependent.",
        },
        {
            "rule": "friday_no_new_trade_plus_close_or_cancel_prior_exposure",
            "reason": f"{len(anomalies['friday_exit_events'])} Friday exit events exist despite Friday setup block.",
        },
    ]


def economic_summary_markdown(report: dict[str, Any], anomalies: dict[str, list[dict[str, Any]]]) -> str:
    trade = report["trade_summary"]
    lifecycle = report["lifecycle_summary"]
    trailing = report["trailing_summary"]
    loss_tail = report["loss_tail_summary"]
    rules = "\n".join(
        f"- `{item['rule']}`: {item['reason']}" for item in report["risk_engine_rules_recommended"]
    )
    return (
        "# Azir Economic Audit\n\n"
        "## Executive Summary\n\n"
        "- Benchmark economico: NOT FROZEN.\n"
        f"- Closed trades in MT5 log: {trade['closed_trades']}.\n"
        f"- Net PnL: {trade['net_pnl']}; PF: {trade['profit_factor']}; expectancy: {trade['expectancy']}.\n"
        f"- Win rate: {trade['win_rate']}%; payoff: {trade['payoff']}.\n"
        f"- Order lifecycles: {lifecycle['setup_lifecycles']}; overnight survival lifecycles: "
        f"{lifecycle['overnight_survival_lifecycles']}.\n"
        f"- Out-of-window fills: {len(anomalies['out_of_window_fills'])}; multi-exit days: "
        f"{len(anomalies['multi_exit_days'])}; Friday exit events: {len(anomalies['friday_exit_events'])}; "
        f"open order cleanup issues: {len(anomalies['open_order_cleanup_issues'])}.\n\n"
        "## Order Lifecycle Findings\n\n"
        "- The event log supports setup/fill/exit reconstruction, but blank `event_id` rows show that some "
        "fills/trailing/exits belong to prior live state rather than the event day itself.\n"
        "- The most likely mechanism behind the anomalies is GTC pending survival plus cleanup/cancel logic "
        "depending on later ticks.\n"
        "- This is not necessarily a broken EA, but it is an operational risk that must be explicitly guarded "
        "before Risk Engine or PPO.\n\n"
        "## Trailing Findings\n\n"
        f"- Trades with trailing activation: {trailing['trailing_activated_trades']}.\n"
        f"- Winning trades with trailing: {trailing['winning_trades_with_trailing_pct']}%.\n"
        f"- Losing trades without trailing: {trailing['losing_trades_without_trailing_pct']}%.\n"
        f"- Median time to first trailing: {trailing['median_time_to_first_trailing_seconds']} seconds.\n"
        "- This is strong observational correlation, not causal proof: trailing activates only after favorable "
        "movement, so a no-trailing group is naturally adverse-selected.\n\n"
        "## Loss Tail Findings\n\n"
        f"- Gross loss abs: {loss_tail['gross_loss_abs']}.\n"
        f"- Worst loss: {loss_tail['worst_loss']}.\n"
        f"- Top 5 losses account for {loss_tail['top_5_losses_pct_of_total_loss']}% of gross loss.\n"
        f"- Max consecutive losses: {loss_tail['max_consecutive_losses']}.\n"
        "- Azir's economics are high-win-rate and low-payoff; the Risk Engine should be designed around "
        "protecting tail losses and stale order exposure.\n\n"
        "## Decision\n\n"
        "- Economic benchmark cannot be frozen yet.\n"
        "- We are ready to design a Risk Engine, but it must be driven by lifecycle guards, not only PnL metrics.\n"
        "- PPO is still premature until execution economics are approximated and protected consistently.\n\n"
        "## Risk Engine Rules Recommended\n\n"
        f"{rules}\n"
    )


def _new_lifecycle(setup_day: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "setup_day": setup_day,
        "setup_timestamp": row.get("timestamp", ""),
        "setup_event_type": row.get("event_type", ""),
        "is_friday": _is_true(row.get("is_friday")) or row.get("event_type") == "blocked_friday",
        "opportunity_rows": 0,
        "buy_order_placed": False,
        "sell_order_placed": False,
        "side_placed": "",
        "order_placed": False,
        "buy_entry": "",
        "sell_entry": "",
        "pending_distance_points": "",
        "atr_points": "",
        "trend_bias": "",
        "first_order_timestamp": "",
        "cleanup_timestamp": "",
        "cleanup_count": 0,
        "fill_count": 0,
        "exit_count": 0,
        "first_fill_timestamp": "",
        "last_fill_timestamp": "",
        "last_exit_timestamp": "",
        "trailing_modifications": 0,
        "assigned_blank_fill_count": 0,
        "assigned_blank_exit_count": 0,
        "survived_change_of_day": False,
        "out_of_window_fill_count": 0,
    }


def _update_lifecycle_from_setup(lifecycle: dict[str, Any], row: dict[str, Any]) -> None:
    if row.get("event_type") == "opportunity":
        lifecycle["opportunity_rows"] += 1
    lifecycle["setup_event_type"] = row.get("event_type", lifecycle["setup_event_type"])
    lifecycle["setup_timestamp"] = row.get("timestamp", lifecycle["setup_timestamp"])
    lifecycle["is_friday"] = _is_true(row.get("is_friday")) or row.get("event_type") == "blocked_friday"
    buy = _is_true(row.get("buy_order_placed"))
    sell = _is_true(row.get("sell_order_placed"))
    if buy or sell:
        lifecycle["buy_order_placed"] = buy
        lifecycle["sell_order_placed"] = sell
        lifecycle["order_placed"] = True
        lifecycle["first_order_timestamp"] = lifecycle["first_order_timestamp"] or row.get("timestamp", "")
        lifecycle["buy_entry"] = row.get("buy_entry", "")
        lifecycle["sell_entry"] = row.get("sell_entry", "")
        lifecycle["pending_distance_points"] = row.get("pending_distance_points", "")
        lifecycle["atr_points"] = row.get("atr_points", "")
        lifecycle["side_placed"] = "both" if buy and sell else "buy" if buy else "sell"
        lifecycle["trend_bias"] = lifecycle["side_placed"]


def _new_trade(row: dict[str, Any], lifecycle: dict[str, Any], sequence: int) -> dict[str, Any]:
    return {
        "setup_day": lifecycle["setup_day"],
        "trade_sequence": sequence,
        "setup_timestamp": lifecycle["setup_timestamp"],
        "side_placed": lifecycle["side_placed"],
        "fill_timestamp": row.get("timestamp", ""),
        "fill_event_day": row["_event_day"],
        "fill_hour": row["_timestamp_dt"].hour,
        "fill_side": row.get("fill_side", ""),
        "fill_price": row.get("fill_price", ""),
        "duration_to_fill_seconds": row.get("duration_to_fill_seconds", ""),
        "assigned_from_blank_event_id": False,
        "assignment_reason": "",
        "trailing_activated": False,
        "trailing_modifications": 0,
        "first_trailing_timestamp": "",
        "time_to_first_trailing_seconds": "",
        "assigned_blank_trailing_count": 0,
        "exit_timestamp": "",
        "exit_event_day": "",
        "exit_reason": "",
        "gross_pnl": "",
        "net_pnl": "",
        "commission": "",
        "swap": "",
        "mfe_points": "",
        "mae_points": "",
        "assigned_blank_exit_count": 0,
        "has_exit": False,
    }


def _orphan_trade(row: dict[str, Any], lifecycle: dict[str, Any], sequence: int) -> dict[str, Any]:
    trade = _new_trade(row, lifecycle, sequence)
    trade["fill_timestamp"] = ""
    trade["fill_event_day"] = ""
    trade["fill_hour"] = ""
    trade["assignment_reason"] = "orphan_exit_without_active_fill"
    return trade


def _apply_exit_to_trade(trade: dict[str, Any], row: dict[str, Any]) -> None:
    trade["exit_timestamp"] = row.get("timestamp", "")
    trade["exit_event_day"] = row["_event_day"]
    trade["exit_reason"] = row.get("exit_reason", "")
    trade["gross_pnl"] = row.get("gross_pnl", "")
    trade["net_pnl"] = row.get("net_pnl", "")
    trade["commission"] = row.get("commission", "")
    trade["swap"] = row.get("swap", "")
    trade["mfe_points"] = row.get("mfe_points", "")
    trade["mae_points"] = row.get("mae_points", "")
    trade["assigned_blank_exit_count"] += 1 if row["_raw_event_id_blank"] else 0
    if _is_true(row.get("trailing_activated")):
        trade["trailing_activated"] = True
    trade["has_exit"] = True


def _finalize_lifecycle(row: dict[str, Any], close_hour: int) -> dict[str, Any]:
    terminal = row["cleanup_timestamp"] or row["first_fill_timestamp"] or row["last_exit_timestamp"]
    row["pending_alive_seconds_to_first_fill_or_cleanup"] = (
        "" if not row["first_order_timestamp"] or not terminal else _seconds_between(row["first_order_timestamp"], terminal)
    )
    if row["setup_event_type"] == "blocked_friday":
        row["lifecycle_status"] = "blocked_friday"
    elif not row["order_placed"]:
        row["lifecycle_status"] = "no_order"
    elif row["fill_count"] > 0:
        row["lifecycle_status"] = "filled"
    elif row["cleanup_count"] > 0:
        row["lifecycle_status"] = "cancelled_no_fill"
    else:
        row["lifecycle_status"] = "missing_cleanup_or_unresolved"
    row["survived_past_close_hour"] = _survived_past_close(row, close_hour)
    row["hypothesis"] = _lifecycle_hypothesis(row)
    return row


def _assign_setup_day(
    row: dict[str, Any],
    lifecycles: dict[str, dict[str, Any]],
    pending_open: list[str],
    active_trades: dict[str, dict[str, Any]],
) -> str:
    event_id_day = row.get("_event_id_day")
    if event_id_day and event_id_day in lifecycles and not row.get("_raw_event_id_blank"):
        return event_id_day
    if row.get("event_type") == "fill" and pending_open:
        return pending_open[-1]
    if row.get("event_type") in {"trailing_modified", "exit"}:
        if active_trades:
            return max(active_trades.values(), key=lambda item: item.get("fill_timestamp", "")).get("setup_day")
    return event_id_day or row["_event_day"]


def _assignment_reason(row: dict[str, Any], setup_day: str, event_day: str) -> str:
    if not row.get("_raw_event_id_blank") and row.get("_event_id_day") == setup_day:
        return "event_id_day_match"
    if setup_day != event_day:
        return "blank_event_id_assigned_to_prior_pending_lifecycle"
    if row.get("_raw_event_id_blank"):
        return "blank_event_id_assigned_to_event_day"
    return "fallback_event_day"


def _latest_active_trade(active_trades: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not active_trades:
        return None
    return max(active_trades.values(), key=lambda item: item.get("fill_timestamp", ""))


def _is_outside_fill_window(row: dict[str, Any], session_start_hour: int, session_end_hour: int) -> bool:
    hour = row["_timestamp_dt"].hour
    return not (session_start_hour <= hour <= session_end_hour)


def _survived_past_close(row: dict[str, Any], close_hour: int) -> bool:
    for field in ("first_fill_timestamp", "last_exit_timestamp"):
        if row.get(field):
            timestamp = _parse_timestamp(row[field])
            if timestamp.date() > _parse_day(row["setup_day"]) or timestamp.hour >= close_hour:
                return True
    return False


def _lifecycle_hypothesis(row: dict[str, Any]) -> str:
    if row["lifecycle_status"] == "blocked_friday":
        return "friday_setup_blocked_no_new_orders"
    if row["lifecycle_status"] == "no_order":
        return "filters_or_order_send_prevented_pending"
    if row["lifecycle_status"] == "cancelled_no_fill":
        return "pending_cancelled_by_logged_cleanup"
    if row["lifecycle_status"] == "missing_cleanup_or_unresolved":
        return "missing_cleanup_log_or_pending_state_not_resolved_in_events"
    if row["survived_change_of_day"] or row["out_of_window_fill_count"]:
        return "gtc_pending_or_position_survived_beyond_intraday_window"
    if row["fill_count"] > 1 or row["exit_count"] > 1:
        return "multiple_fills_or_exits_from_same_setup_lifecycle"
    return "same_day_lifecycle"


def _trade_summary_row(dimension: str, group: str, trades: list[dict[str, Any]]) -> dict[str, Any]:
    pnl = [_to_float(row.get("net_pnl")) or 0.0 for row in trades if row.get("has_exit")]
    wins = [value for value in pnl if value > 0]
    losses = [value for value in pnl if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    times = [_to_float(row.get("time_to_first_trailing_seconds")) for row in trades]
    times = [value for value in times if value is not None]
    return {
        "dimension": dimension,
        "group": group,
        "trades": len(pnl),
        "net_pnl": _round(sum(pnl)),
        "win_rate": _pct(len(wins), len(pnl)),
        "average_win": _round(mean(wins)) if wins else None,
        "average_loss": _round(mean(losses)) if losses else None,
        "profit_factor": _round(gross_profit / gross_loss) if gross_loss else None,
        "expectancy": _round(mean(pnl)) if pnl else None,
        "max_loss": _round(min(pnl)) if pnl else None,
        "median_time_to_first_trailing_seconds": _safe_median(times),
        "causality_level": "observational_grouping_only",
        "interpretation": "Do not interpret as causal without counterfactual/tick replay.",
    }


def _find_lifecycle(lifecycle_rows: list[dict[str, Any]], setup_day: str) -> dict[str, Any] | None:
    return next((row for row in lifecycle_rows if row["setup_day"] == setup_day), None)


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _event_day(row: dict[str, Any]) -> str:
    return _event_day_from_text(row.get("timestamp")) or "unknown_date"


def _event_day_from_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split(" ")[0].replace(".", "-")


def _event_id_day(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split("_")[0].replace(".", "-")


def _parse_timestamp(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.combine(date.min, datetime.min.time())
    text = text.replace(".", "-").replace("T", " ")
    if len(text) == 10:
        text = f"{text} 00:00:00"
    return datetime.fromisoformat(text)


def _parse_day(value: Any) -> date:
    return _parse_timestamp(str(value)[:10]).date()


def _seconds_between(start: Any, end: Any) -> int:
    return int((_parse_timestamp(end) - _parse_timestamp(start)).total_seconds())


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _pct(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator * 100.0, 4)


def _safe_median(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return _round(median(clean)) if clean else None


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


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "economic_benchmark_can_be_frozen": report["decision"]["economic_benchmark_can_be_frozen"],
        "ready_for_risk_engine_design": report["decision"]["ready_for_risk_engine_design"],
        "ready_for_ppo": report["decision"]["ready_for_ppo"],
        "trade_summary": report["trade_summary"],
        "anomaly_summary": report["anomaly_summary"],
        "recommended_next_sprint": report["decision"]["recommended_next_sprint"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
