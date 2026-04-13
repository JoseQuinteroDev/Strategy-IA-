"""Freeze candidate for Azir protected economics after forced-close repricing."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

from hybrid_quant.risk.azir_state import AzirRiskConfig

from .economic_audit import (
    _read_raw_event_log,
    _round,
    _to_float,
    _write_csv,
    reconstruct_lifecycles,
    summarize_trades,
)
from .replica import OhlcvBar, load_ohlcv_csv
from .risk_reaudit import apply_risk_engine_to_lifecycle


DEFAULT_SYMBOL = "XAUUSD-STD"
BENCHMARK_NAME = "baseline_azir_protected_economic_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reprice Azir forced closes and freeze protected benchmark candidate.")
    parser.add_argument("--mt5-log-path", required=True, help="Canonical Azir MT5 event log CSV.")
    parser.add_argument("--m1-input-path", required=True, help="XAUUSD M1 OHLCV CSV with internal schema.")
    parser.add_argument("--output-dir", required=True, help="Directory for freeze artifacts.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--lot-size", type=float, default=0.10)
    parser.add_argument("--contract-size", type=float, default=100.0)
    parser.add_argument("--config-name", default="risk_engine_azir_v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_protected_benchmark_freeze(
        mt5_log_path=Path(args.mt5_log_path),
        m1_input_path=Path(args.m1_input_path),
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
        lot_size=args.lot_size,
        contract_size=args.contract_size,
        config=AzirRiskConfig(name=args.config_name),
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_protected_benchmark_freeze(
    *,
    mt5_log_path: Path,
    m1_input_path: Path,
    output_dir: Path,
    symbol: str = DEFAULT_SYMBOL,
    lot_size: float = 0.10,
    contract_size: float = 100.0,
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
    simulation = apply_risk_engine_to_lifecycle(
        rows=rows,
        lifecycle_rows=reconstruction["lifecycles"],
        trade_rows=reconstruction["trades"],
        config=config,
    )
    bars = load_ohlcv_csv(m1_input_path)
    forced_decisions = [row for row in simulation["trade_decisions"] if row["risk_status"] == "forced_close_unpriced"]
    forced_cases = [
        revalue_forced_close(
            decision=row,
            bars=bars,
            config=config,
            lot_size=lot_size,
            contract_size=contract_size,
        )
        for row in forced_decisions
    ]
    revalued_trades = _build_revalued_trades(simulation["protected_trades"], reconstruction["trades"], forced_cases)

    original_metrics = summarize_trades(reconstruction["trades"])
    reaudited_known_metrics = summarize_trades(simulation["protected_trades"])
    revalued_metrics = summarize_trades(revalued_trades)
    comparison_rows = _comparison_rows(original_metrics, reaudited_known_metrics, revalued_metrics)
    freeze_decision = _freeze_decision(forced_cases, revalued_metrics)
    report = {
        "benchmark_name": BENCHMARK_NAME,
        "source_log": str(mt5_log_path),
        "m1_input_path": str(m1_input_path),
        "symbol": symbol,
        "risk_engine": asdict(config),
        "pricing_convention": {
            "preferred_source": "M1 OHLCV",
            "base_rule": "Use the close of the latest fully closed M1 bar before the configured Risk Engine close hour.",
            "fallback_rule": "If no pre-close M1 bar exists after fill, the case remains unpriced and benchmark cannot freeze.",
            "sensitivity_rule": "Report OHLC of selected bar plus first available post-close M1 open/close where present.",
            "lot_size": lot_size,
            "contract_size": contract_size,
            "commission_policy": "Use zero additional commission/swap because the observed MT5 close rows for these cases logged 0.00.",
        },
        "forced_close_cases": forced_cases,
        "metrics": {
            "azir_original_observed": original_metrics,
            "azir_with_risk_engine_v1_reaudited_known_pnl": reaudited_known_metrics,
            "azir_with_risk_engine_v1_forced_closes_revalued": revalued_metrics,
        },
        "comparison": comparison_rows,
        "decision": freeze_decision,
        "limitations": [
            "No tick exists exactly at 22:00 in the M1 file for the two forced-close days.",
            "The selected price is the last fully closed M1 quote before the operational close target, not a broker-confirmed close ticket.",
            "The uncertainty band is explicitly reported and does not include unknown broker execution latency.",
            "Azir entry/trailing logic remains unchanged.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(forced_cases, output_dir / "forced_close_cases.csv")
    _write_csv(_metric_rows(revalued_metrics), output_dir / "protected_benchmark_metrics.csv")
    _write_csv(comparison_rows, output_dir / "original_vs_protected_vs_revalued.csv")
    (output_dir / "forced_close_revaluation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "protected_benchmark_freeze_summary.md").write_text(
        _summary_markdown(report),
        encoding="utf-8",
    )
    return report


def revalue_forced_close(
    *,
    decision: dict[str, Any],
    bars: list[OhlcvBar],
    config: AzirRiskConfig,
    lot_size: float,
    contract_size: float,
) -> dict[str, Any]:
    fill_timestamp = str(decision["fill_timestamp"])
    setup_day = str(decision["setup_day"])
    fill_dt = _parse_timestamp_like(fill_timestamp)
    target_close_dt = datetime.combine(fill_dt.date(), time(config.close_hour, 0))
    candidates = [bar for bar in bars if fill_dt <= bar.open_time < target_close_dt and bar.open_time.date() == fill_dt.date()]
    if not candidates:
        return _unpriced_case(decision, "no_m1_bar_after_fill_before_risk_close")

    selected = candidates[-1]
    selected_close_time = selected.open_time + timedelta(minutes=1)
    next_bar = next((bar for bar in bars if bar.open_time >= target_close_dt), None)
    side = str(decision["fill_side"])
    entry = _to_float(decision.get("fill_price"))
    if entry is None:
        return _unpriced_case(decision, "missing_fill_price")

    base_gross = _pnl(side, entry, selected.close, lot_size, contract_size)
    sensitivity = {
        "selected_bar_open_pnl": _round(_pnl(side, entry, selected.open, lot_size, contract_size)),
        "selected_bar_high_pnl": _round(_pnl(side, entry, selected.high, lot_size, contract_size)),
        "selected_bar_low_pnl": _round(_pnl(side, entry, selected.low, lot_size, contract_size)),
        "selected_bar_close_pnl": _round(base_gross),
        "next_available_open_pnl": _round(_pnl(side, entry, next_bar.open, lot_size, contract_size)) if next_bar else "",
        "next_available_close_pnl": _round(_pnl(side, entry, next_bar.close, lot_size, contract_size)) if next_bar else "",
    }
    numeric_sensitivity = [value for value in sensitivity.values() if isinstance(value, float)]
    return {
        "setup_day": setup_day,
        "fill_timestamp": fill_timestamp,
        "fill_side": side,
        "entry_price": entry,
        "observed_exit_timestamp": decision.get("exit_timestamp", ""),
        "observed_net_pnl": decision.get("observed_net_pnl", ""),
        "risk_close_target_timestamp": target_close_dt.isoformat(sep=" "),
        "selected_m1_bar_open_time": selected.open_time.isoformat(sep=" "),
        "selected_m1_bar_close_time": selected_close_time.isoformat(sep=" "),
        "selected_open": selected.open,
        "selected_high": selected.high,
        "selected_low": selected.low,
        "selected_close": selected.close,
        "next_available_bar_open_time": next_bar.open_time.isoformat(sep=" ") if next_bar else "",
        "next_available_open": next_bar.open if next_bar else "",
        "next_available_close": next_bar.close if next_bar else "",
        "minutes_between_selected_bar_close_and_risk_close": _round(
            (target_close_dt - selected_close_time).total_seconds() / 60.0
        ),
        "price_source": "m1_last_closed_bar_before_risk_close",
        "revalued_exit_price": selected.close,
        "revalued_gross_pnl": _round(base_gross),
        "revalued_net_pnl": _round(base_gross),
        "commission_assumption": 0.0,
        "swap_assumption": 0.0,
        "sensitivity_min_pnl": min(numeric_sensitivity) if numeric_sensitivity else "",
        "sensitivity_max_pnl": max(numeric_sensitivity) if numeric_sensitivity else "",
        "sensitivity_range_pnl": _round(max(numeric_sensitivity) - min(numeric_sensitivity)) if numeric_sensitivity else "",
        **sensitivity,
        "revaluation_status": "priced_with_m1_proxy",
        "uncertainty_note": (
            "No M1 bar exists exactly at risk close. Base price uses latest closed M1 before close; "
            "sensitivity includes selected-bar OHLC and first post-close quote."
        ),
    }


def _build_revalued_trades(
    protected_trades: list[dict[str, Any]],
    original_trades: list[dict[str, Any]],
    forced_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    originals = {(row.get("setup_day"), row.get("fill_timestamp")): row for row in original_trades}
    trades = [dict(row) for row in protected_trades]
    for case in forced_cases:
        if case["revaluation_status"] != "priced_with_m1_proxy":
            continue
        original = dict(originals[(case["setup_day"], case["fill_timestamp"])])
        original["exit_timestamp"] = case["risk_close_target_timestamp"].replace("-", ".")
        original["exit_event_day"] = case["setup_day"]
        original["exit_reason"] = "risk_engine_forced_close_revalued"
        original["gross_pnl"] = case["revalued_gross_pnl"]
        original["net_pnl"] = case["revalued_net_pnl"]
        original["commission"] = "0.00"
        original["swap"] = "0.00"
        original["has_exit"] = True
        original["risk_status"] = "forced_close_revalued"
        original["price_source"] = case["price_source"]
        trades.append(original)
    trades.sort(key=lambda row: row.get("fill_timestamp") or row.get("exit_timestamp") or "")
    return trades


def _comparison_rows(
    original: dict[str, Any],
    reaudited: dict[str, Any],
    revalued: dict[str, Any],
) -> list[dict[str, Any]]:
    metrics = [
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
    return [
        {
            "metric": metric,
            "azir_original_observed": original.get(metric),
            "azir_with_risk_engine_v1_reaudited_known_pnl": reaudited.get(metric),
            "azir_with_risk_engine_v1_forced_closes_revalued": revalued.get(metric),
            "delta_revalued_vs_original": _delta(original.get(metric), revalued.get(metric)),
            "delta_revalued_vs_reaudited_known": _delta(reaudited.get(metric), revalued.get(metric)),
        }
        for metric in metrics
    ]


def _freeze_decision(forced_cases: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    all_priced = all(row["revaluation_status"] == "priced_with_m1_proxy" for row in forced_cases)
    net_positive = (metrics.get("net_pnl") or 0.0) > 0.0
    pf_ok = (metrics.get("profit_factor") or 0.0) > 1.0
    sensitivity_worst_total = sum(
        _to_float(row.get("sensitivity_min_pnl")) or 0.0
        for row in forced_cases
        if row["revaluation_status"] == "priced_with_m1_proxy"
    )
    base_total = sum(
        _to_float(row.get("revalued_net_pnl")) or 0.0
        for row in forced_cases
        if row["revaluation_status"] == "priced_with_m1_proxy"
    )
    uncertainty_abs = abs(base_total - sensitivity_worst_total)
    return {
        "benchmark_name": BENCHMARK_NAME,
        "can_freeze_baseline_azir_protected_economic_v1": bool(all_priced and net_positive and pf_ok),
        "ready_for_rl_environment_design": bool(all_priced and net_positive and pf_ok),
        "ready_for_ppo_training": False,
        "recommended_next_sprint": "design_rl_env_for_azir_v1" if all_priced and net_positive and pf_ok else "additional_economic_pricing_audit",
        "reason": (
            "All forced-close cases are revalued with an explicit M1 proxy and the protected benchmark remains profitable."
            if all_priced and net_positive and pf_ok
            else "At least one forced-close case remains unpriced or the protected economics are not strong enough."
        ),
        "forced_close_cases": len(forced_cases),
        "forced_close_cases_priced": len([row for row in forced_cases if row["revaluation_status"] == "priced_with_m1_proxy"]),
        "forced_close_base_total_pnl": _round(base_total),
        "forced_close_worst_sensitivity_total_pnl": _round(sensitivity_worst_total),
        "forced_close_uncertainty_abs_vs_worst_sensitivity": _round(uncertainty_abs),
    }


def _metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"metric": key, "value": value} for key, value in metrics.items()]


def _summary_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    metrics = report["metrics"]["azir_with_risk_engine_v1_forced_closes_revalued"]
    cases = "\n".join(
        "- "
        f"{case['setup_day']} {case['fill_side']} entry={case['entry_price']} "
        f"price={case['revalued_exit_price']} pnl={case['revalued_net_pnl']} "
        f"source={case['price_source']} gap_min={case['minutes_between_selected_bar_close_and_risk_close']} "
        f"sensitivity=[{case['sensitivity_min_pnl']}, {case['sensitivity_max_pnl']}]."
        for case in report["forced_close_cases"]
    )
    return (
        "# Protected Azir Economic Benchmark Freeze\n\n"
        "## Executive Summary\n\n"
        f"- Benchmark: `{BENCHMARK_NAME}`.\n"
        f"- Can freeze: {decision['can_freeze_baseline_azir_protected_economic_v1']}.\n"
        f"- Ready for RL environment design: {decision['ready_for_rl_environment_design']}.\n"
        f"- Ready for PPO training: {decision['ready_for_ppo_training']}.\n"
        f"- Net PnL protected/revalued: {metrics['net_pnl']}.\n"
        f"- Profit factor protected/revalued: {metrics['profit_factor']}.\n"
        f"- Expectancy protected/revalued: {metrics['expectancy']}.\n"
        f"- Max DD abs protected/revalued: {metrics['max_drawdown_abs']}.\n\n"
        "## Forced Close Revaluation\n\n"
        f"{cases}\n\n"
        "## Pricing Convention\n\n"
        "- Source: M1 OHLCV.\n"
        "- Base rule: close of the latest fully closed M1 bar before the configured 22:00 Risk Engine close.\n"
        "- Sensitivity: selected-bar OHLC plus first available post-close M1 quote.\n"
        "- Commission/swap: 0.00, matching the observed MT5 close rows for these two cases.\n\n"
        "## Decision\n\n"
        f"- {decision['reason']}\n"
        f"- Recommended next sprint: `{decision['recommended_next_sprint']}`.\n\n"
        "## Caveat\n\n"
        "This is now freezeable as a protected economic benchmark with an explicit M1 pricing convention. "
        "It is still not a tick-exact replication of MT5 trailing/execution internals.\n"
    )


def _unpriced_case(decision: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "setup_day": decision.get("setup_day", ""),
        "fill_timestamp": decision.get("fill_timestamp", ""),
        "fill_side": decision.get("fill_side", ""),
        "entry_price": decision.get("fill_price", ""),
        "observed_exit_timestamp": decision.get("exit_timestamp", ""),
        "observed_net_pnl": decision.get("observed_net_pnl", ""),
        "revaluation_status": "unpriced",
        "unpriced_reason": reason,
    }


def _pnl(side: str, entry: float, exit_price: float, lot_size: float, contract_size: float) -> float:
    direction = 1.0 if side == "buy" else -1.0
    return (exit_price - entry) * direction * lot_size * contract_size


def _parse_timestamp_like(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace(".", "-"))


def _delta(before: Any, after: Any) -> float | str:
    before_float = _to_float(before)
    after_float = _to_float(after)
    if before_float is None or after_float is None:
        return ""
    return _round(after_float - before_float)


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_name": report["benchmark_name"],
        "can_freeze": report["decision"]["can_freeze_baseline_azir_protected_economic_v1"],
        "ready_for_rl_environment_design": report["decision"]["ready_for_rl_environment_design"],
        "ready_for_ppo_training": report["decision"]["ready_for_ppo_training"],
        "metrics": report["metrics"]["azir_with_risk_engine_v1_forced_closes_revalued"],
        "forced_close_cases": report["forced_close_cases"],
        "recommended_next_sprint": report["decision"]["recommended_next_sprint"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
