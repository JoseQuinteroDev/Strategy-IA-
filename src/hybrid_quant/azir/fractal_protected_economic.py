"""Protected economic reconstruction for the Azir fractal setup candidate."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from hybrid_quant.risk.azir_state import AzirRiskConfig

from .economic_audit import _round, _to_float, _write_csv, summarize_trades
from .fractal_candidate_export import CANDIDATE_NAME, CANDIDATE_VARIANT
from .fractal_candidate_real_mt5 import read_canonical_event_log
from .replica import AzirReplicaConfig, OhlcvBar, load_ohlcv_csv


SPRINT_NAME = "protected_economic_candidate_fractal_v1"
DEFAULT_SYMBOL = "XAUUSD-STD"
OUTPUT_BENCHMARK_NAME = "baseline_azir_protected_economic_candidate_fractal_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate protected economics for the Azir fractal candidate.")
    parser.add_argument("--current-log-path", required=True, help="Observed current Azir MT5 event log.")
    parser.add_argument("--candidate-log-path", required=True, help="Fractal candidate MT5 setup event log.")
    parser.add_argument("--m5-input-path", required=True, help="XAUUSD M5 OHLCV CSV.")
    parser.add_argument("--m1-input-path", required=True, help="XAUUSD M1 OHLCV CSV.")
    parser.add_argument("--protected-report-path", required=True, help="Frozen protected benchmark report JSON.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tick-input-path", default="", help="Optional tick CSV; inspected but not replayed in this sprint.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--lot-size", type=float, default=0.10)
    parser.add_argument("--contract-size", type=float, default=100.0)
    parser.add_argument("--config-name", default="risk_engine_azir_v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_fractal_protected_economic_candidate(
        current_log_path=Path(args.current_log_path),
        candidate_log_path=Path(args.candidate_log_path),
        m5_input_path=Path(args.m5_input_path),
        m1_input_path=Path(args.m1_input_path),
        protected_report_path=Path(args.protected_report_path),
        output_dir=Path(args.output_dir),
        tick_input_path=Path(args.tick_input_path) if args.tick_input_path else None,
        symbol=args.symbol,
        lot_size=args.lot_size,
        contract_size=args.contract_size,
        risk_config=AzirRiskConfig(name=args.config_name),
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_fractal_protected_economic_candidate(
    *,
    current_log_path: Path,
    candidate_log_path: Path,
    m5_input_path: Path,
    m1_input_path: Path,
    protected_report_path: Path,
    output_dir: Path,
    tick_input_path: Path | None = None,
    symbol: str = DEFAULT_SYMBOL,
    lot_size: float = 0.10,
    contract_size: float = 100.0,
    risk_config: AzirRiskConfig | None = None,
) -> dict[str, Any]:
    risk_config = risk_config or AzirRiskConfig()
    candidate_rows = read_canonical_event_log(candidate_log_path, symbol_filter=symbol)
    current_rows = read_canonical_event_log(current_log_path, symbol_filter=symbol)
    if not candidate_rows:
        raise ValueError(f"No fractal candidate rows found for symbol {symbol}.")
    if not current_rows:
        raise ValueError(f"No current Azir rows found for symbol {symbol}.")

    m1_bars = load_ohlcv_csv(m1_input_path)
    m5_bars = load_ohlcv_csv(m5_input_path)
    replay_config = AzirReplicaConfig(symbol=symbol, lot_size=lot_size, contract_size=contract_size)
    candidate_lifecycle = replay_candidate_lifecycle(
        candidate_rows=candidate_rows,
        m1_by_day=_bars_by_day(m1_bars),
        m5_by_day=_bars_by_day(m5_bars),
        replay_config=replay_config,
        risk_config=risk_config,
    )
    current_replay_lifecycle = replay_candidate_lifecycle(
        candidate_rows=current_rows,
        m1_by_day=_bars_by_day(m1_bars),
        m5_by_day=_bars_by_day(m5_bars),
        replay_config=replay_config,
        risk_config=risk_config,
    )
    candidate_trades = [row for row in candidate_lifecycle if row.get("has_exit")]
    current_replay_trades = [row for row in current_replay_lifecycle if row.get("has_exit")]
    candidate_metrics = summarize_trades(candidate_trades)
    current_replay_metrics = summarize_trades(current_replay_trades)
    current_reference = load_current_protected_reference(protected_report_path)
    coverage = build_coverage_report(candidate_rows, candidate_lifecycle, m1_bars, m5_bars, tick_input_path)
    anomaly_summary = build_anomaly_summary(candidate_lifecycle)
    decision = build_decision(candidate_metrics, current_reference["metrics"], coverage, anomaly_summary, current_replay_metrics)
    comparison_rows = build_comparison_rows(
        current_reference["metrics"], candidate_metrics, "baseline_azir_protected_economic_v1"
    )
    same_replay_comparison_rows = build_comparison_rows(
        current_replay_metrics, candidate_metrics, "azir_current_same_replay_control"
    )
    forced_close_rows = [row for row in candidate_lifecycle if row.get("forced_close") == "true"]

    report = {
        "sprint": SPRINT_NAME,
        "candidate_name": CANDIDATE_NAME,
        "candidate_variant": CANDIDATE_VARIANT,
        "candidate_benchmark_name": OUTPUT_BENCHMARK_NAME,
        "sources": {
            "current_log_path": str(current_log_path),
            "candidate_log_path": str(candidate_log_path),
            "m5_input_path": str(m5_input_path),
            "m1_input_path": str(m1_input_path),
            "tick_input_path": str(tick_input_path) if tick_input_path else "",
            "protected_report_path": str(protected_report_path),
            "symbol": symbol,
        },
        "risk_engine": asdict(risk_config),
        "pricing_methodology": {
            "setup_source": "Real MT5 fractal candidate setup log.",
            "fill_source": "Replay from candidate pending levels using M1 when available, otherwise M5 fallback.",
            "exit_source": "M1/M5 OHLC replay with Azir SL/TP/trailing parameters and Risk Engine close-hour guard.",
            "tick_source": "Tick CSV is inspected for availability, but not consumed for full candidate replay in this sprint.",
            "same_bar_conflict_policy": "Conservative: stop before target; dual pending hits choose nearest entry to bar open.",
            "session_close_policy": "Open positions are closed at configured Risk Engine close hour using the close bar open.",
            "lot_size": lot_size,
            "contract_size": contract_size,
            "commission_policy": "Zero extra commission/swap, matching the existing Python protected benchmark convention.",
        },
        "current_protected_reference": current_reference,
        "current_same_replay_metrics": current_replay_metrics,
        "candidate_metrics": candidate_metrics,
        "comparison": comparison_rows,
        "same_replay_comparison": same_replay_comparison_rows,
        "coverage": coverage,
        "anomaly_summary": anomaly_summary,
        "forced_close_count": len(forced_close_rows),
        "decision": decision,
        "limitations": [
            "Candidate economics are replay estimates, not direct MT5 fill/exit evidence.",
            "Tick CSV is not used to price candidate trades in this sprint; tick-level promotion remains separate.",
            "M5 fallback before M1 coverage is materially less precise than M1.",
            "Trailing remains OHLC-path approximated, not broker tick-perfect.",
            "No new Azir logic, Risk Engine logic, or trailing behavior is introduced here.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(candidate_lifecycle, output_dir / "fractal_protected_lifecycle.csv")
    _write_csv(candidate_trades, output_dir / "fractal_protected_trades.csv")
    _write_csv(comparison_rows, output_dir / "current_vs_fractal_protected_comparison.csv")
    _write_csv(same_replay_comparison_rows, output_dir / "current_vs_fractal_same_replay_comparison.csv")
    _write_csv(forced_close_rows, output_dir / "fractal_forced_close_cases.csv")
    _write_csv([coverage], output_dir / "fractal_economic_coverage_report.csv")
    (output_dir / "fractal_protected_economic_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "fractal_protected_economic_summary.md").write_text(summary_markdown(report), encoding="utf-8")
    (output_dir / "fractal_candidate_promotion_assessment.md").write_text(promotion_markdown(report), encoding="utf-8")
    return report


def replay_candidate_lifecycle(
    *,
    candidate_rows: list[dict[str, Any]],
    m1_by_day: dict[date, list[OhlcvBar]],
    m5_by_day: dict[date, list[OhlcvBar]],
    replay_config: AzirReplicaConfig,
    risk_config: AzirRiskConfig,
) -> list[dict[str, Any]]:
    lifecycle_rows: list[dict[str, Any]] = []
    daily_realized: defaultdict[str, float] = defaultdict(float)
    daily_loss_streak: defaultdict[str, int] = defaultdict(int)
    daily_trades: defaultdict[str, int] = defaultdict(int)
    setup_rows = [row for row in candidate_rows if row.get("event_type") in {"opportunity", "blocked_friday"}]

    for row in sorted(setup_rows, key=lambda item: item.get("_timestamp_dt") or datetime.min):
        setup_dt = _timestamp(row)
        setup_day = setup_dt.date().isoformat()
        if row.get("event_type") == "blocked_friday":
            lifecycle_rows.append(_blocked_row(row, risk_status="blocked_friday"))
            continue

        guard = _pre_trade_risk_guard(
            setup_day=setup_day,
            setup_row=row,
            daily_realized=daily_realized,
            daily_loss_streak=daily_loss_streak,
            daily_trades=daily_trades,
            risk_config=risk_config,
        )
        if guard:
            lifecycle_rows.append(_prevented_row(row, guard))
            continue

        bars, pricing_source = _select_replay_bars(setup_dt, m1_by_day, m5_by_day, risk_config)
        if not bars:
            lifecycle_rows.append(_unpriced_setup_row(row, "no_m1_or_m5_bars_for_setup_day"))
            continue

        replay = simulate_setup_from_bars(
            setup_row=row,
            bars=bars,
            pricing_source=pricing_source,
            replay_config=replay_config,
            risk_config=risk_config,
        )
        lifecycle_rows.append(replay)
        if replay.get("has_exit"):
            pnl = _to_float(replay.get("net_pnl")) or 0.0
            daily_realized[setup_day] += pnl
            daily_loss_streak[setup_day] = daily_loss_streak[setup_day] + 1 if pnl < 0 else 0
            daily_trades[setup_day] += 1
    return lifecycle_rows


def simulate_setup_from_bars(
    *,
    setup_row: dict[str, Any],
    bars: list[OhlcvBar],
    pricing_source: str,
    replay_config: AzirReplicaConfig,
    risk_config: AzirRiskConfig,
) -> dict[str, Any]:
    setup_dt = _timestamp(setup_row)
    close_dt = datetime.combine(setup_dt.date(), time(risk_config.close_hour, 0))
    intraday = [bar for bar in bars if setup_dt <= bar.open_time <= close_dt]
    if not intraday:
        return _unpriced_setup_row(setup_row, "no_replay_bars_inside_session")

    buy_placed = _is_true(setup_row.get("buy_order_placed"))
    sell_placed = _is_true(setup_row.get("sell_order_placed"))
    buy_entry = _to_float(setup_row.get("buy_entry"))
    sell_entry = _to_float(setup_row.get("sell_entry"))
    if not buy_placed and not sell_placed:
        return _no_order_row(setup_row, pricing_source)
    if buy_placed and buy_entry is None:
        return _unpriced_setup_row(setup_row, "missing_buy_entry")
    if sell_placed and sell_entry is None:
        return _unpriced_setup_row(setup_row, "missing_sell_entry")

    close_index = _close_index(intraday, close_dt)
    fill_index: int | None = None
    fill_bar: OhlcvBar | None = None
    fill_side = ""
    ambiguity = False
    for index, bar in enumerate(intraday[:close_index]):
        hit_buy = bool(buy_placed and buy_entry is not None and bar.high >= buy_entry)
        hit_sell = bool(sell_placed and sell_entry is not None and bar.low <= sell_entry)
        if not hit_buy and not hit_sell:
            continue
        ambiguity = hit_buy and hit_sell
        fill_side = _resolve_fill_side(bar, hit_buy, hit_sell, buy_entry or 0.0, sell_entry or 0.0)
        fill_index = index
        fill_bar = bar
        break

    if fill_index is None or fill_bar is None:
        return {
            **_base_lifecycle_row(setup_row),
            "protected_status": "cancelled_no_fill_at_close",
            "risk_status": "cancelled_no_fill_at_close",
            "risk_reason": "Risk Engine hard close cancels live pendings at operational close.",
            "pricing_source": pricing_source,
            "has_fill": False,
            "has_exit": False,
            "forced_close": "false",
            "fill_ambiguity": "false",
        }

    entry = buy_entry if fill_side == "buy" else sell_entry
    if entry is None:
        return _unpriced_setup_row(setup_row, "missing_entry_after_fill_resolution")

    exit_result = simulate_exit_from_bars(
        bars=intraday,
        fill_index=fill_index,
        close_index=close_index,
        side=fill_side,
        entry=entry,
        replay_config=replay_config,
    )
    return {
        **_base_lifecycle_row(setup_row),
        "protected_status": "filled_replayed_exit",
        "risk_status": "kept_replayed_exit",
        "risk_reason": "Candidate setup priced with external Risk Engine close-hour and lifecycle guards.",
        "pricing_source": pricing_source,
        "has_fill": True,
        "has_exit": True,
        "fill_timestamp": _format_dt(fill_bar.open_time),
        "fill_side": fill_side,
        "fill_price": _round(entry),
        "duration_to_fill_seconds": int((fill_bar.open_time - setup_dt).total_seconds()),
        "exit_timestamp": _format_dt(exit_result["exit_timestamp"]),
        "exit_reason": exit_result["exit_reason"],
        "exit_price": _round(exit_result["exit_price"]),
        "duration_seconds": int((exit_result["exit_timestamp"] - fill_bar.open_time).total_seconds()),
        "gross_pnl": _round(_pnl(fill_side, entry, exit_result["exit_price"], replay_config)),
        "net_pnl": _round(_pnl(fill_side, entry, exit_result["exit_price"], replay_config)),
        "mfe_points": _round(exit_result["mfe_points"]),
        "mae_points": _round(exit_result["mae_points"]),
        "trailing_activated": str(exit_result["trailing_activated"]).lower(),
        "trailing_modifications": exit_result["trailing_modifications"],
        "forced_close": str(exit_result["forced_close"]).lower(),
        "fill_ambiguity": str(ambiguity).lower(),
        "commission": 0.0,
        "swap": 0.0,
    }


def simulate_exit_from_bars(
    *,
    bars: list[OhlcvBar],
    fill_index: int,
    close_index: int,
    side: str,
    entry: float,
    replay_config: AzirReplicaConfig,
) -> dict[str, Any]:
    is_buy = side == "buy"
    stop = entry - replay_config.sl_points * replay_config.point if is_buy else entry + replay_config.sl_points * replay_config.point
    target = entry + replay_config.tp_points * replay_config.point if is_buy else entry - replay_config.tp_points * replay_config.point
    mfe_points = 0.0
    mae_points = 0.0
    trailing_activated = False
    trailing_modifications = 0

    for index in range(fill_index, close_index):
        bar = bars[index]
        favorable = (bar.high - entry) / replay_config.point if is_buy else (entry - bar.low) / replay_config.point
        adverse = (entry - bar.low) / replay_config.point if is_buy else (bar.high - entry) / replay_config.point
        mfe_points = max(mfe_points, favorable)
        mae_points = max(mae_points, adverse)
        if is_buy:
            if bar.low <= stop:
                return _exit_result(bar.open_time, stop, "stop_loss_or_trailing_stop", mfe_points, mae_points, trailing_activated, trailing_modifications, False)
            if bar.high >= target:
                return _exit_result(bar.open_time, target, "take_profit", mfe_points, mae_points, trailing_activated, trailing_modifications, False)
            if favorable >= replay_config.trailing_start_points:
                new_stop = bar.high - replay_config.trailing_step_points * replay_config.point
                if new_stop > stop:
                    stop = new_stop
                    trailing_activated = True
                    trailing_modifications += 1
        else:
            if bar.high >= stop:
                return _exit_result(bar.open_time, stop, "stop_loss_or_trailing_stop", mfe_points, mae_points, trailing_activated, trailing_modifications, False)
            if bar.low <= target:
                return _exit_result(bar.open_time, target, "take_profit", mfe_points, mae_points, trailing_activated, trailing_modifications, False)
            if favorable >= replay_config.trailing_start_points:
                new_stop = bar.low + replay_config.trailing_step_points * replay_config.point
                if new_stop < stop:
                    stop = new_stop
                    trailing_activated = True
                    trailing_modifications += 1

    close_bar = bars[min(close_index, len(bars) - 1)]
    return _exit_result(
        close_bar.open_time,
        close_bar.open,
        "risk_engine_forced_session_close",
        mfe_points,
        mae_points,
        trailing_activated,
        trailing_modifications,
        True,
    )


def load_current_protected_reference(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Protected benchmark report does not exist: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {}).get("azir_with_risk_engine_v1_forced_closes_revalued")
    if not metrics:
        raise ValueError("Protected benchmark report does not contain revalued protected metrics.")
    return {
        "benchmark_name": report.get("benchmark_name", "baseline_azir_protected_economic_v1"),
        "path": str(path),
        "metrics": metrics,
        "pricing_convention": report.get("pricing_convention", {}),
    }


def build_comparison_rows(
    current_metrics: dict[str, Any], candidate_metrics: dict[str, Any], current_label: str
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
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        current_value = current_metrics.get(metric)
        candidate_value = candidate_metrics.get(metric)
        rows.append(
            {
                "metric": metric,
                current_label: current_value,
                OUTPUT_BENCHMARK_NAME: candidate_value,
                "delta_candidate_vs_current": _delta(current_value, candidate_value),
            }
        )
    return rows


def build_coverage_report(
    candidate_rows: list[dict[str, Any]],
    lifecycle_rows: list[dict[str, Any]],
    m1_bars: list[OhlcvBar],
    m5_bars: list[OhlcvBar],
    tick_input_path: Path | None,
) -> dict[str, Any]:
    priced = [row for row in lifecycle_rows if row.get("has_exit")]
    source_counts = Counter(row.get("pricing_source", "") for row in priced)
    setup_rows = [row for row in candidate_rows if row.get("event_type") in {"opportunity", "blocked_friday"}]
    unpriced = [row for row in lifecycle_rows if row.get("risk_status") == "unpriced"]
    no_fill = [row for row in lifecycle_rows if row.get("protected_status") == "cancelled_no_fill_at_close"]
    no_order = [row for row in lifecycle_rows if row.get("protected_status") == "no_order_placed"]
    blocked = [row for row in lifecycle_rows if row.get("risk_status") in {"blocked_friday", "prevented"}]
    tick_inspection = inspect_tick_csv_fast(tick_input_path) if tick_input_path else {}
    tick_coverable = _count_trades_in_tick_range(priced, tick_inspection)
    return {
        "candidate_setup_rows": len(setup_rows),
        "candidate_lifecycle_rows": len(lifecycle_rows),
        "closed_trades_priced": len(priced),
        "closed_trades_priced_pct_of_setups": _pct(len(priced), len(setup_rows)),
        "m1_priced_trades": source_counts.get("m1_replay", 0),
        "m1_priced_pct_of_priced": _pct(source_counts.get("m1_replay", 0), len(priced)),
        "m5_priced_trades": source_counts.get("m5_fallback_proxy", 0),
        "m5_priced_pct_of_priced": _pct(source_counts.get("m5_fallback_proxy", 0), len(priced)),
        "tick_priced_trades": 0,
        "tick_priced_pct_of_priced": 0.0,
        "tick_coverable_priced_trades_not_replayed": tick_coverable,
        "unpriced_setups": len(unpriced),
        "unpriced_pct_of_setups": _pct(len(unpriced), len(setup_rows)),
        "cancelled_no_fill_setups": len(no_fill),
        "no_order_setups": len(no_order),
        "blocked_or_prevented_setups": len(blocked),
        "m1_first_bar": _bar_range(m1_bars)["first"],
        "m1_last_bar": _bar_range(m1_bars)["last"],
        "m5_first_bar": _bar_range(m5_bars)["first"],
        "m5_last_bar": _bar_range(m5_bars)["last"],
        "tick_inspection": tick_inspection,
    }


def build_anomaly_summary(lifecycle_rows: list[dict[str, Any]]) -> dict[str, Any]:
    fills = [row for row in lifecycle_rows if row.get("has_fill")]
    out_of_window = []
    for row in fills:
        fill_dt = _parse_dt(row.get("fill_timestamp"))
        if fill_dt.year <= 1 or fill_dt.hour < 16 or fill_dt.hour > 21:
            out_of_window.append(row)
    multi_exit_days = Counter(row.get("setup_day") for row in lifecycle_rows if row.get("has_exit"))
    return {
        "out_of_window_fills": len(out_of_window),
        "multi_exit_days": len([day for day, count in multi_exit_days.items() if day and count > 1]),
        "friday_filled_trades": len([row for row in fills if _is_friday_text(row.get("setup_day"))]),
        "cleanup_or_persistence_issues": 0,
        "forced_closes": len([row for row in lifecycle_rows if row.get("forced_close") == "true"]),
        "fill_ambiguity_rows": len([row for row in lifecycle_rows if row.get("fill_ambiguity") == "true"]),
    }


def build_decision(
    candidate_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
    coverage: dict[str, Any],
    anomaly_summary: dict[str, Any],
    current_same_replay_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_net = _float(candidate_metrics.get("net_pnl"))
    current_net = _float(current_metrics.get("net_pnl"))
    current_replay_net = _float((current_same_replay_metrics or {}).get("net_pnl"))
    candidate_pf = _float(candidate_metrics.get("profit_factor"))
    candidate_expectancy = _float(candidate_metrics.get("expectancy"))
    unpriced = int(coverage.get("unpriced_setups", 0) or 0)
    m5_pct = _float(coverage.get("m5_priced_pct_of_priced"))
    better = candidate_net > current_net and candidate_pf > _float(current_metrics.get("profit_factor"))
    better_same_replay = not current_same_replay_metrics or candidate_net > current_replay_net
    formal_candidate = better and better_same_replay and candidate_pf > 1.1 and candidate_expectancy > 0.0 and unpriced == 0
    freeze_ready = formal_candidate and m5_pct < 25.0 and anomaly_summary.get("fill_ambiguity_rows", 0) == 0
    return {
        "improvement_survives_protected_economic_replay": better,
        "improvement_survives_same_replay_control": better_same_replay,
        "may_formalize_as_protected_economic_candidate": formal_candidate,
        "may_replace_official_baseline_now": False,
        "may_freeze_final_candidate_benchmark_now": freeze_ready,
        "recommended_next_sprint": (
            "tick_level_fractal_candidate_replay_v1"
            if formal_candidate and not freeze_ready
            else "fractal_candidate_operational_revalidation_or_discard"
        ),
        "reason": _decision_reason(formal_candidate, freeze_ready, m5_pct, unpriced, better),
    }


def inspect_tick_csv_fast(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {"available": False, "path": str(path) if path else ""}
    first_data = ""
    with path.open("rb") as handle:
        header = handle.readline().decode("utf-8-sig", errors="replace").strip()
        first_data = handle.readline().decode("utf-8-sig", errors="replace").strip()
        last_data = _tail_last_line(handle)
    columns = [item.strip() for item in header.split(",") if item.strip()]
    return {
        "available": True,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "columns": columns,
        "first_time": first_data.split(",")[0] if first_data else "",
        "last_time": last_data.split(",")[0] if last_data else "",
        "has_bid_ask_columns": "bid" in columns and "ask" in columns,
        "note": "Fast inspection only; row count intentionally not computed for multi-GB file.",
    }


def summary_markdown(report: dict[str, Any]) -> str:
    current = report["current_protected_reference"]["metrics"]
    current_replay = report["current_same_replay_metrics"]
    candidate = report["candidate_metrics"]
    coverage = report["coverage"]
    decision = report["decision"]
    return (
        "# Fractal Protected Economic Candidate v1\n\n"
        "## Executive Summary\n\n"
        f"- Current protected benchmark net/PF/expectancy: {current.get('net_pnl')} / "
        f"{current.get('profit_factor')} / {current.get('expectancy')}.\n"
        f"- Current Azir same-replay control net/PF/expectancy: {current_replay.get('net_pnl')} / "
        f"{current_replay.get('profit_factor')} / {current_replay.get('expectancy')}.\n"
        f"- Fractal candidate replay net/PF/expectancy: {candidate.get('net_pnl')} / "
        f"{candidate.get('profit_factor')} / {candidate.get('expectancy')}.\n"
        f"- Priced closed trades: {coverage.get('closed_trades_priced')} "
        f"({coverage.get('closed_trades_priced_pct_of_setups')}% of setup rows).\n"
        f"- Pricing sources: M1 {coverage.get('m1_priced_trades')}, "
        f"M5 fallback {coverage.get('m5_priced_trades')}, tick priced 0, "
        f"unpriced setups {coverage.get('unpriced_setups')}.\n"
        f"- Promotion to protected economic candidate: {decision['may_formalize_as_protected_economic_candidate']}.\n"
        f"- Final benchmark freeze now: {decision['may_freeze_final_candidate_benchmark_now']}.\n\n"
        "## Method\n\n"
        "- Setup evidence comes from the real MT5 fractal candidate event log.\n"
        "- Fills/exits are reconstructed with Azir order levels, SL/TP/trailing, and Risk Engine close-hour guard.\n"
        "- M1 is used when available; M5 is a labelled fallback for the older period.\n"
        "- Tick data is inspected but not used for final trade pricing in this sprint.\n\n"
        "## Decision\n\n"
        f"{decision['reason']}\n"
    )


def promotion_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    coverage = report["coverage"]
    anomalies = report["anomaly_summary"]
    return (
        "# Fractal Candidate Promotion Assessment\n\n"
        f"- Candidate benchmark label: `{report['candidate_benchmark_name']}`.\n"
        f"- May formalize as protected economic candidate: {decision['may_formalize_as_protected_economic_candidate']}.\n"
        f"- May replace official baseline now: {decision['may_replace_official_baseline_now']}.\n"
        f"- May freeze final candidate benchmark now: {decision['may_freeze_final_candidate_benchmark_now']}.\n"
        f"- M1 priced trades: {coverage['m1_priced_trades']}.\n"
        f"- M5 fallback trades: {coverage['m5_priced_trades']}.\n"
        f"- Unpriced setups: {coverage['unpriced_setups']}.\n"
        f"- Fill ambiguities: {anomalies['fill_ambiguity_rows']}.\n\n"
        "The candidate can be promoted only as a research economic candidate if the uplift survives, "
        "because its economics are still replay-based and not directly observed in MT5.\n"
    )


def _bars_by_day(bars: list[OhlcvBar]) -> dict[date, list[OhlcvBar]]:
    grouped: dict[date, list[OhlcvBar]] = defaultdict(list)
    for bar in bars:
        grouped[bar.open_time.date()].append(bar)
    return dict(grouped)


def _select_replay_bars(
    setup_dt: datetime,
    m1_by_day: dict[date, list[OhlcvBar]],
    m5_by_day: dict[date, list[OhlcvBar]],
    risk_config: AzirRiskConfig,
) -> tuple[list[OhlcvBar], str]:
    close_dt = datetime.combine(setup_dt.date(), time(risk_config.close_hour, 0))
    m1 = [bar for bar in m1_by_day.get(setup_dt.date(), []) if setup_dt <= bar.open_time <= close_dt]
    if m1:
        return m1, "m1_replay"
    m5 = [bar for bar in m5_by_day.get(setup_dt.date(), []) if setup_dt <= bar.open_time <= close_dt]
    if m5:
        return m5, "m5_fallback_proxy"
    return [], "unpriced"


def _pre_trade_risk_guard(
    *,
    setup_day: str,
    setup_row: dict[str, Any],
    daily_realized: dict[str, float],
    daily_loss_streak: dict[str, int],
    daily_trades: dict[str, int],
    risk_config: AzirRiskConfig,
) -> str:
    if risk_config.friday_no_new_trade and _is_friday_text(setup_day):
        return "friday_no_new_trade_plus_close_or_cancel_prior_exposure"
    spread = _to_float(setup_row.get("spread_points"))
    if risk_config.spread_guard_enabled and spread is not None and spread > risk_config.max_spread_points:
        return "spread_guard_if_available"
    if daily_trades[setup_day] >= risk_config.max_trades_per_day:
        return "max_trades_per_day"
    if daily_realized[setup_day] <= -abs(risk_config.max_daily_loss):
        return "daily_max_loss_guard"
    if risk_config.max_consecutive_losses > 0 and daily_loss_streak[setup_day] >= risk_config.max_consecutive_losses:
        return "consecutive_losses_kill_switch"
    return ""


def _base_lifecycle_row(setup_row: dict[str, Any]) -> dict[str, Any]:
    setup_dt = _timestamp(setup_row)
    keys = [
        "timestamp",
        "event_id",
        "event_type",
        "symbol",
        "day_of_week",
        "is_friday",
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
        "atr_filter_passed",
        "rsi_gate_required",
        "rsi_gate_passed",
        "buy_order_placed",
        "sell_order_placed",
    ]
    row = {f"setup_{key}": setup_row.get(key, "") for key in keys}
    row.update(
        {
            "setup_day": setup_dt.date().isoformat(),
            "setup_timestamp": _format_dt(setup_dt),
            "candidate_variant": CANDIDATE_VARIANT,
        }
    )
    return row


def _blocked_row(setup_row: dict[str, Any], *, risk_status: str) -> dict[str, Any]:
    return {
        **_base_lifecycle_row(setup_row),
        "protected_status": risk_status,
        "risk_status": risk_status,
        "risk_reason": "Setup is blocked before order placement.",
        "pricing_source": "not_applicable",
        "has_fill": False,
        "has_exit": False,
        "forced_close": "false",
        "fill_ambiguity": "false",
    }


def _prevented_row(setup_row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        **_base_lifecycle_row(setup_row),
        "protected_status": "prevented_by_risk_engine",
        "risk_status": "prevented",
        "risk_reason": reason,
        "pricing_source": "not_applicable",
        "has_fill": False,
        "has_exit": False,
        "forced_close": "false",
        "fill_ambiguity": "false",
    }


def _unpriced_setup_row(setup_row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        **_base_lifecycle_row(setup_row),
        "protected_status": "unpriced",
        "risk_status": "unpriced",
        "risk_reason": reason,
        "pricing_source": "unpriced",
        "has_fill": False,
        "has_exit": False,
        "forced_close": "false",
        "fill_ambiguity": "false",
    }


def _no_order_row(setup_row: dict[str, Any], pricing_source: str) -> dict[str, Any]:
    return {
        **_base_lifecycle_row(setup_row),
        "protected_status": "no_order_placed",
        "risk_status": "no_order_placed",
        "risk_reason": "Candidate setup placed no pending order after Azir filters.",
        "pricing_source": pricing_source,
        "has_fill": False,
        "has_exit": False,
        "forced_close": "false",
        "fill_ambiguity": "false",
    }


def _exit_result(
    timestamp: datetime,
    price: float,
    reason: str,
    mfe_points: float,
    mae_points: float,
    trailing_activated: bool,
    trailing_modifications: int,
    forced_close: bool,
) -> dict[str, Any]:
    return {
        "exit_timestamp": timestamp,
        "exit_price": price,
        "exit_reason": reason,
        "mfe_points": mfe_points,
        "mae_points": mae_points,
        "trailing_activated": trailing_activated,
        "trailing_modifications": trailing_modifications,
        "forced_close": forced_close,
    }


def _resolve_fill_side(bar: OhlcvBar, hit_buy: bool, hit_sell: bool, buy_entry: float, sell_entry: float) -> str:
    if hit_buy and not hit_sell:
        return "buy"
    if hit_sell and not hit_buy:
        return "sell"
    return "buy" if abs(bar.open - buy_entry) <= abs(bar.open - sell_entry) else "sell"


def _close_index(bars: list[OhlcvBar], close_dt: datetime) -> int:
    for index, bar in enumerate(bars):
        if bar.open_time >= close_dt:
            return index
    return len(bars) - 1


def _pnl(side: str, entry: float, exit_price: float, replay_config: AzirReplicaConfig) -> float:
    direction = 1.0 if side == "buy" else -1.0
    return (exit_price - entry) * direction * replay_config.lot_size * replay_config.contract_size


def _delta(left: Any, right: Any) -> float | str:
    left_float = _to_float(left)
    right_float = _to_float(right)
    if left_float is None or right_float is None:
        return ""
    return _round(right_float - left_float)


def _float(value: Any) -> float:
    parsed = _to_float(value)
    return 0.0 if parsed is None else float(parsed)


def _pct(numerator: int | float, denominator: int | float) -> float:
    return _round((float(numerator) / float(denominator) * 100.0) if denominator else 0.0) or 0.0


def _timestamp(row: dict[str, Any]) -> datetime:
    value = row.get("_timestamp_dt")
    if isinstance(value, datetime):
        return value
    return _parse_dt(row.get("timestamp"))


def _parse_dt(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.min
    return datetime.fromisoformat(text.replace(".", "-").replace("T", " "))


def _format_dt(value: datetime) -> str:
    return value.strftime("%Y.%m.%d %H:%M:%S")


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _is_friday_text(day_text: Any) -> bool:
    try:
        return datetime.fromisoformat(str(day_text)[:10]).weekday() == 4
    except ValueError:
        return False


def _bar_range(bars: list[OhlcvBar]) -> dict[str, str]:
    if not bars:
        return {"first": "", "last": ""}
    return {"first": bars[0].open_time.isoformat(sep=" "), "last": bars[-1].open_time.isoformat(sep=" ")}


def _count_trades_in_tick_range(trades: list[dict[str, Any]], tick_inspection: dict[str, Any]) -> int:
    if not tick_inspection.get("available"):
        return 0
    first = _parse_dt(tick_inspection.get("first_time"))
    last = _parse_dt(tick_inspection.get("last_time"))
    if first.year <= 1 or last.year <= 1:
        return 0
    count = 0
    for row in trades:
        fill_dt = _parse_dt(row.get("fill_timestamp"))
        exit_dt = _parse_dt(row.get("exit_timestamp"))
        if first <= fill_dt <= last and first <= exit_dt <= last:
            count += 1
    return count


def _tail_last_line(handle: Any, block_size: int = 8192) -> str:
    handle.seek(0, 2)
    file_size = handle.tell()
    if file_size == 0:
        return ""
    data = b""
    offset = 0
    while file_size - offset > 0:
        offset = min(file_size, offset + block_size)
        handle.seek(file_size - offset)
        data = handle.read(min(block_size, file_size - offset + block_size)) + data
        lines = [line for line in data.splitlines() if line.strip()]
        if len(lines) >= 2 or offset == file_size:
            return lines[-1].decode("utf-8-sig", errors="replace")
    return ""


def _decision_reason(formal_candidate: bool, freeze_ready: bool, m5_pct: float, unpriced: int, better: bool) -> str:
    if freeze_ready:
        return "Candidate improves the protected reference and has enough high-granularity coverage to consider freezing later."
    if formal_candidate:
        return (
            "Candidate improves the protected reference and has no unpriced setups, but too much of the result still "
            f"uses M5 fallback ({m5_pct}%). It may be promoted as a protected economic research candidate, not final benchmark."
        )
    if not better:
        return "Candidate does not improve the current protected benchmark after protected economic replay."
    if unpriced:
        return f"Candidate improves proxy metrics, but {unpriced} setups remain unpriced."
    return "Candidate remains promising but fails one or more quality gates for formal promotion."


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "sprint": report["sprint"],
        "candidate_net_pnl": report["candidate_metrics"].get("net_pnl"),
        "candidate_profit_factor": report["candidate_metrics"].get("profit_factor"),
        "current_net_pnl": report["current_protected_reference"]["metrics"].get("net_pnl"),
        "current_same_replay_net_pnl": report["current_same_replay_metrics"].get("net_pnl"),
        "closed_trades_priced": report["coverage"].get("closed_trades_priced"),
        "m1_priced_trades": report["coverage"].get("m1_priced_trades"),
        "m5_priced_trades": report["coverage"].get("m5_priced_trades"),
        "unpriced_setups": report["coverage"].get("unpriced_setups"),
        "formal_candidate": report["decision"]["may_formalize_as_protected_economic_candidate"],
        "freeze_ready_now": report["decision"]["may_freeze_final_candidate_benchmark_now"],
        "recommended_next_sprint": report["decision"]["recommended_next_sprint"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
