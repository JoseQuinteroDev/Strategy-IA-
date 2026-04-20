"""Final protected-baseline comparison for the Azir fractal lifecycle log."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hybrid_quant.risk.azir_state import AzirRiskConfig

from .economic_audit import (
    _event_day,
    _event_id_day,
    _parse_timestamp,
    _round,
    _to_float,
    _write_csv,
    build_anomaly_reports,
    build_trailing_report,
    reconstruct_lifecycles,
    summarize_trades,
)
from .event_log import AZIR_EVENT_COLUMNS
from .risk_reaudit import apply_risk_engine_to_lifecycle


SPRINT_NAME = "compare_fractal_full_lifecycle_mt5_vs_baseline_protected_v1"
BASELINE_NAME = "baseline_azir_protected_economic_v1"
CANDIDATE_NAME = "baseline_azir_protected_economic_candidate_fractal_v1"
BASELINE_METRIC_KEY = "azir_with_risk_engine_v1_forced_closes_revalued"
DEFAULT_SYMBOL = "XAUUSD-STD"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare protected Azir baseline vs real MT5 fractal lifecycle.")
    parser.add_argument("--current-log-path", required=True)
    parser.add_argument("--candidate-full-lifecycle-log-path", required=True)
    parser.add_argument("--candidate-setup-log-path", required=True)
    parser.add_argument("--m5-input-path", required=True)
    parser.add_argument("--m1-input-path", required=True)
    parser.add_argument("--tick-input-path", required=True)
    parser.add_argument("--forced-close-report-path", required=True)
    parser.add_argument("--fractal-protected-report-path", required=True)
    parser.add_argument("--fractal-tick-replay-report-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_fractal_lifecycle_final_comparison(
        current_log_path=Path(args.current_log_path),
        candidate_full_lifecycle_log_path=Path(args.candidate_full_lifecycle_log_path),
        candidate_setup_log_path=Path(args.candidate_setup_log_path),
        m5_input_path=Path(args.m5_input_path),
        m1_input_path=Path(args.m1_input_path),
        tick_input_path=Path(args.tick_input_path),
        forced_close_report_path=Path(args.forced_close_report_path),
        fractal_protected_report_path=Path(args.fractal_protected_report_path),
        fractal_tick_replay_report_path=Path(args.fractal_tick_replay_report_path),
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_fractal_lifecycle_final_comparison(
    *,
    current_log_path: Path,
    candidate_full_lifecycle_log_path: Path,
    candidate_setup_log_path: Path,
    m5_input_path: Path,
    m1_input_path: Path,
    tick_input_path: Path,
    forced_close_report_path: Path,
    fractal_protected_report_path: Path,
    fractal_tick_replay_report_path: Path,
    output_dir: Path,
    symbol: str = DEFAULT_SYMBOL,
    risk_config: AzirRiskConfig | None = None,
) -> dict[str, Any]:
    risk_config = risk_config or AzirRiskConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    current_rows = flatten_segments(read_segmented_event_log(current_log_path, symbol))
    candidate_segments = read_segmented_event_log(candidate_full_lifecycle_log_path, symbol)
    if not current_rows:
        raise ValueError(f"No current Azir rows found for {symbol}.")
    if not candidate_segments:
        raise ValueError(f"No candidate lifecycle rows found for {symbol}.")

    selected = select_primary_segment(candidate_segments)
    candidate_rows = selected["rows"]
    baseline_lot = infer_setup_lot(current_rows) or 0.01
    candidate_lot = selected["primary_lot"] or infer_setup_lot(candidate_rows) or baseline_lot
    pnl_scale = baseline_lot / candidate_lot if candidate_lot else 1.0
    baseline_report = read_json(forced_close_report_path)
    baseline_metrics = baseline_report.get("metrics", {}).get(BASELINE_METRIC_KEY, {})
    if not baseline_metrics:
        raise ValueError(f"Missing protected metrics key: {BASELINE_METRIC_KEY}")

    current_recon = reconstruct_lifecycles(
        current_rows,
        session_start_hour=risk_config.session_fill_start_hour,
        session_end_hour=risk_config.session_fill_end_hour,
        close_hour=risk_config.close_hour,
    )
    candidate_recon = reconstruct_lifecycles(
        candidate_rows,
        session_start_hour=risk_config.session_fill_start_hour,
        session_end_hour=risk_config.session_fill_end_hour,
        close_hour=risk_config.close_hour,
    )
    current_risk = apply_risk_engine_to_lifecycle(
        rows=current_rows,
        lifecycle_rows=current_recon["lifecycles"],
        trade_rows=current_recon["trades"],
        config=risk_config,
    )
    candidate_risk = apply_risk_engine_to_lifecycle(
        rows=candidate_rows,
        lifecycle_rows=candidate_recon["lifecycles"],
        trade_rows=candidate_recon["trades"],
        config=risk_config,
    )

    candidate_trades = [row for row in candidate_recon["trades"] if row.get("has_exit")]
    candidate_protected = [row for row in candidate_risk["protected_trades"] if row.get("has_exit")]
    candidate_metrics_raw = summarize_trades(candidate_trades)
    candidate_metrics_norm = summarize_trades(scale_trades(candidate_trades, pnl_scale))
    candidate_protected_raw = summarize_trades(candidate_protected)
    candidate_protected_norm = summarize_trades(scale_trades(candidate_protected, pnl_scale))
    current_trades = [row for row in current_recon["trades"] if row.get("has_exit")]
    current_metrics = summarize_trades(current_trades)

    candidate_anomalies = build_anomaly_reports(
        candidate_rows,
        candidate_recon["lifecycles"],
        candidate_recon["trades"],
        risk_config.session_fill_start_hour,
        risk_config.session_fill_end_hour,
    )
    current_anomalies = build_anomaly_reports(
        current_rows,
        current_recon["lifecycles"],
        current_recon["trades"],
        risk_config.session_fill_start_hour,
        risk_config.session_fill_end_hour,
    )
    schema_check = schema_report(candidate_full_lifecycle_log_path, candidate_segments, selected, symbol)
    comparability = {
        "same_symbol": set(row.get("symbol", "") for row in current_rows) == set(row.get("symbol", "") for row in candidate_rows),
        "baseline_lot": baseline_lot,
        "candidate_lot": candidate_lot,
        "same_lot": abs(baseline_lot - candidate_lot) < 1e-12,
        "pnl_normalization_scale_to_baseline_lot": pnl_scale,
        "multiple_candidate_segments": len(candidate_segments) > 1,
        "selected_candidate_segment_index": selected["segment_index"],
        "same_risk_engine_policy": risk_config.name == "risk_engine_azir_v1",
        "same_rules_except_swing": True,
        "structural_contamination": len(candidate_segments) > 1 or abs(baseline_lot - candidate_lot) >= 1e-12,
    }
    comparison_rows = metric_comparison_rows(
        baseline_metrics,
        candidate_metrics_raw,
        candidate_metrics_norm,
        candidate_protected_raw,
        candidate_protected_norm,
    )
    exit_rows = exit_distribution_rows(current_trades, candidate_trades, candidate_protected)
    risk_rows = risk_profile_rows(
        baseline_metrics,
        current_recon,
        current_risk,
        current_anomalies,
        build_trailing_report(current_trades),
        candidate_recon,
        candidate_risk,
        candidate_anomalies,
        build_trailing_report(candidate_trades),
        candidate_protected_norm,
    )
    decision = promotion_decision(baseline_metrics, candidate_protected_norm, schema_check, comparability, candidate_risk)
    report = {
        "sprint": SPRINT_NAME,
        "baseline_name": BASELINE_NAME,
        "candidate_name": CANDIDATE_NAME,
        "symbol": symbol,
        "sources": {
            "current_log_path": file_info(current_log_path),
            "candidate_full_lifecycle_log_path": file_info(candidate_full_lifecycle_log_path),
            "candidate_setup_log_path": file_info(candidate_setup_log_path),
            "m5_input_path": file_info(m5_input_path),
            "m1_input_path": file_info(m1_input_path),
            "tick_input_path": file_info(tick_input_path),
            "forced_close_report_path": file_info(forced_close_report_path),
            "fractal_protected_report_path": file_info(fractal_protected_report_path),
            "fractal_tick_replay_report_path": file_info(fractal_tick_replay_report_path),
        },
        "risk_engine": asdict(risk_config),
        "candidate_lifecycle_schema_check": schema_check,
        "comparability": comparability,
        "metrics": {
            "baseline_protected": baseline_metrics,
            "current_observed_reference": current_metrics,
            "candidate_observed_raw": candidate_metrics_raw,
            "candidate_observed_normalized_to_baseline_lot": candidate_metrics_norm,
            "candidate_protected_raw": candidate_protected_raw,
            "candidate_protected_normalized_to_baseline_lot": candidate_protected_norm,
        },
        "comparison_rows": comparison_rows,
        "exit_distribution_rows": exit_rows,
        "risk_profile_rows": risk_rows,
        "candidate_anomaly_summary": {key: len(value) for key, value in candidate_anomalies.items()},
        "current_anomaly_summary": {key: len(value) for key, value in current_anomalies.items()},
        "candidate_risk_decision_summary": dict(Counter(row["risk_status"] for row in candidate_risk["trade_decisions"])),
        "current_risk_decision_summary": dict(Counter(row["risk_status"] for row in current_risk["trade_decisions"])),
        "decision": decision,
        "limitations": limitations(comparability, decision),
    }
    _write_csv(comparison_rows, output_dir / "baseline_vs_fractal_protected_comparison.csv")
    _write_csv(exit_rows, output_dir / "fractal_vs_baseline_exit_distribution.csv")
    _write_csv(risk_rows, output_dir / "fractal_vs_baseline_risk_profile.csv")
    (output_dir / "fractal_lifecycle_schema_check.json").write_text(json.dumps(schema_check, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "fractal_vs_baseline_protected_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "fractal_vs_baseline_protected_summary.md").write_text(summary_md(report), encoding="utf-8")
    (output_dir / "fractal_promotion_final_assessment.md").write_text(promotion_md(report), encoding="utf-8")
    return report


def read_segmented_event_log(path: Path, symbol: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    prev = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(AZIR_EVENT_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Event log missing canonical columns: {sorted(missing)}")
        for raw in reader:
            if symbol and raw.get("symbol") and raw.get("symbol") != symbol:
                continue
            row = {column: raw.get(column, "") for column in AZIR_EVENT_COLUMNS}
            row["_raw_event_id_blank"] = not str(row.get("event_id", "")).strip()
            row["_event_day"] = _event_day(row)
            row["_timestamp_dt"] = _parse_timestamp(row.get("timestamp"))
            row["_event_id_day"] = _event_id_day(row.get("event_id"))
            if prev and row["_timestamp_dt"] < prev and current:
                segments.append(current)
                current = []
            current.append(row)
            prev = row["_timestamp_dt"]
    if current:
        segments.append(current)
    return [make_segment(rows, index) for index, rows in enumerate(segments, start=1)]


def make_segment(rows: list[dict[str, Any]], index: int) -> dict[str, Any]:
    rows.sort(key=lambda row: row["_timestamp_dt"])
    lots = Counter(row.get("lot_size", "") for row in rows if row.get("lot_size"))
    events = Counter(row.get("event_type", "") for row in rows)
    return {
        "segment_index": index,
        "rows": rows,
        "row_count": len(rows),
        "first_timestamp": rows[0].get("timestamp", "") if rows else "",
        "last_timestamp": rows[-1].get("timestamp", "") if rows else "",
        "event_counts": dict(events),
        "lot_counts": dict(lots),
        "primary_lot": _to_float(lots.most_common(1)[0][0]) if lots else None,
        "fill_count": events.get("fill", 0),
        "exit_count": events.get("exit", 0),
    }


def flatten_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for segment in segments for row in segment["rows"]]


def select_primary_segment(segments: list[dict[str, Any]]) -> dict[str, Any]:
    return max(segments, key=lambda item: (item["exit_count"], item["fill_count"], item["row_count"], item["segment_index"]))


def infer_setup_lot(rows: list[dict[str, Any]]) -> float | None:
    counts = Counter(row.get("lot_size", "") for row in rows if row.get("event_type") in {"opportunity", "blocked_friday"} and row.get("lot_size"))
    return _to_float(counts.most_common(1)[0][0]) if counts else None


def scale_trades(trades: list[dict[str, Any]], scale: float) -> list[dict[str, Any]]:
    scaled = []
    for trade in trades:
        row = dict(trade)
        for key in ("net_pnl", "gross_pnl", "commission", "swap"):
            value = _to_float(row.get(key))
            if value is not None:
                row[key] = _round(value * scale)
        scaled.append(row)
    return scaled


def schema_report(path: Path, segments: list[dict[str, Any]], selected: dict[str, Any], symbol: str) -> dict[str, Any]:
    rows = selected["rows"]
    events = Counter(row.get("event_type", "") for row in rows)
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "symbol": symbol,
        "columns": list(AZIR_EVENT_COLUMNS),
        "schema_compatible": True,
        "segments_detected": len(segments),
        "segments": [public_segment(segment) for segment in segments],
        "selected_segment_index": selected["segment_index"],
        "selected_rows": len(rows),
        "first_timestamp": selected["first_timestamp"],
        "last_timestamp": selected["last_timestamp"],
        "event_counts": dict(events),
        "has_fills": events.get("fill", 0) > 0,
        "has_trailing_modified": events.get("trailing_modified", 0) > 0,
        "has_exits": events.get("exit", 0) > 0,
        "has_pnl": any(row.get("net_pnl") for row in rows if row.get("event_type") == "exit"),
        "usable_as_direct_economic_evidence": events.get("fill", 0) > 0 and events.get("exit", 0) > 0,
        "mixed_or_multiple_runs_detected": len(segments) > 1,
    }


def public_segment(segment: dict[str, Any]) -> dict[str, Any]:
    return {key: segment[key] for key in ("segment_index", "row_count", "first_timestamp", "last_timestamp", "event_counts", "lot_counts", "primary_lot")}


def metric_comparison_rows(baseline: dict[str, Any], cand_raw: dict[str, Any], cand_norm: dict[str, Any], cand_prot_raw: dict[str, Any], cand_prot_norm: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = ["closed_trades", "net_pnl", "win_rate", "average_win", "average_loss", "payoff", "profit_factor", "expectancy", "max_drawdown_abs", "max_consecutive_losses"]
    return [
        {
            "metric": metric,
            "baseline_azir_protected_economic_v1": baseline.get(metric),
            "fractal_observed_raw_lot": cand_raw.get(metric),
            "fractal_observed_normalized_to_baseline_lot": cand_norm.get(metric),
            "fractal_risk_applied_raw_lot": cand_prot_raw.get(metric),
            "fractal_risk_applied_normalized_to_baseline_lot": cand_prot_norm.get(metric),
            "delta_fractal_risk_applied_normalized_vs_baseline": delta(cand_prot_norm.get(metric), baseline.get(metric)),
        }
        for metric in metrics
    ]


def exit_distribution_rows(current: list[dict[str, Any]], cand: list[dict[str, Any]], cand_prot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label, trades in [("current_azir_observed_log", current), ("fractal_candidate_observed_selected_segment", cand), ("fractal_candidate_risk_applied_selected_segment", cand_prot)]:
        total = len(trades)
        for reason, count in sorted(Counter(row.get("exit_reason", "") or "unknown" for row in trades).items()):
            pnl = sum(_to_float(row.get("net_pnl")) or 0.0 for row in trades if (row.get("exit_reason", "") or "unknown") == reason)
            rows.append({"profile": label, "exit_reason": reason, "trades": count, "pct_of_profile_trades": _round((count / total * 100.0) if total else 0.0, 4), "net_pnl_raw_profile_scale": _round(pnl)})
    return rows


def risk_profile_rows(baseline: dict[str, Any], current_recon: dict[str, Any], current_risk: dict[str, Any], current_anom: dict[str, Any], current_trailing: dict[str, Any], candidate_recon: dict[str, Any], candidate_risk: dict[str, Any], candidate_anom: dict[str, Any], candidate_trailing: dict[str, Any], candidate_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    def row(label: str, metrics: dict[str, Any], recon: dict[str, Any], risk: dict[str, Any], anom: dict[str, Any], trailing: dict[str, Any], source: str) -> dict[str, Any]:
        decisions = Counter(item["risk_status"] for item in risk["trade_decisions"])
        return {
            "profile": label,
            "closed_trades": metrics.get("closed_trades"),
            "net_pnl": metrics.get("net_pnl"),
            "max_drawdown_abs": metrics.get("max_drawdown_abs"),
            "max_consecutive_losses": metrics.get("max_consecutive_losses"),
            "risk_kept_observed_exit": decisions.get("kept_observed_exit", 0),
            "risk_prevented": decisions.get("prevented", 0),
            "risk_forced_close_unpriced": decisions.get("forced_close_unpriced", 0),
            "multi_exit_days": len(anom["multi_exit_days"]),
            "out_of_window_fills": len(anom["out_of_window_fills"]),
            "friday_exit_events": len(anom["friday_exit_events"]),
            "cleanup_issues": len(anom["open_order_cleanup_issues"]),
            "no_fill_cleanups": sum(1 for item in recon["lifecycles"] if int(item.get("cleanup_count", 0) or 0) > 0),
            "trailing_activated_trades": trailing["summary"].get("trailing_activated_trades"),
            "source": source,
        }
    return [
        row(BASELINE_NAME, baseline, current_recon, current_risk, current_anom, current_trailing, "frozen metrics plus current-log risk profile"),
        row(f"{CANDIDATE_NAME}_normalized", candidate_metrics, candidate_recon, candidate_risk, candidate_anom, candidate_trailing, "real MT5 candidate segment plus risk_engine_azir_v1"),
    ]


def promotion_decision(baseline: dict[str, Any], candidate: dict[str, Any], schema: dict[str, Any], comparability: dict[str, Any], risk: dict[str, Any]) -> dict[str, Any]:
    net_delta = delta(candidate.get("net_pnl"), baseline.get("net_pnl")) or 0.0
    pf_delta = delta(candidate.get("profit_factor"), baseline.get("profit_factor")) or 0.0
    dd_delta = delta(candidate.get("max_drawdown_abs"), baseline.get("max_drawdown_abs")) or 0.0
    forced_unpriced = Counter(row["risk_status"] for row in risk["trade_decisions"]).get("forced_close_unpriced", 0)
    promote = bool(schema["usable_as_direct_economic_evidence"] and comparability["same_symbol"] and forced_unpriced == 0 and net_delta > 0 and pf_delta >= 0 and dd_delta <= 0 and not comparability["structural_contamination"])
    if net_delta <= 0:
        recommendation = "keep_baseline_azir_protected_economic_v1_official"
        reason = "After lot normalization and Risk Engine application, the fractal candidate does not beat baseline net PnL."
    elif comparability["structural_contamination"]:
        recommendation = "rerun_clean_mt5_fractal_lifecycle_before_promotion"
        reason = "The candidate evidence is promising but the CSV has multiple runs or different lot sizing."
    elif promote:
        recommendation = "promote_fractal_candidate_to_official_protected_baseline"
        reason = "The candidate beats net PnL, PF and drawdown with comparable evidence."
    else:
        recommendation = "keep_fractal_as_serious_candidate_without_final_promotion"
        reason = "The evidence is not strong enough for final replacement."
    return {
        "promote_candidate": promote,
        "recommendation": recommendation,
        "reason": reason,
        "net_pnl_delta_normalized_vs_baseline": _round(net_delta),
        "profit_factor_delta_vs_baseline": _round(pf_delta),
        "max_drawdown_abs_delta_vs_baseline": _round(dd_delta),
        "candidate_forced_close_unpriced": forced_unpriced,
        "next_recommended_sprint": "clean_mt5_fractal_lifecycle_rerun_lot_0_01_then_final_decision_v1" if comparability["structural_contamination"] else "keep_baseline_and_archive_fractal_candidate_findings_v1",
    }


def limitations(comparability: dict[str, Any], decision: dict[str, Any]) -> list[str]:
    notes = []
    if comparability["multiple_candidate_segments"]:
        notes.append("Candidate CSV contains more than one chronological Strategy Tester segment; only the primary full segment is used.")
    if not comparability["same_lot"]:
        notes.append("Candidate PnL is normalized to baseline lot size before comparison.")
    if not decision["promote_candidate"]:
        notes.append("No final replacement is made in this sprint.")
    return notes


def file_info(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def delta(candidate: Any, baseline: Any) -> float | None:
    left = _to_float(candidate)
    right = _to_float(baseline)
    return None if left is None or right is None else _round(left - right)


def summary_md(report: dict[str, Any]) -> str:
    base = report["metrics"]["baseline_protected"]
    cand = report["metrics"]["candidate_protected_normalized_to_baseline_lot"]
    comp = report["comparability"]
    dec = report["decision"]
    return (
        "# Fractal vs Protected Baseline Final Comparison\n\n"
        f"- Candidate promoted: `{dec['promote_candidate']}`.\n"
        f"- Recommendation: `{dec['recommendation']}`.\n"
        f"- Baseline net/PF/DD: `{base.get('net_pnl')}` / `{base.get('profit_factor')}` / `{base.get('max_drawdown_abs')}`.\n"
        f"- Candidate normalized protected net/PF/DD: `{cand.get('net_pnl')}` / `{cand.get('profit_factor')}` / `{cand.get('max_drawdown_abs')}`.\n"
        f"- Candidate selected segment: `{comp['selected_candidate_segment_index']}`.\n"
        f"- Lot normalization scale: `{comp['pnl_normalization_scale_to_baseline_lot']}`.\n\n"
        f"Decision: {dec['reason']}\n"
    )


def promotion_md(report: dict[str, Any]) -> str:
    dec = report["decision"]
    limits = "\n".join(f"- {item}" for item in report["limitations"]) or "- None."
    return (
        "# Fractal Promotion Final Assessment\n\n"
        f"- Promote candidate: `{dec['promote_candidate']}`.\n"
        f"- Recommendation: `{dec['recommendation']}`.\n"
        f"- Net PnL delta: `{dec['net_pnl_delta_normalized_vs_baseline']}`.\n"
        f"- PF delta: `{dec['profit_factor_delta_vs_baseline']}`.\n"
        f"- Max DD delta: `{dec['max_drawdown_abs_delta_vs_baseline']}`.\n"
        f"- Next sprint: `{dec['next_recommended_sprint']}`.\n\n"
        "## Limitations\n\n"
        f"{limits}\n"
    )


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "sprint": SPRINT_NAME,
        "promote_candidate": report["decision"]["promote_candidate"],
        "recommendation": report["decision"]["recommendation"],
        "baseline_net_pnl": report["metrics"]["baseline_protected"].get("net_pnl"),
        "candidate_normalized_protected_net_pnl": report["metrics"]["candidate_protected_normalized_to_baseline_lot"].get("net_pnl"),
        "candidate_segments_detected": report["candidate_lifecycle_schema_check"]["segments_detected"],
        "next_recommended_sprint": report["decision"]["next_recommended_sprint"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
