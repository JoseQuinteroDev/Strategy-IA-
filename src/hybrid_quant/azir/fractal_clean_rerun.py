"""Validate a clean MT5 fractal rerun and prepare/finalize promotion decision."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from .economic_audit import _write_csv
from .fractal_lifecycle_final_comparison import (
    DEFAULT_SYMBOL,
    run_fractal_lifecycle_final_comparison,
    read_segmented_event_log,
)


SPRINT_NAME = "clean_mt5_fractal_lifecycle_rerun_lot_0_01_then_final_decision_v1"
EXPECTED_CLEAN_FILE = "fractal_candidate_full_lifecycle_event_log_clean_001.csv"
EXPECTED_START = date(2021, 1, 4)
EXPECTED_END = date(2025, 12, 31)
LAST_EVENT_MIN = date(2025, 12, 30)
EXPECTED_LOT = 0.01
REQUIRED_EVENTS = ("opportunity", "fill", "trailing_modified", "exit")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate clean MT5 fractal rerun and run final decision if valid.")
    parser.add_argument("--current-log-path", required=True)
    parser.add_argument("--clean-candidate-log-path", required=True)
    parser.add_argument("--candidate-setup-log-path", required=True)
    parser.add_argument("--m5-input-path", required=True)
    parser.add_argument("--m1-input-path", required=True)
    parser.add_argument("--tick-input-path", required=True)
    parser.add_argument("--forced-close-report-path", required=True)
    parser.add_argument("--fractal-protected-report-path", required=True)
    parser.add_argument("--previous-final-report-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_clean_rerun_gate(
        current_log_path=Path(args.current_log_path),
        clean_candidate_log_path=Path(args.clean_candidate_log_path),
        candidate_setup_log_path=Path(args.candidate_setup_log_path),
        m5_input_path=Path(args.m5_input_path),
        m1_input_path=Path(args.m1_input_path),
        tick_input_path=Path(args.tick_input_path),
        forced_close_report_path=Path(args.forced_close_report_path),
        fractal_protected_report_path=Path(args.fractal_protected_report_path),
        previous_final_report_path=Path(args.previous_final_report_path),
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_clean_rerun_gate(
    *,
    current_log_path: Path,
    clean_candidate_log_path: Path,
    candidate_setup_log_path: Path,
    m5_input_path: Path,
    m1_input_path: Path,
    tick_input_path: Path,
    forced_close_report_path: Path,
    fractal_protected_report_path: Path,
    previous_final_report_path: Path,
    output_dir: Path,
    symbol: str = DEFAULT_SYMBOL,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_clean_csv(clean_candidate_log_path, symbol)
    comparison_report: dict[str, Any] | None = None
    final_output_dir = output_dir / "final_comparison"
    if validation["accepted"]:
        comparison_report = run_fractal_lifecycle_final_comparison(
            current_log_path=current_log_path,
            candidate_full_lifecycle_log_path=clean_candidate_log_path,
            candidate_setup_log_path=candidate_setup_log_path,
            m5_input_path=m5_input_path,
            m1_input_path=m1_input_path,
            tick_input_path=tick_input_path,
            forced_close_report_path=forced_close_report_path,
            fractal_protected_report_path=fractal_protected_report_path,
            fractal_tick_replay_report_path=previous_final_report_path,
            output_dir=final_output_dir,
            symbol=symbol,
        )
    readiness = build_readiness(validation, comparison_report)
    report = {
        "sprint": SPRINT_NAME,
        "expected_clean_file": EXPECTED_CLEAN_FILE,
        "sources": {
            "current_log_path": file_info(current_log_path),
            "clean_candidate_log_path": file_info(clean_candidate_log_path),
            "candidate_setup_log_path": file_info(candidate_setup_log_path),
            "m5_input_path": file_info(m5_input_path),
            "m1_input_path": file_info(m1_input_path),
            "tick_input_path": file_info(tick_input_path),
            "forced_close_report_path": file_info(forced_close_report_path),
            "fractal_protected_report_path": file_info(fractal_protected_report_path),
            "previous_final_report_path": file_info(previous_final_report_path),
        },
        "mt5_rerun_requirements": rerun_requirements(),
        "validation": validation,
        "final_comparison": comparison_report,
        "readiness": readiness,
    }
    write_artifacts(output_dir, report)
    return report


def validate_clean_csv(path: Path, symbol: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "accepted": False,
            "checks": [{"name": "file_exists", "passed": False, "detail": str(path)}],
        }
    segments = read_segmented_event_log(path, symbol)
    rows = [row for segment in segments for row in segment["rows"]]
    selected = segments[0] if segments else {}
    event_counts = selected.get("event_counts", {})
    lot_counts = selected.get("lot_counts", {})
    timestamps = [row["_timestamp_dt"].date() for row in rows if row.get("timestamp")]
    first_date = min(timestamps) if timestamps else None
    last_date = max(timestamps) if timestamps else None
    setup_rows = [row for row in rows if row.get("event_type") in {"opportunity", "blocked_friday"}]
    exit_rows = [row for row in rows if row.get("event_type") == "exit"]

    checks.append(check("file_exists", True, str(path)))
    checks.append(check("schema_compatible", True, "canonical Azir event columns parsed"))
    checks.append(check("symbol_is_expected", all(row.get("symbol") == symbol for row in rows), symbol))
    checks.append(check("single_chronological_segment", len(segments) == 1, f"segments={len(segments)}"))
    checks.append(check("official_lot_size_0_01", set(lot_counts) == {"0.01"}, f"lot_counts={lot_counts}"))
    checks.append(check("timeframe_m5", all(row.get("timeframe") in {"", "M5"} for row in rows), "all non-empty timeframe values must be M5"))
    for event in REQUIRED_EVENTS:
        checks.append(check(f"has_{event}", int(event_counts.get(event, 0)) > 0, f"count={event_counts.get(event, 0)}"))
    checks.append(check("has_usable_net_pnl", all(str(row.get("net_pnl", "")).strip() for row in exit_rows), f"exit_rows={len(exit_rows)}"))
    checks.append(check("has_usable_gross_pnl", all(str(row.get("gross_pnl", "")).strip() for row in exit_rows), f"exit_rows={len(exit_rows)}"))
    checks.append(check("has_usable_exit_reason", all(str(row.get("exit_reason", "")).strip() for row in exit_rows), f"exit_rows={len(exit_rows)}"))
    checks.append(check("start_date_matches_baseline", first_date == EXPECTED_START, str(first_date)))
    checks.append(check("end_date_inside_requested_range", bool(last_date and LAST_EVENT_MIN <= last_date <= EXPECTED_END), str(last_date)))
    checks.append(check("has_setup_rows", len(setup_rows) > 0, f"setup_rows={len(setup_rows)}"))
    accepted = all(item["passed"] for item in checks)
    return {
        "path": str(path),
        "exists": True,
        "accepted": accepted,
        "row_count": len(rows),
        "segments_detected": len(segments),
        "first_date": str(first_date) if first_date else "",
        "last_date": str(last_date) if last_date else "",
        "event_counts": event_counts,
        "lot_counts": lot_counts,
        "checks": checks,
    }


def build_readiness(validation: dict[str, Any], comparison_report: dict[str, Any] | None) -> dict[str, Any]:
    if not validation["accepted"]:
        return {
            "status": "clean_csv_not_accepted",
            "final_comparison_ready": False,
            "candidate_promoted": False,
            "next_recommended_sprint": "rerun_mt5_clean_fractal_lifecycle_fix_validation_failures_v1",
        }
    decision = (comparison_report or {}).get("decision", {})
    return {
        "status": "clean_csv_accepted_and_final_comparison_executed",
        "final_comparison_ready": True,
        "candidate_promoted": bool(decision.get("promote_candidate")),
        "recommendation": decision.get("recommendation", ""),
        "next_recommended_sprint": decision.get("next_recommended_sprint", ""),
    }


def write_artifacts(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "fractal_clean_rerun_requirements.md").write_text(requirements_md(report), encoding="utf-8")
    (output_dir / "fractal_clean_csv_validation_report.json").write_text(
        json.dumps(report["validation"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "fractal_final_decision_runner_ready.md").write_text(runner_ready_md(report), encoding="utf-8")
    _write_csv(clean_template_rows(report), output_dir / "fractal_clean_vs_baseline_template.csv")
    (output_dir / "fractal_clean_readiness_assessment.md").write_text(readiness_md(report), encoding="utf-8")


def rerun_requirements() -> dict[str, Any]:
    return {
        "symbol": DEFAULT_SYMBOL,
        "timeframe": "M5",
        "tester_start": str(EXPECTED_START),
        "tester_end": str(EXPECTED_END),
        "LotSize": EXPECTED_LOT,
        "EventLogFileName": EXPECTED_CLEAN_FILE,
        "InpOverwrite": True,
        "required_segments": 1,
        "required_events": list(REQUIRED_EVENTS),
        "notes": "Use the auxiliary fractal lifecycle EA, not the official Azir EA, and start from a clean output file.",
    }


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def clean_template_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    comparison = (report.get("final_comparison") or {}).get("comparison_rows", [])
    if comparison:
        return [
            {
                "metric": row["metric"],
                "baseline_azir_protected_economic_v1": row["baseline_azir_protected_economic_v1"],
                "fractal_clean_risk_applied": row["fractal_risk_applied_normalized_to_baseline_lot"],
                "delta": row["delta_fractal_risk_applied_normalized_vs_baseline"],
            }
            for row in comparison
        ]
    return [
        {"metric": metric, "baseline_azir_protected_economic_v1": "", "fractal_clean_risk_applied": "", "delta": ""}
        for metric in ("closed_trades", "net_pnl", "profit_factor", "expectancy", "max_drawdown_abs")
    ]


def requirements_md(report: dict[str, Any]) -> str:
    req = report["mt5_rerun_requirements"]
    return (
        "# Clean MT5 Fractal Rerun Requirements\n\n"
        f"- Symbol: `{req['symbol']}`\n"
        f"- Timeframe: `{req['timeframe']}`\n"
        f"- Tester range: `{req['tester_start']}` to `{req['tester_end']}`\n"
        f"- LotSize: `{req['LotSize']}`\n"
        f"- Output CSV: `{req['EventLogFileName']}`\n"
        "- Required: one chronological segment, no appended prior tester run, full lifecycle events.\n"
    )


def runner_ready_md(report: dict[str, Any]) -> str:
    validation = report["validation"]
    return (
        "# Fractal Final Decision Runner Ready\n\n"
        f"- Clean CSV accepted: `{validation['accepted']}`\n"
        f"- Clean CSV path: `{validation['path']}`\n"
        f"- Final comparison executed: `{report['readiness']['final_comparison_ready']}`\n"
        "- Runner: `python -m hybrid_quant.azir.fractal_clean_rerun ...`\n"
    )


def readiness_md(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    failed = [item for item in report["validation"]["checks"] if not item["passed"]]
    failed_text = "\n".join(f"- `{item['name']}`: {item['detail']}" for item in failed) or "- None."
    return (
        "# Fractal Clean Readiness Assessment\n\n"
        f"- Status: `{readiness['status']}`\n"
        f"- Candidate promoted: `{readiness['candidate_promoted']}`\n"
        f"- Recommendation: `{readiness.get('recommendation', '')}`\n"
        f"- Next sprint: `{readiness['next_recommended_sprint']}`\n\n"
        "## Failed Checks\n\n"
        f"{failed_text}\n"
    )


def file_info(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "sprint": report["sprint"],
        "clean_csv_accepted": report["validation"]["accepted"],
        "candidate_promoted": report["readiness"]["candidate_promoted"],
        "recommendation": report["readiness"].get("recommendation", ""),
        "next_recommended_sprint": report["readiness"]["next_recommended_sprint"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
