"""Prepare MT5 full-lifecycle export validation for the Azir fractal candidate.

This sprint intentionally does not promote the fractal candidate.  It creates a
clean MT5 auxiliary-export path and records what evidence is still missing
before a final protected economic comparison can be made.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .event_log import AZIR_EVENT_COLUMNS


SPRINT_NAME = "mt5_fractal_candidate_full_lifecycle_export_or_tick_gap_closure_v1"
FULL_LIFECYCLE_EA_NAME = "AzirFractalCandidateFullLifecycleExport.mq5"
FULL_LIFECYCLE_LOG_NAME = "fractal_candidate_full_lifecycle_event_log.csv"
CANONICAL_LIFECYCLE_EVENTS = (
    "opportunity",
    "fill",
    "trailing_modified",
    "opposite_pending_cancelled",
    "exit",
    "no_fill_at_close",
    "friday_blocked",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Azir fractal candidate MT5 full-lifecycle export artifacts.")
    parser.add_argument("--current-log-path", required=True)
    parser.add_argument("--candidate-setup-log-path", required=True)
    parser.add_argument("--m5-input-path", required=True)
    parser.add_argument("--m1-input-path", required=True)
    parser.add_argument("--tick-input-path", required=True)
    parser.add_argument("--forced-close-report-path", required=True)
    parser.add_argument("--fractal-protected-report-path", required=True)
    parser.add_argument("--fractal-tick-replay-report-path", required=True)
    parser.add_argument("--mql5-ea-path", required=True)
    parser.add_argument("--mql5-logger-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--full-lifecycle-log-path", default="")
    parser.add_argument("--symbol", default="XAUUSD-STD")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_fractal_full_lifecycle_export_assessment(
        current_log_path=Path(args.current_log_path),
        candidate_setup_log_path=Path(args.candidate_setup_log_path),
        m5_input_path=Path(args.m5_input_path),
        m1_input_path=Path(args.m1_input_path),
        tick_input_path=Path(args.tick_input_path),
        forced_close_report_path=Path(args.forced_close_report_path),
        fractal_protected_report_path=Path(args.fractal_protected_report_path),
        fractal_tick_replay_report_path=Path(args.fractal_tick_replay_report_path),
        mql5_ea_path=Path(args.mql5_ea_path),
        mql5_logger_path=Path(args.mql5_logger_path),
        output_dir=Path(args.output_dir),
        full_lifecycle_log_path=Path(args.full_lifecycle_log_path) if args.full_lifecycle_log_path else None,
        symbol=args.symbol,
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_fractal_full_lifecycle_export_assessment(
    *,
    current_log_path: Path,
    candidate_setup_log_path: Path,
    m5_input_path: Path,
    m1_input_path: Path,
    tick_input_path: Path,
    forced_close_report_path: Path,
    fractal_protected_report_path: Path,
    fractal_tick_replay_report_path: Path,
    mql5_ea_path: Path,
    mql5_logger_path: Path,
    output_dir: Path,
    full_lifecycle_log_path: Path | None = None,
    symbol: str = "XAUUSD-STD",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    previous_tick_report = _read_json(fractal_tick_replay_report_path)
    previous_coverage = previous_tick_report.get("coverage", {})
    previous_metrics = previous_tick_report.get("candidate_tick_metrics", {})
    full_lifecycle_inspection = inspect_event_log(full_lifecycle_log_path, symbol) if full_lifecycle_log_path else {
        "path": "",
        "exists": False,
        "reason": "No full lifecycle log path supplied yet. Run the auxiliary MT5 EA first.",
    }
    current_log_inspection = inspect_event_log(current_log_path, symbol)
    candidate_setup_inspection = inspect_event_log(candidate_setup_log_path, symbol)

    source_files = {
        "current_log": inspect_file(current_log_path),
        "candidate_setup_log": inspect_file(candidate_setup_log_path),
        "m5_input": inspect_file(m5_input_path),
        "m1_input": inspect_file(m1_input_path),
        "tick_input": inspect_file(tick_input_path),
        "forced_close_report": inspect_file(forced_close_report_path),
        "fractal_protected_report": inspect_file(fractal_protected_report_path),
        "fractal_tick_replay_report": inspect_file(fractal_tick_replay_report_path),
        "mql5_ea": inspect_file(mql5_ea_path),
        "mql5_logger": inspect_file(mql5_logger_path),
    }
    missing_sources = [name for name, meta in source_files.items() if not meta["exists"]]
    real_lifecycle_ready = _has_real_lifecycle(full_lifecycle_inspection)
    gap_rows = build_tick_gap_closure_rows(previous_coverage, full_lifecycle_inspection)
    readiness = build_readiness(
        missing_sources=missing_sources,
        full_lifecycle_inspection=full_lifecycle_inspection,
        previous_coverage=previous_coverage,
        real_lifecycle_ready=real_lifecycle_ready,
    )

    report = {
        "sprint": SPRINT_NAME,
        "symbol": symbol,
        "decision": {
            "candidate_promoted": False,
            "reason": "This sprint implements the MT5 lifecycle export path. Promotion still requires a real full-lifecycle candidate log and final economic comparison.",
            "next_recommended_sprint": readiness["next_recommended_sprint"],
        },
        "sources": source_files,
        "missing_sources": missing_sources,
        "mt5_export_path": {
            "ea_path": str(mql5_ea_path),
            "logger_include_path": str(mql5_logger_path),
            "expected_output_file": FULL_LIFECYCLE_LOG_NAME,
            "event_log_use_common_folder_default": True,
            "definition": "swing_10_fractal: latest confirmed M5 pivot high/low inside the last 10 closed M5 bars, 2 bars left and 2 bars right, rolling high/low fallback.",
            "official_azir_touched": False,
        },
        "current_log_inspection": current_log_inspection,
        "candidate_setup_log_inspection": candidate_setup_inspection,
        "full_lifecycle_log_inspection": full_lifecycle_inspection,
        "previous_tick_replay": {
            "report_path": str(fractal_tick_replay_report_path),
            "coverage": previous_coverage,
            "candidate_metrics": previous_metrics,
        },
        "gap_closure": {
            "rows": gap_rows,
            "new_tick_coverage_added_in_this_sprint": 0,
            "reason": "The valuable path is MT5 full-lifecycle export. Tick/M1/M5 gap closure remains unchanged until a real candidate lifecycle log is generated.",
        },
        "readiness": readiness,
        "limitations": build_limitations(full_lifecycle_inspection, previous_coverage),
    }

    write_csv(gap_rows, output_dir / "fractal_tick_gap_closure_report.csv")
    write_lifecycle_sample(output_dir / "fractal_lifecycle_log_sample.csv")
    (output_dir / "fractal_lifecycle_schema.md").write_text(schema_markdown(), encoding="utf-8")
    (output_dir / "fractal_lifecycle_readiness_assessment.md").write_text(readiness_markdown(report), encoding="utf-8")
    (output_dir / "fractal_full_lifecycle_export_summary.md").write_text(summary_markdown(report), encoding="utf-8")
    (output_dir / "fractal_full_lifecycle_export_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def inspect_file(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
    }


def inspect_event_log(path: Path | None, symbol: str) -> dict[str, Any]:
    if path is None:
        return {"path": "", "exists": False}
    if not path.exists():
        return {"path": str(path), "exists": False}

    event_counts: Counter[str] = Counter()
    first_timestamp = ""
    last_timestamp = ""
    columns: list[str] = []
    rows = 0
    symbol_rows = 0
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        for row in reader:
            rows += 1
            if row.get("symbol", symbol) != symbol:
                continue
            symbol_rows += 1
            timestamp = row.get("timestamp", "")
            if timestamp:
                if not first_timestamp:
                    first_timestamp = timestamp
                last_timestamp = timestamp
            event_counts[row.get("event_type", "")] += 1

    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "columns": columns,
        "schema_compatible": set(AZIR_EVENT_COLUMNS).issubset(set(columns)),
        "rows": rows,
        "symbol_rows": symbol_rows,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "event_counts": dict(event_counts),
    }


def build_tick_gap_closure_rows(previous_coverage: dict[str, Any], full_lifecycle_inspection: dict[str, Any]) -> list[dict[str, Any]]:
    previous_priced = int(previous_coverage.get("closed_trades_priced", 0) or 0)
    previous_tick = int(previous_coverage.get("tick_priced_trades", 0) or 0)
    previous_m1 = int(previous_coverage.get("m1_fallback_trades", 0) or 0)
    previous_m5 = int(previous_coverage.get("m5_fallback_trades", 0) or 0)
    previous_unpriced = int(previous_coverage.get("unpriced_trades", 0) or 0)
    lifecycle_rows = int(full_lifecycle_inspection.get("symbol_rows", 0) or 0)
    has_real_lifecycle = _has_real_lifecycle(full_lifecycle_inspection)
    return [
        {
            "layer": "previous_tick_m1_m5_replay",
            "priced_trades": previous_priced,
            "tick_priced_trades": previous_tick,
            "m1_fallback_trades": previous_m1,
            "m5_fallback_trades": previous_m5,
            "unpriced_trades": previous_unpriced,
            "real_mt5_lifecycle_rows": 0,
            "status": "baseline_before_this_sprint",
            "notes": "State from fractal_tick_replay_report.json before the MT5 lifecycle exporter is run.",
        },
        {
            "layer": "mt5_full_lifecycle_exporter",
            "priced_trades": "",
            "tick_priced_trades": "",
            "m1_fallback_trades": "",
            "m5_fallback_trades": "",
            "unpriced_trades": "",
            "real_mt5_lifecycle_rows": lifecycle_rows,
            "status": "real_lifecycle_log_available" if has_real_lifecycle else "exporter_ready_log_not_generated",
            "notes": "Run the auxiliary EA in MT5 Strategy Tester to replace proxy lifecycle evidence with real fill/trailing/exit events.",
        },
    ]


def build_readiness(
    *,
    missing_sources: list[str],
    full_lifecycle_inspection: dict[str, Any],
    previous_coverage: dict[str, Any],
    real_lifecycle_ready: bool,
) -> dict[str, Any]:
    m5_fallback = int(previous_coverage.get("m5_fallback_trades", 0) or 0)
    if missing_sources:
        status = "blocked_missing_inputs"
        next_sprint = "fix_missing_inputs_for_fractal_lifecycle_export_v1"
    elif not real_lifecycle_ready:
        status = "exporter_ready_waiting_for_mt5_run"
        next_sprint = "run_mt5_fractal_lifecycle_export_then_compare_v1"
    else:
        status = "ready_for_final_candidate_economic_comparison"
        next_sprint = "compare_fractal_full_lifecycle_mt5_vs_baseline_protected_v1"

    return {
        "status": status,
        "missing_sources": missing_sources,
        "real_full_lifecycle_log_available": real_lifecycle_ready,
        "previous_m5_fallback_trades": m5_fallback,
        "can_freeze_or_promote_candidate_now": False,
        "next_recommended_sprint": next_sprint,
    }


def build_limitations(full_lifecycle_inspection: dict[str, Any], previous_coverage: dict[str, Any]) -> list[str]:
    limitations = [
        "The official Azir EA is not modified; this is an auxiliary candidate export path.",
        "No candidate economic benchmark can be frozen until the auxiliary EA is run in MT5 and its lifecycle log is compared to the protected baseline.",
    ]
    if not _has_real_lifecycle(full_lifecycle_inspection):
        limitations.append("No real MT5 full-lifecycle candidate log was available during this run.")
    m5_fallback = int(previous_coverage.get("m5_fallback_trades", 0) or 0)
    if m5_fallback > 0:
        limitations.append(f"Previous replay still had {m5_fallback} trades valued with M5 fallback.")
    return limitations


def _has_real_lifecycle(inspection: dict[str, Any]) -> bool:
    counts = inspection.get("event_counts", {})
    return bool(inspection.get("exists")) and int(counts.get("fill", 0) or 0) > 0 and int(counts.get("exit", 0) or 0) > 0


def write_lifecycle_sample(path: Path) -> None:
    rows = []
    for event_type in ("opportunity", "fill", "trailing_modified", "exit"):
        row = {column: "" for column in AZIR_EVENT_COLUMNS}
        row.update(
            {
                "timestamp": "2025.01.06 16:30:00",
                "event_id": "2025-01-06_XAUUSD-STD_123456321",
                "event_type": event_type,
                "symbol": "XAUUSD-STD",
                "magic": "123456321",
                "timeframe": "M5",
                "swing_bars": "10",
                "notes": f"schema sample for {FULL_LIFECYCLE_LOG_NAME}",
            }
        )
        rows.append(row)
    write_csv(rows, path)


def schema_markdown() -> str:
    event_lines = "\n".join(f"- `{event}`" for event in CANONICAL_LIFECYCLE_EVENTS)
    column_lines = "\n".join(f"- `{column}`" for column in AZIR_EVENT_COLUMNS)
    return (
        "# Fractal Candidate Lifecycle Schema\n\n"
        "The auxiliary MT5 EA writes the same canonical Azir event-log columns as the official logger.\n\n"
        "## Expected Event Types\n"
        f"{event_lines}\n\n"
        "## Columns\n"
        f"{column_lines}\n\n"
        "## Candidate Definition\n"
        "`swing_10_fractal` uses the latest confirmed M5 pivot high/low inside the last 10 closed M5 bars, "
        "with 2 bars to the left and 2 bars to the right. If no confirmed pivot exists, it falls back to the "
        "official rolling high/low over the same 10 closed bars. EMA20, ATR, RSI gate, Friday filter, offset, "
        "SL/TP, trailing and close-hour behavior remain aligned with Azir.\n"
    )


def summary_markdown(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    previous = report["previous_tick_replay"]["coverage"]
    return (
        "# Fractal Full Lifecycle Export Summary\n\n"
        f"- Sprint: `{SPRINT_NAME}`\n"
        f"- Auxiliary EA: `{report['mt5_export_path']['ea_path']}`\n"
        f"- Real lifecycle log available now: `{readiness['real_full_lifecycle_log_available']}`\n"
        f"- Previous tick-priced trades: `{previous.get('tick_priced_trades', 0)}`\n"
        f"- Previous M1 fallback trades: `{previous.get('m1_fallback_trades', 0)}`\n"
        f"- Previous M5 fallback trades: `{previous.get('m5_fallback_trades', 0)}`\n"
        f"- Candidate promoted now: `false`\n\n"
        "## MT5 Usage\n"
        "Compile `AzirFractalCandidateFullLifecycleExport.mq5` together with `AzirEventLogger.mqh`, then run it in MT5 Strategy Tester on `XAUUSD-STD`, M5, over the same historical range as Azir. "
        f"Keep `EventLogFileName={FULL_LIFECYCLE_LOG_NAME}` and `EventLogUseCommonFolder=true` unless you deliberately choose another export location.\n\n"
        "## Decision\n"
        f"{report['decision']['reason']}\n\n"
        f"Next recommended sprint: `{readiness['next_recommended_sprint']}`.\n"
    )


def readiness_markdown(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    return (
        "# Fractal Lifecycle Readiness Assessment\n\n"
        f"- Status: `{readiness['status']}`\n"
        f"- Missing sources: `{', '.join(readiness['missing_sources']) if readiness['missing_sources'] else 'none'}`\n"
        f"- Real full lifecycle log available: `{readiness['real_full_lifecycle_log_available']}`\n"
        f"- Can freeze/promote now: `{readiness['can_freeze_or_promote_candidate_now']}`\n"
        f"- Next sprint: `{readiness['next_recommended_sprint']}`\n\n"
        "## Limitations\n"
        f"{limitations}\n"
    )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    readiness = report["readiness"]
    previous = report["previous_tick_replay"]["coverage"]
    return {
        "sprint": report["sprint"],
        "status": readiness["status"],
        "real_lifecycle_log_available": readiness["real_full_lifecycle_log_available"],
        "previous_tick_priced_trades": previous.get("tick_priced_trades", 0),
        "previous_m1_fallback_trades": previous.get("m1_fallback_trades", 0),
        "previous_m5_fallback_trades": previous.get("m5_fallback_trades", 0),
        "next_recommended_sprint": readiness["next_recommended_sprint"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
