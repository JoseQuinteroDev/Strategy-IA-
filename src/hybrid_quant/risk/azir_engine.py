"""Azir-specific lifecycle risk engine and anomaly impact evaluator."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hybrid_quant.azir.economic_audit import (
    _read_raw_event_log,
    build_anomaly_reports,
    reconstruct_lifecycles,
)

from .azir_rules import DEFAULT_AZIR_RULES, AzirRiskRule
from .azir_state import AzirRiskConfig, AzirRiskDecision, AzirRiskState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate risk_engine_azir_v1 against MT5 lifecycle anomalies.")
    parser.add_argument("--mt5-log-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbol", default="XAUUSD-STD")
    parser.add_argument("--config-name", default="risk_engine_azir_v1")
    return parser


class AzirRiskEngine:
    """Hard guardrail layer around Azir.

    The engine does not generate signals and does not modify Azir's entry
    logic. It only approves/blocks setup attempts and emits lifecycle cleanup
    actions that an integration layer can execute in MT5/Python.
    """

    def __init__(
        self,
        config: AzirRiskConfig | None = None,
        rules: tuple[AzirRiskRule, ...] = DEFAULT_AZIR_RULES,
    ) -> None:
        self.config = config or AzirRiskConfig()
        self.rules = rules

    def evaluate(self, state: AzirRiskState, *, context: str) -> AzirRiskDecision:
        results = [rule.evaluate(state, self.config, context) for rule in self.rules]
        blockers = tuple(result.code for result in results if result.block)
        actions = _dedupe(action for result in results for action in result.actions)
        warnings = _dedupe(warning for result in results for warning in result.warnings)
        rationale = " ".join(result.rationale for result in results if result.rationale)
        return AzirRiskDecision(
            approved=not blockers,
            reason_code=blockers[0] if blockers else "approved",
            blocked_by=blockers,
            actions=actions,
            warnings=warnings,
            rationale=rationale or "Azir lifecycle state approved by risk_engine_azir_v1.",
            metadata={
                "context": context,
                "config": self.config.name,
                "pending_orders": state.pending_orders,
                "open_positions": state.open_positions,
                "trades_today": state.trades_today,
                "daily_realized_pnl": state.daily_realized_pnl,
                "consecutive_losses_today": state.consecutive_losses_today,
                "spread_points": state.spread_points,
            },
        )


def run_anomaly_evaluation(
    *,
    mt5_log_path: Path,
    output_dir: Path,
    symbol: str = "XAUUSD-STD",
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
    anomalies = build_anomaly_reports(
        rows,
        reconstruction["lifecycles"],
        reconstruction["trades"],
        config.session_fill_start_hour,
        config.session_fill_end_hour,
    )
    evaluation_rows = evaluate_anomalies(anomalies, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(evaluation_rows, output_dir / "azir_risk_anomaly_mitigation.csv")
    report = {
        "risk_engine": asdict(config),
        "source_log": str(mt5_log_path),
        "anomaly_counts": {key: len(value) for key, value in anomalies.items()},
        "mitigation_counts": dict(_count_by(evaluation_rows, "impact")),
        "evaluation": evaluation_rows,
        "decision": {
            "ready_for_reaudit_with_risk_layer": True,
            "ready_for_ppo": False,
            "economic_benchmark_can_be_frozen": False,
            "recommended_next_sprint": "reaudit_azir_with_risk_engine_v1",
        },
    }
    (output_dir / "azir_risk_engine_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "azir_risk_engine_summary.md").write_text(
        _summary_markdown(report),
        encoding="utf-8",
    )
    return report


def evaluate_anomalies(anomalies: dict[str, list[dict[str, Any]]], config: AzirRiskConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for anomaly in anomalies.get("out_of_window_fills", []):
        rows.append(
            {
                "category": "out_of_window_fill",
                "date": anomaly.get("event_day"),
                "setup_day": anomaly.get("setup_day"),
                "impact": "would_prevent",
                "rules": "hard_cancel_all_pendings_at_close|force_reconcile_orders_positions_before_setup",
                "reason": "Prior pending should have been cancelled at close and reconciled before next setup.",
            }
        )
    for anomaly in anomalies.get("friday_exit_events", []):
        rows.append(
            {
                "category": "friday_exit_event",
                "date": anomaly.get("event_day"),
                "setup_day": anomaly.get("setup_day"),
                "impact": "would_prevent_or_force_close",
                "rules": "friday_no_new_trade_plus_close_or_cancel_prior_exposure|hard_cancel_all_pendings_at_close",
                "reason": "Friday guard should cancel/close prior exposure before a Friday exit can occur.",
            }
        )
    for anomaly in anomalies.get("multi_exit_days", []):
        same_setup = len(str(anomaly.get("assigned_setup_days", "")).split("|")) == 1
        rows.append(
            {
                "category": "multi_exit_day",
                "date": anomaly.get("event_day"),
                "setup_day": anomaly.get("assigned_setup_days"),
                "impact": "would_mitigate" if same_setup else "would_prevent_or_mitigate",
                "rules": "block_new_setups_if_any_position_or_pending_exists|cancel_remaining_pendings_after_fill|max_trades_per_day",
                "reason": (
                    "Same-day multi-fill needs post-fill pending cleanup; prior-day overlap needs setup block "
                    "while exposure exists."
                ),
            }
        )
    for anomaly in anomalies.get("open_order_cleanup_issues", []):
        unresolved = anomaly.get("status") == "missing_cleanup_or_unresolved"
        rows.append(
            {
                "category": "open_order_cleanup_issue",
                "date": anomaly.get("setup_day"),
                "setup_day": anomaly.get("setup_day"),
                "impact": "would_prevent" if unresolved else "would_mitigate",
                "rules": "hard_cancel_all_pendings_at_close|force_reconcile_orders_positions_before_setup",
                "reason": "Mandatory cleanup/reconciliation directly targets stale pending lifecycle state.",
            }
        )
    return rows


def _dedupe(values: Any) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(str(value))
    return tuple(result)


def _count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_markdown(report: dict[str, Any]) -> str:
    counts = report["mitigation_counts"]
    rules = report["risk_engine"]
    return (
        "# risk_engine_azir_v1 Impact Evaluation\n\n"
        "## Executive Summary\n\n"
        f"- Would prevent: {counts.get('would_prevent', 0)} anomaly records.\n"
        f"- Would prevent or force close: {counts.get('would_prevent_or_force_close', 0)} anomaly records.\n"
        f"- Would prevent or mitigate: {counts.get('would_prevent_or_mitigate', 0)} anomaly records.\n"
        f"- Would mitigate: {counts.get('would_mitigate', 0)} anomaly records.\n"
        "- This is a policy impact estimate, not a rerun of MT5 execution.\n\n"
        "## Active Guardrails\n\n"
        f"- hard_cancel_all_pendings_at_close: {rules['hard_cancel_all_pendings_at_close']}\n"
        f"- block_new_setups_if_any_position_or_pending_exists: "
        f"{rules['block_new_setups_if_any_position_or_pending_exists']}\n"
        f"- force_reconcile_orders_positions_before_setup: "
        f"{rules['force_reconcile_orders_positions_before_setup']}\n"
        f"- friday_exposure_policy: `{rules['friday_exposure_policy']}`\n"
        f"- max_daily_loss: {rules['max_daily_loss']}\n"
        f"- max_consecutive_losses: {rules['max_consecutive_losses']}\n"
        f"- max_trades_per_day: {rules['max_trades_per_day']}\n"
        f"- max_spread_points: {rules['max_spread_points']}\n\n"
        "## Decision\n\n"
        "- Ready to re-audit Azir with this risk layer.\n"
        "- Not ready for PPO until the risk-layer re-audit confirms lifecycle anomalies are controlled.\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_anomaly_evaluation(
        mt5_log_path=Path(args.mt5_log_path),
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
        config=AzirRiskConfig(name=args.config_name),
    )
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
