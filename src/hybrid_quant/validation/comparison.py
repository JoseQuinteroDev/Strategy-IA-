from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


CLASSIFICATION_SCORE = {
    "NO-GO": 0,
    "GO WITH CAUTION": 1,
    "GO": 2,
}


@dataclass(slots=True)
class RobustnessComparisonArtifacts:
    output_dir: Path
    comparison_path: Path
    summary_path: Path
    comparison: dict[str, Any]


class RobustnessComparisonRunner:
    def compare(
        self,
        *,
        baseline_report_path: str | Path,
        extended_report_path: str | Path,
        output_dir: str | Path,
        baseline_label: str = "q1_2024",
        extended_label: str = "extended_sample",
    ) -> RobustnessComparisonArtifacts:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        baseline_report = json.loads(Path(baseline_report_path).read_text(encoding="utf-8"))
        extended_report = json.loads(Path(extended_report_path).read_text(encoding="utf-8"))

        baseline_snapshot = self._snapshot(label=baseline_label, report=baseline_report)
        extended_snapshot = self._snapshot(label=extended_label, report=extended_report)
        delta = self._delta(baseline_snapshot, extended_snapshot)
        conclusion = self._conclusion(baseline_snapshot, extended_snapshot, delta)

        comparison = {
            "baseline_label": baseline_label,
            "extended_label": extended_label,
            "variant": extended_report.get("variant") or baseline_report.get("variant"),
            "runs": {
                baseline_label: baseline_snapshot,
                extended_label: extended_snapshot,
            },
            "delta": delta,
            "conclusion": conclusion,
        }

        comparison_path = output_path / "robustness_comparison.json"
        summary_path = output_path / "robustness_comparison_summary.md"
        comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        summary_path.write_text(self._summary_markdown(comparison), encoding="utf-8")

        return RobustnessComparisonArtifacts(
            output_dir=output_path,
            comparison_path=comparison_path,
            summary_path=summary_path,
            comparison=comparison,
        )

    def _snapshot(self, *, label: str, report: dict[str, Any]) -> dict[str, Any]:
        full_metrics = report["full_dataset"]["metrics"]
        walk_forward = report["walk_forward"]["summary"]
        temporal = report["temporal_blocks"]["summary"]
        monte_carlo = report["monte_carlo"]
        cost_summary = report["cost_sensitivity"]["summary"]
        decision = report["decision"]
        dataset = report["dataset"]

        return {
            "label": label,
            "start": dataset["start"],
            "end": dataset["end"],
            "bars": int(dataset["bars"]),
            "classification": decision["classification"],
            "full_trades": int(full_metrics["number_of_trades"]),
            "full_net_pnl": float(full_metrics["net_pnl"]),
            "full_gross_pnl": float(full_metrics["gross_pnl"]),
            "full_fees_paid": float(full_metrics["fees_paid"]),
            "full_sharpe": float(full_metrics["sharpe"]),
            "full_max_drawdown": float(full_metrics["max_drawdown"]),
            "walk_forward_test_windows": int(walk_forward["test_windows"]),
            "walk_forward_positive_ratio": float(walk_forward["positive_test_window_ratio"]),
            "walk_forward_total_test_trades": int(walk_forward["total_test_trades"]),
            "walk_forward_total_test_net_pnl": float(walk_forward["total_test_net_pnl"]),
            "walk_forward_mean_test_sharpe": float(walk_forward["mean_test_sharpe"]),
            "temporal_blocks": int(temporal["blocks"]),
            "temporal_positive_ratio": float(temporal["positive_block_ratio"]),
            "monte_carlo_drawdown_p95": float(monte_carlo["max_drawdown"]["p95"]),
            "cost_survival_ratio": float(cost_summary["survival_ratio"]),
            "failed_caution_checks": list(decision.get("failed_caution_checks", [])),
            "failed_go_checks": list(decision.get("failed_go_checks", [])),
            "limitations": list(report.get("limitations", [])),
        }

    def _delta(self, baseline_snapshot: dict[str, Any], extended_snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "classification_score_delta": CLASSIFICATION_SCORE[extended_snapshot["classification"]]
            - CLASSIFICATION_SCORE[baseline_snapshot["classification"]],
            "bars_delta": int(extended_snapshot["bars"] - baseline_snapshot["bars"]),
            "full_trades_delta": int(extended_snapshot["full_trades"] - baseline_snapshot["full_trades"]),
            "full_net_pnl_delta": float(extended_snapshot["full_net_pnl"] - baseline_snapshot["full_net_pnl"]),
            "walk_forward_total_test_trades_delta": int(
                extended_snapshot["walk_forward_total_test_trades"]
                - baseline_snapshot["walk_forward_total_test_trades"]
            ),
            "walk_forward_positive_ratio_delta": float(
                extended_snapshot["walk_forward_positive_ratio"]
                - baseline_snapshot["walk_forward_positive_ratio"]
            ),
            "walk_forward_mean_test_sharpe_delta": float(
                extended_snapshot["walk_forward_mean_test_sharpe"]
                - baseline_snapshot["walk_forward_mean_test_sharpe"]
            ),
            "temporal_positive_ratio_delta": float(
                extended_snapshot["temporal_positive_ratio"]
                - baseline_snapshot["temporal_positive_ratio"]
            ),
            "cost_survival_ratio_delta": float(
                extended_snapshot["cost_survival_ratio"]
                - baseline_snapshot["cost_survival_ratio"]
            ),
        }

    def _conclusion(
        self,
        baseline_snapshot: dict[str, Any],
        extended_snapshot: dict[str, Any],
        delta: dict[str, Any],
    ) -> dict[str, Any]:
        extended_classification = extended_snapshot["classification"]
        evidence_strengthened = bool(
            delta["bars_delta"] > 0
            and delta["walk_forward_total_test_trades_delta"] > 0
            and (
                delta["classification_score_delta"] > 0
                or delta["walk_forward_positive_ratio_delta"] > 0.0
                or delta["walk_forward_mean_test_sharpe_delta"] > 0.0
                or delta["temporal_positive_ratio_delta"] > 0.0
            )
        )

        if extended_classification == "GO":
            direct_answer = "La muestra ampliada ya justifica `GO` y la baseline merece pasar al siguiente nivel de investigacion."
        elif extended_classification == "GO WITH CAUTION":
            direct_answer = (
                "La muestra ampliada eleva la evidencia hasta `GO WITH CAUTION`; todavia hace falta prudencia, "
                "pero ya no seria razonable descartarla antes de la siguiente fase."
            )
        elif evidence_strengthened:
            direct_answer = (
                "La muestra ampliada mejora algo la evidencia, pero no lo suficiente: `baseline_v3` sigue en `NO-GO`."
            )
        else:
            direct_answer = (
                "La muestra ampliada no rescata la hipotesis: `baseline_v3` sigue en `NO-GO` y todavia no merece subir de nivel."
            )

        return {
            "evidence_strengthened": evidence_strengthened,
            "direct_answer": direct_answer,
            "recommended_status": extended_classification,
        }

    def _summary_markdown(self, comparison: dict[str, Any]) -> str:
        baseline_label = comparison["baseline_label"]
        extended_label = comparison["extended_label"]
        baseline = comparison["runs"][baseline_label]
        extended = comparison["runs"][extended_label]
        delta = comparison["delta"]
        conclusion = comparison["conclusion"]

        lines = [
            "# Robustness Comparison",
            "",
            "## Direct Answer",
            f"- Recommended status: `{conclusion['recommended_status']}`",
            f"- {conclusion['direct_answer']}",
            "",
            "## Baseline Window",
            f"- Label: `{baseline_label}`",
            f"- Period: `{baseline['start']}` -> `{baseline['end']}`",
            f"- Bars: `{baseline['bars']}`",
            f"- Classification: `{baseline['classification']}`",
            f"- Full trades: `{baseline['full_trades']}`",
            f"- Walk-forward test trades: `{baseline['walk_forward_total_test_trades']}`",
            f"- Walk-forward positive ratio: `{baseline['walk_forward_positive_ratio'] * 100:.2f}%`",
            f"- Mean test Sharpe: `{baseline['walk_forward_mean_test_sharpe']:.4f}`",
            f"- Temporal positive ratio: `{baseline['temporal_positive_ratio'] * 100:.2f}%`",
            "",
            "## Extended Window",
            f"- Label: `{extended_label}`",
            f"- Period: `{extended['start']}` -> `{extended['end']}`",
            f"- Bars: `{extended['bars']}`",
            f"- Classification: `{extended['classification']}`",
            f"- Full trades: `{extended['full_trades']}`",
            f"- Walk-forward test trades: `{extended['walk_forward_total_test_trades']}`",
            f"- Walk-forward positive ratio: `{extended['walk_forward_positive_ratio'] * 100:.2f}%`",
            f"- Mean test Sharpe: `{extended['walk_forward_mean_test_sharpe']:.4f}`",
            f"- Temporal positive ratio: `{extended['temporal_positive_ratio'] * 100:.2f}%`",
            "",
            "## Delta",
            f"- Bars delta: `{delta['bars_delta']}`",
            f"- Full trades delta: `{delta['full_trades_delta']}`",
            f"- Full net PnL delta: `{delta['full_net_pnl_delta']:.2f}`",
            f"- Walk-forward test trades delta: `{delta['walk_forward_total_test_trades_delta']}`",
            f"- Walk-forward positive ratio delta: `{delta['walk_forward_positive_ratio_delta'] * 100:.2f} pp`",
            f"- Mean test Sharpe delta: `{delta['walk_forward_mean_test_sharpe_delta']:.4f}`",
            f"- Temporal positive ratio delta: `{delta['temporal_positive_ratio_delta'] * 100:.2f} pp`",
            f"- Cost survival ratio delta: `{delta['cost_survival_ratio_delta'] * 100:.2f} pp`",
            "",
            "## Failed Checks",
            f"- {baseline_label} failed caution checks: `{', '.join(baseline['failed_caution_checks']) or 'none'}`",
            f"- {extended_label} failed caution checks: `{', '.join(extended['failed_caution_checks']) or 'none'}`",
            f"- {baseline_label} failed GO checks: `{', '.join(baseline['failed_go_checks']) or 'none'}`",
            f"- {extended_label} failed GO checks: `{', '.join(extended['failed_go_checks']) or 'none'}`",
        ]
        return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two robustness validation reports.")
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--extended-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-label", default="q1_2024")
    parser.add_argument("--extended-label", default="extended_sample")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    runner = RobustnessComparisonRunner()
    artifacts = runner.compare(
        baseline_report_path=args.baseline_report,
        extended_report_path=args.extended_report,
        output_dir=args.output_dir,
        baseline_label=args.baseline_label,
        extended_label=args.extended_label,
    )
    print(f"robustness_comparison={artifacts.comparison_path}")
    print(f"robustness_comparison_summary={artifacts.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
