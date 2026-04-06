from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from hybrid_quant.data import read_ohlcv_frame

from .diagnostics import BaselineDiagnosticsRunner
from .runner import BaselineRunner


DEFAULT_VARIANTS = (
    "baseline_trend_nasdaq",
    "baseline_trend_nasdaq_v2",
    "baseline_trend_nasdaq_v2_long_only",
    "baseline_trend_nasdaq_v2_short_only",
)


@dataclass(slots=True)
class RefinementVariantArtifacts:
    variant_name: str
    artifact_dir: Path
    diagnostics_dir: Path
    report_path: Path
    diagnostics_path: Path
    summary_path: Path
    metrics: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the trend Nasdaq refinement comparison suite.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-gaps", action="store_true")
    parser.add_argument("--variant", action="append", dest="variants")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    variants = tuple(args.variants or DEFAULT_VARIANTS)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_ohlcv_frame(args.input_path)

    variant_artifacts = _run_variants(
        config_dir=args.config_dir,
        frame=frame,
        output_dir=output_dir,
        variants=variants,
        allow_gaps=args.allow_gaps,
    )
    payload = _build_comparison_payload(
        input_path=args.input_path,
        frame=frame,
        variant_artifacts=variant_artifacts,
        variant_names=variants,
    )

    comparison_path = output_dir / "trend_nasdaq_refinement_comparison.json"
    summary_path = output_dir / "trend_nasdaq_refinement_summary.md"
    comparison_path.write_text(json.dumps(_sanitize_value(payload), indent=2), encoding="utf-8")
    summary_path.write_text(_build_summary_markdown(payload), encoding="utf-8")

    _export_root_aliases(output_dir=output_dir, variant_artifacts=variant_artifacts)

    best_variant_name, best_variant = max(
        payload["variants"].items(),
        key=lambda item: (float(item[1]["net_pnl"]), -float(item[1]["max_drawdown"])),
    )
    print(f"Trend refinement comparison: {comparison_path}")
    print(f"Trend refinement summary: {summary_path}")
    print(
        " ".join(
            [
                f"selected_variant={best_variant_name}",
                f"trades={best_variant['number_of_trades']}",
                f"net_pnl={best_variant['net_pnl']}",
                f"drawdown={best_variant['max_drawdown']}",
            ]
        )
    )
    return 0


def _run_variants(
    *,
    config_dir: str | Path,
    frame,
    output_dir: Path,
    variants: Sequence[str],
    allow_gaps: bool,
) -> dict[str, RefinementVariantArtifacts]:
    artifacts: dict[str, RefinementVariantArtifacts] = {}
    for variant_name in variants:
        runner = BaselineRunner.from_config(config_dir, variant_name=variant_name)
        artifact_dir = output_dir / variant_name
        diagnostics_dir = output_dir / f"{variant_name}_diagnostics"
        baseline_artifacts = runner.run(
            output_dir=artifact_dir,
            input_frame=frame,
            allow_gaps=allow_gaps or runner.application.settings.data.allow_gaps,
        )
        diagnostics_runner = BaselineDiagnosticsRunner(runner.application)
        diagnostics_artifacts = diagnostics_runner.run(
            artifact_dir=artifact_dir,
            output_dir=diagnostics_dir,
        )
        diagnostics_payload = json.loads(diagnostics_artifacts.diagnostics_path.read_text(encoding="utf-8"))
        metrics = _build_variant_metrics(
            variant_name=variant_name,
            report_payload=json.loads(baseline_artifacts.report_path.read_text(encoding="utf-8")),
            diagnostics_payload=diagnostics_payload,
        )
        artifacts[variant_name] = RefinementVariantArtifacts(
            variant_name=variant_name,
            artifact_dir=artifact_dir,
            diagnostics_dir=diagnostics_dir,
            report_path=baseline_artifacts.report_path,
            diagnostics_path=diagnostics_artifacts.diagnostics_path,
            summary_path=diagnostics_artifacts.summary_path,
            metrics=metrics,
        )
    return artifacts


def _build_variant_metrics(
    *,
    variant_name: str,
    report_payload: dict[str, Any],
    diagnostics_payload: dict[str, Any],
) -> dict[str, Any]:
    baseline_metrics = diagnostics_payload["baseline_metrics"]
    return {
        "variant": variant_name,
        "symbol": report_payload["symbol"],
        "execution_timeframe": report_payload["execution_timeframe"],
        "filter_timeframe": report_payload["filter_timeframe"],
        "strategy_family": baseline_metrics["strategy_family"],
        "number_of_trades": int(baseline_metrics["number_of_trades"]),
        "win_rate": float(baseline_metrics["win_rate"]),
        "average_win": float(baseline_metrics["average_win"]),
        "average_loss": float(baseline_metrics["average_loss"]),
        "payoff": float(baseline_metrics["payoff_real"]),
        "expectancy": float(baseline_metrics["expectancy"]),
        "gross_pnl": float(baseline_metrics["gross_pnl"]),
        "net_pnl": float(baseline_metrics["net_pnl"]),
        "fees_paid": float(baseline_metrics["fees_paid_total"]),
        "max_drawdown": float(baseline_metrics["max_drawdown"]),
        "sharpe": float(baseline_metrics["sharpe"]),
        "sortino": float(baseline_metrics["sortino"]),
        "calmar": float(baseline_metrics["calmar"]),
        "profitable_months_pct": float(baseline_metrics["profitable_months_pct"]),
        "max_consecutive_losses": int(baseline_metrics["max_consecutive_losses"]),
        "average_holding_bars": float(baseline_metrics["average_holding_bars"]),
        "estimated_fee_drag": float(baseline_metrics["estimated_fee_drag"] or 0.0),
        "estimated_slippage_drag": float(baseline_metrics["estimated_slippage_drag"] or 0.0),
        "estimated_total_cost_drag": float(baseline_metrics["estimated_total_cost_drag"] or 0.0),
        "report_path": report_payload.get("report_path"),
        "validation": report_payload["validation"],
    }


def _build_comparison_payload(
    *,
    input_path: str,
    frame,
    variant_artifacts: dict[str, RefinementVariantArtifacts],
    variant_names: Sequence[str],
) -> dict[str, Any]:
    variants = {name: _sanitize_value(artifact.metrics) for name, artifact in variant_artifacts.items()}
    pair_deltas = {
        "baseline_trend_nasdaq_v2_vs_baseline_trend_nasdaq": _build_delta_payload(
            variants.get("baseline_trend_nasdaq"),
            variants.get("baseline_trend_nasdaq_v2"),
        ),
        "baseline_trend_nasdaq_v2_long_only_vs_baseline_trend_nasdaq_v2": _build_delta_payload(
            variants.get("baseline_trend_nasdaq_v2"),
            variants.get("baseline_trend_nasdaq_v2_long_only"),
        ),
        "baseline_trend_nasdaq_v2_short_only_vs_baseline_trend_nasdaq_v2": _build_delta_payload(
            variants.get("baseline_trend_nasdaq_v2"),
            variants.get("baseline_trend_nasdaq_v2_short_only"),
        ),
        "baseline_trend_nasdaq_v2_short_only_vs_baseline_trend_nasdaq": _build_delta_payload(
            variants.get("baseline_trend_nasdaq"),
            variants.get("baseline_trend_nasdaq_v2_short_only"),
        ),
    }
    conclusion = _build_conclusion(variants=variants, pair_deltas=pair_deltas)
    return {
        "comparison_mode": "baseline_runner_with_risk_engine",
        "input_path": str(input_path),
        "period": {
            "symbol": "NQ",
            "execution_timeframe": "5m",
            "filter_timeframe": "1H",
            "start": frame.index[0].isoformat() if not frame.empty else None,
            "end": frame.index[-1].isoformat() if not frame.empty else None,
            "bars": int(len(frame)),
        },
        "variants": variants,
        "pair_deltas": _sanitize_value(pair_deltas),
        "conclusion": conclusion,
        "variant_order": list(variant_names),
    }


def _build_delta_payload(
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if baseline is None or candidate is None:
        return None

    baseline_trades = float(baseline["number_of_trades"])
    baseline_fees = float(baseline["fees_paid"])
    baseline_drawdown = float(baseline["max_drawdown"])
    return {
        "trade_count_delta": int(candidate["number_of_trades"] - baseline["number_of_trades"]),
        "trade_reduction_pct": ((baseline_trades - float(candidate["number_of_trades"])) / baseline_trades)
        if baseline_trades > 0.0
        else 0.0,
        "win_rate_delta": float(candidate["win_rate"] - baseline["win_rate"]),
        "payoff_delta": float(candidate["payoff"] - baseline["payoff"]),
        "expectancy_delta": float(candidate["expectancy"] - baseline["expectancy"]),
        "gross_pnl_delta": float(candidate["gross_pnl"] - baseline["gross_pnl"]),
        "net_pnl_delta": float(candidate["net_pnl"] - baseline["net_pnl"]),
        "fees_paid_delta": float(candidate["fees_paid"] - baseline["fees_paid"]),
        "fee_reduction_pct": ((baseline_fees - float(candidate["fees_paid"])) / baseline_fees)
        if baseline_fees > 0.0
        else 0.0,
        "max_drawdown_delta": float(candidate["max_drawdown"] - baseline["max_drawdown"]),
        "max_drawdown_reduction_pct": ((baseline_drawdown - float(candidate["max_drawdown"])) / baseline_drawdown)
        if baseline_drawdown > 0.0
        else 0.0,
        "sharpe_delta": float(candidate["sharpe"] - baseline["sharpe"]),
        "sortino_delta": float(candidate["sortino"] - baseline["sortino"]),
        "calmar_delta": float(candidate["calmar"] - baseline["calmar"]),
        "profitable_months_pct_delta": float(candidate["profitable_months_pct"] - baseline["profitable_months_pct"]),
        "max_consecutive_losses_delta": int(candidate["max_consecutive_losses"] - baseline["max_consecutive_losses"]),
    }


def _build_conclusion(
    *,
    variants: dict[str, dict[str, Any]],
    pair_deltas: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    baseline = variants.get("baseline_trend_nasdaq")
    v2 = variants.get("baseline_trend_nasdaq_v2")
    long_only = variants.get("baseline_trend_nasdaq_v2_long_only")
    short_only = variants.get("baseline_trend_nasdaq_v2_short_only")

    if baseline is None or v2 is None:
        return {
            "headline": "Trend refinement comparison is incomplete because one or more required variants are missing.",
            "v2_merits_robust_validation": False,
            "strongest_variant": None,
        }

    strongest_name, strongest_metrics = max(
        variants.items(),
        key=lambda item: (float(item[1]["net_pnl"]), -float(item[1]["max_drawdown"])),
    )

    v2_improves_net = float(v2["net_pnl"]) > float(baseline["net_pnl"])
    v2_improves_gross = float(v2["gross_pnl"]) > float(baseline["gross_pnl"])
    v2_reduces_drawdown = float(v2["max_drawdown"]) <= float(baseline["max_drawdown"])
    v2_reduces_trades = int(v2["number_of_trades"]) < int(baseline["number_of_trades"])
    v2_still_negative = float(v2["net_pnl"]) < 0.0

    if v2_improves_net and v2_reduces_drawdown and v2_reduces_trades:
        headline = (
            "`baseline_trend_nasdaq_v2` improves the original trend baseline materially, "
            "but it still does not clear break-even and should not move to robust validation yet."
        )
    else:
        headline = (
            "`baseline_trend_nasdaq_v2` does not improve the original baseline enough to justify a "
            "robust-validation sprint yet."
        )

    directional_note = None
    if long_only is not None and short_only is not None:
        if float(short_only["net_pnl"]) > float(long_only["net_pnl"]):
            directional_note = (
                "After the hour/quality refinement, the short-only diagnostic slice is stronger than the long-only one."
            )
        else:
            directional_note = (
                "After the hour/quality refinement, the long-only diagnostic slice is stronger than the short-only one."
            )

    top_changes = [
        (
            f"`baseline_trend_nasdaq_v2` changed net PnL by "
            f"`{pair_deltas['baseline_trend_nasdaq_v2_vs_baseline_trend_nasdaq']['net_pnl_delta']:.2f}` "
            f"while cutting trades by "
            f"`{pair_deltas['baseline_trend_nasdaq_v2_vs_baseline_trend_nasdaq']['trade_reduction_pct'] * 100:.2f}%`."
        ),
        (
            f"`baseline_trend_nasdaq_v2` changed gross PnL by "
            f"`{pair_deltas['baseline_trend_nasdaq_v2_vs_baseline_trend_nasdaq']['gross_pnl_delta']:.2f}` "
            f"and max drawdown by "
            f"`{pair_deltas['baseline_trend_nasdaq_v2_vs_baseline_trend_nasdaq']['max_drawdown_delta']:.4f}`."
        ),
        (
            f"The strongest requested diagnostic variant is `{strongest_name}` with net PnL "
            f"`{strongest_metrics['net_pnl']:.2f}` and drawdown `{strongest_metrics['max_drawdown']:.4f}`."
        ),
    ]
    if directional_note is not None:
        top_changes.append(directional_note)

    return {
        "headline": headline,
        "top_changes": top_changes,
        "v2_improves_net_pnl": v2_improves_net,
        "v2_improves_gross_pnl": v2_improves_gross,
        "v2_reduces_drawdown": v2_reduces_drawdown,
        "v2_reduces_trades": v2_reduces_trades,
        "v2_merits_robust_validation": (
            v2_improves_net
            and v2_improves_gross
            and v2_reduces_drawdown
            and not v2_still_negative
        ),
        "strongest_variant": strongest_name,
        "strongest_variant_metrics": strongest_metrics,
        "next_step": (
            "Use the diagnostic directional winner as the hypothesis for one more very small rule-based refinement "
            "before starting a robust-validation sprint."
            if strongest_name != "baseline_trend_nasdaq_v2"
            else "Run a dedicated diagnostic on baseline_trend_nasdaq_v2 and only then consider robust validation."
        ),
    }


def _build_summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Trend Nasdaq Refinement",
        "",
        "## Setup",
        f"- Input: `{payload['input_path']}`",
        f"- Period: `{payload['period']['start']}` -> `{payload['period']['end']}`",
        f"- Bars: `{payload['period']['bars']}`",
        "",
        "## Direct Answer",
        payload["conclusion"]["headline"],
    ]

    for variant_name in payload["variant_order"]:
        metrics = payload["variants"][variant_name]
        lines.extend(
            [
                "",
                f"## {variant_name}",
                f"- Trades: `{metrics['number_of_trades']}`",
                f"- Win rate: `{metrics['win_rate'] * 100:.2f}%`",
                f"- Average win: `{metrics['average_win']:.2f}`",
                f"- Average loss: `{metrics['average_loss']:.2f}`",
                f"- Payoff: `{metrics['payoff']:.4f}`",
                f"- Expectancy: `{metrics['expectancy']:.2f}`",
                f"- Gross PnL: `{metrics['gross_pnl']:.2f}`",
                f"- Net PnL: `{metrics['net_pnl']:.2f}`",
                f"- Fees paid: `{metrics['fees_paid']:.2f}`",
                f"- Max drawdown: `{metrics['max_drawdown']:.4f}`",
                f"- Sharpe: `{metrics['sharpe']:.4f}`",
                f"- Sortino: `{metrics['sortino']:.4f}`",
                f"- Calmar: `{metrics['calmar']:.4f}`",
                f"- Profitable months pct: `{metrics['profitable_months_pct'] * 100:.2f}%`",
                f"- Max consecutive losses: `{metrics['max_consecutive_losses']}`",
            ]
        )

    lines.extend(["", "## Pair Deltas"])
    for label, delta in payload["pair_deltas"].items():
        if delta is None:
            continue
        lines.extend(
            [
                f"- `{label}`: net PnL delta `{delta['net_pnl_delta']:.2f}`, gross PnL delta `{delta['gross_pnl_delta']:.2f}`, trade reduction `{delta['trade_reduction_pct'] * 100:.2f}%`, drawdown delta `{delta['max_drawdown_delta']:.4f}`.",
            ]
        )

    lines.extend(["", "## Key Takeaways"])
    for item in payload["conclusion"]["top_changes"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Verdict",
            f"- `baseline_trend_nasdaq_v2` merits robust validation now: `{payload['conclusion']['v2_merits_robust_validation']}`",
            f"- Strongest tested variant on this slice: `{payload['conclusion']['strongest_variant']}`",
            f"- Recommended next step: {payload['conclusion']['next_step']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _export_root_aliases(
    *,
    output_dir: Path,
    variant_artifacts: dict[str, RefinementVariantArtifacts],
) -> None:
    for variant_name, artifact in variant_artifacts.items():
        shutil.copyfile(artifact.report_path, output_dir / f"{variant_name}_report.json")
        for source_name, target_name in {
            "monthly_breakdown.csv": f"{variant_name}_monthly_breakdown.csv",
            "hourly_breakdown.csv": f"{variant_name}_hourly_breakdown.csv",
            "exit_reason_breakdown.csv": f"{variant_name}_exit_reason_breakdown.csv",
            "side_breakdown.csv": f"{variant_name}_side_breakdown.csv",
        }.items():
            source = artifact.diagnostics_dir / source_name
            if source.exists():
                shutil.copyfile(source, output_dir / target_name)


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_value(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


if __name__ == "__main__":
    raise SystemExit(main())
