from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from hybrid_quant.bootstrap import build_application_from_settings
from hybrid_quant.core import Settings, apply_settings_overrides
from hybrid_quant.data import (
    BinanceHistoricalDownloader,
    HistoricalDataIngestionService,
    ParquetDatasetStore,
    read_ohlcv_frame,
)

from .diagnostics import BaselineDiagnosticsRunner
from .orb_intraday_active_research import (
    _extract_activity_row,
    _filter_frame_by_range,
    _max_inactive_weeks,
    _parse_datetime,
    _pct_periods_with_trade,
    _read_enriched_trades,
    _reference_snapshot,
    _sanitize_value,
    _temporal_snapshot,
)
from .runner import BaselineRunner
from .variants import load_variant_settings, variant_summary_payload


@dataclass(slots=True)
class IntradayContextualThresholds:
    minimum_profit_factor: float = 1.05
    minimum_expectancy: float = 0.0
    maximum_drawdown: float = 0.08
    minimum_positive_year_ratio: float = 0.50
    minimum_positive_quarter_ratio: float = 0.40


@dataclass(slots=True)
class IntradayContextualVariantConfig:
    name: str
    label: str
    candidate: bool
    source_variant: str | None
    overrides: dict[str, Any]


@dataclass(slots=True)
class IntradayContextualResearchConfig:
    name: str
    base_variant: str
    summary_thresholds: IntradayContextualThresholds
    variants: tuple[IntradayContextualVariantConfig, ...]


@dataclass(slots=True)
class IntradayContextualVariantSpec:
    name: str
    label: str
    candidate: bool
    source_variant: str
    settings: Settings


@dataclass(slots=True)
class IntradayContextualVariantArtifacts:
    variant_name: str
    label: str
    candidate: bool
    artifact_dir: Path
    diagnostics_dir: Path
    report_path: Path
    diagnostics_path: Path
    summary_path: Path
    metrics: dict[str, Any]


@dataclass(slots=True)
class IntradayContextualResearchArtifacts:
    output_dir: Path
    comparison_path: Path
    results_path: Path
    summary_path: Path
    activity_summary_path: Path
    yearly_path: Path
    quarterly_path: Path
    ranking_path: Path
    variant_artifacts: dict[str, IntradayContextualVariantArtifacts]


class IntradayContextualResearchRunner:
    def __init__(self, config_dir: str | Path, experiment: IntradayContextualResearchConfig) -> None:
        self.config_dir = Path(config_dir)
        self.experiment = experiment
        self.base_settings = load_variant_settings(self.config_dir, experiment.base_variant)

    def run(
        self,
        *,
        input_frame: pd.DataFrame,
        output_dir: str | Path,
        allow_gaps: bool = False,
        start: datetime | None = None,
        end: datetime | None = None,
        selected_variants: Sequence[str] | None = None,
    ) -> IntradayContextualResearchArtifacts:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        frame = _filter_frame_by_range(input_frame, start=start, end=end)
        specs = self._build_variant_specs(selected_variants=selected_variants)
        if not specs:
            raise ValueError("The contextual intraday research config produced no variants to run.")

        variant_artifacts: dict[str, IntradayContextualVariantArtifacts] = {}
        yearly_rows: list[pd.DataFrame] = []
        quarterly_rows: list[pd.DataFrame] = []
        activity_rows: list[dict[str, Any]] = []

        for spec in specs:
            runner = _build_runner_from_settings(spec.settings)
            artifact_dir = output_path / "variants" / spec.name / "baseline"
            diagnostics_dir = output_path / "variants" / spec.name / "diagnostics"
            baseline_artifacts = runner.run(
                output_dir=artifact_dir,
                input_frame=frame,
                allow_gaps=allow_gaps or runner.application.settings.data.allow_gaps,
            )
            diagnostics_runner = BaselineDiagnosticsRunner(runner.application)
            diagnostics_artifacts = diagnostics_runner.run(
                artifact_dir=artifact_dir,
                output_dir=diagnostics_dir,
                include_variants=True,
                include_risk_replay=False,
            )
            report_payload = json.loads(baseline_artifacts.report_path.read_text(encoding="utf-8"))
            diagnostics_payload = json.loads(diagnostics_artifacts.diagnostics_path.read_text(encoding="utf-8"))
            yearly_breakdown = pd.read_csv(diagnostics_artifacts.yearly_breakdown_path)
            quarterly_breakdown = pd.read_csv(diagnostics_artifacts.quarterly_breakdown_path)
            enriched_trades = _read_enriched_trades(diagnostics_dir / "enriched_trades.csv")

            if not yearly_breakdown.empty:
                yearly_breakdown.insert(0, "variant", spec.name)
                yearly_breakdown.insert(1, "label", spec.label)
                yearly_rows.append(yearly_breakdown)
            if not quarterly_breakdown.empty:
                quarterly_breakdown.insert(0, "variant", spec.name)
                quarterly_breakdown.insert(1, "label", spec.label)
                quarterly_rows.append(quarterly_breakdown)

            metrics = self._build_variant_metrics(
                spec=spec,
                report_payload=report_payload,
                diagnostics_payload=diagnostics_payload,
                yearly_breakdown=yearly_breakdown,
                quarterly_breakdown=quarterly_breakdown,
                enriched_trades=enriched_trades,
            )
            activity_rows.append(_extract_activity_row(metrics))
            variant_artifacts[spec.name] = IntradayContextualVariantArtifacts(
                variant_name=spec.name,
                label=spec.label,
                candidate=spec.candidate,
                artifact_dir=artifact_dir,
                diagnostics_dir=diagnostics_dir,
                report_path=baseline_artifacts.report_path,
                diagnostics_path=diagnostics_artifacts.diagnostics_path,
                summary_path=diagnostics_artifacts.summary_path,
                metrics=metrics,
            )

        results_frame = pd.DataFrame([artifact.metrics for artifact in variant_artifacts.values()])
        yearly_frame = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()
        quarterly_frame = pd.concat(quarterly_rows, ignore_index=True) if quarterly_rows else pd.DataFrame()
        activity_summary = pd.DataFrame(activity_rows)
        ranking = self._build_candidate_ranking(results_frame)
        conclusion = self._build_conclusion(
            results_frame=results_frame,
            yearly_frame=yearly_frame,
            quarterly_frame=quarterly_frame,
        )

        payload = {
            "experiment_name": self.experiment.name,
            "base_variant": self.experiment.base_variant,
            "input_period": {
                "start": frame.index[0].isoformat() if not frame.empty else None,
                "end": frame.index[-1].isoformat() if not frame.empty else None,
                "bars": int(len(frame)),
            },
            "summary_thresholds": {
                "minimum_profit_factor": self.experiment.summary_thresholds.minimum_profit_factor,
                "minimum_expectancy": self.experiment.summary_thresholds.minimum_expectancy,
                "maximum_drawdown": self.experiment.summary_thresholds.maximum_drawdown,
                "minimum_positive_year_ratio": self.experiment.summary_thresholds.minimum_positive_year_ratio,
                "minimum_positive_quarter_ratio": self.experiment.summary_thresholds.minimum_positive_quarter_ratio,
            },
            "variants": {name: _sanitize_value(artifact.metrics) for name, artifact in variant_artifacts.items()},
            "conclusion": conclusion,
        }

        comparison_path = output_path / "intraday_nasdaq_contextual_comparison.json"
        results_path = output_path / "intraday_nasdaq_contextual_results.csv"
        summary_path = output_path / "intraday_nasdaq_contextual_summary.md"
        activity_summary_path = output_path / "activity_summary.csv"
        yearly_path = output_path / "yearly_variant_summary.csv"
        quarterly_path = output_path / "quarterly_variant_summary.csv"
        ranking_path = output_path / "candidate_ranking.csv"

        comparison_path.write_text(json.dumps(_sanitize_value(payload), indent=2), encoding="utf-8")
        results_frame.sort_values(
            ["candidate", "passes_quality_guard", "net_pnl", "expectancy", "profit_factor", "max_drawdown"],
            ascending=[False, False, False, False, False, True],
        ).to_csv(results_path, index=False)
        activity_summary.to_csv(activity_summary_path, index=False)
        yearly_frame.to_csv(yearly_path, index=False)
        quarterly_frame.to_csv(quarterly_path, index=False)
        ranking.to_csv(ranking_path, index=False)
        summary_path.write_text(
            self._build_summary_markdown(
                results_frame=results_frame,
                activity_summary=activity_summary,
                conclusion=conclusion,
            ),
            encoding="utf-8",
        )

        return IntradayContextualResearchArtifacts(
            output_dir=output_path,
            comparison_path=comparison_path,
            results_path=results_path,
            summary_path=summary_path,
            activity_summary_path=activity_summary_path,
            yearly_path=yearly_path,
            quarterly_path=quarterly_path,
            ranking_path=ranking_path,
            variant_artifacts=variant_artifacts,
        )

    def _build_variant_specs(self, *, selected_variants: Sequence[str] | None) -> list[IntradayContextualVariantSpec]:
        requested = set(selected_variants or ())
        specs: list[IntradayContextualVariantSpec] = []
        for variant in self.experiment.variants:
            if requested and variant.name not in requested:
                continue
            source_variant = variant.source_variant or self.experiment.base_variant
            source_settings = (
                self.base_settings
                if source_variant == self.experiment.base_variant
                else load_variant_settings(self.config_dir, source_variant)
            )
            settings = apply_settings_overrides(source_settings, variant.overrides)
            settings = apply_settings_overrides(
                settings,
                {"strategy": {"variant_name": variant.name}},
            )
            specs.append(
                IntradayContextualVariantSpec(
                    name=variant.name,
                    label=variant.label,
                    candidate=variant.candidate,
                    source_variant=source_variant,
                    settings=settings,
                )
            )
        return specs

    def _build_variant_metrics(
        self,
        *,
        spec: IntradayContextualVariantSpec,
        report_payload: dict[str, Any],
        diagnostics_payload: dict[str, Any],
        yearly_breakdown: pd.DataFrame,
        quarterly_breakdown: pd.DataFrame,
        enriched_trades: pd.DataFrame,
    ) -> dict[str, Any]:
        baseline_metrics = diagnostics_payload["baseline_metrics"]
        thresholds = self.experiment.summary_thresholds
        positive_year_ratio = float((yearly_breakdown["net_pnl"] > 0.0).mean()) if not yearly_breakdown.empty else 0.0
        positive_quarter_ratio = (
            float((quarterly_breakdown["net_pnl"] > 0.0).mean()) if not quarterly_breakdown.empty else 0.0
        )
        long_trades = (
            int((enriched_trades["side"].astype(str).str.lower() == "long").sum()) if not enriched_trades.empty else 0
        )
        short_trades = (
            int((enriched_trades["side"].astype(str).str.lower() == "short").sum()) if not enriched_trades.empty else 0
        )
        trades_total = int(baseline_metrics["number_of_trades"])

        return {
            "variant": spec.name,
            "label": spec.label,
            "candidate": spec.candidate,
            "source_variant": spec.source_variant,
            "symbol": report_payload["symbol"],
            "strategy": variant_summary_payload(spec.settings),
            "number_of_trades": trades_total,
            "trades_per_year": float(baseline_metrics["trades_per_year"]),
            "trades_per_week_avg": float(baseline_metrics["trades_per_week_avg"]),
            "trades_per_month_avg": float(baseline_metrics["trades_per_year"]) / 12.0 if baseline_metrics["trades_per_year"] else 0.0,
            "pct_days_with_trade": _pct_periods_with_trade(enriched_trades, "entry_timestamp", "D"),
            "pct_weeks_with_trade": _pct_periods_with_trade(enriched_trades, "entry_timestamp", "W-SUN"),
            "max_inactive_weeks": _max_inactive_weeks(enriched_trades),
            "win_rate": float(baseline_metrics["win_rate"]),
            "average_win": float(baseline_metrics["average_win"]),
            "average_loss": float(baseline_metrics["average_loss"]),
            "payoff": float(baseline_metrics["payoff_real"]),
            "profit_factor": float(baseline_metrics["profit_factor"]),
            "expectancy": float(baseline_metrics["expectancy"]),
            "gross_pnl": float(baseline_metrics["gross_pnl"]),
            "net_pnl": float(baseline_metrics["net_pnl"]),
            "fees_paid": float(baseline_metrics["fees_paid_total"]),
            "max_drawdown": float(baseline_metrics["max_drawdown"]),
            "sharpe": float(baseline_metrics["sharpe"]),
            "sortino": float(baseline_metrics["sortino"]),
            "calmar": float(baseline_metrics["calmar"]),
            "average_holding_bars": float(baseline_metrics["average_holding_bars"]),
            "positive_year_ratio": positive_year_ratio,
            "positive_quarter_ratio": positive_quarter_ratio,
            "long_trades": long_trades,
            "short_trades": short_trades,
            "passes_quality_guard": (
                float(baseline_metrics["profit_factor"]) > thresholds.minimum_profit_factor
                and float(baseline_metrics["expectancy"]) > thresholds.minimum_expectancy
                and float(baseline_metrics["max_drawdown"]) <= thresholds.maximum_drawdown
                and positive_year_ratio >= thresholds.minimum_positive_year_ratio
                and positive_quarter_ratio >= thresholds.minimum_positive_quarter_ratio
            ),
        }

    def _build_candidate_ranking(self, results_frame: pd.DataFrame) -> pd.DataFrame:
        if results_frame.empty:
            return pd.DataFrame(columns=["variant"])
        ranking = results_frame.copy()
        ranking = ranking.sort_values(
            [
                "candidate",
                "passes_quality_guard",
                "net_pnl",
                "expectancy",
                "profit_factor",
                "max_drawdown",
                "sharpe",
                "sortino",
                "trades_per_week_avg",
            ],
            ascending=[False, False, False, False, False, True, False, False, False],
        ).reset_index(drop=True)
        ranking.insert(0, "rank", range(1, len(ranking) + 1))
        return ranking

    def _build_conclusion(
        self,
        *,
        results_frame: pd.DataFrame,
        yearly_frame: pd.DataFrame,
        quarterly_frame: pd.DataFrame,
    ) -> dict[str, Any]:
        candidates = results_frame.loc[results_frame["candidate"]].copy()
        if candidates.empty:
            return {
                "headline": "No contextual intraday variants were executed.",
                "best_profitability_variant": None,
                "best_drawdown_variant": None,
                "most_active_variant": None,
                "best_balance_variant": None,
                "family_verdict": "No verdict.",
            }

        ranking = self._build_candidate_ranking(candidates)
        best_balance = ranking.iloc[0]
        profitability_pool = candidates.loc[candidates["passes_quality_guard"]].copy()
        if profitability_pool.empty:
            profitability_pool = candidates
        best_profitability = profitability_pool.sort_values(
            ["net_pnl", "expectancy", "profit_factor", "max_drawdown", "sharpe", "trades_per_week_avg"],
            ascending=[False, False, False, True, False, False],
        ).iloc[0]
        best_drawdown = profitability_pool.sort_values(
            ["max_drawdown", "net_pnl", "profit_factor", "expectancy", "trades_per_week_avg"],
            ascending=[True, False, False, False, False],
        ).iloc[0]
        most_active = candidates.sort_values(
            ["trades_per_week_avg", "net_pnl", "profit_factor", "expectancy"],
            ascending=[False, False, False, False],
        ).iloc[0]

        if bool(best_profitability["passes_quality_guard"]):
            family_verdict = (
                "Yes: the new contextual intraday family already has at least one candidate with positive economics, acceptable drawdown, and usable activity."
            )
        else:
            family_verdict = (
                "Mixed: the new contextual intraday family found activity, but no candidate passes the profitability/drawdown guard cleanly yet."
            )

        return {
            "headline": (
                f"`{best_profitability['variant']}` is the strongest profitability-first candidate in the new contextual intraday family, "
                f"while `{most_active['variant']}` is the most active variant."
            ),
            "best_profitability_variant": _sanitize_value(best_profitability.to_dict()),
            "best_drawdown_variant": _sanitize_value(best_drawdown.to_dict()),
            "most_active_variant": _sanitize_value(most_active.to_dict()),
            "best_balance_variant": _sanitize_value(best_balance.to_dict()),
            "active_orb_control": _reference_snapshot(results_frame, "active_orb_reclaim_30m_control"),
            "family_verdict": family_verdict,
            "yearly_snapshot": _temporal_snapshot(yearly_frame, best_profitability["variant"], "exit_year"),
            "quarterly_snapshot": _temporal_snapshot(quarterly_frame, best_profitability["variant"], "exit_quarter"),
        }

    def _build_summary_markdown(
        self,
        *,
        results_frame: pd.DataFrame,
        activity_summary: pd.DataFrame,
        conclusion: dict[str, Any],
    ) -> str:
        best_profitability = conclusion["best_profitability_variant"]
        best_drawdown = conclusion["best_drawdown_variant"]
        most_active = conclusion["most_active_variant"]
        best_balance = conclusion["best_balance_variant"]
        ordered = results_frame.sort_values(
            [
                "candidate",
                "passes_quality_guard",
                "net_pnl",
                "expectancy",
                "profit_factor",
                "max_drawdown",
                "trades_per_week_avg",
            ],
            ascending=[False, False, False, False, False, True, False],
        )
        lines = [
            "# Contextual Intraday Nasdaq Research Summary",
            "",
            "## Direct Answer",
            conclusion["headline"],
            "",
            f"- Family verdict: {conclusion['family_verdict']}",
            (
                f"- Best profitability-first candidate: `{best_profitability['variant']}` | net `{best_profitability['net_pnl']:.2f}` | "
                f"expectancy `{best_profitability['expectancy']:.2f}` | PF `{best_profitability['profit_factor']:.2f}` | DD `{best_profitability['max_drawdown']:.4f}`."
            ),
            (
                f"- Lowest-drawdown viable candidate: `{best_drawdown['variant']}` | DD `{best_drawdown['max_drawdown']:.4f}` | "
                f"net `{best_drawdown['net_pnl']:.2f}` | PF `{best_drawdown['profit_factor']:.2f}`."
            ),
            (
                f"- Most active variant: `{most_active['variant']}` | `{most_active['trades_per_week_avg']:.3f}` trades/week | "
                f"net `{most_active['net_pnl']:.2f}` | PF `{most_active['profit_factor']:.2f}`."
            ),
            (
                f"- Best overall balance with the new ranking: `{best_balance['variant']}`."
            ),
            "",
            "## Variants",
        ]
        for _, row in ordered.iterrows():
            lines.append(
                f"- `{row['variant']}`: net `{row['net_pnl']:.2f}`, expectancy `{row['expectancy']:.2f}`, PF `{row['profit_factor']:.2f}`, "
                f"DD `{row['max_drawdown']:.4f}`, Sharpe `{row['sharpe']:.2f}`, trades/week `{row['trades_per_week_avg']:.3f}`."
            )

        lines.extend(["", "## Activity Summary"])
        for _, row in activity_summary.sort_values("trades_per_week_avg", ascending=False).iterrows():
            lines.append(
                f"- `{row['variant']}`: `{row['trades_per_month_avg']:.2f}` trades/month, `{row['pct_days_with_trade'] * 100:.1f}%` of days with trades, "
                f"`{row['pct_weeks_with_trade'] * 100:.1f}%` of weeks active."
            )

        lines.extend(["", "## Temporal Snapshot"])
        for item in conclusion["yearly_snapshot"]:
            lines.append(
                f"- Year `{item['block']}`: trades `{item['trades']}`, net `{item['net_pnl']:.2f}`, PF `{item['profit_factor']:.2f}`."
            )
        for item in conclusion["quarterly_snapshot"][:8]:
            lines.append(
                f"- Quarter `{item['block']}`: trades `{item['trades']}`, net `{item['net_pnl']:.2f}`, PF `{item['profit_factor']:.2f}`."
            )

        lines.extend(
            [
                "",
                "## How To Read This",
                "- `intraday_nasdaq_contextual_results.csv` is ranked profit-first, drawdown-second, activity-third.",
                "- `candidate_ranking.csv` is the quick shortlist of the most serious contextual intraday candidates.",
                "- Each variant keeps its own `baseline/` and `diagnostics/` artifacts inside `variants/`.",
            ]
        )
        return "\n".join(lines) + "\n"


def load_intraday_contextual_research_config(path: str | Path) -> IntradayContextualResearchConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Contextual intraday research config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Contextual intraday research config must be a mapping: {config_path}")

    thresholds_payload = payload.get("summary_thresholds", {}) or {}
    thresholds = IntradayContextualThresholds(
        minimum_profit_factor=float(thresholds_payload.get("minimum_profit_factor", 1.05)),
        minimum_expectancy=float(thresholds_payload.get("minimum_expectancy", 0.0)),
        maximum_drawdown=float(thresholds_payload.get("maximum_drawdown", 0.08)),
        minimum_positive_year_ratio=float(thresholds_payload.get("minimum_positive_year_ratio", 0.50)),
        minimum_positive_quarter_ratio=float(thresholds_payload.get("minimum_positive_quarter_ratio", 0.40)),
    )

    raw_variants = payload.get("variants", [])
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("Contextual intraday research config requires a non-empty 'variants' list.")

    variants: list[IntradayContextualVariantConfig] = []
    for raw_variant in raw_variants:
        if not isinstance(raw_variant, dict):
            raise ValueError("Each contextual intraday variant must be a mapping.")
        overrides = raw_variant.get("overrides", {}) or {}
        if not isinstance(overrides, dict):
            raise ValueError("Variant overrides must be a mapping.")
        source_variant = raw_variant.get("source_variant")
        variants.append(
            IntradayContextualVariantConfig(
                name=str(raw_variant["name"]),
                label=str(raw_variant.get("label", raw_variant["name"])),
                candidate=bool(raw_variant.get("candidate", True)),
                source_variant=str(source_variant) if source_variant else None,
                overrides=overrides,
            )
        )

    return IntradayContextualResearchConfig(
        name=str(payload.get("name", "intraday_nasdaq_contextual_research")),
        base_variant=str(payload.get("base_variant", "baseline_nq_intraday_contextual")),
        summary_thresholds=thresholds,
        variants=tuple(variants),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the new contextual intraday Nasdaq research matrix.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument(
        "--experiment-config",
        default="configs/experiments/intraday_nasdaq_contextual_research.yaml",
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-gaps", action="store_true")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--variant", action="append", dest="variants")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    experiment = load_intraday_contextual_research_config(args.experiment_config)
    runner = IntradayContextualResearchRunner(args.config_dir, experiment)
    frame = read_ohlcv_frame(args.input_path)
    artifacts = runner.run(
        input_frame=frame,
        output_dir=args.output_dir,
        allow_gaps=args.allow_gaps,
        start=_parse_datetime(args.start) if args.start else None,
        end=_parse_datetime(args.end) if args.end else None,
        selected_variants=tuple(args.variants or ()),
    )

    payload = json.loads(artifacts.comparison_path.read_text(encoding="utf-8"))
    best = payload["conclusion"]["best_profitability_variant"]
    print(f"Contextual intraday comparison: {artifacts.comparison_path}")
    print(f"Contextual intraday summary: {artifacts.summary_path}")
    print(
        " ".join(
            [
                f"variants={len(payload['variants'])}",
                f"best_profitability={best['variant'] if best else 'n/a'}",
                f"family_verdict={payload['conclusion']['family_verdict']}",
            ]
        )
    )
    return 0


def _build_runner_from_settings(settings: Settings) -> BaselineRunner:
    application = build_application_from_settings(settings)
    data_service = HistoricalDataIngestionService(
        downloader=BinanceHistoricalDownloader(
            base_url=settings.data.historical_api_url,
            timeout_seconds=settings.data.request_timeout_seconds,
        ),
        store=ParquetDatasetStore(
            compression=settings.data.parquet_compression,
            engine=settings.data.parquet_engine,
        ),
    )
    return BaselineRunner(application=application, data_service=data_service)


if __name__ == "__main__":
    raise SystemExit(main())
