from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from hybrid_quant.core import Settings, apply_settings_overrides
from hybrid_quant.data import read_ohlcv_frame

from .diagnostics import BaselineDiagnosticsRunner
from .orb_intraday_active_research import (
    _build_runner_from_settings,
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
from .variants import load_variant_settings, variant_summary_payload


@dataclass(slots=True)
class IntradayHybridThresholds:
    minimum_profit_factor: float = 1.10
    minimum_expectancy: float = 0.0
    maximum_drawdown: float = 0.08
    maximum_daily_loss_pct: float = 0.025
    minimum_positive_year_ratio: float = 0.50
    minimum_positive_quarter_ratio: float = 0.40
    minimum_positive_split_ratio: float = 0.50
    minimum_trades: int = 40


@dataclass(slots=True)
class IntradayHybridSplitConfig:
    train_ratio: float = 0.60
    validation_ratio: float = 0.20
    test_ratio: float = 0.20


@dataclass(slots=True)
class IntradayHybridVariantConfig:
    name: str
    label: str
    candidate: bool
    source_variant: str | None
    overrides: dict[str, Any]


@dataclass(slots=True)
class IntradayHybridResearchConfig:
    name: str
    base_variant: str
    summary_thresholds: IntradayHybridThresholds
    temporal_splits: IntradayHybridSplitConfig
    variants: tuple[IntradayHybridVariantConfig, ...]


@dataclass(slots=True)
class IntradayHybridVariantSpec:
    name: str
    label: str
    candidate: bool
    source_variant: str
    settings: Settings


@dataclass(slots=True)
class IntradayHybridVariantArtifacts:
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
class IntradayHybridResearchArtifacts:
    output_dir: Path
    comparison_path: Path
    results_path: Path
    summary_path: Path
    ranking_path: Path
    activity_summary_path: Path
    split_results_path: Path
    yearly_path: Path
    quarterly_path: Path
    cost_sensitivity_path: Path
    variant_artifacts: dict[str, IntradayHybridVariantArtifacts]


class IntradayHybridResearchRunner:
    def __init__(self, config_dir: str | Path, experiment: IntradayHybridResearchConfig) -> None:
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
    ) -> IntradayHybridResearchArtifacts:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        frame = _filter_frame_by_range(input_frame, start=start, end=end)
        specs = self._build_variant_specs(selected_variants=selected_variants)
        if not specs:
            raise ValueError("The intraday hybrid research config produced no variants to run.")

        variant_artifacts: dict[str, IntradayHybridVariantArtifacts] = {}
        yearly_rows: list[pd.DataFrame] = []
        quarterly_rows: list[pd.DataFrame] = []
        hourly_rows: list[pd.DataFrame] = []
        side_rows: list[pd.DataFrame] = []
        activity_rows: list[dict[str, Any]] = []
        split_rows: list[dict[str, Any]] = []
        cost_rows: list[pd.DataFrame] = []

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
            hourly_breakdown = pd.read_csv(diagnostics_artifacts.hourly_breakdown_path)
            side_breakdown = pd.read_csv(diagnostics_artifacts.side_breakdown_path)
            cost_impact = pd.read_csv(diagnostics_artifacts.cost_impact_path)
            enriched_trades = _read_enriched_trades(diagnostics_dir / "enriched_trades.csv")

            for collection, breakdown in (
                (yearly_rows, yearly_breakdown),
                (quarterly_rows, quarterly_breakdown),
                (hourly_rows, hourly_breakdown),
                (side_rows, side_breakdown),
            ):
                if not breakdown.empty:
                    breakdown.insert(0, "variant", spec.name)
                    breakdown.insert(1, "label", spec.label)
                    collection.append(breakdown)
            if not cost_impact.empty:
                cost_impact.insert(0, "research_variant", spec.name)
                cost_impact.insert(1, "label", spec.label)
                cost_rows.append(cost_impact)

            split_frame = self._build_split_results(
                variant=spec.name,
                label=spec.label,
                enriched_trades=enriched_trades,
                frame=frame,
                initial_capital=spec.settings.backtest.initial_capital,
            )
            if not split_frame.empty:
                split_rows.extend(split_frame.to_dict(orient="records"))

            metrics = self._build_variant_metrics(
                spec=spec,
                report_payload=report_payload,
                diagnostics_payload=diagnostics_payload,
                yearly_breakdown=yearly_breakdown,
                quarterly_breakdown=quarterly_breakdown,
                split_frame=split_frame,
                enriched_trades=enriched_trades,
            )
            activity_rows.append(_extract_activity_row(metrics))
            variant_artifacts[spec.name] = IntradayHybridVariantArtifacts(
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

        return self._write_outputs(
            output_path=output_path,
            frame=frame,
            variant_artifacts=variant_artifacts,
            yearly_rows=yearly_rows,
            quarterly_rows=quarterly_rows,
            hourly_rows=hourly_rows,
            side_rows=side_rows,
            activity_rows=activity_rows,
            split_rows=split_rows,
            cost_rows=cost_rows,
        )

    def _build_variant_specs(self, *, selected_variants: Sequence[str] | None) -> list[IntradayHybridVariantSpec]:
        requested = set(selected_variants or ())
        specs: list[IntradayHybridVariantSpec] = []
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
            settings = apply_settings_overrides(settings, {"strategy": {"variant_name": variant.name}})
            specs.append(
                IntradayHybridVariantSpec(
                    name=variant.name,
                    label=variant.label,
                    candidate=variant.candidate,
                    source_variant=source_variant,
                    settings=settings,
                )
            )
        return specs

    def _write_outputs(
        self,
        *,
        output_path: Path,
        frame: pd.DataFrame,
        variant_artifacts: dict[str, IntradayHybridVariantArtifacts],
        yearly_rows: list[pd.DataFrame],
        quarterly_rows: list[pd.DataFrame],
        hourly_rows: list[pd.DataFrame],
        side_rows: list[pd.DataFrame],
        activity_rows: list[dict[str, Any]],
        split_rows: list[dict[str, Any]],
        cost_rows: list[pd.DataFrame],
    ) -> IntradayHybridResearchArtifacts:
        results_frame = pd.DataFrame([artifact.metrics for artifact in variant_artifacts.values()])
        yearly_frame = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()
        quarterly_frame = pd.concat(quarterly_rows, ignore_index=True) if quarterly_rows else pd.DataFrame()
        hourly_frame = pd.concat(hourly_rows, ignore_index=True) if hourly_rows else pd.DataFrame()
        side_frame = pd.concat(side_rows, ignore_index=True) if side_rows else pd.DataFrame()
        activity_summary = pd.DataFrame(activity_rows)
        split_results = pd.DataFrame(split_rows)
        cost_sensitivity = pd.concat(cost_rows, ignore_index=True) if cost_rows else pd.DataFrame()
        ranking = self._build_candidate_ranking(results_frame)
        conclusion = self._build_conclusion(
            results_frame=results_frame,
            yearly_frame=yearly_frame,
            quarterly_frame=quarterly_frame,
            split_results=split_results,
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
                "maximum_daily_loss_pct": self.experiment.summary_thresholds.maximum_daily_loss_pct,
                "minimum_positive_year_ratio": self.experiment.summary_thresholds.minimum_positive_year_ratio,
                "minimum_positive_quarter_ratio": self.experiment.summary_thresholds.minimum_positive_quarter_ratio,
                "minimum_positive_split_ratio": self.experiment.summary_thresholds.minimum_positive_split_ratio,
                "minimum_trades": self.experiment.summary_thresholds.minimum_trades,
            },
            "temporal_splits": {
                "train_ratio": self.experiment.temporal_splits.train_ratio,
                "validation_ratio": self.experiment.temporal_splits.validation_ratio,
                "test_ratio": self.experiment.temporal_splits.test_ratio,
            },
            "variants": {name: _sanitize_value(artifact.metrics) for name, artifact in variant_artifacts.items()},
            "conclusion": conclusion,
        }

        comparison_path = output_path / "intraday_hybrid_comparison.json"
        results_path = output_path / "intraday_hybrid_results.csv"
        summary_path = output_path / "intraday_hybrid_summary.md"
        ranking_path = output_path / "candidate_ranking.csv"
        activity_summary_path = output_path / "activity_summary.csv"
        split_results_path = output_path / "split_results.csv"
        yearly_path = output_path / "yearly_variant_summary.csv"
        quarterly_path = output_path / "quarterly_variant_summary.csv"
        cost_sensitivity_path = output_path / "cost_sensitivity.csv"

        comparison_path.write_text(json.dumps(_sanitize_value(payload), indent=2), encoding="utf-8")
        results_frame.sort_values(
            ["candidate", "passes_quality_guard", "net_pnl", "expectancy", "profit_factor", "max_drawdown"],
            ascending=[False, False, False, False, False, True],
        ).to_csv(results_path, index=False)
        ranking.to_csv(ranking_path, index=False)
        activity_summary.to_csv(activity_summary_path, index=False)
        split_results.to_csv(split_results_path, index=False)
        yearly_frame.to_csv(yearly_path, index=False)
        quarterly_frame.to_csv(quarterly_path, index=False)
        cost_sensitivity.to_csv(cost_sensitivity_path, index=False)
        hourly_frame.to_csv(output_path / "hourly_variant_summary.csv", index=False)
        side_frame.to_csv(output_path / "side_variant_summary.csv", index=False)
        summary_path.write_text(
            self._build_summary_markdown(
                results_frame=results_frame,
                activity_summary=activity_summary,
                conclusion=conclusion,
            ),
            encoding="utf-8",
        )

        return IntradayHybridResearchArtifacts(
            output_dir=output_path,
            comparison_path=comparison_path,
            results_path=results_path,
            summary_path=summary_path,
            ranking_path=ranking_path,
            activity_summary_path=activity_summary_path,
            split_results_path=split_results_path,
            yearly_path=yearly_path,
            quarterly_path=quarterly_path,
            cost_sensitivity_path=cost_sensitivity_path,
            variant_artifacts=variant_artifacts,
        )

    def _build_variant_metrics(
        self,
        *,
        spec: IntradayHybridVariantSpec,
        report_payload: dict[str, Any],
        diagnostics_payload: dict[str, Any],
        yearly_breakdown: pd.DataFrame,
        quarterly_breakdown: pd.DataFrame,
        split_frame: pd.DataFrame,
        enriched_trades: pd.DataFrame,
    ) -> dict[str, Any]:
        baseline_metrics = diagnostics_payload["baseline_metrics"]
        thresholds = self.experiment.summary_thresholds
        positive_year_ratio = float((yearly_breakdown["net_pnl"] > 0.0).mean()) if not yearly_breakdown.empty else 0.0
        positive_quarter_ratio = (
            float((quarterly_breakdown["net_pnl"] > 0.0).mean()) if not quarterly_breakdown.empty else 0.0
        )
        positive_split_ratio = float((split_frame["net_pnl"] > 0.0).mean()) if not split_frame.empty else 0.0
        long_trades = (
            int((enriched_trades["side"].astype(str).str.lower() == "long").sum()) if not enriched_trades.empty else 0
        )
        short_trades = (
            int((enriched_trades["side"].astype(str).str.lower() == "short").sum()) if not enriched_trades.empty else 0
        )
        max_daily_loss = self._max_daily_net_loss(enriched_trades)
        initial_capital = spec.settings.backtest.initial_capital
        max_daily_loss_pct = abs(max_daily_loss) / initial_capital if initial_capital > 0.0 else 0.0
        number_of_trades = int(baseline_metrics["number_of_trades"])

        return {
            "variant": spec.name,
            "label": spec.label,
            "candidate": spec.candidate,
            "source_variant": spec.source_variant,
            "symbol": report_payload["symbol"],
            "strategy": variant_summary_payload(spec.settings),
            "number_of_trades": number_of_trades,
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
            "max_daily_net_loss": max_daily_loss,
            "max_daily_net_loss_pct": max_daily_loss_pct,
            "sharpe": float(baseline_metrics["sharpe"]),
            "sortino": float(baseline_metrics["sortino"]),
            "calmar": float(baseline_metrics["calmar"]),
            "max_consecutive_losses": int(baseline_metrics["max_consecutive_losses"]),
            "average_holding_bars": float(baseline_metrics["average_holding_bars"]),
            "average_mfe_atr": float(baseline_metrics["average_mfe_atr"]),
            "average_mae_atr": float(baseline_metrics["average_mae_atr"]),
            "positive_year_ratio": positive_year_ratio,
            "positive_quarter_ratio": positive_quarter_ratio,
            "positive_split_ratio": positive_split_ratio,
            "long_trades": long_trades,
            "short_trades": short_trades,
            "risk_blocked_actionable_signals": int(report_payload["risk"]["blocked_actionable_signals"]),
            "risk_raw_actionable_signals": int(report_payload["risk"]["raw_actionable_signals"]),
            "kill_switch_days": len(report_payload["risk"]["kill_switch_triggered_days"]),
            "passes_quality_guard": (
                number_of_trades >= thresholds.minimum_trades
                and float(baseline_metrics["net_pnl"]) > 0.0
                and float(baseline_metrics["profit_factor"]) > thresholds.minimum_profit_factor
                and float(baseline_metrics["expectancy"]) > thresholds.minimum_expectancy
                and float(baseline_metrics["max_drawdown"]) <= thresholds.maximum_drawdown
                and max_daily_loss_pct <= thresholds.maximum_daily_loss_pct
                and positive_year_ratio >= thresholds.minimum_positive_year_ratio
                and positive_quarter_ratio >= thresholds.minimum_positive_quarter_ratio
                and positive_split_ratio >= thresholds.minimum_positive_split_ratio
            ),
        }

    def _build_split_results(
        self,
        *,
        variant: str,
        label: str,
        enriched_trades: pd.DataFrame,
        frame: pd.DataFrame,
        initial_capital: float,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for split_name, start, end in self._split_ranges(frame):
            subset = enriched_trades
            if not subset.empty and "exit_timestamp" in subset.columns:
                exits = pd.to_datetime(subset["exit_timestamp"], utc=True, errors="coerce")
                subset = subset.loc[(exits >= start) & (exits <= end)].copy()
            rows.append(
                {
                    "variant": variant,
                    "label": label,
                    "split": split_name,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    **self._metrics_from_trades(subset, initial_capital=initial_capital),
                }
            )
        return pd.DataFrame(rows)

    def _split_ranges(self, frame: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
        index = pd.to_datetime(frame.index, utc=True)
        start = index.min()
        end = index.max()
        total_seconds = max((end - start).total_seconds(), 1.0)
        train_end = start + pd.Timedelta(seconds=total_seconds * self.experiment.temporal_splits.train_ratio)
        validation_end = train_end + pd.Timedelta(
            seconds=total_seconds * self.experiment.temporal_splits.validation_ratio
        )
        return [
            ("train", start, train_end),
            ("validation", train_end, validation_end),
            ("test", validation_end, end),
        ]

    def _metrics_from_trades(self, trades: pd.DataFrame, *, initial_capital: float) -> dict[str, Any]:
        if trades.empty:
            return {
                "trades": 0,
                "net_pnl": 0.0,
                "gross_pnl": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "win_rate": 0.0,
                "max_daily_net_loss_pct": 0.0,
            }
        wins = trades.loc[trades["net_pnl"] > 0.0, "net_pnl"]
        losses = trades.loc[trades["net_pnl"] <= 0.0, "net_pnl"]
        gross_profit = float(wins.sum()) if not wins.empty else 0.0
        gross_loss = abs(float(losses.sum())) if not losses.empty else 0.0
        max_daily_loss = self._max_daily_net_loss(trades)
        return {
            "trades": int(len(trades)),
            "net_pnl": float(trades["net_pnl"].sum()),
            "gross_pnl": float(trades["gross_pnl"].sum()) if "gross_pnl" in trades else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else 0.0,
            "expectancy": float(trades["net_pnl"].mean()),
            "win_rate": float((trades["net_pnl"] > 0.0).mean()),
            "max_daily_net_loss_pct": abs(max_daily_loss) / initial_capital if initial_capital > 0.0 else 0.0,
        }

    def _max_daily_net_loss(self, enriched_trades: pd.DataFrame) -> float:
        if enriched_trades.empty or "exit_timestamp" not in enriched_trades.columns:
            return 0.0
        exits = pd.to_datetime(enriched_trades["exit_timestamp"], utc=True, errors="coerce")
        frame = enriched_trades.copy()
        frame["exit_date"] = exits.dt.date
        daily = frame.groupby("exit_date", observed=False)["net_pnl"].sum()
        return float(daily.min()) if not daily.empty else 0.0

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
                "positive_split_ratio",
                "trades_per_week_avg",
            ],
            ascending=[False, False, False, False, False, True, False, False],
        ).reset_index(drop=True)
        ranking.insert(0, "rank", range(1, len(ranking) + 1))
        return ranking

    def _build_conclusion(
        self,
        *,
        results_frame: pd.DataFrame,
        yearly_frame: pd.DataFrame,
        quarterly_frame: pd.DataFrame,
        split_results: pd.DataFrame,
    ) -> dict[str, Any]:
        candidates = results_frame.loc[results_frame["candidate"]].copy()
        if candidates.empty:
            return {
                "headline": "No candidate hybrid intraday variants were executed.",
                "final_baseline_variant": None,
                "family_verdict": "NO-GO",
            }

        ranking = self._build_candidate_ranking(candidates)
        final_candidate = ranking.iloc[0]
        profitable_pool = candidates.loc[candidates["passes_quality_guard"]].copy()
        if profitable_pool.empty:
            profitable_pool = candidates
        best_profit = profitable_pool.sort_values(
            ["net_pnl", "expectancy", "profit_factor", "max_drawdown", "positive_split_ratio"],
            ascending=[False, False, False, True, False],
        ).iloc[0]
        best_drawdown = profitable_pool.sort_values(
            ["max_drawdown", "net_pnl", "expectancy", "profit_factor"],
            ascending=[True, False, False, False],
        ).iloc[0]
        if bool(final_candidate["passes_quality_guard"]):
            verdict = "GO WITH CAUTION"
            headline = (
                f"`{final_candidate['variant']}` is the best frozen baseline candidate: positive economics, "
                "controlled drawdown, and acceptable temporal consistency."
            )
        elif float(final_candidate["net_pnl"]) > 0.0 and float(final_candidate["expectancy"]) > 0.0:
            verdict = "NEEDS MORE VALIDATION"
            headline = (
                f"`{final_candidate['variant']}` is the least fragile candidate, but it does not pass every robustness guard."
            )
        else:
            verdict = "NO-GO"
            headline = "No hybrid intraday candidate currently has enough economic quality to be a safe PPO baseline."

        return {
            "headline": headline,
            "family_verdict": verdict,
            "final_baseline_variant": _sanitize_value(final_candidate.to_dict()),
            "best_profitability_variant": _sanitize_value(best_profit.to_dict()),
            "best_drawdown_variant": _sanitize_value(best_drawdown.to_dict()),
            "legacy_orb_control": _reference_snapshot(results_frame, "legacy_orb_control"),
            "legacy_session_trend_control": _reference_snapshot(results_frame, "legacy_session_trend_control"),
            "yearly_snapshot": _temporal_snapshot(yearly_frame, final_candidate["variant"], "exit_year"),
            "quarterly_snapshot": _temporal_snapshot(quarterly_frame, final_candidate["variant"], "exit_quarter"),
            "split_snapshot": (
                split_results.loc[split_results["variant"] == final_candidate["variant"]].to_dict(orient="records")
                if not split_results.empty
                else []
            ),
        }

    def _build_summary_markdown(
        self,
        *,
        results_frame: pd.DataFrame,
        activity_summary: pd.DataFrame,
        conclusion: dict[str, Any],
    ) -> str:
        final_candidate = conclusion.get("final_baseline_variant")
        ordered = self._build_candidate_ranking(results_frame)
        lines = [
            "# Intraday Hybrid Contextual Research Summary",
            "",
            "## Direct Answer",
            conclusion["headline"],
            "",
            f"- Final classification: `{conclusion['family_verdict']}`.",
            "- Ranking priority: net PnL / expectancy / profit factor first, drawdown second, activity third.",
            "- Max drawdown is reported as a fraction, so `0.024` means `2.4%`.",
        ]
        if final_candidate:
            lines.append(
                f"- Frozen candidate: `{final_candidate['variant']}` | net `{final_candidate['net_pnl']:.2f}` | "
                f"expectancy `{final_candidate['expectancy']:.2f}` | PF `{final_candidate['profit_factor']:.2f}` | "
                f"DD `{final_candidate['max_drawdown']:.4f}` | trades/week `{final_candidate['trades_per_week_avg']:.3f}`."
            )

        lines.extend(["", "## Variants"])
        for _, row in ordered.iterrows():
            lines.append(
                f"- `{row['variant']}`: net `{row['net_pnl']:.2f}`, expectancy `{row['expectancy']:.2f}`, "
                f"PF `{row['profit_factor']:.2f}`, DD `{row['max_drawdown']:.4f}`, "
                f"daily loss `{row['max_daily_net_loss_pct']:.4f}`, trades `{int(row['number_of_trades'])}`, "
                f"split+ `{row['positive_split_ratio']:.2f}`, guard `{bool(row['passes_quality_guard'])}`."
            )

        lines.extend(["", "## Activity"])
        if not activity_summary.empty:
            for _, row in activity_summary.sort_values("trades_per_week_avg", ascending=False).iterrows():
                lines.append(
                    f"- `{row['variant']}`: `{row['trades_per_week_avg']:.3f}` trades/week, "
                    f"`{row['pct_weeks_with_trade'] * 100:.1f}%` active weeks, max inactive weeks `{row['max_inactive_weeks']}`."
                )

        lines.extend(["", "## Temporal Snapshot"])
        for item in conclusion.get("yearly_snapshot", []):
            lines.append(
                f"- Year `{item['block']}`: trades `{item['trades']}`, net `{item['net_pnl']:.2f}`, PF `{item['profit_factor']:.2f}`."
            )
        for item in conclusion.get("split_snapshot", []):
            lines.append(
                f"- Split `{item['split']}`: trades `{item['trades']}`, net `{item['net_pnl']:.2f}`, PF `{item['profit_factor']:.2f}`."
            )

        lines.extend(
            [
                "",
                "## How To Read Artifacts",
                "- `intraday_hybrid_results.csv` is the full comparison table.",
                "- `candidate_ranking.csv` is the shortlist ordered by profitability, drawdown, then activity.",
                "- `split_results.csv` is the train/validation/test temporal check.",
                "- `cost_sensitivity.csv` replays baseline signals without fees/slippage to expose cost fragility.",
                "- Each variant has isolated `baseline/` and `diagnostics/` artifacts in `variants/`.",
            ]
        )
        return "\n".join(lines) + "\n"


def load_intraday_hybrid_research_config(path: str | Path) -> IntradayHybridResearchConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Intraday hybrid research config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Intraday hybrid research config must be a mapping: {config_path}")

    thresholds_payload = payload.get("summary_thresholds", {}) or {}
    thresholds = IntradayHybridThresholds(
        minimum_profit_factor=float(thresholds_payload.get("minimum_profit_factor", 1.10)),
        minimum_expectancy=float(thresholds_payload.get("minimum_expectancy", 0.0)),
        maximum_drawdown=float(thresholds_payload.get("maximum_drawdown", 0.08)),
        maximum_daily_loss_pct=float(thresholds_payload.get("maximum_daily_loss_pct", 0.025)),
        minimum_positive_year_ratio=float(thresholds_payload.get("minimum_positive_year_ratio", 0.50)),
        minimum_positive_quarter_ratio=float(thresholds_payload.get("minimum_positive_quarter_ratio", 0.40)),
        minimum_positive_split_ratio=float(thresholds_payload.get("minimum_positive_split_ratio", 0.50)),
        minimum_trades=int(thresholds_payload.get("minimum_trades", 40)),
    )
    split_payload = payload.get("temporal_splits", {}) or {}
    splits = IntradayHybridSplitConfig(
        train_ratio=float(split_payload.get("train_ratio", 0.60)),
        validation_ratio=float(split_payload.get("validation_ratio", 0.20)),
        test_ratio=float(split_payload.get("test_ratio", 0.20)),
    )

    raw_variants = payload.get("variants", [])
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("Intraday hybrid research config requires a non-empty 'variants' list.")
    variants: list[IntradayHybridVariantConfig] = []
    for raw_variant in raw_variants:
        if not isinstance(raw_variant, dict):
            raise ValueError("Each intraday hybrid variant must be a mapping.")
        overrides = raw_variant.get("overrides", {}) or {}
        if not isinstance(overrides, dict):
            raise ValueError("Variant overrides must be a mapping.")
        source_variant = raw_variant.get("source_variant")
        variants.append(
            IntradayHybridVariantConfig(
                name=str(raw_variant["name"]),
                label=str(raw_variant.get("label", raw_variant["name"])),
                candidate=bool(raw_variant.get("candidate", True)),
                source_variant=str(source_variant) if source_variant else None,
                overrides=overrides,
            )
        )

    return IntradayHybridResearchConfig(
        name=str(payload.get("name", "intraday_hybrid_research")),
        base_variant=str(payload.get("base_variant", "baseline_intraday_hybrid")),
        summary_thresholds=thresholds,
        temporal_splits=splits,
        variants=tuple(variants),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the primary intraday hybrid contextual research matrix.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--experiment-config", default="configs/experiments/intraday_hybrid_research.yaml")
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
    experiment = load_intraday_hybrid_research_config(args.experiment_config)
    runner = IntradayHybridResearchRunner(args.config_dir, experiment)
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
    final_candidate = payload["conclusion"].get("final_baseline_variant")
    print(f"Intraday hybrid comparison: {artifacts.comparison_path}")
    print(f"Intraday hybrid summary: {artifacts.summary_path}")
    print(
        " ".join(
            [
                f"variants={len(payload['variants'])}",
                f"final={final_candidate['variant'] if final_candidate else 'n/a'}",
                f"verdict={payload['conclusion']['family_verdict']}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
