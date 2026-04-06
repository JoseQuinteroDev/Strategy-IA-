from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from hybrid_quant.core import Settings, apply_settings_overrides
from hybrid_quant.data import read_ohlcv_frame

from .diagnostics import BaselineDiagnosticsRunner
from .orb_focus_validation import (
    _build_runner_from_settings,
    _filter_frame_by_range,
    _net_concentration,
    _parse_datetime,
    _sanitize_value,
    _tag_temporal_frame,
)
from .orb_intraday_active_research import (
    _extract_activity_row,
    _max_inactive_weeks,
    _pct_periods_with_trade,
    _read_enriched_trades,
    _reference_snapshot,
    _temporal_snapshot,
)
from .variants import load_variant_settings, variant_summary_payload


@dataclass(slots=True)
class SessionTrendZoomThresholds:
    minimum_trades_total: int = 40
    minimum_trades_per_year: float = 12.0
    minimum_profit_factor: float = 1.05
    strong_profit_factor: float = 1.10
    minimum_expectancy: float = 0.0
    maximum_drawdown: float = 0.08
    target_drawdown_min: float = 0.01
    target_drawdown_max: float = 0.05
    minimum_positive_year_ratio: float = 0.50
    minimum_positive_quarter_ratio: float = 0.50
    maximum_year_net_pnl_concentration: float = 0.75


@dataclass(slots=True)
class SessionTrendZoomVariantConfig:
    name: str
    label: str
    candidate: bool
    source_variant: str | None
    tags: dict[str, str]
    overrides: dict[str, Any]


@dataclass(slots=True)
class SessionTrendZoomConfig:
    name: str
    base_variant: str
    summary_thresholds: SessionTrendZoomThresholds
    variants: tuple[SessionTrendZoomVariantConfig, ...]


@dataclass(slots=True)
class SessionTrendZoomVariantSpec:
    name: str
    label: str
    candidate: bool
    source_variant: str
    tags: dict[str, str]
    settings: Settings


@dataclass(slots=True)
class SessionTrendZoomVariantArtifacts:
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
class SessionTrendZoomArtifacts:
    output_dir: Path
    comparison_path: Path
    results_path: Path
    summary_path: Path
    ranking_path: Path
    hourly_summary_path: Path
    side_summary_path: Path
    temporal_summary_path: Path
    filter_ablation_summary_path: Path
    variant_artifacts: dict[str, SessionTrendZoomVariantArtifacts]


class SessionTrend30mZoomRunner:
    def __init__(self, config_dir: str | Path, experiment: SessionTrendZoomConfig) -> None:
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
    ) -> SessionTrendZoomArtifacts:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        frame = _filter_frame_by_range(input_frame, start=start, end=end)
        specs = self._build_variant_specs(selected_variants=selected_variants)
        if not specs:
            raise ValueError("The session_trend_30m zoom config produced no variants to run.")

        variant_artifacts: dict[str, SessionTrendZoomVariantArtifacts] = {}
        yearly_rows: list[pd.DataFrame] = []
        quarterly_rows: list[pd.DataFrame] = []
        hourly_rows: list[pd.DataFrame] = []
        side_rows: list[pd.DataFrame] = []
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
            hourly_breakdown = pd.read_csv(diagnostics_artifacts.hourly_breakdown_path)
            side_breakdown = pd.read_csv(diagnostics_artifacts.side_breakdown_path)
            enriched_trades = _read_enriched_trades(diagnostics_dir / "enriched_trades.csv")

            if not yearly_breakdown.empty:
                yearly_breakdown = self._tag_variant_frame(spec, yearly_breakdown)
                yearly_rows.append(yearly_breakdown)
            if not quarterly_breakdown.empty:
                quarterly_breakdown = self._tag_variant_frame(spec, quarterly_breakdown)
                quarterly_rows.append(quarterly_breakdown)
            if not hourly_breakdown.empty:
                hourly_breakdown = self._tag_variant_frame(spec, hourly_breakdown)
                hourly_rows.append(hourly_breakdown)
            if not side_breakdown.empty:
                side_breakdown = self._tag_variant_frame(spec, side_breakdown)
                side_rows.append(side_breakdown)

            metrics = self._build_variant_metrics(
                spec=spec,
                report_payload=report_payload,
                diagnostics_payload=diagnostics_payload,
                yearly_breakdown=yearly_breakdown,
                quarterly_breakdown=quarterly_breakdown,
                enriched_trades=enriched_trades,
            )
            activity_rows.append(_extract_activity_row(metrics))
            variant_artifacts[spec.name] = SessionTrendZoomVariantArtifacts(
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
        activity_summary = pd.DataFrame(activity_rows)
        yearly_frame = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()
        quarterly_frame = pd.concat(quarterly_rows, ignore_index=True) if quarterly_rows else pd.DataFrame()
        hourly_frame = pd.concat(hourly_rows, ignore_index=True) if hourly_rows else pd.DataFrame()
        side_frame = pd.concat(side_rows, ignore_index=True) if side_rows else pd.DataFrame()
        temporal_frame = pd.concat(
            [
                _tag_temporal_frame(yearly_frame, block_type="year", block_column="exit_year"),
                _tag_temporal_frame(quarterly_frame, block_type="quarter", block_column="exit_quarter"),
            ],
            ignore_index=True,
        )

        results_frame = self._append_reference_deltas(results_frame)
        hourly_frame = self._append_hourly_deltas(hourly_frame)
        side_frame = self._append_side_deltas(side_frame)
        ranking = self._build_candidate_ranking(results_frame)
        filter_ablation = self._build_filter_ablation_summary(results_frame)
        conclusion = self._build_conclusion(
            results_frame=results_frame,
            ranking=ranking,
            filter_ablation=filter_ablation,
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
            "drawdown_convention": "max_drawdown is stored as a fraction of equity. Example: 0.0272 means 2.72%.",
            "summary_thresholds": asdict(self.experiment.summary_thresholds),
            "variants": {name: _sanitize_value(artifact.metrics) for name, artifact in variant_artifacts.items()},
            "conclusion": conclusion,
        }

        comparison_path = output_path / "session_trend_30m_zoom_comparison.json"
        results_path = output_path / "session_trend_30m_zoom_results.csv"
        summary_path = output_path / "session_trend_30m_zoom_summary.md"
        ranking_path = output_path / "candidate_ranking.csv"
        hourly_summary_path = output_path / "hourly_variant_summary.csv"
        side_summary_path = output_path / "side_variant_summary.csv"
        temporal_summary_path = output_path / "temporal_variant_summary.csv"
        filter_ablation_summary_path = output_path / "filter_ablation_summary.csv"

        comparison_path.write_text(json.dumps(_sanitize_value(payload), indent=2), encoding="utf-8")
        results_frame.sort_values(
            [
                "candidate",
                "passes_quality_guard",
                "within_target_drawdown_band",
                "net_pnl",
                "expectancy",
                "profit_factor",
                "max_drawdown",
                "trades_per_week_avg",
            ],
            ascending=[False, False, False, False, False, False, True, False],
        ).to_csv(results_path, index=False)
        ranking.to_csv(ranking_path, index=False)
        hourly_frame.to_csv(hourly_summary_path, index=False)
        side_frame.to_csv(side_summary_path, index=False)
        temporal_frame.to_csv(temporal_summary_path, index=False)
        filter_ablation.to_csv(filter_ablation_summary_path, index=False)
        activity_summary.to_csv(output_path / "activity_summary.csv", index=False)
        yearly_frame.to_csv(output_path / "yearly_variant_summary.csv", index=False)
        quarterly_frame.to_csv(output_path / "quarterly_variant_summary.csv", index=False)
        summary_path.write_text(
            self._build_summary_markdown(
                results_frame=results_frame,
                ranking=ranking,
                filter_ablation=filter_ablation,
                conclusion=conclusion,
            ),
            encoding="utf-8",
        )

        return SessionTrendZoomArtifacts(
            output_dir=output_path,
            comparison_path=comparison_path,
            results_path=results_path,
            summary_path=summary_path,
            ranking_path=ranking_path,
            hourly_summary_path=hourly_summary_path,
            side_summary_path=side_summary_path,
            temporal_summary_path=temporal_summary_path,
            filter_ablation_summary_path=filter_ablation_summary_path,
            variant_artifacts=variant_artifacts,
        )

    def _tag_variant_frame(self, spec: SessionTrendZoomVariantSpec, frame: pd.DataFrame) -> pd.DataFrame:
        tagged = frame.copy()
        tagged.insert(0, "variant", spec.name)
        tagged.insert(1, "label", spec.label)
        tagged.insert(2, "candidate", spec.candidate)
        tagged.insert(3, "source_variant", spec.source_variant)
        for key, value in spec.tags.items():
            tagged[f"tag_{key}"] = value
        return tagged

    def _build_variant_specs(self, *, selected_variants: Sequence[str] | None) -> list[SessionTrendZoomVariantSpec]:
        requested = set(selected_variants or ())
        specs: list[SessionTrendZoomVariantSpec] = []
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
                SessionTrendZoomVariantSpec(
                    name=variant.name,
                    label=variant.label,
                    candidate=variant.candidate,
                    source_variant=source_variant,
                    tags=variant.tags,
                    settings=settings,
                )
            )
        return specs

    def _build_variant_metrics(
        self,
        *,
        spec: SessionTrendZoomVariantSpec,
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
        trades_per_year = float(baseline_metrics["trades_per_year"])
        max_drawdown = float(baseline_metrics["max_drawdown"])
        within_target_band = thresholds.target_drawdown_min <= max_drawdown <= thresholds.target_drawdown_max
        passes_sample_guard = (
            trades_total >= thresholds.minimum_trades_total
            and trades_per_year >= thresholds.minimum_trades_per_year
        )
        passes_temporal_gate = (
            positive_year_ratio >= thresholds.minimum_positive_year_ratio
            and positive_quarter_ratio >= thresholds.minimum_positive_quarter_ratio
            and _net_concentration(yearly_breakdown) <= thresholds.maximum_year_net_pnl_concentration
        )

        metrics = {
            "variant": spec.name,
            "label": spec.label,
            "candidate": spec.candidate,
            "source_variant": spec.source_variant,
            "symbol": report_payload["symbol"],
            "strategy": variant_summary_payload(spec.settings),
            "number_of_trades": trades_total,
            "trades_per_year": trades_per_year,
            "trades_per_month": trades_per_year / 12.0 if trades_per_year else 0.0,
            "trades_per_month_avg": trades_per_year / 12.0 if trades_per_year else 0.0,
            "trades_per_week_avg": float(baseline_metrics["trades_per_week_avg"]),
            "percentage_of_days_with_trade": _pct_periods_with_trade(enriched_trades, "entry_timestamp", "D"),
            "percentage_of_weeks_with_trade": _pct_periods_with_trade(enriched_trades, "entry_timestamp", "W-SUN"),
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
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100.0,
            "sharpe": float(baseline_metrics["sharpe"]),
            "sortino": float(baseline_metrics["sortino"]),
            "calmar": float(baseline_metrics["calmar"]),
            "average_holding_bars": float(baseline_metrics["average_holding_bars"]),
            "positive_year_ratio": positive_year_ratio,
            "positive_quarter_ratio": positive_quarter_ratio,
            "year_net_pnl_concentration": _net_concentration(yearly_breakdown),
            "quarter_net_pnl_concentration": _net_concentration(quarterly_breakdown),
            "long_trades": long_trades,
            "short_trades": short_trades,
            "passes_sample_guard": passes_sample_guard,
            "passes_temporal_gate": passes_temporal_gate,
            "within_target_drawdown_band": within_target_band,
            "passes_quality_guard": (
                spec.candidate
                and float(baseline_metrics["profit_factor"]) > thresholds.minimum_profit_factor
                and float(baseline_metrics["expectancy"]) > thresholds.minimum_expectancy
                and float(baseline_metrics["net_pnl"]) > 0.0
                and max_drawdown <= thresholds.maximum_drawdown
                and passes_temporal_gate
            ),
            "strong_profit_factor": float(baseline_metrics["profit_factor"]) > thresholds.strong_profit_factor,
        }
        for key, value in spec.tags.items():
            metrics[f"tag_{key}"] = value
        return metrics

    def _append_reference_deltas(self, results_frame: pd.DataFrame) -> pd.DataFrame:
        if results_frame.empty:
            return results_frame
        enriched = results_frame.copy()
        reference = enriched.loc[enriched["variant"] == "reference"]
        if reference.empty:
            return enriched
        ref = reference.iloc[0]
        for metric in ["net_pnl", "gross_pnl", "profit_factor", "expectancy", "max_drawdown", "trades_per_week_avg"]:
            enriched[f"delta_{metric}_vs_reference"] = enriched[metric] - float(ref[metric])
        return enriched

    def _append_hourly_deltas(self, hourly_frame: pd.DataFrame) -> pd.DataFrame:
        if hourly_frame.empty:
            return hourly_frame
        reference = hourly_frame.loc[hourly_frame["variant"] == "reference", ["signal_hour_utc", "net_pnl"]].rename(
            columns={"net_pnl": "reference_net_pnl"}
        )
        if reference.empty:
            return hourly_frame
        merged = hourly_frame.merge(reference, on="signal_hour_utc", how="left")
        merged["delta_net_pnl_vs_reference"] = merged["net_pnl"] - merged["reference_net_pnl"].fillna(0.0)
        return merged

    def _append_side_deltas(self, side_frame: pd.DataFrame) -> pd.DataFrame:
        if side_frame.empty:
            return side_frame
        reference = side_frame.loc[side_frame["variant"] == "reference", ["side", "net_pnl"]].rename(
            columns={"net_pnl": "reference_net_pnl"}
        )
        if reference.empty:
            return side_frame
        merged = side_frame.merge(reference, on="side", how="left")
        merged["delta_net_pnl_vs_reference"] = merged["net_pnl"] - merged["reference_net_pnl"].fillna(0.0)
        return merged

    def _build_candidate_ranking(self, results_frame: pd.DataFrame) -> pd.DataFrame:
        ranking = results_frame.loc[results_frame["candidate"]].copy()
        if ranking.empty:
            return pd.DataFrame(columns=["rank", "variant"])
        ranking = ranking.sort_values(
            [
                "passes_quality_guard",
                "passes_sample_guard",
                "within_target_drawdown_band",
                "net_pnl",
                "expectancy",
                "profit_factor",
                "max_drawdown",
                "sharpe",
                "trades_per_week_avg",
            ],
            ascending=[False, False, False, False, False, False, True, False, False],
        ).reset_index(drop=True)
        ranking.insert(0, "rank", range(1, len(ranking) + 1))
        return ranking

    def _build_filter_ablation_summary(self, results_frame: pd.DataFrame) -> pd.DataFrame:
        if results_frame.empty:
            return pd.DataFrame(columns=["variant"])
        columns = [
            "variant",
            "label",
            "candidate",
            "tag_profile",
            "tag_focus_group",
            "net_pnl",
            "gross_pnl",
            "profit_factor",
            "expectancy",
            "max_drawdown",
            "max_drawdown_pct",
            "trades_per_week_avg",
            "positive_year_ratio",
            "positive_quarter_ratio",
            "passes_quality_guard",
            "within_target_drawdown_band",
            "delta_net_pnl_vs_reference",
            "delta_profit_factor_vs_reference",
            "delta_expectancy_vs_reference",
            "delta_max_drawdown_vs_reference",
            "delta_trades_per_week_avg_vs_reference",
        ]
        extra_tag_columns = sorted(column for column in results_frame.columns if column.startswith("tag_"))
        available: list[str] = []
        for column in columns + extra_tag_columns:
            if column in results_frame.columns and column not in available:
                available.append(column)
        return results_frame[available].sort_values(
            ["tag_profile", "tag_focus_group", "net_pnl", "trades_per_week_avg"],
            ascending=[True, True, False, False],
        )

    def _best_variant_row(
        self,
        frame: pd.DataFrame,
        *,
        fallback: pd.DataFrame,
        sort_columns: list[str],
        ascending: list[bool],
    ) -> pd.Series:
        pool = frame if not frame.empty else fallback
        return pool.sort_values(sort_columns, ascending=ascending).iloc[0]

    def _top_filter_deltas(self, results_frame: pd.DataFrame, *, positive: bool) -> list[dict[str, Any]]:
        profile_series = (
            results_frame["tag_profile"]
            if "tag_profile" in results_frame.columns
            else pd.Series("single_axis", index=results_frame.index)
        )
        single_axis = results_frame.loc[results_frame["candidate"] & (profile_series == "single_axis")].copy()
        if single_axis.empty:
            return []
        comparator = (
            single_axis.loc[
                (single_axis["delta_trades_per_week_avg_vs_reference"] > 0.0)
                & (single_axis["passes_quality_guard"])
            ]
            if positive
            else single_axis
        )
        if comparator.empty:
            comparator = single_axis
        comparator = comparator.sort_values(
            ["delta_trades_per_week_avg_vs_reference", "net_pnl", "profit_factor", "max_drawdown"],
            ascending=[False, False, False, True],
        )
        return [_sanitize_value(row.to_dict()) for _, row in comparator.head(3).iterrows()]

    def _build_conclusion(
        self,
        *,
        results_frame: pd.DataFrame,
        ranking: pd.DataFrame,
        filter_ablation: pd.DataFrame,
        yearly_frame: pd.DataFrame,
        quarterly_frame: pd.DataFrame,
    ) -> dict[str, Any]:
        if results_frame.empty or ranking.empty:
            return {
                "headline": "No session_trend_30m zoom variants were executed.",
                "best_profitability_variant": None,
                "best_balance_variant": None,
                "best_drawdown_band_variant": None,
                "controls": {},
                "frequency_killer_filters": [],
                "edge_helping_filters": [],
                "best_combo_variants": [],
                "ppo_readiness": "Do not move to PPO.",
            }

        candidates = results_frame.loc[results_frame["candidate"]].copy()
        strong_pool = candidates.loc[candidates["passes_quality_guard"]].copy()
        if strong_pool.empty:
            strong_pool = candidates

        best_profitability = self._best_variant_row(
            strong_pool,
            fallback=candidates,
            sort_columns=["net_pnl", "expectancy", "profit_factor", "max_drawdown", "sharpe", "trades_per_week_avg"],
            ascending=[False, False, False, True, False, False],
        )
        best_balance = ranking.iloc[0]
        drawdown_band_pool = strong_pool.loc[strong_pool["within_target_drawdown_band"]].copy()
        best_drawdown_band = self._best_variant_row(
            drawdown_band_pool,
            fallback=strong_pool,
            sort_columns=["within_target_drawdown_band", "net_pnl", "profit_factor", "expectancy", "max_drawdown", "trades_per_week_avg"],
            ascending=[False, False, False, False, True, False],
        )

        combo_profile = (
            candidates["tag_profile"]
            if "tag_profile" in candidates.columns
            else pd.Series("single_axis", index=candidates.index)
        )
        combo_pool = candidates.loc[combo_profile == "combo"].copy()
        combo_pool = combo_pool.sort_values(
            ["passes_quality_guard", "net_pnl", "profit_factor", "expectancy", "max_drawdown", "trades_per_week_avg"],
            ascending=[False, False, False, False, True, False],
        )

        headline = (
            f"`{best_balance['variant']}` is the strongest zoomed candidate around `session_trend_30m` when ranking profitability first, "
            f"drawdown second, and frequency third."
        )
        if bool(best_balance["within_target_drawdown_band"]):
            headline += " It also sits inside the target drawdown band (1%-5%)."

        ppo_readiness = (
            "Closer, but still no PPO yet. This subfamily now deserves a dedicated robustness phase first."
            if bool(best_balance["passes_quality_guard"])
            else "Still no PPO. The zoom did not produce a clean enough candidate yet."
        )

        return {
            "headline": headline,
            "best_profitability_variant": _sanitize_value(best_profitability.to_dict()),
            "best_balance_variant": _sanitize_value(best_balance.to_dict()),
            "best_drawdown_band_variant": _sanitize_value(best_drawdown_band.to_dict()),
            "controls": {
                "active_orb_reclaim_30m_control": _reference_snapshot(results_frame, "active_orb_reclaim_30m_control"),
                "context_reclaim_15m_control": _reference_snapshot(results_frame, "context_reclaim_15m_control"),
            },
            "frequency_killer_filters": self._top_filter_deltas(filter_ablation, positive=True),
            "edge_helping_filters": [
                _sanitize_value(row.to_dict())
                for _, row in strong_pool.sort_values(
                    ["delta_net_pnl_vs_reference", "profit_factor", "expectancy", "max_drawdown"],
                    ascending=[False, False, False, True],
                ).head(3).iterrows()
            ],
            "best_combo_variants": [_sanitize_value(row.to_dict()) for _, row in combo_pool.head(3).iterrows()],
            "yearly_snapshot": _temporal_snapshot(yearly_frame, best_balance["variant"], "exit_year"),
            "quarterly_snapshot": _temporal_snapshot(quarterly_frame, best_balance["variant"], "exit_quarter"),
            "ppo_readiness": ppo_readiness,
        }

    def _build_summary_markdown(
        self,
        *,
        results_frame: pd.DataFrame,
        ranking: pd.DataFrame,
        filter_ablation: pd.DataFrame,
        conclusion: dict[str, Any],
    ) -> str:
        best_profitability = conclusion["best_profitability_variant"]
        best_balance = conclusion["best_balance_variant"]
        best_drawdown_band = conclusion["best_drawdown_band_variant"]
        ordered = results_frame.sort_values(
            [
                "candidate",
                "passes_quality_guard",
                "within_target_drawdown_band",
                "net_pnl",
                "expectancy",
                "profit_factor",
                "max_drawdown",
                "trades_per_week_avg",
            ],
            ascending=[False, False, False, False, False, False, True, False],
        )
        lines = [
            "# Session Trend 30m Zoom Summary",
            "",
            "## Direct Answer",
            conclusion["headline"],
            "",
            "- Drawdown convention: `max_drawdown` is stored as a fraction of equity. Example: `0.0272 = 2.72%`.",
            (
                f"- Best profitability variant: `{best_profitability['variant']}` | net `{best_profitability['net_pnl']:.2f}` | "
                f"PF `{best_profitability['profit_factor']:.2f}` | expectancy `{best_profitability['expectancy']:.2f}` | "
                f"DD `{best_profitability['max_drawdown_pct']:.2f}%` | trades/week `{best_profitability['trades_per_week_avg']:.3f}`."
            ),
            (
                f"- Best balance variant: `{best_balance['variant']}` | net `{best_balance['net_pnl']:.2f}` | "
                f"PF `{best_balance['profit_factor']:.2f}` | expectancy `{best_balance['expectancy']:.2f}` | "
                f"DD `{best_balance['max_drawdown_pct']:.2f}%` | trades/week `{best_balance['trades_per_week_avg']:.3f}`."
            ),
            (
                f"- Best variant inside the 1%-5% DD band: `{best_drawdown_band['variant']}` | DD `{best_drawdown_band['max_drawdown_pct']:.2f}%` | "
                f"net `{best_drawdown_band['net_pnl']:.2f}` | PF `{best_drawdown_band['profit_factor']:.2f}`."
            ),
            f"- PPO readiness: {conclusion['ppo_readiness']}",
            "",
            "## Ranking",
        ]
        for _, row in ranking.head(8).iterrows():
            lines.append(
                f"- `#{int(row['rank'])} {row['variant']}`: net `{row['net_pnl']:.2f}`, expectancy `{row['expectancy']:.2f}`, "
                f"PF `{row['profit_factor']:.2f}`, DD `{row['max_drawdown_pct']:.2f}%`, trades/week `{row['trades_per_week_avg']:.3f}`."
            )

        lines.extend(["", "## Variants"])
        for _, row in ordered.iterrows():
            lines.append(
                f"- `{row['variant']}`: net `{row['net_pnl']:.2f}`, gross `{row['gross_pnl']:.2f}`, trades `{int(row['number_of_trades'])}`, "
                f"trades/year `{row['trades_per_year']:.2f}`, trades/month `{row['trades_per_month']:.2f}`, "
                f"weeks active `{row['percentage_of_weeks_with_trade'] * 100:.1f}%`, PF `{row['profit_factor']:.2f}`, "
                f"expectancy `{row['expectancy']:.2f}`, DD `{row['max_drawdown_pct']:.2f}%`."
            )

        lines.extend(["", "## Filters That Seem To Kill Too Much Frequency"])
        for item in conclusion["frequency_killer_filters"]:
            lines.append(
                f"- `{item['variant']}`: delta trades/week `{item['delta_trades_per_week_avg_vs_reference']:.3f}`, "
                f"delta net `{item['delta_net_pnl_vs_reference']:.2f}`, PF `{item['profit_factor']:.2f}`, DD `{item['max_drawdown_pct']:.2f}%`."
            )

        lines.extend(["", "## Filters That Still Add Edge"])
        for item in conclusion["edge_helping_filters"]:
            lines.append(
                f"- `{item['variant']}`: delta net `{item['delta_net_pnl_vs_reference']:.2f}`, "
                f"delta PF `{item['delta_profit_factor_vs_reference']:.2f}`, delta DD `{item['delta_max_drawdown_vs_reference'] * 100:.2f} pp`, "
                f"trades/week `{item['trades_per_week_avg']:.3f}`."
            )

        lines.extend(["", "## Best Simultaneous Combinations"])
        for item in conclusion["best_combo_variants"]:
            lines.append(
                f"- `{item['variant']}`: net `{item['net_pnl']:.2f}`, PF `{item['profit_factor']:.2f}`, "
                f"expectancy `{item['expectancy']:.2f}`, DD `{item['max_drawdown_pct']:.2f}%`, trades/week `{item['trades_per_week_avg']:.3f}`."
            )

        lines.extend(["", "## Temporal Snapshot Of The Best Balance Variant"])
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
                "- `session_trend_30m_zoom_results.csv` is the main table for profitability first, drawdown second, frequency third.",
                "- `candidate_ranking.csv` is the shortlist of the most serious session-trend zoom candidates.",
                "- `hourly_variant_summary.csv` and `side_variant_summary.csv` show where improvements or damage concentrate.",
                "- `filter_ablation_summary.csv` is the fastest view of which filter changes helped or hurt relative to `reference`.",
                "- Each variant keeps its own `baseline/` and `diagnostics/` artifacts inside `variants/`.",
            ]
        )
        return "\n".join(lines) + "\n"


def load_session_trend_30m_zoom_config(path: str | Path) -> SessionTrendZoomConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Session-trend zoom config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Session-trend zoom config must be a mapping: {config_path}")

    thresholds_payload = payload.get("summary_thresholds", {}) or {}
    thresholds = SessionTrendZoomThresholds(
        minimum_trades_total=int(thresholds_payload.get("minimum_trades_total", 40)),
        minimum_trades_per_year=float(thresholds_payload.get("minimum_trades_per_year", 12.0)),
        minimum_profit_factor=float(thresholds_payload.get("minimum_profit_factor", 1.05)),
        strong_profit_factor=float(thresholds_payload.get("strong_profit_factor", 1.10)),
        minimum_expectancy=float(thresholds_payload.get("minimum_expectancy", 0.0)),
        maximum_drawdown=float(thresholds_payload.get("maximum_drawdown", 0.08)),
        target_drawdown_min=float(thresholds_payload.get("target_drawdown_min", 0.01)),
        target_drawdown_max=float(thresholds_payload.get("target_drawdown_max", 0.05)),
        minimum_positive_year_ratio=float(thresholds_payload.get("minimum_positive_year_ratio", 0.50)),
        minimum_positive_quarter_ratio=float(thresholds_payload.get("minimum_positive_quarter_ratio", 0.50)),
        maximum_year_net_pnl_concentration=float(
            thresholds_payload.get("maximum_year_net_pnl_concentration", 0.75)
        ),
    )

    raw_variants = payload.get("variants", [])
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("Session-trend zoom config requires a non-empty 'variants' list.")

    variants: list[SessionTrendZoomVariantConfig] = []
    for raw_variant in raw_variants:
        if not isinstance(raw_variant, dict):
            raise ValueError("Each session-trend zoom variant must be a mapping.")
        overrides = raw_variant.get("overrides", {}) or {}
        if not isinstance(overrides, dict):
            raise ValueError("Variant overrides must be a mapping.")
        tags = raw_variant.get("tags", {}) or {}
        if not isinstance(tags, dict):
            raise ValueError("Variant tags must be a mapping.")
        source_variant = raw_variant.get("source_variant")
        variants.append(
            SessionTrendZoomVariantConfig(
                name=str(raw_variant["name"]),
                label=str(raw_variant.get("label", raw_variant["name"])),
                candidate=bool(raw_variant.get("candidate", True)),
                source_variant=str(source_variant) if source_variant else None,
                tags={str(key): str(value) for key, value in tags.items()},
                overrides=overrides,
            )
        )

    return SessionTrendZoomConfig(
        name=str(payload.get("name", "session_trend_30m_zoom")),
        base_variant=str(payload.get("base_variant", "session_trend_30m")),
        summary_thresholds=thresholds,
        variants=tuple(variants),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zoom in on session_trend_30m with controlled filter combinations.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--experiment-config", default="configs/experiments/session_trend_30m_zoom.yaml")
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

    experiment = load_session_trend_30m_zoom_config(args.experiment_config)
    runner = SessionTrend30mZoomRunner(args.config_dir, experiment)
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
    best = payload["conclusion"]["best_balance_variant"]
    print(f"Session-trend zoom comparison: {artifacts.comparison_path}")
    print(f"Session-trend zoom summary: {artifacts.summary_path}")
    print(
        " ".join(
            [
                f"variants={len(payload['variants'])}",
                f"best_balance={best['variant'] if best else 'n/a'}",
                f"best_profitability={payload['conclusion']['best_profitability_variant']['variant'] if payload['conclusion']['best_profitability_variant'] else 'n/a'}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
