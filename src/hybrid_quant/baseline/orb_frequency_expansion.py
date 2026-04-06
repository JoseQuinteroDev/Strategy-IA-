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
    _reference_temporal_snapshot,
    _sanitize_value,
    _tag_temporal_frame,
)
from .variants import load_variant_settings, variant_summary_payload


@dataclass(slots=True)
class OrbFrequencySummaryThresholds:
    minimum_trades_total: int = 30
    minimum_trades_per_year: float = 12.0
    target_trades_per_year: float = 15.0
    minimum_frequency_uplift_ratio: float = 1.05
    minimum_expectancy_ratio_vs_reference: float = 0.75
    minimum_profit_factor_ratio_vs_reference: float = 0.90
    maximum_drawdown_multiple_vs_reference: float = 2.00
    minimum_positive_year_ratio: float = 0.50
    minimum_positive_quarter_ratio: float = 0.35
    maximum_year_net_pnl_concentration: float = 0.80


@dataclass(slots=True)
class OrbFrequencyVariantConfig:
    name: str
    label: str
    candidate: bool
    overrides: dict[str, Any]


@dataclass(slots=True)
class OrbFrequencyExpansionConfig:
    name: str
    base_variant: str
    summary_thresholds: OrbFrequencySummaryThresholds
    variants: tuple[OrbFrequencyVariantConfig, ...]


@dataclass(slots=True)
class OrbFrequencyVariantSpec:
    name: str
    label: str
    candidate: bool
    settings: Settings


@dataclass(slots=True)
class OrbFrequencyVariantArtifacts:
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
class OrbFrequencyExpansionArtifacts:
    output_dir: Path
    comparison_path: Path
    results_path: Path
    summary_path: Path
    temporal_blocks_path: Path
    activity_summary_path: Path
    variant_artifacts: dict[str, OrbFrequencyVariantArtifacts]


class OrbFrequencyExpansionRunner:
    def __init__(self, config_dir: str | Path, experiment: OrbFrequencyExpansionConfig) -> None:
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
    ) -> OrbFrequencyExpansionArtifacts:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        frame = _filter_frame_by_range(input_frame, start=start, end=end)
        specs = self._build_variant_specs(selected_variants=selected_variants)
        if not specs:
            raise ValueError("The ORB frequency expansion config produced no variants to run.")

        variant_artifacts: dict[str, OrbFrequencyVariantArtifacts] = {}
        yearly_rows: list[pd.DataFrame] = []
        quarterly_rows: list[pd.DataFrame] = []
        curve_rows: list[pd.DataFrame] = []

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
            yearly_curve = pd.read_csv(diagnostics_artifacts.yearly_equity_curve_path)
            trades_frame = _read_trade_frame(artifact_dir / "trades.csv")

            if not yearly_breakdown.empty:
                yearly_breakdown.insert(0, "variant", spec.name)
                yearly_breakdown.insert(1, "label", spec.label)
                yearly_breakdown.insert(2, "candidate", spec.candidate)
                yearly_rows.append(yearly_breakdown)
            if not quarterly_breakdown.empty:
                quarterly_breakdown.insert(0, "variant", spec.name)
                quarterly_breakdown.insert(1, "label", spec.label)
                quarterly_breakdown.insert(2, "candidate", spec.candidate)
                quarterly_rows.append(quarterly_breakdown)
            if not yearly_curve.empty:
                yearly_curve.insert(0, "variant", spec.name)
                yearly_curve.insert(1, "label", spec.label)
                curve_rows.append(yearly_curve)

            activity_metrics = _build_trade_activity_metrics(frame=frame, trades_frame=trades_frame)
            metrics = self._build_variant_metrics(
                spec=spec,
                report_payload=report_payload,
                diagnostics_payload=diagnostics_payload,
                yearly_breakdown=yearly_breakdown,
                quarterly_breakdown=quarterly_breakdown,
                activity_metrics=activity_metrics,
            )
            variant_artifacts[spec.name] = OrbFrequencyVariantArtifacts(
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
        results_frame = self._attach_reference_relative_metrics(results_frame)

        yearly_frame = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()
        quarterly_frame = pd.concat(quarterly_rows, ignore_index=True) if quarterly_rows else pd.DataFrame()
        yearly_curve_frame = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
        temporal_blocks = pd.concat(
            [
                _tag_temporal_frame(yearly_frame, block_type="year", block_column="exit_year"),
                _tag_temporal_frame(quarterly_frame, block_type="quarter", block_column="exit_quarter"),
            ],
            ignore_index=True,
        )

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
            "summary_thresholds": asdict(self.experiment.summary_thresholds),
            "variants": {name: _sanitize_value(artifact.metrics) for name, artifact in variant_artifacts.items()},
            "conclusion": conclusion,
        }

        comparison_path = output_path / "orb_frequency_expansion_comparison.json"
        results_path = output_path / "orb_frequency_expansion_results.csv"
        summary_path = output_path / "orb_frequency_expansion_summary.md"
        temporal_blocks_path = output_path / "temporal_block_results.csv"
        activity_summary_path = output_path / "activity_summary.csv"

        comparison_path.write_text(json.dumps(_sanitize_value(payload), indent=2), encoding="utf-8")
        results_frame.sort_values(
            ["passes_balance_gate", "trades_per_year", "net_pnl", "sharpe"],
            ascending=[False, False, False, False],
        ).to_csv(results_path, index=False)
        temporal_blocks.to_csv(temporal_blocks_path, index=False)
        yearly_frame.to_csv(output_path / "yearly_variant_summary.csv", index=False)
        quarterly_frame.to_csv(output_path / "quarterly_variant_summary.csv", index=False)
        yearly_curve_frame.to_csv(output_path / "yearly_equity_curve_summary.csv", index=False)
        results_frame[
            [
                "variant",
                "label",
                "candidate",
                "number_of_trades",
                "trades_per_year",
                "trades_per_month_avg",
                "trades_per_week_avg",
                "pct_days_with_trade",
                "pct_weeks_with_trade",
                "frequency_uplift_ratio",
                "expectancy",
                "profit_factor",
                "max_drawdown",
                "sharpe",
                "positive_year_ratio",
                "positive_quarter_ratio",
                "passes_frequency_gate",
                "passes_quality_guard",
                "passes_temporal_gate",
                "passes_balance_gate",
                "meets_reasonable_frequency",
            ]
        ].sort_values(["passes_balance_gate", "trades_per_year"], ascending=[False, False]).to_csv(
            activity_summary_path,
            index=False,
        )
        summary_path.write_text(
            self._build_summary_markdown(
                results_frame=results_frame,
                yearly_frame=yearly_frame,
                quarterly_frame=quarterly_frame,
                conclusion=conclusion,
            ),
            encoding="utf-8",
        )

        return OrbFrequencyExpansionArtifacts(
            output_dir=output_path,
            comparison_path=comparison_path,
            results_path=results_path,
            summary_path=summary_path,
            temporal_blocks_path=temporal_blocks_path,
            activity_summary_path=activity_summary_path,
            variant_artifacts=variant_artifacts,
        )

    def _build_variant_specs(self, *, selected_variants: Sequence[str] | None) -> list[OrbFrequencyVariantSpec]:
        specs: list[OrbFrequencyVariantSpec] = []
        requested = set(selected_variants or ())
        for variant in self.experiment.variants:
            if requested and variant.name not in requested:
                continue
            settings = apply_settings_overrides(self.base_settings, variant.overrides)
            settings = apply_settings_overrides(
                settings,
                {
                    "strategy": {
                        "variant_name": variant.name,
                    }
                },
            )
            specs.append(
                OrbFrequencyVariantSpec(
                    name=variant.name,
                    label=variant.label,
                    candidate=variant.candidate,
                    settings=settings,
                )
            )
        return specs

    def _build_variant_metrics(
        self,
        *,
        spec: OrbFrequencyVariantSpec,
        report_payload: dict[str, Any],
        diagnostics_payload: dict[str, Any],
        yearly_breakdown: pd.DataFrame,
        quarterly_breakdown: pd.DataFrame,
        activity_metrics: dict[str, float],
    ) -> dict[str, Any]:
        baseline_metrics = diagnostics_payload["baseline_metrics"]
        thresholds = self.experiment.summary_thresholds
        positive_year_ratio = float((yearly_breakdown["net_pnl"] > 0.0).mean()) if not yearly_breakdown.empty else 0.0
        positive_quarter_ratio = (
            float((quarterly_breakdown["net_pnl"] > 0.0).mean()) if not quarterly_breakdown.empty else 0.0
        )
        year_net_pnl_concentration = _net_concentration(yearly_breakdown)
        quarter_net_pnl_concentration = _net_concentration(quarterly_breakdown)
        trades_total = int(baseline_metrics["number_of_trades"])
        trades_per_year = float(baseline_metrics["trades_per_year"])
        passes_min_sample = (
            trades_total >= thresholds.minimum_trades_total
            and trades_per_year >= thresholds.minimum_trades_per_year
        )
        return {
            "variant": spec.name,
            "label": spec.label,
            "candidate": spec.candidate,
            "symbol": report_payload["symbol"],
            "strategy": variant_summary_payload(spec.settings),
            "number_of_trades": trades_total,
            "trades_per_year": trades_per_year,
            "trades_per_month_avg": float(activity_metrics["trades_per_month_avg"]),
            "trades_per_week_avg": float(activity_metrics["trades_per_week_avg"]),
            "pct_days_with_trade": float(activity_metrics["pct_days_with_trade"]),
            "pct_weeks_with_trade": float(activity_metrics["pct_weeks_with_trade"]),
            "trade_days": int(activity_metrics["trade_days"]),
            "trade_weeks": int(activity_metrics["trade_weeks"]),
            "trade_months": int(activity_metrics["trade_months"]),
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
            "profitable_years_pct": float(baseline_metrics["profitable_years_pct"]),
            "profitable_months_pct": float(baseline_metrics["profitable_months_pct"]),
            "max_consecutive_losses": int(baseline_metrics["max_consecutive_losses"]),
            "average_holding_bars": float(baseline_metrics["average_holding_bars"]),
            "average_mfe_atr": float(baseline_metrics["average_mfe_atr"]),
            "average_mae_atr": float(baseline_metrics["average_mae_atr"]),
            "positive_year_ratio": positive_year_ratio,
            "positive_quarter_ratio": positive_quarter_ratio,
            "years_with_trades": int(len(yearly_breakdown)),
            "quarters_with_trades": int(len(quarterly_breakdown)),
            "year_net_pnl_concentration": year_net_pnl_concentration,
            "quarter_net_pnl_concentration": quarter_net_pnl_concentration,
            "best_year_net_pnl": float(yearly_breakdown["net_pnl"].max()) if not yearly_breakdown.empty else 0.0,
            "worst_year_net_pnl": float(yearly_breakdown["net_pnl"].min()) if not yearly_breakdown.empty else 0.0,
            "best_quarter_net_pnl": float(quarterly_breakdown["net_pnl"].max()) if not quarterly_breakdown.empty else 0.0,
            "worst_quarter_net_pnl": float(quarterly_breakdown["net_pnl"].min()) if not quarterly_breakdown.empty else 0.0,
            "passes_min_sample": passes_min_sample,
            "passes_temporal_gate": (
                positive_year_ratio >= thresholds.minimum_positive_year_ratio
                and positive_quarter_ratio >= thresholds.minimum_positive_quarter_ratio
                and year_net_pnl_concentration <= thresholds.maximum_year_net_pnl_concentration
            ),
        }

    def _attach_reference_relative_metrics(self, results_frame: pd.DataFrame) -> pd.DataFrame:
        if results_frame.empty:
            return results_frame
        enriched = results_frame.copy()
        reference = enriched.loc[enriched["variant"] == "reference"]
        if reference.empty:
            raise ValueError("ORB frequency expansion requires a `reference` variant.")

        reference_row = reference.iloc[0]
        thresholds = self.experiment.summary_thresholds
        reference_trades_per_year = max(float(reference_row["trades_per_year"]), 1e-9)
        reference_expectancy = float(reference_row["expectancy"])
        reference_profit_factor = max(float(reference_row["profit_factor"]), 1e-9)
        reference_drawdown = max(float(reference_row["max_drawdown"]), 1e-9)

        enriched["frequency_uplift_ratio"] = enriched["trades_per_year"] / reference_trades_per_year
        enriched["expectancy_ratio_vs_reference"] = enriched["expectancy"].apply(
            lambda value: float(value) / reference_expectancy if abs(reference_expectancy) > 1e-9 else 0.0
        )
        enriched["profit_factor_ratio_vs_reference"] = enriched["profit_factor"] / reference_profit_factor
        enriched["drawdown_multiple_vs_reference"] = enriched["max_drawdown"] / reference_drawdown
        enriched["meets_reasonable_frequency"] = enriched["trades_per_year"] >= thresholds.target_trades_per_year
        enriched["passes_frequency_gate"] = (
            enriched["passes_min_sample"]
            & (enriched["frequency_uplift_ratio"] >= thresholds.minimum_frequency_uplift_ratio)
        )
        enriched["passes_quality_guard"] = (
            (enriched["expectancy_ratio_vs_reference"] >= thresholds.minimum_expectancy_ratio_vs_reference)
            & (enriched["profit_factor_ratio_vs_reference"] >= thresholds.minimum_profit_factor_ratio_vs_reference)
            & (enriched["drawdown_multiple_vs_reference"] <= thresholds.maximum_drawdown_multiple_vs_reference)
            & (enriched["net_pnl"] > 0.0)
            & (enriched["sharpe"] > 0.0)
        )
        enriched["passes_balance_gate"] = (
            enriched["candidate"]
            & enriched["passes_frequency_gate"]
            & enriched["passes_quality_guard"]
            & enriched["passes_temporal_gate"]
        )
        return enriched

    def _build_conclusion(
        self,
        *,
        results_frame: pd.DataFrame,
        yearly_frame: pd.DataFrame,
        quarterly_frame: pd.DataFrame,
    ) -> dict[str, Any]:
        thresholds = self.experiment.summary_thresholds
        if results_frame.empty:
            return {
                "headline": "No ORB frequency variants were executed.",
                "best_frequency_variant": None,
                "best_balance_variant": None,
                "best_drawdown_variant": None,
                "still_promising": False,
                "frequency_assessment": "No results to assess.",
                "next_small_adjustment": "No recommendation available.",
                "ppo_readiness": "Do not move to PPO.",
            }

        reference = results_frame.loc[results_frame["variant"] == "reference"]
        if reference.empty:
            raise ValueError("ORB frequency expansion requires a reference row.")
        reference_row = reference.iloc[0]

        candidates = results_frame.loc[(results_frame["candidate"]) & (results_frame["variant"] != "reference")].copy()
        best_frequency = candidates.sort_values(
            ["trades_per_year", "net_pnl", "sharpe"],
            ascending=[False, False, False],
        ).iloc[0]

        balance_candidates = candidates.loc[candidates["passes_balance_gate"]].copy()
        best_balance = None
        if not balance_candidates.empty:
            best_balance = balance_candidates.sort_values(
                [
                    "meets_reasonable_frequency",
                    "trades_per_year",
                    "sharpe",
                    "net_pnl",
                    "max_drawdown",
                ],
                ascending=[False, False, False, False, True],
            ).iloc[0]

        drawdown_candidates = candidates.loc[candidates["passes_min_sample"]].copy()
        if drawdown_candidates.empty:
            drawdown_candidates = candidates
        best_drawdown = drawdown_candidates.sort_values(["max_drawdown", "net_pnl"], ascending=[True, False]).iloc[0]

        still_promising = bool(best_balance is not None)

        if best_balance is None:
            frequency_assessment = (
                "No local variant increased frequency enough while preserving expectancy, profit factor, drawdown control and temporal stability."
            )
            next_small_adjustment = (
                "Do not broaden the search space yet. One more tiny local pass is safer than opening a larger sweep."
            )
        else:
            frequency_assessment = (
                "A locally-expanded ORB variant does improve cadence without obviously breaking robustness."
                if bool(best_balance["meets_reasonable_frequency"])
                else "The ORB subfamily improves cadence somewhat, but still remains below a truly comfortable trading frequency."
            )
            next_small_adjustment = (
                f"`{best_balance['variant']}` is the best frequency-aware candidate because it clears the guard rails while improving activity over the reference."
            )

        if float(best_frequency["expectancy_ratio_vs_reference"]) < thresholds.minimum_expectancy_ratio_vs_reference:
            frequency_assessment += " The biggest frequency lift currently comes with a clear expectancy haircut."
        if float(best_frequency["profit_factor_ratio_vs_reference"]) < thresholds.minimum_profit_factor_ratio_vs_reference:
            frequency_assessment += " The biggest frequency lift also weakens profit factor more than the guard rails allow."

        ppo_readiness = (
            "Still no PPO. One more temporal robustness pass on the best frequency-aware candidate is the right next step."
            if still_promising
            else "Still no PPO. The frequency expansion did not yet produce a clean enough candidate."
        )

        return {
            "headline": (
                f"Frequency expansion around `{self.experiment.base_variant}` points to "
                f"`{best_frequency['variant']}` as the biggest cadence lift and "
                f"`{best_balance['variant'] if best_balance is not None else 'no variant'}` "
                "as the best frequency-quality balance."
            ),
            "reference_variant": _sanitize_value(reference_row.to_dict()),
            "best_frequency_variant": _sanitize_value(best_frequency.to_dict()),
            "best_balance_variant": _sanitize_value(best_balance.to_dict()) if best_balance is not None else None,
            "best_drawdown_variant": _sanitize_value(best_drawdown.to_dict()),
            "still_promising": still_promising,
            "frequency_assessment": frequency_assessment,
            "next_small_adjustment": next_small_adjustment,
            "ppo_readiness": ppo_readiness,
            "sample_warning": (
                f"The best balance candidate still runs only `{best_balance['trades_per_year']:.2f}` trades/year."
                if best_balance is not None and not bool(best_balance["meets_reasonable_frequency"])
                else None
            ),
            "yearly_reference": _reference_temporal_snapshot(yearly_frame, variant="reference", block_column="exit_year"),
            "quarterly_reference": _reference_temporal_snapshot(
                quarterly_frame,
                variant="reference",
                block_column="exit_quarter",
            ),
            "yearly_best_balance": (
                _reference_temporal_snapshot(
                    yearly_frame,
                    variant=str(best_balance["variant"]),
                    block_column="exit_year",
                )
                if best_balance is not None
                else []
            ),
            "quarterly_best_balance": (
                _reference_temporal_snapshot(
                    quarterly_frame,
                    variant=str(best_balance["variant"]),
                    block_column="exit_quarter",
                )
                if best_balance is not None
                else []
            ),
        }

    def _build_summary_markdown(
        self,
        *,
        results_frame: pd.DataFrame,
        yearly_frame: pd.DataFrame,
        quarterly_frame: pd.DataFrame,
        conclusion: dict[str, Any],
    ) -> str:
        lines = [
            "# ORB Frequency Expansion Summary",
            "",
            "## Direct Answer",
            conclusion["headline"],
            "",
            f"- Still promising: `{conclusion['still_promising']}`",
            f"- Frequency assessment: {conclusion['frequency_assessment']}",
            f"- PPO readiness: {conclusion['ppo_readiness']}",
        ]
        if conclusion.get("sample_warning"):
            lines.append(f"- Sample warning: {conclusion['sample_warning']}")

        reference = conclusion["reference_variant"]
        best_frequency = conclusion["best_frequency_variant"]
        best_balance = conclusion["best_balance_variant"]
        best_drawdown = conclusion["best_drawdown_variant"]

        lines.extend(
            [
                "",
                "## Reference",
                (
                    f"- `reference`: net `{reference['net_pnl']:.2f}` | DD `{reference['max_drawdown']:.4f}` "
                    f"| trades/year `{reference['trades_per_year']:.2f}` | weeks with trade `{reference['pct_weeks_with_trade']:.1%}`."
                ),
                "",
                "## Best Variants",
                (
                    f"- Biggest frequency lift: `{best_frequency['variant']}` | trades/year `{best_frequency['trades_per_year']:.2f}` "
                    f"| uplift `{best_frequency['frequency_uplift_ratio']:.2f}x` | net `{best_frequency['net_pnl']:.2f}` "
                    f"| PF `{best_frequency['profit_factor']:.2f}` | expectancy `{best_frequency['expectancy']:.2f}`."
                ),
            ]
        )
        if best_balance is None:
            lines.append("- Best balance: no candidate cleared the guard rails.")
        else:
            lines.append(
                f"- Best balance: `{best_balance['variant']}` | trades/year `{best_balance['trades_per_year']:.2f}` "
                f"| uplift `{best_balance['frequency_uplift_ratio']:.2f}x` | net `{best_balance['net_pnl']:.2f}` "
                f"| PF `{best_balance['profit_factor']:.2f}` | expectancy `{best_balance['expectancy']:.2f}` "
                f"| DD `{best_balance['max_drawdown']:.4f}`."
            )
        lines.append(
            f"- Lowest drawdown with usable sample: `{best_drawdown['variant']}` | DD `{best_drawdown['max_drawdown']:.4f}`."
        )

        lines.extend(["", "## Variants"])
        display = results_frame.sort_values(
            ["passes_balance_gate", "trades_per_year", "net_pnl"],
            ascending=[False, False, False],
        )
        for _, row in display.iterrows():
            guard_flag = "yes" if bool(row["passes_balance_gate"]) else "no"
            lines.append(
                f"- `{row['variant']}`: trades/year `{row['trades_per_year']:.2f}`, "
                f"trades/month `{row['trades_per_month_avg']:.2f}`, weeks with trade `{row['pct_weeks_with_trade']:.1%}`, "
                f"net `{row['net_pnl']:.2f}`, PF `{row['profit_factor']:.2f}`, expectancy `{row['expectancy']:.2f}`, "
                f"DD `{row['max_drawdown']:.4f}`, Sharpe `{row['sharpe']:.4f}`, guard rails `{guard_flag}`."
            )

        lines.extend(["", "## Temporal Reference"])
        for item in conclusion["yearly_reference"]:
            lines.append(
                f"- Reference year `{item['block']}`: trades `{item['trades']}`, net `{item['net_pnl']:.2f}`, PF `{item['profit_factor']:.2f}`."
            )
        for item in conclusion["quarterly_reference"][:8]:
            lines.append(
                f"- Reference quarter `{item['block']}`: trades `{item['trades']}`, net `{item['net_pnl']:.2f}`, PF `{item['profit_factor']:.2f}`."
            )

        if best_balance is not None:
            lines.extend(["", "## Temporal Best Balance"])
            for item in conclusion["yearly_best_balance"]:
                lines.append(
                    f"- Best-balance year `{item['block']}`: trades `{item['trades']}`, net `{item['net_pnl']:.2f}`, PF `{item['profit_factor']:.2f}`."
                )
            for item in conclusion["quarterly_best_balance"][:8]:
                lines.append(
                    f"- Best-balance quarter `{item['block']}`: trades `{item['trades']}`, net `{item['net_pnl']:.2f}`, PF `{item['profit_factor']:.2f}`."
                )

        lines.extend(
            [
                "",
                "## How To Read This",
                "- `orb_frequency_expansion_results.csv` is the clean comparison table by local frequency-aware variant.",
                "- `activity_summary.csv` isolates the cadence and guard-rail metrics.",
                "- `temporal_block_results.csv` merges yearly and quarterly blocks for all variants.",
                "- Each variant keeps its own `baseline/` and `diagnostics/` artifacts under `variants/`.",
                "",
                "## Next Small Step",
                f"- {conclusion['next_small_adjustment']}",
            ]
        )
        return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frequency-aware ORB sensitivity around the width_wider subfamily.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--experiment-config", default="configs/experiments/orb_frequency_expansion.yaml")
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

    experiment = load_orb_frequency_expansion_config(args.experiment_config)
    runner = OrbFrequencyExpansionRunner(args.config_dir, experiment)
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
    best_balance = payload["conclusion"]["best_balance_variant"]
    best_frequency = payload["conclusion"]["best_frequency_variant"]
    print(f"ORB frequency expansion comparison: {artifacts.comparison_path}")
    print(f"ORB frequency expansion summary: {artifacts.summary_path}")
    print(
        " ".join(
            [
                f"variants={len(payload['variants'])}",
                f"best_frequency={best_frequency['variant'] if best_frequency else 'n/a'}",
                f"best_balance={best_balance['variant'] if best_balance else 'n/a'}",
            ]
        )
    )
    return 0


def load_orb_frequency_expansion_config(path: str | Path) -> OrbFrequencyExpansionConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"ORB frequency expansion config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"ORB frequency expansion config must be a mapping: {config_path}")

    thresholds_payload = payload.get("summary_thresholds", {}) or {}
    thresholds = OrbFrequencySummaryThresholds(
        minimum_trades_total=int(thresholds_payload.get("minimum_trades_total", 30)),
        minimum_trades_per_year=float(thresholds_payload.get("minimum_trades_per_year", 12.0)),
        target_trades_per_year=float(thresholds_payload.get("target_trades_per_year", 15.0)),
        minimum_frequency_uplift_ratio=float(thresholds_payload.get("minimum_frequency_uplift_ratio", 1.05)),
        minimum_expectancy_ratio_vs_reference=float(
            thresholds_payload.get("minimum_expectancy_ratio_vs_reference", 0.75)
        ),
        minimum_profit_factor_ratio_vs_reference=float(
            thresholds_payload.get("minimum_profit_factor_ratio_vs_reference", 0.90)
        ),
        maximum_drawdown_multiple_vs_reference=float(
            thresholds_payload.get("maximum_drawdown_multiple_vs_reference", 2.00)
        ),
        minimum_positive_year_ratio=float(thresholds_payload.get("minimum_positive_year_ratio", 0.50)),
        minimum_positive_quarter_ratio=float(thresholds_payload.get("minimum_positive_quarter_ratio", 0.35)),
        maximum_year_net_pnl_concentration=float(
            thresholds_payload.get("maximum_year_net_pnl_concentration", 0.80)
        ),
    )

    raw_variants = payload.get("variants", [])
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("ORB frequency expansion config requires a non-empty 'variants' list.")

    variants: list[OrbFrequencyVariantConfig] = []
    for raw_variant in raw_variants:
        if not isinstance(raw_variant, dict):
            raise ValueError("Each ORB frequency expansion variant must be a mapping.")
        overrides = raw_variant.get("overrides", {}) or {}
        if not isinstance(overrides, dict):
            raise ValueError("Variant overrides must be a mapping.")
        variants.append(
            OrbFrequencyVariantConfig(
                name=str(raw_variant["name"]),
                label=str(raw_variant.get("label", raw_variant["name"])),
                candidate=bool(raw_variant.get("candidate", True)),
                overrides=overrides,
            )
        )

    return OrbFrequencyExpansionConfig(
        name=str(payload.get("name", "orb_frequency_expansion")),
        base_variant=str(payload.get("base_variant", "orb30_close_multi_no_slope_no_rvol_width_wider")),
        summary_thresholds=thresholds,
        variants=tuple(variants),
    )


def _read_trade_frame(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    trade_frame = pd.read_csv(path)
    if trade_frame.empty:
        return trade_frame
    for column in ("entry_timestamp", "exit_timestamp"):
        if column in trade_frame.columns:
            trade_frame[column] = pd.to_datetime(trade_frame[column], utc=True, errors="coerce")
    return trade_frame


def _build_trade_activity_metrics(*, frame: pd.DataFrame, trades_frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "trades_per_month_avg": 0.0,
            "trades_per_week_avg": 0.0,
            "pct_days_with_trade": 0.0,
            "pct_weeks_with_trade": 0.0,
            "trade_days": 0,
            "trade_weeks": 0,
            "trade_months": 0,
        }

    frame_index = pd.DatetimeIndex(frame.index)
    if frame_index.tz is None:
        frame_index = frame_index.tz_localize(UTC)
    else:
        frame_index = frame_index.tz_convert(UTC)
    frame_naive = frame_index.tz_localize(None)
    total_days = max(1, frame_index.normalize().nunique())
    total_weeks = max(1, pd.PeriodIndex(frame_naive, freq="W-SUN").nunique())
    total_months = max(1, pd.PeriodIndex(frame_naive, freq="M").nunique())

    if trades_frame.empty or "entry_timestamp" not in trades_frame.columns:
        return {
            "trades_per_month_avg": 0.0,
            "trades_per_week_avg": 0.0,
            "pct_days_with_trade": 0.0,
            "pct_weeks_with_trade": 0.0,
            "trade_days": 0,
            "trade_weeks": 0,
            "trade_months": 0,
        }

    entry_index = pd.DatetimeIndex(trades_frame["entry_timestamp"].dropna())
    if entry_index.empty:
        return {
            "trades_per_month_avg": 0.0,
            "trades_per_week_avg": 0.0,
            "pct_days_with_trade": 0.0,
            "pct_weeks_with_trade": 0.0,
            "trade_days": 0,
            "trade_weeks": 0,
            "trade_months": 0,
        }

    entry_naive = entry_index.tz_convert(UTC).tz_localize(None)
    trade_days = int(entry_index.normalize().nunique())
    trade_weeks = int(pd.PeriodIndex(entry_naive, freq="W-SUN").nunique())
    trade_months = int(pd.PeriodIndex(entry_naive, freq="M").nunique())
    trades_total = int(len(trades_frame))

    return {
        "trades_per_month_avg": trades_total / total_months,
        "trades_per_week_avg": trades_total / total_weeks,
        "pct_days_with_trade": trade_days / total_days,
        "pct_weeks_with_trade": trade_weeks / total_weeks,
        "trade_days": trade_days,
        "trade_weeks": trade_weeks,
        "trade_months": trade_months,
    }


if __name__ == "__main__":
    raise SystemExit(main())
