from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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
from .runner import BaselineRunner
from .variants import load_variant_settings, variant_summary_payload


@dataclass(slots=True)
class OrbFocusSummaryThresholds:
    minimum_trades_total: int = 24
    minimum_trades_per_year: float = 8.0
    max_drawdown: float = 0.08
    minimum_positive_year_ratio: float = 0.50
    minimum_positive_quarter_ratio: float = 0.35
    maximum_year_net_pnl_concentration: float = 0.75


@dataclass(slots=True)
class OrbFocusVariantConfig:
    name: str
    label: str
    candidate: bool
    overrides: dict[str, Any]


@dataclass(slots=True)
class OrbFocusValidationConfig:
    name: str
    base_variant: str
    summary_thresholds: OrbFocusSummaryThresholds
    variants: tuple[OrbFocusVariantConfig, ...]


@dataclass(slots=True)
class OrbFocusVariantSpec:
    name: str
    label: str
    candidate: bool
    settings: Settings


@dataclass(slots=True)
class OrbFocusVariantArtifacts:
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
class OrbFocusValidationArtifacts:
    output_dir: Path
    comparison_path: Path
    results_path: Path
    summary_path: Path
    temporal_blocks_path: Path
    variant_artifacts: dict[str, OrbFocusVariantArtifacts]


class OrbFocusValidationRunner:
    def __init__(self, config_dir: str | Path, experiment: OrbFocusValidationConfig) -> None:
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
    ) -> OrbFocusValidationArtifacts:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        frame = _filter_frame_by_range(input_frame, start=start, end=end)
        specs = self._build_variant_specs(selected_variants=selected_variants)
        if not specs:
            raise ValueError("The ORB focus validation config produced no variants to run.")

        variant_artifacts: dict[str, OrbFocusVariantArtifacts] = {}
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

            metrics = self._build_variant_metrics(
                spec=spec,
                report_payload=report_payload,
                diagnostics_payload=diagnostics_payload,
                yearly_breakdown=yearly_breakdown,
                quarterly_breakdown=quarterly_breakdown,
            )
            variant_artifacts[spec.name] = OrbFocusVariantArtifacts(
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

        comparison_path = output_path / "orb_focus_validation_comparison.json"
        results_path = output_path / "orb_focus_validation_results.csv"
        summary_path = output_path / "orb_focus_validation_summary.md"
        temporal_blocks_path = output_path / "temporal_block_results.csv"

        comparison_path.write_text(json.dumps(_sanitize_value(payload), indent=2), encoding="utf-8")
        results_frame.sort_values(["candidate", "net_pnl", "sharpe"], ascending=[False, False, False]).to_csv(
            results_path,
            index=False,
        )
        temporal_blocks.to_csv(temporal_blocks_path, index=False)
        yearly_frame.to_csv(output_path / "yearly_variant_summary.csv", index=False)
        quarterly_frame.to_csv(output_path / "quarterly_variant_summary.csv", index=False)
        yearly_curve_frame.to_csv(output_path / "yearly_equity_curve_summary.csv", index=False)
        summary_path.write_text(
            self._build_summary_markdown(
                results_frame=results_frame,
                yearly_frame=yearly_frame,
                quarterly_frame=quarterly_frame,
                conclusion=conclusion,
            ),
            encoding="utf-8",
        )

        return OrbFocusValidationArtifacts(
            output_dir=output_path,
            comparison_path=comparison_path,
            results_path=results_path,
            summary_path=summary_path,
            temporal_blocks_path=temporal_blocks_path,
            variant_artifacts=variant_artifacts,
        )

    def _build_variant_specs(self, *, selected_variants: Sequence[str] | None) -> list[OrbFocusVariantSpec]:
        specs: list[OrbFocusVariantSpec] = []
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
                OrbFocusVariantSpec(
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
        spec: OrbFocusVariantSpec,
        report_payload: dict[str, Any],
        diagnostics_payload: dict[str, Any],
        yearly_breakdown: pd.DataFrame,
        quarterly_breakdown: pd.DataFrame,
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
            "trades_per_week_avg": float(baseline_metrics["trades_per_week_avg"]),
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
            "passes_drawdown_gate": float(baseline_metrics["max_drawdown"]) <= thresholds.max_drawdown,
            "passes_temporal_gate": (
                positive_year_ratio >= thresholds.minimum_positive_year_ratio
                and positive_quarter_ratio >= thresholds.minimum_positive_quarter_ratio
                and year_net_pnl_concentration <= thresholds.maximum_year_net_pnl_concentration
            ),
        }

    def _build_conclusion(
        self,
        *,
        results_frame: pd.DataFrame,
        yearly_frame: pd.DataFrame,
        quarterly_frame: pd.DataFrame,
    ) -> dict[str, Any]:
        thresholds = self.experiment.summary_thresholds
        candidates = results_frame.loc[results_frame["candidate"]].copy()
        if candidates.empty:
            return {
                "headline": "No candidate variants were executed.",
                "best_stability_variant": None,
                "best_net_variant": None,
                "best_drawdown_variant": None,
                "still_promising": False,
                "edge_assessment": "No candidates to assess.",
                "next_small_adjustment": "No recommendation available.",
                "ppo_readiness": "Do not move to PPO.",
            }

        best_stability = candidates.sort_values(
            by=[
                "passes_min_sample",
                "passes_temporal_gate",
                "positive_year_ratio",
                "positive_quarter_ratio",
                "sharpe",
                "net_pnl",
                "max_drawdown",
            ],
            ascending=[False, False, False, False, False, False, True],
        ).iloc[0]
        best_net = candidates.sort_values(["net_pnl", "sharpe"], ascending=[False, False]).iloc[0]
        drawdown_candidates = candidates.loc[candidates["passes_min_sample"]].copy()
        if drawdown_candidates.empty:
            drawdown_candidates = candidates
        best_drawdown = drawdown_candidates.sort_values(["max_drawdown", "net_pnl"], ascending=[True, False]).iloc[0]

        still_promising = bool(
            best_stability["passes_min_sample"]
            and best_stability["passes_temporal_gate"]
            and best_stability["passes_drawdown_gate"]
            and best_stability["net_pnl"] > 0.0
            and best_stability["sharpe"] > 0.0
        )

        edge_assessment = (
            "The ORB subfamily still looks promising, but only if you judge it through the temporally-stable candidate rather than the highest-PnL candidate."
            if still_promising
            else "The ORB subfamily still looks alive enough for research, but the edge remains too concentrated or too sparse to claim robustness."
        )
        if best_stability["year_net_pnl_concentration"] > thresholds.maximum_year_net_pnl_concentration:
            edge_assessment = (
                "The ORB subfamily remains concentrated in too few yearly blocks; the best local candidate still depends too much on one period."
            )

        next_small_adjustment = (
            f"`{best_net['variant']}` is the strongest local tweak to inspect next because it keeps the highest net PnL while preserving a tradable cadence."
            if best_net["variant"] != "reference"
            else "No local tweak clearly dominates the current reference; the next step should be more temporal validation, not more parameter chasing."
        )
        if best_drawdown["variant"] != best_stability["variant"] and best_drawdown["passes_min_sample"]:
            next_small_adjustment = (
                f"Compare `{best_stability['variant']}` against `{best_drawdown['variant']}` next: one leads in stability, the other is the cleanest drawdown trade-off."
            )

        ppo_readiness = (
            "Still no PPO. Validate this subfamily more deeply first."
            if not still_promising
            else "Still hold PPO for now; one more temporal validation pass is safer before any RL layer."
        )

        return {
            "headline": (
                f"Focused validation around `{self.experiment.base_variant}` points to "
                f"`{best_stability['variant']}` as the most stable local candidate and "
                f"`{best_net['variant']}` as the highest-net local candidate."
            ),
            "best_stability_variant": _sanitize_value(best_stability.to_dict()),
            "best_net_variant": _sanitize_value(best_net.to_dict()),
            "best_drawdown_variant": _sanitize_value(best_drawdown.to_dict()),
            "still_promising": still_promising,
            "edge_assessment": edge_assessment,
            "next_small_adjustment": next_small_adjustment,
            "ppo_readiness": ppo_readiness,
            "sample_warning": (
                f"The best stability candidate still runs only `{best_stability['trades_per_year']:.2f}` trades/year."
                if float(best_stability["trades_per_year"]) < thresholds.minimum_trades_per_year
                else None
            ),
            "yearly_reference": _reference_temporal_snapshot(yearly_frame, variant="reference", block_column="exit_year"),
            "quarterly_reference": _reference_temporal_snapshot(
                quarterly_frame,
                variant="reference",
                block_column="exit_quarter",
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
        best_stability = conclusion["best_stability_variant"]
        best_net = conclusion["best_net_variant"]
        best_drawdown = conclusion["best_drawdown_variant"]
        ordered = results_frame.sort_values(["candidate", "net_pnl", "sharpe"], ascending=[False, False, False])
        lines = [
            "# ORB Focus Validation Summary",
            "",
            "## Direct Answer",
            conclusion["headline"],
            "",
            f"- Still promising: `{conclusion['still_promising']}`",
            f"- Edge assessment: {conclusion['edge_assessment']}",
            f"- PPO readiness: {conclusion['ppo_readiness']}",
        ]
        if conclusion.get("sample_warning"):
            lines.append(f"- Sample warning: {conclusion['sample_warning']}")

        lines.extend(
            [
                "",
                "## Best Candidates",
                (
                    f"- Best stability: `{best_stability['variant']}` | net `{best_stability['net_pnl']:.2f}` | "
                    f"DD `{best_stability['max_drawdown']:.4f}` | trades/year `{best_stability['trades_per_year']:.2f}` | "
                    f"positive years `{best_stability['positive_year_ratio'] * 100:.1f}%`."
                ),
                (
                    f"- Best net: `{best_net['variant']}` | net `{best_net['net_pnl']:.2f}` | "
                    f"Sharpe `{best_net['sharpe']:.4f}` | year concentration `{best_net['year_net_pnl_concentration']:.2f}`."
                ),
                (
                    f"- Lowest drawdown with usable sample: `{best_drawdown['variant']}` | DD `{best_drawdown['max_drawdown']:.4f}` | "
                    f"net `{best_drawdown['net_pnl']:.2f}`."
                ),
                "",
                "## Variants",
            ]
        )
        for _, row in ordered.iterrows():
            lines.append(
                f"- `{row['variant']}`: net `{row['net_pnl']:.2f}`, gross `{row['gross_pnl']:.2f}`, "
                f"DD `{row['max_drawdown']:.4f}`, Sharpe `{row['sharpe']:.4f}`, trades/year `{row['trades_per_year']:.2f}`, "
                f"positive years `{row['positive_year_ratio'] * 100:.1f}%`, positive quarters `{row['positive_quarter_ratio'] * 100:.1f}%`."
            )

        lines.extend(["", "## Reference Temporal Breakdown"])
        for item in conclusion["yearly_reference"]:
            lines.append(
                f"- Year `{item['block']}`: trades `{item['trades']}`, net `{item['net_pnl']:.2f}`, "
                f"PF `{item['profit_factor']:.2f}`, DD `{item.get('year_drawdown', float('nan')):.4f}`."
            )
        for item in conclusion["quarterly_reference"][:8]:
            lines.append(
                f"- Quarter `{item['block']}`: trades `{item['trades']}`, net `{item['net_pnl']:.2f}`, "
                f"PF `{item['profit_factor']:.2f}`."
            )

        lines.extend(
            [
                "",
                "## How To Read This",
                "- `orb_focus_validation_results.csv` is the clean comparison table by local variant.",
                "- `temporal_block_results.csv` merges yearly and quarterly blocks for all variants.",
                "- Each variant keeps its own `baseline/` and `diagnostics/` artifacts under `variants/`.",
                "",
                "## Next Small Step",
                f"- {conclusion['next_small_adjustment']}",
            ]
        )
        return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run focused ORB sensitivity around the winning subfamily.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--experiment-config", default="configs/experiments/orb_focus_validation.yaml")
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

    experiment = load_orb_focus_validation_config(args.experiment_config)
    runner = OrbFocusValidationRunner(args.config_dir, experiment)
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
    best = payload["conclusion"]["best_stability_variant"]
    print(f"ORB focus validation comparison: {artifacts.comparison_path}")
    print(f"ORB focus validation summary: {artifacts.summary_path}")
    print(
        " ".join(
            [
                f"variants={len(payload['variants'])}",
                f"best_stability={best['variant'] if best else 'n/a'}",
                f"best_net={payload['conclusion']['best_net_variant']['variant'] if payload['conclusion']['best_net_variant'] else 'n/a'}",
            ]
        )
    )
    return 0


def load_orb_focus_validation_config(path: str | Path) -> OrbFocusValidationConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"ORB focus validation config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"ORB focus validation config must be a mapping: {config_path}")

    thresholds_payload = payload.get("summary_thresholds", {}) or {}
    thresholds = OrbFocusSummaryThresholds(
        minimum_trades_total=int(thresholds_payload.get("minimum_trades_total", 24)),
        minimum_trades_per_year=float(thresholds_payload.get("minimum_trades_per_year", 8.0)),
        max_drawdown=float(thresholds_payload.get("max_drawdown", 0.08)),
        minimum_positive_year_ratio=float(thresholds_payload.get("minimum_positive_year_ratio", 0.50)),
        minimum_positive_quarter_ratio=float(thresholds_payload.get("minimum_positive_quarter_ratio", 0.35)),
        maximum_year_net_pnl_concentration=float(
            thresholds_payload.get("maximum_year_net_pnl_concentration", 0.75)
        ),
    )

    raw_variants = payload.get("variants", [])
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("ORB focus validation config requires a non-empty 'variants' list.")

    variants: list[OrbFocusVariantConfig] = []
    for raw_variant in raw_variants:
        if not isinstance(raw_variant, dict):
            raise ValueError("Each ORB focus validation variant must be a mapping.")
        overrides = raw_variant.get("overrides", {}) or {}
        if not isinstance(overrides, dict):
            raise ValueError("Variant overrides must be a mapping.")
        variants.append(
            OrbFocusVariantConfig(
                name=str(raw_variant["name"]),
                label=str(raw_variant.get("label", raw_variant["name"])),
                candidate=bool(raw_variant.get("candidate", True)),
                overrides=overrides,
            )
        )

    return OrbFocusValidationConfig(
        name=str(payload.get("name", "orb_focus_validation")),
        base_variant=str(payload.get("base_variant", "orb30_close_multi_no_slope_no_rvol")),
        summary_thresholds=thresholds,
        variants=tuple(variants),
    )


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


def _tag_temporal_frame(frame: pd.DataFrame, *, block_type: str, block_column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["variant", "label", "candidate", "block_type", "block"])
    tagged = frame.copy()
    tagged.insert(3, "block_type", block_type)
    tagged.insert(4, "block", tagged[block_column])
    return tagged


def _net_concentration(frame: pd.DataFrame) -> float:
    if frame.empty or "net_pnl" not in frame.columns:
        return 0.0
    abs_net = frame["net_pnl"].abs()
    total = float(abs_net.sum())
    if total <= 0.0:
        return 0.0
    return float(abs_net.max() / total)


def _reference_temporal_snapshot(
    frame: pd.DataFrame,
    *,
    variant: str,
    block_column: str,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    subset = frame.loc[frame["variant"] == variant].copy()
    if subset.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in subset.sort_values(block_column).iterrows():
        item = {
            "block": row[block_column],
            "trades": int(row["trades"]),
            "net_pnl": float(row["net_pnl"]),
            "profit_factor": float(row["profit_factor"]),
        }
        if "year_drawdown" in row.index and pd.notna(row["year_drawdown"]):
            item["year_drawdown"] = float(row["year_drawdown"])
        rows.append(item)
    return rows


def _parse_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _filter_frame_by_range(
    frame: pd.DataFrame,
    *,
    start: datetime | None,
    end: datetime | None,
) -> pd.DataFrame:
    filtered = frame
    if start is not None:
        filtered = filtered.loc[filtered.index >= pd.Timestamp(start)]
    if end is not None:
        filtered = filtered.loc[filtered.index <= pd.Timestamp(end)]
    if filtered.empty:
        raise ValueError("The requested ORB focus validation range produced an empty OHLCV frame.")
    return filtered


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
