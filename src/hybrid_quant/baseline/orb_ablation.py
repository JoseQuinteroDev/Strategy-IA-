from __future__ import annotations

import argparse
import itertools
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
from .runner import BaselineRunner
from .variants import load_variant_settings, variant_summary_payload


@dataclass(slots=True)
class AblationOption:
    key: str
    label: str
    value: Any
    overrides: dict[str, Any]


@dataclass(slots=True)
class AblationDimension:
    key: str
    label: str
    options: tuple[AblationOption, ...]


@dataclass(slots=True)
class OrbAblationConfig:
    name: str
    base_variant: str
    dimensions: tuple[AblationDimension, ...]


@dataclass(slots=True)
class OrbAblationVariantSpec:
    name: str
    label: str
    factors: dict[str, Any]
    settings: Settings


@dataclass(slots=True)
class OrbAblationVariantArtifacts:
    variant_name: str
    label: str
    factors: dict[str, Any]
    artifact_dir: Path
    diagnostics_dir: Path
    report_path: Path
    diagnostics_path: Path
    summary_path: Path
    metrics: dict[str, Any]


@dataclass(slots=True)
class OrbAblationArtifacts:
    output_dir: Path
    comparison_path: Path
    results_path: Path
    summary_path: Path
    variant_artifacts: dict[str, OrbAblationVariantArtifacts]


class OrbAblationRunner:
    def __init__(self, config_dir: str | Path, experiment: OrbAblationConfig) -> None:
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
    ) -> OrbAblationArtifacts:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        frame = _filter_frame_by_range(input_frame, start=start, end=end)
        variant_specs = self._build_variant_specs(selected_variants=selected_variants)
        if not variant_specs:
            raise ValueError("The ORB ablation matrix produced no variants to run.")

        variant_artifacts: dict[str, OrbAblationVariantArtifacts] = {}
        for spec in variant_specs:
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
                include_risk_replay=True,
            )
            report_payload = json.loads(baseline_artifacts.report_path.read_text(encoding="utf-8"))
            diagnostics_payload = json.loads(diagnostics_artifacts.diagnostics_path.read_text(encoding="utf-8"))
            metrics = self._build_variant_metrics(
                spec=spec,
                report_payload=report_payload,
                diagnostics_payload=diagnostics_payload,
            )
            variant_artifacts[spec.name] = OrbAblationVariantArtifacts(
                variant_name=spec.name,
                label=spec.label,
                factors=spec.factors,
                artifact_dir=artifact_dir,
                diagnostics_dir=diagnostics_dir,
                report_path=baseline_artifacts.report_path,
                diagnostics_path=diagnostics_artifacts.diagnostics_path,
                summary_path=diagnostics_artifacts.summary_path,
                metrics=metrics,
            )

        results_frame = pd.DataFrame([artifact.metrics for artifact in variant_artifacts.values()])
        factor_summaries = {
            "opening_range": _build_factor_summary(results_frame, "opening_range_minutes"),
            "entry_mode": _build_factor_summary(results_frame, "entry_mode"),
            "breakout_budget": _build_factor_summary(results_frame, "daily_breakout_policy"),
            "ema_slope": _build_factor_summary(results_frame, "use_ema_200_1h_slope"),
            "relative_volume": _build_factor_summary(results_frame, "use_relative_volume_filter"),
        }
        conclusion = _build_ablation_conclusion(results_frame, factor_summaries)
        payload = {
            "experiment_name": self.experiment.name,
            "base_variant": self.experiment.base_variant,
            "input_period": {
                "start": frame.index[0].isoformat() if not frame.empty else None,
                "end": frame.index[-1].isoformat() if not frame.empty else None,
                "bars": int(len(frame)),
            },
            "variants": {name: _sanitize_value(artifact.metrics) for name, artifact in variant_artifacts.items()},
            "factor_summaries": {
                key: summary.to_dict(orient="records")
                for key, summary in factor_summaries.items()
            },
            "conclusion": conclusion,
            "variant_order": list(variant_artifacts),
        }

        comparison_path = output_path / "orb_ablation_comparison.json"
        results_path = output_path / "orb_ablation_results.csv"
        summary_path = output_path / "orb_ablation_summary.md"
        comparison_path.write_text(json.dumps(_sanitize_value(payload), indent=2), encoding="utf-8")
        results_frame.sort_values("net_pnl", ascending=False).to_csv(results_path, index=False)
        summary_path.write_text(
            _build_summary_markdown(payload=payload, results_frame=results_frame, factor_summaries=factor_summaries),
            encoding="utf-8",
        )

        for name, summary in factor_summaries.items():
            summary.to_csv(output_path / f"{name}_summary.csv", index=False)

        return OrbAblationArtifacts(
            output_dir=output_path,
            comparison_path=comparison_path,
            results_path=results_path,
            summary_path=summary_path,
            variant_artifacts=variant_artifacts,
        )

    def _build_variant_specs(self, *, selected_variants: Sequence[str] | None) -> list[OrbAblationVariantSpec]:
        specs: list[OrbAblationVariantSpec] = []
        for option_tuple in itertools.product(*(dimension.options for dimension in self.experiment.dimensions)):
            settings = self.base_settings
            factors: dict[str, Any] = {}
            name_parts: list[str] = []
            labels: list[str] = []
            for dimension, option in zip(self.experiment.dimensions, option_tuple, strict=True):
                settings = apply_settings_overrides(settings, option.overrides)
                factors[dimension.key] = option.value
                name_parts.append(option.key)
                labels.append(option.label)

            variant_name = "_".join(name_parts)
            settings = apply_settings_overrides(
                settings,
                {
                    "strategy": {
                        "variant_name": variant_name,
                    }
                },
            )
            spec = OrbAblationVariantSpec(
                name=variant_name,
                label=" | ".join(labels),
                factors=factors,
                settings=settings,
            )
            if selected_variants and spec.name not in selected_variants:
                continue
            specs.append(spec)
        return specs

    def _build_variant_metrics(
        self,
        *,
        spec: OrbAblationVariantSpec,
        report_payload: dict[str, Any],
        diagnostics_payload: dict[str, Any],
    ) -> dict[str, Any]:
        baseline_metrics = diagnostics_payload["baseline_metrics"]
        factor_values = {
            "opening_range_minutes": spec.factors.get(
                "opening_range_minutes",
                spec.settings.strategy.opening_range_minutes,
            ),
            "entry_mode": spec.factors.get(
                "entry_mode",
                spec.settings.strategy.entry_mode,
            ),
            "daily_breakout_policy": spec.factors.get(
                "daily_breakout_policy",
                "first_breakout_only"
                if spec.settings.strategy.max_breakouts_per_day == 1
                else "multiple_breakouts",
            ),
            "use_ema_200_1h_slope": spec.factors.get(
                "use_ema_200_1h_slope",
                spec.settings.strategy.use_ema_200_1h_slope,
            ),
            "use_relative_volume_filter": spec.factors.get(
                "use_relative_volume_filter",
                spec.settings.strategy.minimum_relative_volume > 0.0,
            ),
        }
        return {
            "variant": spec.name,
            "label": spec.label,
            "symbol": report_payload["symbol"],
            "execution_timeframe": report_payload["execution_timeframe"],
            "filter_timeframe": report_payload["filter_timeframe"],
            "strategy": variant_summary_payload(spec.settings),
            "opening_range_minutes": int(factor_values["opening_range_minutes"]),
            "entry_mode": str(factor_values["entry_mode"]),
            "daily_breakout_policy": str(factor_values["daily_breakout_policy"]),
            "use_ema_200_1h_slope": bool(factor_values["use_ema_200_1h_slope"]),
            "use_relative_volume_filter": bool(factor_values["use_relative_volume_filter"]),
            "number_of_trades": int(baseline_metrics["number_of_trades"]),
            "trades_per_year": float(baseline_metrics["trades_per_year"]),
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
            "profitable_years_pct": float(baseline_metrics.get("profitable_years_pct", 0.0)),
            "profitable_months_pct": float(baseline_metrics["profitable_months_pct"]),
            "max_consecutive_losses": int(baseline_metrics["max_consecutive_losses"]),
            "average_holding_bars": float(baseline_metrics["average_holding_bars"]),
            "average_mfe_atr": float(baseline_metrics["average_mfe_atr"]),
            "average_mae_atr": float(baseline_metrics["average_mae_atr"]),
            "estimated_fee_drag": float(baseline_metrics["estimated_fee_drag"] or 0.0),
            "estimated_slippage_drag": float(baseline_metrics["estimated_slippage_drag"] or 0.0),
            "estimated_total_cost_drag": float(baseline_metrics["estimated_total_cost_drag"] or 0.0),
            "validation": report_payload["validation"],
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Nasdaq ORB ablation matrix.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--experiment-config", default="configs/experiments/orb_ablation.yaml")
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

    experiment = load_orb_ablation_config(args.experiment_config)
    runner = OrbAblationRunner(args.config_dir, experiment)
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
    best = payload["conclusion"]["best_robustness_variant"]
    print(f"ORB ablation comparison: {artifacts.comparison_path}")
    print(f"ORB ablation summary: {artifacts.summary_path}")
    print(
        " ".join(
            [
                f"variants={len(payload['variants'])}",
                f"best={best['variant'] if best else 'n/a'}",
                f"best_net_pnl={best['net_pnl'] if best else 'n/a'}",
            ]
        )
    )
    return 0


def load_orb_ablation_config(path: str | Path) -> OrbAblationConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"ORB ablation config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"ORB ablation config must be a mapping: {config_path}")

    name = str(payload.get("name", "orb_ablation"))
    base_variant = str(payload.get("base_variant", "baseline_nq_orb"))
    raw_dimensions = payload.get("dimensions", [])
    if not isinstance(raw_dimensions, list) or not raw_dimensions:
        raise ValueError("ORB ablation config requires a non-empty 'dimensions' list.")

    dimensions: list[AblationDimension] = []
    for raw_dimension in raw_dimensions:
        if not isinstance(raw_dimension, dict):
            raise ValueError("Each ORB ablation dimension must be a mapping.")
        key = str(raw_dimension["key"])
        label = str(raw_dimension.get("label", key))
        raw_options = raw_dimension.get("options", [])
        if not isinstance(raw_options, list) or not raw_options:
            raise ValueError(f"ORB ablation dimension '{key}' requires at least one option.")

        options: list[AblationOption] = []
        for raw_option in raw_options:
            if not isinstance(raw_option, dict):
                raise ValueError(f"ORB ablation option in '{key}' must be a mapping.")
            option_key = str(raw_option["key"])
            option_label = str(raw_option.get("label", option_key))
            option_value = raw_option.get("value", option_key)
            overrides = raw_option.get("overrides", {}) or {}
            if not isinstance(overrides, dict):
                raise ValueError(f"Overrides for ORB ablation option '{option_key}' must be a mapping.")
            options.append(
                AblationOption(
                    key=option_key,
                    label=option_label,
                    value=option_value,
                    overrides=overrides,
                )
            )
        dimensions.append(AblationDimension(key=key, label=label, options=tuple(options)))

    return OrbAblationConfig(name=name, base_variant=base_variant, dimensions=tuple(dimensions))


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


def _build_factor_summary(results_frame: pd.DataFrame, factor: str) -> pd.DataFrame:
    if results_frame.empty or factor not in results_frame.columns:
        return pd.DataFrame(columns=[factor])

    rows: list[dict[str, Any]] = []
    grouped = results_frame.groupby(factor, dropna=False, observed=False)
    for value, frame in grouped:
        rows.append(
            {
                factor: value,
                "variants": int(len(frame)),
                "avg_net_pnl": float(frame["net_pnl"].mean()),
                "median_net_pnl": float(frame["net_pnl"].median()),
                "best_net_pnl": float(frame["net_pnl"].max()),
                "avg_gross_pnl": float(frame["gross_pnl"].mean()),
                "avg_max_drawdown": float(frame["max_drawdown"].mean()),
                "median_max_drawdown": float(frame["max_drawdown"].median()),
                "avg_sharpe": float(frame["sharpe"].mean()),
                "avg_sortino": float(frame["sortino"].mean()),
                "avg_calmar": float(frame["calmar"].mean()),
                "avg_profit_factor": float(frame["profit_factor"].mean()),
                "avg_expectancy": float(frame["expectancy"].mean()),
                "avg_trades_per_year": float(frame["trades_per_year"].mean()),
                "avg_trades_per_week": float(frame["trades_per_week_avg"].mean()),
                "positive_variant_ratio": float((frame["net_pnl"] > 0.0).mean()),
                "avg_profitable_years_pct": float(frame["profitable_years_pct"].mean()),
                "avg_profitable_months_pct": float(frame["profitable_months_pct"].mean()),
            }
        )

    summary = pd.DataFrame(rows)
    if factor == "opening_range_minutes":
        summary = summary.sort_values(factor)
    return summary.reset_index(drop=True)


def _build_ablation_conclusion(
    results_frame: pd.DataFrame,
    factor_summaries: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    if results_frame.empty:
        return {
            "headline": "No ORB ablation variants were executed.",
            "best_robustness_variant": None,
            "lowest_drawdown_variant": None,
            "lowest_overtrading_variant": None,
            "entry_mode_verdict": "No verdict.",
            "opening_range_verdict": "No verdict.",
        }

    best_robustness = results_frame.sort_values(
        by=["profitable_years_pct", "sharpe", "net_pnl", "profit_factor", "max_drawdown"],
        ascending=[False, False, False, False, True],
    ).iloc[0]
    lowest_drawdown = results_frame.sort_values(
        by=["max_drawdown", "net_pnl"],
        ascending=[True, False],
    ).iloc[0]
    lowest_overtrading = results_frame.sort_values(
        by=["trades_per_year", "net_pnl"],
        ascending=[True, False],
    ).iloc[0]

    entry_mode_summary = factor_summaries.get("entry_mode", pd.DataFrame())
    opening_range_summary = factor_summaries.get("opening_range", pd.DataFrame())

    entry_mode_verdict = _compare_factor_values(
        entry_mode_summary,
        factor="entry_mode",
        left_value="breakout_close_entry",
        right_value="breakout_retest_entry",
        left_label="entry on close",
        right_label="retest",
    )
    opening_range_verdict = _compare_factor_values(
        opening_range_summary,
        factor="opening_range_minutes",
        left_value=15,
        right_value=30,
        left_label="OR 15m",
        right_label="OR 30m",
    )

    return {
        "headline": (
            f"The strongest robustness candidate on this ORB ablation matrix is `{best_robustness['variant']}` "
            f"with net PnL `{best_robustness['net_pnl']:.2f}`, Sharpe `{best_robustness['sharpe']:.4f}`, "
            f"drawdown `{best_robustness['max_drawdown']:.4f}` and `{best_robustness['trades_per_year']:.2f}` trades/year."
        ),
        "best_robustness_variant": _sanitize_value(best_robustness.to_dict()),
        "lowest_drawdown_variant": _sanitize_value(lowest_drawdown.to_dict()),
        "lowest_overtrading_variant": _sanitize_value(lowest_overtrading.to_dict()),
        "entry_mode_verdict": entry_mode_verdict,
        "opening_range_verdict": opening_range_verdict,
    }


def _compare_factor_values(
    summary: pd.DataFrame,
    *,
    factor: str,
    left_value: Any,
    right_value: Any,
    left_label: str,
    right_label: str,
) -> str:
    if summary.empty or factor not in summary.columns:
        return f"There is not enough data to compare {left_label} vs {right_label}."

    left = summary.loc[summary[factor] == left_value]
    right = summary.loc[summary[factor] == right_value]
    if left.empty or right.empty:
        return f"There is not enough data to compare {left_label} vs {right_label}."

    left_row = left.iloc[0]
    right_row = right.iloc[0]
    left_score = _factor_score(left_row)
    right_score = _factor_score(right_row)

    if left_score > right_score:
        winner, loser = left_row, right_row
        winner_label, loser_label = left_label, right_label
    elif right_score > left_score:
        winner, loser = right_row, left_row
        winner_label, loser_label = right_label, left_label
    else:
        return (
            f"{left_label} and {right_label} look mixed on this matrix: neither dominates clearly across "
            "net PnL, drawdown and Sharpe."
        )

    return (
        f"{winner_label} looks more stable than {loser_label}: average net PnL "
        f"`{winner['avg_net_pnl']:.2f}` vs `{loser['avg_net_pnl']:.2f}`, average drawdown "
        f"`{winner['avg_max_drawdown']:.4f}` vs `{loser['avg_max_drawdown']:.4f}`, and average Sharpe "
        f"`{winner['avg_sharpe']:.4f}` vs `{loser['avg_sharpe']:.4f}`."
    )


def _factor_score(row: pd.Series) -> tuple[float, float, float, float]:
    return (
        float(row["positive_variant_ratio"]),
        float(row["avg_sharpe"]),
        float(row["avg_net_pnl"]),
        -float(row["avg_max_drawdown"]),
    )


def _build_summary_markdown(
    *,
    payload: dict[str, Any],
    results_frame: pd.DataFrame,
    factor_summaries: dict[str, pd.DataFrame],
) -> str:
    conclusion = payload["conclusion"]
    top_variants = (
        results_frame.sort_values(["net_pnl", "sharpe"], ascending=[False, False]).head(5)
        if not results_frame.empty
        else pd.DataFrame()
    )
    lines = [
        "# ORB Ablation Summary",
        "",
        "## Setup",
        f"- Base variant: `{payload['base_variant']}`",
        f"- Period: `{payload['input_period']['start']}` -> `{payload['input_period']['end']}`",
        f"- Bars: `{payload['input_period']['bars']}`",
        f"- Variants executed: `{len(payload['variants'])}`",
        "",
        "## Direct Answer",
        conclusion["headline"],
        "",
        "## Best Robustness Variant",
        (
            f"- Variant: `{conclusion['best_robustness_variant']['variant']}`"
            if conclusion["best_robustness_variant"]
            else "- Variant: `n/a`"
        ),
        (
            f"- Net PnL: `{conclusion['best_robustness_variant']['net_pnl']:.2f}`"
            if conclusion["best_robustness_variant"]
            else "- Net PnL: `n/a`"
        ),
        (
            f"- Max drawdown: `{conclusion['best_robustness_variant']['max_drawdown']:.4f}`"
            if conclusion["best_robustness_variant"]
            else "- Max drawdown: `n/a`"
        ),
        (
            f"- Trades/year: `{conclusion['best_robustness_variant']['trades_per_year']:.2f}`"
            if conclusion["best_robustness_variant"]
            else "- Trades/year: `n/a`"
        ),
        "",
        "## Factor Verdicts",
        f"- Entry mode: {conclusion['entry_mode_verdict']}",
        f"- Opening range: {conclusion['opening_range_verdict']}",
        (
            f"- Lowest drawdown variant: `{conclusion['lowest_drawdown_variant']['variant']}` "
            f"at `{conclusion['lowest_drawdown_variant']['max_drawdown']:.4f}`."
            if conclusion["lowest_drawdown_variant"]
            else "- Lowest drawdown variant: `n/a`."
        ),
        (
            f"- Strongest overtrading reduction: `{conclusion['lowest_overtrading_variant']['variant']}` "
            f"with `{conclusion['lowest_overtrading_variant']['trades_per_year']:.2f}` trades/year."
            if conclusion["lowest_overtrading_variant"]
            else "- Strongest overtrading reduction: `n/a`."
        ),
        "",
        "## Top Variants",
    ]
    for _, row in top_variants.iterrows():
        lines.append(
            f"- `{row['variant']}`: net `{row['net_pnl']:.2f}`, gross `{row['gross_pnl']:.2f}`, "
            f"DD `{row['max_drawdown']:.4f}`, Sharpe `{row['sharpe']:.4f}`, trades/year `{row['trades_per_year']:.2f}`."
        )

    lines.extend(["", "## Factor Summaries"])
    for factor_name, summary in factor_summaries.items():
        if summary.empty:
            continue
        lines.append(f"- `{factor_name}` summary exported to `{factor_name}_summary.csv`.")

    lines.extend(
        [
            "",
            "## How To Read This",
            "- `orb_ablation_results.csv` ranks every variant with its exact factor combination.",
            "- Each variant has its own `baseline/` and `diagnostics/` folder under `variants/`.",
            "- Prefer variants that improve net PnL and Sharpe without collapsing trade frequency or inflating drawdown.",
            "",
            "## Pitfalls",
            "- This matrix is still a multiple-comparison exercise; a single winning variant is not enough by itself.",
            "- Lower trade count can look cleaner simply because the strategy trades less, not because it is more robust.",
            "- The retest mode can reduce entries mechanically; compare it against drawdown and trades/year, not net PnL alone.",
        ]
    )
    return "\n".join(lines) + "\n"


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
        raise ValueError("The requested ORB ablation range produced an empty OHLCV frame.")
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
