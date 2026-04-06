from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from hybrid_quant.data import read_ohlcv_frame

from .orb_focus_validation import _parse_datetime, _sanitize_value
from .orb_frequency_expansion import (
    OrbFrequencyExpansionConfig,
    OrbFrequencyExpansionRunner,
    OrbFrequencySummaryThresholds,
    OrbFrequencyVariantConfig,
)


@dataclass(slots=True)
class OrbFrequencyPushThresholds:
    minimum_trades_total: int = 45
    minimum_trades_per_year: float = 15.0
    target_trades_per_week_avg: float = 1.0
    ambitious_trades_per_week_avg: float = 2.0
    minimum_profit_factor: float = 1.10
    minimum_expectancy_ratio_vs_reference: float = 0.70
    minimum_profit_factor_ratio_vs_reference: float = 0.85
    maximum_drawdown_multiple_vs_reference: float = 2.50
    minimum_positive_year_ratio: float = 0.50
    minimum_positive_quarter_ratio: float = 0.35
    maximum_year_net_pnl_concentration: float = 0.80


@dataclass(slots=True)
class OrbFrequencyPushVariantConfig:
    name: str
    label: str
    candidate: bool
    overrides: dict[str, Any]


@dataclass(slots=True)
class OrbFrequencyPushConfig:
    name: str
    base_variant: str
    summary_thresholds: OrbFrequencyPushThresholds
    variants: tuple[OrbFrequencyPushVariantConfig, ...]


@dataclass(slots=True)
class OrbFrequencyPushArtifacts:
    output_dir: Path
    comparison_path: Path
    results_path: Path
    summary_path: Path
    activity_summary_path: Path
    candidate_ranking_path: Path


class OrbFrequencyPushRunner:
    def __init__(self, config_dir: str | Path, experiment: OrbFrequencyPushConfig) -> None:
        self.config_dir = Path(config_dir)
        self.experiment = experiment

    def run(
        self,
        *,
        input_frame: pd.DataFrame,
        output_dir: str | Path,
        allow_gaps: bool = False,
        start: datetime | None = None,
        end: datetime | None = None,
        selected_variants: Sequence[str] | None = None,
    ) -> OrbFrequencyPushArtifacts:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        expansion_results_path = output_path / "orb_frequency_expansion_results.csv"
        expansion_comparison_path = output_path / "orb_frequency_expansion_comparison.json"
        if (
            selected_variants
            or not expansion_results_path.exists()
            or not expansion_comparison_path.exists()
            or not (output_path / "yearly_variant_summary.csv").exists()
            or not (output_path / "quarterly_variant_summary.csv").exists()
        ):
            expansion_runner = OrbFrequencyExpansionRunner(
                self.config_dir,
                self._build_compatible_expansion_config(),
            )
            expansion_runner.run(
                input_frame=input_frame,
                output_dir=output_path,
                allow_gaps=allow_gaps,
                start=start,
                end=end,
                selected_variants=selected_variants,
            )

        results_frame = pd.read_csv(expansion_results_path)
        yearly_frame = _read_csv_or_empty(output_path / "yearly_variant_summary.csv")
        quarterly_frame = _read_csv_or_empty(output_path / "quarterly_variant_summary.csv")
        enriched = self._enrich_results_with_push_metrics(results_frame, output_path)
        candidate_ranking = self._build_candidate_ranking(enriched)
        conclusion = self._build_conclusion(enriched)

        payload = {
            "experiment_name": self.experiment.name,
            "base_variant": self.experiment.base_variant,
            "summary_thresholds": asdict(self.experiment.summary_thresholds),
            "input_period": json.loads(expansion_comparison_path.read_text(encoding="utf-8"))["input_period"],
            "variants": {
                row["variant"]: _sanitize_value(row.drop(labels=["strategy"]).to_dict())
                for _, row in enriched.iterrows()
            },
            "conclusion": conclusion,
        }

        comparison_path = output_path / "orb_frequency_push_comparison.json"
        results_path = output_path / "orb_frequency_push_results.csv"
        summary_path = output_path / "orb_frequency_push_summary.md"
        activity_summary_path = output_path / "activity_summary.csv"
        candidate_ranking_path = output_path / "candidate_ranking.csv"

        comparison_path.write_text(json.dumps(_sanitize_value(payload), indent=2), encoding="utf-8")
        enriched.sort_values(
            ["passes_push_guard", "trades_per_week_avg", "net_pnl"],
            ascending=[False, False, False],
        ).to_csv(results_path, index=False)
        enriched[
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
                "max_inactive_days",
                "max_inactive_weeks",
                "frequency_uplift_ratio",
                "expectancy",
                "profit_factor",
                "payoff",
                "net_pnl",
                "gross_pnl",
                "max_drawdown",
                "sharpe",
                "sortino",
                "calmar",
                "reaches_one_trade_per_week",
                "reaches_two_trades_per_week",
                "passes_absolute_quality",
                "passes_relative_quality",
                "passes_temporal_gate",
                "passes_push_guard",
            ]
        ].sort_values(["passes_push_guard", "trades_per_week_avg"], ascending=[False, False]).to_csv(
            activity_summary_path,
            index=False,
        )
        candidate_ranking.to_csv(candidate_ranking_path, index=False)
        summary_path.write_text(
            self._build_summary_markdown(
                results_frame=enriched,
                yearly_frame=yearly_frame,
                quarterly_frame=quarterly_frame,
                conclusion=conclusion,
            ),
            encoding="utf-8",
        )

        return OrbFrequencyPushArtifacts(
            output_dir=output_path,
            comparison_path=comparison_path,
            results_path=results_path,
            summary_path=summary_path,
            activity_summary_path=activity_summary_path,
            candidate_ranking_path=candidate_ranking_path,
        )

    def _build_compatible_expansion_config(self) -> OrbFrequencyExpansionConfig:
        thresholds = self.experiment.summary_thresholds
        compatible_thresholds = OrbFrequencySummaryThresholds(
            minimum_trades_total=thresholds.minimum_trades_total,
            minimum_trades_per_year=thresholds.minimum_trades_per_year,
            target_trades_per_year=thresholds.target_trades_per_week_avg * 52.0,
            minimum_frequency_uplift_ratio=1.0,
            minimum_expectancy_ratio_vs_reference=thresholds.minimum_expectancy_ratio_vs_reference,
            minimum_profit_factor_ratio_vs_reference=thresholds.minimum_profit_factor_ratio_vs_reference,
            maximum_drawdown_multiple_vs_reference=thresholds.maximum_drawdown_multiple_vs_reference,
            minimum_positive_year_ratio=thresholds.minimum_positive_year_ratio,
            minimum_positive_quarter_ratio=thresholds.minimum_positive_quarter_ratio,
            maximum_year_net_pnl_concentration=thresholds.maximum_year_net_pnl_concentration,
        )
        variants = tuple(
            OrbFrequencyVariantConfig(
                name=variant.name,
                label=variant.label,
                candidate=variant.candidate,
                overrides=variant.overrides,
            )
            for variant in self.experiment.variants
        )
        return OrbFrequencyExpansionConfig(
            name=self.experiment.name,
            base_variant=self.experiment.base_variant,
            summary_thresholds=compatible_thresholds,
            variants=variants,
        )

    def _enrich_results_with_push_metrics(self, results_frame: pd.DataFrame, output_path: Path) -> pd.DataFrame:
        enriched = results_frame.copy()
        if enriched.empty:
            return enriched

        reference_row = enriched.loc[enriched["variant"] == "reference"].iloc[0]
        thresholds = self.experiment.summary_thresholds
        reference_expectancy = float(reference_row["expectancy"])
        reference_profit_factor = max(float(reference_row["profit_factor"]), 1e-9)
        reference_drawdown = max(float(reference_row["max_drawdown"]), 1e-9)

        inactivity_days: list[int] = []
        inactivity_weeks: list[int] = []
        for _, row in enriched.iterrows():
            trade_frame = _read_trade_frame(output_path / "variants" / str(row["variant"]) / "baseline" / "trades.csv")
            streaks = _build_inactivity_metrics(trade_frame)
            inactivity_days.append(streaks["max_inactive_days"])
            inactivity_weeks.append(streaks["max_inactive_weeks"])

        enriched["max_inactive_days"] = inactivity_days
        enriched["max_inactive_weeks"] = inactivity_weeks
        enriched["reaches_one_trade_per_week"] = enriched["trades_per_week_avg"] >= thresholds.target_trades_per_week_avg
        enriched["reaches_two_trades_per_week"] = (
            enriched["trades_per_week_avg"] >= thresholds.ambitious_trades_per_week_avg
        )
        enriched["expectancy_ratio_vs_reference"] = enriched["expectancy"].apply(
            lambda value: float(value) / reference_expectancy if abs(reference_expectancy) > 1e-9 else 0.0
        )
        enriched["profit_factor_ratio_vs_reference"] = enriched["profit_factor"] / reference_profit_factor
        enriched["drawdown_multiple_vs_reference"] = enriched["max_drawdown"] / reference_drawdown
        enriched["passes_absolute_quality"] = (
            (enriched["profit_factor"] > thresholds.minimum_profit_factor)
            & (enriched["expectancy"] > 0.0)
            & (enriched["net_pnl"] > 0.0)
            & (enriched["sharpe"] > 0.0)
        )
        enriched["passes_relative_quality"] = (
            (enriched["expectancy_ratio_vs_reference"] >= thresholds.minimum_expectancy_ratio_vs_reference)
            & (enriched["profit_factor_ratio_vs_reference"] >= thresholds.minimum_profit_factor_ratio_vs_reference)
            & (enriched["drawdown_multiple_vs_reference"] <= thresholds.maximum_drawdown_multiple_vs_reference)
        )
        enriched["passes_temporal_gate"] = (
            (enriched["positive_year_ratio"] >= thresholds.minimum_positive_year_ratio)
            & (enriched["positive_quarter_ratio"] >= thresholds.minimum_positive_quarter_ratio)
            & (enriched["year_net_pnl_concentration"] <= thresholds.maximum_year_net_pnl_concentration)
        )
        enriched["passes_push_guard"] = (
            enriched["candidate"]
            & enriched["passes_min_sample"]
            & enriched["passes_absolute_quality"]
            & enriched["passes_relative_quality"]
            & enriched["passes_temporal_gate"]
        )
        return enriched

    def _build_candidate_ranking(self, results_frame: pd.DataFrame) -> pd.DataFrame:
        candidates = results_frame.loc[results_frame["candidate"]].copy()
        candidates = candidates.sort_values(
            [
                "reaches_one_trade_per_week",
                "passes_push_guard",
                "trades_per_week_avg",
                "profit_factor",
                "expectancy",
                "net_pnl",
                "max_drawdown",
            ],
            ascending=[False, False, False, False, False, False, True],
        ).reset_index(drop=True)
        if not candidates.empty:
            candidates.insert(0, "rank", range(1, len(candidates) + 1))
        return candidates

    def _build_conclusion(self, results_frame: pd.DataFrame) -> dict[str, Any]:
        if results_frame.empty:
            return {
                "headline": "No ORB push variants were executed.",
                "highest_frequency_variant": None,
                "best_high_frequency_variant": None,
                "best_balance_variant": None,
                "still_promising": False,
                "one_trade_week_candidate_exists": False,
                "two_trade_week_candidate_exists": False,
                "research_verdict": "No results to assess.",
                "next_small_adjustment": "No recommendation available.",
                "ppo_readiness": "Do not move to PPO.",
            }

        reference = results_frame.loc[results_frame["variant"] == "reference"].iloc[0]
        candidates = results_frame.loc[results_frame["candidate"]].copy()
        non_reference = candidates.loc[candidates["variant"] != "reference"].copy()
        highest_frequency = non_reference.sort_values(
            ["trades_per_week_avg", "net_pnl"],
            ascending=[False, False],
        ).iloc[0]

        guarded = non_reference.loc[non_reference["passes_push_guard"]].copy()
        one_week = guarded.loc[guarded["reaches_one_trade_per_week"]].copy()
        two_week = guarded.loc[guarded["reaches_two_trades_per_week"]].copy()

        best_high_frequency = None
        if not one_week.empty:
            best_high_frequency = one_week.sort_values(
                ["trades_per_week_avg", "profit_factor", "expectancy", "net_pnl", "max_drawdown"],
                ascending=[False, False, False, False, True],
            ).iloc[0]

        best_balance = None
        if not guarded.empty:
            best_balance = guarded.sort_values(
                ["reaches_one_trade_per_week", "trades_per_week_avg", "profit_factor", "expectancy", "net_pnl", "max_drawdown"],
                ascending=[False, False, False, False, False, True],
            ).iloc[0]

        still_promising = bool(best_balance is not None)
        if best_balance is None:
            research_verdict = (
                "No existe todavía una variante que lleve esta familia ORB a una frecuencia más práctica sin romper calidad."
            )
        elif best_high_frequency is None:
            research_verdict = (
                "Sí existe una variante más frecuente y defendible, pero todavía no llega a 1 trade por semana de media."
            )
        else:
            research_verdict = (
                "Sí existe una variante más frecuente y defendible que llega al menos a 1 trade por semana sin romper los guard rails."
            )

        next_small_adjustment = (
            f"`{best_balance['variant']}` es la mejor candidata high-frequency-aware para la siguiente validación temporal."
            if best_balance is not None
            else "La familia probablemente tiene un techo natural de frecuencia más bajo; no abriría más parámetros sin validar primero este límite."
        )

        ppo_readiness = (
            "Todavía no PPO. Primero validar temporalmente la mejor candidata high-frequency-aware."
            if best_balance is not None
            else "Todavía no PPO. Esta fase no produjo una candidata suficientemente fuerte."
        )

        return {
            "headline": (
                f"Frequency push around `{self.experiment.base_variant}` points to "
                f"`{highest_frequency['variant']}` as the maximum cadence probe and "
                f"`{best_balance['variant'] if best_balance is not None else 'no variant'}` "
                "as the best high-frequency-aware candidate."
            ),
            "reference_variant": _sanitize_value(reference.to_dict()),
            "highest_frequency_variant": _sanitize_value(highest_frequency.to_dict()),
            "best_high_frequency_variant": _sanitize_value(best_high_frequency.to_dict()) if best_high_frequency is not None else None,
            "best_balance_variant": _sanitize_value(best_balance.to_dict()) if best_balance is not None else None,
            "still_promising": still_promising,
            "one_trade_week_candidate_exists": bool(best_high_frequency is not None),
            "two_trade_week_candidate_exists": bool(not two_week.empty),
            "research_verdict": research_verdict,
            "next_small_adjustment": next_small_adjustment,
            "ppo_readiness": ppo_readiness,
        }

    def _build_summary_markdown(
        self,
        *,
        results_frame: pd.DataFrame,
        yearly_frame: pd.DataFrame,
        quarterly_frame: pd.DataFrame,
        conclusion: dict[str, Any],
    ) -> str:
        reference = conclusion["reference_variant"]
        highest_frequency = conclusion["highest_frequency_variant"]
        best_balance = conclusion["best_balance_variant"]
        best_high_frequency = conclusion["best_high_frequency_variant"]

        lines = [
            "# ORB Frequency Push Summary",
            "",
            "## Direct Answer",
            conclusion["headline"],
            "",
            f"- Research verdict: {conclusion['research_verdict']}",
            f"- Still promising: `{conclusion['still_promising']}`",
            f"- Reaches >= 1 trade/week: `{conclusion['one_trade_week_candidate_exists']}`",
            f"- Reaches >= 2 trades/week: `{conclusion['two_trade_week_candidate_exists']}`",
            f"- PPO readiness: {conclusion['ppo_readiness']}",
            "",
            "## Reference Control",
            (
                f"- `reference`: trades/week `{reference['trades_per_week_avg']:.3f}`, trades/year `{reference['trades_per_year']:.2f}`, "
                f"net `{reference['net_pnl']:.2f}`, PF `{reference['profit_factor']:.2f}`, expectancy `{reference['expectancy']:.2f}`, "
                f"DD `{reference['max_drawdown']:.4f}`."
            ),
            "",
            "## Frequency Push Leaders",
            (
                f"- Highest frequency: `{highest_frequency['variant']}` | trades/week `{highest_frequency['trades_per_week_avg']:.3f}` "
                f"| trades/year `{highest_frequency['trades_per_year']:.2f}` | net `{highest_frequency['net_pnl']:.2f}` "
                f"| PF `{highest_frequency['profit_factor']:.2f}` | expectancy `{highest_frequency['expectancy']:.2f}`."
            ),
        ]
        if best_high_frequency is None:
            lines.append("- >= 1 trade/week with guard rails: no variant cleared that bar.")
        else:
            lines.append(
                f"- Best >= 1 trade/week candidate: `{best_high_frequency['variant']}` | trades/week `{best_high_frequency['trades_per_week_avg']:.3f}` "
                f"| net `{best_high_frequency['net_pnl']:.2f}` | PF `{best_high_frequency['profit_factor']:.2f}` "
                f"| expectancy `{best_high_frequency['expectancy']:.2f}` | DD `{best_high_frequency['max_drawdown']:.4f}`."
            )
        if best_balance is None:
            lines.append("- Best balance: no candidate preserved enough quality while pushing frequency.")
        else:
            lines.append(
                f"- Best balance: `{best_balance['variant']}` | trades/week `{best_balance['trades_per_week_avg']:.3f}` "
                f"| net `{best_balance['net_pnl']:.2f}` | PF `{best_balance['profit_factor']:.2f}` "
                f"| expectancy `{best_balance['expectancy']:.2f}` | DD `{best_balance['max_drawdown']:.4f}`."
            )

        lines.extend(["", "## Variants"])
        display = results_frame.sort_values(
            ["reaches_one_trade_per_week", "passes_push_guard", "trades_per_week_avg", "net_pnl"],
            ascending=[False, False, False, False],
        )
        for _, row in display.iterrows():
            lines.append(
                f"- `{row['variant']}`: trades/week `{row['trades_per_week_avg']:.3f}`, trades/year `{row['trades_per_year']:.2f}`, "
                f"weeks with trade `{row['pct_weeks_with_trade']:.1%}`, max inactive weeks `{int(row['max_inactive_weeks'])}`, "
                f"net `{row['net_pnl']:.2f}`, PF `{row['profit_factor']:.2f}`, expectancy `{row['expectancy']:.2f}`, "
                f"DD `{row['max_drawdown']:.4f}`, push-guard `{('yes' if bool(row['passes_push_guard']) else 'no')}`."
            )

        lines.extend(["", "## Temporal View"])
        if not yearly_frame.empty and {"variant", "exit_year", "trades", "net_pnl", "profit_factor"}.issubset(
            set(yearly_frame.columns)
        ):
            reference_yearly = yearly_frame.loc[yearly_frame["variant"] == "reference"].copy()
            for _, row in reference_yearly.sort_values("exit_year").iterrows():
                lines.append(
                    f"- Reference year `{row['exit_year']}`: trades `{int(row['trades'])}`, net `{float(row['net_pnl']):.2f}`, PF `{float(row['profit_factor']):.2f}`."
                )
            if best_balance is not None:
                best_yearly = yearly_frame.loc[yearly_frame["variant"] == best_balance["variant"]].copy()
                for _, row in best_yearly.sort_values("exit_year").iterrows():
                    lines.append(
                        f"- Best-balance year `{row['exit_year']}`: trades `{int(row['trades'])}`, net `{float(row['net_pnl']):.2f}`, PF `{float(row['profit_factor']):.2f}`."
                    )
        else:
            lines.append("- Temporal yearly breakdown is empty for this run.")

        lines.extend(
            [
                "",
                "## How To Read This",
                "- `orb_frequency_push_results.csv` is the main comparison table.",
                "- `activity_summary.csv` isolates cadence, inactivity streaks and guard rails.",
                "- `candidate_ranking.csv` orders the serious variants by frequency-quality balance.",
                "- `yearly_variant_summary.csv` and `quarterly_variant_summary.csv` show where the gains are concentrated.",
                "",
                "## Next Small Step",
                f"- {conclusion['next_small_adjustment']}",
            ]
        )
        return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run advanced ORB frequency push research around the best local subfamily.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--experiment-config", default="configs/experiments/orb_frequency_push.yaml")
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

    experiment = load_orb_frequency_push_config(args.experiment_config)
    runner = OrbFrequencyPushRunner(args.config_dir, experiment)
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
    highest_frequency = payload["conclusion"]["highest_frequency_variant"]
    print(f"ORB frequency push comparison: {artifacts.comparison_path}")
    print(f"ORB frequency push summary: {artifacts.summary_path}")
    print(
        " ".join(
            [
                f"variants={len(payload['variants'])}",
                f"highest_frequency={highest_frequency['variant'] if highest_frequency else 'n/a'}",
                f"best_balance={best_balance['variant'] if best_balance else 'n/a'}",
            ]
        )
    )
    return 0


def load_orb_frequency_push_config(path: str | Path) -> OrbFrequencyPushConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"ORB frequency push config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"ORB frequency push config must be a mapping: {config_path}")

    thresholds_payload = payload.get("summary_thresholds", {}) or {}
    thresholds = OrbFrequencyPushThresholds(
        minimum_trades_total=int(thresholds_payload.get("minimum_trades_total", 45)),
        minimum_trades_per_year=float(thresholds_payload.get("minimum_trades_per_year", 15.0)),
        target_trades_per_week_avg=float(thresholds_payload.get("target_trades_per_week_avg", 1.0)),
        ambitious_trades_per_week_avg=float(thresholds_payload.get("ambitious_trades_per_week_avg", 2.0)),
        minimum_profit_factor=float(thresholds_payload.get("minimum_profit_factor", 1.10)),
        minimum_expectancy_ratio_vs_reference=float(
            thresholds_payload.get("minimum_expectancy_ratio_vs_reference", 0.70)
        ),
        minimum_profit_factor_ratio_vs_reference=float(
            thresholds_payload.get("minimum_profit_factor_ratio_vs_reference", 0.85)
        ),
        maximum_drawdown_multiple_vs_reference=float(
            thresholds_payload.get("maximum_drawdown_multiple_vs_reference", 2.50)
        ),
        minimum_positive_year_ratio=float(thresholds_payload.get("minimum_positive_year_ratio", 0.50)),
        minimum_positive_quarter_ratio=float(thresholds_payload.get("minimum_positive_quarter_ratio", 0.35)),
        maximum_year_net_pnl_concentration=float(
            thresholds_payload.get("maximum_year_net_pnl_concentration", 0.80)
        ),
    )

    raw_variants = payload.get("variants", [])
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("ORB frequency push config requires a non-empty 'variants' list.")

    variants: list[OrbFrequencyPushVariantConfig] = []
    for raw_variant in raw_variants:
        if not isinstance(raw_variant, dict):
            raise ValueError("Each ORB frequency push variant must be a mapping.")
        overrides = raw_variant.get("overrides", {}) or {}
        if not isinstance(overrides, dict):
            raise ValueError("Variant overrides must be a mapping.")
        variants.append(
            OrbFrequencyPushVariantConfig(
                name=str(raw_variant["name"]),
                label=str(raw_variant.get("label", raw_variant["name"])),
                candidate=bool(raw_variant.get("candidate", True)),
                overrides=overrides,
            )
        )

    return OrbFrequencyPushConfig(
        name=str(payload.get("name", "orb_frequency_push")),
        base_variant=str(payload.get("base_variant", "orb30_close_multi_no_slope_no_rvol_width_laxer_extension_laxer")),
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


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _build_inactivity_metrics(trades_frame: pd.DataFrame) -> dict[str, int]:
    if trades_frame.empty or "entry_timestamp" not in trades_frame.columns:
        return {"max_inactive_days": 0, "max_inactive_weeks": 0}

    entry_index = pd.DatetimeIndex(trades_frame["entry_timestamp"].dropna())
    if entry_index.empty:
        return {"max_inactive_days": 0, "max_inactive_weeks": 0}

    entry_naive = entry_index.tz_convert(UTC).tz_localize(None)
    day_numbers = sorted({int(value.ordinal) for value in entry_naive.to_period("D")})
    week_numbers = sorted({int(value.ordinal) for value in entry_naive.to_period("W-SUN")})

    return {
        "max_inactive_days": _max_gap(day_numbers),
        "max_inactive_weeks": _max_gap(week_numbers),
    }


def _max_gap(values: list[int]) -> int:
    if len(values) <= 1:
        return 0
    gaps = [(current - previous) - 1 for previous, current in zip(values[:-1], values[1:])]
    return int(max(0, max(gaps)))


if __name__ == "__main__":
    raise SystemExit(main())
