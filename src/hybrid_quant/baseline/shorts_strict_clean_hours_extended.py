from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import shutil
from typing import Any, Sequence

import pandas as pd

from hybrid_quant.data import read_ohlcv_frame

from .orb_focus_validation import _filter_frame_by_range, _parse_datetime, _sanitize_value
from .session_trend_30m_zoom import (
    SessionTrend30mZoomRunner,
    load_session_trend_30m_zoom_config,
)


@dataclass(slots=True)
class ShortsStrictExtendedArtifacts:
    output_dir: Path
    comparison_path: Path
    results_path: Path
    summary_path: Path
    ranking_path: Path
    comparison_summary_path: Path
    yearly_summary_path: Path
    quarterly_summary_path: Path
    hourly_summary_path: Path
    side_summary_path: Path


def _build_coverage_summary(
    *,
    frame: pd.DataFrame,
    requested_start: datetime | None,
    requested_end: datetime | None,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "requested_start": requested_start.isoformat() if requested_start else None,
            "requested_end": requested_end.isoformat() if requested_end else None,
            "actual_start": None,
            "actual_end": None,
            "bars": 0,
            "dominant_interval": None,
            "gap_count": 0,
            "gaps_ge_1h": 0,
            "gaps_ge_1d": 0,
            "largest_gap_hours": 0.0,
            "largest_gaps": [],
        }

    deltas = frame.index.to_series().diff().dropna()
    dominant = deltas.mode().iloc[0] if not deltas.empty and not deltas.mode().empty else pd.Timedelta(minutes=5)
    gap_deltas = deltas.loc[deltas > dominant]
    largest_gaps: list[dict[str, Any]] = []
    for timestamp, delta in gap_deltas.sort_values(ascending=False).head(8).items():
        previous = timestamp - delta
        largest_gaps.append(
            {
                "previous_bar": previous.isoformat(),
                "next_bar": timestamp.isoformat(),
                "gap_minutes": round(float(delta.total_seconds() / 60.0), 2),
                "gap_hours": round(float(delta.total_seconds() / 3600.0), 3),
            }
        )

    return {
        "requested_start": requested_start.isoformat() if requested_start else None,
        "requested_end": requested_end.isoformat() if requested_end else None,
        "actual_start": frame.index[0].isoformat(),
        "actual_end": frame.index[-1].isoformat(),
        "bars": int(len(frame)),
        "dominant_interval": str(dominant),
        "gap_count": int((deltas > dominant).sum()),
        "gaps_ge_1h": int((gap_deltas >= pd.Timedelta(hours=1)).sum()) if not gap_deltas.empty else 0,
        "gaps_ge_1d": int((gap_deltas >= pd.Timedelta(days=1)).sum()) if not gap_deltas.empty else 0,
        "largest_gap_hours": round(float(gap_deltas.max().total_seconds() / 3600.0), 3) if not gap_deltas.empty else 0.0,
        "largest_gaps": largest_gaps,
    }


def _rename_or_copy(source: Path, target: Path) -> Path:
    shutil.copyfile(source, target)
    return target


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _build_comparison_summary(results_frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "variant",
        "label",
        "candidate",
        "source_variant",
        "number_of_trades",
        "trades_per_year",
        "trades_per_month",
        "trades_per_week_avg",
        "win_rate",
        "payoff",
        "expectancy",
        "profit_factor",
        "gross_pnl",
        "net_pnl",
        "max_drawdown",
        "max_drawdown_pct",
        "sharpe",
        "sortino",
        "calmar",
        "positive_year_ratio",
        "positive_quarter_ratio",
        "year_net_pnl_concentration",
        "long_trades",
        "short_trades",
        "passes_quality_guard",
        "within_target_drawdown_band",
        "delta_net_pnl_vs_reference",
        "delta_profit_factor_vs_reference",
        "delta_expectancy_vs_reference",
        "delta_max_drawdown_vs_reference",
        "delta_trades_per_week_avg_vs_reference",
    ]
    available = [column for column in columns if column in results_frame.columns]
    return results_frame[available].sort_values(
        ["candidate", "passes_quality_guard", "net_pnl", "profit_factor"],
        ascending=[False, False, False, False],
    )


def _build_extended_summary_markdown(
    *,
    coverage: dict[str, Any],
    results_frame: pd.DataFrame,
    ranking: pd.DataFrame,
    comparison_payload: dict[str, Any],
) -> str:
    conclusion = comparison_payload["conclusion"]
    best_profitability = conclusion["best_profitability_variant"]
    best_balance = conclusion["best_balance_variant"]
    controls = conclusion.get("controls", {})
    reference_row = results_frame.loc[results_frame["variant"] == "reference"].iloc[0]

    lines = [
        "# Shorts Strict Clean Hours Extended Summary",
        "",
        "## Direct Answer",
        conclusion["headline"],
        "",
        "- Drawdown convention: `max_drawdown` is stored as a fraction of equity. Example: `0.02448 = 2.448%`.",
        (
            f"- Dataset coverage used: `{coverage['actual_start']}` -> `{coverage['actual_end']}` | "
            f"bars `{coverage['bars']}` | dominant interval `{coverage['dominant_interval']}`."
        ),
        (
            f"- Requested range: `{coverage['requested_start']}` -> `{coverage['requested_end']}` | "
            f"gaps >= 1h `{coverage['gaps_ge_1h']}` | gaps >= 1d `{coverage['gaps_ge_1d']}`."
        ),
        (
            f"- Reference `shorts_strict_clean_hours`: net `{reference_row['net_pnl']:.2f}` | PF `{reference_row['profit_factor']:.2f}` | "
            f"expectancy `{reference_row['expectancy']:.2f}` | DD `{reference_row['max_drawdown_pct']:.2f}%` | "
            f"trades/week `{reference_row['trades_per_week_avg']:.3f}`."
        ),
        (
            f"- Positive in all observed years? {'yes' if conclusion.get('reference_positive_all_years') else 'no'} | "
            f"best year `{conclusion.get('reference_best_year')}` | worst year `{conclusion.get('reference_worst_year')}`."
        ),
        (
            f"- Best profitability variant: `{best_profitability['variant']}` | net `{best_profitability['net_pnl']:.2f}` | "
            f"PF `{best_profitability['profit_factor']:.2f}` | DD `{best_profitability['max_drawdown_pct']:.2f}%`."
        ),
        (
            f"- Best balance variant: `{best_balance['variant']}` | net `{best_balance['net_pnl']:.2f}` | "
            f"PF `{best_balance['profit_factor']:.2f}` | expectancy `{best_balance['expectancy']:.2f}` | "
            f"DD `{best_balance['max_drawdown_pct']:.2f}%` | trades/week `{best_balance['trades_per_week_avg']:.3f}`."
        ),
        f"- PPO readiness: {conclusion['ppo_readiness']}",
        "",
        "## Ranking",
    ]

    for _, row in ranking.head(10).iterrows():
        lines.append(
            f"- `#{int(row['rank'])} {row['variant']}`: net `{row['net_pnl']:.2f}`, PF `{row['profit_factor']:.2f}`, "
            f"expectancy `{row['expectancy']:.2f}`, DD `{row['max_drawdown_pct']:.2f}%`, trades/week `{row['trades_per_week_avg']:.3f}`."
        )

    lines.extend(["", "## Required Controls"])
    for control_name in [
        "session_trend_30m_original",
        "long_only_clean_hours_control",
        "context_reclaim_15m_control",
    ]:
        control = controls.get(control_name)
        if not control:
            continue
        lines.append(
            f"- `{control_name}`: net `{control['net_pnl']:.2f}`, PF `{control['profit_factor']:.2f}`, "
            f"DD `{control['max_drawdown_pct']:.2f}%`, trades/week `{control['trades_per_week_avg']:.3f}`."
        )

    lines.extend(["", "## Variant Table"])
    ordered = results_frame.sort_values(
        ["candidate", "passes_quality_guard", "net_pnl", "profit_factor", "max_drawdown"],
        ascending=[False, False, False, False, True],
    )
    for _, row in ordered.iterrows():
        lines.append(
            f"- `{row['variant']}`: net `{row['net_pnl']:.2f}`, gross `{row['gross_pnl']:.2f}`, trades `{int(row['number_of_trades'])}`, "
            f"trades/year `{row['trades_per_year']:.2f}`, trades/month `{row['trades_per_month']:.2f}`, "
            f"trades/week `{row['trades_per_week_avg']:.3f}`, PF `{row['profit_factor']:.2f}`, expectancy `{row['expectancy']:.2f}`, "
            f"DD `{row['max_drawdown_pct']:.2f}%`, positive years `{row['positive_year_ratio'] * 100:.1f}%`, "
            f"positive quarters `{row['positive_quarter_ratio'] * 100:.1f}%`."
        )

    lines.extend(["", "## How To Read This"])
    lines.append("- `shorts_strict_clean_hours_extended_results.csv` is the main table for profitability first, drawdown second, frequency third.")
    lines.append("- `candidate_ranking.csv` is the shortlist around `shorts_strict_clean_hours`.")
    lines.append("- `comparison_summary.csv` is the clean side-by-side view of the required controls and all local variants.")
    lines.append("- `yearly_variant_summary.csv` and `quarterly_variant_summary.csv` are the main temporal robustness tables.")
    lines.append("- `hourly_variant_summary.csv` and `side_variant_summary.csv` show whether shorts remain the damage center or are already controlled.")
    return "\n".join(lines) + "\n"


def run_extended_research(
    *,
    config_dir: str | Path,
    experiment_config: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    allow_gaps: bool = False,
    start: datetime | None = None,
    end: datetime | None = None,
    selected_variants: Sequence[str] | None = None,
) -> ShortsStrictExtendedArtifacts:
    frame = read_ohlcv_frame(input_path)
    requested_start = start or datetime(2020, 1, 1, tzinfo=UTC)
    requested_end = end or datetime(2026, 1, 1, tzinfo=UTC)
    filtered = _filter_frame_by_range(frame, start=requested_start, end=requested_end)
    coverage = _build_coverage_summary(frame=filtered, requested_start=requested_start, requested_end=requested_end)

    experiment = load_session_trend_30m_zoom_config(experiment_config)
    runner = SessionTrend30mZoomRunner(config_dir, experiment)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    runner.run(
        input_frame=frame,
        output_dir=output_path,
        allow_gaps=allow_gaps,
        start=requested_start,
        end=requested_end,
        selected_variants=selected_variants,
    )

    base_comparison_path = output_path / "session_trend_30m_zoom_comparison.json"
    base_results_path = output_path / "session_trend_30m_zoom_results.csv"
    base_summary_path = output_path / "session_trend_30m_zoom_summary.md"
    ranking_path = output_path / "candidate_ranking.csv"
    yearly_summary_path = output_path / "yearly_variant_summary.csv"
    quarterly_summary_path = output_path / "quarterly_variant_summary.csv"
    hourly_summary_path = output_path / "hourly_variant_summary.csv"
    side_summary_path = output_path / "side_variant_summary.csv"

    comparison_payload = json.loads(base_comparison_path.read_text(encoding="utf-8"))
    comparison_payload["coverage"] = coverage
    comparison_payload["drawdown_convention"] = "max_drawdown is stored as a fraction of equity. Example: 0.02448 means 2.448%."

    results_frame = _safe_read_csv(base_results_path)
    ranking_frame = _safe_read_csv(ranking_path)
    yearly_frame = _safe_read_csv(yearly_summary_path)
    comparison_summary = _build_comparison_summary(results_frame)

    reference_years = yearly_frame.loc[yearly_frame["variant"] == "reference"].copy() if not yearly_frame.empty else pd.DataFrame()
    if not reference_years.empty:
        best_row = reference_years.sort_values(["net_pnl", "profit_factor"], ascending=[False, False]).iloc[0]
        worst_row = reference_years.sort_values(["net_pnl", "profit_factor"], ascending=[True, True]).iloc[0]
        trades_column = "number_of_trades" if "number_of_trades" in reference_years.columns else "trades"
        comparison_payload["conclusion"]["reference_positive_all_years"] = bool((reference_years["net_pnl"] > 0.0).all())
        comparison_payload["conclusion"]["reference_best_year"] = str(best_row["exit_year"])
        comparison_payload["conclusion"]["reference_worst_year"] = str(worst_row["exit_year"])
        comparison_payload["conclusion"]["reference_yearly_snapshot"] = [
            _sanitize_value(row)
            for row in reference_years[["exit_year", trades_column, "net_pnl", "profit_factor"]]
            .rename(
                columns={
                    "exit_year": "block",
                    trades_column: "trades",
                }
            )
            .to_dict(orient="records")
        ]

    comparison_path = output_path / "shorts_strict_clean_hours_extended_comparison.json"
    results_path = output_path / "shorts_strict_clean_hours_extended_results.csv"
    summary_path = output_path / "shorts_strict_clean_hours_extended_summary.md"
    comparison_summary_path = output_path / "comparison_summary.csv"

    comparison_path.write_text(json.dumps(_sanitize_value(comparison_payload), indent=2), encoding="utf-8")
    _rename_or_copy(base_results_path, results_path)
    comparison_summary.to_csv(comparison_summary_path, index=False)
    summary_path.write_text(
        _build_extended_summary_markdown(
            coverage=coverage,
            results_frame=results_frame,
            ranking=ranking_frame,
            comparison_payload=comparison_payload,
        ),
        encoding="utf-8",
    )
    (output_path / "dataset_coverage.json").write_text(
        json.dumps(_sanitize_value(coverage), indent=2),
        encoding="utf-8",
    )
    (output_path / "base_session_trend_30m_zoom_summary.md").write_text(
        base_summary_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    return ShortsStrictExtendedArtifacts(
        output_dir=output_path,
        comparison_path=comparison_path,
        results_path=results_path,
        summary_path=summary_path,
        ranking_path=ranking_path,
        comparison_summary_path=comparison_summary_path,
        yearly_summary_path=yearly_summary_path,
        quarterly_summary_path=quarterly_summary_path,
        hourly_summary_path=hourly_summary_path,
        side_summary_path=side_summary_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extended validation and local zoom around shorts_strict_clean_hours."
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument(
        "--experiment-config",
        default="configs/experiments/shorts_strict_clean_hours_extended.yaml",
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
    start = _parse_datetime(args.start) if args.start else None
    end = _parse_datetime(args.end) if args.end else None
    artifacts = run_extended_research(
        config_dir=args.config_dir,
        experiment_config=args.experiment_config,
        input_path=args.input_path,
        output_dir=args.output_dir,
        allow_gaps=args.allow_gaps,
        start=start,
        end=end,
        selected_variants=args.variants,
    )
    print(f"Extended comparison written to {artifacts.comparison_path}")
    print(f"Summary written to {artifacts.summary_path}")
    print(
        " ".join(
            [
                f"results={artifacts.results_path.name}",
                f"ranking={artifacts.ranking_path.name}",
                f"comparison={artifacts.comparison_summary_path.name}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
