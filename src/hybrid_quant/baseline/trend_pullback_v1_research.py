from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from hybrid_quant.core import Settings, apply_settings_overrides
from hybrid_quant.data import read_ohlcv_frame

from .intraday_hybrid_research import (
    IntradayHybridResearchConfig,
    IntradayHybridResearchRunner,
    load_intraday_hybrid_research_config,
)
from .orb_intraday_active_research import _build_runner_from_settings, _parse_datetime, _sanitize_value
from .variants import load_variant_settings


class TrendPullbackV1ResearchRunner:
    """Thin research harness for the gold trend-pullback baseline family.

    The heavy lifting intentionally reuses the existing comparable-baseline
    research runner. This wrapper keeps artifact names and focused robustness
    checks specific to `baseline_trend_pullback_v1`.
    """

    def __init__(self, config_dir: str | Path, experiment: IntradayHybridResearchConfig) -> None:
        self.config_dir = Path(config_dir)
        self.experiment = experiment
        self.delegate = IntradayHybridResearchRunner(config_dir, experiment)

    def run(
        self,
        *,
        input_frame: pd.DataFrame,
        output_dir: str | Path,
        allow_gaps: bool = False,
        start: datetime | None = None,
        end: datetime | None = None,
        selected_variants: Sequence[str] | None = None,
    ) -> dict[str, Path]:
        artifacts = self.delegate.run(
            input_frame=input_frame,
            output_dir=output_dir,
            allow_gaps=allow_gaps,
            start=start,
            end=end,
            selected_variants=selected_variants,
        )
        output_path = artifacts.output_dir
        self._alias_delegate_artifacts(output_path)
        final_variant = self._final_variant(output_path)
        frame = _filter_frame_by_range(input_frame, start=start, end=end)
        if final_variant is not None:
            settings = self._settings_for_variant(final_variant)
            self._write_cost_sensitivity(
                settings=settings,
                variant=final_variant,
                input_frame=frame,
                output_path=output_path,
                allow_gaps=allow_gaps,
            )
            self._write_walk_forward(
                settings=settings,
                variant=final_variant,
                input_frame=frame,
                output_path=output_path,
                allow_gaps=allow_gaps,
            )
        return {
            "comparison": output_path / "trend_pullback_v1_comparison.json",
            "results": output_path / "trend_pullback_v1_results.csv",
            "summary": output_path / "trend_pullback_v1_summary.md",
            "ranking": output_path / "candidate_ranking.csv",
            "cost_sensitivity": output_path / "trend_pullback_v1_cost_sensitivity.csv",
            "walk_forward": output_path / "trend_pullback_v1_walk_forward.csv",
        }

    def _alias_delegate_artifacts(self, output_path: Path) -> None:
        aliases = {
            "intraday_hybrid_comparison.json": "trend_pullback_v1_comparison.json",
            "intraday_hybrid_results.csv": "trend_pullback_v1_results.csv",
            "intraday_hybrid_summary.md": "trend_pullback_v1_summary.md",
        }
        for source_name, target_name in aliases.items():
            source = output_path / source_name
            target = output_path / target_name
            if not source.exists():
                continue
            text = source.read_text(encoding="utf-8")
            text = text.replace("Intraday Hybrid Contextual", "Trend Pullback V1")
            text = text.replace("hybrid intraday", "trend pullback")
            text = text.replace("No trend pullback candidate", "No trend pullback candidate")
            text = text.replace("intraday_hybrid", "trend_pullback_v1")
            target.write_text(text, encoding="utf-8")

    def _final_variant(self, output_path: Path) -> str | None:
        ranking_path = output_path / "candidate_ranking.csv"
        if not ranking_path.exists():
            return None
        ranking = pd.read_csv(ranking_path)
        if ranking.empty or "variant" not in ranking.columns:
            return None
        return str(ranking.iloc[0]["variant"])

    def _settings_for_variant(self, variant_name: str) -> Settings:
        base_settings = load_variant_settings(self.config_dir, self.experiment.base_variant)
        for variant in self.experiment.variants:
            if variant.name != variant_name:
                continue
            source_variant = variant.source_variant or self.experiment.base_variant
            source_settings = (
                base_settings
                if source_variant == self.experiment.base_variant
                else load_variant_settings(self.config_dir, source_variant)
            )
            settings = apply_settings_overrides(source_settings, variant.overrides)
            return apply_settings_overrides(settings, {"strategy": {"variant_name": variant.name}})
        return base_settings

    def _write_cost_sensitivity(
        self,
        *,
        settings: Settings,
        variant: str,
        input_frame: pd.DataFrame,
        output_path: Path,
        allow_gaps: bool,
    ) -> None:
        rows: list[dict[str, Any]] = []
        scenarios = {
            "base": 1.0,
            "costs_x1_5": 1.5,
            "costs_x2": 2.0,
            "costs_x3": 3.0,
        }
        for scenario, scale in scenarios.items():
            scenario_settings = replace(
                settings,
                backtest=replace(
                    settings.backtest,
                    fee_bps=settings.backtest.fee_bps * scale,
                    slippage_bps=settings.backtest.slippage_bps * scale,
                    fee_per_contract_per_side=settings.backtest.fee_per_contract_per_side * scale,
                    slippage_points=settings.backtest.slippage_points * scale,
                ),
            )
            runner = _build_runner_from_settings(scenario_settings)
            artifact_dir = output_path / "cost_sensitivity" / scenario
            baseline_artifacts = runner.run(
                output_dir=artifact_dir,
                input_frame=input_frame,
                allow_gaps=allow_gaps or scenario_settings.data.allow_gaps,
            )
            report = json.loads(baseline_artifacts.report_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "variant": variant,
                    "scenario": scenario,
                    "cost_scale": scale,
                    "number_of_trades": int(report["number_of_trades"]),
                    "net_pnl": _report_float(report, "pnl_net"),
                    "win_rate": _report_float(report, "win_rate"),
                    "payoff": _report_float(report, "payoff"),
                    "expectancy": _report_float(report, "expectancy"),
                    "max_drawdown": _report_float(report, "max_drawdown"),
                    "sharpe": _report_float(report, "sharpe"),
                    "sortino": _report_float(report, "sortino"),
                    "calmar": _report_float(report, "calmar"),
                    "fee_bps": float(scenario_settings.backtest.fee_bps),
                    "slippage_bps": float(scenario_settings.backtest.slippage_bps),
                    "fee_per_contract_per_side": float(scenario_settings.backtest.fee_per_contract_per_side),
                    "slippage_points": float(scenario_settings.backtest.slippage_points),
                }
            )
        pd.DataFrame(rows).to_csv(output_path / "trend_pullback_v1_cost_sensitivity.csv", index=False)

    def _write_walk_forward(
        self,
        *,
        settings: Settings,
        variant: str,
        input_frame: pd.DataFrame,
        output_path: Path,
        allow_gaps: bool,
    ) -> None:
        windows = _anchored_walk_forward_ranges(input_frame, splits=3)
        rows: list[dict[str, Any]] = []
        for window in windows:
            test_frame = input_frame.loc[
                (input_frame.index >= window["test_start"]) & (input_frame.index <= window["test_end"])
            ]
            if test_frame.empty:
                continue
            runner = _build_runner_from_settings(settings)
            artifact_dir = output_path / "walk_forward" / str(window["split"])
            baseline_artifacts = runner.run(
                output_dir=artifact_dir,
                input_frame=test_frame,
                allow_gaps=allow_gaps or settings.data.allow_gaps,
            )
            report = json.loads(baseline_artifacts.report_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "variant": variant,
                    **{key: value.isoformat() if isinstance(value, pd.Timestamp) else value for key, value in window.items()},
                    "number_of_trades": int(report["number_of_trades"]),
                    "net_pnl": _report_float(report, "pnl_net"),
                    "win_rate": _report_float(report, "win_rate"),
                    "payoff": _report_float(report, "payoff"),
                    "expectancy": _report_float(report, "expectancy"),
                    "max_drawdown": _report_float(report, "max_drawdown"),
                    "sharpe": _report_float(report, "sharpe"),
                    "sortino": _report_float(report, "sortino"),
                    "calmar": _report_float(report, "calmar"),
                }
            )
        pd.DataFrame(rows).to_csv(output_path / "trend_pullback_v1_walk_forward.csv", index=False)


def _anchored_walk_forward_ranges(frame: pd.DataFrame, *, splits: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    index = pd.to_datetime(frame.index, utc=True)
    start = index.min()
    end = index.max()
    total = max((end - start).total_seconds(), 1.0)
    rows: list[dict[str, Any]] = []
    train_ratios = [0.45, 0.55, 0.65][:splits]
    validation_ratio = 0.15
    test_ratio = 0.15
    for idx, train_ratio in enumerate(train_ratios, start=1):
        train_end = start + pd.Timedelta(seconds=total * train_ratio)
        validation_end = train_end + pd.Timedelta(seconds=total * validation_ratio)
        test_end = min(validation_end + pd.Timedelta(seconds=total * test_ratio), end)
        if validation_end >= end:
            break
        rows.append(
            {
                "split": f"wf_{idx}",
                "train_start": start,
                "train_end": train_end,
                "validation_start": train_end,
                "validation_end": validation_end,
                "test_start": validation_end,
                "test_end": test_end,
            }
        )
    return rows


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
        raise ValueError("The requested trend pullback research range produced an empty OHLCV frame.")
    return filtered


def _report_float(report: dict[str, Any], key: str) -> float:
    value = report.get(key)
    return float(value) if value is not None else 0.0


def load_trend_pullback_v1_research_config(path: str | Path) -> IntradayHybridResearchConfig:
    return load_intraday_hybrid_research_config(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the baseline_trend_pullback_v1 research matrix.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--experiment-config", default="configs/experiments/trend_pullback_v1_research.yaml")
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
    experiment = load_trend_pullback_v1_research_config(args.experiment_config)
    runner = TrendPullbackV1ResearchRunner(args.config_dir, experiment)
    frame = read_ohlcv_frame(args.input_path)
    artifacts = runner.run(
        input_frame=frame,
        output_dir=args.output_dir,
        allow_gaps=args.allow_gaps,
        start=_parse_datetime(args.start) if args.start else None,
        end=_parse_datetime(args.end) if args.end else None,
        selected_variants=tuple(args.variants or ()),
    )
    payload = json.loads(artifacts["comparison"].read_text(encoding="utf-8"))
    final_candidate = payload["conclusion"].get("final_baseline_variant")
    print(f"Trend pullback comparison: {artifacts['comparison']}")
    print(f"Trend pullback summary: {artifacts['summary']}")
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
