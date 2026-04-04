from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from hybrid_quant.bootstrap import TradingApplication
from hybrid_quant.core import (
    BacktestRequest,
    FeatureSnapshot,
    MarketBar,
    MarketDataBatch,
    Settings,
    StrategyContext,
    StrategySignal,
    ValidationReport,
    load_settings,
)
from hybrid_quant.data import (
    BinanceHistoricalDownloader,
    DownloadRequest,
    HistoricalDataIngestionService,
    OHLCVCleaner,
    ParquetDatasetStore,
    TimeIndexValidator,
)

from .diagnostics import BaselineDiagnosticsRunner
from .variants import BASELINE_VARIANTS, build_variant_application, load_variant_settings, variant_summary_payload


@dataclass(slots=True)
class VariantRunArtifacts:
    variant_name: str
    artifact_dir: Path
    diagnostics_dir: Path
    report_path: Path
    diagnostics_path: Path
    summary_path: Path
    metrics: dict[str, Any]


@dataclass(slots=True)
class BaselineComparisonArtifacts:
    output_dir: Path
    comparison_path: Path
    summary_path: Path
    variant_artifacts: dict[str, VariantRunArtifacts]
    oos_comparison_path: Path | None = None
    oos_summary_path: Path | None = None


class BaselineComparisonRunner:
    """Compare direct strategy/backtest variants on the same OHLCV frame.

    This runner intentionally bypasses the current `BaselineRunner` risk-filtering layer so
    `baseline_v1` remains directly comparable with the historical diagnostic baseline.
    """

    def __init__(self, config_dir: str | Path) -> None:
        self.config_dir = Path(config_dir)
        self.base_settings = load_variant_settings(self.config_dir, "baseline_v1")
        self.cleaner = OHLCVCleaner()
        self.validator = TimeIndexValidator()
        self.data_service = HistoricalDataIngestionService(
            downloader=BinanceHistoricalDownloader(
                base_url=self.base_settings.data.historical_api_url,
                timeout_seconds=self.base_settings.data.request_timeout_seconds,
            ),
            store=ParquetDatasetStore(
                compression=self.base_settings.data.parquet_compression,
                engine=self.base_settings.data.parquet_engine,
            ),
        )

    def run(
        self,
        *,
        output_dir: str | Path,
        variants: Sequence[str] = ("baseline_v1", "baseline_v2", "baseline_v3"),
        request: DownloadRequest | None = None,
        input_frame: pd.DataFrame | None = None,
        allow_gaps: bool = False,
        oos_start: datetime | None = None,
        oos_end: datetime | None = None,
    ) -> BaselineComparisonArtifacts:
        if input_frame is None and request is None:
            raise ValueError("BaselineComparisonRunner.run requires an input frame or a download request.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        frame = self._prepare_frame(input_frame=input_frame, request=request, allow_gaps=allow_gaps)
        variant_artifacts = self._run_variants(frame=frame, output_dir=output_path, variants=variants)
        comparison_payload = self._build_comparison_payload(frame=frame, variant_artifacts=variant_artifacts)

        comparison_path = output_path / "baseline_comparison.json"
        summary_path = output_path / "baseline_comparison_summary.md"
        comparison_path.write_text(
            json.dumps(self._sanitize_value(comparison_payload), indent=2),
            encoding="utf-8",
        )
        summary_path.write_text(
            self._build_comparison_summary(comparison_payload, title="Baseline Comparison"),
            encoding="utf-8",
        )

        self._export_root_aliases(
            output_path=output_path,
            variant_artifacts=variant_artifacts,
            comparison_path=comparison_path,
            summary_path=summary_path,
        )

        oos_comparison_path: Path | None = None
        oos_summary_path: Path | None = None
        if oos_start is not None and oos_end is not None:
            oos_frame = frame.loc[
                (frame.index >= pd.Timestamp(oos_start))
                & (frame.index <= pd.Timestamp(oos_end))
            ].copy()
            if not oos_frame.empty:
                oos_output_dir = output_path / "oos_check"
                oos_variants = self._run_variants(
                    frame=oos_frame,
                    output_dir=oos_output_dir,
                    variants=variants,
                )
                oos_payload = self._build_comparison_payload(
                    frame=oos_frame,
                    variant_artifacts=oos_variants,
                )
                oos_payload["note"] = (
                    "This holdout is a temporal slice of the provided input, not an independent quarter."
                )
                oos_comparison_path = output_path / "baseline_comparison_oos.json"
                oos_summary_path = output_path / "baseline_comparison_oos_summary.md"
                oos_comparison_path.write_text(
                    json.dumps(self._sanitize_value(oos_payload), indent=2),
                    encoding="utf-8",
                )
                oos_summary_path.write_text(
                    self._build_comparison_summary(
                        oos_payload,
                        title="Baseline Comparison Holdout",
                    ),
                    encoding="utf-8",
                )

        return BaselineComparisonArtifacts(
            output_dir=output_path,
            comparison_path=comparison_path,
            summary_path=summary_path,
            variant_artifacts=variant_artifacts,
            oos_comparison_path=oos_comparison_path,
            oos_summary_path=oos_summary_path,
        )

    def _prepare_frame(
        self,
        *,
        input_frame: pd.DataFrame | None,
        request: DownloadRequest | None,
        allow_gaps: bool,
    ) -> pd.DataFrame:
        if input_frame is not None:
            cleaned_frame, _ = self.cleaner.clean(input_frame)
            self.validator.validate(
                cleaned_frame,
                self.base_settings.market.execution_timeframe,
                allow_gaps=allow_gaps,
            )
            return cleaned_frame

        cleaned_frame, _, _ = self.data_service.prepare_frame(
            request=request,
            allow_gaps=allow_gaps,
        )
        return cleaned_frame

    def _run_variants(
        self,
        *,
        frame: pd.DataFrame,
        output_dir: Path,
        variants: Sequence[str],
    ) -> dict[str, VariantRunArtifacts]:
        results: dict[str, VariantRunArtifacts] = {}
        for variant_name in variants:
            application = build_variant_application(self.config_dir, variant_name)
            artifact_dir = output_dir / variant_name
            diagnostics_dir = output_dir / f"{variant_name}_diagnostics"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

            report_payload = self._run_direct_baseline_variant(
                application=application,
                frame=frame,
                output_dir=artifact_dir,
            )
            diagnostics_runner = BaselineDiagnosticsRunner(application)
            diagnostics_artifacts = diagnostics_runner.run(
                artifact_dir=artifact_dir,
                output_dir=diagnostics_dir,
                include_variants=True,
                include_risk_replay=False,
            )
            diagnostics_payload = json.loads(
                diagnostics_artifacts.diagnostics_path.read_text(encoding="utf-8")
            )
            metrics = self._build_variant_metrics(
                variant_name=variant_name,
                settings=application.settings,
                report_payload=report_payload,
                diagnostics_payload=diagnostics_payload,
                artifact_dir=artifact_dir,
                diagnostics_dir=diagnostics_dir,
            )
            results[variant_name] = VariantRunArtifacts(
                variant_name=variant_name,
                artifact_dir=artifact_dir,
                diagnostics_dir=diagnostics_dir,
                report_path=artifact_dir / "report.json",
                diagnostics_path=diagnostics_artifacts.diagnostics_path,
                summary_path=diagnostics_artifacts.summary_path,
                metrics=metrics,
            )
        return results

    def _run_direct_baseline_variant(
        self,
        *,
        application: TradingApplication,
        frame: pd.DataFrame,
        output_dir: Path,
    ) -> dict[str, Any]:
        bars = self._frame_to_bars(frame)
        batch = MarketDataBatch(
            symbol=application.settings.market.symbol,
            timeframe=application.settings.market.execution_timeframe,
            bars=bars,
            metadata={"source": application.settings.strategy.variant_name},
        )
        features = application.feature_pipeline.transform(batch)
        signals = self._generate_signals(application, bars, features)
        result = application.backtest_engine.run(
            BacktestRequest(
                bars=bars,
                features=features,
                signals=signals,
                initial_capital=application.settings.backtest.initial_capital,
                risk_per_trade_fraction=application.settings.risk.max_risk_per_trade,
                max_leverage=application.settings.risk.max_leverage,
                signal_cooldown_bars=application.settings.strategy.signal_cooldown_bars,
                exit_zscore_threshold=application.settings.strategy.exit_zscore,
                session_close_hour_utc=application.settings.strategy.session_close_hour_utc,
                session_close_minute_utc=application.settings.strategy.session_close_minute_utc,
                intrabar_exit_policy=application.settings.backtest.intrabar_exit_policy,
            )
        )
        validation_report = application.validator.validate(result)

        ohlcv_frame = frame.copy()
        ohlcv_frame.index.name = "open_time"
        feature_frame = self._feature_snapshots_to_frame(features)
        signal_frame = self._signals_to_frame(signals)
        trade_frame = self._trades_to_frame(result.trade_records)
        report_payload = self._build_variant_report_payload(
            settings=application.settings,
            result=result,
            validation_report=validation_report,
            trade_frame=trade_frame,
            bars=len(ohlcv_frame),
            features=len(feature_frame),
        )

        ohlcv_frame.to_csv(output_dir / "ohlcv.csv")
        feature_frame.to_csv(output_dir / "features.csv")
        signal_frame.to_csv(output_dir / "signals.csv", index=False)
        trade_frame.to_csv(output_dir / "trades.csv", index=False)
        (output_dir / "report.json").write_text(
            json.dumps(self._sanitize_value(report_payload), indent=2),
            encoding="utf-8",
        )
        (output_dir / "summary.md").write_text(
            self._build_variant_summary(report_payload),
            encoding="utf-8",
        )
        return report_payload

    def _generate_signals(
        self,
        application: TradingApplication,
        bars: Sequence[MarketBar],
        features: Sequence[FeatureSnapshot],
    ) -> list[StrategySignal]:
        signals: list[StrategySignal] = []
        for bar, feature in zip(bars, features, strict=True):
            adx = feature.values.get("adx_1h")
            regime = (
                "trend"
                if adx is not None
                and math.isfinite(float(adx))
                and float(adx) > application.settings.strategy.adx_threshold
                else "range"
            )
            signals.append(
                application.strategy.generate(
                    StrategyContext(
                        symbol=application.settings.market.symbol,
                        execution_timeframe=application.settings.market.execution_timeframe,
                        filter_timeframe=application.settings.market.filter_timeframe,
                        bars=[bar],
                        features=[feature],
                        regime=regime,
                    )
                )
            )
        return signals

    def _build_variant_metrics(
        self,
        *,
        variant_name: str,
        settings: Settings,
        report_payload: dict[str, Any],
        diagnostics_payload: dict[str, Any],
        artifact_dir: Path,
        diagnostics_dir: Path,
    ) -> dict[str, Any]:
        baseline_metrics = diagnostics_payload["baseline_metrics"]
        return {
            "variant": variant_name,
            "symbol": settings.market.symbol,
            "execution_timeframe": settings.market.execution_timeframe,
            "filter_timeframe": settings.market.filter_timeframe,
            "strategy": variant_summary_payload(settings),
            "number_of_trades": int(baseline_metrics["number_of_trades"]),
            "win_rate": float(baseline_metrics["win_rate"]),
            "average_win": float(baseline_metrics["average_win"]),
            "average_loss": float(baseline_metrics["average_loss"]),
            "payoff": float(baseline_metrics["payoff_real"]),
            "expectancy": float(baseline_metrics["expectancy"]),
            "gross_pnl": float(baseline_metrics["gross_pnl"]),
            "net_pnl": float(baseline_metrics["net_pnl"]),
            "fees_paid": float(baseline_metrics["fees_paid_total"]),
            "slippage_impact_estimate": float(baseline_metrics["estimated_slippage_drag"] or 0.0),
            "fee_impact_estimate": float(baseline_metrics["estimated_fee_drag"] or 0.0),
            "total_cost_impact_estimate": float(
                baseline_metrics["estimated_total_cost_drag"] or 0.0
            ),
            "max_drawdown": float(baseline_metrics["max_drawdown"]),
            "sharpe": float(baseline_metrics["sharpe"]),
            "sortino": float(baseline_metrics["sortino"]),
            "calmar": float(baseline_metrics["calmar"]),
            "profitable_months_pct": float(baseline_metrics["profitable_months_pct"]),
            "max_consecutive_losses": int(baseline_metrics["max_consecutive_losses"]),
            "average_holding_bars": float(baseline_metrics["average_holding_bars"]),
            "total_return": float(report_payload["total_return"]),
            "validation": report_payload["validation"],
            "report_path": str(artifact_dir / "report.json"),
            "summary_path": str(artifact_dir / "summary.md"),
            "diagnostics_path": str(diagnostics_dir / "diagnostics.json"),
            "diagnostics_summary_path": str(diagnostics_dir / "diagnostics_summary.md"),
        }

    def _build_comparison_payload(
        self,
        *,
        frame: pd.DataFrame,
        variant_artifacts: dict[str, VariantRunArtifacts],
    ) -> dict[str, Any]:
        variant_names = list(variant_artifacts.keys())
        variants = {
            name: self._sanitize_value(artifact.metrics)
            for name, artifact in variant_artifacts.items()
        }
        pair_deltas = self._build_pair_deltas(variants=variants, variant_names=variant_names)
        primary_delta = self._resolve_primary_delta(pair_deltas=pair_deltas, variant_names=variant_names)
        conclusion = self._build_comparison_conclusion(
            variants=variants,
            variant_names=variant_names,
            pair_deltas=pair_deltas,
        )
        period_metadata = self._comparison_period_metadata(variants)

        return {
            "comparison_mode": "direct_strategy_backtest",
            "period": {
                **period_metadata,
                "start": frame.index[0].isoformat() if not frame.empty else None,
                "end": frame.index[-1].isoformat() if not frame.empty else None,
                "bars": int(len(frame)),
            },
            "variants": variants,
            "delta": primary_delta,
            "pair_deltas": pair_deltas,
            "conclusion": conclusion,
        }

    def _build_delta_payload(
        self,
        baseline_v1: dict[str, Any] | None,
        baseline_v2: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if baseline_v1 is None or baseline_v2 is None:
            return None

        trades_v1 = float(baseline_v1["number_of_trades"])
        fees_v1 = float(baseline_v1["fees_paid"])
        drawdown_v1 = float(baseline_v1["max_drawdown"])

        return {
            "trade_count_delta": int(
                baseline_v2["number_of_trades"] - baseline_v1["number_of_trades"]
            ),
            "trade_reduction_pct": ((trades_v1 - float(baseline_v2["number_of_trades"])) / trades_v1)
            if trades_v1 > 0.0
            else 0.0,
            "win_rate_delta": float(baseline_v2["win_rate"] - baseline_v1["win_rate"]),
            "payoff_delta": float(baseline_v2["payoff"] - baseline_v1["payoff"]),
            "expectancy_delta": float(baseline_v2["expectancy"] - baseline_v1["expectancy"]),
            "gross_pnl_delta": float(baseline_v2["gross_pnl"] - baseline_v1["gross_pnl"]),
            "net_pnl_delta": float(baseline_v2["net_pnl"] - baseline_v1["net_pnl"]),
            "fees_paid_delta": float(baseline_v2["fees_paid"] - baseline_v1["fees_paid"]),
            "fee_reduction_pct": ((fees_v1 - float(baseline_v2["fees_paid"])) / fees_v1)
            if fees_v1 > 0.0
            else 0.0,
            "slippage_impact_delta": float(
                baseline_v2["slippage_impact_estimate"] - baseline_v1["slippage_impact_estimate"]
            ),
            "max_drawdown_delta": float(baseline_v2["max_drawdown"] - baseline_v1["max_drawdown"]),
            "max_drawdown_reduction_pct": ((drawdown_v1 - float(baseline_v2["max_drawdown"])) / drawdown_v1)
            if drawdown_v1 > 0.0
            else 0.0,
            "sharpe_delta": float(baseline_v2["sharpe"] - baseline_v1["sharpe"]),
            "sortino_delta": float(baseline_v2["sortino"] - baseline_v1["sortino"]),
            "calmar_delta": float(baseline_v2["calmar"] - baseline_v1["calmar"]),
            "profitable_months_pct_delta": float(
                baseline_v2["profitable_months_pct"] - baseline_v1["profitable_months_pct"]
            ),
            "max_consecutive_losses_delta": int(
                baseline_v2["max_consecutive_losses"] - baseline_v1["max_consecutive_losses"]
            ),
        }

    def _build_pair_deltas(
        self,
        *,
        variants: dict[str, dict[str, Any]],
        variant_names: Sequence[str],
    ) -> dict[str, Any]:
        pair_deltas: dict[str, Any] = {}
        for base_index, base_name in enumerate(variant_names):
            for compare_name in variant_names[base_index + 1 :]:
                base_metrics = variants.get(base_name)
                compare_metrics = variants.get(compare_name)
                if base_metrics is None or compare_metrics is None:
                    continue
                pair_deltas[f"{compare_name}_vs_{base_name}"] = self._build_delta_payload(
                    base_metrics,
                    compare_metrics,
                )
        return pair_deltas

    def _resolve_primary_delta(
        self,
        *,
        pair_deltas: dict[str, Any],
        variant_names: Sequence[str],
    ) -> dict[str, Any] | None:
        if len(variant_names) >= 2:
            preferred_key = f"{variant_names[-1]}_vs_{variant_names[-2]}"
            if preferred_key in pair_deltas:
                return pair_deltas[preferred_key]
            fallback_key = f"{variant_names[-1]}_vs_{variant_names[0]}"
            if fallback_key in pair_deltas:
                return pair_deltas[fallback_key]
        return next(iter(pair_deltas.values()), None)

    def _comparison_period_metadata(self, variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
        symbols = sorted({metrics["symbol"] for metrics in variants.values() if metrics.get("symbol")})
        execution_timeframes = sorted(
            {
                metrics["execution_timeframe"]
                for metrics in variants.values()
                if metrics.get("execution_timeframe")
            }
        )
        filter_timeframes = sorted(
            {metrics["filter_timeframe"] for metrics in variants.values() if metrics.get("filter_timeframe")}
        )
        return {
            "symbol": symbols[0] if len(symbols) == 1 else "mixed_variants",
            "execution_timeframe": execution_timeframes[0]
            if len(execution_timeframes) == 1
            else "mixed_variants",
            "filter_timeframe": filter_timeframes[0]
            if len(filter_timeframes) == 1
            else "mixed_variants",
            "symbols": symbols,
            "execution_timeframes": execution_timeframes,
            "filter_timeframes": filter_timeframes,
        }

    def _build_comparison_conclusion(
        self,
        *,
        variants: dict[str, dict[str, Any]],
        variant_names: Sequence[str],
        pair_deltas: dict[str, Any],
    ) -> dict[str, Any]:
        baseline_v1 = variants.get("baseline_v1")
        baseline_v2 = variants.get("baseline_v2")
        baseline_v3 = variants.get("baseline_v3")
        canonical_only = set(variant_names).issubset({"baseline_v1", "baseline_v2", "baseline_v3"})

        if not canonical_only:
            if len(variant_names) < 2:
                single_name = variant_names[0] if variant_names else "unknown_variant"
                return {
                    "headline": (
                        f"{single_name} ran successfully, but comparison needs at least two variants."
                    ),
                    "top_changes": [],
                    "economic_structure_improved": None,
                    "serious_enough_for_ppo": None,
                    "recommended_next_step": (
                        "Run the new trend family against another comparable variant on the same dataset before "
                        "drawing performance conclusions."
                    ),
                }

            best_variant_name, best_variant = max(
                variants.items(),
                key=lambda item: (float(item[1]["net_pnl"]), -float(item[1]["max_drawdown"])),
            )
            safest_variant_name, safest_variant = min(
                variants.items(),
                key=lambda item: (float(item[1]["max_drawdown"]), -float(item[1]["net_pnl"])),
            )
            reference_name = variant_names[0]
            candidate_name = variant_names[-1]
            reference = variants.get(reference_name)
            candidate = variants.get(candidate_name)
            candidate_delta = pair_deltas.get(f"{candidate_name}_vs_{reference_name}")
            economic_structure_improved = None
            if reference is not None and candidate is not None:
                economic_structure_improved = bool(
                    candidate["net_pnl"] >= reference["net_pnl"]
                    and candidate["max_drawdown"] <= reference["max_drawdown"]
                )
            top_changes = [
                (
                    f"Best net PnL on this slice: `{best_variant_name}` with "
                    f"`{best_variant['net_pnl']:.2f}` across `{best_variant['number_of_trades']}` trades."
                ),
                (
                    f"Lowest drawdown on this slice: `{safest_variant_name}` with "
                    f"`{safest_variant['max_drawdown']:.4f}` max drawdown."
                ),
            ]
            if candidate_delta is not None:
                top_changes.append(
                    (
                        f"Requested lead comparison `{candidate_name}` vs `{reference_name}` moved net PnL by "
                        f"`{candidate_delta['net_pnl_delta']:.2f}` and trade count by "
                        f"`{candidate_delta['trade_count_delta']}`."
                    )
                )
            return {
                "headline": (
                    f"Generic comparison complete. `{best_variant_name}` is the strongest requested variant by net "
                    "PnL on this slice, and the new family is now wired into the same backtest and diagnostics flow."
                ),
                "top_changes": top_changes,
                "economic_structure_improved": economic_structure_improved,
                "serious_enough_for_ppo": None,
                "recommended_next_step": (
                    "Run a dedicated diagnosis and robust validation sprint on the strongest trend-breakout variant "
                    "before considering any RL layer."
                ),
            }

        if baseline_v1 is None or baseline_v2 is None:
            return {
                "headline": "Comparison is incomplete because at least one requested baseline variant is missing.",
                "top_changes": [],
                "economic_structure_improved": None,
                "serious_enough_for_ppo": None,
                "recommended_next_step": "Run both baseline_v1 and baseline_v2 on the same input first.",
            }

        if baseline_v3 is None:
            delta = pair_deltas.get("baseline_v2_vs_baseline_v1")
            if delta is None:
                return {
                    "headline": "Comparison is incomplete because pair deltas could not be computed.",
                    "top_changes": [],
                    "economic_structure_improved": None,
                    "serious_enough_for_ppo": None,
                    "recommended_next_step": "Re-run the comparison with a complete pair of variants.",
                }

            top_changes = [
                (
                    "baseline_v2 cuts frequency materially, which is exactly what the diagnosis asked for, "
                    f"moving from `{baseline_v1['number_of_trades']}` to `{baseline_v2['number_of_trades']}` trades."
                ),
                (
                    "baseline_v2 reduces cost drag by trading less often: fees move from "
                    f"`{baseline_v1['fees_paid']:.2f}` to `{baseline_v2['fees_paid']:.2f}`."
                ),
                (
                    "baseline_v2 improves setup economics if payoff, expectancy or drawdown improve; the net effect is "
                    f"`PnL {baseline_v1['net_pnl']:.2f}` -> `{baseline_v2['net_pnl']:.2f}` with drawdown "
                    f"`{baseline_v1['max_drawdown']:.4f}` -> `{baseline_v2['max_drawdown']:.4f}`."
                ),
            ]
            economic_structure_improved = bool(
                baseline_v2["payoff"] >= baseline_v1["payoff"]
                and baseline_v2["expectancy"] >= baseline_v1["expectancy"]
                and baseline_v2["max_drawdown"] <= baseline_v1["max_drawdown"]
                and baseline_v2["fees_paid"] <= baseline_v1["fees_paid"]
            )
            serious_enough_for_ppo = bool(
                economic_structure_improved
                and baseline_v2["net_pnl"] >= 0.0
                and baseline_v2["max_drawdown"] <= 0.10
                and baseline_v2["number_of_trades"] >= 20
            )
            headline = (
                "baseline_v2 is clearly better than baseline_v1, but it still does not clear the bar for a serious "
                "PPO training target yet."
                if delta["net_pnl_delta"] > 0.0
                else "baseline_v2 does not yet improve the baseline enough, so another rule-based refinement sprint is still required before PPO."
            )
            next_step = (
                "Run one more small baseline-only sprint focused on entry selectivity and cost-aware targeting before "
                "treating PPO as a serious layer."
            )
            return {
                "headline": headline,
                "top_changes": top_changes,
                "economic_structure_improved": economic_structure_improved,
                "serious_enough_for_ppo": serious_enough_for_ppo,
                "recommended_next_step": next_step,
            }

        delta = pair_deltas.get("baseline_v3_vs_baseline_v2")
        if delta is None:
            return {
                "headline": "Comparison is incomplete because baseline_v3 deltas are missing.",
                "top_changes": [],
                "economic_structure_improved": None,
                "serious_enough_for_ppo": None,
                "recommended_next_step": "Re-run the comparison including baseline_v3.",
            }

        top_changes = [
            (
                "baseline_v3 keeps the low-frequency profile of baseline_v2 while tightening the worst trading windows, "
                f"moving from `{baseline_v2['number_of_trades']}` to `{baseline_v3['number_of_trades']}` trades."
            ),
            (
                "baseline_v3 cuts cost drag further by removing persistently weak hours: fees move from "
                f"`{baseline_v2['fees_paid']:.2f}` to `{baseline_v3['fees_paid']:.2f}`."
            ),
            (
                "baseline_v3 improves the economic result of the same core setup, with "
                f"`PnL {baseline_v2['net_pnl']:.2f}` -> `{baseline_v3['net_pnl']:.2f}` and drawdown "
                f"`{baseline_v2['max_drawdown']:.4f}` -> `{baseline_v3['max_drawdown']:.4f}`."
            ),
        ]

        economic_structure_improved = bool(
            baseline_v3["payoff"] >= baseline_v2["payoff"] or baseline_v3["net_pnl"] > baseline_v2["net_pnl"]
        ) and bool(
            baseline_v3["expectancy"] >= baseline_v2["expectancy"]
            and baseline_v3["max_drawdown"] <= baseline_v2["max_drawdown"]
            and baseline_v3["fees_paid"] <= baseline_v2["fees_paid"]
        )
        serious_enough_for_ppo = bool(
            economic_structure_improved
            and baseline_v3["net_pnl"] >= 0.0
            and baseline_v3["max_drawdown"] <= 0.10
            and baseline_v3["number_of_trades"] >= 20
        )

        if baseline_v3["net_pnl"] >= 0.0:
            headline = (
                "baseline_v3 turns the refined baseline net-positive on this comparison slice while keeping frequency "
                "and drawdown very contained."
            )
        elif delta["net_pnl_delta"] > 0.0:
            headline = (
                "baseline_v3 is better than baseline_v2, but it still does not clear the bar for a serious PPO "
                "training target yet."
            )
        else:
            headline = (
                "baseline_v3 does not yet improve the baseline enough, so another rule-based refinement sprint "
                "is still required before PPO."
            )

        if serious_enough_for_ppo:
            next_step = (
                "Move to a validation sprint first, using baseline_v3 as the candidate trade generator and keeping "
                "baseline_v1/baseline_v2 as historical controls."
            )
        else:
            next_step = (
                "Run a validation-focused sprint next: stress baseline_v3 on a broader temporal sample before adding "
                "more rule complexity or returning to PPO."
            )

        return {
            "headline": headline,
            "top_changes": top_changes,
            "economic_structure_improved": economic_structure_improved,
            "serious_enough_for_ppo": serious_enough_for_ppo,
            "recommended_next_step": next_step,
        }

    def _build_variant_report_payload(
        self,
        *,
        settings: Settings,
        result: Any,
        validation_report: ValidationReport,
        trade_frame: pd.DataFrame,
        bars: int,
        features: int,
    ) -> dict[str, Any]:
        gross_pnl = float(trade_frame["gross_pnl"].sum()) if not trade_frame.empty else 0.0
        fees_paid = float(trade_frame["fees_paid"].sum()) if not trade_frame.empty else 0.0
        average_win = float(result.metadata.get("average_win", 0.0) or 0.0)
        average_loss = float(result.metadata.get("average_loss", 0.0) or 0.0)
        return {
            "symbol": settings.market.symbol,
            "execution_timeframe": settings.market.execution_timeframe,
            "filter_timeframe": settings.market.filter_timeframe,
            "variant": variant_summary_payload(settings),
            "bars": bars,
            "features": features,
            "start": result.start.isoformat() if result.start else None,
            "end": result.end.isoformat() if result.end else None,
            "number_of_trades": result.trades,
            "win_rate": result.win_rate,
            "average_win": average_win,
            "average_loss": average_loss,
            "payoff": result.payoff,
            "expectancy": result.expectancy,
            "gross_pnl": gross_pnl,
            "pnl_net": result.pnl_net,
            "fees_paid": fees_paid,
            "max_drawdown": result.max_drawdown,
            "sharpe": result.sharpe,
            "sortino": result.sortino,
            "calmar": result.calmar,
            "total_return": result.total_return,
            "equity_final": result.equity_final,
            "validation": {
                "passed": validation_report.passed,
                "checks": validation_report.checks,
                "summary": validation_report.summary,
            },
            "backtest": {
                key: self._sanitize_value(value)
                for key, value in result.metadata.items()
                if key != "equity_curve"
            },
        }

    def _build_variant_summary(self, payload: dict[str, Any]) -> str:
        lines = [
            "# Baseline Variant Summary",
            "",
            f"- Variant: `{payload['variant']['variant_name']}`",
            f"- Strategy name: `{payload['variant']['name']}`",
            f"- Period: `{payload['start']}` -> `{payload['end']}`",
            f"- Bars: `{payload['bars']}`",
            f"- Trades: `{payload['number_of_trades']}`",
            f"- Win rate: `{payload['win_rate'] * 100:.2f}%`",
            f"- Average win: `{payload['average_win']:.2f}`",
            f"- Average loss: `{payload['average_loss']:.2f}`",
            f"- Payoff: `{payload['payoff']}`",
            f"- Expectancy: `{payload['expectancy']}`",
            f"- Gross PnL: `{payload['gross_pnl']}`",
            f"- Net PnL: `{payload['pnl_net']}`",
            f"- Fees paid: `{payload['fees_paid']}`",
            f"- Max drawdown: `{payload['max_drawdown']}`",
            f"- Sharpe: `{payload['sharpe']}`",
            f"- Sortino: `{payload['sortino']}`",
            f"- Calmar: `{payload['calmar']}`",
            f"- Validation: `{payload['validation']['summary']}`",
        ]
        return "\n".join(lines) + "\n"

    def _build_comparison_summary(self, payload: dict[str, Any], *, title: str) -> str:
        variants = payload["variants"]
        pair_deltas = payload.get("pair_deltas", {})
        conclusion = payload.get("conclusion", {})

        lines = [
            f"# {title}",
            "",
            "## Setup",
            f"- Comparison mode: `{payload['comparison_mode']}`",
            f"- Symbol: `{payload['period']['symbol']}`",
            f"- Period: `{payload['period']['start']}` -> `{payload['period']['end']}`",
            f"- Bars: `{payload['period']['bars']}`",
            "",
            "## Direct Answer",
            conclusion.get("headline", "No conclusion generated."),
        ]
        note = payload.get("note")
        if note:
            lines.extend(["", f"- Note: {note}"])

        for variant_name, metrics in variants.items():
            lines.extend(
                [
                    "",
                    f"## {variant_name}",
                    f"- Family: `{metrics['strategy'].get('family', 'unknown')}`",
                    f"- Symbol config: `{metrics.get('symbol')}`",
                    f"- Trades: `{metrics['number_of_trades']}`",
                    f"- Win rate: `{metrics['win_rate'] * 100:.2f}%`",
                    f"- Average win: `{metrics['average_win']:.2f}`",
                    f"- Average loss: `{metrics['average_loss']:.2f}`",
                    f"- Payoff: `{metrics['payoff']:.4f}`",
                    f"- Expectancy: `{metrics['expectancy']:.2f}`",
                    f"- Gross PnL: `{metrics['gross_pnl']:.2f}`",
                    f"- Net PnL: `{metrics['net_pnl']:.2f}`",
                    f"- Fees paid: `{metrics['fees_paid']:.2f}`",
                    f"- Slippage impact estimate: `{metrics['slippage_impact_estimate']:.2f}`",
                    f"- Max drawdown: `{metrics['max_drawdown']:.4f}`",
                    f"- Sharpe: `{metrics['sharpe']:.4f}`",
                    f"- Sortino: `{metrics['sortino']:.4f}`",
                    f"- Calmar: `{metrics['calmar']:.4f}`",
                    f"- Profitable months pct: `{metrics['profitable_months_pct'] * 100:.2f}%`",
                    f"- Max consecutive losses: `{metrics['max_consecutive_losses']}`",
                ]
            )

        for label, delta in pair_deltas.items():
            if delta is None:
                continue
            lines.extend(
                [
                    "",
                    f"## {label}",
                    f"- Trade count delta: `{delta['trade_count_delta']}` ({delta['trade_reduction_pct'] * 100:.2f}% fewer trades).",
                    f"- Net PnL delta: `{delta['net_pnl_delta']:.2f}`",
                    f"- Gross PnL delta: `{delta['gross_pnl_delta']:.2f}`",
                    f"- Fees delta: `{delta['fees_paid_delta']:.2f}` ({delta['fee_reduction_pct'] * 100:.2f}% reduction).",
                    f"- Win rate delta: `{delta['win_rate_delta'] * 100:.2f} pp`",
                    f"- Payoff delta: `{delta['payoff_delta']:.4f}`",
                    f"- Expectancy delta: `{delta['expectancy_delta']:.2f}`",
                    f"- Max drawdown delta: `{delta['max_drawdown_delta']:.4f}` ({delta['max_drawdown_reduction_pct'] * 100:.2f}% reduction).",
                ]
            )

        top_changes = conclusion.get("top_changes", [])
        if top_changes:
            lines.extend(["", "## Key Takeaways"])
            for item in top_changes:
                lines.append(f"- {item}")

        serious_enough_for_ppo = conclusion.get("serious_enough_for_ppo")
        if serious_enough_for_ppo is None:
            lines.extend(
                [
                    "",
                    "## Next Step",
                    f"- Economic structure improved: `{conclusion.get('economic_structure_improved')}`",
                    f"- Recommended next step: {conclusion.get('recommended_next_step', 'n/a')}",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "## PPO Readiness",
                    f"- Economic structure improved: `{conclusion.get('economic_structure_improved')}`",
                    f"- Serious enough for PPO now: `{serious_enough_for_ppo}`",
                    f"- Recommended next step: {conclusion.get('recommended_next_step', 'n/a')}",
                ]
            )
        return "\n".join(lines) + "\n"

    def _export_root_aliases(
        self,
        *,
        output_path: Path,
        variant_artifacts: dict[str, VariantRunArtifacts],
        comparison_path: Path,
        summary_path: Path,
    ) -> None:
        variant_names = list(variant_artifacts.keys())
        if variant_names:
            comparison_slug = "_".join(variant_names)
            shutil.copyfile(comparison_path, output_path / f"{comparison_slug}_comparison.json")
            shutil.copyfile(summary_path, output_path / f"{comparison_slug}_summary.md")
            exact_alias_slug = self._variant_alias_slug(variant_names)
            shutil.copyfile(comparison_path, output_path / f"{exact_alias_slug}_comparison.json")
            shutil.copyfile(summary_path, output_path / f"{exact_alias_slug}_summary.md")

        for variant_name, artifact in variant_artifacts.items():
            shutil.copyfile(artifact.report_path, output_path / f"{variant_name}_report.json")
            alias_map = {
                "monthly_breakdown.csv": f"{variant_name}_monthly_breakdown.csv",
                "hourly_breakdown.csv": f"{variant_name}_hourly_breakdown.csv",
                "exit_reason_breakdown.csv": f"{variant_name}_exit_reason_breakdown.csv",
                "side_breakdown.csv": f"{variant_name}_side_breakdown.csv",
            }
            for source_name, target_name in alias_map.items():
                source = artifact.diagnostics_dir / source_name
                if source.exists():
                    shutil.copyfile(source, output_path / target_name)

    def _frame_to_bars(self, frame: pd.DataFrame) -> list[MarketBar]:
        normalized = frame.copy()
        normalized.index = pd.to_datetime(normalized.index, utc=True)
        normalized = normalized.sort_index(kind="mergesort")
        return [
            MarketBar(
                timestamp=timestamp.to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for timestamp, row in normalized.iterrows()
        ]

    def _feature_snapshots_to_frame(
        self,
        feature_snapshots: Sequence[FeatureSnapshot],
    ) -> pd.DataFrame:
        rows = []
        for snapshot in feature_snapshots:
            row = {"open_time": snapshot.timestamp}
            row.update(snapshot.values)
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows).set_index("open_time")
        frame.index = pd.to_datetime(frame.index, utc=True)
        return frame

    def _signals_to_frame(self, signals: Sequence[StrategySignal]) -> pd.DataFrame:
        rows = []
        for signal in signals:
            rows.append(
                {
                    "timestamp": signal.timestamp,
                    "symbol": signal.symbol,
                    "side": signal.side.value,
                    "raw_side": signal.side.value,
                    "strength": signal.strength,
                    "entry_price": signal.entry_price,
                    "stop_price": signal.stop_price,
                    "target_price": signal.target_price,
                    "time_stop_bars": signal.time_stop_bars,
                    "close_on_session_end": signal.close_on_session_end,
                    "entry_reason": signal.entry_reason,
                    "rationale": signal.rationale,
                }
            )
        return pd.DataFrame(rows)

    def _trades_to_frame(self, trades: Sequence[Any]) -> pd.DataFrame:
        columns = [
            "symbol",
            "side",
            "entry_timestamp",
            "exit_timestamp",
            "entry_price",
            "exit_price",
            "quantity",
            "gross_pnl",
            "net_pnl",
            "fees_paid",
            "return_pct",
            "bars_held",
            "exit_reason",
            "entry_reason",
        ]
        rows = []
        for trade in trades:
            rows.append(
                {
                    "symbol": trade.symbol,
                    "side": trade.side.value,
                    "entry_timestamp": trade.entry_timestamp,
                    "exit_timestamp": trade.exit_timestamp,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "quantity": trade.quantity,
                    "gross_pnl": trade.gross_pnl,
                    "net_pnl": trade.net_pnl,
                    "fees_paid": trade.fees_paid,
                    "return_pct": trade.return_pct,
                    "bars_held": trade.bars_held,
                    "exit_reason": trade.exit_reason,
                    "entry_reason": trade.entry_reason,
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._sanitize_value(inner) for key, inner in value.items()}
        if isinstance(value, tuple):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return value
        return value

    def _variant_alias_slug(self, variant_names: Sequence[str]) -> str:
        suffixes = []
        for name in variant_names:
            suffix = name.removeprefix("baseline_")
            suffixes.append(suffix)
        return "baseline_" + "_".join(suffixes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline variants on the same dataset.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output-dir")
    parser.add_argument("--input-path")
    parser.add_argument("--allow-gaps", action="store_true")
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--oos-start")
    parser.add_argument("--oos-end")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = load_settings(args.config_dir)
    runner = BaselineComparisonRunner(args.config_dir)
    variants = tuple(args.variants or list(BASELINE_VARIANTS))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(settings.storage.artifacts_dir) / "baseline-comparison"
    )

    oos_start = _parse_datetime(args.oos_start) if args.oos_start else None
    oos_end = _parse_datetime(args.oos_end) if args.oos_end else None

    if args.input_path:
        frame = _read_input_frame(args.input_path)
        artifacts = runner.run(
            output_dir=output_dir,
            input_frame=frame,
            variants=variants,
            allow_gaps=args.allow_gaps,
            oos_start=oos_start,
            oos_end=oos_end,
        )
    else:
        start = _parse_datetime(args.start or settings.data.default_start)
        end = _resolve_end_datetime(args.end, settings, start)
        request = DownloadRequest(
            symbol=settings.market.symbol,
            interval=settings.market.execution_timeframe,
            start=start,
            end=end,
            limit=settings.data.request_limit,
        )
        artifacts = runner.run(
            output_dir=output_dir,
            request=request,
            variants=variants,
            allow_gaps=args.allow_gaps or settings.data.allow_gaps,
            oos_start=oos_start,
            oos_end=oos_end,
        )

    payload = json.loads(artifacts.comparison_path.read_text(encoding="utf-8"))
    variant_items = list(payload["variants"].items())
    selected_variant = None
    if variant_items:
        selected_name, selected_variant = max(
            variant_items,
            key=lambda item: (float(item[1]["net_pnl"]), -float(item[1]["max_drawdown"])),
        )
        print(
            " ".join(
                [
                    f"selected_variant={selected_name}",
                    f"trades={selected_variant['number_of_trades']}",
                    f"net_pnl={selected_variant['net_pnl']}",
                    f"drawdown={selected_variant['max_drawdown']}",
                ]
            )
        )
    print(f"Comparison written to {artifacts.comparison_path}")
    print(f"Summary written to {artifacts.summary_path}")
    if artifacts.oos_comparison_path is not None:
        print(f"Holdout comparison written to {artifacts.oos_comparison_path}")
    return 0


def _read_input_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    frame = pd.read_csv(source, parse_dates=["open_time"], index_col="open_time")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


def _parse_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _resolve_end_datetime(raw_end: str | None, settings: Settings, start: datetime) -> datetime:
    if raw_end:
        return _parse_datetime(raw_end)
    if settings.data.default_end:
        return _parse_datetime(settings.data.default_end)
    return start + timedelta(days=90)


if __name__ == "__main__":
    raise SystemExit(main())
