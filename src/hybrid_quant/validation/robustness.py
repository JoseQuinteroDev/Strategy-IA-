from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

import pandas as pd

from hybrid_quant.baseline.variants import (
    BASELINE_VARIANTS,
    build_variant_application,
    load_variant_settings,
    variant_summary_payload,
)
from hybrid_quant.bootstrap import TradingApplication, build_application_from_settings
from hybrid_quant.core import (
    BacktestRequest,
    FeatureSnapshot,
    MarketBar,
    MarketDataBatch,
    Settings,
    StrategyContext,
    StrategySignal,
    apply_settings_overrides,
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
from hybrid_quant.validation.walk_forward import RollingWindow, build_rolling_windows


@dataclass(slots=True)
class VariantExecution:
    label: str
    settings: Settings
    result: Any
    trade_frame: pd.DataFrame
    metrics: dict[str, Any]
    bars: int
    start: str | None
    end: str | None


@dataclass(slots=True)
class RobustnessArtifacts:
    output_dir: Path
    report_path: Path
    summary_path: Path
    walk_forward_results_path: Path
    temporal_block_results_path: Path
    monte_carlo_summary_path: Path
    cost_sensitivity_path: Path
    report: dict[str, Any]


class RobustnessValidationRunner:
    def __init__(
        self,
        config_dir: str | Path,
        *,
        variant_name: str = "baseline_v3",
        settings: Settings | None = None,
    ) -> None:
        self.config_dir = Path(config_dir)
        self.variant_name = variant_name
        self.settings = settings or load_variant_settings(self.config_dir, variant_name)
        self.cleaner = OHLCVCleaner()
        self.validator = TimeIndexValidator()
        self.data_service = HistoricalDataIngestionService(
            downloader=BinanceHistoricalDownloader(
                base_url=self.settings.data.historical_api_url,
                timeout_seconds=self.settings.data.request_timeout_seconds,
            ),
            store=ParquetDatasetStore(
                compression=self.settings.data.parquet_compression,
                engine=self.settings.data.parquet_engine,
            ),
        )

    @classmethod
    def from_config(
        cls,
        config_dir: str | Path,
        *,
        variant_name: str = "baseline_v3",
    ) -> "RobustnessValidationRunner":
        return cls(config_dir, variant_name=variant_name)

    def run(
        self,
        *,
        output_dir: str | Path,
        request: DownloadRequest | None = None,
        input_frame: pd.DataFrame | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        allow_gaps: bool = False,
        artifact_suffix: str | None = None,
    ) -> RobustnessArtifacts:
        if input_frame is None and request is None:
            raise ValueError("RobustnessValidationRunner.run requires an input frame or a download request.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        frame = self._prepare_frame(
            input_frame=input_frame,
            request=request,
            allow_gaps=allow_gaps,
            start=start,
            end=end,
        )
        limitations = self._detect_limitations(frame)
        full_execution = self._execute_full_frame(frame=frame, settings=self.settings, label="full_dataset")
        walk_forward_frame, walk_forward_summary = self._run_walk_forward(frame)
        temporal_block_frame, temporal_block_summary = self._run_temporal_blocks(frame)
        monte_carlo_summary = self._run_monte_carlo(full_execution.trade_frame)
        cost_sensitivity_frame, cost_sensitivity_summary = self._run_cost_sensitivity(frame)
        decision = self._classify_robustness(
            walk_forward_summary=walk_forward_summary,
            temporal_block_summary=temporal_block_summary,
            monte_carlo_summary=monte_carlo_summary,
            cost_sensitivity_summary=cost_sensitivity_summary,
            limitations=limitations,
        )

        report = {
            "variant": self.variant_name,
            "strategy": variant_summary_payload(self.settings),
            "dataset": {
                "symbol": self.settings.market.symbol,
                "execution_timeframe": self.settings.market.execution_timeframe,
                "filter_timeframe": self.settings.market.filter_timeframe,
                "start": frame.index[0].isoformat() if not frame.empty else None,
                "end": frame.index[-1].isoformat() if not frame.empty else None,
                "bars": int(len(frame)),
            },
            "limitations": limitations,
            "full_dataset": {
                "metrics": full_execution.metrics,
                "bars": full_execution.bars,
                "start": full_execution.start,
                "end": full_execution.end,
            },
            "walk_forward": {
                "summary": walk_forward_summary,
                "rows": walk_forward_frame.to_dict(orient="records"),
            },
            "temporal_blocks": {
                "summary": temporal_block_summary,
                "rows": temporal_block_frame.to_dict(orient="records"),
            },
            "monte_carlo": monte_carlo_summary,
            "cost_sensitivity": {
                "summary": cost_sensitivity_summary,
                "rows": cost_sensitivity_frame.to_dict(orient="records"),
            },
            "decision": decision,
        }

        report_path = output_path / "robustness_report.json"
        summary_path = output_path / "robustness_summary.md"
        walk_forward_results_path = output_path / "walk_forward_results.csv"
        temporal_block_results_path = output_path / "temporal_block_results.csv"
        monte_carlo_summary_path = output_path / "monte_carlo_summary.json"
        cost_sensitivity_path = output_path / "cost_sensitivity.csv"

        report_path.write_text(json.dumps(self._sanitize_value(report), indent=2), encoding="utf-8")
        summary_path.write_text(self._build_summary_markdown(report), encoding="utf-8")
        walk_forward_frame.to_csv(walk_forward_results_path, index=False)
        temporal_block_frame.to_csv(temporal_block_results_path, index=False)
        monte_carlo_summary_path.write_text(
            json.dumps(self._sanitize_value(monte_carlo_summary), indent=2),
            encoding="utf-8",
        )
        cost_sensitivity_frame.to_csv(cost_sensitivity_path, index=False)
        self._export_alias_artifacts(
            output_path=output_path,
            artifact_suffix=artifact_suffix,
            report_path=report_path,
            summary_path=summary_path,
            walk_forward_results_path=walk_forward_results_path,
            temporal_block_results_path=temporal_block_results_path,
            monte_carlo_summary_path=monte_carlo_summary_path,
            cost_sensitivity_path=cost_sensitivity_path,
        )

        return RobustnessArtifacts(
            output_dir=output_path,
            report_path=report_path,
            summary_path=summary_path,
            walk_forward_results_path=walk_forward_results_path,
            temporal_block_results_path=temporal_block_results_path,
            monte_carlo_summary_path=monte_carlo_summary_path,
            cost_sensitivity_path=cost_sensitivity_path,
            report=report,
        )

    def _export_alias_artifacts(
        self,
        *,
        output_path: Path,
        artifact_suffix: str | None,
        report_path: Path,
        summary_path: Path,
        walk_forward_results_path: Path,
        temporal_block_results_path: Path,
        monte_carlo_summary_path: Path,
        cost_sensitivity_path: Path,
    ) -> None:
        if artifact_suffix is None:
            return
        suffix = artifact_suffix.strip().lower().replace(" ", "_")
        if not suffix:
            return

        alias_map = {
            report_path: output_path / f"robustness_report_{suffix}.json",
            summary_path: output_path / f"robustness_summary_{suffix}.md",
            walk_forward_results_path: output_path / f"walk_forward_results_{suffix}.csv",
            temporal_block_results_path: output_path / f"temporal_block_results_{suffix}.csv",
            monte_carlo_summary_path: output_path / f"monte_carlo_summary_{suffix}.json",
            cost_sensitivity_path: output_path / f"cost_sensitivity_{suffix}.csv",
        }
        for source, target in alias_map.items():
            shutil.copyfile(source, target)

    def _prepare_frame(
        self,
        *,
        input_frame: pd.DataFrame | None,
        request: DownloadRequest | None,
        allow_gaps: bool,
        start: datetime | None,
        end: datetime | None,
    ) -> pd.DataFrame:
        if input_frame is not None:
            cleaned_frame, _ = self.cleaner.clean(input_frame)
            self.validator.validate(
                cleaned_frame,
                self.settings.market.execution_timeframe,
                allow_gaps=allow_gaps,
            )
            frame = cleaned_frame
        else:
            frame, _, _ = self.data_service.prepare_frame(
                request=request,
                allow_gaps=allow_gaps,
            )

        if start is not None:
            frame = frame.loc[frame.index >= pd.Timestamp(start)]
        if end is not None:
            frame = frame.loc[frame.index <= pd.Timestamp(end)]
        if frame.empty:
            raise ValueError("The selected validation frame is empty after applying the date filters.")
        return frame

    def _detect_limitations(self, frame: pd.DataFrame) -> list[str]:
        limitations: list[str] = []
        month_periods = frame.index.tz_convert("UTC").tz_localize(None).to_period("M").nunique()
        if month_periods < 4:
            limitations.append(
                f"Only {month_periods} monthly blocks are available in the local dataset; no independent quarter is present in the workspace."
            )
        if len(frame) < 5000:
            limitations.append("The dataset is short for robust intraday validation, so split-level evidence will be thin.")
        return limitations

    def _run_walk_forward(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        validation = self.settings.validation
        windows = build_rolling_windows(
            total_bars=len(frame),
            splits=validation.walk_forward_splits,
            train_ratio=validation.walk_forward_train_ratio,
            validation_ratio=validation.walk_forward_validation_ratio,
            test_ratio=validation.walk_forward_test_ratio,
        )
        rows: list[dict[str, Any]] = []
        test_trade_frames: list[pd.DataFrame] = []

        for window in windows:
            for phase_name, start_idx, end_idx in [
                ("train", window.train_start, window.train_end),
                ("validation", window.validation_start, window.validation_end),
                ("test", window.test_start, window.test_end),
            ]:
                execution = self._execute_window_slice(
                    full_frame=frame,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    settings=self.settings,
                    label=f"wf_{window.window_id}_{phase_name}",
                )
                row = {
                    "window_id": window.window_id,
                    "phase": phase_name,
                    "start": execution.start,
                    "end": execution.end,
                    "bars": execution.bars,
                }
                row.update(execution.metrics)
                rows.append(row)
                if phase_name == "test":
                    test_trade_frames.append(execution.trade_frame)

        results = pd.DataFrame(rows)
        test_rows = results.loc[results["phase"] == "test"].copy()
        non_empty_test_frames = [trade_frame for trade_frame in test_trade_frames if not trade_frame.empty]
        combined_test_trades = (
            pd.concat(non_empty_test_frames, ignore_index=True) if non_empty_test_frames else pd.DataFrame()
        )
        summary = self._build_walk_forward_summary(test_rows, combined_test_trades)
        summary["window_definitions"] = [window.to_dict() for window in windows]
        return results, summary

    def _run_temporal_blocks(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        frequency = self.settings.validation.temporal_block_frequency.lower()
        if frequency != "monthly":
            raise ValueError(f"Unsupported temporal block frequency: {self.settings.validation.temporal_block_frequency}")

        period_index = frame.index.tz_convert("UTC").tz_localize(None).to_period("M")
        rows: list[dict[str, Any]] = []
        trade_frames: list[pd.DataFrame] = []
        for block_id, period in enumerate(period_index.unique()):
            mask = period_index == period
            block_positions = [index for index, keep in enumerate(mask.tolist()) if keep]
            if not block_positions:
                continue
            start_idx = block_positions[0]
            end_idx = block_positions[-1] + 1
            execution = self._execute_window_slice(
                full_frame=frame,
                start_idx=start_idx,
                end_idx=end_idx,
                settings=self.settings,
                label=f"block_{period}",
            )
            row = {
                "block_id": block_id,
                "block_label": str(period),
                "start": execution.start,
                "end": execution.end,
                "bars": execution.bars,
            }
            row.update(execution.metrics)
            rows.append(row)
            trade_frames.append(execution.trade_frame)

        results = pd.DataFrame(rows)
        summary = self._build_temporal_block_summary(results, trade_frames)
        return results, summary

    def _run_monte_carlo(self, trade_frame: pd.DataFrame) -> dict[str, Any]:
        simulations = self.settings.validation.monte_carlo_simulations
        seed = self.settings.validation.monte_carlo_seed
        trade_pnls = [float(value) for value in trade_frame.get("net_pnl", pd.Series(dtype=float)).tolist()]
        initial_capital = self.settings.backtest.initial_capital

        if not trade_pnls:
            return {
                "simulations": simulations,
                "seed": seed,
                "trades": 0,
                "note": "Monte Carlo was skipped because the baseline produced no trades.",
                "equity_final": self._percentile_summary([initial_capital]),
                "total_return": self._percentile_summary([0.0]),
                "max_drawdown": self._percentile_summary([0.0]),
                "expectancy": self._percentile_summary([0.0]),
            }

        rng = random.Random(seed)
        equity_final: list[float] = []
        total_return: list[float] = []
        max_drawdown: list[float] = []
        expectancy: list[float] = []

        for _ in range(simulations):
            permuted = list(trade_pnls)
            rng.shuffle(permuted)
            current_equity = initial_capital
            peak_equity = initial_capital
            worst_drawdown = 0.0
            for pnl in permuted:
                current_equity += pnl
                peak_equity = max(peak_equity, current_equity)
                if peak_equity > 0.0:
                    worst_drawdown = max(worst_drawdown, (peak_equity - current_equity) / peak_equity)
            equity_final.append(current_equity)
            total_return.append((current_equity - initial_capital) / initial_capital)
            max_drawdown.append(worst_drawdown)
            expectancy.append(sum(permuted) / len(permuted))

        return {
            "simulations": simulations,
            "seed": seed,
            "trades": len(trade_pnls),
            "note": (
                "Pure trade reordering changes path-dependent drawdown, but leaves final equity, total return and expectancy invariant."
            ),
            "equity_final": self._percentile_summary(equity_final),
            "total_return": self._percentile_summary(total_return),
            "max_drawdown": self._percentile_summary(max_drawdown),
            "expectancy": self._percentile_summary(expectancy),
        }

    def _run_cost_sensitivity(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        positive_scenarios = 0
        for scenario in self.settings.validation.cost_scenarios:
            stressed_settings = apply_settings_overrides(
                self.settings,
                {
                    "backtest": {
                        "fee_bps": self.settings.backtest.fee_bps * scenario.fee_multiplier,
                        "slippage_bps": self.settings.backtest.slippage_bps * scenario.slippage_multiplier,
                    }
                },
            )
            execution = self._execute_full_frame(
                frame=frame,
                settings=stressed_settings,
                label=scenario.name,
            )
            row = {
                "scenario": scenario.name,
                "fee_multiplier": scenario.fee_multiplier,
                "slippage_multiplier": scenario.slippage_multiplier,
            }
            row.update(execution.metrics)
            rows.append(row)
            if execution.metrics["net_pnl"] >= 0.0:
                positive_scenarios += 1

        results = pd.DataFrame(rows)
        scenario_count = len(results)
        summary = {
            "scenario_count": scenario_count,
            "positive_scenarios": positive_scenarios,
            "survival_ratio": (positive_scenarios / scenario_count) if scenario_count else 0.0,
            "base_net_pnl": float(results.loc[results["scenario"] == "base", "net_pnl"].iloc[0]) if not results.empty and (results["scenario"] == "base").any() else None,
            "worst_case_net_pnl": float(results["net_pnl"].min()) if not results.empty else 0.0,
            "worst_case_drawdown": float(results["max_drawdown"].max()) if not results.empty else 0.0,
            "note": (
                "Trade counts can change across stress scenarios because baseline_v3 uses cost-aware filters, "
                "so higher fees or slippage may block marginal setups before execution."
            ),
        }
        return results, summary

    def _execute_full_frame(
        self,
        *,
        frame: pd.DataFrame,
        settings: Settings,
        label: str,
    ) -> VariantExecution:
        return self._execute_window_slice(
            full_frame=frame,
            start_idx=0,
            end_idx=len(frame),
            settings=settings,
            label=label,
        )

    def _execute_window_slice(
        self,
        *,
        full_frame: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        settings: Settings,
        label: str,
    ) -> VariantExecution:
        start_idx = max(0, int(start_idx))
        end_idx = min(len(full_frame), int(end_idx))
        if end_idx <= start_idx:
            raise ValueError("Invalid slice requested for robustness validation.")

        warmup_bars = min(settings.data.warmup_bars, start_idx)
        warmup_start = start_idx - warmup_bars
        working_frame = full_frame.iloc[warmup_start:end_idx].copy()
        phase_offset = start_idx - warmup_start

        application = build_application_from_settings(settings)
        all_bars = self._frame_to_bars(working_frame)
        batch = MarketDataBatch(
            symbol=settings.market.symbol,
            timeframe=settings.market.execution_timeframe,
            bars=all_bars,
            metadata={"source": label},
        )
        all_features = application.feature_pipeline.transform(batch)
        actual_bars = all_bars[phase_offset:]
        actual_features = all_features[phase_offset:]
        signals = self._generate_signals(application, actual_bars, actual_features)
        result = application.backtest_engine.run(
            BacktestRequest(
                bars=actual_bars,
                features=actual_features,
                signals=signals,
                initial_capital=settings.backtest.initial_capital,
                risk_per_trade_fraction=settings.risk.max_risk_per_trade,
                max_leverage=settings.risk.max_leverage,
                signal_cooldown_bars=settings.strategy.signal_cooldown_bars,
                exit_zscore_threshold=settings.strategy.exit_zscore,
                session_close_hour_utc=settings.strategy.session_close_hour_utc,
                session_close_minute_utc=settings.strategy.session_close_minute_utc,
                intrabar_exit_policy=settings.backtest.intrabar_exit_policy,
            )
        )
        trade_frame = self._trades_to_frame(result.trade_records)
        metrics = self._build_metrics_from_result(result=result, trade_frame=trade_frame)
        actual_frame = full_frame.iloc[start_idx:end_idx]
        return VariantExecution(
            label=label,
            settings=settings,
            result=result,
            trade_frame=trade_frame,
            metrics=metrics,
            bars=len(actual_frame),
            start=actual_frame.index[0].isoformat() if not actual_frame.empty else None,
            end=actual_frame.index[-1].isoformat() if not actual_frame.empty else None,
        )

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

    def _build_metrics_from_result(self, *, result: Any, trade_frame: pd.DataFrame) -> dict[str, Any]:
        wins = trade_frame.loc[trade_frame["net_pnl"] > 0.0, "net_pnl"] if not trade_frame.empty else pd.Series(dtype=float)
        losses = trade_frame.loc[trade_frame["net_pnl"] <= 0.0, "net_pnl"] if not trade_frame.empty else pd.Series(dtype=float)
        average_win = float(wins.mean()) if not wins.empty else 0.0
        average_loss = float(losses.mean()) if not losses.empty else 0.0
        payoff = average_win / abs(average_loss) if average_loss < 0.0 else 0.0
        expectancy = float(trade_frame["net_pnl"].mean()) if not trade_frame.empty else 0.0

        return {
            "number_of_trades": int(result.trades),
            "gross_pnl": float(trade_frame["gross_pnl"].sum()) if not trade_frame.empty else 0.0,
            "net_pnl": float(trade_frame["net_pnl"].sum()) if not trade_frame.empty else 0.0,
            "fees_paid": float(trade_frame["fees_paid"].sum()) if not trade_frame.empty else 0.0,
            "win_rate": float(result.win_rate),
            "average_win": average_win,
            "average_loss": average_loss,
            "payoff": payoff,
            "expectancy": expectancy,
            "max_drawdown": float(result.max_drawdown),
            "sharpe": float(result.sharpe),
            "sortino": float(result.sortino),
            "calmar": float(result.calmar),
            "total_return": float(result.total_return),
        }

    def _build_walk_forward_summary(
        self,
        test_rows: pd.DataFrame,
        combined_test_trades: pd.DataFrame,
    ) -> dict[str, Any]:
        if test_rows.empty:
            return {
                "test_windows": 0,
                "positive_test_windows": 0,
                "positive_test_window_ratio": 0.0,
                "total_test_trades": 0,
                "total_test_gross_pnl": 0.0,
                "total_test_net_pnl": 0.0,
                "total_test_fees_paid": 0.0,
                "mean_test_total_return": 0.0,
                "worst_test_drawdown": 0.0,
                "mean_test_sharpe": 0.0,
                "combined_trade_metrics": self._empty_trade_summary(),
            }

        return {
            "test_windows": int(len(test_rows)),
            "positive_test_windows": int((test_rows["net_pnl"] > 0.0).sum()),
            "positive_test_window_ratio": float((test_rows["net_pnl"] > 0.0).mean()),
            "total_test_trades": int(test_rows["number_of_trades"].sum()),
            "total_test_gross_pnl": float(test_rows["gross_pnl"].sum()),
            "total_test_net_pnl": float(test_rows["net_pnl"].sum()),
            "total_test_fees_paid": float(test_rows["fees_paid"].sum()),
            "mean_test_total_return": float(test_rows["total_return"].mean()),
            "worst_test_drawdown": float(test_rows["max_drawdown"].max()),
            "mean_test_sharpe": float(test_rows["sharpe"].mean()),
            "combined_trade_metrics": self._trade_frame_summary(combined_test_trades),
        }

    def _build_temporal_block_summary(
        self,
        results: pd.DataFrame,
        trade_frames: Sequence[pd.DataFrame],
    ) -> dict[str, Any]:
        non_empty_trade_frames = [trade_frame for trade_frame in trade_frames if not trade_frame.empty]
        combined_trades = pd.concat(non_empty_trade_frames, ignore_index=True) if non_empty_trade_frames else pd.DataFrame()
        if results.empty:
            return {
                "blocks": 0,
                "positive_blocks": 0,
                "positive_block_ratio": 0.0,
                "worst_block_drawdown": 0.0,
                "mean_block_sharpe": 0.0,
                "combined_trade_metrics": self._empty_trade_summary(),
            }

        return {
            "blocks": int(len(results)),
            "positive_blocks": int((results["net_pnl"] > 0.0).sum()),
            "positive_block_ratio": float((results["net_pnl"] > 0.0).mean()),
            "worst_block_drawdown": float(results["max_drawdown"].max()),
            "mean_block_sharpe": float(results["sharpe"].mean()),
            "combined_trade_metrics": self._trade_frame_summary(combined_trades),
        }

    def _trade_frame_summary(self, trade_frame: pd.DataFrame) -> dict[str, Any]:
        if trade_frame.empty:
            return self._empty_trade_summary()

        wins = trade_frame.loc[trade_frame["net_pnl"] > 0.0, "net_pnl"]
        losses = trade_frame.loc[trade_frame["net_pnl"] <= 0.0, "net_pnl"]
        average_win = float(wins.mean()) if not wins.empty else 0.0
        average_loss = float(losses.mean()) if not losses.empty else 0.0
        payoff = average_win / abs(average_loss) if average_loss < 0.0 else 0.0
        expectancy = float(trade_frame["net_pnl"].mean()) if not trade_frame.empty else 0.0
        return {
            "number_of_trades": int(len(trade_frame)),
            "gross_pnl": float(trade_frame["gross_pnl"].sum()),
            "net_pnl": float(trade_frame["net_pnl"].sum()),
            "fees_paid": float(trade_frame["fees_paid"].sum()),
            "win_rate": float((trade_frame["net_pnl"] > 0.0).mean()),
            "average_win": average_win,
            "average_loss": average_loss,
            "payoff": payoff,
            "expectancy": expectancy,
        }

    def _empty_trade_summary(self) -> dict[str, Any]:
        return {
            "number_of_trades": 0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "fees_paid": 0.0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "payoff": 0.0,
            "expectancy": 0.0,
        }

    def _classify_robustness(
        self,
        *,
        walk_forward_summary: dict[str, Any],
        temporal_block_summary: dict[str, Any],
        monte_carlo_summary: dict[str, Any],
        cost_sensitivity_summary: dict[str, Any],
        limitations: Sequence[str],
    ) -> dict[str, Any]:
        decision = self.settings.validation.decision
        go_checks = {
            "total_test_trades": walk_forward_summary["total_test_trades"] >= decision.min_total_test_trades,
            "positive_test_window_ratio": walk_forward_summary["positive_test_window_ratio"] >= decision.min_positive_test_window_ratio_go,
            "positive_temporal_block_ratio": temporal_block_summary["positive_block_ratio"] >= decision.min_positive_temporal_block_ratio_go,
            "mean_test_total_return": walk_forward_summary["mean_test_total_return"] >= decision.min_aggregated_test_total_return_go,
            "worst_test_drawdown": walk_forward_summary["worst_test_drawdown"] <= decision.max_test_drawdown_go,
            "mean_test_sharpe": walk_forward_summary["mean_test_sharpe"] >= decision.min_test_sharpe_go,
            "monte_carlo_drawdown_p95": monte_carlo_summary["max_drawdown"]["p95"] <= decision.max_monte_carlo_drawdown_p95_go,
            "cost_survival_ratio": cost_sensitivity_summary["survival_ratio"] >= decision.min_cost_survival_ratio_go,
        }
        caution_checks = {
            "total_test_trades": walk_forward_summary["total_test_trades"] >= decision.min_total_test_trades,
            "positive_test_window_ratio": walk_forward_summary["positive_test_window_ratio"] >= decision.min_positive_test_window_ratio_caution,
            "positive_temporal_block_ratio": temporal_block_summary["positive_block_ratio"] >= decision.min_positive_temporal_block_ratio_caution,
            "mean_test_total_return": walk_forward_summary["mean_test_total_return"] >= decision.min_aggregated_test_total_return_caution,
            "worst_test_drawdown": walk_forward_summary["worst_test_drawdown"] <= decision.max_test_drawdown_caution,
            "mean_test_sharpe": walk_forward_summary["mean_test_sharpe"] >= decision.min_test_sharpe_caution,
            "monte_carlo_drawdown_p95": monte_carlo_summary["max_drawdown"]["p95"] <= decision.max_monte_carlo_drawdown_p95_caution,
            "cost_survival_ratio": cost_sensitivity_summary["survival_ratio"] >= decision.min_cost_survival_ratio_caution,
        }

        if all(go_checks.values()) and not limitations:
            classification = "GO"
        elif all(caution_checks.values()):
            classification = "GO WITH CAUTION"
        else:
            classification = "NO-GO"

        failed_go_checks = [name for name, passed in go_checks.items() if not passed]
        failed_caution_checks = [name for name, passed in caution_checks.items() if not passed]
        passed_go_checks = [name for name, passed in go_checks.items() if passed]
        passed_caution_checks = [name for name, passed in caution_checks.items() if passed]
        rationale = self._decision_rationale(
            classification=classification,
            go_checks=go_checks,
            caution_checks=caution_checks,
            walk_forward_summary=walk_forward_summary,
            temporal_block_summary=temporal_block_summary,
            monte_carlo_summary=monte_carlo_summary,
            cost_sensitivity_summary=cost_sensitivity_summary,
            limitations=limitations,
        )
        return {
            "classification": classification,
            "go_checks": go_checks,
            "caution_checks": caution_checks,
            "passed_go_checks": passed_go_checks,
            "failed_go_checks": failed_go_checks,
            "passed_caution_checks": passed_caution_checks,
            "failed_caution_checks": failed_caution_checks,
            "rationale": rationale,
        }

    def _decision_rationale(
        self,
        *,
        classification: str,
        go_checks: dict[str, bool],
        caution_checks: dict[str, bool],
        walk_forward_summary: dict[str, Any],
        temporal_block_summary: dict[str, Any],
        monte_carlo_summary: dict[str, Any],
        cost_sensitivity_summary: dict[str, Any],
        limitations: Sequence[str],
    ) -> list[str]:
        lines: list[str] = []
        caution_failures = [name for name, passed in caution_checks.items() if not passed]
        go_failures = [name for name, passed in go_checks.items() if not passed]

        if caution_failures:
            details = "At least one minimum robustness threshold is still missing."
            if {"total_test_trades", "mean_test_sharpe"} & set(caution_failures):
                details = (
                    f"Walk-forward only produced {walk_forward_summary['total_test_trades']} test trades "
                    f"with a mean test Sharpe of {walk_forward_summary['mean_test_sharpe']:.4f}."
                )
            lines.append("Caution gate failures: " + ", ".join(caution_failures) + f". {details}")
        else:
            lines.append("Caution thresholds pass overall, but the baseline still needs careful reading of the sample limits.")

        lines.append(
            "Temporal stability remains weak: "
            f"{walk_forward_summary['positive_test_windows']}/{walk_forward_summary['test_windows']} walk-forward test windows "
            f"and {temporal_block_summary['positive_blocks']}/{temporal_block_summary['blocks']} temporal blocks were net positive."
        )
        lines.append(
            "Path risk looks contained: Monte Carlo max drawdown p95 is "
            f"{monte_carlo_summary['max_drawdown']['p95']:.4f}."
        )
        lines.append(
            "Cost stress is mixed rather than catastrophic: "
            f"{cost_sensitivity_summary['positive_scenarios']}/{cost_sensitivity_summary['scenario_count']} stress scenarios stayed non-negative."
        )
        if limitations:
            lines.append("Dataset limitations reduce confidence: " + " ".join(limitations))
        if classification == "GO":
            lines.append(
                "GO means the temporal evidence, trade count and stress behavior are strong enough to justify the next research step."
            )
        elif classification == "GO WITH CAUTION":
            lines.append(
                "GO WITH CAUTION means the baseline is promising, but it still needs more out-of-sample evidence before returning to PPO."
            )
        else:
            lines.append(
                "NO-GO means the baseline is not robust enough yet to move up the stack; the next sprint should stay on validation or light baseline refinement."
            )
        if go_failures:
            lines.append("GO gate failures: " + ", ".join(go_failures) + ".")
        return lines

    def _build_summary_markdown(self, report: dict[str, Any]) -> str:
        dataset = report["dataset"]
        full_metrics = report["full_dataset"]["metrics"]
        walk_forward = report["walk_forward"]["summary"]
        temporal = report["temporal_blocks"]["summary"]
        monte_carlo = report["monte_carlo"]
        cost_summary = report["cost_sensitivity"]["summary"]
        decision = report["decision"]

        lines = [
            "# Robustness Summary",
            "",
            "## Direct Answer",
            f"- Classification: `{decision['classification']}`",
            "",
            "## Dataset Scope",
            f"- Variant: `{report['variant']}`",
            f"- Symbol: `{dataset['symbol']}`",
            f"- Period: `{dataset['start']}` -> `{dataset['end']}`",
            f"- Bars: `{dataset['bars']}`",
        ]
        if report["limitations"]:
            lines.extend(["", "## Limitations"])
            for limitation in report["limitations"]:
                lines.append(f"- {limitation}")

        lines.extend(
            [
                "",
                "## Full Dataset Snapshot",
                f"- Trades: `{full_metrics['number_of_trades']}`",
                f"- Net PnL: `{full_metrics['net_pnl']:.2f}`",
                f"- Gross PnL: `{full_metrics['gross_pnl']:.2f}`",
                f"- Fees paid: `{full_metrics['fees_paid']:.2f}`",
                f"- Max drawdown: `{full_metrics['max_drawdown']:.4f}`",
                f"- Sharpe: `{full_metrics['sharpe']:.4f}`",
            ]
        )

        lines.extend(
            [
                "",
                "## Walk-Forward",
                f"- Test windows: `{walk_forward['test_windows']}`",
                f"- Positive test windows: `{walk_forward['positive_test_windows']}` / `{walk_forward['test_windows']}`",
                f"- Positive test window ratio: `{walk_forward['positive_test_window_ratio'] * 100:.2f}%`",
                f"- Total test trades: `{walk_forward['total_test_trades']}`",
                f"- Total test net PnL: `{walk_forward['total_test_net_pnl']:.2f}`",
                f"- Mean test total return: `{walk_forward['mean_test_total_return'] * 100:.2f}%`",
                f"- Worst test drawdown: `{walk_forward['worst_test_drawdown']:.4f}`",
                f"- Mean test Sharpe: `{walk_forward['mean_test_sharpe']:.4f}`",
            ]
        )

        lines.extend(
            [
                "",
                "## Temporal Blocks",
                f"- Blocks: `{temporal['blocks']}`",
                f"- Positive block ratio: `{temporal['positive_block_ratio'] * 100:.2f}%`",
                f"- Worst block drawdown: `{temporal['worst_block_drawdown']:.4f}`",
                f"- Mean block Sharpe: `{temporal['mean_block_sharpe']:.4f}`",
            ]
        )

        lines.extend(
            [
                "",
                "## Monte Carlo",
                f"- Simulations: `{monte_carlo['simulations']}`",
                f"- Max drawdown p95: `{monte_carlo['max_drawdown']['p95']:.4f}`",
                f"- Equity final p5/p50/p95: `{monte_carlo['equity_final']['p5']:.2f}` / `{monte_carlo['equity_final']['p50']:.2f}` / `{monte_carlo['equity_final']['p95']:.2f}`",
                f"- Note: {monte_carlo['note']}",
            ]
        )

        lines.extend(
            [
                "",
                "## Cost Sensitivity",
                f"- Positive scenarios: `{cost_summary['positive_scenarios']}` / `{cost_summary['scenario_count']}`",
                f"- Survival ratio: `{cost_summary['survival_ratio'] * 100:.2f}%`",
                f"- Worst stressed net PnL: `{cost_summary['worst_case_net_pnl']:.2f}`",
                f"- Worst stressed drawdown: `{cost_summary['worst_case_drawdown']:.4f}`",
                f"- Note: {cost_summary['note']}",
            ]
        )

        lines.extend(
            [
                "",
                "## Decision Checks",
                f"- Failed caution checks: `{', '.join(decision['failed_caution_checks']) or 'none'}`",
                f"- Failed GO checks: `{', '.join(decision['failed_go_checks']) or 'none'}`",
            ]
        )

        lines.extend(["", "## Decision Logic"])
        for reason in decision["rationale"]:
            lines.append(f"- {reason}")
        return "\n".join(lines) + "\n"

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

    def _percentile_summary(self, values: Sequence[float]) -> dict[str, float]:
        series = pd.Series(list(values), dtype=float)
        return {
            "p5": float(series.quantile(0.05)),
            "p50": float(series.quantile(0.50)),
            "p95": float(series.quantile(0.95)),
            "mean": float(series.mean()),
        }

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._sanitize_value(inner) for key, inner in value.items()}
        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, float):
            return None if not math.isfinite(value) else value
        return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run robust validation on a baseline variant.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--input-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--variant", default="baseline_v3")
    parser.add_argument("--symbol")
    parser.add_argument("--interval")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--allow-gaps", action="store_true")
    parser.add_argument("--artifact-suffix")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = load_settings(args.config_dir)
    runner = RobustnessValidationRunner.from_config(args.config_dir, variant_name=args.variant)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(settings.storage.artifacts_dir) / f"{args.variant}-robustness"
    )
    start = _parse_datetime(args.start) if args.start else None
    end = _parse_datetime(args.end) if args.end else None

    if args.input_path:
        frame = _read_input_frame(args.input_path)
        artifacts = runner.run(
            output_dir=output_dir,
            input_frame=frame,
            start=start,
            end=end,
            allow_gaps=args.allow_gaps,
            artifact_suffix=args.artifact_suffix,
        )
    else:
        request_start = start or _parse_datetime(settings.data.default_start)
        request_end = end or _resolve_end_datetime(settings)
        request = DownloadRequest(
            symbol=args.symbol or settings.market.symbol,
            interval=args.interval or settings.market.execution_timeframe,
            start=request_start,
            end=request_end,
            limit=settings.data.request_limit,
        )
        artifacts = runner.run(
            output_dir=output_dir,
            request=request,
            allow_gaps=args.allow_gaps or settings.data.allow_gaps,
            artifact_suffix=args.artifact_suffix,
        )

    decision = artifacts.report["decision"]
    print(f"classification={decision['classification']}")
    print(f"robustness_report={artifacts.report_path}")
    print(f"robustness_summary={artifacts.summary_path}")
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


def _resolve_end_datetime(settings: Settings) -> datetime | None:
    if settings.data.default_end:
        return _parse_datetime(settings.data.default_end)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
