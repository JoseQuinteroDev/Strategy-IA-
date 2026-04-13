from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from hybrid_quant.bootstrap import TradingApplication, build_application
from hybrid_quant.core import BacktestRequest, BacktestResult, FeatureSnapshot, MarketBar, PortfolioState, SignalSide, StrategySignal
from hybrid_quant.execution import PortfolioSimulator, is_within_session, signal_has_executable_levels

from .variants import build_variant_application


@dataclass(slots=True)
class BaselineArtifactSet:
    artifact_dir: Path
    report: dict[str, Any]
    ohlcv_frame: pd.DataFrame
    feature_frame: pd.DataFrame
    signal_frame: pd.DataFrame
    trade_frame: pd.DataFrame
    bars: tuple[MarketBar, ...]
    features: tuple[FeatureSnapshot, ...]
    signals: tuple[StrategySignal, ...]
    baseline_replay: BacktestResult
    timeframe_step: pd.Timedelta


@dataclass(slots=True)
class BaselineDiagnosticsArtifacts:
    output_dir: Path
    diagnostics_path: Path
    summary_path: Path
    yearly_breakdown_path: Path
    quarterly_breakdown_path: Path
    monthly_breakdown_path: Path
    hourly_breakdown_path: Path
    exit_reason_breakdown_path: Path
    side_breakdown_path: Path
    cost_impact_path: Path
    variant_comparison_path: Path
    risk_execution_breakdown_path: Path
    yearly_equity_curve_path: Path
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class ComparisonVariant:
    name: str
    description: str
    fee_bps: float | None = None
    slippage_bps: float | None = None
    side_filter: SignalSide | None = None
    disable_time_stop: bool = False
    disable_session_close: bool = False
    target_scale: float | None = None


@dataclass(slots=True)
class RiskReplaySummary:
    rows: pd.DataFrame
    actionable_signals: int
    approved_signals: int
    blocked_signals: int
    blocked_by_reason: dict[str, int]
    approved_by_side: dict[str, int]
    blocked_by_side: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "actionable_signals": self.actionable_signals,
            "approved_signals": self.approved_signals,
            "blocked_signals": self.blocked_signals,
            "approval_rate": self.approved_signals / self.actionable_signals if self.actionable_signals else 0.0,
            "blocked_by_reason": self.blocked_by_reason,
            "approved_by_side": self.approved_by_side,
            "blocked_by_side": self.blocked_by_side,
        }


class BaselineDiagnosticsRunner:
    def __init__(self, application: TradingApplication) -> None:
        self.application = application

    @classmethod
    def from_config(
        cls,
        config_dir: str | Path,
        variant_name: str | None = None,
    ) -> "BaselineDiagnosticsRunner":
        application = (
            build_variant_application(config_dir, variant_name)
            if variant_name is not None
            else build_application(config_dir)
        )
        return cls(application)

    def run(
        self,
        *,
        artifact_dir: str | Path,
        output_dir: str | Path | None = None,
        include_variants: bool = True,
        include_risk_replay: bool = True,
    ) -> BaselineDiagnosticsArtifacts:
        artifact_path = Path(artifact_dir)
        output_path = Path(output_dir) if output_dir is not None else artifact_path / "diagnostics"
        output_path.mkdir(parents=True, exist_ok=True)

        artifact_set = self._load_artifact_set(artifact_path)
        enriched_trades = self._build_enriched_trade_frame(artifact_set)
        yearly_breakdown = self._build_trade_breakdown(enriched_trades, "exit_year")
        quarterly_breakdown = self._build_quarterly_breakdown(enriched_trades, artifact_set.baseline_replay)
        monthly_breakdown = self._build_monthly_breakdown(enriched_trades, artifact_set.baseline_replay)
        weekday_breakdown = self._build_trade_breakdown(enriched_trades, "signal_weekday")
        hourly_breakdown = self._build_trade_breakdown(enriched_trades, "signal_hour_utc")
        exit_reason_breakdown = self._build_trade_breakdown(enriched_trades, "exit_reason")
        side_breakdown = self._build_trade_breakdown(enriched_trades, "side")
        entry_mode_breakdown = self._build_trade_breakdown(enriched_trades, "entry_mode")
        entry_trigger_breakdown = self._build_trade_breakdown(enriched_trades, "entry_trigger")
        regime_breakdown = self._build_trade_breakdown(enriched_trades, "regime")
        duration_breakdown = self._build_duration_breakdown(enriched_trades)
        first_breakout_breakdown = self._build_trade_breakdown(enriched_trades, "breakout_order_bucket")
        zscore_breakdown = self._build_bucket_breakdown(
            enriched_trades,
            column="abs_entry_zscore",
            bins=[0.0, 2.0, 2.5, 3.0, math.inf],
            labels=["<=2.0", "2.0-2.5", "2.5-3.0", ">3.0"],
            bucket_name="zscore_bucket",
        )
        stop_target_breakdown = self._build_stop_target_breakdown(enriched_trades)
        breakout_distance_breakdown = self._build_bucket_breakdown(
            enriched_trades,
            column="breakout_distance_atr",
            bins=[0.0, 0.10, 0.25, 0.50, math.inf],
            labels=["<=0.10 ATR", "0.10-0.25 ATR", "0.25-0.50 ATR", ">0.50 ATR"],
            bucket_name="breakout_distance_bucket",
        )
        momentum_breakdown = self._build_bucket_breakdown(
            enriched_trades,
            column="abs_momentum",
            bins=[0.0, 0.0025, 0.0050, 0.0100, math.inf],
            labels=["<=0.25%", "0.25-0.50%", "0.50-1.00%", ">1.00%"],
            bucket_name="momentum_bucket",
        )
        breakout_range_breakdown = self._build_bucket_breakdown(
            enriched_trades,
            column="breakout_range_width_atr",
            bins=[0.0, 1.0, 1.5, 2.0, math.inf],
            labels=["<=1.0 ATR", "1.0-1.5 ATR", "1.5-2.0 ATR", ">2.0 ATR"],
            bucket_name="breakout_range_bucket",
        )
        opening_range_width_breakdown = self._build_bucket_breakdown(
            enriched_trades,
            column="opening_range_width_atr",
            bins=[0.0, 0.5, 1.0, 1.5, math.inf],
            labels=["<=0.5 ATR", "0.5-1.0 ATR", "1.0-1.5 ATR", ">1.5 ATR"],
            bucket_name="opening_range_width_bucket",
        )
        target_to_cost_breakdown = self._build_bucket_breakdown(
            enriched_trades,
            column="target_to_cost_ratio",
            bins=[0.0, 1.0, 2.0, 3.0, math.inf],
            labels=["<=1.0x", "1.0-2.0x", "2.0-3.0x", ">3.0x"],
            bucket_name="target_to_cost_bucket",
        )
        signal_side_breakdown = self._build_signal_side_breakdown(artifact_set, enriched_trades)
        risk_replay = self._replay_risk_engine(artifact_set) if include_risk_replay else None
        risk_execution_breakdown = self._build_risk_execution_breakdown(artifact_set, risk_replay)
        variant_frame = self._run_variant_suite(artifact_set) if include_variants else pd.DataFrame()
        cost_impact = self._build_cost_impact(enriched_trades, variant_frame)
        yearly_equity_curve = self._build_yearly_equity_curve(artifact_set.baseline_replay)
        overall_metrics = self._build_overall_metrics(
            enriched_trades=enriched_trades,
            replay_result=artifact_set.baseline_replay,
            yearly_breakdown=yearly_breakdown,
            monthly_breakdown=monthly_breakdown,
            cost_impact=cost_impact,
        )
        automatic_conclusion = self._build_automatic_conclusion(
            overall_metrics=overall_metrics,
            enriched_trades=enriched_trades,
            monthly_breakdown=monthly_breakdown,
            hourly_breakdown=hourly_breakdown,
            side_breakdown=side_breakdown,
            regime_breakdown=regime_breakdown,
            exit_reason_breakdown=exit_reason_breakdown,
            variant_frame=variant_frame,
        )

        diagnostics = {
            "artifact_dir": str(artifact_set.artifact_dir),
            "question_answer": automatic_conclusion["short_answer"],
            "baseline_metrics": overall_metrics,
            "report_snapshot": artifact_set.report,
            "breakdowns": {
                "yearly": yearly_breakdown.to_dict(orient="records"),
                "quarterly": quarterly_breakdown.to_dict(orient="records"),
                "monthly": monthly_breakdown.to_dict(orient="records"),
                "weekday": weekday_breakdown.to_dict(orient="records"),
                "hourly": hourly_breakdown.to_dict(orient="records"),
                "exit_reason": exit_reason_breakdown.to_dict(orient="records"),
                "side": side_breakdown.to_dict(orient="records"),
                "entry_mode": entry_mode_breakdown.to_dict(orient="records"),
                "entry_trigger": entry_trigger_breakdown.to_dict(orient="records"),
                "regime": regime_breakdown.to_dict(orient="records"),
                "duration": duration_breakdown.to_dict(orient="records"),
                "first_breakout": first_breakout_breakdown.to_dict(orient="records"),
                "zscore": zscore_breakdown.to_dict(orient="records"),
                "stop_target": stop_target_breakdown.to_dict(orient="records"),
                "breakout_distance": breakout_distance_breakdown.to_dict(orient="records"),
                "momentum": momentum_breakdown.to_dict(orient="records"),
                "breakout_range": breakout_range_breakdown.to_dict(orient="records"),
                "opening_range_width": opening_range_width_breakdown.to_dict(orient="records"),
                "target_to_cost": target_to_cost_breakdown.to_dict(orient="records"),
                "signal_side": signal_side_breakdown.to_dict(orient="records"),
                "risk_execution": risk_execution_breakdown.to_dict(orient="records"),
            },
            "risk_replay": risk_replay.to_dict() if risk_replay is not None else None,
            "variants": variant_frame.to_dict(orient="records"),
            "cost_impact": cost_impact.to_dict(orient="records"),
            "yearly_equity_curve": yearly_equity_curve.to_dict(orient="records"),
            "automatic_conclusion": automatic_conclusion,
        }

        diagnostics_path = output_path / "diagnostics.json"
        summary_path = output_path / "diagnostics_summary.md"
        yearly_breakdown_path = output_path / "yearly_breakdown.csv"
        quarterly_breakdown_path = output_path / "quarterly_breakdown.csv"
        monthly_breakdown_path = output_path / "monthly_breakdown.csv"
        hourly_breakdown_path = output_path / "hourly_breakdown.csv"
        exit_reason_breakdown_path = output_path / "exit_reason_breakdown.csv"
        side_breakdown_path = output_path / "side_breakdown.csv"
        cost_impact_path = output_path / "cost_impact.csv"
        variant_comparison_path = output_path / "variant_comparison.csv"
        risk_execution_breakdown_path = output_path / "risk_execution_breakdown.csv"
        yearly_equity_curve_path = output_path / "yearly_equity_curve.csv"

        diagnostics_path.write_text(json.dumps(self._sanitize_value(diagnostics), indent=2), encoding="utf-8")
        summary_path.write_text(
            self._build_summary_markdown(
                diagnostics=diagnostics,
                overall_metrics=overall_metrics,
                yearly_breakdown=yearly_breakdown,
                monthly_breakdown=monthly_breakdown,
                hourly_breakdown=hourly_breakdown,
                variant_frame=variant_frame,
                signal_side_breakdown=signal_side_breakdown,
            ),
            encoding="utf-8",
        )

        yearly_breakdown.to_csv(yearly_breakdown_path, index=False)
        quarterly_breakdown.to_csv(quarterly_breakdown_path, index=False)
        monthly_breakdown.to_csv(monthly_breakdown_path, index=False)
        hourly_breakdown.to_csv(hourly_breakdown_path, index=False)
        exit_reason_breakdown.to_csv(exit_reason_breakdown_path, index=False)
        side_breakdown.to_csv(side_breakdown_path, index=False)
        cost_impact.to_csv(cost_impact_path, index=False)
        variant_frame.to_csv(variant_comparison_path, index=False)
        risk_execution_breakdown.to_csv(risk_execution_breakdown_path, index=False)
        yearly_equity_curve.to_csv(yearly_equity_curve_path, index=False)
        weekday_breakdown.to_csv(output_path / "weekday_breakdown.csv", index=False)
        regime_breakdown.to_csv(output_path / "regime_breakdown.csv", index=False)
        signal_side_breakdown.to_csv(output_path / "signal_side_breakdown.csv", index=False)
        entry_mode_breakdown.to_csv(output_path / "entry_mode_breakdown.csv", index=False)
        entry_trigger_breakdown.to_csv(output_path / "entry_trigger_breakdown.csv", index=False)
        duration_breakdown.to_csv(output_path / "duration_breakdown.csv", index=False)
        first_breakout_breakdown.to_csv(output_path / "first_breakout_breakdown.csv", index=False)
        zscore_breakdown.to_csv(output_path / "zscore_breakdown.csv", index=False)
        stop_target_breakdown.to_csv(output_path / "stop_target_breakdown.csv", index=False)
        breakout_distance_breakdown.to_csv(output_path / "breakout_distance_breakdown.csv", index=False)
        momentum_breakdown.to_csv(output_path / "momentum_breakdown.csv", index=False)
        breakout_range_breakdown.to_csv(output_path / "breakout_range_breakdown.csv", index=False)
        opening_range_width_breakdown.to_csv(output_path / "opening_range_width_breakdown.csv", index=False)
        target_to_cost_breakdown.to_csv(output_path / "target_to_cost_breakdown.csv", index=False)
        enriched_trades.to_csv(output_path / "enriched_trades.csv", index=False)

        return BaselineDiagnosticsArtifacts(
            output_dir=output_path,
            diagnostics_path=diagnostics_path,
            summary_path=summary_path,
            yearly_breakdown_path=yearly_breakdown_path,
            quarterly_breakdown_path=quarterly_breakdown_path,
            monthly_breakdown_path=monthly_breakdown_path,
            hourly_breakdown_path=hourly_breakdown_path,
            exit_reason_breakdown_path=exit_reason_breakdown_path,
            side_breakdown_path=side_breakdown_path,
            cost_impact_path=cost_impact_path,
            variant_comparison_path=variant_comparison_path,
            risk_execution_breakdown_path=risk_execution_breakdown_path,
            yearly_equity_curve_path=yearly_equity_curve_path,
            diagnostics=diagnostics,
        )

    def _load_artifact_set(self, artifact_dir: Path) -> BaselineArtifactSet:
        report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
        ohlcv_frame = self._read_indexed_frame(artifact_dir / "ohlcv.csv", "open_time")
        feature_frame = self._read_indexed_frame(artifact_dir / "features.csv", "open_time")
        signal_frame = pd.read_csv(artifact_dir / "signals.csv", parse_dates=["timestamp"], low_memory=False)
        signal_frame["timestamp"] = pd.to_datetime(signal_frame["timestamp"], utc=True)
        trade_frame = pd.read_csv(
            artifact_dir / "trades.csv",
            parse_dates=["entry_timestamp", "exit_timestamp"],
        )
        trade_frame["entry_timestamp"] = pd.to_datetime(trade_frame["entry_timestamp"], utc=True)
        trade_frame["exit_timestamp"] = pd.to_datetime(trade_frame["exit_timestamp"], utc=True)
        for column in [
            "entry_price",
            "stop_price",
            "target_price",
            "strength",
            "quantity",
            "gross_pnl",
            "net_pnl",
            "fees_paid",
            "return_pct",
            "bars_held",
        ]:
            if column in signal_frame.columns:
                signal_frame[column] = pd.to_numeric(signal_frame[column], errors="coerce")
            if column in trade_frame.columns:
                trade_frame[column] = pd.to_numeric(trade_frame[column], errors="coerce")

        bars = tuple(self._frame_to_bars(ohlcv_frame))
        features = tuple(self._frame_to_features(feature_frame))
        signals = tuple(self._frame_to_signals(signal_frame))
        replay = self._run_backtest_replay(bars=bars, features=features, signals=signals)
        step_series = ohlcv_frame.index.to_series().diff().dropna()
        timeframe_step = step_series.median() if not step_series.empty else pd.Timedelta(minutes=5)

        return BaselineArtifactSet(
            artifact_dir=artifact_dir,
            report=report,
            ohlcv_frame=ohlcv_frame,
            feature_frame=feature_frame,
            signal_frame=signal_frame,
            trade_frame=trade_frame,
            bars=bars,
            features=features,
            signals=signals,
            baseline_replay=replay,
            timeframe_step=timeframe_step,
        )

    def _run_backtest_replay(
        self,
        *,
        bars: Sequence[MarketBar],
        features: Sequence[FeatureSnapshot],
        signals: Sequence[StrategySignal],
        fee_bps: float | None = None,
        slippage_bps: float | None = None,
    ) -> BacktestResult:
        engine = replace(
            self.application.backtest_engine,
            fee_bps=self.application.settings.backtest.fee_bps if fee_bps is None else fee_bps,
            slippage_bps=self.application.settings.backtest.slippage_bps if slippage_bps is None else slippage_bps,
        )
        request = BacktestRequest(
            bars=bars,
            features=features,
            signals=signals,
            initial_capital=self.application.settings.backtest.initial_capital,
            risk_per_trade_fraction=self.application.settings.risk.max_risk_per_trade,
            max_leverage=self.application.settings.risk.max_leverage,
            signal_cooldown_bars=self.application.settings.strategy.signal_cooldown_bars,
            exit_zscore_threshold=self.application.settings.strategy.exit_zscore,
            session_close_hour_utc=self.application.settings.strategy.session_close_hour_utc,
            session_close_minute_utc=self.application.settings.strategy.session_close_minute_utc,
            session_close_timezone=self.application.settings.strategy.session_close_timezone,
            session_close_windows=tuple(self.application.settings.strategy.entry_session_windows),
            intrabar_exit_policy=self.application.settings.backtest.intrabar_exit_policy,
            gap_exit_policy=self.application.settings.backtest.gap_exit_policy,
        )
        return engine.run(request)

    def _build_enriched_trade_frame(self, artifact_set: BaselineArtifactSet) -> pd.DataFrame:
        trades = artifact_set.trade_frame.copy()
        if trades.empty:
            return trades

        settings = self.application.settings
        ohlcv_index = artifact_set.ohlcv_frame.index
        position_lookup = pd.Series(range(len(ohlcv_index)), index=ohlcv_index)
        entry_positions = position_lookup.reindex(trades["entry_timestamp"])
        signal_positions = entry_positions - 1
        signal_times = signal_positions.apply(
            lambda value: ohlcv_index[int(value)] if pd.notna(value) and int(value) >= 0 else pd.NaT
        )

        signal_lookup = artifact_set.signal_frame.set_index("timestamp")
        feature_lookup = artifact_set.feature_frame.copy()
        feature_lookup.index = pd.to_datetime(feature_lookup.index, utc=True)
        ohlcv_lookup = artifact_set.ohlcv_frame.copy()
        ohlcv_lookup.index = pd.to_datetime(ohlcv_lookup.index, utc=True)

        joined_signals = signal_lookup.reindex(pd.DatetimeIndex(signal_times)).reset_index(drop=True)
        joined_features = feature_lookup.reindex(pd.DatetimeIndex(signal_times)).reset_index(drop=True)
        joined_prices = ohlcv_lookup.reindex(pd.DatetimeIndex(signal_times)).reset_index(drop=True)

        anchor_column = self._anchor_column_name()
        breakout_window = settings.strategy.breakout_lookback_bars
        momentum_window = settings.strategy.momentum_lookback_bars
        estimated_round_trip_cost_bps = settings.strategy.estimated_round_trip_cost_bps or (
            2.0 * (settings.backtest.fee_bps + settings.backtest.slippage_bps)
        )
        trades["signal_time"] = pd.to_datetime(signal_times.values, utc=True)
        trades["signal_side"] = (
            joined_signals["raw_side"]
            if "raw_side" in joined_signals.columns
            else joined_signals["side"]
        )
        trades["signal_entry_price"] = pd.to_numeric(joined_signals.get("entry_price"), errors="coerce")
        trades["signal_stop_price"] = pd.to_numeric(joined_signals.get("stop_price"), errors="coerce")
        trades["signal_target_price"] = pd.to_numeric(joined_signals.get("target_price"), errors="coerce")
        trades["candidate_signal_side"] = trades["signal_side"].fillna(trades["side"])
        trades["stop_distance"] = (trades["signal_entry_price"] - trades["signal_stop_price"]).abs()
        trades["target_distance"] = (trades["signal_target_price"] - trades["signal_entry_price"]).abs()
        trades["stop_distance_pct"] = trades["stop_distance"] / trades["signal_entry_price"]
        trades["target_distance_pct"] = trades["target_distance"] / trades["signal_entry_price"]
        trades["risk_reward_ratio"] = trades["target_distance"] / trades["stop_distance"]
        trades["strategy_family"] = (
            joined_signals.get("strategy_family")
            if "strategy_family" in joined_signals.columns
            else settings.strategy.family
        )
        if not isinstance(trades["strategy_family"], pd.Series):
            trades["strategy_family"] = pd.Series(settings.strategy.family, index=trades.index, dtype=object)
        trades["strategy_family"] = trades["strategy_family"].fillna(settings.strategy.family)
        trades["signal_variant_name"] = (
            joined_signals.get("variant_name")
            if "variant_name" in joined_signals.columns
            else settings.strategy.variant_name
        )
        if not isinstance(trades["signal_variant_name"], pd.Series):
            trades["signal_variant_name"] = pd.Series(settings.strategy.variant_name, index=trades.index, dtype=object)
        trades["signal_variant_name"] = trades["signal_variant_name"].fillna(settings.strategy.variant_name)
        trades["entry_mode"] = (
            joined_signals.get("entry_mode")
            if "entry_mode" in joined_signals.columns
            else settings.strategy.entry_mode
        )
        if not isinstance(trades["entry_mode"], pd.Series):
            trades["entry_mode"] = pd.Series(settings.strategy.entry_mode, index=trades.index, dtype=object)
        trades["entry_mode"] = trades["entry_mode"].fillna(settings.strategy.entry_mode)
        trades["entry_trigger"] = (
            joined_signals.get("entry_trigger")
            if "entry_trigger" in joined_signals.columns
            else trades["entry_mode"]
        )
        if not isinstance(trades["entry_trigger"], pd.Series):
            trades["entry_trigger"] = pd.Series(index=trades.index, dtype=object)
        trades["entry_trigger"] = trades["entry_trigger"].fillna(trades["entry_mode"])
        trades["entry_zscore"] = pd.to_numeric(joined_features.get("zscore_distance_to_mean"), errors="coerce")
        trades["abs_entry_zscore"] = trades["entry_zscore"].abs()
        trades["adx_1h"] = pd.to_numeric(joined_features.get("adx_1h"), errors="coerce")
        trades["ema_200_1h"] = pd.to_numeric(joined_features.get("ema_200_1h"), errors="coerce")
        trades["atr_14"] = pd.to_numeric(joined_features.get("atr_14"), errors="coerce")
        trades["signal_close"] = pd.to_numeric(joined_prices.get("close"), errors="coerce")
        trades["anchor_name"] = anchor_column or ""
        if anchor_column is None:
            trades["anchor_value"] = float("nan")
            trades["anchor_distance"] = float("nan")
            trades["anchor_distance_pct"] = float("nan")
        else:
            trades["anchor_value"] = pd.to_numeric(joined_features.get(anchor_column), errors="coerce")
            trades["anchor_distance"] = trades["signal_close"] - trades["anchor_value"]
            trades["anchor_distance_pct"] = trades["anchor_distance"] / trades["anchor_value"]

        breakout_high_column = f"breakout_high_{breakout_window}"
        breakout_low_column = f"breakout_low_{breakout_window}"
        breakout_width_column = f"breakout_range_width_{breakout_window}"
        breakout_width_atr_column = f"breakout_range_width_atr_{breakout_window}"
        momentum_column = f"momentum_{momentum_window}"

        trades["breakout_high"] = pd.to_numeric(joined_features.get(breakout_high_column), errors="coerce")
        trades["breakout_low"] = pd.to_numeric(joined_features.get(breakout_low_column), errors="coerce")
        trades["breakout_range_width"] = pd.to_numeric(joined_features.get(breakout_width_column), errors="coerce")
        trades["breakout_range_width_atr"] = pd.to_numeric(
            joined_features.get(breakout_width_atr_column),
            errors="coerce",
        )
        trades["momentum"] = pd.to_numeric(joined_features.get(momentum_column), errors="coerce")
        trades["abs_momentum"] = trades["momentum"].abs()
        trades["candle_range_atr"] = pd.to_numeric(joined_features.get("candle_range_atr"), errors="coerce")
        trades["price_vs_ema_200_1h_pct"] = pd.to_numeric(
            joined_features.get("price_vs_ema_200_1h_pct"),
            errors="coerce",
        )
        trades["ema_200_1h_slope"] = pd.to_numeric(joined_features.get("ema_200_1h_slope"), errors="coerce")
        trades["relative_volume"] = self._coalesce_numeric_series(
            joined_signals, joined_features, "relative_volume"
        )
        trades["opening_range_high"] = self._coalesce_numeric_series(
            joined_signals, joined_features, "opening_range_high"
        )
        trades["opening_range_low"] = self._coalesce_numeric_series(
            joined_signals, joined_features, "opening_range_low"
        )
        trades["opening_range_width"] = self._coalesce_numeric_series(
            joined_signals, joined_features, "opening_range_width"
        )
        trades["opening_range_width_atr"] = self._coalesce_numeric_series(
            joined_signals, joined_features, "opening_range_width_atr"
        )
        trades["opening_range_breakout_count_today"] = self._coalesce_numeric_series(
            joined_signals, joined_features, "opening_range_breakout_count_today"
        )
        trades["first_breakout_of_day"] = (
            self._coalesce_numeric_series(
                joined_signals,
                joined_features,
                "first_breakout_of_day",
                secondary_column="opening_range_first_breakout_of_day",
            )
            .astype(float)
            .fillna(0.0)
        )
        trades["breakout_level"] = pd.to_numeric(joined_signals.get("breakout_level"), errors="coerce")
        missing_breakout_level = trades["breakout_level"].isna()
        long_mask = trades["candidate_signal_side"].astype(str).str.lower() == "long"
        short_mask = trades["candidate_signal_side"].astype(str).str.lower() == "short"
        trades.loc[missing_breakout_level & long_mask, "breakout_level"] = trades.loc[
            missing_breakout_level & long_mask,
            "breakout_high",
        ]
        trades.loc[missing_breakout_level & short_mask, "breakout_level"] = trades.loc[
            missing_breakout_level & short_mask,
            "breakout_low",
        ]
        missing_breakout_level = trades["breakout_level"].isna()
        trades.loc[missing_breakout_level & long_mask, "breakout_level"] = trades.loc[
            missing_breakout_level & long_mask,
            "opening_range_high",
        ]
        trades.loc[missing_breakout_level & short_mask, "breakout_level"] = trades.loc[
            missing_breakout_level & short_mask,
            "opening_range_low",
        ]
        missing_breakout_range = trades["breakout_range_width_atr"].isna()
        trades.loc[missing_breakout_range, "breakout_range_width_atr"] = trades.loc[
            missing_breakout_range,
            "opening_range_width_atr",
        ]
        trades.loc[trades["breakout_range_width"].isna(), "breakout_range_width"] = trades.loc[
            trades["breakout_range_width"].isna(),
            "opening_range_width",
        ]
        trades["breakout_distance"] = float("nan")
        trades.loc[long_mask, "breakout_distance"] = (
            trades.loc[long_mask, "signal_entry_price"] - trades.loc[long_mask, "breakout_level"]
        )
        trades.loc[short_mask, "breakout_distance"] = (
            trades.loc[short_mask, "breakout_level"] - trades.loc[short_mask, "signal_entry_price"]
        )
        trades["breakout_distance_atr"] = trades["breakout_distance"] / trades["atr_14"]
        trades["estimated_round_trip_cost_bps"] = estimated_round_trip_cost_bps
        trades["expected_move_bps"] = (
            trades["target_distance_pct"] * 10000.0
        )
        trades["target_to_cost_ratio"] = pd.to_numeric(
            joined_signals.get("target_to_cost_ratio"),
            errors="coerce",
        )
        missing_target_to_cost = trades["target_to_cost_ratio"].isna()
        if estimated_round_trip_cost_bps > 0.0:
            trades.loc[missing_target_to_cost, "target_to_cost_ratio"] = (
                trades.loc[missing_target_to_cost, "expected_move_bps"] / estimated_round_trip_cost_bps
            )
        trades["signal_hour_utc"] = trades["signal_time"].dt.hour
        trades["entry_hour_utc"] = trades["entry_timestamp"].dt.hour
        trades["signal_weekday"] = trades["signal_time"].dt.day_name()
        trades["entry_weekday"] = trades["entry_timestamp"].dt.day_name()
        trades["signal_year"] = trades["signal_time"].dt.strftime("%Y")
        trades["entry_year"] = trades["entry_timestamp"].dt.strftime("%Y")
        trades["exit_year"] = trades["exit_timestamp"].dt.strftime("%Y")
        trades["signal_quarter"] = (
            trades["signal_time"].dt.year.astype(str) + "-Q" + trades["signal_time"].dt.quarter.astype(str)
        )
        trades["exit_quarter"] = (
            trades["exit_timestamp"].dt.year.astype(str) + "-Q" + trades["exit_timestamp"].dt.quarter.astype(str)
        )
        trades["exit_month"] = trades["exit_timestamp"].dt.strftime("%Y-%m")
        trades["regime"] = trades["adx_1h"].apply(self._regime_label)
        trades["breakout_order_bucket"] = trades["first_breakout_of_day"].apply(
            lambda value: "first_breakout" if float(value) >= 1.0 else "later_breakout"
        )
        mfe_mae = trades.apply(
            lambda row: self._calculate_mfe_mae(
                ohlcv_lookup=ohlcv_lookup,
                entry_timestamp=row["entry_timestamp"],
                exit_timestamp=row["exit_timestamp"],
                side=str(row["side"]),
                entry_price=float(row["entry_price"]),
                atr=float(row["atr_14"]) if pd.notna(row["atr_14"]) else None,
            ),
            axis=1,
            result_type="expand",
        )
        mfe_mae.columns = ["mfe", "mae", "mfe_atr", "mae_atr"]
        trades = pd.concat([trades, mfe_mae], axis=1)
        return trades

    def _build_overall_metrics(
        self,
        *,
        enriched_trades: pd.DataFrame,
        replay_result: BacktestResult,
        yearly_breakdown: pd.DataFrame,
        monthly_breakdown: pd.DataFrame,
        cost_impact: pd.DataFrame,
    ) -> dict[str, Any]:
        wins = enriched_trades.loc[enriched_trades["net_pnl"] > 0.0, "net_pnl"]
        losses = enriched_trades.loc[enriched_trades["net_pnl"] <= 0.0, "net_pnl"]
        gross_profit = float(wins.sum()) if not wins.empty else 0.0
        gross_loss = abs(float(losses.sum())) if not losses.empty else 0.0
        average_win = float(wins.mean()) if not wins.empty else 0.0
        average_loss = float(losses.mean()) if not losses.empty else 0.0
        payoff_real = average_win / abs(average_loss) if average_loss < 0.0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else 0.0
        expectancy = float(enriched_trades["net_pnl"].mean()) if not enriched_trades.empty else 0.0
        profitable_months_pct = (
            float((monthly_breakdown["net_pnl"] > 0.0).mean()) if not monthly_breakdown.empty else 0.0
        )
        profitable_years_pct = (
            float((yearly_breakdown["net_pnl"] > 0.0).mean()) if not yearly_breakdown.empty else 0.0
        )
        break_even_win_rate = 1.0 / (1.0 + payoff_real) if payoff_real > 0.0 else 1.0
        variant_lookup = cost_impact.set_index("variant") if not cost_impact.empty else pd.DataFrame()
        period_days = 0.0
        if replay_result.start is not None and replay_result.end is not None:
            period_days = max(
                (pd.Timestamp(replay_result.end) - pd.Timestamp(replay_result.start)).total_seconds() / 86_400.0,
                0.0,
            )
        period_years = period_days / 365.25 if period_days > 0.0 else 0.0
        period_weeks = period_days / 7.0 if period_days > 0.0 else 0.0

        return {
            "strategy_family": self.application.settings.strategy.family,
            "variant_name": self.application.settings.strategy.variant_name,
            "bars": int(replay_result.metadata.get("bars", 0)),
            "analysis_period_days": period_days,
            "analysis_period_years": period_years,
            "number_of_trades": int(replay_result.trades),
            "trades_per_year": (float(replay_result.trades) / period_years) if period_years > 0.0 else 0.0,
            "trades_per_week_avg": (float(replay_result.trades) / period_weeks) if period_weeks > 0.0 else 0.0,
            "gross_pnl": float(enriched_trades["gross_pnl"].sum()),
            "net_pnl": float(enriched_trades["net_pnl"].sum()),
            "fees_paid_total": float(enriched_trades["fees_paid"].sum()),
            "average_win": average_win,
            "average_loss": average_loss,
            "payoff_real": payoff_real,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "win_rate": float(replay_result.win_rate),
            "break_even_win_rate": break_even_win_rate,
            "break_even_gap": break_even_win_rate - float(replay_result.win_rate),
            "max_drawdown": float(replay_result.max_drawdown),
            "sharpe": float(replay_result.sharpe),
            "sortino": float(replay_result.sortino),
            "calmar": float(replay_result.calmar),
            "max_consecutive_losses": self._max_consecutive_losses(enriched_trades["net_pnl"].tolist()),
            "average_holding_bars": float(enriched_trades["bars_held"].mean()) if not enriched_trades.empty else 0.0,
            "average_mfe_atr": (
                float(enriched_trades["mfe_atr"].dropna().mean())
                if "mfe_atr" in enriched_trades and not enriched_trades["mfe_atr"].dropna().empty
                else 0.0
            ),
              "average_mae_atr": (
                  float(enriched_trades["mae_atr"].dropna().mean())
                  if "mae_atr" in enriched_trades and not enriched_trades["mae_atr"].dropna().empty
                  else 0.0
              ),
              "profitable_years_pct": profitable_years_pct,
              "profitable_months_pct": profitable_months_pct,
            "estimated_fee_drag": self._variant_delta(variant_lookup, "no_fees"),
            "estimated_slippage_drag": self._variant_delta(variant_lookup, "no_slippage"),
            "estimated_total_cost_drag": self._variant_delta(variant_lookup, "no_costs"),
        }

    def _build_monthly_breakdown(self, enriched_trades: pd.DataFrame, replay_result: BacktestResult) -> pd.DataFrame:
        monthly = self._build_trade_breakdown(enriched_trades, "exit_month")
        if monthly.empty:
            return monthly

        equity_curve = pd.DataFrame(replay_result.metadata.get("equity_curve", []))
        if not equity_curve.empty:
            equity_curve["timestamp"] = pd.to_datetime(equity_curve["timestamp"], utc=True)
            equity_curve["month"] = equity_curve["timestamp"].dt.strftime("%Y-%m")
            monthly_dd = equity_curve.groupby("month", observed=False).agg(
                month_drawdown=("equity", lambda series: abs((series / series.cummax() - 1.0).min()))
            ).reset_index()
            month_return = (
                equity_curve.groupby("month", observed=False)["equity"]
                .agg(month_start_equity="first", month_end_equity="last")
                .reset_index()
            )
            month_return["month_return"] = (
                month_return["month_end_equity"] / month_return["month_start_equity"] - 1.0
            )
            monthly = monthly.merge(monthly_dd, left_on="exit_month", right_on="month", how="left").drop(
                columns=["month"]
            )
            monthly = monthly.merge(month_return, left_on="exit_month", right_on="month", how="left").drop(
                columns=["month"]
            )

        monthly["profitable_month"] = monthly["net_pnl"] > 0.0
        return monthly.sort_values("exit_month").reset_index(drop=True)

    def _build_quarterly_breakdown(self, enriched_trades: pd.DataFrame, replay_result: BacktestResult) -> pd.DataFrame:
        quarterly = self._build_trade_breakdown(enriched_trades, "exit_quarter")
        if quarterly.empty:
            return quarterly

        equity_curve = pd.DataFrame(replay_result.metadata.get("equity_curve", []))
        if not equity_curve.empty:
            equity_curve["timestamp"] = pd.to_datetime(equity_curve["timestamp"], utc=True)
            equity_curve["quarter"] = (
                equity_curve["timestamp"].dt.year.astype(str)
                + "-Q"
                + equity_curve["timestamp"].dt.quarter.astype(str)
            )
            quarter_dd = equity_curve.groupby("quarter", observed=False).agg(
                quarter_drawdown=("equity", lambda series: abs((series / series.cummax() - 1.0).min()))
            ).reset_index()
            quarter_return = (
                equity_curve.groupby("quarter", observed=False)["equity"]
                .agg(quarter_start_equity="first", quarter_end_equity="last")
                .reset_index()
            )
            quarter_return["quarter_return"] = (
                quarter_return["quarter_end_equity"] / quarter_return["quarter_start_equity"] - 1.0
            )
            quarterly = quarterly.merge(quarter_dd, left_on="exit_quarter", right_on="quarter", how="left").drop(
                columns=["quarter"]
            )
            quarterly = quarterly.merge(
                quarter_return,
                left_on="exit_quarter",
                right_on="quarter",
                how="left",
            ).drop(columns=["quarter"])

        quarterly["profitable_quarter"] = quarterly["net_pnl"] > 0.0
        return quarterly.sort_values("exit_quarter").reset_index(drop=True)

    def _build_yearly_equity_curve(self, replay_result: BacktestResult) -> pd.DataFrame:
        equity_curve = pd.DataFrame(replay_result.metadata.get("equity_curve", []))
        if equity_curve.empty:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "year",
                    "equity",
                    "year_start_equity",
                    "normalized_equity",
                    "year_drawdown",
                ]
            )

        equity_curve["timestamp"] = pd.to_datetime(equity_curve["timestamp"], utc=True)
        equity_curve = equity_curve.sort_values("timestamp").reset_index(drop=True)
        equity_curve["year"] = equity_curve["timestamp"].dt.strftime("%Y")
        equity_curve["year_start_equity"] = equity_curve.groupby("year", observed=False)["equity"].transform("first")
        equity_curve["normalized_equity"] = equity_curve["equity"] / equity_curve["year_start_equity"]
        equity_curve["year_drawdown"] = equity_curve.groupby("year", observed=False)["equity"].transform(
            lambda series: abs(series / series.cummax() - 1.0)
        )
        return equity_curve[
            ["timestamp", "year", "equity", "year_start_equity", "normalized_equity", "year_drawdown"]
        ]

    def _build_trade_breakdown(self, enriched_trades: pd.DataFrame, column: str) -> pd.DataFrame:
        if enriched_trades.empty or column not in enriched_trades.columns:
            return pd.DataFrame(
                columns=[
                    column,
                    "trades",
                    "gross_pnl",
                    "net_pnl",
                    "win_rate",
                    "average_win",
                    "average_loss",
                    "payoff_real",
                    "profit_factor",
                    "expectancy",
                    "average_holding_bars",
                    "average_mfe_atr",
                    "average_mae_atr",
                    "fees_paid",
                ]
            )

        rows: list[dict[str, Any]] = []
        grouped = enriched_trades.groupby(column, dropna=False, observed=False)
        for value, frame in grouped:
            wins = frame.loc[frame["net_pnl"] > 0.0, "net_pnl"]
            losses = frame.loc[frame["net_pnl"] <= 0.0, "net_pnl"]
            gross_profit = float(wins.sum()) if not wins.empty else 0.0
            gross_loss = abs(float(losses.sum())) if not losses.empty else 0.0
            average_win = float(wins.mean()) if not wins.empty else 0.0
            average_loss = float(losses.mean()) if not losses.empty else 0.0
            payoff_real = average_win / abs(average_loss) if average_loss < 0.0 else 0.0
            rows.append(
                {
                    column: value,
                    "trades": int(len(frame)),
                    "gross_pnl": float(frame["gross_pnl"].sum()),
                    "net_pnl": float(frame["net_pnl"].sum()),
                    "win_rate": float((frame["net_pnl"] > 0.0).mean()) if not frame.empty else 0.0,
                    "average_win": average_win,
                    "average_loss": average_loss,
                    "payoff_real": payoff_real,
                    "profit_factor": (gross_profit / gross_loss) if gross_loss > 0.0 else 0.0,
                    "expectancy": float(frame["net_pnl"].mean()) if not frame.empty else 0.0,
                    "average_holding_bars": float(frame["bars_held"].mean()) if not frame.empty else 0.0,
                    "average_mfe_atr": (
                        float(frame["mfe_atr"].dropna().mean()) if "mfe_atr" in frame and not frame["mfe_atr"].dropna().empty else 0.0
                    ),
                    "average_mae_atr": (
                        float(frame["mae_atr"].dropna().mean()) if "mae_atr" in frame and not frame["mae_atr"].dropna().empty else 0.0
                    ),
                    "fees_paid": float(frame["fees_paid"].sum()),
                }
            )

        result = pd.DataFrame(rows)
        if column == "signal_weekday":
            weekday_order = {
                "Monday": 0,
                "Tuesday": 1,
                "Wednesday": 2,
                "Thursday": 3,
                "Friday": 4,
                "Saturday": 5,
                "Sunday": 6,
            }
            result["_order"] = result[column].map(weekday_order).fillna(999)
            result = result.sort_values("_order").drop(columns=["_order"])
        elif column in {"signal_hour_utc", "exit_month", "exit_quarter", "exit_year"}:
            result = result.sort_values(column)
        else:
            result = result.sort_values("net_pnl")
        return result.reset_index(drop=True)

    def _build_duration_breakdown(self, enriched_trades: pd.DataFrame) -> pd.DataFrame:
        if enriched_trades.empty:
            return pd.DataFrame(columns=["holding_bucket"])
        frame = enriched_trades.copy()
        frame["holding_bucket"] = pd.cut(
            frame["bars_held"],
            bins=[0, 1, 2, 3, 6, 12, math.inf],
            labels=["1", "2", "3", "4-6", "7-12", "13+"],
            include_lowest=True,
        )
        return self._build_trade_breakdown(frame, "holding_bucket")

    def _build_bucket_breakdown(
        self,
        enriched_trades: pd.DataFrame,
        *,
        column: str,
        bins: Sequence[float],
        labels: Sequence[str],
        bucket_name: str,
    ) -> pd.DataFrame:
        if enriched_trades.empty or column not in enriched_trades.columns:
            return pd.DataFrame(columns=[bucket_name])
        frame = enriched_trades.copy()
        frame[bucket_name] = pd.cut(
            frame[column],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
        valid = frame.loc[frame[bucket_name].notna()].copy()
        if valid.empty:
            return self._empty_bucket_breakdown(bucket_name, labels)
        return self._build_trade_breakdown(valid, bucket_name)

    def _empty_bucket_breakdown(self, bucket_name: str, labels: Sequence[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for label in labels:
            rows.append(
                {
                    bucket_name: label,
                    "trades": 0,
                    "gross_pnl": 0.0,
                    "net_pnl": 0.0,
                    "win_rate": 0.0,
                    "average_win": 0.0,
                    "average_loss": 0.0,
                    "payoff_real": 0.0,
                    "profit_factor": 0.0,
                    "expectancy": 0.0,
                    "average_holding_bars": 0.0,
                    "average_mfe_atr": 0.0,
                    "average_mae_atr": 0.0,
                    "fees_paid": 0.0,
                }
            )
        return pd.DataFrame(rows)

    def _build_stop_target_breakdown(self, enriched_trades: pd.DataFrame) -> pd.DataFrame:
        if enriched_trades.empty:
            return pd.DataFrame(columns=["measure", "bucket"])

        stop = self._build_bucket_breakdown(
            enriched_trades,
            column="stop_distance_pct",
            bins=[0.0, 0.001, 0.002, 0.003, 0.005, math.inf],
            labels=["<=0.10%", "0.10-0.20%", "0.20-0.30%", "0.30-0.50%", ">0.50%"],
            bucket_name="bucket",
        )
        stop.insert(0, "measure", "stop_distance_pct")

        target = self._build_bucket_breakdown(
            enriched_trades,
            column="target_distance_pct",
            bins=[0.0, 0.001, 0.002, 0.003, 0.005, math.inf],
            labels=["<=0.10%", "0.10-0.20%", "0.20-0.30%", "0.30-0.50%", ">0.50%"],
            bucket_name="bucket",
        )
        target.insert(0, "measure", "target_distance_pct")
        return pd.concat([stop, target], ignore_index=True)

    def _build_signal_side_breakdown(self, artifact_set: BaselineArtifactSet, enriched_trades: pd.DataFrame) -> pd.DataFrame:
        actionable = artifact_set.signal_frame.loc[artifact_set.signal_frame["side"].isin(["long", "short"])].copy()
        if actionable.empty:
            return pd.DataFrame(columns=["signal_side"])

        candidate_counts = actionable.groupby("side", observed=False).size().rename("candidate_signals")
        executed_counts = enriched_trades.groupby("candidate_signal_side", observed=False).size().rename("executed_trades")
        result = candidate_counts.to_frame().join(executed_counts, how="left").fillna(0.0).reset_index()
        result = result.rename(columns={"side": "signal_side"})
        result["executed_trades"] = result["executed_trades"].astype(int)
        result["execution_rate"] = result["executed_trades"] / result["candidate_signals"]
        return result.sort_values("signal_side").reset_index(drop=True)

    def _run_variant_suite(self, artifact_set: BaselineArtifactSet) -> pd.DataFrame:
        variants = [
            ComparisonVariant(name="baseline", description="Historical fixed-signal baseline replay."),
            ComparisonVariant(name="no_fees", description="Replay with fees removed.", fee_bps=0.0),
            ComparisonVariant(name="no_slippage", description="Replay with slippage removed.", slippage_bps=0.0),
            ComparisonVariant(name="no_costs", description="Replay with both fees and slippage removed.", fee_bps=0.0, slippage_bps=0.0),
            ComparisonVariant(name="only_longs", description="Replay only long candidate trades.", side_filter=SignalSide.LONG),
            ComparisonVariant(name="only_shorts", description="Replay only short candidate trades.", side_filter=SignalSide.SHORT),
            ComparisonVariant(name="no_time_stop", description="Replay with time stop disabled.", disable_time_stop=True),
            ComparisonVariant(name="no_session_close", description="Replay with session close exits disabled.", disable_session_close=True),
            ComparisonVariant(name="target_1p5x", description="Replay with target distance widened to 1.5x.", target_scale=1.5),
        ]

        baseline_result = artifact_set.baseline_replay
        rows: list[dict[str, Any]] = []
        for variant in variants:
            variant_signals = self._transform_signals(artifact_set.signals, variant)
            result = self._run_backtest_replay(
                bars=artifact_set.bars,
                features=artifact_set.features,
                signals=variant_signals,
                fee_bps=variant.fee_bps,
                slippage_bps=variant.slippage_bps,
            )
            rows.append(
                {
                    "variant": variant.name,
                    "description": variant.description,
                    "trades": int(result.trades),
                    "win_rate": float(result.win_rate),
                    "payoff": float(result.payoff),
                    "expectancy": float(result.expectancy),
                    "net_pnl": float(result.pnl_net),
                    "max_drawdown": float(result.max_drawdown),
                    "delta_pnl_vs_baseline": float(result.pnl_net - baseline_result.pnl_net),
                    "delta_trades_vs_baseline": int(result.trades - baseline_result.trades),
                }
            )

        return pd.DataFrame(rows)

    def _build_cost_impact(self, enriched_trades: pd.DataFrame, variant_frame: pd.DataFrame) -> pd.DataFrame:
        if variant_frame.empty:
            return pd.DataFrame(columns=["variant"])
        selected = variant_frame.loc[variant_frame["variant"].isin(["baseline", "no_fees", "no_slippage", "no_costs"])].copy()
        selected["fees_paid_total"] = float(enriched_trades["fees_paid"].sum()) if not enriched_trades.empty else 0.0
        return selected.reset_index(drop=True)

    def _replay_risk_engine(self, artifact_set: BaselineArtifactSet) -> RiskReplaySummary:
        settings = self.application.settings
        simulator = PortfolioSimulator(
            initial_capital=settings.backtest.initial_capital,
            fee_bps=settings.backtest.fee_bps,
            slippage_bps=settings.backtest.slippage_bps,
            intrabar_exit_policy=settings.backtest.intrabar_exit_policy,
            gap_exit_policy=settings.backtest.gap_exit_policy,
            point_value=settings.backtest.point_value,
            contract_step=settings.backtest.contract_step,
            min_contracts=settings.backtest.min_contracts,
            max_contracts=settings.backtest.max_contracts,
            fee_per_contract_per_side=settings.backtest.fee_per_contract_per_side,
            slippage_points=settings.backtest.slippage_points,
        )

        day_start_equity = settings.backtest.initial_capital
        current_day = None
        trades_today = 0
        consecutive_losses_today = 0
        daily_kill_switch_active = False
        peak_equity = settings.backtest.initial_capital
        rows: list[dict[str, Any]] = []
        blocked_by_reason: Counter[str] = Counter()
        approved_by_side: Counter[str] = Counter()
        blocked_by_side: Counter[str] = Counter()

        for index, (bar, feature, signal) in enumerate(zip(artifact_set.bars, artifact_set.features, artifact_set.signals, strict=True)):
            timestamp = pd.Timestamp(bar.timestamp)
            timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")

            if current_day != timestamp.date():
                current_day = timestamp.date()
                day_start_equity = simulator.equity(bar.open)
                trades_today = 0
                consecutive_losses_today = 0
                daily_kill_switch_active = False

            trade = simulator.step(
                index=index,
                bar=bar,
                feature=feature,
                exit_zscore_threshold=settings.strategy.exit_zscore,
                session_close_hour_utc=settings.strategy.session_close_hour_utc,
                session_close_minute_utc=settings.strategy.session_close_minute_utc,
                session_close_timezone=settings.strategy.session_close_timezone,
                session_close_windows=tuple(settings.strategy.entry_session_windows),
            )
            if trade is not None:
                trades_today += 1
                consecutive_losses_today = consecutive_losses_today + 1 if trade.net_pnl <= 0.0 else 0

            current_equity = simulator.equity(bar.close)
            peak_equity = max(peak_equity, current_equity)
            total_drawdown_pct = ((peak_equity - current_equity) / peak_equity) if peak_equity > 0.0 else 0.0
            daily_pnl_pct = ((current_equity - day_start_equity) / day_start_equity) if day_start_equity > 0.0 else 0.0
            if settings.risk.daily_kill_switch and daily_pnl_pct <= -settings.risk.max_daily_loss:
                daily_kill_switch_active = True

            portfolio = PortfolioState(
                equity=current_equity,
                cash=simulator.cash,
                daily_pnl_pct=daily_pnl_pct,
                open_positions=int(simulator.position is not None) + int(simulator.pending_entry is not None),
                gross_exposure=(
                    simulator.position.quantity * bar.close * simulator.position.point_value
                    if simulator.position is not None
                    else 0.0
                ),
                peak_equity=peak_equity,
                total_drawdown_pct=total_drawdown_pct,
                trades_today=trades_today,
                consecutive_losses_today=consecutive_losses_today,
                daily_kill_switch_active=daily_kill_switch_active,
                session_allowed=is_within_session(
                    timestamp,
                    start_hour_utc=settings.risk.session_start_hour_utc,
                    start_minute_utc=settings.risk.session_start_minute_utc,
                    end_hour_utc=settings.risk.session_end_hour_utc,
                    end_minute_utc=settings.risk.session_end_minute_utc,
                    timezone=settings.risk.session_timezone,
                    session_windows=tuple(settings.risk.session_windows),
                ),
                timestamp=timestamp.to_pydatetime(),
            )
            decision = self.application.risk_engine.evaluate(signal, portfolio)
            actionable = signal.side in {SignalSide.LONG, SignalSide.SHORT}
            execution_ready = False
            if actionable and decision.approved and index < len(artifact_set.bars) - 1 and simulator.position is None and simulator.pending_entry is None and signal_has_executable_levels(signal):
                execution_ready = simulator.queue_signal(
                    signal=signal,
                    index=index,
                    size_fraction=decision.size_fraction,
                    max_leverage=decision.max_leverage,
                )

            if actionable:
                if decision.approved:
                    approved_by_side[signal.side.value] += 1
                else:
                    blocked_by_side[signal.side.value] += 1
                    if decision.reason_code is not None:
                        blocked_by_reason[decision.reason_code] += 1

            rows.append(
                {
                    "timestamp": timestamp,
                    "raw_side": signal.side.value,
                    "actionable": actionable,
                    "approved": decision.approved,
                    "execution_ready": execution_ready,
                    "reason_code": decision.reason_code,
                    "trades_today": trades_today,
                    "daily_pnl_pct": daily_pnl_pct,
                    "total_drawdown_pct": total_drawdown_pct,
                }
            )

        frame = pd.DataFrame(rows)
        actionable_frame = frame.loc[frame["actionable"]].copy()
        return RiskReplaySummary(
            rows=frame,
            actionable_signals=int(len(actionable_frame)),
            approved_signals=int(actionable_frame["approved"].sum()),
            blocked_signals=int((~actionable_frame["approved"]).sum()),
            blocked_by_reason=dict(sorted(blocked_by_reason.items())),
            approved_by_side=dict(sorted(approved_by_side.items())),
            blocked_by_side=dict(sorted(blocked_by_side.items())),
        )

    def _build_risk_execution_breakdown(self, artifact_set: BaselineArtifactSet, risk_replay: RiskReplaySummary | None) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        candidate_counts = artifact_set.signal_frame.loc[artifact_set.signal_frame["side"].isin(["long", "short"])].groupby("side", observed=False).size()
        executed_counts = artifact_set.trade_frame.groupby("side", observed=False).size()

        for side, count in candidate_counts.items():
            rows.append({"source": "historical_artifact", "status": "candidate_signal", "side": side, "reason_code": None, "count": int(count)})
        for side, count in executed_counts.items():
            rows.append({"source": "historical_artifact", "status": "executed_trade", "side": side, "reason_code": None, "count": int(count)})

        if risk_replay is not None and not risk_replay.rows.empty:
            actionable = risk_replay.rows.loc[risk_replay.rows["actionable"]].copy()
            grouped = actionable.groupby(["raw_side", "approved", "reason_code"], observed=False).size()
            for (side, approved, reason_code), count in grouped.items():
                rows.append(
                    {
                        "source": "current_risk_replay",
                        "status": "approved" if bool(approved) else "blocked",
                        "side": side,
                        "reason_code": reason_code,
                        "count": int(count),
                    }
                )

        columns = ["source", "status", "side", "reason_code", "count"]
        frame = pd.DataFrame(rows, columns=columns)
        if frame.empty:
            return frame
        return frame.sort_values(["source", "side", "status", "reason_code"], na_position="last")

    def _build_automatic_conclusion(
        self,
        *,
        overall_metrics: dict[str, Any],
        enriched_trades: pd.DataFrame,
        monthly_breakdown: pd.DataFrame,
        hourly_breakdown: pd.DataFrame,
        side_breakdown: pd.DataFrame,
        regime_breakdown: pd.DataFrame,
        exit_reason_breakdown: pd.DataFrame,
        variant_frame: pd.DataFrame,
    ) -> dict[str, Any]:
        return self._build_family_aware_automatic_conclusion(
            overall_metrics=overall_metrics,
            enriched_trades=enriched_trades,
            monthly_breakdown=monthly_breakdown,
            hourly_breakdown=hourly_breakdown,
            side_breakdown=side_breakdown,
            regime_breakdown=regime_breakdown,
            exit_reason_breakdown=exit_reason_breakdown,
            variant_frame=variant_frame,
        )

    def _build_family_aware_automatic_conclusion(
        self,
        *,
        overall_metrics: dict[str, Any],
        enriched_trades: pd.DataFrame,
        monthly_breakdown: pd.DataFrame,
        hourly_breakdown: pd.DataFrame,
        side_breakdown: pd.DataFrame,
        regime_breakdown: pd.DataFrame,
        exit_reason_breakdown: pd.DataFrame,
        variant_frame: pd.DataFrame,
    ) -> dict[str, Any]:
        family = self.application.settings.strategy.family
        baseline_pnl = float(overall_metrics["net_pnl"])
        no_costs_pnl = self._variant_metric(variant_frame, "no_costs", "net_pnl")
        no_fees_pnl = self._variant_metric(variant_frame, "no_fees", "net_pnl")
        no_slippage_pnl = self._variant_metric(variant_frame, "no_slippage", "net_pnl")
        no_time_stop_pnl = self._variant_metric(variant_frame, "no_time_stop", "net_pnl")
        no_session_close_pnl = self._variant_metric(variant_frame, "no_session_close", "net_pnl")
        only_longs_pnl = self._variant_metric(variant_frame, "only_longs", "net_pnl")
        only_shorts_pnl = self._variant_metric(variant_frame, "only_shorts", "net_pnl")

        hourly_loss_share = 0.0
        toxic_hours: list[str] = []
        if not hourly_breakdown.empty and baseline_pnl < 0.0:
            worst_hours = hourly_breakdown.nsmallest(5, "net_pnl")
            hourly_loss_share = abs(float(worst_hours["net_pnl"].sum())) / abs(baseline_pnl)
            toxic_hours = [str(int(value)) for value in worst_hours["signal_hour_utc"].tolist()]

        top_causes: list[str] = []
        if no_costs_pnl is not None and no_costs_pnl > 0.0 > baseline_pnl:
            top_causes.append(
                "Execution costs are crushing a thin gross edge: the fixed-signal replay moves "
                f"from `{baseline_pnl:.2f}` to `{no_costs_pnl:.2f}` without fees or slippage."
            )
        if overall_metrics["break_even_gap"] > 0.10:
            top_causes.append(
                "The current win-rate and payoff mix is not strong enough: the baseline wins "
                f"`{overall_metrics['win_rate'] * 100:.2f}%` but would need "
                f"`{overall_metrics['break_even_win_rate'] * 100:.2f}%` to break even at the current payoff."
            )
        if not exit_reason_breakdown.empty:
            stop_loss_row = exit_reason_breakdown.loc[exit_reason_breakdown["exit_reason"] == "stop_loss"]
            take_profit_row = exit_reason_breakdown.loc[exit_reason_breakdown["exit_reason"] == "take_profit"]
            if not stop_loss_row.empty and not take_profit_row.empty:
                top_causes.append(
                    "Stop losses are dominating the damage: "
                    f"`{int(stop_loss_row.iloc[0]['trades'])}` stop exits versus "
                    f"`{int(take_profit_row.iloc[0]['trades'])}` take-profit exits."
                )
        if family in {"trend_breakout", "opening_range_breakout"} and len(top_causes) < 3 and not enriched_trades.empty:
            median_breakout_distance = self._safe_series_median(enriched_trades.get("breakout_distance_atr"))
            median_target_to_cost = self._safe_series_median(enriched_trades.get("target_to_cost_ratio"))
            median_momentum = self._safe_series_median(enriched_trades.get("abs_momentum"))
            if math.isfinite(median_target_to_cost) and median_target_to_cost < 2.0:
                top_causes.append(
                    "Breakouts look too small versus execution friction: the median target-to-cost ratio is only "
                    f"`{median_target_to_cost:.2f}x`."
                )
            elif math.isfinite(median_breakout_distance) and median_breakout_distance < 0.15:
                top_causes.append(
                    "Entries are triggering too close to the breakout level: median breakout distance is only "
                    f"`{median_breakout_distance:.2f} ATR`."
                )
            elif math.isfinite(median_momentum) and median_momentum < 0.0030:
                top_causes.append(
                    "Momentum confirmation on executed trades is weak: median absolute momentum is only "
                    f"`{median_momentum * 100:.2f}%`."
                )
        if len(top_causes) < 3 and hourly_loss_share > 0.30:
            top_causes.append(
                "Losses are clustering in toxic hours; the five worst signal hours explain "
                f"`{hourly_loss_share * 100:.1f}%` of the total loss."
            )
        if len(top_causes) < 3 and only_longs_pnl is not None and only_shorts_pnl is not None:
            if only_longs_pnl < only_shorts_pnl:
                top_causes.append("Long trades are the weaker side of the system on this slice.")
            elif only_shorts_pnl < only_longs_pnl:
                top_causes.append("Short trades are the weaker side of the system on this slice.")
        if len(top_causes) < 3 and not regime_breakdown.empty and len(regime_breakdown) == 1:
            top_causes.append(
                "The trend/regime filter is working mechanically but not economically: "
                f"all trades land in `{regime_breakdown.iloc[0]['regime']}` and the PnL still does not hold up."
            )
        top_causes = top_causes[:3]

        promising_parts: list[str] = []
        if no_costs_pnl is not None and no_costs_pnl > 0.0:
            promising_parts.append(
                "The setup is not obviously dead: the same signal sequence is profitable before costs."
            )
        if overall_metrics["max_drawdown"] <= 0.10:
            promising_parts.append(
                f"Drawdown is contained enough for further research at `{overall_metrics['max_drawdown']:.4f}`."
            )
        if overall_metrics["payoff_real"] >= 1.0:
            promising_parts.append(
                f"The realized payoff is respectable at `{overall_metrics['payoff_real']:.2f}`."
            )
        if family in {"trend_breakout", "opening_range_breakout"} and not enriched_trades.empty:
            median_breakout_range = self._safe_series_median(enriched_trades.get("breakout_range_width_atr"))
            if math.isfinite(median_breakout_range):
                promising_parts.append(
                    f"Executed breakouts are not trivially tiny: median prior range width is `{median_breakout_range:.2f} ATR`."
                )

        weak_parts: list[str] = []
        if overall_metrics["break_even_gap"] > 0.0:
            weak_parts.append(
                f"The strategy still needs `{overall_metrics['break_even_gap'] * 100:.2f}` percentage points more hit rate at the current payoff."
            )
        if no_fees_pnl is not None and baseline_pnl < no_fees_pnl:
            weak_parts.append(
                f"Fee drag is material: removing fees moves net PnL from `{baseline_pnl:.2f}` to `{no_fees_pnl:.2f}`."
            )
        if no_slippage_pnl is not None and baseline_pnl < no_slippage_pnl:
            weak_parts.append(
                f"Slippage drag is material: removing slippage moves net PnL from `{baseline_pnl:.2f}` to `{no_slippage_pnl:.2f}`."
            )
        if hourly_loss_share > 0.30 and toxic_hours:
            weak_parts.append(f"Losses are concentrated in specific hours: `{', '.join(toxic_hours)}` UTC.")
        if family in {"trend_breakout", "opening_range_breakout"} and not enriched_trades.empty:
            median_target_to_cost = self._safe_series_median(enriched_trades.get("target_to_cost_ratio"))
            if math.isfinite(median_target_to_cost) and median_target_to_cost < 2.5:
                weak_parts.append(
                    f"The median target-to-cost ratio is only `{median_target_to_cost:.2f}x`, which leaves little room for execution noise."
                )

        merits_robust_validation = bool(
            overall_metrics["number_of_trades"] >= 8
            and overall_metrics["max_drawdown"] <= 0.12
            and (
                baseline_pnl > 0.0
                or (no_costs_pnl is not None and no_costs_pnl > 0.0 and overall_metrics["payoff_real"] >= 0.8)
            )
        )
        validation_verdict = "GO WITH CAUTION" if merits_robust_validation else "NO-GO"

        next_changes = [
            "Use this diagnostic evidence to decide whether the current breakout family deserves robust temporal validation.",
            "If it does not, the next experiment should be a very small refinement around hours or setup quality, not RL.",
        ]
        if family in {"trend_breakout", "opening_range_breakout"}:
            next_changes.insert(
                0,
                "Check whether removing the worst trading hours materially improves the economics before changing the strategy structure.",
            )

        do_not_touch = [
            "Do not add PPO or more model complexity until the standalone rule-based baseline is economically defendable.",
            "Do not turn this diagnostic into a parameter search; keep the next step small and evidence-driven.",
        ]

        if baseline_pnl > 0.0:
            short_answer = (
                "This breakout baseline looks economically plausible on this slice and deserves robust validation before any RL layer."
            )
        elif no_costs_pnl is not None and no_costs_pnl > 0.0:
            short_answer = (
                "This breakout baseline may have a fragile raw edge, but execution costs are likely erasing it. It is interesting enough for careful validation, not for RL."
            )
        else:
            short_answer = (
                "This breakout baseline does not yet show a defendable economic edge on this slice. It should not move to RL, and robust validation only makes sense if you still want to test whether the weak edge is real."
            )

        return {
            "short_answer": short_answer,
            "top_causes": top_causes,
            "promising_parts": promising_parts,
            "weak_parts": weak_parts,
            "merits_robust_validation": merits_robust_validation,
            "validation_verdict": validation_verdict,
            "next_minimal_changes": next_changes,
            "not_worth_touching_yet": do_not_touch,
            "supporting_observations": {
                "baseline_pnl": baseline_pnl,
                "no_costs_pnl": no_costs_pnl,
                "no_fees_pnl": no_fees_pnl,
                "no_slippage_pnl": no_slippage_pnl,
                "no_time_stop_delta": no_time_stop_pnl - baseline_pnl if no_time_stop_pnl is not None else None,
                "no_session_close_delta": no_session_close_pnl - baseline_pnl if no_session_close_pnl is not None else None,
                "only_longs_pnl": only_longs_pnl,
                "only_shorts_pnl": only_shorts_pnl,
                "toxic_hours_utc": toxic_hours,
            },
        }
        baseline_pnl = float(overall_metrics["net_pnl"])
        no_costs_pnl = self._variant_metric(variant_frame, "no_costs", "net_pnl")
        no_fees_pnl = self._variant_metric(variant_frame, "no_fees", "net_pnl")
        no_slippage_pnl = self._variant_metric(variant_frame, "no_slippage", "net_pnl")
        no_time_stop_pnl = self._variant_metric(variant_frame, "no_time_stop", "net_pnl")
        no_session_close_pnl = self._variant_metric(variant_frame, "no_session_close", "net_pnl")

        hourly_loss_share = 0.0
        toxic_hours: list[str] = []
        if not hourly_breakdown.empty and baseline_pnl < 0.0:
            worst_hours = hourly_breakdown.nsmallest(5, "net_pnl")
            hourly_loss_share = abs(float(worst_hours["net_pnl"].sum())) / abs(baseline_pnl)
            toxic_hours = [str(int(value)) for value in worst_hours["signal_hour_utc"].tolist()]

        top_causes: list[str] = []
        if no_costs_pnl is not None and no_costs_pnl > 0.0 > baseline_pnl:
            top_causes.append(
                "Los costes de ejecución están destruyendo un edge bruto muy fino: el replay con señales fijas pasa "
                f"de `{baseline_pnl:.2f}` a `{no_costs_pnl:.2f}` sin fees ni slippage."
            )
        if overall_metrics["break_even_gap"] > 0.10:
            top_causes.append(
                "La combinación `win rate + payoff` es insostenible en neto: el baseline gana "
                f"`{overall_metrics['win_rate'] * 100:.2f}%` pero necesitaría "
                f"`{overall_metrics['break_even_win_rate'] * 100:.2f}%` para quedar en break-even con su payoff actual."
            )
        if not exit_reason_breakdown.empty:
            stop_loss_row = exit_reason_breakdown.loc[exit_reason_breakdown["exit_reason"] == "stop_loss"]
            take_profit_row = exit_reason_breakdown.loc[exit_reason_breakdown["exit_reason"] == "take_profit"]
            if not stop_loss_row.empty and not take_profit_row.empty:
                top_causes.append(
                    "Las pérdidas por `stop_loss` dominan el resultado: "
                    f"`{int(stop_loss_row.iloc[0]['trades'])}` stops frente a "
                    f"`{int(take_profit_row.iloc[0]['trades'])}` take profits, con una asimetría neta muy desfavorable."
                )
        if len(top_causes) < 3 and hourly_loss_share > 0.30:
            top_causes.append(
                "Las pérdidas se concentran en franjas horarias tóxicas; las peores cinco horas aportan "
                f"`{hourly_loss_share * 100:.1f}%` de la pérdida total."
            )
        if len(top_causes) < 3 and not regime_breakdown.empty and len(regime_breakdown) == 1:
            top_causes.append(
                "El filtro de régimen parece funcionar de forma mecánica, pero no discrimina calidad: "
                f"todos los trades quedan en `{regime_breakdown.iloc[0]['regime']}` y aun así el PnL es muy negativo."
            )
        top_causes = top_causes[:3]

        rescuable = []
        if no_costs_pnl is not None and no_costs_pnl > 0.0:
            rescuable.append("La idea base no está completamente muerta: sin costes, el replay fijo queda en positivo.")
        if not side_breakdown.empty and (side_breakdown["net_pnl"] < 0.0).all():
            rescuable.append("La degradación no depende de un único lado; eso sugiere atacar fricción y horarios antes que reescribir la lógica.")
        if monthly_breakdown.empty or (monthly_breakdown["net_pnl"] <= 0.0).all():
            rescuable.append("Lo más rescatable hoy es el edge pre-coste y la posibilidad de recortar mucho el número de trades en las horas peores.")

        next_changes = [
            "Reducir fricción por trade antes de tocar RL: menos operaciones, sesiones más selectivas y filtros para horas claramente tóxicas.",
            "Probar un filtro simple de calendario/horario sobre el baseline fijo, especialmente para las peores horas y fines de semana.",
            "Probar una versión más selectiva que aumente la magnitud esperada por trade en lugar de sólo mover targets.",
        ]

        do_not_touch = [
            "No merece la pena tocar PPO, SAC, LSTM ni más IA mientras el baseline neto siga tan débil.",
            "No parece prioritario tocar `time_stop`: su impacto aislado es pequeño frente a la pérdida total.",
            "Tampoco parece prioritario tocar `close_on_session_end`: su replay aislado casi no mueve el resultado.",
        ]

        short_answer = (
            "Este baseline pierde dinero sobre todo porque su edge bruto es muy pequeño y queda aplastado por los costes, "
            "mientras que el win rate neto es demasiado bajo para el payoff que deja después de fees/slippage. "
            "Además, las pérdidas no vienen de un solo bug: aparecen en ambos lados y se concentran en horas claramente malas."
        )

        return {
            "short_answer": short_answer,
            "top_causes": top_causes,
            "rescues": rescuable,
            "next_minimal_changes": next_changes,
            "not_worth_touching_yet": do_not_touch,
            "supporting_observations": {
                "baseline_pnl": baseline_pnl,
                "no_costs_pnl": no_costs_pnl,
                "no_fees_pnl": no_fees_pnl,
                "no_slippage_pnl": no_slippage_pnl,
                "no_time_stop_delta": no_time_stop_pnl - baseline_pnl if no_time_stop_pnl is not None else None,
                "no_session_close_delta": no_session_close_pnl - baseline_pnl if no_session_close_pnl is not None else None,
                "toxic_hours_utc": toxic_hours,
            },
        }

    def _build_summary_markdown(
        self,
        *,
        diagnostics: dict[str, Any],
        overall_metrics: dict[str, Any],
        yearly_breakdown: pd.DataFrame,
        monthly_breakdown: pd.DataFrame,
        hourly_breakdown: pd.DataFrame,
        variant_frame: pd.DataFrame,
        signal_side_breakdown: pd.DataFrame,
    ) -> str:
        return self._build_family_aware_summary_markdown(
            diagnostics=diagnostics,
            overall_metrics=overall_metrics,
            yearly_breakdown=yearly_breakdown,
            monthly_breakdown=monthly_breakdown,
            hourly_breakdown=hourly_breakdown,
            variant_frame=variant_frame,
            signal_side_breakdown=signal_side_breakdown,
        )

    def _build_family_aware_summary_markdown(
        self,
        *,
        diagnostics: dict[str, Any],
        overall_metrics: dict[str, Any],
        yearly_breakdown: pd.DataFrame,
        monthly_breakdown: pd.DataFrame,
        hourly_breakdown: pd.DataFrame,
        variant_frame: pd.DataFrame,
        signal_side_breakdown: pd.DataFrame,
    ) -> str:
        automatic = diagnostics["automatic_conclusion"]
        worst_hours = hourly_breakdown.nsmallest(5, "net_pnl") if not hourly_breakdown.empty else pd.DataFrame()
        worst_month = monthly_breakdown.nsmallest(1, "net_pnl") if not monthly_breakdown.empty else pd.DataFrame()

        lines = [
            "# Baseline Diagnostics",
            "",
            "## Direct Answer",
            automatic["short_answer"],
            "",
            "## Verdict",
            f"- Robust validation readiness: `{automatic['validation_verdict']}`",
            f"- Merits robust validation now: `{automatic['merits_robust_validation']}`",
            "",
            "## Baseline Snapshot",
            f"- Strategy family: `{overall_metrics['strategy_family']}`",
            f"- Variant: `{overall_metrics['variant_name']}`",
            f"- Trades: `{overall_metrics['number_of_trades']}`",
            f"- Trades per year: `{overall_metrics['trades_per_year']:.2f}`",
            f"- Trades per week avg: `{overall_metrics['trades_per_week_avg']:.2f}`",
            f"- Gross PnL: `{overall_metrics['gross_pnl']:.2f}`",
            f"- Net PnL: `{overall_metrics['net_pnl']:.2f}`",
            f"- Fees paid: `{overall_metrics['fees_paid_total']:.2f}`",
            f"- Estimated fee drag: `{(overall_metrics['estimated_fee_drag'] or 0.0):.2f}`",
            f"- Estimated slippage drag: `{(overall_metrics['estimated_slippage_drag'] or 0.0):.2f}`",
            f"- Estimated total cost drag: `{(overall_metrics['estimated_total_cost_drag'] or 0.0):.2f}`",
            f"- Win rate: `{overall_metrics['win_rate'] * 100:.2f}%`",
            f"- Average win: `{overall_metrics['average_win']:.2f}`",
            f"- Average loss: `{overall_metrics['average_loss']:.2f}`",
            f"- Payoff real: `{overall_metrics['payoff_real']:.4f}`",
            f"- Profit factor: `{overall_metrics['profit_factor']:.4f}`",
            f"- Expectancy: `{overall_metrics['expectancy']:.2f}`",
            f"- Break-even win rate: `{overall_metrics['break_even_win_rate'] * 100:.2f}%`",
            f"- Max drawdown: `{overall_metrics['max_drawdown']:.4f}`",
            f"- Sharpe: `{overall_metrics['sharpe']:.4f}`",
            f"- Sortino: `{overall_metrics['sortino']:.4f}`",
            f"- Calmar: `{overall_metrics['calmar']:.4f}`",
            f"- Max consecutive losses: `{overall_metrics['max_consecutive_losses']}`",
            f"- Average holding bars: `{overall_metrics['average_holding_bars']:.2f}`",
            f"- Average MFE (ATR): `{overall_metrics['average_mfe_atr']:.2f}`",
            f"- Average MAE (ATR): `{overall_metrics['average_mae_atr']:.2f}`",
            f"- Percent profitable years: `{overall_metrics['profitable_years_pct'] * 100:.2f}%`",
            f"- Percent profitable months: `{overall_metrics['profitable_months_pct'] * 100:.2f}%`",
            "",
            "## Top 3 Probable Causes",
        ]
        for cause in automatic["top_causes"]:
            lines.append(f"- {cause}")

        lines.extend(["", "## Promising Parts"])
        for item in automatic["promising_parts"]:
            lines.append(f"- {item}")

        lines.extend(["", "## Weak Parts"])
        for item in automatic["weak_parts"]:
            lines.append(f"- {item}")

        lines.extend(["", "## Minimum Next Experiment"])
        for item in automatic["next_minimal_changes"]:
            lines.append(f"- {item}")

        lines.extend(["", "## What Not To Touch Yet"])
        for item in automatic["not_worth_touching_yet"]:
            lines.append(f"- {item}")

        lines.extend(["", "## Key Observations"])
        if not worst_month.empty:
            month = worst_month.iloc[0]
            month_drawdown = month.get("month_drawdown", float("nan"))
            lines.append(
                f"- Worst month: `{month['exit_month']}` with net PnL `{month['net_pnl']:.2f}` and monthly drawdown `{month_drawdown:.4f}`."
            )
        if not yearly_breakdown.empty:
            rendered_years = ", ".join(
                f"`{row['exit_year']}` ({row['net_pnl']:.2f}, PF {row['profit_factor']:.2f})"
                for _, row in yearly_breakdown.iterrows()
            )
            lines.append(f"- Yearly results: {rendered_years}.")
        if not worst_hours.empty:
            rendered_hours = ", ".join(
                f"`{int(row['signal_hour_utc']):02d}:00` ({row['net_pnl']:.2f})"
                for _, row in worst_hours.iterrows()
            )
            lines.append(f"- Worst signal hours: {rendered_hours}.")
        if not signal_side_breakdown.empty:
            for _, row in signal_side_breakdown.iterrows():
                lines.append(
                    f"- Candidate signals `{row['signal_side']}`: `{int(row['candidate_signals'])}`; executed trades `{int(row['executed_trades'])}`; execution rate `{row['execution_rate'] * 100:.2f}%`."
                )

        if not variant_frame.empty:
            lines.extend(["", "## Diagnostic Variants"])
            for variant_name in [
                "baseline",
                "no_fees",
                "no_slippage",
                "no_costs",
                "only_longs",
                "only_shorts",
                "no_time_stop",
                "no_session_close",
                "target_1p5x",
            ]:
                row = variant_frame.loc[variant_frame["variant"] == variant_name]
                if row.empty:
                    continue
                metric = row.iloc[0]
                lines.append(
                    f"- `{variant_name}`: net PnL `{metric['net_pnl']:.2f}`, trades `{int(metric['trades'])}`, win rate `{metric['win_rate'] * 100:.2f}%`, delta vs baseline `{metric['delta_pnl_vs_baseline']:.2f}`."
                )

        return "\n".join(lines) + "\n"
        automatic = diagnostics["automatic_conclusion"]
        worst_hours = hourly_breakdown.nsmallest(5, "net_pnl") if not hourly_breakdown.empty else pd.DataFrame()
        worst_month = monthly_breakdown.nsmallest(1, "net_pnl") if not monthly_breakdown.empty else pd.DataFrame()

        lines = [
            "# Baseline Diagnostics",
            "",
            "## Respuesta Directa",
            automatic["short_answer"],
            "",
            "## Baseline Snapshot",
            f"- Trades: `{overall_metrics['number_of_trades']}`",
            f"- Gross PnL: `{overall_metrics['gross_pnl']:.2f}`",
            f"- Net PnL: `{overall_metrics['net_pnl']:.2f}`",
            f"- Fees pagadas: `{overall_metrics['fees_paid_total']:.2f}`",
            f"- Win rate: `{overall_metrics['win_rate'] * 100:.2f}%`",
            f"- Payoff real: `{overall_metrics['payoff_real']:.4f}`",
            f"- Expectancy: `{overall_metrics['expectancy']:.2f}`",
            f"- Break-even win rate: `{overall_metrics['break_even_win_rate'] * 100:.2f}%`",
            f"- Max drawdown: `{overall_metrics['max_drawdown']:.4f}`",
            f"- Max consecutive losses: `{overall_metrics['max_consecutive_losses']}`",
            f"- Average holding bars: `{overall_metrics['average_holding_bars']:.2f}`",
            f"- Percent profitable months: `{overall_metrics['profitable_months_pct'] * 100:.2f}%`",
            "",
            "## Top 3 Causas Más Probables",
        ]
        for cause in automatic["top_causes"]:
            lines.append(f"- {cause}")

        lines.extend(["", "## Partes Rescatables"])
        for item in automatic["rescues"]:
            lines.append(f"- {item}")

        lines.extend(["", "## Cambios Mínimos a Probar Después"])
        for item in automatic["next_minimal_changes"]:
            lines.append(f"- {item}")

        lines.extend(["", "## Qué No Tocar Todavía"])
        for item in automatic["not_worth_touching_yet"]:
            lines.append(f"- {item}")

        lines.extend(["", "## Hallazgos Clave"])
        if not worst_month.empty:
            month = worst_month.iloc[0]
            month_drawdown = month.get("month_drawdown", float("nan"))
            lines.append(f"- El peor mes fue `{month['exit_month']}` con `PnL {month['net_pnl']:.2f}` y drawdown mensual `{month_drawdown:.4f}`.")
        if not worst_hours.empty:
            rendered_hours = ", ".join(
                f"`{int(row['signal_hour_utc']):02d}:00` ({row['net_pnl']:.2f})"
                for _, row in worst_hours.iterrows()
            )
            lines.append(f"- Horas más tóxicas por señal: {rendered_hours}.")
        if not signal_side_breakdown.empty:
            for _, row in signal_side_breakdown.iterrows():
                lines.append(f"- Candidate signals `{row['signal_side']}`: `{int(row['candidate_signals'])}`; trades ejecutados `{int(row['executed_trades'])}`; tasa de ejecución `{row['execution_rate'] * 100:.2f}%`.")

        if not variant_frame.empty:
            lines.extend(["", "## Variantes Diagnósticas"])
            for variant_name in ["baseline", "no_fees", "no_slippage", "no_costs", "only_longs", "only_shorts", "no_time_stop", "no_session_close", "target_1p5x"]:
                row = variant_frame.loc[variant_frame["variant"] == variant_name]
                if row.empty:
                    continue
                metric = row.iloc[0]
                lines.append(f"- `{variant_name}`: PnL `{metric['net_pnl']:.2f}`, trades `{int(metric['trades'])}`, win rate `{metric['win_rate'] * 100:.2f}%`, delta vs baseline `{metric['delta_pnl_vs_baseline']:.2f}`.")

        return "\n".join(lines) + "\n"

    def _transform_signals(self, signals: Sequence[StrategySignal], variant: ComparisonVariant) -> tuple[StrategySignal, ...]:
        transformed: list[StrategySignal] = []
        for signal in signals:
            current_signal = signal
            if variant.side_filter is not None and signal.side not in {SignalSide.FLAT, variant.side_filter}:
                current_signal = StrategySignal(symbol=signal.symbol, timestamp=signal.timestamp, side=SignalSide.FLAT, strength=0.0, rationale=f"Filtered out by variant {variant.name}.")
            elif signal.side in {SignalSide.LONG, SignalSide.SHORT}:
                target_price = signal.target_price
                if variant.target_scale is not None and signal.entry_price is not None and signal.target_price is not None:
                    target_distance = abs(float(signal.target_price) - float(signal.entry_price)) * variant.target_scale
                    target_price = float(signal.entry_price) + target_distance if signal.side == SignalSide.LONG else float(signal.entry_price) - target_distance
                current_signal = StrategySignal(
                    symbol=signal.symbol,
                    timestamp=signal.timestamp,
                    side=signal.side,
                    strength=signal.strength,
                    rationale=signal.rationale,
                    entry_price=signal.entry_price,
                    stop_price=signal.stop_price,
                    target_price=target_price,
                    time_stop_bars=None if variant.disable_time_stop else signal.time_stop_bars,
                    close_on_session_end=False if variant.disable_session_close else signal.close_on_session_end,
                    entry_reason=signal.entry_reason,
                    metadata=dict(signal.metadata),
                )
            transformed.append(current_signal)
        return tuple(transformed)

    def _frame_to_bars(self, frame: pd.DataFrame) -> list[MarketBar]:
        return [MarketBar(timestamp=timestamp.to_pydatetime(), open=float(row["open"]), high=float(row["high"]), low=float(row["low"]), close=float(row["close"]), volume=float(row["volume"])) for timestamp, row in frame.iterrows()]

    def _frame_to_features(self, frame: pd.DataFrame) -> list[FeatureSnapshot]:
        rows = []
        for timestamp, row in frame.iterrows():
            values = {column: (float(row[column]) if pd.notna(row[column]) else float("nan")) for column in frame.columns}
            rows.append(FeatureSnapshot(timestamp=timestamp.to_pydatetime(), values=values, metadata={}))
        return rows

    def _frame_to_signals(self, frame: pd.DataFrame) -> list[StrategySignal]:
        signals: list[StrategySignal] = []
        for _, row in frame.iterrows():
            metadata: dict[str, Any] = {}
            for column in ["raw_side", "risk_approved", "risk_reason_code", "risk_blocked_by", "risk_size_fraction", "risk_max_leverage"]:
                if column in row.index and pd.notna(row[column]) and row[column] != "":
                    metadata[column] = row[column]

            signals.append(
                StrategySignal(
                    symbol=str(row.get("symbol", self.application.settings.market.symbol)),
                    timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                    side=SignalSide(str(row["side"])),
                    strength=float(row.get("strength", 0.0) or 0.0),
                    rationale=str(row.get("rationale", "")),
                    entry_price=self._parse_optional_float(row.get("entry_price")),
                    stop_price=self._parse_optional_float(row.get("stop_price")),
                    target_price=self._parse_optional_float(row.get("target_price")),
                    time_stop_bars=self._parse_optional_int(row.get("time_stop_bars")),
                    close_on_session_end=self._parse_optional_bool(row.get("close_on_session_end"), default=True),
                    entry_reason=self._parse_optional_text(row.get("entry_reason")),
                    metadata=metadata,
                )
            )
        return signals

    def _read_indexed_frame(self, path: Path, index_name: str) -> pd.DataFrame:
        frame = pd.read_csv(path)
        if index_name not in frame.columns:
            first_column = frame.columns[0]
            frame = frame.rename(columns={first_column: index_name})
        frame[index_name] = pd.to_datetime(frame[index_name], utc=True)
        return frame.set_index(index_name)

    def _anchor_column_name(self) -> str | None:
        if self.application.settings.strategy.family != "mean_reversion":
            return None
        anchor = self.application.settings.strategy.mean_reversion_anchor.lower()
        return "intraday_vwap" if anchor == "vwap" else "ema_50"

    def _regime_label(self, adx_value: float) -> str:
        if pd.isna(adx_value):
            return "unknown"
        return "trend" if float(adx_value) > self.application.settings.strategy.adx_threshold else "range"

    def _max_consecutive_losses(self, pnls: Iterable[float]) -> int:
        max_losses = 0
        streak = 0
        for pnl in pnls:
            if pnl <= 0.0:
                streak += 1
                max_losses = max(max_losses, streak)
            else:
                streak = 0
        return max_losses

    def _variant_metric(self, variant_frame: pd.DataFrame, variant: str, column: str) -> float | None:
        if variant_frame.empty or "variant" not in variant_frame.columns or column not in variant_frame.columns:
            return None
        row = variant_frame.loc[variant_frame["variant"] == variant]
        return None if row.empty else float(row.iloc[0][column])

    def _variant_delta(self, variant_lookup: pd.DataFrame, variant_name: str) -> float | None:
        if variant_lookup.empty or variant_name not in variant_lookup.index:
            return None
        return float(variant_lookup.loc[variant_name, "delta_pnl_vs_baseline"])

    def _coalesce_numeric_series(
        self,
        primary_frame: pd.DataFrame,
        secondary_frame: pd.DataFrame,
        primary_column: str,
        *,
        secondary_column: str | None = None,
    ) -> pd.Series:
        index = primary_frame.index
        primary = primary_frame.get(primary_column)
        if primary is None:
            primary_series = pd.Series(np.nan, index=index, dtype=float)
        else:
            primary_series = pd.to_numeric(primary, errors="coerce").astype(float)

        secondary_name = secondary_column or primary_column
        secondary = secondary_frame.get(secondary_name)
        if secondary is None:
            secondary_series = pd.Series(np.nan, index=index, dtype=float)
        else:
            secondary_series = pd.to_numeric(secondary, errors="coerce").astype(float)
        result = primary_series.copy()
        missing = result.isna()
        result.loc[missing] = secondary_series.loc[missing]
        return result

    def _safe_series_median(self, series: pd.Series | None) -> float:
        if series is None:
            return float("nan")
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            return float("nan")
        return float(clean.median())

    def _calculate_mfe_mae(
        self,
        *,
        ohlcv_lookup: pd.DataFrame,
        entry_timestamp: pd.Timestamp,
        exit_timestamp: pd.Timestamp,
        side: str,
        entry_price: float,
        atr: float | None,
    ) -> tuple[float, float, float | None, float | None]:
        path = ohlcv_lookup.loc[
            (ohlcv_lookup.index >= pd.Timestamp(entry_timestamp))
            & (ohlcv_lookup.index <= pd.Timestamp(exit_timestamp))
        ]
        if path.empty:
            return 0.0, 0.0, None, None

        if side == "long":
            mfe = float(max(0.0, path["high"].max() - entry_price))
            mae = float(max(0.0, entry_price - path["low"].min()))
        else:
            mfe = float(max(0.0, entry_price - path["low"].min()))
            mae = float(max(0.0, path["high"].max() - entry_price))

        if atr is None or not math.isfinite(float(atr)) or atr <= 0.0:
            return mfe, mae, None, None
        return mfe, mae, mfe / atr, mae / atr

    def _parse_optional_float(self, value: Any) -> float | None:
        if value is None or value == "" or pd.isna(value):
            return None
        return float(value)

    def _parse_optional_int(self, value: Any) -> int | None:
        if value is None or value == "" or pd.isna(value):
            return None
        return int(float(value))

    def _parse_optional_text(self, value: Any) -> str | None:
        if value is None or value == "" or pd.isna(value):
            return None
        return str(value)

    def _parse_optional_bool(self, value: Any, *, default: bool) -> bool:
        if value is None or value == "" or pd.isna(value):
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes"}

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
    parser = argparse.ArgumentParser(description="Diagnose an existing baseline artifact directory.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--variant")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--skip-variants", action="store_true")
    parser.add_argument("--skip-risk-replay", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    runner = BaselineDiagnosticsRunner.from_config(args.config_dir, variant_name=args.variant)
    artifacts = runner.run(
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        include_variants=not args.skip_variants,
        include_risk_replay=not args.skip_risk_replay,
    )

    print(f"Diagnostics written to {artifacts.diagnostics_path}")
    print(f"Summary written to {artifacts.summary_path}")
    print(f"Variant comparison written to {artifacts.variant_comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
