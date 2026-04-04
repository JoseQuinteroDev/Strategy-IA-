from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from hybrid_quant.bootstrap import TradingApplication, build_application_from_settings
from hybrid_quant.core import (
    BacktestRequest,
    FeatureSnapshot,
    MarketBar,
    MarketDataBatch,
    PortfolioState,
    RiskDecision,
    Settings,
    SignalSide,
    StrategyContext,
    StrategySignal,
    ValidationReport,
    load_settings,
)
from hybrid_quant.data import (
    BinanceHistoricalDownloader,
    DownloadRequest,
    HistoricalDataIngestionService,
    ParquetDatasetStore,
    read_ohlcv_frame,
)
from hybrid_quant.execution import PortfolioSimulator, is_within_session, signal_has_executable_levels

from .variants import load_variant_settings


@dataclass(slots=True)
class BaselineArtifacts:
    output_dir: Path
    ohlcv_path: Path
    features_path: Path
    signals_path: Path
    trades_path: Path
    risk_decisions_path: Path
    risk_log_path: Path
    report_path: Path
    summary_path: Path
    result: Any
    validation_report: ValidationReport


@dataclass(slots=True)
class BaselineRunner:
    application: TradingApplication
    data_service: HistoricalDataIngestionService

    @classmethod
    def from_config(cls, config_dir: str | Path, variant_name: str | None = None) -> "BaselineRunner":
        settings = (
            load_variant_settings(config_dir, variant_name)
            if variant_name is not None
            else load_settings(config_dir)
        )
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
        return cls(application=application, data_service=data_service)

    def run(
        self,
        *,
        output_dir: str | Path,
        request: DownloadRequest | None = None,
        input_frame: pd.DataFrame | None = None,
        allow_gaps: bool = False,
    ) -> BaselineArtifacts:
        if input_frame is None and request is None:
            raise ValueError("BaselineRunner.run requires either a download request or an input frame.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if input_frame is not None:
            frame, _, _ = self._prepare_local_frame(input_frame, allow_gaps=allow_gaps)
        else:
            frame, _, _ = self.data_service.prepare_frame(request=request, allow_gaps=allow_gaps)

        bars = self._frame_to_bars(frame)
        batch = MarketDataBatch(
            symbol=self.application.settings.market.symbol,
            timeframe=self.application.settings.market.execution_timeframe,
            bars=bars,
            metadata={"source": "baseline_runner"},
        )
        feature_snapshots = self.application.feature_pipeline.transform(batch)
        raw_signals = self._generate_signals(bars, feature_snapshots)
        filtered_signals, risk_rows, risk_log_lines, risk_summary = self._apply_risk_engine(
            bars=bars,
            feature_snapshots=feature_snapshots,
            raw_signals=raw_signals,
        )

        result = self.application.backtest_engine.run(
            BacktestRequest(
                bars=bars,
                features=feature_snapshots,
                signals=filtered_signals,
                initial_capital=self.application.settings.backtest.initial_capital,
                risk_per_trade_fraction=self.application.settings.risk.max_risk_per_trade,
                max_leverage=self.application.settings.risk.max_leverage,
                signal_cooldown_bars=self.application.settings.strategy.signal_cooldown_bars,
                exit_zscore_threshold=self.application.settings.strategy.exit_zscore,
                session_close_hour_utc=self.application.settings.strategy.session_close_hour_utc,
                session_close_minute_utc=self.application.settings.strategy.session_close_minute_utc,
                intrabar_exit_policy=self.application.settings.backtest.intrabar_exit_policy,
            )
        )
        validation_report = self.application.validator.validate(result)

        ohlcv_frame = frame.copy()
        ohlcv_frame.index.name = "open_time"
        feature_frame = self._feature_snapshots_to_frame(feature_snapshots)
        signal_frame = self._signals_to_frame(filtered_signals)
        trade_frame = self._trades_to_frame(result.trade_records)
        risk_frame = self._risk_rows_to_frame(risk_rows)

        ohlcv_path = output_path / "ohlcv.csv"
        features_path = output_path / "features.csv"
        signals_path = output_path / "signals.csv"
        trades_path = output_path / "trades.csv"
        risk_decisions_path = output_path / "risk_decisions.csv"
        risk_log_path = output_path / "risk.log"
        report_path = output_path / "report.json"
        summary_path = output_path / "summary.md"

        ohlcv_frame.to_csv(ohlcv_path)
        feature_frame.to_csv(features_path)
        signal_frame.to_csv(signals_path, index=False)
        trade_frame.to_csv(trades_path, index=False)
        risk_frame.to_csv(risk_decisions_path, index=False)
        risk_log_path.write_text(self._render_risk_log(risk_log_lines), encoding="utf-8")

        report_payload = self._build_report_payload(
            result=result,
            validation_report=validation_report,
            settings=self.application.settings,
            ohlcv_rows=len(ohlcv_frame),
            feature_rows=len(feature_frame),
            risk_summary=risk_summary,
        )
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        summary_path.write_text(self._build_summary_markdown(report_payload), encoding="utf-8")

        return BaselineArtifacts(
            output_dir=output_path,
            ohlcv_path=ohlcv_path,
            features_path=features_path,
            signals_path=signals_path,
            trades_path=trades_path,
            risk_decisions_path=risk_decisions_path,
            risk_log_path=risk_log_path,
            report_path=report_path,
            summary_path=summary_path,
            result=result,
            validation_report=validation_report,
        )

    def _apply_risk_engine(
        self,
        *,
        bars: Sequence[MarketBar],
        feature_snapshots: Sequence[FeatureSnapshot],
        raw_signals: Sequence[StrategySignal],
    ) -> tuple[list[StrategySignal], list[dict[str, Any]], list[str], dict[str, Any]]:
        if len(bars) != len(feature_snapshots) or len(bars) != len(raw_signals):
            raise ValueError("Bars, features, and raw signals must have the same length for risk evaluation.")

        settings = self.application.settings
        simulator = PortfolioSimulator(
            initial_capital=settings.backtest.initial_capital,
            fee_bps=settings.backtest.fee_bps,
            slippage_bps=settings.backtest.slippage_bps,
            intrabar_exit_policy=settings.backtest.intrabar_exit_policy,
        )

        day_start_equity = settings.backtest.initial_capital
        current_day: date | None = None
        trades_today = 0
        daily_kill_switch_active = False
        peak_equity = settings.backtest.initial_capital

        filtered_signals: list[StrategySignal] = []
        risk_rows: list[dict[str, Any]] = []
        risk_log_lines: list[str] = []
        blocked_by_reason: dict[str, int] = {}
        kill_switch_days: set[str] = set()
        raw_actionable_signals = 0
        approved_actionable_signals = 0
        blocked_actionable_signals = 0

        for index, (bar, feature, raw_signal) in enumerate(zip(bars, feature_snapshots, raw_signals, strict=True)):
            timestamp = self._normalize_timestamp(bar.timestamp)
            if current_day != timestamp.date():
                current_day = timestamp.date()
                day_start_equity = simulator.equity(bar.open)
                trades_today = 0
                daily_kill_switch_active = False

            trade = simulator.step(
                index=index,
                bar=bar,
                feature=feature,
                exit_zscore_threshold=settings.strategy.exit_zscore,
                session_close_hour_utc=settings.strategy.session_close_hour_utc,
                session_close_minute_utc=settings.strategy.session_close_minute_utc,
            )
            if trade is not None:
                trades_today += 1

            current_equity = simulator.equity(bar.close)
            peak_equity = max(peak_equity, current_equity)
            total_drawdown_pct = ((peak_equity - current_equity) / peak_equity) if peak_equity > 0.0 else 0.0
            daily_pnl_pct = ((current_equity - day_start_equity) / day_start_equity) if day_start_equity > 0.0 else 0.0
            if settings.risk.daily_kill_switch and daily_pnl_pct <= -settings.risk.max_daily_loss:
                daily_kill_switch_active = True
                kill_switch_days.add(timestamp.date().isoformat())

            open_positions = int(simulator.position is not None) + int(simulator.pending_entry is not None)
            session_allowed = is_within_session(
                timestamp,
                start_hour_utc=settings.risk.session_start_hour_utc,
                start_minute_utc=settings.risk.session_start_minute_utc,
                end_hour_utc=settings.risk.session_end_hour_utc,
                end_minute_utc=settings.risk.session_end_minute_utc,
            )
            gross_exposure = (simulator.position.quantity * bar.close) if simulator.position is not None else 0.0
            portfolio = PortfolioState(
                equity=current_equity,
                cash=simulator.cash,
                daily_pnl_pct=daily_pnl_pct,
                open_positions=open_positions,
                gross_exposure=gross_exposure,
                peak_equity=peak_equity,
                total_drawdown_pct=total_drawdown_pct,
                trades_today=trades_today,
                daily_kill_switch_active=daily_kill_switch_active,
                session_allowed=session_allowed,
                timestamp=timestamp.to_pydatetime(),
            )

            decision = self.application.risk_engine.evaluate(raw_signal, portfolio)
            actionable = raw_signal.side in {SignalSide.LONG, SignalSide.SHORT}
            if actionable:
                raw_actionable_signals += 1
                if decision.approved:
                    approved_actionable_signals += 1
                else:
                    blocked_actionable_signals += 1
                    if decision.reason_code:
                        blocked_by_reason[decision.reason_code] = blocked_by_reason.get(decision.reason_code, 0) + 1
                    risk_log_lines.append(self._format_risk_log_line(raw_signal, decision))

            filtered_signal = self._filter_signal(raw_signal, decision)
            execution_ready = False
            if (
                decision.approved
                and actionable
                and index < len(bars) - 1
                and simulator.position is None
                and simulator.pending_entry is None
                and signal_has_executable_levels(filtered_signal)
            ):
                execution_ready = simulator.queue_signal(
                    signal=filtered_signal,
                    index=index,
                    size_fraction=decision.size_fraction,
                    max_leverage=decision.max_leverage,
                )

            filtered_signals.append(filtered_signal)
            risk_rows.append(
                self._build_risk_row(
                    raw_signal=raw_signal,
                    filtered_signal=filtered_signal,
                    decision=decision,
                    portfolio=portfolio,
                    execution_ready=execution_ready,
                )
            )

        risk_summary = {
            "raw_actionable_signals": raw_actionable_signals,
            "approved_actionable_signals": approved_actionable_signals,
            "blocked_actionable_signals": blocked_actionable_signals,
            "blocked_by_reason": dict(sorted(blocked_by_reason.items())),
            "daily_kill_switch_enabled": settings.risk.daily_kill_switch,
            "kill_switch_triggered_days": sorted(kill_switch_days),
            "max_trades_per_day": settings.risk.max_trades_per_day,
            "max_open_positions": settings.risk.max_open_positions,
            "block_outside_session": settings.risk.block_outside_session,
            "require_stop_loss": settings.risk.require_stop_loss,
        }
        return filtered_signals, risk_rows, risk_log_lines, risk_summary

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

    def _prepare_local_frame(
        self,
        frame: pd.DataFrame,
        *,
        allow_gaps: bool,
    ) -> tuple[pd.DataFrame, Any, Any]:
        cleaned_frame, cleaning_report = self.data_service.cleaner.clean(frame)
        validation_report = self.data_service.validator.validate(
            cleaned_frame,
            self.application.settings.market.execution_timeframe,
            allow_gaps=allow_gaps,
        )
        return cleaned_frame, cleaning_report, validation_report

    def _generate_signals(
        self,
        bars: list[MarketBar],
        feature_snapshots: list[FeatureSnapshot],
    ) -> list[StrategySignal]:
        signals: list[StrategySignal] = []
        for bar, feature in zip(bars, feature_snapshots, strict=True):
            adx = feature.values.get("adx_1h")
            regime = (
                "trend"
                if adx is not None
                and math.isfinite(float(adx))
                and float(adx) > self.application.settings.strategy.adx_threshold
                else "range"
            )
            signals.append(
                self.application.strategy.generate(
                    StrategyContext(
                        symbol=self.application.settings.market.symbol,
                        execution_timeframe=self.application.settings.market.execution_timeframe,
                        filter_timeframe=self.application.settings.market.filter_timeframe,
                        bars=[bar],
                        features=[feature],
                        regime=regime,
                    )
                )
            )
        return signals

    def _normalize_timestamp(self, timestamp: object) -> pd.Timestamp:
        normalized = pd.Timestamp(timestamp)
        if normalized.tzinfo is None:
            return normalized.tz_localize("UTC")
        return normalized.tz_convert("UTC")

    def _filter_signal(self, signal: StrategySignal, decision: RiskDecision) -> StrategySignal:
        metadata = dict(signal.metadata)
        metadata.update(decision.metadata)
        metadata.update(
            {
                "raw_side": signal.side.value,
                "raw_rationale": signal.rationale,
                "raw_entry_reason": signal.entry_reason,
                "risk_approved": decision.approved,
                "risk_reason_code": decision.reason_code,
                "risk_blocked_by": list(decision.blocked_by),
                "risk_rationale": decision.rationale,
                "risk_size_fraction": decision.size_fraction,
                "risk_max_leverage": decision.max_leverage,
            }
        )

        if decision.approved and signal.side in {SignalSide.LONG, SignalSide.SHORT}:
            return StrategySignal(
                symbol=signal.symbol,
                timestamp=signal.timestamp,
                side=signal.side,
                strength=signal.strength,
                rationale=signal.rationale,
                entry_price=signal.entry_price,
                stop_price=signal.stop_price,
                target_price=signal.target_price,
                time_stop_bars=signal.time_stop_bars,
                close_on_session_end=signal.close_on_session_end,
                entry_reason=signal.entry_reason,
                metadata=metadata,
            )

        rationale = signal.rationale if signal.side == SignalSide.FLAT else decision.rationale
        return StrategySignal(
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            side=SignalSide.FLAT,
            strength=0.0,
            rationale=rationale,
            entry_price=None,
            stop_price=None,
            target_price=None,
            time_stop_bars=signal.time_stop_bars,
            close_on_session_end=signal.close_on_session_end,
            entry_reason=None,
            metadata=metadata,
        )

    def _build_risk_row(
        self,
        *,
        raw_signal: StrategySignal,
        filtered_signal: StrategySignal,
        decision: RiskDecision,
        portfolio: PortfolioState,
        execution_ready: bool,
    ) -> dict[str, Any]:
        return {
            "timestamp": raw_signal.timestamp,
            "symbol": raw_signal.symbol,
            "raw_side": raw_signal.side.value,
            "filtered_side": filtered_signal.side.value,
            "actionable": raw_signal.side in {SignalSide.LONG, SignalSide.SHORT},
            "approved": decision.approved,
            "execution_ready": execution_ready,
            "reason_code": decision.reason_code,
            "blocked_by": json.dumps(list(decision.blocked_by)),
            "rationale": decision.rationale,
            "entry_price": raw_signal.entry_price,
            "stop_price": raw_signal.stop_price,
            "target_price": raw_signal.target_price,
            "size_fraction": decision.size_fraction,
            "max_leverage": decision.max_leverage,
            "equity": portfolio.equity,
            "cash": portfolio.cash,
            "daily_pnl_pct": portfolio.daily_pnl_pct,
            "total_drawdown_pct": portfolio.total_drawdown_pct,
            "trades_today": portfolio.trades_today,
            "open_positions": portfolio.open_positions,
            "gross_exposure": portfolio.gross_exposure,
            "session_allowed": portfolio.session_allowed,
            "daily_kill_switch_active": portfolio.daily_kill_switch_active,
        }

    def _feature_snapshots_to_frame(self, feature_snapshots: Sequence[FeatureSnapshot]) -> pd.DataFrame:
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
        exported_metadata_keys = [
            "strategy_family",
            "variant_name",
            "regime",
            "anchor_name",
            "anchor_value",
            "breakout_level",
            "breakout_trigger",
            "breakout_distance",
            "breakout_distance_atr",
            "breakout_range_width_atr",
            "momentum",
            "target_to_cost_ratio",
            "expected_move_bps",
        ]
        for signal in signals:
            row = {
                "timestamp": signal.timestamp,
                "symbol": signal.symbol,
                "side": signal.side.value,
                "raw_side": signal.metadata.get("raw_side", signal.side.value),
                "strength": signal.strength,
                "entry_price": signal.entry_price,
                "stop_price": signal.stop_price,
                "target_price": signal.target_price,
                "time_stop_bars": signal.time_stop_bars,
                "close_on_session_end": signal.close_on_session_end,
                "entry_reason": signal.entry_reason,
                "rationale": signal.rationale,
                "risk_approved": signal.metadata.get("risk_approved"),
                "risk_reason_code": signal.metadata.get("risk_reason_code"),
                "risk_blocked_by": json.dumps(signal.metadata.get("risk_blocked_by", [])),
                "risk_size_fraction": signal.metadata.get("risk_size_fraction"),
                "risk_max_leverage": signal.metadata.get("risk_max_leverage"),
            }
            for key in exported_metadata_keys:
                row[key] = signal.metadata.get(key)
            rows.append(row)
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

    def _risk_rows_to_frame(self, rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
        columns = [
            "timestamp",
            "symbol",
            "raw_side",
            "filtered_side",
            "actionable",
            "approved",
            "execution_ready",
            "reason_code",
            "blocked_by",
            "rationale",
            "entry_price",
            "stop_price",
            "target_price",
            "size_fraction",
            "max_leverage",
            "equity",
            "cash",
            "daily_pnl_pct",
            "total_drawdown_pct",
            "trades_today",
            "open_positions",
            "gross_exposure",
            "session_allowed",
            "daily_kill_switch_active",
        ]
        return pd.DataFrame(rows, columns=columns)

    def _build_report_payload(
        self,
        *,
        result: Any,
        validation_report: ValidationReport,
        settings: Settings,
        ohlcv_rows: int,
        feature_rows: int,
        risk_summary: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "symbol": settings.market.symbol,
            "execution_timeframe": settings.market.execution_timeframe,
            "filter_timeframe": settings.market.filter_timeframe,
            "bars": ohlcv_rows,
            "features": feature_rows,
            "start": result.start.isoformat() if result.start else None,
            "end": result.end.isoformat() if result.end else None,
            "number_of_trades": result.trades,
            "win_rate": result.win_rate,
            "payoff": result.payoff,
            "expectancy": result.expectancy,
            "max_drawdown": result.max_drawdown,
            "sharpe": result.sharpe,
            "sortino": result.sortino,
            "calmar": result.calmar,
            "pnl_net": result.pnl_net,
            "total_return": result.total_return,
            "equity_final": result.equity_final,
            "validation": {
                "passed": validation_report.passed,
                "checks": validation_report.checks,
                "summary": validation_report.summary,
            },
            "risk": risk_summary,
            "backtest": {
                key: self._sanitize_value(value)
                for key, value in result.metadata.items()
                if key != "equity_curve"
            },
        }
        return {key: self._sanitize_value(value) for key, value in payload.items()}

    def _build_summary_markdown(self, payload: dict[str, Any]) -> str:
        win_rate_pct = (payload["win_rate"] or 0.0) * 100.0 if payload["win_rate"] is not None else None
        return "\n".join(
            [
                "# Baseline Summary",
                "",
                f"- Symbol: `{payload['symbol']}`",
                f"- Period: `{payload['start']}` -> `{payload['end']}`",
                f"- Bars: `{payload['bars']}`",
                f"- Trades: `{payload['number_of_trades']}`",
                f"- Win rate: `{win_rate_pct:.2f}%`" if win_rate_pct is not None else "- Win rate: `n/a`",
                f"- Payoff: `{payload['payoff']}`",
                f"- Expectancy: `{payload['expectancy']}`",
                f"- Max drawdown: `{payload['max_drawdown']}`",
                f"- Sharpe: `{payload['sharpe']}`",
                f"- Sortino: `{payload['sortino']}`",
                f"- Calmar: `{payload['calmar']}`",
                f"- PnL neto: `{payload['pnl_net']}`",
                (
                    f"- Risk blocked signals: `{payload['risk']['blocked_actionable_signals']}` / "
                    f"`{payload['risk']['raw_actionable_signals']}` actionable"
                ),
                f"- Validation: `{payload['validation']['summary']}`",
            ]
        )

    def _render_risk_log(self, lines: Sequence[str]) -> str:
        if not lines:
            return "# No actionable signals were blocked by the risk engine.\n"
        return "\n".join(lines) + "\n"

    def _format_risk_log_line(self, signal: StrategySignal, decision: RiskDecision) -> str:
        blocked_by = ",".join(decision.blocked_by) if decision.blocked_by else "unknown"
        return (
            f"{pd.Timestamp(signal.timestamp).isoformat()} | raw_side={signal.side.value} "
            f"| reason={decision.reason_code or 'unknown'} | blocked_by={blocked_by} "
            f"| rationale={decision.rationale}"
        )

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._sanitize_value(inner) for key, inner in value.items()}
        if isinstance(value, tuple):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return value
        return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the end-to-end baseline pipeline.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--variant")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output-dir")
    parser.add_argument("--input-path")
    parser.add_argument("--allow-gaps", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = (
        load_variant_settings(args.config_dir, args.variant)
        if args.variant is not None
        else load_settings(args.config_dir)
    )
    runner = BaselineRunner.from_config(args.config_dir, variant_name=args.variant)
    output_dir = Path(args.output_dir) if args.output_dir else Path(settings.storage.artifacts_dir) / "baseline"

    if args.input_path:
        frame = _read_input_frame(args.input_path)
        artifacts = runner.run(
            output_dir=output_dir,
            input_frame=frame,
            allow_gaps=args.allow_gaps or settings.data.allow_gaps,
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
            allow_gaps=args.allow_gaps or settings.data.allow_gaps,
        )

    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    print(f"Baseline report written to {artifacts.report_path}")
    print(f"Summary written to {artifacts.summary_path}")
    print(
        " ".join(
            [
                f"Trades={report['number_of_trades']}",
                f"WinRate={report['win_rate']}",
                f"PnL={report['pnl_net']}",
                f"BlockedSignals={report['risk']['blocked_actionable_signals']}",
            ]
        )
    )
    return 0


def _read_input_frame(path: str | Path) -> pd.DataFrame:
    return read_ohlcv_frame(path)


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
