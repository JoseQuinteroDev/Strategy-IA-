from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from hybrid_quant.bootstrap import TradingApplication, build_application
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
    ParquetDatasetStore,
)


@dataclass(slots=True)
class BaselineArtifacts:
    output_dir: Path
    ohlcv_path: Path
    features_path: Path
    signals_path: Path
    trades_path: Path
    report_path: Path
    summary_path: Path
    result: Any
    validation_report: ValidationReport


@dataclass(slots=True)
class BaselineRunner:
    application: TradingApplication
    data_service: HistoricalDataIngestionService

    @classmethod
    def from_config(cls, config_dir: str | Path) -> "BaselineRunner":
        settings = load_settings(config_dir)
        application = build_application(config_dir)
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
        signals = self._generate_signals(bars, feature_snapshots)

        result = self.application.backtest_engine.run(
            BacktestRequest(
                bars=bars,
                features=feature_snapshots,
                signals=signals,
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
        signal_frame = self._signals_to_frame(signals)
        trade_frame = self._trades_to_frame(result.trade_records)

        ohlcv_path = output_path / "ohlcv.csv"
        features_path = output_path / "features.csv"
        signals_path = output_path / "signals.csv"
        trades_path = output_path / "trades.csv"
        report_path = output_path / "report.json"
        summary_path = output_path / "summary.md"

        ohlcv_frame.to_csv(ohlcv_path)
        feature_frame.to_csv(features_path)
        signal_frame.to_csv(signals_path, index=False)
        trade_frame.to_csv(trades_path, index=False)

        report_payload = self._build_report_payload(
            result=result,
            validation_report=validation_report,
            settings=self.application.settings,
            ohlcv_rows=len(ohlcv_frame),
            feature_rows=len(feature_frame),
        )
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        summary_path.write_text(self._build_summary_markdown(report_payload), encoding="utf-8")

        return BaselineArtifacts(
            output_dir=output_path,
            ohlcv_path=ohlcv_path,
            features_path=features_path,
            signals_path=signals_path,
            trades_path=trades_path,
            report_path=report_path,
            summary_path=summary_path,
            result=result,
            validation_report=validation_report,
        )

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
            regime = "trend" if adx is not None and math.isfinite(float(adx)) and float(adx) > self.application.settings.strategy.adx_threshold else "range"
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
        for signal in signals:
            rows.append(
                {
                    "timestamp": signal.timestamp,
                    "symbol": signal.symbol,
                    "side": signal.side.value,
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

    def _build_report_payload(
        self,
        *,
        result: Any,
        validation_report: ValidationReport,
        settings: Settings,
        ohlcv_rows: int,
        feature_rows: int,
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
                f"- Validation: `{payload['validation']['summary']}`",
            ]
        )

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._sanitize_value(inner) for key, inner in value.items()}
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
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output-dir")
    parser.add_argument("--input-path")
    parser.add_argument("--allow-gaps", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = load_settings(args.config_dir)
    runner = BaselineRunner.from_config(args.config_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(settings.storage.artifacts_dir) / "baseline"

    if args.input_path:
        frame = _read_input_frame(args.input_path)
        artifacts = runner.run(output_dir=output_dir, input_frame=frame, allow_gaps=args.allow_gaps)
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
    print(f"Trades={report['number_of_trades']} WinRate={report['win_rate']} PnL={report['pnl_net']}")
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
