from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hybrid_quant.backtest.engine import IntradayBacktestEngine
from hybrid_quant.baseline.diagnostics import BaselineDiagnosticsRunner
from hybrid_quant.bootstrap import build_application
from hybrid_quant.core import BacktestRequest, FeatureSnapshot, MarketBar, SignalSide, StrategySignal


def _bars() -> tuple[MarketBar, ...]:
    index = pd.date_range("2024-01-03T00:00:00Z", periods=10, freq="5min", tz="UTC")
    rows = [
        (100.0, 100.3, 99.8, 100.0),
        (100.0, 101.7, 99.9, 101.1),
        (101.1, 101.3, 100.7, 100.9),
        (100.9, 101.1, 100.4, 100.6),
        (100.6, 101.9, 100.1, 101.6),
        (101.6, 101.8, 101.0, 101.2),
        (101.2, 102.6, 100.9, 102.3),
        (101.0, 101.2, 100.8, 101.1),
        (101.1, 101.3, 100.9, 101.2),
        (101.2, 101.4, 101.0, 101.3),
    ]
    return tuple(
        MarketBar(
            timestamp=timestamp.to_pydatetime(),
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=100.0,
        )
        for timestamp, (open_, high, low, close) in zip(index, rows, strict=True)
    )


def _features(bars: tuple[MarketBar, ...]) -> tuple[FeatureSnapshot, ...]:
    return tuple(
        FeatureSnapshot(
            timestamp=bar.timestamp,
            values={
                "log_return": 0.0,
                "atr_14": 1.0,
                "ema_200_1h": 99.0,
                "adx_1h": 10.0,
                "intraday_vwap": 100.5,
                "ema_50": 100.4,
                "realized_volatility_20": 0.01,
                "candle_range": bar.high - bar.low,
                "zscore_distance_to_mean": 2.6 if index in {0, 5} else 0.5,
                "hour_utc": float(pd.Timestamp(bar.timestamp).hour),
                "day_of_week": float(pd.Timestamp(bar.timestamp).day_of_week),
                "session_asia": 1.0,
                "session_europe": 0.0,
                "session_us": 0.0,
            },
            metadata={},
        )
        for index, bar in enumerate(bars)
    )


def _signals(bars: tuple[MarketBar, ...]) -> tuple[StrategySignal, ...]:
    signals: list[StrategySignal] = []
    for index, bar in enumerate(bars):
        if index == 0:
            signals.append(
                StrategySignal(
                    symbol="BTCUSDT",
                    timestamp=bar.timestamp,
                    side=SignalSide.LONG,
                    strength=1.0,
                    rationale="synthetic long",
                    entry_price=bar.close,
                    stop_price=bar.close - 1.0,
                    target_price=bar.close + 1.0,
                    time_stop_bars=4,
                    close_on_session_end=True,
                    entry_reason="synthetic long",
                )
            )
        elif index == 5:
            signals.append(
                StrategySignal(
                    symbol="BTCUSDT",
                    timestamp=bar.timestamp,
                    side=SignalSide.SHORT,
                    strength=1.0,
                    rationale="synthetic short",
                    entry_price=bar.close,
                    stop_price=bar.close + 1.0,
                    target_price=bar.close - 1.0,
                    time_stop_bars=4,
                    close_on_session_end=True,
                    entry_reason="synthetic short",
                )
            )
        else:
            signals.append(
                StrategySignal(
                    symbol="BTCUSDT",
                    timestamp=bar.timestamp,
                    side=SignalSide.FLAT,
                    strength=0.0,
                    rationale="flat",
                    time_stop_bars=4,
                    close_on_session_end=True,
                )
            )
    return tuple(signals)


def _write_synthetic_artifact(config_dir: Path, artifact_dir: Path) -> None:
    app = build_application(config_dir)
    bars = _bars()
    features = _features(bars)
    signals = _signals(bars)
    result = IntradayBacktestEngine(
        initial_capital=app.settings.backtest.initial_capital,
        fee_bps=app.settings.backtest.fee_bps,
        slippage_bps=app.settings.backtest.slippage_bps,
        latency_ms=app.settings.backtest.latency_ms,
        intrabar_exit_policy=app.settings.backtest.intrabar_exit_policy,
    ).run(
        BacktestRequest(
            bars=bars,
            features=features,
            signals=signals,
            initial_capital=app.settings.backtest.initial_capital,
            risk_per_trade_fraction=app.settings.risk.max_risk_per_trade,
            max_leverage=app.settings.risk.max_leverage,
            signal_cooldown_bars=app.settings.strategy.signal_cooldown_bars,
            exit_zscore_threshold=app.settings.strategy.exit_zscore,
            session_close_hour_utc=app.settings.strategy.session_close_hour_utc,
            session_close_minute_utc=app.settings.strategy.session_close_minute_utc,
            intrabar_exit_policy=app.settings.backtest.intrabar_exit_policy,
        )
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    ohlcv_frame = pd.DataFrame(
        [
            {
                "open_time": pd.Timestamp(bar.timestamp).tz_convert("UTC"),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
    ).set_index("open_time")
    ohlcv_frame.to_csv(artifact_dir / "ohlcv.csv")

    feature_frame = pd.DataFrame(
        [{"open_time": pd.Timestamp(snapshot.timestamp).tz_convert("UTC"), **snapshot.values} for snapshot in features]
    ).set_index("open_time")
    feature_frame.to_csv(artifact_dir / "features.csv")

    signal_frame = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(signal.timestamp).tz_convert("UTC"),
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
            for signal in signals
        ]
    )
    signal_frame.to_csv(artifact_dir / "signals.csv", index=False)

    trade_frame = pd.DataFrame(
        [
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
            for trade in result.trade_records
        ]
    )
    trade_frame.to_csv(artifact_dir / "trades.csv", index=False)

    report = {
        "symbol": "BTCUSDT",
        "execution_timeframe": "5m",
        "filter_timeframe": "1H",
        "bars": len(bars),
        "features": len(features),
        "start": bars[0].timestamp.isoformat(),
        "end": bars[-1].timestamp.isoformat(),
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
        "validation": {"passed": False, "checks": {}, "summary": "synthetic"},
        "backtest": result.metadata,
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (artifact_dir / "summary.md").write_text("# synthetic\n", encoding="utf-8")


class BaselineDiagnosticsRunnerTests(unittest.TestCase):
    def test_diagnostics_runner_generates_expected_artifacts(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        runner = BaselineDiagnosticsRunner.from_config(config_dir)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            artifact_dir = tmp_path / "artifact"
            output_dir = tmp_path / "diagnostics"
            _write_synthetic_artifact(config_dir, artifact_dir)

            artifacts = runner.run(artifact_dir=artifact_dir, output_dir=output_dir)

            self.assertTrue(artifacts.diagnostics_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.monthly_breakdown_path.exists())
            self.assertTrue(artifacts.hourly_breakdown_path.exists())
            self.assertTrue(artifacts.exit_reason_breakdown_path.exists())
            self.assertTrue(artifacts.side_breakdown_path.exists())
            self.assertTrue(artifacts.cost_impact_path.exists())
            self.assertTrue(artifacts.variant_comparison_path.exists())
            self.assertTrue(artifacts.risk_execution_breakdown_path.exists())

            payload = json.loads(artifacts.diagnostics_path.read_text(encoding="utf-8"))
            self.assertIn("baseline_metrics", payload)
            self.assertIn("automatic_conclusion", payload)
            self.assertIn("question_answer", payload)
            self.assertEqual(payload["baseline_metrics"]["number_of_trades"], 2)

            variants = pd.read_csv(artifacts.variant_comparison_path)
            self.assertTrue({"baseline", "no_fees", "no_slippage", "no_costs"}.issubset(set(variants["variant"])))
