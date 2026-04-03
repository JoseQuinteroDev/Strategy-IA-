from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from hybrid_quant.backtest import IntradayBacktestEngine
from hybrid_quant.baseline import BaselineRunner
from hybrid_quant.core import BacktestRequest, FeatureSnapshot, MarketBar, SignalSide, StrategySignal


def _bar(timestamp: datetime, open_: float, high: float, low: float, close: float) -> MarketBar:
    return MarketBar(
        timestamp=timestamp,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100.0,
    )


def _feature(timestamp: datetime, zscore: float = 0.0) -> FeatureSnapshot:
    return FeatureSnapshot(timestamp=timestamp, values={"zscore_distance_to_mean": zscore}, metadata={})


class BacktestEngineTests(unittest.TestCase):
    def test_engine_executes_target_hit_with_costs_and_position_sizing(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        bars = [
            _bar(start, 100.0, 100.3, 99.7, 100.0),
            _bar(start + timedelta(minutes=5), 100.0, 101.2, 99.9, 100.8),
            _bar(start + timedelta(minutes=10), 100.8, 101.0, 100.4, 100.9),
        ]
        features = [_feature(bar.timestamp) for bar in bars]
        signals = [
            StrategySignal(
                symbol="BTCUSDT",
                timestamp=bars[0].timestamp,
                side=SignalSide.LONG,
                strength=1.0,
                rationale="synthetic long",
                entry_price=100.0,
                stop_price=99.0,
                target_price=101.0,
                time_stop_bars=12,
                close_on_session_end=True,
                entry_reason="synthetic long",
            ),
            StrategySignal(
                symbol="BTCUSDT",
                timestamp=bars[1].timestamp,
                side=SignalSide.FLAT,
                strength=0.0,
                rationale="hold",
            ),
            StrategySignal(
                symbol="BTCUSDT",
                timestamp=bars[2].timestamp,
                side=SignalSide.FLAT,
                strength=0.0,
                rationale="hold",
            ),
        ]

        engine = IntradayBacktestEngine(initial_capital=10_000.0, fee_bps=0.0, slippage_bps=0.0, latency_ms=0)
        result = engine.run(
            BacktestRequest(
                bars=bars,
                features=features,
                signals=signals,
                initial_capital=10_000.0,
                risk_per_trade_fraction=0.01,
                max_leverage=1.0,
            )
        )

        self.assertEqual(result.trades, 1)
        self.assertAlmostEqual(result.pnl_net, 100.0)
        self.assertAlmostEqual(result.win_rate, 1.0)
        self.assertAlmostEqual(result.trade_records[0].quantity, 100.0)
        self.assertEqual(result.trade_records[0].exit_reason, "take_profit")

    def test_engine_forces_session_close_exit(self) -> None:
        bars = [
            _bar(datetime(2024, 1, 1, 23, 40, tzinfo=UTC), 100.0, 100.4, 99.8, 100.0),
            _bar(datetime(2024, 1, 1, 23, 45, tzinfo=UTC), 100.0, 100.2, 99.9, 100.1),
            _bar(datetime(2024, 1, 1, 23, 55, tzinfo=UTC), 100.1, 100.2, 99.9, 100.05),
        ]
        features = [_feature(bar.timestamp, zscore=-3.0) for bar in bars]
        signals = [
            StrategySignal(
                symbol="BTCUSDT",
                timestamp=bars[0].timestamp,
                side=SignalSide.LONG,
                strength=1.0,
                rationale="synthetic long",
                entry_price=100.0,
                stop_price=99.0,
                target_price=101.5,
                time_stop_bars=20,
                close_on_session_end=True,
                entry_reason="synthetic long",
            ),
            StrategySignal(symbol="BTCUSDT", timestamp=bars[1].timestamp, side=SignalSide.FLAT, strength=0.0, rationale="hold"),
            StrategySignal(symbol="BTCUSDT", timestamp=bars[2].timestamp, side=SignalSide.FLAT, strength=0.0, rationale="hold"),
        ]

        engine = IntradayBacktestEngine(initial_capital=10_000.0, fee_bps=0.0, slippage_bps=0.0, latency_ms=0)
        result = engine.run(
            BacktestRequest(
                bars=bars,
                features=features,
                signals=signals,
                initial_capital=10_000.0,
                risk_per_trade_fraction=0.01,
                max_leverage=1.0,
                session_close_hour_utc=23,
                session_close_minute_utc=55,
            )
        )

        self.assertEqual(result.trades, 1)
        self.assertEqual(result.trade_records[0].exit_reason, "session_close")


class BaselineRunnerTests(unittest.TestCase):
    def test_runner_writes_reproducible_artifacts_from_input_frame(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        runner = BaselineRunner.from_config(config_dir)

        index = pd.date_range("2024-01-01T00:00:00Z", periods=24 * 12 * 3, freq="5min", tz="UTC")
        step = pd.Series(range(len(index)), dtype=float)
        close = 100.0 + (step * 0.01) + (step / 8.0).apply(lambda value: __import__("math").sin(value))
        frame = pd.DataFrame(index=index)
        frame["open"] = close - 0.1
        frame["high"] = close + 0.4
        frame["low"] = close - 0.5
        frame["close"] = close
        frame["volume"] = 50.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = runner.run(output_dir=tmp_dir, input_frame=frame)

            self.assertTrue(artifacts.ohlcv_path.exists())
            self.assertTrue(artifacts.features_path.exists())
            self.assertTrue(artifacts.signals_path.exists())
            self.assertTrue(artifacts.trades_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())

