from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from hybrid_quant.backtest import IntradayBacktestEngine
from hybrid_quant.baseline import BaselineRunner
from hybrid_quant.baseline.runner import _filter_frame_by_range
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

    def test_intrabar_policy_stop_first_prefers_stop_when_both_levels_hit(self) -> None:
        result = self._run_intrabar_collision(policy="stop_first")

        self.assertEqual(result.trades, 1)
        self.assertEqual(result.trade_records[0].exit_reason, "stop_loss")
        self.assertLess(result.trade_records[0].net_pnl, 0.0)

    def test_intrabar_policy_target_first_prefers_target_when_both_levels_hit(self) -> None:
        result = self._run_intrabar_collision(policy="target_first")

        self.assertEqual(result.trades, 1)
        self.assertEqual(result.trade_records[0].exit_reason, "take_profit")
        self.assertGreater(result.trade_records[0].net_pnl, 0.0)

    def test_intrabar_policy_conservative_prefers_worse_outcome_when_both_levels_hit(self) -> None:
        result = self._run_intrabar_collision(policy="conservative")

        self.assertEqual(result.trades, 1)
        self.assertEqual(result.trade_records[0].exit_reason, "stop_loss")
        self.assertLess(result.trade_records[0].net_pnl, 0.0)

    def test_invalid_intrabar_policy_raises_clear_error(self) -> None:
        bars = [
            _bar(datetime(2024, 1, 1, 0, 0, tzinfo=UTC), 100.0, 100.3, 99.7, 100.0),
            _bar(datetime(2024, 1, 1, 0, 5, tzinfo=UTC), 100.0, 101.2, 98.8, 100.5),
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
            ),
            StrategySignal(symbol="BTCUSDT", timestamp=bars[1].timestamp, side=SignalSide.FLAT, strength=0.0, rationale="hold"),
        ]

        engine = IntradayBacktestEngine(
            initial_capital=10_000.0,
            fee_bps=0.0,
            slippage_bps=0.0,
            latency_ms=0,
            intrabar_exit_policy="unsupported",
        )

        with self.assertRaises(ValueError):
            engine.run(
                BacktestRequest(
                    bars=bars,
                    features=features,
                    signals=signals,
                    initial_capital=10_000.0,
                )
            )

    def _run_intrabar_collision(self, *, policy: str):
        start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        bars = [
            _bar(start, 100.0, 100.2, 99.8, 100.0),
            _bar(start + timedelta(minutes=5), 100.0, 101.3, 98.7, 100.2),
            _bar(start + timedelta(minutes=10), 100.2, 100.5, 99.9, 100.1),
        ]
        features = [_feature(bar.timestamp) for bar in bars]
        signals = [
            StrategySignal(
                symbol="BTCUSDT",
                timestamp=bars[0].timestamp,
                side=SignalSide.LONG,
                strength=1.0,
                rationale="intrabar collision",
                entry_price=100.0,
                stop_price=99.0,
                target_price=101.0,
                time_stop_bars=12,
                close_on_session_end=True,
                entry_reason="intrabar collision",
            ),
            StrategySignal(symbol="BTCUSDT", timestamp=bars[1].timestamp, side=SignalSide.FLAT, strength=0.0, rationale="hold"),
            StrategySignal(symbol="BTCUSDT", timestamp=bars[2].timestamp, side=SignalSide.FLAT, strength=0.0, rationale="hold"),
        ]

        engine = IntradayBacktestEngine(
            initial_capital=10_000.0,
            fee_bps=0.0,
            slippage_bps=0.0,
            latency_ms=0,
            intrabar_exit_policy=policy,
        )
        return engine.run(
            BacktestRequest(
                bars=bars,
                features=features,
                signals=signals,
                initial_capital=10_000.0,
                risk_per_trade_fraction=0.01,
                max_leverage=1.0,
            )
        )


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
            self.assertTrue(artifacts.risk_decisions_path.exists())
            self.assertTrue(artifacts.risk_log_path.exists())
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())

    def test_filter_frame_by_range_supports_local_cli_windows(self) -> None:
        index = pd.date_range("2024-01-01T00:00:00Z", periods=12, freq="5min", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
            },
            index=index,
        )
        frame.index.name = "open_time"

        filtered = _filter_frame_by_range(
            frame,
            start=datetime(2024, 1, 1, 0, 10, tzinfo=UTC),
            end=datetime(2024, 1, 1, 0, 30, tzinfo=UTC),
        )

        self.assertEqual(len(filtered), 5)
        self.assertEqual(filtered.index[0], pd.Timestamp("2024-01-01T00:10:00Z"))
        self.assertEqual(filtered.index[-1], pd.Timestamp("2024-01-01T00:30:00Z"))

    def test_filter_frame_by_range_raises_on_empty_slice(self) -> None:
        index = pd.date_range("2024-01-01T00:00:00Z", periods=4, freq="5min", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
            },
            index=index,
        )
        frame.index.name = "open_time"

        with self.assertRaises(ValueError):
            _filter_frame_by_range(
                frame,
                start=datetime(2024, 2, 1, 0, 0, tzinfo=UTC),
                end=datetime(2024, 2, 1, 1, 0, tzinfo=UTC),
            )

            report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
            required_keys = {
                "symbol",
                "execution_timeframe",
                "filter_timeframe",
                "number_of_trades",
                "win_rate",
                "payoff",
                "expectancy",
                "max_drawdown",
                "sharpe",
                "sortino",
                "calmar",
                "pnl_net",
                "total_return",
                "equity_final",
                "validation",
                "risk",
                "backtest",
            }
            self.assertTrue(required_keys.issubset(report.keys()))
            self.assertGreaterEqual(report["number_of_trades"], 0)
            self.assertGreaterEqual(report["win_rate"], 0.0)
            self.assertLessEqual(report["win_rate"], 1.0)
            self.assertGreaterEqual(report["max_drawdown"], 0.0)
            self.assertAlmostEqual(
                report["equity_final"],
                100000.0 + report["pnl_net"],
                places=6,
            )
            self.assertAlmostEqual(
                report["total_return"],
                report["pnl_net"] / 100000.0,
                places=6,
            )
            self.assertIn("intrabar_exit_policy", report["backtest"])
            self.assertIn("blocked_actionable_signals", report["risk"])
            self.assertIn("blocked_by_reason", report["risk"])

            trades_frame = pd.read_csv(artifacts.trades_path)
            expected_trade_columns = [
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
            self.assertEqual(list(trades_frame.columns), expected_trade_columns)
            self.assertEqual(len(trades_frame), report["number_of_trades"])
            if not trades_frame.empty:
                pd.to_datetime(trades_frame["entry_timestamp"], utc=True)
                pd.to_datetime(trades_frame["exit_timestamp"], utc=True)
                for column in ["entry_price", "exit_price", "quantity", "gross_pnl", "net_pnl", "fees_paid"]:
                    self.assertTrue(pd.to_numeric(trades_frame[column], errors="coerce").notna().all())

            risk_frame = pd.read_csv(artifacts.risk_decisions_path)
            self.assertIn("reason_code", risk_frame.columns)
            self.assertIn("approved", risk_frame.columns)
            self.assertEqual(len(risk_frame), len(frame))

            risk_log = artifacts.risk_log_path.read_text(encoding="utf-8")
            self.assertTrue(risk_log.strip())

    def test_runner_supports_trend_nasdaq_variant_from_config(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        runner = BaselineRunner.from_config(config_dir, variant_name="baseline_trend_nasdaq")
        self.assertEqual(runner.application.strategy.__class__.__name__, "TrendBreakoutStrategy")

        session_index = pd.date_range("2024-01-02T13:30:00Z", periods=78 * 3, freq="5min", tz="UTC")
        step = pd.Series(range(len(session_index)), dtype=float)
        close = 17000.0 + (step * 0.8)
        frame = pd.DataFrame(index=session_index)
        frame["open"] = close - 0.4
        frame["high"] = close + 1.6
        frame["low"] = close - 1.0
        frame["close"] = close
        frame["volume"] = 200.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = runner.run(output_dir=tmp_dir, input_frame=frame, allow_gaps=True)

            report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["symbol"], "NQ")
            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.signals_path.exists())
            self.assertTrue(artifacts.trades_path.exists())
