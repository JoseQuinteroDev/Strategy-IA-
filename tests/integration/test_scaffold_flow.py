from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from hybrid_quant.backtest import IntradayBacktestEngine
from hybrid_quant.core import (
    BacktestRequest,
    MarketBar,
    MarketDataBatch,
    PortfolioState,
    SignalSide,
    StrategySignal,
    StrategyContext,
)
from hybrid_quant.env import HybridTradingEnvironment
from hybrid_quant.features import FeaturePipeline
from hybrid_quant.paper import PaperTradingRunner
from hybrid_quant.risk import PropFirmRiskEngine
from hybrid_quant.rl import PPOTrainer
from hybrid_quant.strategy import MeanReversionStrategy
from hybrid_quant.validation import WalkForwardValidator


class ScaffoldFlowTests(unittest.TestCase):
    def test_end_to_end_scaffold_remains_executable(self) -> None:
        bars = [
            MarketBar(
                timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=UTC) + timedelta(minutes=5 * index),
                open=100.0 + index,
                high=101.0 + index,
                low=99.0 + index,
                close=100.5 + index,
                volume=10.0 + index,
            )
            for index in range(4)
        ]

        pipeline = FeaturePipeline(
            feature_names=["log_return", "candle_range", "hour_utc"],
            lookback_window=96,
            regime_window=288,
            normalize=True,
        )
        features = pipeline.transform(
            MarketDataBatch(symbol="BTCUSDT", timeframe="5m", bars=bars, metadata={})
        )

        strategy = MeanReversionStrategy(
            name="mean_reversion_trend_regime",
            variant_name="baseline_v1",
            entry_zscore=2.0,
            exit_zscore=0.5,
            trend_filter="ema_200_1h",
            regime_filter="adx_1h",
            execution_timeframe="5m",
            filter_timeframe="1H",
            mean_reversion_anchor="vwap",
            adx_threshold=25.0,
            atr_multiple_stop=1.0,
            atr_multiple_target=1.0,
            time_stop_bars=12,
            session_close_hour_utc=23,
            session_close_minute_utc=55,
            no_entry_minutes_before_close=30,
        )
        signal = strategy.generate(
            StrategyContext(
                symbol="BTCUSDT",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=bars,
                features=features,
                regime="neutral",
            )
        )

        risk_engine = PropFirmRiskEngine(
            max_risk_per_trade=0.0025,
            max_daily_loss=0.03,
            max_open_positions=1,
            max_leverage=2.0,
            prop_firm_mode=True,
        )
        decision = risk_engine.evaluate(signal, PortfolioState())

        backtest_engine = IntradayBacktestEngine(
            initial_capital=100000.0,
            fee_bps=4.0,
            slippage_bps=2.0,
            latency_ms=250,
        )
        result = backtest_engine.run(
            BacktestRequest(
                bars=bars,
                features=features,
                signal=signal,
                initial_capital=100000.0,
            )
        )

        validator = WalkForwardValidator(
            walk_forward_splits=4,
            min_trades=1,
            max_drawdown_limit=0.08,
            sharpe_floor=0.0,
        )
        report = validator.validate(result)

        environment = HybridTradingEnvironment(
            observation_window=64,
            max_steps=100,
            reward_mode="risk_adjusted",
            risk_engine=risk_engine,
            initial_capital=100000.0,
            fee_bps=0.0,
            slippage_bps=0.0,
        )
        candidate_signals = [
            StrategySignal(
                symbol="BTCUSDT",
                timestamp=bars[0].timestamp,
                side=SignalSide.LONG,
                strength=1.0,
                rationale="synthetic env candidate",
                entry_price=bars[0].close,
                stop_price=bars[0].close - 1.0,
                target_price=bars[0].close + 1.0,
                time_stop_bars=12,
                close_on_session_end=True,
                entry_reason="synthetic env candidate",
            ),
            StrategySignal(symbol="BTCUSDT", timestamp=bars[1].timestamp, side=SignalSide.FLAT, strength=0.0, rationale="hold"),
            StrategySignal(symbol="BTCUSDT", timestamp=bars[2].timestamp, side=SignalSide.FLAT, strength=0.0, rationale="hold"),
            StrategySignal(symbol="BTCUSDT", timestamp=bars[3].timestamp, side=SignalSide.FLAT, strength=0.0, rationale="hold"),
        ]
        environment.attach_market_data(bars, features, candidate_signals=candidate_signals)
        initial_observation, reset_info = environment.reset()
        next_observation, reward, terminated, truncated, info = environment.step(HybridTradingEnvironment.ACTION_TAKE_TRADE)

        trainer = PPOTrainer(algorithm="PPO", total_timesteps=1000000, enabled=False)
        artifact = trainer.fit(environment)

        paper = PaperTradingRunner(venue="simulator", dry_run=True, heartbeat_seconds=30)
        execution = paper.submit(signal, decision)

        self.assertEqual(len(features), len(bars))
        self.assertEqual(signal.side, SignalSide.FLAT)
        self.assertFalse(decision.approved)
        self.assertEqual(result.metadata["mode"], "baseline")
        self.assertFalse(report.passed)
        self.assertEqual(initial_observation.shape, environment.observation_space.shape)
        self.assertEqual(next_observation.shape, environment.observation_space.shape)
        self.assertIn("observation_keys", reset_info)
        self.assertNotEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["risk_approved"])
        self.assertIn(info["trades_closed"][0], {"stop_loss", "take_profit"})
        self.assertEqual(artifact.status, "disabled")
        self.assertFalse(execution.accepted)
