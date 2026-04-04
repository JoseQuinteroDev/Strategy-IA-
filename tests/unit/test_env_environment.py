from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

import numpy as np

from hybrid_quant.core import FeatureSnapshot, MarketBar, SignalSide, StrategySignal
from hybrid_quant.env import HybridTradingEnvironment
from hybrid_quant.risk import PropFirmRiskEngine


def _bar(timestamp: datetime, open_: float, high: float, low: float, close: float) -> MarketBar:
    return MarketBar(
        timestamp=timestamp,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100.0,
    )


def _feature(timestamp: datetime, zscore: float = 1.0) -> FeatureSnapshot:
    return FeatureSnapshot(
        timestamp=timestamp,
        values={
            "log_return": 0.0,
            "zscore_distance_to_mean": zscore,
            "atr_14": 1.0,
            "adx_1h": 12.0,
        },
        metadata={},
    )


def _signal(
    timestamp: datetime,
    *,
    side: SignalSide = SignalSide.LONG,
    entry_price: float | None = 100.0,
    stop_price: float | None = 99.0,
    target_price: float | None = 101.0,
) -> StrategySignal:
    return StrategySignal(
        symbol="BTCUSDT",
        timestamp=timestamp,
        side=side,
        strength=1.0 if side != SignalSide.FLAT else 0.0,
        rationale="synthetic signal",
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        time_stop_bars=12,
        close_on_session_end=True,
        entry_reason="synthetic signal" if side != SignalSide.FLAT else None,
    )


class HybridTradingEnvironmentTests(unittest.TestCase):
    def _build_environment(self) -> HybridTradingEnvironment:
        return HybridTradingEnvironment(
            observation_window=64,
            max_steps=50,
            reward_mode="risk_adjusted",
            risk_engine=PropFirmRiskEngine(
                max_risk_per_trade=0.01,
                max_daily_loss=0.03,
                max_total_drawdown=0.2,
                daily_kill_switch=True,
                max_trades_per_day=6,
                max_open_positions=1,
                max_leverage=1.0,
                block_outside_session=True,
                session_start_hour_utc=0,
                session_start_minute_utc=0,
                session_end_hour_utc=23,
                session_end_minute_utc=55,
                require_stop_loss=True,
            ),
            initial_capital=10_000.0,
            fee_bps=0.0,
            slippage_bps=0.0,
            intrabar_exit_policy="conservative",
        )

    def test_reset_returns_expected_observation_shape(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        env = self._build_environment()
        bars = [_bar(start + timedelta(minutes=5 * idx), 100.0, 100.2, 99.8, 100.0) for idx in range(3)]
        features = [_feature(bar.timestamp) for bar in bars]
        signals = [_signal(bars[0].timestamp), _signal(bars[1].timestamp, side=SignalSide.FLAT), _signal(bars[2].timestamp, side=SignalSide.FLAT)]
        env.attach_market_data(bars, features, candidate_signals=signals)

        observation, info = env.reset()

        self.assertEqual(env.action_space.n, 3)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertIn("observation_keys", info)
        self.assertEqual(info["observation_mode"], HybridTradingEnvironment.OBSERVATION_MODE)
        self.assertEqual(info["state_context_bars"], 64)
        self.assertEqual(info["reward_breakdown"]["reward"], 0.0)
        self.assertTrue(info["candidate_actionable"])

    def test_take_trade_action_drives_simulator_and_nonzero_reward(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        env = self._build_environment()
        bars = [
            _bar(start, 100.0, 100.2, 99.8, 100.0),
            _bar(start + timedelta(minutes=5), 100.0, 101.2, 99.9, 100.8),
            _bar(start + timedelta(minutes=10), 100.8, 101.0, 100.5, 100.9),
        ]
        features = [_feature(bar.timestamp) for bar in bars]
        signals = [
            _signal(bars[0].timestamp),
            _signal(bars[1].timestamp, side=SignalSide.FLAT),
            _signal(bars[2].timestamp, side=SignalSide.FLAT),
        ]
        env.attach_market_data(bars, features, candidate_signals=signals)
        env.reset()

        observation, reward, terminated, truncated, info = env.step(HybridTradingEnvironment.ACTION_TAKE_TRADE)

        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertGreater(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["action_effect"], "trade_queued")
        self.assertIn("reward_breakdown", info)
        self.assertAlmostEqual(info["reward_breakdown"]["reward"], reward)
        self.assertIn("take_profit", info["trades_closed"])

    def test_close_early_action_flattens_open_position(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        env = self._build_environment()
        bars = [
            _bar(start, 100.0, 100.2, 99.8, 100.0),
            _bar(start + timedelta(minutes=5), 100.0, 100.4, 99.7, 100.3),
            _bar(start + timedelta(minutes=10), 100.3, 100.5, 100.1, 100.2),
        ]
        features = [_feature(bar.timestamp, zscore=2.0) for bar in bars]
        signals = [
            _signal(bars[0].timestamp, target_price=105.0),
            _signal(bars[1].timestamp, side=SignalSide.FLAT),
            _signal(bars[2].timestamp, side=SignalSide.FLAT),
        ]
        env.attach_market_data(bars, features, candidate_signals=signals)
        env.reset()

        env.step(HybridTradingEnvironment.ACTION_TAKE_TRADE)
        _, reward, _, _, info = env.step(HybridTradingEnvironment.ACTION_CLOSE_EARLY)

        self.assertEqual(reward, 0.0)
        self.assertEqual(info["action_effect"], "closed_early")
        self.assertIn("agent_close_early", info["trades_closed"])

    def test_blocked_trade_attempt_produces_negative_reward_and_reason(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        env = self._build_environment()
        bars = [
            _bar(start, 100.0, 100.2, 99.8, 100.0),
            _bar(start + timedelta(minutes=5), 100.0, 100.1, 99.9, 100.0),
        ]
        features = [_feature(bar.timestamp) for bar in bars]
        signals = [
            _signal(bars[0].timestamp, stop_price=None),
            _signal(bars[1].timestamp, side=SignalSide.FLAT),
        ]
        env.attach_market_data(bars, features, candidate_signals=signals)
        env.reset()

        _, reward, _, _, info = env.step(HybridTradingEnvironment.ACTION_TAKE_TRADE)

        self.assertLess(reward, 0.0)
        self.assertEqual(info["risk_reason_code"], "missing_or_invalid_stop_loss")
        self.assertEqual(info["action_effect"], "trade_blocked")
        self.assertEqual(info["trades_closed"], [])
        self.assertFalse(np.isclose(reward, 0.0))

    def test_state_context_alias_remains_backward_compatible(self) -> None:
        legacy_env = HybridTradingEnvironment(observation_window=32, max_steps=10, reward_mode="risk_adjusted")
        canonical_env = HybridTradingEnvironment(state_context_bars=48, max_steps=10, reward_mode="risk_adjusted")

        self.assertEqual(legacy_env.state_context_bars, 32)
        self.assertEqual(legacy_env.observation_window, 32)
        self.assertEqual(canonical_env.state_context_bars, 48)
        self.assertEqual(canonical_env.observation_window, 48)
