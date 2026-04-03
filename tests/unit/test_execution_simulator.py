from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from hybrid_quant.core import FeatureSnapshot, MarketBar, SignalSide, StrategySignal
from hybrid_quant.execution import (
    PortfolioSimulator,
    build_pending_entry,
    is_session_close,
    is_within_session,
    resolve_intrabar_policy,
    signal_has_executable_levels,
)


def _bar(timestamp: datetime, open_: float, high: float, low: float, close: float) -> MarketBar:
    return MarketBar(
        timestamp=timestamp,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100.0,
    )


def _feature(timestamp: datetime, zscore: float = 2.0) -> FeatureSnapshot:
    return FeatureSnapshot(timestamp=timestamp, values={"zscore_distance_to_mean": zscore}, metadata={})


def _signal(timestamp: datetime) -> StrategySignal:
    return StrategySignal(
        symbol="BTCUSDT",
        timestamp=timestamp,
        side=SignalSide.LONG,
        strength=1.0,
        rationale="synthetic long",
        entry_price=100.0,
        stop_price=99.0,
        target_price=101.0,
        time_stop_bars=12,
        close_on_session_end=True,
        entry_reason="synthetic long",
    )


class ExecutionSimulatorTests(unittest.TestCase):
    def test_portfolio_simulator_reproduces_target_hit_path(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        simulator = PortfolioSimulator(
            initial_capital=10_000.0,
            fee_bps=0.0,
            slippage_bps=0.0,
            intrabar_exit_policy="conservative",
        )
        signal = _signal(start)

        queued = simulator.queue_signal(signal=signal, index=0, size_fraction=0.01, max_leverage=1.0)
        self.assertTrue(queued)

        trade = simulator.step(
            index=1,
            bar=_bar(start + timedelta(minutes=5), 100.0, 101.2, 99.9, 100.8),
            feature=_feature(start + timedelta(minutes=5)),
            exit_zscore_threshold=0.5,
            session_close_hour_utc=23,
            session_close_minute_utc=55,
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade.exit_reason, "take_profit")
        self.assertAlmostEqual(trade.net_pnl, 100.0)
        self.assertAlmostEqual(simulator.cash, 10_100.0)
        self.assertAlmostEqual(simulator.equity(100.8), 10_100.0)

    def test_public_session_helpers_handle_regular_and_overnight_windows(self) -> None:
        morning = datetime(2024, 1, 1, 9, 0, tzinfo=UTC)
        late_night = datetime(2024, 1, 1, 23, 30, tzinfo=UTC)

        self.assertTrue(
            is_within_session(
                morning,
                start_hour_utc=8,
                start_minute_utc=0,
                end_hour_utc=16,
                end_minute_utc=0,
            )
        )
        self.assertTrue(
            is_within_session(
                late_night,
                start_hour_utc=22,
                start_minute_utc=0,
                end_hour_utc=2,
                end_minute_utc=0,
            )
        )
        self.assertTrue(is_session_close(datetime(2024, 1, 1, 23, 55, tzinfo=UTC), 23, 55))

    def test_public_signal_and_policy_helpers_validate_inputs(self) -> None:
        signal = _signal(datetime(2024, 1, 1, 0, 0, tzinfo=UTC))
        pending = build_pending_entry(signal=signal, generated_index=0, size_fraction=0.01, max_leverage=1.0)

        self.assertTrue(signal_has_executable_levels(signal))
        self.assertIsNotNone(pending)
        self.assertEqual(resolve_intrabar_policy(None, "conservative"), "conservative")

        with self.assertRaises(ValueError):
            resolve_intrabar_policy("unsupported", "conservative")
