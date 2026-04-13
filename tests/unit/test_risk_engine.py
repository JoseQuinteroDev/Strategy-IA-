from __future__ import annotations

import unittest
from datetime import UTC, datetime

from hybrid_quant.core import PortfolioState, SignalSide, StrategySignal
from hybrid_quant.risk import PropFirmRiskEngine


def _signal(
    *,
    timestamp: datetime,
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


class PropFirmRiskEngineTests(unittest.TestCase):
    def test_approves_valid_signal_inside_session(self) -> None:
        engine = PropFirmRiskEngine(
            max_risk_per_trade=0.0025,
            max_daily_loss=0.03,
            max_total_drawdown=0.1,
            daily_kill_switch=True,
            max_trades_per_day=6,
            max_open_positions=1,
            max_leverage=2.0,
            block_outside_session=True,
            session_start_hour_utc=0,
            session_start_minute_utc=0,
            session_end_hour_utc=23,
            session_end_minute_utc=55,
            require_stop_loss=True,
        )
        decision = engine.evaluate(
            _signal(timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=UTC)),
            PortfolioState(),
        )

        self.assertTrue(decision.approved)
        self.assertEqual(decision.reason_code, "approved")
        self.assertAlmostEqual(decision.size_fraction, 0.0025)

    def test_blocks_missing_or_invalid_stop_loss(self) -> None:
        engine = PropFirmRiskEngine(require_stop_loss=True)
        decision = engine.evaluate(
            _signal(
                timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
                stop_price=None,
            ),
            PortfolioState(),
        )

        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason_code, "missing_or_invalid_stop_loss")
        self.assertIn("missing_or_invalid_stop_loss", decision.blocked_by)

    def test_blocks_signal_outside_session_using_utc_clock(self) -> None:
        engine = PropFirmRiskEngine(
            block_outside_session=True,
            session_start_hour_utc=8,
            session_start_minute_utc=0,
            session_end_hour_utc=16,
            session_end_minute_utc=0,
        )
        decision = engine.evaluate(
            _signal(timestamp=datetime(2024, 1, 1, 7, 55, tzinfo=UTC)),
            PortfolioState(),
        )

        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason_code, "outside_session")

    def test_blocks_nq_baseline_entries_outside_14_to_19_window(self) -> None:
        engine = PropFirmRiskEngine(
            block_outside_session=True,
            session_start_hour_utc=14,
            session_start_minute_utc=0,
            session_end_hour_utc=19,
            session_end_minute_utc=0,
        )

        before_open = engine.evaluate(
            _signal(timestamp=datetime(2024, 1, 1, 13, 55, tzinfo=UTC)),
            PortfolioState(),
        )
        inside = engine.evaluate(
            _signal(timestamp=datetime(2024, 1, 1, 14, 0, tzinfo=UTC)),
            PortfolioState(),
        )
        after_close = engine.evaluate(
            _signal(timestamp=datetime(2024, 1, 1, 19, 5, tzinfo=UTC)),
            PortfolioState(),
        )

        self.assertEqual(before_open.reason_code, "outside_session")
        self.assertTrue(inside.approved)
        self.assertEqual(after_close.reason_code, "outside_session")

    def test_blocks_outside_multiple_madrid_session_windows(self) -> None:
        engine = PropFirmRiskEngine(
            block_outside_session=True,
            session_timezone="Europe/Madrid",
            session_windows=["09:00-11:00", "14:00-16:30"],
        )

        morning = engine.evaluate(
            _signal(timestamp=datetime(2024, 1, 2, 8, 15, tzinfo=UTC)),
            PortfolioState(),
        )
        lunch = engine.evaluate(
            _signal(timestamp=datetime(2024, 1, 2, 11, 30, tzinfo=UTC)),
            PortfolioState(),
        )
        afternoon = engine.evaluate(
            _signal(timestamp=datetime(2024, 1, 2, 14, 15, tzinfo=UTC)),
            PortfolioState(),
        )

        self.assertTrue(morning.approved)
        self.assertEqual(lunch.reason_code, "outside_session")
        self.assertTrue(afternoon.approved)

    def test_blocks_after_configured_consecutive_losses_per_day(self) -> None:
        engine = PropFirmRiskEngine(max_consecutive_losses_per_day=2)
        decision = engine.evaluate(
            _signal(timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=UTC)),
            PortfolioState(consecutive_losses_today=2),
        )

        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason_code, "max_consecutive_losses_per_day")

    def test_session_filter_can_use_new_york_timezone_with_dst(self) -> None:
        engine = PropFirmRiskEngine(
            block_outside_session=True,
            session_start_hour_utc=9,
            session_start_minute_utc=30,
            session_end_hour_utc=14,
            session_end_minute_utc=30,
            session_timezone="America/New_York",
        )

        inside_summer = engine.evaluate(
            _signal(timestamp=datetime(2024, 7, 1, 13, 35, tzinfo=UTC)),
            PortfolioState(),
        )
        outside_summer = engine.evaluate(
            _signal(timestamp=datetime(2024, 7, 1, 13, 25, tzinfo=UTC)),
            PortfolioState(),
        )
        inside_winter = engine.evaluate(
            _signal(timestamp=datetime(2024, 1, 2, 14, 35, tzinfo=UTC)),
            PortfolioState(),
        )

        self.assertTrue(inside_summer.approved)
        self.assertEqual(outside_summer.reason_code, "outside_session")
        self.assertTrue(inside_winter.approved)

    def test_blocks_trade_after_daily_limit_and_kill_switch(self) -> None:
        engine = PropFirmRiskEngine(
            max_daily_loss=0.02,
            daily_kill_switch=True,
        )
        decision = engine.evaluate(
            _signal(timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=UTC)),
            PortfolioState(
                daily_pnl_pct=-0.03,
                daily_kill_switch_active=True,
            ),
        )

        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason_code, "daily_loss_limit")
        self.assertIn("daily_kill_switch", decision.blocked_by)

    def test_blocks_when_trade_count_or_total_drawdown_breached(self) -> None:
        engine = PropFirmRiskEngine(
            max_trades_per_day=2,
            max_total_drawdown=0.05,
        )
        decision = engine.evaluate(
            _signal(timestamp=datetime(2024, 1, 1, 13, 0, tzinfo=UTC)),
            PortfolioState(
                trades_today=2,
                total_drawdown_pct=0.06,
            ),
        )

        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason_code, "max_trades_per_day")
        self.assertIn("max_total_drawdown", decision.blocked_by)
