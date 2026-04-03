from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from hybrid_quant.baseline import BaselineRunner
from hybrid_quant.core import SignalSide, StrategySignal
from hybrid_quant.risk import PropFirmRiskEngine


def _frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01T00:00:00Z", periods=4, freq="5min", tz="UTC")
    frame = pd.DataFrame(index=index)
    frame["open"] = [100.0, 100.0, 99.0, 99.1]
    frame["high"] = [100.2, 100.1, 99.2, 99.3]
    frame["low"] = [99.8, 98.5, 98.9, 98.9]
    frame["close"] = [100.0, 99.0, 99.0, 99.1]
    frame["volume"] = 100.0
    return frame


def _long_signal(timestamp: datetime, entry: float, stop: float, target: float) -> StrategySignal:
    return StrategySignal(
        symbol="BTCUSDT",
        timestamp=timestamp,
        side=SignalSide.LONG,
        strength=1.0,
        rationale="synthetic long",
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        time_stop_bars=12,
        close_on_session_end=True,
        entry_reason="synthetic long",
    )


class BaselineRiskIntegrationTests(unittest.TestCase):
    def test_daily_kill_switch_blocks_second_trade_and_writes_risk_artifacts(self) -> None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        runner = BaselineRunner.from_config(config_dir)

        runner.application.backtest_engine.fee_bps = 0.0
        runner.application.backtest_engine.slippage_bps = 0.0
        runner.application.settings.backtest.fee_bps = 0.0
        runner.application.settings.backtest.slippage_bps = 0.0
        runner.application.settings.risk.max_risk_per_trade = 0.01
        runner.application.settings.risk.max_daily_loss = 0.0005
        runner.application.settings.risk.daily_kill_switch = True
        runner.application.risk_engine = PropFirmRiskEngine(
            max_risk_per_trade=0.01,
            max_daily_loss=0.0005,
            max_total_drawdown=0.5,
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
        )

        timestamps = [datetime(2024, 1, 1, 0, 0, tzinfo=UTC) + timedelta(minutes=5 * idx) for idx in range(4)]
        signals = [
            _long_signal(timestamps[0], entry=100.0, stop=99.0, target=101.0),
            _long_signal(timestamps[1], entry=99.0, stop=98.5, target=100.0),
            StrategySignal(symbol="BTCUSDT", timestamp=timestamps[2], side=SignalSide.FLAT, strength=0.0, rationale="hold"),
            StrategySignal(symbol="BTCUSDT", timestamp=timestamps[3], side=SignalSide.FLAT, strength=0.0, rationale="hold"),
        ]
        original_generate_signals = BaselineRunner._generate_signals
        BaselineRunner._generate_signals = lambda self, bars, features: signals  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                artifacts = runner.run(output_dir=tmp_dir, input_frame=_frame())

                report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
                self.assertEqual(report["number_of_trades"], 1)
                self.assertEqual(report["risk"]["raw_actionable_signals"], 2)
                self.assertEqual(report["risk"]["approved_actionable_signals"], 1)
                self.assertEqual(report["risk"]["blocked_actionable_signals"], 1)
                self.assertEqual(report["risk"]["blocked_by_reason"]["daily_loss_limit"], 1)
                self.assertEqual(report["risk"]["kill_switch_triggered_days"], ["2024-01-01"])

                risk_frame = pd.read_csv(artifacts.risk_decisions_path)
                blocked_row = risk_frame.loc[risk_frame["timestamp"] == "2024-01-01 00:05:00+00:00"].iloc[0]
                self.assertEqual(blocked_row["raw_side"], "long")
                self.assertEqual(blocked_row["filtered_side"], "flat")
                self.assertEqual(blocked_row["reason_code"], "daily_loss_limit")
                self.assertIn("daily_kill_switch", blocked_row["blocked_by"])
                self.assertTrue(bool(blocked_row["daily_kill_switch_active"]))

                signals_frame = pd.read_csv(artifacts.signals_path)
                filtered_row = signals_frame.loc[signals_frame["timestamp"] == "2024-01-01 00:05:00+00:00"].iloc[0]
                self.assertEqual(filtered_row["side"], "flat")
                self.assertEqual(filtered_row["raw_side"], "long")
                self.assertEqual(filtered_row["risk_reason_code"], "daily_loss_limit")

                trades_frame = pd.read_csv(artifacts.trades_path)
                self.assertEqual(len(trades_frame), 1)
                self.assertEqual(trades_frame.iloc[0]["exit_reason"], "stop_loss")

                risk_log = artifacts.risk_log_path.read_text(encoding="utf-8")
                self.assertIn("daily_loss_limit", risk_log)
        finally:
            BaselineRunner._generate_signals = original_generate_signals
