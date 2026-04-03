from __future__ import annotations

import unittest
from datetime import UTC, datetime

from hybrid_quant.core import FeatureSnapshot, MarketBar, SignalSide, StrategyContext
from hybrid_quant.strategy import MeanReversionStrategy


def _bar(timestamp: datetime, close: float) -> MarketBar:
    return MarketBar(
        timestamp=timestamp,
        open=close - 0.4,
        high=close + 0.6,
        low=close - 0.8,
        close=close,
        volume=100.0,
    )


def _feature(timestamp: datetime, **values: float) -> FeatureSnapshot:
    return FeatureSnapshot(timestamp=timestamp, values=values, metadata={})


def _strategy(**overrides: float | int | str) -> MeanReversionStrategy:
    params: dict[str, float | int | str] = {
        "name": "mean_reversion_trend_regime",
        "entry_zscore": 2.0,
        "exit_zscore": 0.5,
        "trend_filter": "ema_200_1h",
        "regime_filter": "adx_1h",
        "execution_timeframe": "5m",
        "filter_timeframe": "1H",
        "mean_reversion_anchor": "vwap",
        "adx_threshold": 25.0,
        "atr_multiple_stop": 1.0,
        "atr_multiple_target": 1.0,
        "time_stop_bars": 12,
        "session_close_hour_utc": 23,
        "session_close_minute_utc": 55,
        "no_entry_minutes_before_close": 30,
    }
    params.update(overrides)
    return MeanReversionStrategy(**params)


class MeanReversionStrategyTests(unittest.TestCase):
    def test_generates_long_signal_with_stop_and_target(self) -> None:
        timestamp = datetime(2024, 1, 2, 12, 0, tzinfo=UTC)
        strategy = _strategy(mean_reversion_anchor="vwap")
        signal = strategy.generate(
            StrategyContext(
                symbol="BTCUSDT",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=100.0)],
                features=[
                    _feature(
                        timestamp,
                        ema_200_1h=95.0,
                        adx_1h=18.0,
                        intraday_vwap=101.2,
                        ema_50=100.9,
                        atr_14=1.5,
                        zscore_distance_to_mean=-2.4,
                    )
                ],
            )
        )

        self.assertEqual(signal.side, SignalSide.LONG)
        self.assertEqual(signal.entry_price, 100.0)
        self.assertAlmostEqual(signal.stop_price, 98.5)
        self.assertAlmostEqual(signal.target_price, 101.5)
        self.assertEqual(signal.time_stop_bars, 12)
        self.assertTrue(signal.close_on_session_end)
        self.assertIn("Long mean reversion", signal.entry_reason or "")
        self.assertEqual(signal.metadata["anchor_name"], "intraday_vwap")

    def test_generates_short_signal_with_stop_and_target(self) -> None:
        timestamp = datetime(2024, 1, 2, 14, 0, tzinfo=UTC)
        strategy = _strategy(mean_reversion_anchor="ema50")
        signal = strategy.generate(
            StrategyContext(
                symbol="BTCUSDT",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=102.0)],
                features=[
                    _feature(
                        timestamp,
                        ema_200_1h=105.0,
                        adx_1h=16.0,
                        intraday_vwap=100.5,
                        ema_50=100.8,
                        atr_14=1.2,
                        zscore_distance_to_mean=2.3,
                    )
                ],
            )
        )

        self.assertEqual(signal.side, SignalSide.SHORT)
        self.assertEqual(signal.entry_price, 102.0)
        self.assertAlmostEqual(signal.stop_price, 103.2)
        self.assertAlmostEqual(signal.target_price, 100.8)
        self.assertEqual(signal.metadata["anchor_name"], "ema_50")
        self.assertIn("Short mean reversion", signal.entry_reason or "")

    def test_blocks_trade_when_adx_regime_is_too_strong(self) -> None:
        timestamp = datetime(2024, 1, 2, 11, 0, tzinfo=UTC)
        strategy = _strategy()
        signal = strategy.generate(
            StrategyContext(
                symbol="BTCUSDT",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=100.0)],
                features=[
                    _feature(
                        timestamp,
                        ema_200_1h=95.0,
                        adx_1h=32.0,
                        intraday_vwap=101.0,
                        ema_50=100.8,
                        atr_14=1.0,
                        zscore_distance_to_mean=-2.2,
                    )
                ],
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)
        self.assertIsNone(signal.entry_price)
        self.assertIn("ADX regime filter blocked", signal.rationale)

    def test_flattens_at_session_close(self) -> None:
        timestamp = datetime(2024, 1, 2, 23, 55, tzinfo=UTC)
        strategy = _strategy()
        signal = strategy.generate(
            StrategyContext(
                symbol="BTCUSDT",
                execution_timeframe="5m",
                filter_timeframe="1H",
                bars=[_bar(timestamp, close=100.0)],
                features=[
                    _feature(
                        timestamp,
                        ema_200_1h=95.0,
                        adx_1h=18.0,
                        intraday_vwap=101.0,
                        ema_50=100.8,
                        atr_14=1.0,
                        zscore_distance_to_mean=-2.4,
                    )
                ],
            )
        )

        self.assertEqual(signal.side, SignalSide.FLAT)
        self.assertTrue(signal.metadata["session_close_exit"])
        self.assertIn("Session close", signal.rationale)

