from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from hybrid_quant.azir.comparison import (
    compare_daily_opportunities,
    compare_event_logs,
    compare_fill_exit_coverage,
)
from hybrid_quant.azir.replica import AzirPythonReplica, AzirReplicaConfig, OhlcvBar, atr, ema, rsi


def _bars(start: datetime, count: int, *, step: float = 0.20) -> list[OhlcvBar]:
    bars: list[OhlcvBar] = []
    price = 2000.0
    for index in range(count):
        close = price + index * step
        bars.append(
            OhlcvBar(
                open_time=start + timedelta(minutes=5 * index),
                open=close - 0.05,
                high=close + 0.60,
                low=close - 0.60,
                close=close,
                volume=100.0 + index,
            )
        )
    return bars


class AzirReplicaTests(unittest.TestCase):
    def test_indicators_return_closed_bar_values(self) -> None:
        bars = _bars(datetime(2025, 1, 8, 14, 0), 40)

        self.assertIsNotNone(ema([bar.close for bar in bars], 20)[30])
        self.assertIsNotNone(atr(bars, 14)[30])
        self.assertIsNotNone(rsi(bars, 14)[30])

    def test_atr_uses_simple_average_true_range_like_observed_mt5(self) -> None:
        bars = _bars(datetime(2025, 1, 8, 14, 0), 20)
        true_ranges = []
        for index, bar in enumerate(bars):
            if index == 0:
                true_ranges.append(bar.high - bar.low)
            else:
                previous_close = bars[index - 1].close
                true_ranges.append(
                    max(
                        bar.high - bar.low,
                        abs(bar.high - previous_close),
                        abs(bar.low - previous_close),
                    )
                )

        self.assertAlmostEqual(atr(bars, 14)[13], sum(true_ranges[:14]) / 14)
        self.assertAlmostEqual(atr(bars, 14)[14], sum(true_ranges[1:15]) / 14)

    def test_friday_filter_blocks_daily_opportunity(self) -> None:
        bars = _bars(datetime(2025, 1, 10, 14, 0), 40)  # Friday
        replica = AzirPythonReplica(bars)

        rows = replica.run()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_type"], "blocked_friday")
        self.assertEqual(rows[0]["is_friday"], True)

    def test_opportunity_uses_last_ten_closed_bars_for_swing_levels(self) -> None:
        bars = _bars(datetime(2025, 1, 8, 14, 0), 120)
        setup_index = next(i for i, bar in enumerate(bars) if bar.open_time.hour == 16 and bar.open_time.minute == 30)
        expected_swing_high = max(bar.high for bar in bars[setup_index - 10 : setup_index])
        expected_swing_low = min(bar.low for bar in bars[setup_index - 10 : setup_index])

        rows = AzirPythonReplica(bars).run()
        opportunity = rows[0]

        self.assertEqual(opportunity["event_type"], "opportunity")
        self.assertAlmostEqual(opportunity["swing_high"], expected_swing_high)
        self.assertAlmostEqual(opportunity["swing_low"], expected_swing_low)
        self.assertAlmostEqual(opportunity["buy_entry"], expected_swing_high + 0.05)
        self.assertAlmostEqual(opportunity["sell_entry"], expected_swing_low - 0.05)
        self.assertEqual(opportunity["buy_order_placed"], True)
        self.assertEqual(opportunity["sell_order_placed"], False)
        self.assertEqual(opportunity["rsi_gate_required"], False)

    def test_atr_filter_can_block_order_placement(self) -> None:
        bars = _bars(datetime(2025, 1, 8, 14, 0), 120)
        config = AzirReplicaConfig(atr_minimum=1000.0)

        rows = AzirPythonReplica(bars, config=config).run()

        self.assertEqual(rows[0]["event_type"], "opportunity")
        self.assertEqual(rows[0]["atr_filter_passed"], False)
        self.assertEqual(rows[0]["buy_order_placed"], False)
        self.assertEqual(rows[0]["sell_order_placed"], False)

    def test_comparison_reports_perfect_parity_for_identical_rows(self) -> None:
        bars = _bars(datetime(2025, 1, 8, 14, 0), 120)
        rows = AzirPythonReplica(bars).run()

        report = compare_event_logs(rows, rows)

        self.assertEqual(report["status"], "compared")
        self.assertEqual(report["discrepancy_count"], 0)
        self.assertEqual(report["field_match_pct"], 100.0)

    def test_daily_opportunity_parity_deduplicates_mt5_opportunity_rows(self) -> None:
        python_rows = [
            {
                "timestamp": "2025.01.08 16:30:00",
                "event_type": "opportunity",
                "symbol": "XAUUSD-STD",
                "magic": "123456321",
                "buy_entry": "2001.05",
                "atr_filter_passed": "true",
            }
        ]
        mt5_rows = [
            {
                "timestamp": "2025.01.08 16:30:00",
                "event_type": "opportunity",
                "symbol": "XAUUSD-STD",
                "magic": "123456321",
                "buy_entry": "2001.05",
                "atr_filter_passed": "true",
            },
            {
                "timestamp": "2025.01.08 16:30:05",
                "event_type": "opportunity",
                "symbol": "XAUUSD-STD",
                "magic": "123456321",
                "buy_entry": "2001.05",
                "atr_filter_passed": "true",
            },
        ]

        report = compare_daily_opportunities(python_rows, mt5_rows)

        self.assertEqual(report["setup_day_match_pct"], 100.0)
        self.assertEqual(report["mt5_duplicate_opportunity_rows"], 1)

    def test_fill_exit_coverage_reports_count_match(self) -> None:
        python_rows = [
            {"timestamp": "2025.01.08 16:31:00", "event_type": "fill"},
            {"timestamp": "2025.01.08 16:40:00", "event_type": "exit"},
        ]
        mt5_rows = [
            {"timestamp": "2025.01.08 16:31:00", "event_type": "fill"},
            {"timestamp": "2025.01.08 16:40:00", "event_type": "exit"},
        ]

        report = compare_fill_exit_coverage(python_rows, mt5_rows)

        self.assertEqual(report["fill_count_match_pct"], 100.0)
        self.assertEqual(report["exit_count_match_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
