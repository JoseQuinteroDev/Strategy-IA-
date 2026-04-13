from __future__ import annotations

import unittest
from datetime import datetime

from hybrid_quant.azir.protected_benchmark_freeze import revalue_forced_close
from hybrid_quant.azir.replica import OhlcvBar
from hybrid_quant.risk.azir_state import AzirRiskConfig


class AzirProtectedBenchmarkFreezeTests(unittest.TestCase):
    def test_revalues_buy_with_last_closed_m1_before_close(self) -> None:
        bars = [
            OhlcvBar(datetime(2024, 6, 19, 21, 58), 101.0, 102.0, 100.0, 101.5, 10),
            OhlcvBar(datetime(2024, 6, 20, 1, 0), 99.0, 100.0, 98.0, 99.5, 10),
        ]
        decision = {
            "setup_day": "2024-06-19",
            "fill_timestamp": "2024.06.19 19:21:38",
            "exit_timestamp": "2024.06.20 04:05:03",
            "fill_side": "buy",
            "fill_price": "100.0",
            "observed_net_pnl": "1.0",
        }

        result = revalue_forced_close(
            decision=decision,
            bars=bars,
            config=AzirRiskConfig(close_hour=22),
            lot_size=0.10,
            contract_size=100.0,
        )

        self.assertEqual(result["revaluation_status"], "priced_with_m1_proxy")
        self.assertEqual(result["revalued_exit_price"], 101.5)
        self.assertEqual(result["revalued_net_pnl"], 15.0)
        self.assertEqual(result["selected_m1_bar_open_time"], "2024-06-19 21:58:00")

    def test_revalues_sell_with_short_direction(self) -> None:
        bars = [
            OhlcvBar(datetime(2024, 9, 2, 21, 58), 98.0, 99.0, 97.0, 98.5, 10),
            OhlcvBar(datetime(2024, 9, 3, 1, 0), 98.0, 98.0, 97.0, 97.5, 10),
        ]
        decision = {
            "setup_day": "2024-09-02",
            "fill_timestamp": "2024.09.02 17:01:39",
            "exit_timestamp": "2024.09.03 02:57:28",
            "fill_side": "sell",
            "fill_price": "100.0",
            "observed_net_pnl": "1.0",
        }

        result = revalue_forced_close(
            decision=decision,
            bars=bars,
            config=AzirRiskConfig(close_hour=22),
            lot_size=0.10,
            contract_size=100.0,
        )

        self.assertEqual(result["revalued_net_pnl"], 15.0)
        self.assertEqual(result["selected_bar_high_pnl"], 10.0)
        self.assertEqual(result["selected_bar_low_pnl"], 30.0)

    def test_marks_case_unpriced_when_no_m1_bar_after_fill(self) -> None:
        decision = {
            "setup_day": "2024-09-02",
            "fill_timestamp": "2024.09.02 17:01:39",
            "exit_timestamp": "2024.09.03 02:57:28",
            "fill_side": "sell",
            "fill_price": "100.0",
            "observed_net_pnl": "1.0",
        }

        result = revalue_forced_close(
            decision=decision,
            bars=[],
            config=AzirRiskConfig(close_hour=22),
            lot_size=0.10,
            contract_size=100.0,
        )

        self.assertEqual(result["revaluation_status"], "unpriced")


if __name__ == "__main__":
    unittest.main()
