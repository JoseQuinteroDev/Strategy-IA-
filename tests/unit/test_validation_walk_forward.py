from __future__ import annotations

import unittest

from hybrid_quant.validation.walk_forward import build_rolling_windows


class RollingWindowTests(unittest.TestCase):
    def test_build_rolling_windows_creates_ordered_non_overlapping_phases(self) -> None:
        windows = build_rolling_windows(
            total_bars=140,
            splits=3,
            train_ratio=0.6,
            validation_ratio=0.2,
            test_ratio=0.2,
        )

        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0].train_start, 0)
        self.assertLess(windows[0].train_end, windows[0].validation_end)
        self.assertLess(windows[0].validation_end, windows[0].test_end)
        self.assertEqual(
            windows[1].train_start - windows[0].train_start,
            windows[0].test_end - windows[0].test_start,
        )
        self.assertLessEqual(windows[-1].test_end, 140)
        for window in windows:
            self.assertLess(window.train_start, window.train_end)
            self.assertLess(window.validation_start, window.validation_end)
            self.assertLess(window.test_start, window.test_end)
