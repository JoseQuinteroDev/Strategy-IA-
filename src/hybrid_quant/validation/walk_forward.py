from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from hybrid_quant.core import BacktestResult, ValidationReport


class Validator(ABC):
    @abstractmethod
    def validate(self, result: BacktestResult) -> ValidationReport:
        """Validate a backtest result against acceptance rules."""


@dataclass(slots=True)
class WalkForwardValidator(Validator):
    walk_forward_splits: int
    min_trades: int
    max_drawdown_limit: float
    sharpe_floor: float

    def validate(self, result: BacktestResult) -> ValidationReport:
        checks = {
            "min_trades": result.trades >= self.min_trades,
            "max_drawdown": result.max_drawdown <= self.max_drawdown_limit,
            "sharpe_floor": result.sharpe >= self.sharpe_floor,
        }
        passed = all(checks.values())
        summary = (
            "Validation passed."
            if passed
            else "Validation failed because the scaffold does not produce enough statistical evidence yet."
        )
        return ValidationReport(
            passed=passed,
            checks=checks,
            summary=summary,
            metadata={"walk_forward_splits": self.walk_forward_splits},
        )


@dataclass(slots=True)
class RollingWindow:
    window_id: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int

    def to_dict(self) -> dict[str, int]:
        return {
            "window_id": self.window_id,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }


def build_rolling_windows(
    *,
    total_bars: int,
    splits: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> list[RollingWindow]:
    if total_bars <= 0:
        raise ValueError("build_rolling_windows requires a positive number of bars.")
    if splits <= 0:
        raise ValueError("build_rolling_windows requires at least one split.")
    if train_ratio <= 0.0 or validation_ratio <= 0.0 or test_ratio <= 0.0:
        raise ValueError("Walk-forward ratios must all be strictly positive.")

    denominator = train_ratio + validation_ratio + test_ratio + (test_ratio * (splits - 1))
    if denominator <= 0.0:
        raise ValueError("Walk-forward ratios produced an invalid denominator.")

    base_unit = total_bars / denominator
    train_size = max(1, int(base_unit * train_ratio))
    validation_size = max(1, int(base_unit * validation_ratio))
    test_size = max(1, int(base_unit * test_ratio))

    total_used = train_size + validation_size + test_size + (splits - 1) * test_size
    if total_used > total_bars:
        while total_used > total_bars and train_size > 1:
            train_size -= 1
            total_used = train_size + validation_size + test_size + (splits - 1) * test_size
        while total_used > total_bars and validation_size > 1:
            validation_size -= 1
            total_used = train_size + validation_size + test_size + (splits - 1) * test_size
        while total_used > total_bars and test_size > 1:
            test_size -= 1
            total_used = train_size + validation_size + test_size + (splits - 1) * test_size

    total_used = train_size + validation_size + test_size + (splits - 1) * test_size
    if total_used > total_bars:
        raise ValueError("Not enough bars to build the requested rolling windows.")

    train_size += total_bars - total_used

    windows: list[RollingWindow] = []
    for window_id in range(splits):
        start = window_id * test_size
        train_start = start
        train_end = train_start + train_size
        validation_start = train_end
        validation_end = validation_start + validation_size
        test_start = validation_end
        test_end = test_start + test_size

        if test_end > total_bars:
            break

        windows.append(
            RollingWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    if not windows:
        raise ValueError("No valid walk-forward windows could be created from the provided configuration.")
    return windows
