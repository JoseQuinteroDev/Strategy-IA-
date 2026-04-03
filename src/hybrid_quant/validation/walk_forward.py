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
