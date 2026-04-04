"""Hybrid quantitative trading framework scaffold."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bootstrap import TradingApplication

__all__ = ["TradingApplication", "build_application"]


def __getattr__(name: str) -> Any:
    if name in {"TradingApplication", "build_application"}:
        from .bootstrap import TradingApplication, build_application

        exports = {
            "TradingApplication": TradingApplication,
            "build_application": build_application,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
