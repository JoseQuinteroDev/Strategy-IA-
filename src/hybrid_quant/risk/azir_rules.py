"""Hard lifecycle rules for Azir's prop-firm friendly risk layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .azir_state import AzirRiskConfig, AzirRiskState


class AzirRiskRule(Protocol):
    code: str

    def evaluate(self, state: AzirRiskState, config: AzirRiskConfig, context: str) -> "RuleResult":
        ...


@dataclass(frozen=True)
class RuleResult:
    code: str
    block: bool = False
    actions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    rationale: str = ""


@dataclass(frozen=True)
class ReconciliationRule:
    code: str = "force_reconcile_orders_positions_before_setup"

    def evaluate(self, state: AzirRiskState, config: AzirRiskConfig, context: str) -> RuleResult:
        if not config.force_reconcile_orders_positions_before_setup:
            return RuleResult(self.code)
        if context not in {"before_setup", "setup_attempt"}:
            return RuleResult(self.code)
        if state.reconciled and not state.reconciliation_errors:
            return RuleResult(self.code)
        actions = ("reconcile_broker_state",)
        if config.reconciliation_policy == "block_and_cleanup":
            actions = actions + ("cancel_all_pendings", "close_open_positions")
        return RuleResult(
            self.code,
            block=True,
            actions=actions,
            rationale="Broker/account state is not clean before Azir setup.",
        )


@dataclass(frozen=True)
class ExposureBeforeSetupRule:
    code: str = "block_new_setups_if_any_position_or_pending_exists"

    def evaluate(self, state: AzirRiskState, config: AzirRiskConfig, context: str) -> RuleResult:
        if not config.block_new_setups_if_any_position_or_pending_exists or context != "setup_attempt":
            return RuleResult(self.code)
        if not state.has_exposure:
            return RuleResult(self.code)
        return RuleResult(
            self.code,
            block=True,
            actions=("cancel_all_pendings", "close_open_positions", "block_new_setup"),
            rationale="Azir setup is blocked because prior exposure is still live.",
        )


@dataclass(frozen=True)
class CloseCleanupRule:
    code: str = "hard_cancel_all_pendings_at_close"

    def evaluate(self, state: AzirRiskState, config: AzirRiskConfig, context: str) -> RuleResult:
        if context not in {"close_check", "heartbeat", "after_fill"}:
            return RuleResult(self.code)
        if state.timestamp.hour < config.close_hour:
            return RuleResult(self.code)
        actions: list[str] = []
        if config.hard_cancel_all_pendings_at_close and state.pending_orders > 0:
            actions.append("cancel_all_pendings")
        if config.close_positions_at_close and state.open_positions > 0:
            actions.append("close_open_positions")
        if not actions:
            return RuleResult(self.code)
        return RuleResult(
            self.code,
            actions=tuple(actions),
            warnings=("close_time_cleanup_required",),
            rationale="Close-time lifecycle cleanup is required.",
        )


@dataclass(frozen=True)
class FridayExposureRule:
    code: str = "friday_no_new_trade_plus_close_or_cancel_prior_exposure"

    def evaluate(self, state: AzirRiskState, config: AzirRiskConfig, context: str) -> RuleResult:
        if not config.friday_no_new_trade or state.timestamp.weekday() != 4:
            return RuleResult(self.code)
        block = context == "setup_attempt"
        actions: list[str] = []
        if state.has_exposure and config.friday_exposure_policy == "close_or_cancel":
            actions.extend(["cancel_all_pendings", "close_open_positions"])
        if block or actions:
            return RuleResult(
                self.code,
                block=block,
                actions=tuple(actions),
                rationale="Friday policy blocks new setup and cleans prior exposure.",
            )
        return RuleResult(self.code)


@dataclass(frozen=True)
class DailyMaxLossRule:
    code: str = "daily_max_loss_guard"

    def evaluate(self, state: AzirRiskState, config: AzirRiskConfig, context: str) -> RuleResult:
        if context not in {"before_setup", "setup_attempt", "heartbeat"}:
            return RuleResult(self.code)
        limit = config.max_daily_loss
        if config.max_daily_loss_pct is not None:
            limit = min(limit, config.starting_equity * config.max_daily_loss_pct)
        if state.daily_realized_pnl > -abs(limit):
            return RuleResult(self.code)
        return RuleResult(
            self.code,
            block=context == "setup_attempt",
            actions=("activate_daily_kill_switch", "cancel_all_pendings"),
            rationale="Daily loss guard reached.",
        )


@dataclass(frozen=True)
class ConsecutiveLossKillSwitchRule:
    code: str = "consecutive_losses_kill_switch"

    def evaluate(self, state: AzirRiskState, config: AzirRiskConfig, context: str) -> RuleResult:
        if config.max_consecutive_losses <= 0:
            return RuleResult(self.code)
        if state.consecutive_losses_today < config.max_consecutive_losses:
            return RuleResult(self.code)
        return RuleResult(
            self.code,
            block=context == "setup_attempt",
            actions=("activate_daily_kill_switch", "cancel_all_pendings"),
            rationale="Consecutive loss kill-switch reached.",
        )


@dataclass(frozen=True)
class MaxTradesPerDayRule:
    code: str = "max_trades_per_day"

    def evaluate(self, state: AzirRiskState, config: AzirRiskConfig, context: str) -> RuleResult:
        if context != "setup_attempt" or state.trades_today < config.max_trades_per_day:
            return RuleResult(self.code)
        return RuleResult(
            self.code,
            block=True,
            actions=("block_new_setup",),
            rationale="Maximum Azir trades per day reached.",
        )


@dataclass(frozen=True)
class SpreadGuardRule:
    code: str = "spread_guard_if_available"

    def evaluate(self, state: AzirRiskState, config: AzirRiskConfig, context: str) -> RuleResult:
        if not config.spread_guard_enabled or context != "setup_attempt":
            return RuleResult(self.code)
        if state.spread_points is None:
            return RuleResult(
                self.code,
                warnings=("spread_unavailable",),
                rationale="Spread was unavailable; spread guard could not be evaluated.",
            )
        if state.spread_points <= config.max_spread_points:
            return RuleResult(self.code)
        return RuleResult(
            self.code,
            block=True,
            actions=("block_new_setup",),
            rationale="Spread exceeds configured threshold.",
        )


@dataclass(frozen=True)
class TrailingGuardrailRule:
    code: str = "trailing_guardrails"

    def evaluate(self, state: AzirRiskState, config: AzirRiskConfig, context: str) -> RuleResult:
        if not config.trailing_audit_required or state.open_positions <= 0:
            return RuleResult(self.code)
        warnings: list[str] = []
        if not state.trailing_expected:
            warnings.append("trailing_not_expected_for_open_position")
        if not state.trailing_active:
            warnings.append("trailing_not_yet_active")
        return RuleResult(
            self.code,
            warnings=tuple(warnings),
            rationale="Trailing state is audited but not modified by risk_engine_azir_v1.",
        )


@dataclass(frozen=True)
class CancelRemainingPendingsAfterFillRule:
    code: str = "cancel_remaining_pendings_after_fill"

    def evaluate(self, state: AzirRiskState, config: AzirRiskConfig, context: str) -> RuleResult:
        if not config.cancel_remaining_pendings_after_fill or context != "after_fill":
            return RuleResult(self.code)
        if state.pending_orders <= 0:
            return RuleResult(self.code)
        return RuleResult(
            self.code,
            actions=("cancel_remaining_pendings",),
            warnings=("post_fill_pending_cleanup_required",),
            rationale="A fill exists while pending orders remain live.",
        )


DEFAULT_AZIR_RULES: tuple[AzirRiskRule, ...] = (
    ReconciliationRule(),
    ExposureBeforeSetupRule(),
    CloseCleanupRule(),
    FridayExposureRule(),
    DailyMaxLossRule(),
    ConsecutiveLossKillSwitchRule(),
    MaxTradesPerDayRule(),
    SpreadGuardRule(),
    TrailingGuardrailRule(),
    CancelRemainingPendingsAfterFillRule(),
)
