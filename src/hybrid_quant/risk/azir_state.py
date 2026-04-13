"""State and decision objects for the Azir lifecycle risk engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class AzirRiskConfig:
    """Configurable hard guards for Azir.

    Values are deliberately conservative and designed as a wrapper around the
    EA lifecycle, not as a replacement for Azir's entry logic.
    """

    name: str = "risk_engine_azir_v1"
    setup_hour: int = 16
    setup_minute: int = 30
    close_hour: int = 22
    session_fill_start_hour: int = 16
    session_fill_end_hour: int = 21
    hard_cancel_all_pendings_at_close: bool = True
    close_positions_at_close: bool = True
    block_new_setups_if_any_position_or_pending_exists: bool = True
    force_reconcile_orders_positions_before_setup: bool = True
    reconciliation_policy: str = "block_and_cleanup"
    friday_no_new_trade: bool = True
    friday_exposure_policy: str = "close_or_cancel"
    max_daily_loss: float = 15.0
    max_daily_loss_pct: float | None = None
    starting_equity: float = 10000.0
    max_consecutive_losses: int = 2
    max_trades_per_day: int = 1
    spread_guard_enabled: bool = True
    max_spread_points: float = 50.0
    cancel_remaining_pendings_after_fill: bool = True
    trailing_audit_required: bool = True


@dataclass(frozen=True)
class AzirRiskState:
    """Broker/account state observed by the risk layer."""

    timestamp: datetime
    pending_orders: int = 0
    open_positions: int = 0
    daily_realized_pnl: float = 0.0
    trades_today: int = 0
    consecutive_losses_today: int = 0
    spread_points: float | None = None
    reconciled: bool = True
    reconciliation_errors: tuple[str, ...] = ()
    trailing_expected: bool = True
    trailing_active: bool = False
    notes: tuple[str, ...] = ()

    @property
    def has_exposure(self) -> bool:
        return self.pending_orders > 0 or self.open_positions > 0


@dataclass(frozen=True)
class AzirRiskDecision:
    approved: bool
    reason_code: str
    blocked_by: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    rationale: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
