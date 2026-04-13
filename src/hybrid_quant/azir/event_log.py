"""CSV schema utilities for the AzirIA MT5 event log.

Sprint 0/1 deliberately keeps the MQL5 EA as the source of truth. These helpers
make the exported event stream deterministic and easy to consume from the next
Python replica sprint without re-encoding column names in several places.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

AZIR_EVENT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "event_id",
    "event_type",
    "symbol",
    "magic",
    "day_of_week",
    "is_friday",
    "server_time",
    "broker",
    "account",
    "timeframe",
    "ny_open_hour",
    "ny_open_minute",
    "close_hour",
    "swing_bars",
    "lot_size",
    "sl_points",
    "tp_points",
    "trailing_start_points",
    "trailing_step_points",
    "swing_high",
    "swing_low",
    "buy_entry",
    "sell_entry",
    "pending_distance_points",
    "spread_points",
    "ema20",
    "prev_close",
    "prev_close_vs_ema20_points",
    "prev_close_above_ema20",
    "atr",
    "atr_points",
    "atr_filter_enabled",
    "atr_filter_passed",
    "atr_minimum",
    "rsi",
    "rsi_gate_enabled",
    "rsi_gate_required",
    "rsi_gate_passed",
    "rsi_bullish_threshold",
    "rsi_sell_threshold",
    "allow_buys",
    "allow_sells",
    "trend_filter_enabled",
    "buy_allowed_by_trend",
    "sell_allowed_by_trend",
    "buy_order_placed",
    "sell_order_placed",
    "buy_retcode",
    "sell_retcode",
    "fill_side",
    "fill_price",
    "duration_to_fill_seconds",
    "mfe_points",
    "mae_points",
    "exit_reason",
    "gross_pnl",
    "net_pnl",
    "commission",
    "swap",
    "slippage_points",
    "trailing_activated",
    "trailing_modifications",
    "trailing_outcome",
    "opposite_order_cancelled",
    "notes",
)

REQUIRED_AZIR_EVENT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "event_id",
    "event_type",
    "symbol",
    "magic",
)


def validate_event_row(row: Mapping[str, object]) -> list[str]:
    """Return human-readable schema errors for one Azir event row."""

    errors: list[str] = []
    missing = [column for column in REQUIRED_AZIR_EVENT_COLUMNS if column not in row]
    if missing:
        errors.append("missing required columns: " + ", ".join(missing))

    unknown = [column for column in row if column not in AZIR_EVENT_COLUMNS]
    if unknown:
        errors.append("unknown columns: " + ", ".join(unknown))

    for column in REQUIRED_AZIR_EVENT_COLUMNS:
        value = row.get(column)
        if value is None or value == "":
            errors.append(f"empty required column: {column}")

    event_type = str(row.get("event_type", ""))
    valid_event_types = {
        "opportunity",
        "blocked_friday",
        "fill",
        "trailing_modified",
        "opposite_pending_cancelled",
        "no_fill_close_cleanup",
        "exit",
    }
    if event_type and event_type not in valid_event_types:
        errors.append(f"unsupported event_type: {event_type}")

    return errors


def _normalize_row(row: Mapping[str, object]) -> dict[str, object]:
    errors = validate_event_row(row)
    if errors:
        raise ValueError("; ".join(errors))
    return {column: "" if row.get(column, "") is None else row.get(column, "") for column in AZIR_EVENT_COLUMNS}


def write_event_log(rows: Iterable[Mapping[str, object]], path: str | Path) -> Path:
    """Write Azir event rows with the canonical column order."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AZIR_EVENT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_row(row))
    return output_path
