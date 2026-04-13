"""CSV inspection helpers for Azir empirical inputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


EVENT_LOG_COLUMNS = {"timestamp", "event_id", "event_type", "symbol", "magic"}
OHLCV_COLUMNS = {"open_time", "open", "high", "low", "close", "volume"}
TRADE_HISTORY_HINTS = {"ticket", "order", "deal", "type", "profit", "commission"}


@dataclass(frozen=True)
class CsvInspection:
    path: str
    bytes: int
    columns: tuple[str, ...]
    rows: int
    first_timestamp: str | None
    last_timestamp: str | None
    dataset_type: str
    suitability: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "columns": list(self.columns),
            "rows": self.rows,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "dataset_type": self.dataset_type,
            "suitability": self.suitability,
        }


def inspect_csv(path: str | Path) -> CsvInspection:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {input_path}")

    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        columns = tuple(reader.fieldnames or ())
        timestamp_column = _timestamp_column(columns)
        rows = 0
        first_timestamp: datetime | None = None
        last_timestamp: datetime | None = None
        for row in reader:
            if not any(row.values()):
                continue
            rows += 1
            if timestamp_column:
                parsed = _parse_timestamp(row.get(timestamp_column, ""))
                if parsed is not None:
                    first_timestamp = parsed if first_timestamp is None else min(first_timestamp, parsed)
                    last_timestamp = parsed if last_timestamp is None else max(last_timestamp, parsed)

    dataset_type = classify_columns(columns)
    return CsvInspection(
        path=str(input_path),
        bytes=input_path.stat().st_size,
        columns=columns,
        rows=rows,
        first_timestamp=_format_timestamp(first_timestamp),
        last_timestamp=_format_timestamp(last_timestamp),
        dataset_type=dataset_type,
        suitability=_suitability(dataset_type),
    )


def classify_columns(columns: tuple[str, ...] | list[str]) -> str:
    normalized = {column.strip().lower() for column in columns}
    if EVENT_LOG_COLUMNS.issubset(normalized):
        return "azir_event_log"
    if OHLCV_COLUMNS.issubset(normalized):
        return "ohlcv"
    if len(TRADE_HISTORY_HINTS & normalized) >= 3:
        return "trade_history_or_tester_report"
    return "unknown"


def _timestamp_column(columns: tuple[str, ...]) -> str | None:
    normalized = {column.lower(): column for column in columns}
    for candidate in ("timestamp", "open_time", "time", "datetime", "date"):
        if candidate in normalized:
            return normalized[candidate]
    return None


def _parse_timestamp(value: str) -> datetime | None:
    value = str(value).strip()
    if not value:
        return None
    value = value.replace(".", "-")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _format_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _suitability(dataset_type: str) -> dict[str, str]:
    if dataset_type == "azir_event_log":
        return {
            "opportunity_parity": "yes",
            "fill_exit_parity": "yes, empirical MT5 events are present",
            "economic_audit": "partial; depends on commission/swap/slippage fields being populated",
            "tick_replay": "no",
        }
    if dataset_type == "ohlcv":
        return {
            "opportunity_parity": "python replica input only",
            "fill_exit_parity": "approximate only",
            "economic_audit": "no MT5 execution evidence",
            "tick_replay": "no",
        }
    if dataset_type == "trade_history_or_tester_report":
        return {
            "opportunity_parity": "no",
            "fill_exit_parity": "partial",
            "economic_audit": "yes if PnL fields are populated",
            "tick_replay": "no",
        }
    return {
        "opportunity_parity": "unknown",
        "fill_exit_parity": "unknown",
        "economic_audit": "unknown",
        "tick_replay": "unknown",
    }
