"""Parity comparison between Python Azir replica events and MT5 event logs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .event_log import AZIR_EVENT_COLUMNS, validate_event_row


@dataclass(frozen=True)
class ParityTolerances:
    price: float = 0.011
    points: float = 1.0
    pnl: float = 0.01


NUMERIC_COMPARISONS: tuple[tuple[str, str, float], ...] = (
    ("swing_high", "price", 0.011),
    ("swing_low", "price", 0.011),
    ("buy_entry", "price", 0.011),
    ("sell_entry", "price", 0.011),
    ("pending_distance_points", "points", 1.0),
    ("ema20", "price", 0.011),
    ("prev_close", "price", 0.011),
    ("atr_points", "points", 1.0),
    ("fill_price", "price", 0.011),
    ("mfe_points", "points", 3.0),
    ("mae_points", "points", 3.0),
    ("net_pnl", "pnl", 0.01),
)

EXACT_COMPARISONS: tuple[str, ...] = (
    "event_type",
    "is_friday",
    "atr_filter_passed",
    "rsi_gate_required",
    "rsi_gate_passed",
    "buy_order_placed",
    "sell_order_placed",
    "fill_side",
    "exit_reason",
    "trailing_activated",
)


def read_event_log(path: str | Path) -> list[dict[str, str]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"MT5 event log does not exist: {input_path}")
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(AZIR_EVENT_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Event log missing canonical columns: {sorted(missing)}")
        rows = [_normalize_real_log_row(dict(row)) for row in reader]
    errors = []
    for index, row in enumerate(rows, start=2):
        row_errors = validate_event_row(row)
        if row_errors:
            errors.append(f"line {index}: {'; '.join(row_errors)}")
    if errors:
        raise ValueError("Invalid event log:\n" + "\n".join(errors[:20]))
    return rows


def _normalize_real_log_row(row: dict[str, str]) -> dict[str, str]:
    """Repair safe identity fields that old/real MT5 logs may have omitted.

    The first logger version cleared ``event_id`` on a daily reset. If an old
    pending later filled, fill/trailing/exit rows could be valid but lack an
    event_id. For comparison we reconstruct a stable key from timestamp date,
    symbol and magic instead of rejecting the empirical log.
    """

    if not str(row.get("event_id", "")).strip():
        timestamp = str(row.get("timestamp", "")).strip()
        date_part = timestamp.split(" ")[0].replace(".", "-") if timestamp else "unknown_date"
        symbol = str(row.get("symbol", "")).strip() or "unknown_symbol"
        magic = str(row.get("magic", "")).strip() or "unknown_magic"
        row["event_id"] = f"{date_part}_{symbol}_{magic}"
    return row


def compare_event_logs(
    python_rows: list[dict[str, Any]],
    mt5_rows: list[dict[str, Any]],
    tolerances: ParityTolerances | None = None,
) -> dict[str, Any]:
    """Compare events by event_id and event_type with category breakdowns."""

    tolerances = tolerances or ParityTolerances()
    py_index = _index_rows(python_rows)
    mt5_index = _index_rows(mt5_rows)
    all_keys = sorted(set(py_index) | set(mt5_index))
    discrepancies: list[dict[str, Any]] = []
    compared_fields = 0
    matched_fields = 0
    event_matches = 0

    for key in all_keys:
        py_group = py_index.get(key, [])
        mt5_group = mt5_index.get(key, [])
        if not py_group:
            discrepancies.append(_discrepancy(key, "missing_python_event", "", "", "present only in MT5"))
            continue
        if not mt5_group:
            discrepancies.append(_discrepancy(key, "missing_mt5_event", "present only in Python", "", ""))
            continue

        for pair_index, (py_row, mt5_row) in enumerate(_zip_with_missing(py_group, mt5_group)):
            pair_key = f"{key}#{pair_index + 1}"
            if py_row is None:
                discrepancies.append(
                    _discrepancy(pair_key, "missing_python_duplicate", "", "", "extra duplicate in MT5")
                )
                continue
            if mt5_row is None:
                discrepancies.append(
                    _discrepancy(pair_key, "missing_mt5_duplicate", "extra duplicate in Python", "", "")
                )
                continue
            event_ok = True
            for field in EXACT_COMPARISONS:
                if _blank(py_row.get(field)) and _blank(mt5_row.get(field)):
                    continue
                compared_fields += 1
                if _normal(py_row.get(field)) == _normal(mt5_row.get(field)):
                    matched_fields += 1
                else:
                    event_ok = False
                    discrepancies.append(
                        _discrepancy(
                            pair_key,
                            f"field_mismatch:{field}",
                            py_row.get(field, ""),
                            mt5_row.get(field, ""),
                            "exact field mismatch",
                        )
                    )
            for field, _, default_tolerance in NUMERIC_COMPARISONS:
                py_value = _to_float(py_row.get(field))
                mt5_value = _to_float(mt5_row.get(field))
                if py_value is None and mt5_value is None:
                    continue
                compared_fields += 1
                tolerance = getattr(tolerances, _tolerance_name(field), default_tolerance)
                if py_value is not None and mt5_value is not None and abs(py_value - mt5_value) <= tolerance:
                    matched_fields += 1
                else:
                    event_ok = False
                    discrepancies.append(
                        _discrepancy(
                            pair_key,
                            f"numeric_mismatch:{field}",
                            py_row.get(field, ""),
                            mt5_row.get(field, ""),
                            f"tolerance={tolerance}",
                        )
                    )
            if event_ok:
                event_matches += 1

    discrepancy_categories = Counter(item["category"] for item in discrepancies)
    total_events = len(all_keys)
    parity_pct = (matched_fields / compared_fields * 100.0) if compared_fields else 0.0
    event_match_pct = (event_matches / total_events * 100.0) if total_events else 0.0
    return {
        "status": "compared",
        "python_events": len(python_rows),
        "mt5_events": len(mt5_rows),
        "unique_event_keys": total_events,
        "event_match_pct": round(event_match_pct, 4),
        "field_match_pct": round(parity_pct, 4),
        "compared_fields": compared_fields,
        "matched_fields": matched_fields,
        "discrepancy_count": len(discrepancies),
        "discrepancies_by_category": dict(discrepancy_categories),
        "daily_opportunity_parity": compare_daily_opportunities(python_rows, mt5_rows),
        "fill_exit_coverage": compare_fill_exit_coverage(python_rows, mt5_rows),
        "discrepancies": discrepancies,
        "hard_to_reproduce": [
            "MT5 tick ordering inside an M5 bar",
            "broker-side pending-order fill sequence when both sides are touched",
            "trailing updates from live bid/ask ticks versus OHLC approximation",
            "commission, swap and broker execution slippage from deal history",
            "RSI gate on M1 if only M5 historical data is available",
        ],
    }


def compare_daily_opportunities(
    python_rows: list[dict[str, Any]],
    mt5_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare one setup decision per server-time date.

    MT5 can log repeated opportunity rows in the same setup minute when the EA
    does not mark ``orders_placed_today``. For day-by-day parity we prefer the
    opportunity row that actually placed an order; if none placed orders, we use
    the first opportunity/Friday block for that server-time date.
    """

    py_daily = _first_daily_setup_rows(python_rows)
    mt5_daily = _first_daily_setup_rows(mt5_rows)
    all_days = sorted(set(py_daily) | set(mt5_daily))
    common_days = sorted(set(py_daily) & set(mt5_daily))
    compared_fields = 0
    matched_fields = 0
    field_discrepancies: Counter[str] = Counter()
    fields = (
        "event_type",
        "is_friday",
        "swing_high",
        "swing_low",
        "buy_entry",
        "sell_entry",
        "pending_distance_points",
        "ema20",
        "prev_close",
        "atr_points",
        "atr_filter_passed",
        "rsi_gate_required",
        "buy_order_placed",
        "sell_order_placed",
    )
    for day in common_days:
        py_row = py_daily[day]
        mt5_row = mt5_daily[day]
        for field in fields:
            if field not in {"event_type", "is_friday"} and (
                py_row.get("event_type") != "opportunity" or mt5_row.get("event_type") != "opportunity"
            ):
                continue
            if _blank(py_row.get(field)) and _blank(mt5_row.get(field)):
                continue
            compared_fields += 1
            if _field_matches(field, py_row.get(field), mt5_row.get(field)):
                matched_fields += 1
            else:
                field_discrepancies[field] += 1

    mt5_opportunity_counts = Counter(
        _event_day(row) for row in mt5_rows if row.get("event_type") == "opportunity"
    )
    duplicate_rows = sum(max(0, count - 1) for count in mt5_opportunity_counts.values())
    return {
        "python_setup_days": len(py_daily),
        "mt5_setup_days": len(mt5_daily),
        "common_setup_days": len(common_days),
        "setup_day_match_pct": round((len(common_days) / len(all_days) * 100.0), 4) if all_days else 0.0,
        "field_match_pct": round((matched_fields / compared_fields * 100.0), 4) if compared_fields else 0.0,
        "compared_fields": compared_fields,
        "matched_fields": matched_fields,
        "field_discrepancies": dict(field_discrepancies),
        "missing_in_python": [day for day in all_days if day not in py_daily][:50],
        "missing_in_mt5": [day for day in all_days if day not in mt5_daily][:50],
        "mt5_duplicate_opportunity_rows": duplicate_rows,
        "mt5_max_opportunity_rows_one_day": max(mt5_opportunity_counts.values(), default=0),
    }


def compare_fill_exit_coverage(
    python_rows: list[dict[str, Any]],
    mt5_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    py_fills = [row for row in python_rows if row.get("event_type") == "fill"]
    mt5_fills = [row for row in mt5_rows if row.get("event_type") == "fill"]
    py_exits = [row for row in python_rows if row.get("event_type") == "exit"]
    mt5_exits = [row for row in mt5_rows if row.get("event_type") == "exit"]
    return {
        "python_fills": len(py_fills),
        "mt5_fills": len(mt5_fills),
        "python_exits": len(py_exits),
        "mt5_exits": len(mt5_exits),
        "fill_count_match_pct": _count_match_pct(len(py_fills), len(mt5_fills)),
        "exit_count_match_pct": _count_match_pct(len(py_exits), len(mt5_exits)),
        "notes": [
            "Fill/exit count parity is weaker than setup parity because Python has only OHLC bars.",
            "MT5 trailing rows are tick-driven and can have many duplicates per position.",
        ],
    }


def write_parity_artifacts(report: dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "parity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_path / "discrepancies.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["key", "category", "python_value", "mt5_value", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report.get("discrepancies", []))
    (output_path / "parity_summary.md").write_text(_summary_markdown(report), encoding="utf-8")


def write_layered_parity_artifacts(
    python_rows: list[dict[str, Any]],
    mt5_rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_setup_daily_parity(python_rows, mt5_rows, output_path / "setup_daily_parity.csv")
    _write_fill_parity(python_rows, mt5_rows, output_path / "fill_parity.csv")
    _write_management_parity(python_rows, mt5_rows, output_path / "management_parity.csv")
    _write_mt5_duplicate_report(mt5_rows, output_path / "mt5_opportunity_duplicates.csv")


def _write_setup_daily_parity(
    python_rows: list[dict[str, Any]],
    mt5_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    py_daily = _first_daily_setup_rows(python_rows)
    mt5_daily = _first_daily_setup_rows(mt5_rows)
    mt5_opportunity_counts = Counter(
        _event_day(row) for row in mt5_rows if row.get("event_type") == "opportunity"
    )
    fields = [
        "event_type",
        "is_friday",
        "swing_high",
        "swing_low",
        "buy_entry",
        "sell_entry",
        "pending_distance_points",
        "ema20",
        "prev_close",
        "atr_points",
        "atr_filter_passed",
        "rsi_gate_required",
        "buy_order_placed",
        "sell_order_placed",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "day",
                "python_present",
                "mt5_present",
                "mt5_opportunity_rows",
                *[f"{field}_python" for field in fields],
                *[f"{field}_mt5" for field in fields],
                *[f"{field}_match" for field in fields],
            ],
        )
        writer.writeheader()
        for day in sorted(set(py_daily) | set(mt5_daily)):
            py_row = py_daily.get(day, {})
            mt5_row = mt5_daily.get(day, {})
            row: dict[str, Any] = {
                "day": day,
                "python_present": bool(py_row),
                "mt5_present": bool(mt5_row),
                "mt5_opportunity_rows": mt5_opportunity_counts.get(day, 0),
            }
            for field in fields:
                row[f"{field}_python"] = py_row.get(field, "")
                row[f"{field}_mt5"] = mt5_row.get(field, "")
                row[f"{field}_match"] = (
                    (
                        field in {"event_type", "is_friday"}
                        or (py_row.get("event_type") == "opportunity" and mt5_row.get("event_type") == "opportunity")
                    )
                    and _field_matches(field, py_row.get(field), mt5_row.get(field))
                    if py_row and mt5_row
                    else False
                )
            writer.writerow(row)


def _write_fill_parity(
    python_rows: list[dict[str, Any]],
    mt5_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    py_by_day = _rows_by_day(python_rows, "fill")
    mt5_by_day = _rows_by_day(mt5_rows, "fill")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "day",
                "python_fill_count",
                "mt5_fill_count",
                "count_match",
                "python_first_side",
                "mt5_first_side",
                "fill_side_match",
                "python_first_fill_price",
                "mt5_first_fill_price",
                "fill_price_diff",
                "python_duration_to_fill_seconds",
                "mt5_duration_to_fill_seconds",
                "duration_diff_seconds",
            ],
        )
        writer.writeheader()
        for day in sorted(set(py_by_day) | set(mt5_by_day)):
            py_rows = py_by_day.get(day, [])
            mt5_rows_day = mt5_by_day.get(day, [])
            py_first = py_rows[0] if py_rows else {}
            mt5_first = mt5_rows_day[0] if mt5_rows_day else {}
            py_price = _to_float(py_first.get("fill_price"))
            mt5_price = _to_float(mt5_first.get("fill_price"))
            py_duration = _to_float(py_first.get("duration_to_fill_seconds"))
            mt5_duration = _to_float(mt5_first.get("duration_to_fill_seconds"))
            writer.writerow(
                {
                    "day": day,
                    "python_fill_count": len(py_rows),
                    "mt5_fill_count": len(mt5_rows_day),
                    "count_match": len(py_rows) == len(mt5_rows_day),
                    "python_first_side": py_first.get("fill_side", ""),
                    "mt5_first_side": mt5_first.get("fill_side", ""),
                    "fill_side_match": _normal(py_first.get("fill_side")) == _normal(mt5_first.get("fill_side")),
                    "python_first_fill_price": py_first.get("fill_price", ""),
                    "mt5_first_fill_price": mt5_first.get("fill_price", ""),
                    "fill_price_diff": "" if py_price is None or mt5_price is None else py_price - mt5_price,
                    "python_duration_to_fill_seconds": py_first.get("duration_to_fill_seconds", ""),
                    "mt5_duration_to_fill_seconds": mt5_first.get("duration_to_fill_seconds", ""),
                    "duration_diff_seconds": ""
                    if py_duration is None or mt5_duration is None
                    else py_duration - mt5_duration,
                }
            )


def _write_management_parity(
    python_rows: list[dict[str, Any]],
    mt5_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    py_exits = _rows_by_day(python_rows, "exit")
    mt5_exits = _rows_by_day(mt5_rows, "exit")
    py_trailing = _rows_by_day(python_rows, "trailing_modified")
    mt5_trailing = _rows_by_day(mt5_rows, "trailing_modified")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "day",
                "python_trailing_modifications",
                "mt5_trailing_modifications",
                "python_exit_count",
                "mt5_exit_count",
                "python_exit_reason",
                "mt5_exit_reason",
                "exit_reason_match",
                "python_net_pnl",
                "mt5_net_pnl",
                "net_pnl_diff",
                "python_mfe_points",
                "mt5_mfe_points",
                "python_mae_points",
                "mt5_mae_points",
                "comparability",
            ],
        )
        writer.writeheader()
        for day in sorted(set(py_exits) | set(mt5_exits) | set(py_trailing) | set(mt5_trailing)):
            py_exit = py_exits.get(day, [{}])[0] if py_exits.get(day) else {}
            mt5_exit = mt5_exits.get(day, [{}])[0] if mt5_exits.get(day) else {}
            py_pnl = _to_float(py_exit.get("net_pnl"))
            mt5_pnl = _to_float(mt5_exit.get("net_pnl"))
            writer.writerow(
                {
                    "day": day,
                    "python_trailing_modifications": len(py_trailing.get(day, [])),
                    "mt5_trailing_modifications": len(mt5_trailing.get(day, [])),
                    "python_exit_count": len(py_exits.get(day, [])),
                    "mt5_exit_count": len(mt5_exits.get(day, [])),
                    "python_exit_reason": py_exit.get("exit_reason", ""),
                    "mt5_exit_reason": mt5_exit.get("exit_reason", ""),
                    "exit_reason_match": _normal(py_exit.get("exit_reason")) == _normal(mt5_exit.get("exit_reason")),
                    "python_net_pnl": py_exit.get("net_pnl", ""),
                    "mt5_net_pnl": mt5_exit.get("net_pnl", ""),
                    "net_pnl_diff": "" if py_pnl is None or mt5_pnl is None else py_pnl - mt5_pnl,
                    "python_mfe_points": py_exit.get("mfe_points", ""),
                    "mt5_mfe_points": mt5_exit.get("mfe_points", ""),
                    "python_mae_points": py_exit.get("mae_points", ""),
                    "mt5_mae_points": mt5_exit.get("mae_points", ""),
                    "comparability": "approximate_ohlc_vs_tick_mt5",
                }
            )


def _write_mt5_duplicate_report(mt5_rows: list[dict[str, Any]], output_path: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mt5_rows:
        if row.get("event_type") == "opportunity":
            grouped[_event_day(row)].append(row)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["day", "opportunity_rows", "first_timestamp", "last_timestamp", "first_notes"],
        )
        writer.writeheader()
        for day, rows in sorted(grouped.items()):
            writer.writerow(
                {
                    "day": day,
                    "opportunity_rows": len(rows),
                    "first_timestamp": rows[0].get("timestamp", ""),
                    "last_timestamp": rows[-1].get("timestamp", ""),
                    "first_notes": rows[0].get("notes", ""),
                }
            )


def _rows_by_day(rows: list[dict[str, Any]], event_type: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("event_type") == event_type:
            grouped[_event_day(row)].append(row)
    return grouped


def write_no_log_report(output_dir: str | Path, searched_paths: list[str]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "blocked_missing_mt5_log",
        "message": "No real MT5 azir_events_*.csv log was found or provided; parity was not computed.",
        "searched_paths": searched_paths,
    }
    (output_path / "parity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_path / "discrepancies.csv").write_text(
        "key,category,python_value,mt5_value,notes\n",
        encoding="utf-8",
    )
    (output_path / "parity_summary.md").write_text(_summary_markdown(report), encoding="utf-8")


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _event_key(row)
        grouped[key].append(row)
    return grouped


def _event_key(row: dict[str, Any]) -> str:
    return f"{_event_day(row)}_{row.get('symbol', '')}_{row.get('magic', '')}|{row.get('event_type', '')}"


def _event_day(row: dict[str, Any]) -> str:
    timestamp = str(row.get("timestamp", "")).strip()
    if timestamp:
        return timestamp.split(" ")[0].replace(".", "-")
    event_id = str(row.get("event_id", "")).strip()
    if event_id:
        return event_id.split("_")[0].replace(".", "-")
    return "unknown_date"


def _first_daily_setup_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("event_type") not in {"opportunity", "blocked_friday"}:
            continue
        grouped[_event_day(row)].append(row)
    return {day: _canonical_setup_row(day_rows) for day, day_rows in grouped.items()}


def _canonical_setup_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    opportunity_rows = [row for row in rows if row.get("event_type") == "opportunity"]
    if opportunity_rows:
        placed = [
            row
            for row in opportunity_rows
            if _normal(row.get("buy_order_placed")) == "true"
            or _normal(row.get("sell_order_placed")) == "true"
        ]
        if placed:
            return placed[-1]
        return opportunity_rows[0]
    return rows[0]


def _zip_with_missing(left: list[Any], right: list[Any]) -> list[tuple[Any | None, Any | None]]:
    size = max(len(left), len(right))
    return [
        (left[index] if index < len(left) else None, right[index] if index < len(right) else None)
        for index in range(size)
    ]


def _discrepancy(key: str, category: str, python_value: Any, mt5_value: Any, notes: str) -> dict[str, Any]:
    return {
        "key": key,
        "category": category,
        "python_value": python_value,
        "mt5_value": mt5_value,
        "notes": notes,
    }


def _blank(value: Any) -> bool:
    return value is None or str(value) == ""


def _normal(value: Any) -> str:
    return str(value).strip().lower()


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tolerance_name(field: str) -> str:
    if field.endswith("_points") or field == "pending_distance_points":
        return "points"
    if field.endswith("pnl"):
        return "pnl"
    return "price"


def _field_matches(field: str, left: Any, right: Any) -> bool:
    if field in {name for name, _, _ in NUMERIC_COMPARISONS}:
        left_number = _to_float(left)
        right_number = _to_float(right)
        if left_number is None and right_number is None:
            return True
        if left_number is None or right_number is None:
            return False
        tolerance = next((item[2] for item in NUMERIC_COMPARISONS if item[0] == field), 0.0)
        return abs(left_number - right_number) <= tolerance
    return _normal(left) == _normal(right)


def _count_match_pct(left: int, right: int) -> float:
    if left == right == 0:
        return 100.0
    denominator = max(left, right)
    return round((min(left, right) / denominator * 100.0), 4) if denominator else 0.0


def _summary_markdown(report: dict[str, Any]) -> str:
    if report.get("status") == "blocked_missing_mt5_log":
        searched = "\n".join(f"- `{path}`" for path in report.get("searched_paths", []))
        return (
            "# Azir Python Replica Parity\n\n"
            "Status: BLOCKED - missing real MT5 log.\n\n"
            "The Python replica can generate `python_events.csv`, but no real "
            "`azir_events_*.csv` file was found/provided, so empirical parity cannot "
            "be computed yet.\n\n"
            "Searched paths:\n\n"
            f"{searched}\n"
        )

    categories = report.get("discrepancies_by_category", {})
    category_lines = "\n".join(f"- `{key}`: {value}" for key, value in categories.items())
    hard_parts = "\n".join(f"- {item}" for item in report.get("hard_to_reproduce", []))
    return (
        "# Azir Python Replica Parity\n\n"
        f"Status: {report.get('status')}\n\n"
        f"- Python events: {report.get('python_events')}\n"
        f"- MT5 events: {report.get('mt5_events')}\n"
        f"- Unique event keys: {report.get('unique_event_keys')}\n"
        f"- Event match pct: {report.get('event_match_pct')}%\n"
        f"- Field match pct: {report.get('field_match_pct')}%\n"
        f"- Discrepancies: {report.get('discrepancy_count')}\n\n"
        "## Daily Opportunity Parity\n\n"
        f"- Setup day match pct: {report.get('daily_opportunity_parity', {}).get('setup_day_match_pct')}%\n"
        f"- Daily setup field match pct: {report.get('daily_opportunity_parity', {}).get('field_match_pct')}%\n"
        f"- MT5 duplicate opportunity rows: {report.get('daily_opportunity_parity', {}).get('mt5_duplicate_opportunity_rows')}\n\n"
        "## Fill/Exit Coverage\n\n"
        f"- Python fills: {report.get('fill_exit_coverage', {}).get('python_fills')}\n"
        f"- MT5 fills: {report.get('fill_exit_coverage', {}).get('mt5_fills')}\n"
        f"- Fill count match pct: {report.get('fill_exit_coverage', {}).get('fill_count_match_pct')}%\n\n"
        "## Discrepancies by Category\n\n"
        f"{category_lines or '- None'}\n\n"
        "## Hardest Parts to Reproduce\n\n"
        f"{hard_parts}\n"
    )
