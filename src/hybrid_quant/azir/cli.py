"""Command line runner for the Azir Python replica and MT5 parity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .comparison import (
    compare_event_logs,
    read_event_log,
    write_layered_parity_artifacts,
    write_no_log_report,
    write_parity_artifacts,
)
from .event_log import write_event_log
from .replica import AzirPythonReplica, AzirReplicaConfig, load_ohlcv_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Azir Python replica and compare it to MT5 logs.")
    parser.add_argument("--input-path", required=True, help="Historical XAUUSD M5 OHLCV CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for replica and parity artifacts.")
    parser.add_argument("--mt5-log-path", default="", help="Optional real azir_events_*.csv exported from MT5.")
    parser.add_argument("--m1-input-path", default="", help="Optional M1 OHLCV CSV for RSI gate parity.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--start", default="", help="Optional inclusive start timestamp/date in server time.")
    parser.add_argument("--end", default="", help="Optional inclusive end timestamp/date in server time.")
    parser.add_argument("--point", type=float, default=0.01)
    parser.add_argument("--contract-size", type=float, default=100.0)
    parser.add_argument("--lot-size", type=float, default=0.10)
    parser.add_argument("--ny-open-hour", type=int, default=16)
    parser.add_argument("--ny-open-minute", type=int, default=30)
    parser.add_argument("--close-hour", type=int, default=22)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bars = _filter_bars(load_ohlcv_csv(args.input_path), args.start, args.end)
    rsi_bars = _filter_bars(load_ohlcv_csv(args.m1_input_path), args.start, args.end) if args.m1_input_path else None
    config = AzirReplicaConfig(
        symbol=args.symbol,
        point=args.point,
        contract_size=args.contract_size,
        lot_size=args.lot_size,
        ny_open_hour=args.ny_open_hour,
        ny_open_minute=args.ny_open_minute,
        close_hour=args.close_hour,
    )
    replica = AzirPythonReplica(bars, config=config, rsi_bars=rsi_bars)
    python_rows = replica.run()
    write_event_log(python_rows, output_dir / "python_events.csv")
    _write_run_metadata(args, bars, python_rows, output_dir)

    if args.mt5_log_path:
        mt5_rows = _filter_events(read_event_log(args.mt5_log_path), args.start, args.end)
        report = compare_event_logs(python_rows, mt5_rows)
        write_parity_artifacts(report, output_dir)
        write_layered_parity_artifacts(python_rows, mt5_rows, output_dir)
    else:
        write_no_log_report(
            output_dir,
            searched_paths=[
                "C:\\Users\\joseq\\Documents\\Playground\\**\\azir_events_*.csv",
                "C:\\Users\\joseq\\Documents\\**\\azir_events_*.csv",
                "%APPDATA%\\MetaQuotes\\Terminal\\Common\\Files\\**\\azir_events_*.csv",
            ],
        )

    return 0


def _write_run_metadata(
    args: argparse.Namespace,
    bars: list,
    python_rows: list[dict],
    output_dir: Path,
) -> None:
    metadata = {
        "input_path": args.input_path,
        "mt5_log_path": args.mt5_log_path or None,
        "m1_input_path": args.m1_input_path or None,
        "start": args.start or None,
        "end": args.end or None,
        "first_bar": bars[0].open_time.isoformat(sep=" "),
        "last_bar": bars[-1].open_time.isoformat(sep=" "),
        "bar_count": len(bars),
        "python_event_count": len(python_rows),
        "notes": [
            "Session timing is interpreted as raw MT5 broker/server time.",
            "Fills, trailing and exits are approximated from OHLC bars; tick ordering is not available.",
            "M1 RSI parity requires --m1-input-path if the RSI gate can activate.",
        ],
    }
    (output_dir / "replica_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _filter_bars(bars: list, start: str, end: str) -> list:
    start_dt = _parse_optional_datetime(start, is_end=False)
    end_dt = _parse_optional_datetime(end, is_end=True)
    return [
        bar for bar in bars
        if (start_dt is None or bar.open_time >= start_dt)
        and (end_dt is None or bar.open_time <= end_dt)
    ]


def _filter_events(rows: list[dict], start: str, end: str) -> list[dict]:
    start_dt = _parse_optional_datetime(start, is_end=False)
    end_dt = _parse_optional_datetime(end, is_end=True)
    filtered: list[dict] = []
    for row in rows:
        timestamp = str(row.get("timestamp", "")).replace(".", "-")
        try:
            event_dt = _parse_optional_datetime(timestamp, is_end=False)
        except ValueError:
            continue
        if event_dt is None:
            continue
        if start_dt is not None and event_dt < start_dt:
            continue
        if end_dt is not None and event_dt > end_dt:
            continue
        filtered.append(row)
    return filtered


def _parse_optional_datetime(value: str, *, is_end: bool) -> object | None:
    from datetime import datetime, time

    if not value:
        return None
    value = value.strip().replace(".", "-")
    if len(value) == 10:
        day = datetime.fromisoformat(value).date()
        return datetime.combine(day, time(23, 59, 59) if is_end else time.min)
    return datetime.fromisoformat(value.replace("T", " "))


if __name__ == "__main__":
    raise SystemExit(main())
