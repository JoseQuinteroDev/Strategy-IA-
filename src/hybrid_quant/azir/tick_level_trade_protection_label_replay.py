"""Tick-level early snapshot replay for Azir trade-protection labels.

The goal is coverage repair, not model training. Real ticks are used first for
30s/60s/120s/180s post-fill snapshots. M1 is a documented fallback only when a
closed M1 bar is available at or before the requested snapshot timestamp.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

from hybrid_quant.env.azir_management_env import AzirManagementEvent, build_azir_management_replay_dataset
from hybrid_quant.risk.azir_state import AzirRiskConfig

from .economic_audit import _parse_timestamp, _round, _to_float, _write_csv
from .management_replay_v2 import TickRecord, TickWindow, extract_ticks_for_windows
from .replica import OhlcvBar, load_ohlcv_csv
from .trade_protection_label_diagnostics import (
    LabelDiagnosticsConfig,
    build_label_distribution,
    build_separability_summary,
    diagnose_features,
    label_post_entry_rows,
)
from .trade_protection_research import (
    DEFAULT_M1_PATH,
    DEFAULT_M5_PATH,
    DEFAULT_MT5_LOG,
    DEFAULT_PROTECTED_REPORT,
    DEFAULT_TICK_PATH,
    PriceSeries,
    TradeProtectionConfig,
)


DEFAULT_OUTPUT_DIR = Path("artifacts/azir-tick-level-trade-protection-label-replay-v1")
DEFAULT_RESEARCH_ARTIFACT_DIR = Path("artifacts/azir-trade-protection-research-v1")
DEFAULT_LABEL_ARTIFACT_DIR = Path("artifacts/azir-trade-protection-label-diagnostics-v1")


@dataclass(frozen=True)
class TickProtectionSnapshotConfig:
    snapshot_seconds: tuple[int, ...] = (30, 60, 120, 180)
    point: float = 0.01
    contract_size: float = 100.0
    default_lot_size: float = 0.10
    default_sl_points: float = 500.0
    default_tp_points: float = 500.0
    session_close_hour: int = 22
    min_ticks_for_tick_snapshot: int = 2
    max_m1_fallback_seconds: int = 180


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build tick-level early post-entry labels for Azir trade protection.")
    parser.add_argument("--mt5-log-path", default=str(DEFAULT_MT5_LOG))
    parser.add_argument("--protected-report-path", default=str(DEFAULT_PROTECTED_REPORT))
    parser.add_argument("--tick-input-path", default=str(DEFAULT_TICK_PATH))
    parser.add_argument("--m1-input-path", default=str(DEFAULT_M1_PATH))
    parser.add_argument("--m5-input-path", default=str(DEFAULT_M5_PATH))
    parser.add_argument("--research-artifact-dir", default=str(DEFAULT_RESEARCH_ARTIFACT_DIR))
    parser.add_argument("--label-artifact-dir", default=str(DEFAULT_LABEL_ARTIFACT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbol", default="XAUUSD-STD")
    parser.add_argument("--tick-symbol-label", default="XAUUSD-STD")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_tick_level_trade_protection_label_replay(
        mt5_log_path=Path(args.mt5_log_path),
        protected_report_path=Path(args.protected_report_path),
        tick_input_path=Path(args.tick_input_path),
        m1_input_path=Path(args.m1_input_path),
        m5_input_path=Path(args.m5_input_path),
        research_artifact_dir=Path(args.research_artifact_dir),
        label_artifact_dir=Path(args.label_artifact_dir),
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
        tick_symbol_label=args.tick_symbol_label,
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_tick_level_trade_protection_label_replay(
    *,
    mt5_log_path: Path,
    protected_report_path: Path,
    tick_input_path: Path,
    m1_input_path: Path,
    m5_input_path: Path,
    research_artifact_dir: Path,
    label_artifact_dir: Path,
    output_dir: Path,
    symbol: str = "XAUUSD-STD",
    tick_symbol_label: str = "XAUUSD-STD",
    snapshot_config: TickProtectionSnapshotConfig | None = None,
    label_config: LabelDiagnosticsConfig | None = None,
) -> dict[str, Any]:
    snapshot_config = snapshot_config or TickProtectionSnapshotConfig()
    label_config = label_config or LabelDiagnosticsConfig(min_labeled_snapshots=500)
    events = build_azir_management_replay_dataset(
        mt5_log_path=mt5_log_path,
        protected_report_path=protected_report_path,
        symbol=symbol,
        risk_config=AzirRiskConfig(),
    )
    m1_bars = load_ohlcv_csv(m1_input_path) if m1_input_path.exists() else []
    m5_bars = load_ohlcv_csv(m5_input_path) if m5_input_path.exists() else []
    m1_series = PriceSeries.from_bars("m1_fallback", 1, m1_bars)
    m5_series = PriceSeries.from_bars("m5_reference", 5, m5_bars)

    tick_windows = build_tick_snapshot_windows(events, snapshot_config)
    tick_map, tick_inspection = extract_ticks_for_windows(
        tick_input_path,
        tick_windows,
        tick_symbol_label=tick_symbol_label,
    )
    snapshot_rows = build_tick_level_snapshot_rows(events, tick_map, m1_series, snapshot_config)
    labeled_rows = label_post_entry_rows(snapshot_rows, events, label_config)
    label_distribution_rows = build_label_distribution(labeled_rows)
    feature_rows = diagnose_features(labeled_rows)
    separability = build_separability_summary(feature_rows, label_distribution_rows, label_config)
    coverage_rows = build_tick_coverage_bias_report(events, labeled_rows, snapshot_config)
    report = build_report(
        mt5_log_path=mt5_log_path,
        protected_report_path=protected_report_path,
        tick_input_path=tick_input_path,
        m1_input_path=m1_input_path,
        m5_input_path=m5_input_path,
        research_artifact_dir=research_artifact_dir,
        label_artifact_dir=label_artifact_dir,
        events=events,
        snapshot_rows=snapshot_rows,
        labeled_rows=labeled_rows,
        label_distribution_rows=label_distribution_rows,
        feature_rows=feature_rows,
        coverage_rows=coverage_rows,
        separability=separability,
        tick_inspection=tick_inspection,
        snapshot_config=snapshot_config,
        label_config=label_config,
        m1_bars=m1_bars,
        m5_bars=m5_bars,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(labeled_rows, output_dir / "tick_level_trade_protection_labeled_dataset.csv")
    _write_csv(labeled_rows[:100], output_dir / "tick_level_trade_protection_sample.csv")
    _write_csv(label_distribution_rows, output_dir / "tick_level_label_distribution.csv")
    _write_csv(coverage_rows, output_dir / "tick_level_coverage_bias_report.csv")
    _write_csv(feature_rows, output_dir / "tick_level_feature_diagnostics.csv")
    (output_dir / "tick_level_snapshot_schema.md").write_text(snapshot_schema_markdown(snapshot_config, label_config), encoding="utf-8")
    (output_dir / "tick_level_separability_report.md").write_text(separability_markdown(report), encoding="utf-8")
    (output_dir / "tick_level_trade_protection_summary.md").write_text(summary_markdown(report), encoding="utf-8")
    (output_dir / "tick_level_trade_protection_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_tick_snapshot_windows(
    events: list[AzirManagementEvent],
    config: TickProtectionSnapshotConfig,
) -> list[TickWindow]:
    max_seconds = max(config.snapshot_seconds)
    windows: list[TickWindow] = []
    for event in events:
        exit_dt = _event_exit_dt(event)
        if exit_dt is None or exit_dt <= event.fill_timestamp:
            continue
        end_dt = min(exit_dt, event.fill_timestamp + timedelta(seconds=max_seconds))
        windows.append(
            TickWindow(
                key=_event_key(event),
                start_text=event.fill_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                end_text=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
    return windows


def build_tick_level_snapshot_rows(
    events: list[AzirManagementEvent],
    tick_map: dict[str, list[TickRecord]],
    m1_series: PriceSeries,
    config: TickProtectionSnapshotConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        exit_dt = _event_exit_dt(event)
        if exit_dt is None:
            continue
        ticks = tick_map.get(_event_key(event), [])
        for seconds_after_fill in config.snapshot_seconds:
            snapshot_dt = event.fill_timestamp + timedelta(seconds=seconds_after_fill)
            if snapshot_dt > exit_dt:
                continue
            tick_row = _tick_snapshot_row(event, ticks, snapshot_dt, seconds_after_fill, config)
            if tick_row is not None:
                rows.append(tick_row)
                continue
            m1_row = _m1_fallback_snapshot_row(event, m1_series, snapshot_dt, seconds_after_fill, config)
            if m1_row is not None:
                rows.append(m1_row)
    return rows


def _tick_snapshot_row(
    event: AzirManagementEvent,
    ticks: list[TickRecord],
    snapshot_dt: datetime,
    seconds_after_fill: int,
    config: TickProtectionSnapshotConfig,
) -> dict[str, Any] | None:
    path = [tick for tick in ticks if event.fill_timestamp <= tick.timestamp <= snapshot_dt]
    if len(path) < config.min_ticks_for_tick_snapshot:
        return None
    current_tick = path[-1]
    prices = [_exit_price(event.side, tick) for tick in path]
    spreads = [max(0.0, tick.ask - tick.bid) for tick in path]
    return _snapshot_row_from_path(
        event=event,
        snapshot_dt=current_tick.timestamp,
        seconds_after_fill=seconds_after_fill,
        source="tick",
        current_price=prices[-1],
        high_price=max(prices),
        low_price=min(prices),
        first_price=prices[0],
        tick_count=len(path),
        volume_proxy=float(len(path)),
        spread_current=spreads[-1],
        spread_mean=mean(spreads) if spreads else 0.0,
        config=config,
    )


def _m1_fallback_snapshot_row(
    event: AzirManagementEvent,
    m1_series: PriceSeries,
    snapshot_dt: datetime,
    seconds_after_fill: int,
    config: TickProtectionSnapshotConfig,
) -> dict[str, Any] | None:
    if seconds_after_fill > config.max_m1_fallback_seconds:
        return None
    bars = [
        bar
        for bar in m1_series.bars
        if event.fill_timestamp <= bar.open_time and bar.open_time + timedelta(minutes=1) <= snapshot_dt
    ]
    if not bars:
        return None
    current = bars[-1]
    prices = [_bar_exit_price(event.side, bar, "close") for bar in bars]
    highs = [_bar_exit_price(event.side, bar, "high") for bar in bars]
    lows = [_bar_exit_price(event.side, bar, "low") for bar in bars]
    return _snapshot_row_from_path(
        event=event,
        snapshot_dt=current.open_time + timedelta(minutes=1),
        seconds_after_fill=seconds_after_fill,
        source="m1_fallback",
        current_price=prices[-1],
        high_price=max(highs) if event.side == "buy" else max(prices),
        low_price=min(lows) if event.side == "buy" else min(prices),
        first_price=prices[0],
        tick_count=0,
        volume_proxy=sum(bar.volume for bar in bars),
        spread_current=0.0,
        spread_mean=0.0,
        config=config,
    )


def _snapshot_row_from_path(
    *,
    event: AzirManagementEvent,
    snapshot_dt: datetime,
    seconds_after_fill: int,
    source: str,
    current_price: float,
    high_price: float,
    low_price: float,
    first_price: float,
    tick_count: int,
    volume_proxy: float,
    spread_current: float,
    spread_mean: float,
    config: TickProtectionSnapshotConfig,
) -> dict[str, Any]:
    entry = _entry(event) or 0.0
    setup = event.setup
    atr_points = _float(setup.get("atr_points"))
    sl_points = _float(setup.get("sl_points")) or config.default_sl_points
    tp_points = _float(setup.get("tp_points")) or config.default_tp_points
    mfe = _mfe_points(event.side, entry, high_price, low_price, config.point)
    mae = _mae_points(event.side, entry, high_price, low_price, config.point)
    unrealized = _pnl(event.side, entry, current_price, _lot_size(event, config), config.contract_size)
    minutes_after_fill = seconds_after_fill / 60.0
    direction_proxy = _side_adjusted_move(event.side, first_price, current_price)
    return {
        "event_key": _event_key(event),
        "setup_day": event.setup_day,
        "fill_timestamp": event.fill_timestamp.isoformat(sep=" "),
        "snapshot_timestamp": snapshot_dt.isoformat(sep=" "),
        "snapshot_seconds_after_fill": seconds_after_fill,
        "snapshot_minutes_after_fill": _round(minutes_after_fill),
        "data_source": source,
        "side": event.side,
        "entry_price": _round(entry),
        "initial_sl_points": _round(sl_points),
        "initial_tp_points": _round(tp_points),
        "distance_to_initial_sl_points": _round(max(sl_points - mae, 0.0)),
        "distance_to_initial_tp_points": _round(max(tp_points - mfe, 0.0)),
        "mfe_points_so_far": _round(mfe),
        "mae_points_so_far": _round(mae),
        "unrealized_pnl_so_far": _round(unrealized),
        "time_to_session_close_minutes": _round(_minutes_to_session_close(snapshot_dt, config.session_close_hour)),
        "trailing_activated_so_far_proxy": mfe >= (_float(setup.get("trailing_start_points")) or 90.0),
        "trailing_modifications_so_far_proxy": 0.0,
        "atr_points_setup": _round(atr_points),
        "atr_relative_mfe": _round(mfe / atr_points) if atr_points else "",
        "atr_relative_mae": _round(mae / atr_points) if atr_points else "",
        "post_entry_speed_points_per_min": _round(mfe / minutes_after_fill) if minutes_after_fill else "",
        "m1_m5_close_direction_proxy": _round(direction_proxy),
        "volume_proxy_sum": _round(volume_proxy),
        "tick_count_so_far": tick_count,
        "spread_points_current": _round(spread_current / config.point) if spread_current else "",
        "spread_points_mean": _round(spread_mean / config.point) if spread_mean else "",
        "spread_points_setup": _float(setup.get("spread_points")) or "",
        "spread_to_atr": _round((_float(setup.get("spread_points")) or 0.0) / atr_points) if atr_points else "",
        "fill_hour": event.fill_timestamp.hour,
        "day_of_week": setup.get("day_of_week", ""),
        "risk_context_trades_per_day_policy": 1,
        "risk_context_consecutive_losses_policy": 2,
        "future_outcome_excluded": True,
    }


def build_tick_coverage_bias_report(
    events: list[AzirManagementEvent],
    labeled_rows: list[dict[str, Any]],
    config: TickProtectionSnapshotConfig,
) -> list[dict[str, Any]]:
    labeled_keys = {str(row["event_key"]) for row in labeled_rows}
    by_key_snapshots = Counter(str(row["event_key"]) for row in labeled_rows)
    rows: list[dict[str, Any]] = []
    groups = {
        "year": lambda event: event.setup_day[:4],
        "side": lambda event: event.side,
        "winner_loser": lambda event: "winner" if event.protected_net_pnl > 0 else "loser" if event.protected_net_pnl < 0 else "flat",
        "duration_bucket": lambda event: _duration_bucket(event),
    }
    for group_name, group_fn in groups.items():
        grouped: dict[str, list[AzirManagementEvent]] = defaultdict(list)
        for event in events:
            grouped[group_fn(event)].append(event)
        for value, group_events in sorted(grouped.items()):
            included = [event for event in group_events if _event_key(event) in labeled_keys]
            rows.append(
                {
                    "group": group_name,
                    "value": value,
                    "total_trades": len(group_events),
                    "included_trades": len(included),
                    "excluded_trades": len(group_events) - len(included),
                    "included_pct": _round(len(included) / len(group_events) * 100.0) if group_events else 0.0,
                    "avg_base_pnl_all": _round(mean(event.protected_net_pnl for event in group_events)) if group_events else 0.0,
                    "avg_base_pnl_included": _round(mean(event.protected_net_pnl for event in included)) if included else "",
                    "snapshot_rows": sum(by_key_snapshots.get(_event_key(event), 0) for event in group_events),
                }
            )
    min_seconds = min(config.snapshot_seconds)
    reasons: Counter[str] = Counter()
    for event in events:
        if _event_key(event) in labeled_keys:
            continue
        duration_seconds = _duration_seconds(event)
        if duration_seconds is None:
            reasons["missing_exit_timestamp"] += 1
        elif duration_seconds < min_seconds:
            reasons[f"duration_below_{min_seconds}s_first_snapshot"] += 1
        else:
            reasons["no_tick_or_causal_m1_snapshot"] += 1
    for reason, count in sorted(reasons.items()):
        rows.append(
            {
                "group": "exclusion_reason",
                "value": reason,
                "total_trades": len(events),
                "included_trades": "",
                "excluded_trades": count,
                "included_pct": "",
                "avg_base_pnl_all": "",
                "avg_base_pnl_included": "",
                "snapshot_rows": "",
            }
        )
    source_counts = Counter(str(row.get("data_source", "")) for row in labeled_rows)
    for source, count in sorted(source_counts.items()):
        rows.append(
            {
                "group": "pricing_source",
                "value": source,
                "total_trades": len(labeled_rows),
                "included_trades": count,
                "excluded_trades": "",
                "included_pct": _round(count / len(labeled_rows) * 100.0) if labeled_rows else 0.0,
                "avg_base_pnl_all": "",
                "avg_base_pnl_included": "",
                "snapshot_rows": count,
            }
        )
    return rows


def build_report(
    *,
    mt5_log_path: Path,
    protected_report_path: Path,
    tick_input_path: Path,
    m1_input_path: Path,
    m5_input_path: Path,
    research_artifact_dir: Path,
    label_artifact_dir: Path,
    events: list[AzirManagementEvent],
    snapshot_rows: list[dict[str, Any]],
    labeled_rows: list[dict[str, Any]],
    label_distribution_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    separability: dict[str, Any],
    tick_inspection: Any,
    snapshot_config: TickProtectionSnapshotConfig,
    label_config: LabelDiagnosticsConfig,
    m1_bars: list[OhlcvBar],
    m5_bars: list[OhlcvBar],
) -> dict[str, Any]:
    labeled_keys = {str(row["event_key"]) for row in labeled_rows}
    old_coverage = load_previous_coverage(label_artifact_dir)
    current_coverage = coverage_summary(events, coverage_rows, labeled_rows)
    best_label = max(
        (
            label
            for label, payload in separability.items()
            if payload["separability_assessment"] == "promising_for_diagnostics"
        ),
        key=lambda label: _float(separability[label]["best_auc_edge_abs"]),
        default="",
    )
    bias_acceptable = current_coverage["winner_included_pct"] >= 50.0 and current_coverage["winner_loser_inclusion_gap_pct"] <= 20.0
    ready = (
        bool(best_label)
        and len(labeled_keys) >= 400
        and bias_acceptable
        and _float(separability.get(best_label, {}).get("best_auc_edge_abs")) >= 0.08
    )
    return {
        "sprint": "tick_level_trade_protection_label_replay_v1",
        "source_paths": {
            "mt5_log_path": str(mt5_log_path),
            "protected_report_path": str(protected_report_path),
            "tick_input_path": str(tick_input_path),
            "m1_input_path": str(m1_input_path),
            "m5_input_path": str(m5_input_path),
            "trade_protection_research_artifact_dir": str(research_artifact_dir),
            "previous_label_artifact_dir": str(label_artifact_dir),
        },
        "inputs_found": {
            "mt5_log_path": mt5_log_path.exists(),
            "protected_report_path": protected_report_path.exists(),
            "tick_input_path": tick_input_path.exists(),
            "m1_input_path": m1_input_path.exists(),
            "m5_input_path": m5_input_path.exists(),
            "trade_protection_research_artifact_dir": research_artifact_dir.exists(),
            "previous_label_artifact_dir": label_artifact_dir.exists(),
        },
        "tick_inspection": asdict(tick_inspection),
        "bar_coverage": {
            "m1_bars": len(m1_bars),
            "m1_start": m1_bars[0].open_time.isoformat(sep=" ") if m1_bars else "",
            "m1_end": m1_bars[-1].open_time.isoformat(sep=" ") if m1_bars else "",
            "m5_bars": len(m5_bars),
            "m5_start": m5_bars[0].open_time.isoformat(sep=" ") if m5_bars else "",
            "m5_end": m5_bars[-1].open_time.isoformat(sep=" ") if m5_bars else "",
        },
        "snapshot_config": asdict(snapshot_config),
        "label_config": asdict(label_config),
        "dataset": {
            "protected_trade_events": len(events),
            "snapshot_rows": len(snapshot_rows),
            "labeled_rows": len(labeled_rows),
            "unique_labeled_trades": len(labeled_keys),
            "tick_snapshot_rows": sum(1 for row in labeled_rows if row.get("data_source") == "tick"),
            "m1_fallback_snapshot_rows": sum(1 for row in labeled_rows if row.get("data_source") == "m1_fallback"),
            "feature_count_diagnosed": len(set(row["feature"] for row in feature_rows)),
        },
        "coverage_comparison": {
            "previous": old_coverage,
            "current": current_coverage,
            "delta_unique_labeled_trades": current_coverage["unique_labeled_trades"] - old_coverage.get("unique_labeled_trades", 0),
            "delta_winner_included_pct": _round(current_coverage["winner_included_pct"] - old_coverage.get("winner_included_pct", 0.0)),
            "delta_loser_included_pct": _round(current_coverage["loser_included_pct"] - old_coverage.get("loser_included_pct", 0.0)),
        },
        "label_distribution": label_distribution_rows,
        "separability": separability,
        "top_feature_diagnostics": feature_rows[:25],
        "coverage_bias": coverage_rows,
        "decision": {
            "best_label_candidate": best_label,
            "coverage_materially_improved": current_coverage["unique_labeled_trades"] > old_coverage.get("unique_labeled_trades", 0),
            "winner_loser_bias_acceptable": bias_acceptable,
            "label_momentum_break_true_still_promising": separability.get("label_momentum_break_true", {}).get("separability_assessment") == "promising_for_diagnostics",
            "ready_for_supervised_trade_protection_model_v1": ready,
            "ready_for_ppo_or_rl": False,
            "recommended_next_sprint": "supervised_trade_protection_model_v1" if ready else "tick_level_trade_protection_threshold_refinement_v1",
            "reason": (
                "Tick-level snapshots improved coverage and kept label separability with acceptable winner/loser balance."
                if ready
                else "Tick-level snapshots improve the evidence base, but coverage balance or label robustness is still not strong enough for a final supervised model."
            ),
        },
        "limitations": [
            "Snapshots require the trade to still be open at the requested horizon; very fast exits before 30s remain impossible to label at 30s.",
            "Tick data has no explicit timezone/symbol column; export context is used for symbol/clock interpretation.",
            "M1 fallback is used only when a closed M1 bar exists by the snapshot timestamp.",
            "This sprint diagnoses labels/features only and does not train a final model.",
        ],
    }


def coverage_summary(
    events: list[AzirManagementEvent],
    coverage_rows: list[dict[str, Any]],
    labeled_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def included_pct(group: str, value: str) -> float:
        row = next((item for item in coverage_rows if item.get("group") == group and item.get("value") == value), {})
        return _float(row.get("included_pct"))

    labeled_keys = {str(row["event_key"]) for row in labeled_rows}
    return {
        "unique_labeled_trades": len(labeled_keys),
        "snapshot_rows": len(labeled_rows),
        "unique_labeled_trade_pct": _round(len(labeled_keys) / len(events) * 100.0) if events else 0.0,
        "winner_included_pct": included_pct("winner_loser", "winner"),
        "loser_included_pct": included_pct("winner_loser", "loser"),
        "winner_loser_inclusion_gap_pct": _round(abs(included_pct("winner_loser", "winner") - included_pct("winner_loser", "loser"))),
        "tick_snapshot_rows": sum(1 for row in labeled_rows if row.get("data_source") == "tick"),
        "m1_fallback_snapshot_rows": sum(1 for row in labeled_rows if row.get("data_source") == "m1_fallback"),
    }


def load_previous_coverage(label_artifact_dir: Path) -> dict[str, Any]:
    report_path = label_artifact_dir / "trade_protection_label_report.json"
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report.get("coverage_bias", [])

    def included_pct(group: str, value: str) -> float:
        row = next((item for item in rows if item.get("group") == group and item.get("value") == value), {})
        return _float(row.get("included_pct"))

    return {
        "unique_labeled_trades": int(report.get("dataset", {}).get("unique_labeled_trades", 0)),
        "snapshot_rows": int(report.get("dataset", {}).get("labeled_rows", 0)),
        "winner_included_pct": included_pct("winner_loser", "winner"),
        "loser_included_pct": included_pct("winner_loser", "loser"),
    }


def snapshot_schema_markdown(
    snapshot_config: TickProtectionSnapshotConfig,
    label_config: LabelDiagnosticsConfig,
) -> str:
    horizons = ", ".join(f"{item}s" for item in snapshot_config.snapshot_seconds)
    return (
        "# Tick-Level Trade Protection Snapshot Schema\n\n"
        f"- Horizons: {horizons} post-fill.\n"
        "- Primary source: real broker/MT5 ticks from `tick_level.csv`.\n"
        "- Fallback: M1 only when a fully closed M1 bar exists by the snapshot timestamp.\n"
        "- No snapshot is generated if the trade has already exited before the horizon.\n\n"
        "## Causal Features\n\n"
        "- Current bid/ask-derived exit price, MFE/MAE so far, unrealized PnL so far, side-adjusted movement, tick count, spread, ATR-relative distances, time to session close.\n\n"
        "## Labels\n\n"
        "- Labels reuse `trade_protection_feature_label_diagnostics_v1` definitions.\n"
        f"- Early-close materiality threshold: {label_config.materiality_pnl} net PnL.\n\n"
        "## Leakage Rules\n\n"
        "- Final protected PnL is label-only.\n"
        "- Final exit reason is not a feature.\n"
        "- Ticks after the snapshot timestamp are prohibited from features.\n"
    )


def separability_markdown(report: dict[str, Any]) -> str:
    lines = []
    for label, payload in report["separability"].items():
        top = ", ".join(f"`{item['feature']}` edge={item['auc_edge_abs']}" for item in payload["top_features"][:4])
        lines.append(
            f"- `{label}`: positive={payload['positive_pct']}%, best_auc_edge={payload['best_auc_edge_abs']}, "
            f"assessment=`{payload['separability_assessment']}`. Top: {top}."
        )
    decision = report["decision"]
    return (
        "# Tick-Level Trade Protection Separability Report\n\n"
        f"- Labeled rows: {report['dataset']['labeled_rows']}.\n"
        f"- Unique labeled trades: {report['dataset']['unique_labeled_trades']}.\n"
        f"- Best label: `{decision['best_label_candidate'] or 'none'}`.\n"
        f"- Momentum-break still promising: {decision['label_momentum_break_true_still_promising']}.\n"
        f"- Ready for supervised v1: {decision['ready_for_supervised_trade_protection_model_v1']}.\n\n"
        "## Label Separability\n\n"
        + "\n".join(lines)
        + "\n"
    )


def summary_markdown(report: dict[str, Any]) -> str:
    current = report["coverage_comparison"]["current"]
    previous = report["coverage_comparison"]["previous"]
    decision = report["decision"]
    labels = "\n".join(
        f"- `{row['label']}`: positives={row['positive_rows']}/{row['rows']} ({row['positive_pct']}%)."
        for row in report["label_distribution"]
    )
    return (
        "# Tick-Level Trade Protection Label Replay V1\n\n"
        "## Executive Summary\n\n"
        f"- Protected events: {report['dataset']['protected_trade_events']}.\n"
        f"- Unique labeled trades: {current['unique_labeled_trades']} vs previous {previous.get('unique_labeled_trades', 0)}.\n"
        f"- Winner coverage: {current['winner_included_pct']}% vs previous {previous.get('winner_included_pct', 0.0)}%.\n"
        f"- Loser coverage: {current['loser_included_pct']}% vs previous {previous.get('loser_included_pct', 0.0)}%.\n"
        f"- Tick rows used for snapshots: {report['dataset']['tick_snapshot_rows']}.\n"
        f"- M1 fallback rows: {report['dataset']['m1_fallback_snapshot_rows']}.\n"
        f"- Best label candidate: `{decision['best_label_candidate'] or 'none'}`.\n"
        f"- Ready for supervised model: {decision['ready_for_supervised_trade_protection_model_v1']}.\n\n"
        "## Label Distribution\n\n"
        f"{labels}\n\n"
        "## Decision\n\n"
        f"- {decision['reason']}\n"
        f"- Recommended next sprint: `{decision['recommended_next_sprint']}`.\n"
    )


def _event_exit_dt(event: AzirManagementEvent) -> datetime | None:
    text = str(event.trade.get("exit_timestamp", "") or "")
    if not text:
        return None
    parsed = _parse_timestamp(text)
    return parsed if parsed.year > 1 else None


def _event_key(event: AzirManagementEvent) -> str:
    return f"{event.setup_day}|{event.fill_timestamp.isoformat(sep=' ')}|{event.side}"


def _entry(event: AzirManagementEvent) -> float | None:
    return _to_float(event.trade.get("fill_price"))


def _lot_size(event: AzirManagementEvent, config: TickProtectionSnapshotConfig) -> float:
    parsed = _to_float(event.setup.get("lot_size"))
    return parsed if parsed is not None and parsed > 0.0 else config.default_lot_size


def _exit_price(side: str, tick: TickRecord) -> float:
    return tick.bid if side == "buy" else tick.ask


def _bar_exit_price(side: str, bar: OhlcvBar, field: str) -> float:
    if field == "high":
        return bar.high if side == "buy" else bar.low
    if field == "low":
        return bar.low if side == "buy" else bar.high
    return bar.close


def _mfe_points(side: str, entry: float, high_price: float, low_price: float, point: float) -> float:
    return max(0.0, (high_price - entry) / point) if side == "buy" else max(0.0, (entry - low_price) / point)


def _mae_points(side: str, entry: float, high_price: float, low_price: float, point: float) -> float:
    return max(0.0, (entry - low_price) / point) if side == "buy" else max(0.0, (high_price - entry) / point)


def _side_adjusted_move(side: str, first_price: float, current_price: float) -> float:
    return current_price - first_price if side == "buy" else first_price - current_price


def _pnl(side: str, entry: float, exit_price: float, lot_size: float, contract_size: float) -> float:
    direction = 1.0 if side == "buy" else -1.0
    return (exit_price - entry) * direction * lot_size * contract_size


def _minutes_to_session_close(timestamp: datetime, close_hour: int) -> float:
    close_dt = timestamp.replace(hour=close_hour, minute=0, second=0, microsecond=0)
    return max(0.0, (close_dt - timestamp).total_seconds() / 60.0)


def _duration_seconds(event: AzirManagementEvent) -> float | None:
    exit_dt = _event_exit_dt(event)
    return None if exit_dt is None else (exit_dt - event.fill_timestamp).total_seconds()


def _duration_bucket(event: AzirManagementEvent) -> str:
    seconds = _duration_seconds(event)
    if seconds is None:
        return "missing"
    if seconds < 30:
        return "<30s"
    if seconds < 60:
        return "30-60s"
    if seconds < 120:
        return "60-120s"
    if seconds < 180:
        return "120-180s"
    if seconds < 300:
        return "180-300s"
    if seconds < 900:
        return "5-15m"
    return ">=15m"


def _float(value: Any) -> float:
    parsed = _to_float(value)
    return 0.0 if parsed is None else float(parsed)


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "sprint": report["sprint"],
        "protected_trade_events": report["dataset"]["protected_trade_events"],
        "unique_labeled_trades": report["dataset"]["unique_labeled_trades"],
        "snapshot_rows": report["dataset"]["snapshot_rows"],
        "tick_snapshot_rows": report["dataset"]["tick_snapshot_rows"],
        "m1_fallback_snapshot_rows": report["dataset"]["m1_fallback_snapshot_rows"],
        "best_label_candidate": report["decision"]["best_label_candidate"],
        "momentum_break_still_promising": report["decision"]["label_momentum_break_true_still_promising"],
        "ready_for_supervised_v1": report["decision"]["ready_for_supervised_trade_protection_model_v1"],
        "recommended_next_sprint": report["decision"]["recommended_next_sprint"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
