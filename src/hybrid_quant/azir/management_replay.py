"""M1 price replay for Azir management-action heuristics.

This sprint prices simple management alternatives for already-protected Azir
trades. It does not change Azir, does not train PPO, and deliberately separates
M1-priced cases from unpriced cases so weak proxies cannot masquerade as a
frozen benchmark.
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from hybrid_quant.env.azir_management_env import (
    ACTION_BASE_MANAGEMENT,
    ACTION_CLOSE_EARLY,
    ACTION_MOVE_TO_BREAK_EVEN,
    ACTION_TRAILING_AGGRESSIVE,
    ACTION_TRAILING_CONSERVATIVE,
    MANAGEMENT_ACTIONS,
    AzirManagementEvent,
    build_azir_management_replay_dataset,
)
from hybrid_quant.risk.azir_state import AzirRiskConfig

from .economic_audit import _parse_timestamp, _round, _to_float, _write_csv
from .replica import OhlcvBar, load_ohlcv_csv
from .train_ppo_skip_take import trade_metrics


DEFAULT_CONFIG = {
    "benchmark": "baseline_azir_protected_economic_v1",
    "risk_engine": "risk_engine_azir_v1",
    "instrument": {"point": 0.01, "contract_size": 100.0, "default_lot_size": 0.01},
    "replay": {
        "close_early_after_closed_m1_bars": 1,
        "sl_points": 500.0,
        "tp_points": 500.0,
        "break_even_activation_points": 90.0,
        "trailing_conservative_activation_points": 120.0,
        "trailing_conservative_step_points": 70.0,
        "trailing_aggressive_activation_points": 60.0,
        "trailing_aggressive_step_points": 35.0,
        "intrabar_policy": "conservative_stop_first",
    },
    "heuristics": [
        {"name": "always_base_management", "mode": "always", "action": "base_management"},
        {"name": "always_close_early", "mode": "always", "action": "close_early"},
        {"name": "move_to_be_after_mfe_threshold", "mode": "mfe_threshold", "action": "move_to_break_even", "threshold_points": 90.0},
        {"name": "conservative_trailing_after_mfe_threshold", "mode": "mfe_threshold", "action": "trailing_conservative", "threshold_points": 120.0},
        {"name": "aggressive_trailing_after_mfe_threshold", "mode": "mfe_threshold", "action": "trailing_aggressive", "threshold_points": 60.0},
        {"name": "sell_only_conservative_trailing_after_mfe_threshold", "mode": "side_mfe_threshold", "side": "sell", "action": "trailing_conservative", "threshold_points": 120.0},
        {"name": "buy_only_conservative_trailing_after_mfe_threshold", "mode": "side_mfe_threshold", "side": "buy", "action": "trailing_conservative", "threshold_points": 120.0},
    ],
}


@dataclass(frozen=True)
class ManagementReplayConfig:
    point: float = 0.01
    contract_size: float = 100.0
    default_lot_size: float = 0.01
    close_early_after_closed_m1_bars: int = 1
    sl_points: float = 500.0
    tp_points: float = 500.0
    break_even_activation_points: float = 90.0
    trailing_conservative_activation_points: float = 120.0
    trailing_conservative_step_points: float = 70.0
    trailing_aggressive_activation_points: float = 60.0
    trailing_aggressive_step_points: float = 35.0
    intrabar_policy: str = "conservative_stop_first"


@dataclass(frozen=True)
class HeuristicSpec:
    name: str
    mode: str
    action: str
    threshold_points: float = 0.0
    side: str = ""


@dataclass(frozen=True)
class ActionReplayResult:
    event_key: str
    setup_day: str
    fill_timestamp: str
    exit_timestamp: str
    side: str
    action: str
    action_id: int
    status: str
    pricing_confidence: str
    exit_reason: str
    exit_price: float | None
    net_pnl: float | None
    duration_minutes: float | None
    m1_bars_used: int
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "event_key": self.event_key,
            "setup_day": self.setup_day,
            "fill_timestamp": self.fill_timestamp,
            "exit_timestamp": self.exit_timestamp,
            "side": self.side,
            "action": self.action,
            "action_id": self.action_id,
            "status": self.status,
            "pricing_confidence": self.pricing_confidence,
            "exit_reason": self.exit_reason,
            "exit_price": "" if self.exit_price is None else _round(self.exit_price),
            "net_pnl": "" if self.net_pnl is None else _round(self.net_pnl),
            "duration_minutes": "" if self.duration_minutes is None else _round(self.duration_minutes),
            "m1_bars_used": self.m1_bars_used,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class M1ReplaySeries:
    bars: list[OhlcvBar]
    open_times: list[datetime]
    close_times: list[datetime]

    @classmethod
    def from_bars(cls, bars: list[OhlcvBar]) -> "M1ReplaySeries":
        return cls(
            bars=bars,
            open_times=[bar.open_time for bar in bars],
            close_times=[bar.open_time + timedelta(minutes=1) for bar in bars],
        )

    def bars_with_close_after(self, fill_dt: datetime) -> list[OhlcvBar]:
        index = bisect.bisect_right(self.close_times, fill_dt)
        return self.bars[index:]

    def full_bars_after_fill_before_exit(self, fill_dt: datetime, exit_dt: datetime) -> list[OhlcvBar]:
        start = bisect.bisect_right(self.open_times, fill_dt)
        end = bisect.bisect_right(self.close_times, exit_dt)
        if end <= start:
            return []
        return self.bars[start:end]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay Azir management actions with M1 OHLCV and evaluate heuristics.")
    parser.add_argument("--mt5-log-path", required=True)
    parser.add_argument("--protected-report-path", required=True)
    parser.add_argument("--m1-input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-path", default="")
    parser.add_argument("--symbol", default="XAUUSD-STD")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    replay_config, heuristics = load_config(Path(args.config_path) if args.config_path else None)
    report = run_management_price_replay(
        mt5_log_path=Path(args.mt5_log_path),
        protected_report_path=Path(args.protected_report_path),
        m1_input_path=Path(args.m1_input_path),
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
        replay_config=replay_config,
        heuristics=heuristics,
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def load_config(path: Path | None) -> tuple[ManagementReplayConfig, list[HeuristicSpec]]:
    payload = DEFAULT_CONFIG if path is None else _merge_dicts(DEFAULT_CONFIG, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    instrument = payload.get("instrument", {})
    replay = payload.get("replay", {})
    replay_config = ManagementReplayConfig(
        point=float(instrument.get("point", 0.01)),
        contract_size=float(instrument.get("contract_size", 100.0)),
        default_lot_size=float(instrument.get("default_lot_size", 0.01)),
        close_early_after_closed_m1_bars=int(replay.get("close_early_after_closed_m1_bars", 1)),
        sl_points=float(replay.get("sl_points", 500.0)),
        tp_points=float(replay.get("tp_points", 500.0)),
        break_even_activation_points=float(replay.get("break_even_activation_points", 90.0)),
        trailing_conservative_activation_points=float(replay.get("trailing_conservative_activation_points", 120.0)),
        trailing_conservative_step_points=float(replay.get("trailing_conservative_step_points", 70.0)),
        trailing_aggressive_activation_points=float(replay.get("trailing_aggressive_activation_points", 60.0)),
        trailing_aggressive_step_points=float(replay.get("trailing_aggressive_step_points", 35.0)),
        intrabar_policy=str(replay.get("intrabar_policy", "conservative_stop_first")),
    )
    heuristics = [
        HeuristicSpec(
            name=str(row["name"]),
            mode=str(row.get("mode", "always")),
            action=str(row.get("action", "base_management")),
            threshold_points=float(row.get("threshold_points", 0.0) or 0.0),
            side=str(row.get("side", "") or "").lower(),
        )
        for row in payload.get("heuristics", [])
    ]
    return replay_config, heuristics


def run_management_price_replay(
    *,
    mt5_log_path: Path,
    protected_report_path: Path,
    m1_input_path: Path,
    output_dir: Path,
    symbol: str = "XAUUSD-STD",
    replay_config: ManagementReplayConfig | None = None,
    heuristics: list[HeuristicSpec] | None = None,
) -> dict[str, Any]:
    replay_config = replay_config or ManagementReplayConfig()
    heuristics = heuristics or [HeuristicSpec("always_base_management", "always", "base_management")]
    events = build_azir_management_replay_dataset(
        mt5_log_path=mt5_log_path,
        protected_report_path=protected_report_path,
        symbol=symbol,
        risk_config=AzirRiskConfig(),
    )
    bars = load_ohlcv_csv(m1_input_path)
    action_results = price_all_management_actions(events, bars, replay_config)
    heuristic_rows, exit_distribution = evaluate_management_heuristics(events, action_results, heuristics)
    report = _build_report(
        mt5_log_path=mt5_log_path,
        protected_report_path=protected_report_path,
        m1_input_path=m1_input_path,
        events=events,
        bars=bars,
        action_results=action_results,
        heuristic_rows=heuristic_rows,
        exit_distribution=exit_distribution,
        replay_config=replay_config,
        heuristics=heuristics,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv([result.to_row() for result in action_results], output_dir / "management_price_replay_cases.csv")
    _write_csv(heuristic_rows, output_dir / "management_heuristics_comparison.csv")
    _write_csv(exit_distribution, output_dir / "management_exit_distribution.csv")
    (output_dir / "management_replay_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "management_replay_summary.md").write_text(_summary_markdown(report), encoding="utf-8")
    (output_dir / "management_limitations.md").write_text(_limitations_markdown(report), encoding="utf-8")
    return report


def price_all_management_actions(
    events: list[AzirManagementEvent],
    bars: list[OhlcvBar],
    config: ManagementReplayConfig,
) -> list[ActionReplayResult]:
    series = M1ReplaySeries.from_bars(bars)
    results: list[ActionReplayResult] = []
    for event in events:
        for action_id in MANAGEMENT_ACTIONS:
            results.append(_price_management_action_with_series(event, series, action_id, config))
    return results


def price_management_action(
    event: AzirManagementEvent,
    bars: list[OhlcvBar],
    action_id: int,
    config: ManagementReplayConfig,
) -> ActionReplayResult:
    return _price_management_action_with_series(event, M1ReplaySeries.from_bars(bars), action_id, config)


def _price_management_action_with_series(
    event: AzirManagementEvent,
    series: M1ReplaySeries,
    action_id: int,
    config: ManagementReplayConfig,
) -> ActionReplayResult:
    action_label = MANAGEMENT_ACTIONS[action_id]
    fill_dt = event.fill_timestamp
    exit_dt = _parse_timestamp(event.trade.get("exit_timestamp"))
    entry = _to_float(event.trade.get("fill_price"))
    if action_id == ACTION_BASE_MANAGEMENT:
        return ActionReplayResult(
            event_key=_event_key(event),
            setup_day=event.setup_day,
            fill_timestamp=fill_dt.isoformat(sep=" "),
            exit_timestamp=exit_dt.isoformat(sep=" ") if exit_dt.year > 1 else "",
            side=event.side,
            action=action_label,
            action_id=action_id,
            status="priced_observed_protected",
            pricing_confidence="observed_protected_benchmark",
            exit_reason=str(event.trade.get("exit_reason", "")),
            exit_price=None,
            net_pnl=event.protected_net_pnl,
            duration_minutes=_minutes_between(fill_dt, exit_dt) if exit_dt.year > 1 else None,
            m1_bars_used=0,
            notes="Base management uses the frozen protected economic benchmark, not M1 counterfactual replay.",
        )
    if entry is None or event.side not in {"buy", "sell"} or exit_dt.year <= 1:
        return _unpriced_result(event, action_id, "missing_entry_side_or_exit")

    if action_id == ACTION_CLOSE_EARLY:
        return _price_close_early(event, series, config, entry, fill_dt, exit_dt)
    if action_id == ACTION_MOVE_TO_BREAK_EVEN:
        return _price_stop_policy(
            event,
            series,
            config,
            entry,
            fill_dt,
            exit_dt,
            action_id,
            break_even_activation_points=config.break_even_activation_points,
            trailing_activation_points=None,
            trailing_step_points=None,
        )
    if action_id == ACTION_TRAILING_CONSERVATIVE:
        return _price_stop_policy(
            event,
            series,
            config,
            entry,
            fill_dt,
            exit_dt,
            action_id,
            break_even_activation_points=None,
            trailing_activation_points=config.trailing_conservative_activation_points,
            trailing_step_points=config.trailing_conservative_step_points,
        )
    if action_id == ACTION_TRAILING_AGGRESSIVE:
        return _price_stop_policy(
            event,
            series,
            config,
            entry,
            fill_dt,
            exit_dt,
            action_id,
            break_even_activation_points=None,
            trailing_activation_points=config.trailing_aggressive_activation_points,
            trailing_step_points=config.trailing_aggressive_step_points,
        )
    return _unpriced_result(event, action_id, "unknown_action")


def evaluate_management_heuristics(
    events: list[AzirManagementEvent],
    action_results: list[ActionReplayResult],
    heuristics: list[HeuristicSpec],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results_by_key_action = {(row.event_key, row.action): row for row in action_results}
    rows: list[dict[str, Any]] = []
    exit_distribution: list[dict[str, Any]] = []
    for heuristic in heuristics:
        chosen: list[ActionReplayResult] = []
        base_same_coverage: list[ActionReplayResult] = []
        unpriced = 0
        for event in events:
            action = _choose_heuristic_action(event, heuristic)
            result = results_by_key_action.get((_event_key(event), action))
            if result is None or result.net_pnl is None:
                unpriced += 1
                continue
            chosen.append(result)
            base_same_coverage.append(results_by_key_action[(_event_key(event), MANAGEMENT_ACTIONS[ACTION_BASE_MANAGEMENT])])
        metrics = _management_metrics(chosen)
        base_metrics = _management_metrics(base_same_coverage)
        rows.append(
            {
                "heuristic": heuristic.name,
                "mode": heuristic.mode,
                "action": heuristic.action,
                "side_filter": heuristic.side,
                "threshold_points": heuristic.threshold_points,
                "events": len(events),
                "priced_trades": len(chosen),
                "unpriced_trades": unpriced,
                "coverage_pct": _round(len(chosen) / len(events) * 100.0) if events else 0.0,
                **metrics,
                "base_same_coverage_net_pnl": base_metrics["net_pnl"],
                "base_same_coverage_profit_factor": base_metrics["profit_factor"],
                "base_same_coverage_expectancy": base_metrics["expectancy"],
                "base_same_coverage_max_drawdown": base_metrics["max_drawdown"],
                "delta_net_pnl_vs_base_same_coverage": _delta(base_metrics["net_pnl"], metrics["net_pnl"]),
                "delta_expectancy_vs_base_same_coverage": _delta(base_metrics["expectancy"], metrics["expectancy"]),
                "benchmark_quality": _benchmark_quality(len(chosen), len(events), unpriced),
            }
        )
        exit_counts = Counter(result.exit_reason for result in chosen)
        for exit_reason, count in sorted(exit_counts.items()):
            exit_distribution.append(
                {
                    "heuristic": heuristic.name,
                    "exit_reason": exit_reason,
                    "trades": count,
                    "pct_of_priced_trades": _round(count / len(chosen) * 100.0) if chosen else 0.0,
                }
            )
    rows.sort(
        key=lambda row: (
            row["benchmark_quality"] == "usable_m1_subset",
            _float(row["delta_net_pnl_vs_base_same_coverage"]),
            _float(row["profit_factor"]),
            -_float(row["max_drawdown"]),
        ),
        reverse=True,
    )
    return rows, exit_distribution


def _price_close_early(
    event: AzirManagementEvent,
    series: M1ReplaySeries,
    config: ManagementReplayConfig,
    entry: float,
    fill_dt: datetime,
    exit_dt: datetime,
) -> ActionReplayResult:
    bars_after = series.bars_with_close_after(fill_dt)
    index = max(0, config.close_early_after_closed_m1_bars - 1)
    if len(bars_after) <= index:
        return _unpriced_result(event, ACTION_CLOSE_EARLY, "unpriced_no_m1_after_fill")
    selected = bars_after[index]
    close_time = selected.open_time + timedelta(minutes=1)
    if close_time > exit_dt:
        return _unpriced_result(event, ACTION_CLOSE_EARLY, "unpriced_intervention_after_observed_exit")
    pnl = _pnl(event.side, entry, selected.close, _lot_size(event, config), config.contract_size)
    return ActionReplayResult(
        event_key=_event_key(event),
        setup_day=event.setup_day,
        fill_timestamp=fill_dt.isoformat(sep=" "),
        exit_timestamp=close_time.isoformat(sep=" "),
        side=event.side,
        action=MANAGEMENT_ACTIONS[ACTION_CLOSE_EARLY],
        action_id=ACTION_CLOSE_EARLY,
        status="priced_with_m1_proxy",
        pricing_confidence="m1_close_proxy",
        exit_reason="close_early_m1_checkpoint",
        exit_price=selected.close,
        net_pnl=pnl,
        duration_minutes=_minutes_between(fill_dt, close_time),
        m1_bars_used=index + 1,
        notes="Exit at the close of the first configured closed M1 bar after fill.",
    )


def _price_stop_policy(
    event: AzirManagementEvent,
    series: M1ReplaySeries,
    config: ManagementReplayConfig,
    entry: float,
    fill_dt: datetime,
    exit_dt: datetime,
    action_id: int,
    *,
    break_even_activation_points: float | None,
    trailing_activation_points: float | None,
    trailing_step_points: float | None,
) -> ActionReplayResult:
    path = series.full_bars_after_fill_before_exit(fill_dt, exit_dt)
    if not path:
        return _unpriced_result(event, action_id, "unpriced_no_full_m1_path_before_observed_exit")
    side = event.side
    point = config.point
    stop = entry - config.sl_points * point if side == "buy" else entry + config.sl_points * point
    target = entry + config.tp_points * point if side == "buy" else entry - config.tp_points * point
    activated = False
    best_favorable = entry

    for bar in path:
        stop_hit = bar.low <= stop if side == "buy" else bar.high >= stop
        target_hit = bar.high >= target if side == "buy" else bar.low <= target
        if stop_hit and target_hit:
            return _action_exit(event, action_id, fill_dt, bar.open_time + timedelta(minutes=1), stop, "m1_conservative_stop_before_target", len(path), config)
        if stop_hit:
            return _action_exit(event, action_id, fill_dt, bar.open_time + timedelta(minutes=1), stop, _stop_reason(action_id), len(path), config)
        if target_hit:
            return _action_exit(event, action_id, fill_dt, bar.open_time + timedelta(minutes=1), target, "target_hit_m1", len(path), config)

        favorable = bar.high if side == "buy" else bar.low
        favorable_points = ((favorable - entry) / point) if side == "buy" else ((entry - favorable) / point)
        if break_even_activation_points is not None and favorable_points >= break_even_activation_points:
            stop = max(stop, entry) if side == "buy" else min(stop, entry)
            activated = True
        if trailing_activation_points is not None and trailing_step_points is not None and favorable_points >= trailing_activation_points:
            best_favorable = max(best_favorable, favorable) if side == "buy" else min(best_favorable, favorable)
            trailing_stop = best_favorable - trailing_step_points * point if side == "buy" else best_favorable + trailing_step_points * point
            stop = max(stop, trailing_stop) if side == "buy" else min(stop, trailing_stop)
            activated = True

    horizon_bar = _latest_closed_bar_before_or_at(path, exit_dt)
    if horizon_bar is None:
        return _unpriced_result(event, action_id, "unpriced_no_m1_close_before_horizon")
    close_time = horizon_bar.open_time + timedelta(minutes=1)
    if not activated:
        return ActionReplayResult(
            event_key=_event_key(event),
            setup_day=event.setup_day,
            fill_timestamp=fill_dt.isoformat(sep=" "),
            exit_timestamp=exit_dt.isoformat(sep=" "),
            side=event.side,
            action=MANAGEMENT_ACTIONS[action_id],
            action_id=action_id,
            status="priced_observed_protected_no_management_activation",
            pricing_confidence="observed_protected_benchmark",
            exit_reason="base_management_no_m1_activation",
            exit_price=None,
            net_pnl=event.protected_net_pnl,
            duration_minutes=_minutes_between(fill_dt, exit_dt),
            m1_bars_used=len(path),
            notes="The M1 path did not activate the counterfactual management rule; base protected outcome is retained.",
        )
    reason = "horizon_close_after_management_activation"
    return _action_exit(event, action_id, fill_dt, close_time, horizon_bar.close, reason, len(path), config)


def _action_exit(
    event: AzirManagementEvent,
    action_id: int,
    fill_dt: datetime,
    exit_dt: datetime,
    exit_price: float,
    exit_reason: str,
    m1_bars_used: int,
    config: ManagementReplayConfig,
) -> ActionReplayResult:
    entry = _to_float(event.trade.get("fill_price")) or 0.0
    pnl = _pnl(event.side, entry, exit_price, _lot_size(event, config), config.contract_size)
    return ActionReplayResult(
        event_key=_event_key(event),
        setup_day=event.setup_day,
        fill_timestamp=fill_dt.isoformat(sep=" "),
        exit_timestamp=exit_dt.isoformat(sep=" "),
        side=event.side,
        action=MANAGEMENT_ACTIONS[action_id],
        action_id=action_id,
        status="priced_with_m1_proxy",
        pricing_confidence="m1_ohlc_conservative_proxy",
        exit_reason=exit_reason,
        exit_price=exit_price,
        net_pnl=pnl,
        duration_minutes=_minutes_between(fill_dt, exit_dt),
        m1_bars_used=m1_bars_used,
        notes="M1 OHLC replay; ambiguous stop/target bars resolve conservatively to stop first.",
    )


def _build_report(
    *,
    mt5_log_path: Path,
    protected_report_path: Path,
    m1_input_path: Path,
    events: list[AzirManagementEvent],
    bars: list[OhlcvBar],
    action_results: list[ActionReplayResult],
    heuristic_rows: list[dict[str, Any]],
    exit_distribution: list[dict[str, Any]],
    replay_config: ManagementReplayConfig,
    heuristics: list[HeuristicSpec],
) -> dict[str, Any]:
    priced_by_action = {
        label: len([row for row in action_results if row.action == label and row.net_pnl is not None])
        for label in MANAGEMENT_ACTIONS.values()
    }
    statuses = Counter(row.status for row in action_results if row.action != "base_management")
    base = next((row for row in heuristic_rows if row["heuristic"] == "always_base_management"), {})
    candidates = [
        row
        for row in heuristic_rows
        if row["heuristic"] != "always_base_management"
        and row["benchmark_quality"] in {"full_coverage", "usable_m1_subset"}
        and _float(row["delta_net_pnl_vs_base_same_coverage"]) > 0.0
    ]
    best_candidate = max(
        candidates,
        key=lambda row: (_float(row["delta_net_pnl_vs_base_same_coverage"]), _float(row["profit_factor"]), -_float(row["max_drawdown"])),
        default=None,
    )
    strong_candidate = bool(
        best_candidate
        and _float(best_candidate["coverage_pct"]) >= 60.0
        and _float(best_candidate["profit_factor"]) >= 1.10
        and _float(best_candidate["delta_net_pnl_vs_base_same_coverage"]) >= 10.0
    )
    return {
        "sprint": "management_price_replay_and_heuristic_backtest_v1",
        "mt5_log_path": str(mt5_log_path),
        "protected_report_path": str(protected_report_path),
        "m1_input_path": str(m1_input_path),
        "m1_coverage": {
            "bars": len(bars),
            "start": bars[0].open_time.isoformat(sep=" ") if bars else None,
            "end": bars[-1].open_time.isoformat(sep=" ") if bars else None,
        },
        "events": len(events),
        "replay_config": asdict(replay_config),
        "heuristics": [asdict(heuristic) for heuristic in heuristics],
        "pricing_methodology": {
            "base_management": "Uses observed/revalued protected benchmark net_pnl.",
            "close_early": "Uses the close of the first configured closed M1 bar after fill if it occurs before observed/protected horizon.",
            "move_to_break_even": "Uses M1 OHLC after fill; BE stop is active only after a fully closed M1 bar reaches activation.",
            "trailing": "Uses M1 OHLC after fill; stop ratchets only after a fully closed M1 bar reaches activation.",
            "intrabar_ambiguity": "If stop and target are both inside one M1 bar, the stop is chosen first.",
        },
        "pricing_coverage_by_action": priced_by_action,
        "alternative_action_status_counts": dict(statuses),
        "heuristic_comparison": heuristic_rows,
        "exit_distribution": exit_distribution,
        "decision": {
            "management_valuation_serious_enough_for_heuristic_screening": bool(priced_by_action.get("close_early", 0) > 0),
            "management_benchmark_ready_for_ppo_training": strong_candidate,
            "best_candidate": best_candidate["heuristic"] if best_candidate else "",
            "best_candidate_delta_net_pnl_vs_base_same_coverage": best_candidate["delta_net_pnl_vs_base_same_coverage"] if best_candidate else "",
            "base_management_full_net_pnl": base.get("net_pnl", ""),
            "recommended_next_sprint": "train_management_ppo_for_azir_v1" if strong_candidate else "management_price_replay_v2_with_tick_or_broker_execution",
            "reason": (
                "A same-coverage management heuristic is strong enough on coverage, PF, and net uplift to justify management PPO."
                if strong_candidate
                else "M1 replay is useful for screening, but the best uplift is too small or too thinly covered to freeze a management benchmark for PPO."
                if best_candidate
                else "No M1-priced heuristic improves base management convincingly on fair same-coverage comparison."
            ),
        },
        "limitations": [
            "M1 coverage starts at the first available M1 bar; earlier protected trades remain unpriced for alternative management actions.",
            "M1 OHLC cannot recover tick order inside a minute, so stop/target ambiguity is resolved conservatively.",
            "Close-early decisions are priced at closed M1 bars, not broker-confirmed immediate market fills.",
            "Trailing alternatives are counterfactual approximations and do not modify the real MQL5 trailing implementation.",
            "This sprint is suitable for heuristic screening, not a frozen management benchmark unless coverage and tick fidelity are sufficient.",
        ],
    }


def _summary_markdown(report: dict[str, Any]) -> str:
    top_lines = "\n".join(
        "- "
        f"`{row['heuristic']}`: net={row['net_pnl']}, PF={row['profit_factor']}, "
        f"exp={row['expectancy']}, DD={row['max_drawdown']}, priced={row['priced_trades']}, "
        f"delta_vs_base_same_coverage={row['delta_net_pnl_vs_base_same_coverage']}, quality={row['benchmark_quality']}."
        for row in report["heuristic_comparison"][:5]
    )
    coverage = "\n".join(f"- `{action}`: {count} priced cases." for action, count in report["pricing_coverage_by_action"].items())
    decision = report["decision"]
    return (
        "# Azir Management Price Replay And Heuristic Backtest V1\n\n"
        "## Executive Summary\n\n"
        "- This sprint prices management alternatives with M1 OHLC where available; it does not train PPO.\n"
        f"- Protected management events: {report['events']}.\n"
        f"- M1 coverage: {report['m1_coverage']['start']} -> {report['m1_coverage']['end']} ({report['m1_coverage']['bars']} bars).\n"
        f"- Base full net PnL: {decision['base_management_full_net_pnl']}.\n"
        f"- Best same-coverage candidate: `{decision['best_candidate'] or 'none'}`.\n"
        f"- Ready for management PPO: {decision['management_benchmark_ready_for_ppo_training']}.\n\n"
        "## Pricing Coverage By Action\n\n"
        f"{coverage}\n\n"
        "## Top Heuristics\n\n"
        f"{top_lines}\n\n"
        "## Decision\n\n"
        f"- {decision['reason']}\n"
        f"- Recommended next sprint: `{decision['recommended_next_sprint']}`.\n"
    )


def _limitations_markdown(report: dict[str, Any]) -> str:
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    methodology = "\n".join(f"- `{key}`: {value}" for key, value in report["pricing_methodology"].items())
    return (
        "# Management Replay Limitations\n\n"
        "## Methodology\n\n"
        f"{methodology}\n\n"
        "## Live Limitations\n\n"
        f"{limitations}\n"
    )


def _choose_heuristic_action(event: AzirManagementEvent, heuristic: HeuristicSpec) -> str:
    if heuristic.mode == "always":
        return heuristic.action
    if heuristic.mode == "mfe_threshold":
        return heuristic.action
    if heuristic.mode == "side_mfe_threshold":
        if event.side == heuristic.side:
            return heuristic.action
        return "base_management"
    return "base_management"


def _management_metrics(results: list[ActionReplayResult]) -> dict[str, Any]:
    pnl_values = [float(row.net_pnl or 0.0) for row in results]
    metrics = trade_metrics(pnl_values)
    durations = [row.duration_minutes for row in results if row.duration_minutes is not None]
    return {
        "net_pnl": metrics["net_pnl"],
        "profit_factor": metrics["profit_factor"],
        "expectancy": metrics["expectancy"],
        "win_rate": metrics["win_rate"],
        "average_win": metrics["average_win"],
        "average_loss": metrics["average_loss"],
        "payoff": metrics["payoff"],
        "max_drawdown": metrics["max_drawdown"],
        "max_consecutive_losses": metrics["max_consecutive_losses"],
        "average_duration_minutes": _round(mean(durations)) if durations else 0.0,
    }


def _unpriced_result(event: AzirManagementEvent, action_id: int, reason: str) -> ActionReplayResult:
    return ActionReplayResult(
        event_key=_event_key(event),
        setup_day=event.setup_day,
        fill_timestamp=event.fill_timestamp.isoformat(sep=" "),
        exit_timestamp=str(event.trade.get("exit_timestamp", "")),
        side=event.side,
        action=MANAGEMENT_ACTIONS[action_id],
        action_id=action_id,
        status=reason,
        pricing_confidence="unpriced",
        exit_reason="unpriced",
        exit_price=None,
        net_pnl=None,
        duration_minutes=None,
        m1_bars_used=0,
        notes=reason,
    )


def _bars_with_close_after(bars: list[OhlcvBar], fill_dt: datetime) -> list[OhlcvBar]:
    return [bar for bar in bars if bar.open_time + timedelta(minutes=1) > fill_dt]


def _full_m1_bars_after_fill_before_exit(bars: list[OhlcvBar], fill_dt: datetime, exit_dt: datetime) -> list[OhlcvBar]:
    return [
        bar
        for bar in bars
        if bar.open_time > fill_dt and bar.open_time + timedelta(minutes=1) <= exit_dt
    ]


def _latest_closed_bar_before_or_at(bars: list[OhlcvBar], exit_dt: datetime) -> OhlcvBar | None:
    eligible = [bar for bar in bars if bar.open_time + timedelta(minutes=1) <= exit_dt]
    return eligible[-1] if eligible else None


def _stop_reason(action_id: int) -> str:
    if action_id == ACTION_MOVE_TO_BREAK_EVEN:
        return "break_even_or_original_stop_hit_m1"
    if action_id in {ACTION_TRAILING_CONSERVATIVE, ACTION_TRAILING_AGGRESSIVE}:
        return "trailing_stop_hit_m1"
    return "stop_hit_m1"


def _pnl(side: str, entry: float, exit_price: float, lot_size: float, contract_size: float) -> float:
    direction = 1.0 if side == "buy" else -1.0
    return (exit_price - entry) * direction * lot_size * contract_size


def _lot_size(event: AzirManagementEvent, config: ManagementReplayConfig) -> float:
    parsed = _to_float(event.setup.get("lot_size"))
    return parsed if parsed is not None and parsed > 0.0 else config.default_lot_size


def _event_key(event: AzirManagementEvent) -> str:
    return f"{event.setup_day}|{event.fill_timestamp.isoformat(sep=' ')}|{event.side}"


def _minutes_between(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 60.0


def _benchmark_quality(priced: int, total: int, unpriced: int) -> str:
    if total == 0 or priced == 0:
        return "not_usable"
    coverage = priced / total
    if unpriced == 0:
        return "full_coverage"
    if coverage >= 0.30:
        return "usable_m1_subset"
    return "thin_m1_subset"


def _float(value: Any) -> float:
    parsed = _to_float(value)
    return 0.0 if parsed is None else float(parsed)


def _delta(before: Any, after: Any) -> float | str:
    before_float = _to_float(before)
    after_float = _to_float(after)
    if before_float is None or after_float is None:
        return ""
    return _round(after_float - before_float)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "sprint": report["sprint"],
        "events": report["events"],
        "m1_coverage": report["m1_coverage"],
        "pricing_coverage_by_action": report["pricing_coverage_by_action"],
        "best_candidate": report["decision"]["best_candidate"],
        "ready_for_management_ppo": report["decision"]["management_benchmark_ready_for_ppo_training"],
        "recommended_next_sprint": report["decision"]["recommended_next_sprint"],
        "output_decision_reason": report["decision"]["reason"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
