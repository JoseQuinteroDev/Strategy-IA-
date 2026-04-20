"""Post-entry trade-protection research for Azir.

This sprint does not change Azir, train PPO, or promote a management policy.
It builds a causal post-entry dataset and screens small protection heuristics
against the frozen protected Azir baseline.
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

from hybrid_quant.env.azir_management_env import (
    AzirManagementEvent,
    build_azir_management_replay_dataset,
)
from hybrid_quant.risk.azir_state import AzirRiskConfig

from .economic_audit import _parse_timestamp, _round, _to_float, _write_csv
from .replica import OhlcvBar, load_ohlcv_csv


DEFAULT_OUTPUT_DIR = Path("artifacts/azir-trade-protection-research-v1")
DEFAULT_MT5_LOG = Path(r"C:\Users\joseq\Documents\Playground\todos_los_ticks.csv")
DEFAULT_M1_PATH = Path(r"C:\Users\joseq\Documents\xauusd_m1.csv")
DEFAULT_M5_PATH = Path(r"C:\Users\joseq\Documents\xauusd_m5.csv")
DEFAULT_TICK_PATH = Path(r"C:\Users\joseq\Documents\tick_level.csv")
DEFAULT_PROTECTED_REPORT = Path(
    "artifacts/azir-protected-economic-v1-freeze/forced_close_revaluation_report.json"
)


@dataclass(frozen=True)
class TradeProtectionConfig:
    point: float = 0.01
    contract_size: float = 100.0
    default_lot_size: float = 0.10
    default_sl_points: float = 500.0
    default_tp_points: float = 500.0
    snapshot_minutes: tuple[int, ...] = (5, 15, 30, 60)
    session_close_hour: int = 22
    break_even_points: tuple[float, ...] = (60.0, 90.0)
    break_even_atr_multiples: tuple[float, ...] = (0.50,)
    no_acceleration_minutes: int = 15
    no_acceleration_min_mfe_points: float = 30.0
    momentum_break_minutes: int = 10
    momentum_break_bars: int = 3
    mfe_reversion_min_points: float = 80.0
    mfe_reversion_fraction: float = 0.50
    conservative_trailing_activation_points: float = 120.0
    conservative_trailing_step_points: float = 70.0
    aggressive_trailing_activation_points: float = 60.0
    aggressive_trailing_step_points: float = 35.0


@dataclass(frozen=True)
class PriceSeries:
    label: str
    timeframe_minutes: int
    bars: list[OhlcvBar]
    open_times: list[datetime]
    close_times: list[datetime]

    @classmethod
    def from_bars(cls, label: str, timeframe_minutes: int, bars: list[OhlcvBar]) -> "PriceSeries":
        ordered = sorted(bars, key=lambda bar: bar.open_time)
        delta = timedelta(minutes=timeframe_minutes)
        return cls(
            label=label,
            timeframe_minutes=timeframe_minutes,
            bars=ordered,
            open_times=[bar.open_time for bar in ordered],
            close_times=[bar.open_time + delta for bar in ordered],
        )

    def bars_after_fill_before_horizon(self, fill_dt: datetime, horizon_dt: datetime) -> list[OhlcvBar]:
        start = bisect.bisect_right(self.open_times, fill_dt)
        end = bisect.bisect_right(self.close_times, horizon_dt)
        return [] if end <= start else self.bars[start:end]

    def bars_until_minutes(self, fill_dt: datetime, minutes_after_fill: int) -> list[OhlcvBar]:
        return self.bars_after_fill_before_horizon(
            fill_dt,
            fill_dt + timedelta(minutes=minutes_after_fill),
        )


@dataclass(frozen=True)
class ProtectionResult:
    heuristic: str
    family: str
    event_key: str
    setup_day: str
    fill_timestamp: str
    exit_timestamp: str
    side: str
    pricing_source: str
    exit_reason: str
    net_pnl: float | None
    duration_minutes: float | None
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "heuristic": self.heuristic,
            "family": self.family,
            "event_key": self.event_key,
            "setup_day": self.setup_day,
            "fill_timestamp": self.fill_timestamp,
            "exit_timestamp": self.exit_timestamp,
            "side": self.side,
            "pricing_source": self.pricing_source,
            "exit_reason": self.exit_reason,
            "net_pnl": "" if self.net_pnl is None else _round(self.net_pnl),
            "duration_minutes": "" if self.duration_minutes is None else _round(self.duration_minutes),
            "notes": self.notes,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Azir post-entry trade protection research.")
    parser.add_argument("--mt5-log-path", default=str(DEFAULT_MT5_LOG))
    parser.add_argument("--protected-report-path", default=str(DEFAULT_PROTECTED_REPORT))
    parser.add_argument("--m1-input-path", default=str(DEFAULT_M1_PATH))
    parser.add_argument("--m5-input-path", default=str(DEFAULT_M5_PATH))
    parser.add_argument("--tick-input-path", default=str(DEFAULT_TICK_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbol", default="XAUUSD-STD")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_trade_protection_research(
        mt5_log_path=Path(args.mt5_log_path),
        protected_report_path=Path(args.protected_report_path),
        m1_input_path=Path(args.m1_input_path),
        m5_input_path=Path(args.m5_input_path),
        tick_input_path=Path(args.tick_input_path),
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_trade_protection_research(
    *,
    mt5_log_path: Path,
    protected_report_path: Path,
    m1_input_path: Path,
    m5_input_path: Path,
    tick_input_path: Path,
    output_dir: Path,
    symbol: str = "XAUUSD-STD",
    config: TradeProtectionConfig | None = None,
) -> dict[str, Any]:
    config = config or TradeProtectionConfig()
    events = build_azir_management_replay_dataset(
        mt5_log_path=mt5_log_path,
        protected_report_path=protected_report_path,
        symbol=symbol,
        risk_config=AzirRiskConfig(),
    )
    m1_bars = load_ohlcv_csv(m1_input_path) if m1_input_path.exists() else []
    m5_bars = load_ohlcv_csv(m5_input_path) if m5_input_path.exists() else []
    series_by_source = [
        PriceSeries.from_bars("m1", 1, m1_bars),
        PriceSeries.from_bars("m5_fallback", 5, m5_bars),
    ]

    dataset_rows = build_post_entry_dataset(events, series_by_source, config)
    result_rows = price_protection_heuristics(events, series_by_source, config)
    comparison_rows, exit_rows, risk_rows = compare_protection_results(events, result_rows)
    family_rows = _family_summaries(comparison_rows)
    report = _build_report(
        mt5_log_path=mt5_log_path,
        protected_report_path=protected_report_path,
        m1_input_path=m1_input_path,
        m5_input_path=m5_input_path,
        tick_input_path=tick_input_path,
        events=events,
        m1_bars=m1_bars,
        m5_bars=m5_bars,
        dataset_rows=dataset_rows,
        comparison_rows=comparison_rows,
        family_rows=family_rows,
        exit_distribution_rows=exit_rows,
        risk_profile_rows=risk_rows,
        config=config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(dataset_rows, output_dir / "post_entry_dataset.csv")
    _write_csv(dataset_rows[:100], output_dir / "post_entry_dataset_sample.csv")
    _write_csv(_filter_family(comparison_rows, "break_even"), output_dir / "break_even_heuristics_comparison.csv")
    _write_csv(_filter_family(comparison_rows, "close_early"), output_dir / "close_early_heuristics_comparison.csv")
    _write_csv(_filter_family(comparison_rows, "trailing"), output_dir / "trailing_protection_comparison.csv")
    _write_csv(risk_rows, output_dir / "protection_vs_baseline_risk_profile.csv")
    _write_csv(exit_rows, output_dir / "protection_exit_distribution.csv")
    _write_csv([row.to_row() for row in result_rows], output_dir / "protection_priced_trade_cases.csv")
    (output_dir / "post_entry_dataset_schema.md").write_text(_dataset_schema_markdown(), encoding="utf-8")
    (output_dir / "trade_protection_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "trade_protection_summary.md").write_text(_summary_markdown(report), encoding="utf-8")
    (output_dir / "protection_candidate_assessment.md").write_text(_assessment_markdown(report), encoding="utf-8")
    return report


def build_post_entry_dataset(
    events: list[AzirManagementEvent],
    series_by_source: list[PriceSeries],
    config: TradeProtectionConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        exit_dt = _event_exit_dt(event)
        series = _best_series_for_event(event, exit_dt, series_by_source) if exit_dt else None
        if series is None or exit_dt is None:
            continue
        for minutes_after_fill in config.snapshot_minutes:
            if event.fill_timestamp + timedelta(minutes=minutes_after_fill) > exit_dt:
                continue
            path = series.bars_until_minutes(event.fill_timestamp, minutes_after_fill)
            if path:
                rows.append(_post_entry_row(event, path, series, minutes_after_fill, config))
    return rows


def price_protection_heuristics(
    events: list[AzirManagementEvent],
    series_by_source: list[PriceSeries],
    config: TradeProtectionConfig,
) -> list[ProtectionResult]:
    rows: list[ProtectionResult] = []
    specs = _heuristic_specs(config)
    for event in events:
        rows.append(_base_result(event))
        exit_dt = _event_exit_dt(event)
        series = _best_series_for_event(event, exit_dt, series_by_source) if exit_dt else None
        path = series.bars_after_fill_before_horizon(event.fill_timestamp, exit_dt) if series and exit_dt else []
        for spec in specs:
            if spec["name"] == "always_base_management":
                continue
            if not path or not exit_dt:
                rows.append(_unpriced_result(event, spec, "no causal M1/M5 path for post-entry intervention"))
                continue
            rows.append(_price_spec(event, path, series, spec, config))
    return rows


def compare_protection_results(
    events: list[AzirManagementEvent],
    results: list[ProtectionResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base_by_key = {row.event_key: row for row in results if row.heuristic == "always_base_management"}
    grouped: dict[str, list[ProtectionResult]] = {}
    families: dict[str, str] = {}
    for row in results:
        grouped.setdefault(row.heuristic, []).append(row)
        families.setdefault(row.heuristic, row.family)

    comparison_rows: list[dict[str, Any]] = []
    exit_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    for heuristic, rows in sorted(grouped.items()):
        priced = [row for row in rows if row.net_pnl is not None]
        base_same = [base_by_key[row.event_key] for row in priced if row.event_key in base_by_key]
        metrics = _metrics([float(row.net_pnl or 0.0) for row in priced])
        base_metrics = _metrics([float(row.net_pnl or 0.0) for row in base_same])
        source_counts = Counter(row.pricing_source for row in priced)
        comparison = {
            "heuristic": heuristic,
            "family": families[heuristic],
            "total_events": len(events),
            "priced_trades": len(priced),
            "unpriced_trades": len(rows) - len(priced),
            "coverage_pct": _round(len(priced) / len(events) * 100.0) if events else 0.0,
            "m1_priced_trades": source_counts.get("m1", 0),
            "m5_fallback_trades": source_counts.get("m5_fallback", 0),
            "observed_baseline_trades": source_counts.get("observed_protected_benchmark", 0),
            **metrics,
            "base_same_coverage_net_pnl": base_metrics["net_pnl"],
            "base_same_coverage_profit_factor": base_metrics["profit_factor"],
            "base_same_coverage_expectancy": base_metrics["expectancy"],
            "base_same_coverage_win_rate": base_metrics["win_rate"],
            "base_same_coverage_average_loss": base_metrics["average_loss"],
            "base_same_coverage_max_drawdown": base_metrics["max_drawdown"],
            "delta_net_pnl_vs_base_same_coverage": _delta(base_metrics["net_pnl"], metrics["net_pnl"]),
            "delta_expectancy_vs_base_same_coverage": _delta(base_metrics["expectancy"], metrics["expectancy"]),
            "delta_average_loss_vs_base_same_coverage": _delta(base_metrics["average_loss"], metrics["average_loss"]),
            "delta_win_rate_vs_base_same_coverage": _delta(base_metrics["win_rate"], metrics["win_rate"]),
            "delta_max_drawdown_vs_base_same_coverage": _delta(base_metrics["max_drawdown"], metrics["max_drawdown"]),
        }
        comparison["assessment"] = _heuristic_assessment(metrics, base_metrics, len(priced), len(events), heuristic)
        comparison_rows.append(comparison)
        risk_rows.append(_risk_profile_row(heuristic, families[heuristic], metrics, base_metrics, len(priced)))
        for exit_reason, count in sorted(Counter(row.exit_reason for row in priced).items()):
            exit_rows.append(
                {
                    "heuristic": heuristic,
                    "family": families[heuristic],
                    "exit_reason": exit_reason,
                    "trades": count,
                    "pct": _round(count / len(priced) * 100.0) if priced else 0.0,
                }
            )
    comparison_rows.sort(
        key=lambda row: (
            row["heuristic"] == "always_base_management",
            _float(row["delta_net_pnl_vs_base_same_coverage"]),
            _float(row["profit_factor"]),
            -abs(_float(row["max_drawdown"])),
        ),
        reverse=True,
    )
    return comparison_rows, exit_rows, risk_rows


def _price_spec(
    event: AzirManagementEvent,
    path: list[OhlcvBar],
    series: PriceSeries,
    spec: dict[str, Any],
    config: TradeProtectionConfig,
) -> ProtectionResult:
    family = str(spec["family"])
    if family == "break_even":
        return _price_break_even(event, path, series, spec, _threshold_points(event, spec, config), config)
    if family == "close_early":
        return _price_close_early_rule(event, path, series, spec, config)
    if family == "trailing":
        return _price_trailing_rule(event, path, series, spec, config)
    return _unpriced_result(event, spec, "unknown heuristic family")


def _post_entry_row(
    event: AzirManagementEvent,
    path: list[OhlcvBar],
    series: PriceSeries,
    minutes_after_fill: int,
    config: TradeProtectionConfig,
) -> dict[str, Any]:
    entry = _entry(event) or 0.0
    mfe = _path_mfe_points(event.side, entry, path, config.point)
    mae = _path_mae_points(event.side, entry, path, config.point)
    setup = event.setup
    atr_points = _float(setup.get("atr_points"))
    sl_points = _float(setup.get("sl_points")) or config.default_sl_points
    tp_points = _float(setup.get("tp_points")) or config.default_tp_points
    unrealized = _pnl(event.side, entry, path[-1].close, _lot_size(event, config), config.contract_size)
    trailing_start = _float(setup.get("trailing_start_points")) or 90.0
    trailing_step = max(_float(setup.get("trailing_step_points")) or 50.0, 1.0)
    return {
        "event_key": _event_key(event),
        "setup_day": event.setup_day,
        "fill_timestamp": event.fill_timestamp.isoformat(sep=" "),
        "snapshot_timestamp": (path[-1].open_time + timedelta(minutes=series.timeframe_minutes)).isoformat(sep=" "),
        "snapshot_minutes_after_fill": minutes_after_fill,
        "data_source": series.label,
        "side": event.side,
        "entry_price": _round(entry),
        "initial_sl_points": _round(sl_points),
        "initial_tp_points": _round(tp_points),
        "trailing_start_points": trailing_start,
        "trailing_step_points": trailing_step,
        "distance_to_initial_sl_points": _round(max(sl_points - mae, 0.0)),
        "distance_to_initial_tp_points": _round(max(tp_points - mfe, 0.0)),
        "mfe_points_so_far": _round(mfe),
        "mae_points_so_far": _round(mae),
        "unrealized_pnl_so_far": _round(unrealized),
        "time_to_session_close_minutes": _round(_minutes_to_session_close(path[-1].open_time, config.session_close_hour)),
        "trailing_activated_so_far_proxy": mfe >= trailing_start,
        "trailing_modifications_so_far_proxy": _round(max(0.0, (mfe - trailing_start) / trailing_step)),
        "atr_points_setup": _round(atr_points),
        "atr_relative_mfe": _round(mfe / atr_points) if atr_points else "",
        "atr_relative_mae": _round(mae / atr_points) if atr_points else "",
        "post_entry_speed_points_per_min": _round(mfe / minutes_after_fill) if minutes_after_fill > 0 else "",
        "m1_m5_close_direction_proxy": _direction_proxy(event.side, path),
        "volume_proxy_sum": _round(sum(bar.volume for bar in path)),
        "spread_points_setup": _float(setup.get("spread_points")) or "",
        "spread_to_atr": _round((_float(setup.get("spread_points")) or 0.0) / atr_points) if atr_points else "",
        "fill_hour": event.fill_timestamp.hour,
        "day_of_week": setup.get("day_of_week", ""),
        "risk_context_trades_per_day_policy": 1,
        "risk_context_consecutive_losses_policy": 2,
        "future_outcome_excluded": True,
    }


def _price_break_even(
    event: AzirManagementEvent,
    path: list[OhlcvBar],
    series: PriceSeries,
    spec: dict[str, Any],
    threshold_points: float,
    config: TradeProtectionConfig,
) -> ProtectionResult:
    entry = _entry(event)
    if entry is None:
        return _unpriced_result(event, spec, "missing entry")
    activated = False
    for bar in path:
        if not activated and _bar_mfe_points(event.side, entry, bar, config.point) >= threshold_points:
            activated = True
        if activated and _bar_hits_price(bar, entry):
            close_dt = bar.open_time + timedelta(minutes=series.timeframe_minutes)
            return _priced_result(
                event,
                spec,
                series.label,
                close_dt,
                entry,
                "break_even_stop_hit_after_threshold",
                config,
                notes=f"BE activated after {threshold_points:.2f} points MFE.",
            )
    return _base_result(
        event,
        heuristic=str(spec["name"]),
        family=family_label(spec),
        notes="BE condition did not exit before base management.",
    )


def _price_close_early_rule(
    event: AzirManagementEvent,
    path: list[OhlcvBar],
    series: PriceSeries,
    spec: dict[str, Any],
    config: TradeProtectionConfig,
) -> ProtectionResult:
    entry = _entry(event)
    if entry is None:
        return _unpriced_result(event, spec, "missing entry")
    name = str(spec["name"])
    if name == "close_if_no_acceleration_after_15m":
        checkpoint = _path_until(path, event.fill_timestamp + timedelta(minutes=config.no_acceleration_minutes), series)
        if checkpoint:
            mfe = _path_mfe_points(event.side, entry, checkpoint, config.point)
            current = checkpoint[-1].close
            pnl_now = _pnl(event.side, entry, current, _lot_size(event, config), config.contract_size)
            if mfe < config.no_acceleration_min_mfe_points and pnl_now <= 0.0:
                return _priced_result(
                    event,
                    spec,
                    series.label,
                    checkpoint[-1].open_time + timedelta(minutes=series.timeframe_minutes),
                    current,
                    "close_early_no_acceleration",
                    config,
                    notes=f"MFE {mfe:.2f} points below threshold and unrealized PnL <= 0.",
                )
        return _base_result(event, heuristic=name, family=family_label(spec), notes="No-acceleration close condition did not trigger.")

    if name == "close_if_momentum_breaks_against_trade":
        checkpoint = _path_until(path, event.fill_timestamp + timedelta(minutes=config.momentum_break_minutes), series)
        if len(checkpoint) >= config.momentum_break_bars:
            last = checkpoint[-config.momentum_break_bars :]
            closes = [bar.close for bar in last]
            against = closes[-1] < closes[0] if event.side == "buy" else closes[-1] > closes[0]
            pnl_now = _pnl(event.side, entry, closes[-1], _lot_size(event, config), config.contract_size)
            if against and pnl_now <= 0.0:
                return _priced_result(
                    event,
                    spec,
                    series.label,
                    last[-1].open_time + timedelta(minutes=series.timeframe_minutes),
                    closes[-1],
                    "close_early_momentum_break_against_trade",
                    config,
                    notes="Last closed bars moved against the filled side while unrealized PnL was non-positive.",
                )
        return _base_result(event, heuristic=name, family=family_label(spec), notes="Momentum-break close condition did not trigger.")

    if name == "close_if_price_reverts_50pct_of_mfe":
        best_mfe = 0.0
        for bar in path:
            best_mfe = max(best_mfe, _bar_mfe_points(event.side, entry, bar, config.point))
            current_favorable = _favorable_points_from_price(event.side, entry, bar.close, config.point)
            reversion = best_mfe - current_favorable
            if best_mfe >= config.mfe_reversion_min_points and reversion >= best_mfe * config.mfe_reversion_fraction:
                return _priced_result(
                    event,
                    spec,
                    series.label,
                    bar.open_time + timedelta(minutes=series.timeframe_minutes),
                    bar.close,
                    "close_early_mfe_reversion",
                    config,
                    notes=f"Retraced {config.mfe_reversion_fraction:.0%} of MFE after reaching {best_mfe:.2f} points.",
                )
        return _base_result(event, heuristic=name, family=family_label(spec), notes="MFE-reversion close condition did not trigger.")
    return _unpriced_result(event, spec, "unknown close-early rule")


def _price_trailing_rule(
    event: AzirManagementEvent,
    path: list[OhlcvBar],
    series: PriceSeries,
    spec: dict[str, Any],
    config: TradeProtectionConfig,
) -> ProtectionResult:
    entry = _entry(event)
    if entry is None:
        return _unpriced_result(event, spec, "missing entry")
    side_filter = str(spec.get("side", "") or "")
    if side_filter and event.side != side_filter:
        return _base_result(event, heuristic=str(spec["name"]), family=family_label(spec), notes="Side-specific heuristic not active for this side.")
    activation = float(spec["activation_points"])
    step = float(spec["step_points"])
    best_price = entry
    stop: float | None = None
    for bar in path:
        mfe = _bar_mfe_points(event.side, entry, bar, config.point)
        if mfe >= activation:
            favorable_price = bar.high if event.side == "buy" else bar.low
            best_price = max(best_price, favorable_price) if event.side == "buy" else min(best_price, favorable_price)
            candidate = best_price - step * config.point if event.side == "buy" else best_price + step * config.point
            stop = candidate if stop is None else (max(stop, candidate) if event.side == "buy" else min(stop, candidate))
        if stop is not None and _bar_hits_price(bar, stop):
            return _priced_result(
                event,
                spec,
                series.label,
                bar.open_time + timedelta(minutes=series.timeframe_minutes),
                stop,
                "protective_trailing_stop_hit",
                config,
                notes=f"Activation={activation} points; step={step} points.",
            )
    return _base_result(event, heuristic=str(spec["name"]), family=family_label(spec), notes="Protective trailing did not exit before base management.")


def _heuristic_specs(config: TradeProtectionConfig) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"name": "always_base_management", "family": "baseline"}]
    specs.extend({"name": f"move_to_be_after_{int(points)}_points", "family": "break_even", "threshold_points": points} for points in config.break_even_points)
    specs.extend({"name": f"move_to_be_after_{multiple:.2f}_atr", "family": "break_even", "threshold_atr": multiple} for multiple in config.break_even_atr_multiples)
    specs.extend(
        [
            {"name": "close_if_no_acceleration_after_15m", "family": "close_early"},
            {"name": "close_if_momentum_breaks_against_trade", "family": "close_early"},
            {"name": "close_if_price_reverts_50pct_of_mfe", "family": "close_early"},
            {
                "name": "conservative_trailing_after_120_points",
                "family": "trailing",
                "activation_points": config.conservative_trailing_activation_points,
                "step_points": config.conservative_trailing_step_points,
            },
            {
                "name": "aggressive_trailing_after_60_points",
                "family": "trailing",
                "activation_points": config.aggressive_trailing_activation_points,
                "step_points": config.aggressive_trailing_step_points,
            },
            {
                "name": "sell_only_conservative_trailing_after_120_points",
                "family": "trailing",
                "side": "sell",
                "activation_points": config.conservative_trailing_activation_points,
                "step_points": config.conservative_trailing_step_points,
            },
            {
                "name": "buy_only_conservative_trailing_after_120_points",
                "family": "trailing",
                "side": "buy",
                "activation_points": config.conservative_trailing_activation_points,
                "step_points": config.conservative_trailing_step_points,
            },
        ]
    )
    return specs


def _base_result(
    event: AzirManagementEvent,
    *,
    heuristic: str = "always_base_management",
    family: str = "baseline",
    notes: str = "Observed/revalued protected benchmark outcome retained.",
) -> ProtectionResult:
    exit_dt = _event_exit_dt(event)
    return ProtectionResult(
        heuristic=heuristic,
        family=family,
        event_key=_event_key(event),
        setup_day=event.setup_day,
        fill_timestamp=event.fill_timestamp.isoformat(sep=" "),
        exit_timestamp=exit_dt.isoformat(sep=" ") if exit_dt else "",
        side=event.side,
        pricing_source="observed_protected_benchmark",
        exit_reason=str(event.trade.get("exit_reason", "")),
        net_pnl=event.protected_net_pnl,
        duration_minutes=_minutes_between(event.fill_timestamp, exit_dt) if exit_dt else None,
        notes=notes,
    )


def _priced_result(
    event: AzirManagementEvent,
    spec: dict[str, Any],
    pricing_source: str,
    exit_dt: datetime,
    exit_price: float,
    exit_reason: str,
    config: TradeProtectionConfig,
    *,
    notes: str = "",
) -> ProtectionResult:
    entry = _entry(event) or 0.0
    pnl = _pnl(event.side, entry, exit_price, _lot_size(event, config), config.contract_size)
    return ProtectionResult(
        heuristic=str(spec["name"]),
        family=family_label(spec),
        event_key=_event_key(event),
        setup_day=event.setup_day,
        fill_timestamp=event.fill_timestamp.isoformat(sep=" "),
        exit_timestamp=exit_dt.isoformat(sep=" "),
        side=event.side,
        pricing_source=pricing_source,
        exit_reason=exit_reason,
        net_pnl=pnl,
        duration_minutes=_minutes_between(event.fill_timestamp, exit_dt),
        notes=notes,
    )


def _unpriced_result(event: AzirManagementEvent, spec: dict[str, Any], reason: str) -> ProtectionResult:
    return ProtectionResult(
        heuristic=str(spec["name"]),
        family=family_label(spec),
        event_key=_event_key(event),
        setup_day=event.setup_day,
        fill_timestamp=event.fill_timestamp.isoformat(sep=" "),
        exit_timestamp=str(event.trade.get("exit_timestamp", "")),
        side=event.side,
        pricing_source="unpriced",
        exit_reason="unpriced",
        net_pnl=None,
        duration_minutes=None,
        notes=reason,
    )


def _build_report(
    *,
    mt5_log_path: Path,
    protected_report_path: Path,
    m1_input_path: Path,
    m5_input_path: Path,
    tick_input_path: Path,
    events: list[AzirManagementEvent],
    m1_bars: list[OhlcvBar],
    m5_bars: list[OhlcvBar],
    dataset_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    exit_distribution_rows: list[dict[str, Any]],
    risk_profile_rows: list[dict[str, Any]],
    config: TradeProtectionConfig,
) -> dict[str, Any]:
    base = next((row for row in comparison_rows if row["heuristic"] == "always_base_management"), {})
    candidates = [row for row in comparison_rows if row["heuristic"] != "always_base_management"]
    defendible = [row for row in candidates if row["assessment"] in {"promising", "watchlist"}]
    best_loss = max(
        candidates,
        key=lambda row: (
            _float(row["delta_average_loss_vs_base_same_coverage"]),
            _float(row["delta_net_pnl_vs_base_same_coverage"]),
            _float(row["profit_factor"]),
        ),
        default=None,
    )
    best_defendible = max(
        defendible,
        key=lambda row: (
            _float(row["delta_net_pnl_vs_base_same_coverage"]),
            _float(row["delta_average_loss_vs_base_same_coverage"]),
            _float(row["profit_factor"]),
        ),
        default=None,
    )
    tick_info = _file_info(tick_input_path)
    return {
        "sprint": "trade_protection_research_for_azir_v1",
        "source_paths": {
            "mt5_log_path": str(mt5_log_path),
            "protected_report_path": str(protected_report_path),
            "m1_input_path": str(m1_input_path),
            "m5_input_path": str(m5_input_path),
            "tick_input_path": str(tick_input_path),
        },
        "data_coverage": {
            "protected_trade_events": len(events),
            "m1_bars": len(m1_bars),
            "m1_start": m1_bars[0].open_time.isoformat(sep=" ") if m1_bars else "",
            "m1_end": m1_bars[-1].open_time.isoformat(sep=" ") if m1_bars else "",
            "m5_bars": len(m5_bars),
            "m5_start": m5_bars[0].open_time.isoformat(sep=" ") if m5_bars else "",
            "m5_end": m5_bars[-1].open_time.isoformat(sep=" ") if m5_bars else "",
            "tick_level_file_present": tick_info["exists"],
            "tick_level_size_bytes": tick_info["size_bytes"],
            "tick_level_usage_this_sprint": "located_only; M1/M5 causal bars used for screening, not tick-perfect benchmark freezing",
        },
        "problem_definition": {
            "unit": "one already-filled protected Azir trade",
            "decision_scope": "post-entry protection only",
            "base_entry": "baseline_azir_protected_economic_v1 with risk_engine_azir_v1",
            "forbidden_scope": ["new entries", "setup variants", "PPO/RL training", "Azir base changes", "Risk Engine changes"],
            "objective": "reduce left-tail losses and average loss while preserving most of Azir's win-rate and expectancy",
        },
        "config": asdict(config),
        "post_entry_dataset": {
            "rows": len(dataset_rows),
            "sample_rows_written": min(100, len(dataset_rows)),
            "leakage_policy": "Rows use only bars closed after fill and up to the snapshot timestamp; exit_reason/net_pnl/future MFE/MAE are excluded.",
        },
        "baseline_metrics": base,
        "heuristic_comparison": comparison_rows,
        "family_summary": family_rows,
        "exit_distribution": exit_distribution_rows,
        "risk_profile": risk_profile_rows,
        "decision": {
            "best_loss_reduction_candidate": best_loss["heuristic"] if best_loss else "",
            "best_defendible_candidate": best_defendible["heuristic"] if best_defendible else "",
            "trade_protection_signal_exists": bool(best_defendible),
            "ready_for_supervised_trade_protection_model_v1": bool(best_defendible),
            "ready_for_management_rl_env_reframed_v1": False,
            "recommended_next_sprint": "supervised_trade_protection_model_v1" if best_defendible else "trade_protection_feature_label_diagnostics_v1",
            "reason": (
                "At least one simple post-entry heuristic improves same-coverage economics without destroying core quality; supervised labels should be studied before RL."
                if best_defendible
                else "No simple protection heuristic improved the same-coverage trade-off enough to justify immediate IA/RL. More label diagnostics are safer first."
            ),
        },
        "limitations": [
            "Alternative exits are counterfactual bar replay, not MT5 broker execution.",
            "M1 data starts later than M5, so older trades may rely on M5 fallback for heuristic screening.",
            "M5 fallback preserves coverage but is weaker for intrabar stop/trailing chronology.",
            "Tick-level data is present but not fully consumed here; a future benchmark-grade sprint should price selected candidates tick-first.",
            "Post-entry features deliberately exclude observed final PnL, final exit reason, full-trade MFE and full-trade MAE.",
        ],
    }


def _family_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["family"] != "baseline":
            by_family.setdefault(row["family"], []).append(row)
    for family, family_rows in sorted(by_family.items()):
        best = max(
            family_rows,
            key=lambda row: (
                _float(row["delta_net_pnl_vs_base_same_coverage"]),
                _float(row["delta_average_loss_vs_base_same_coverage"]),
                _float(row["profit_factor"]),
            ),
        )
        result.append(
            {
                "family": family,
                "best_heuristic": best["heuristic"],
                "best_delta_net_pnl": best["delta_net_pnl_vs_base_same_coverage"],
                "best_delta_average_loss": best["delta_average_loss_vs_base_same_coverage"],
                "best_profit_factor": best["profit_factor"],
                "best_assessment": best["assessment"],
            }
        )
    return result


def _risk_profile_row(
    heuristic: str,
    family: str,
    metrics: dict[str, Any],
    base_metrics: dict[str, Any],
    closed_trades: int,
) -> dict[str, Any]:
    return {
        "heuristic": heuristic,
        "family": family,
        "closed_trades": closed_trades,
        "average_loss": metrics["average_loss"],
        "worst_loss": metrics["worst_loss"],
        "loss_p05": metrics["loss_p05"],
        "loss_to_average_win_abs": metrics["loss_to_average_win_abs"],
        "max_drawdown": metrics["max_drawdown"],
        "max_consecutive_losses": metrics["max_consecutive_losses"],
        "base_same_coverage_average_loss": base_metrics["average_loss"],
        "base_same_coverage_worst_loss": base_metrics["worst_loss"],
        "base_same_coverage_loss_to_average_win_abs": base_metrics["loss_to_average_win_abs"],
    }


def _metrics(pnl: list[float]) -> dict[str, Any]:
    wins = [value for value in pnl if value > 0.0]
    losses = [value for value in pnl if value < 0.0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    average_win = mean(wins) if wins else None
    average_loss = mean(losses) if losses else None
    return {
        "closed_trades": len(pnl),
        "net_pnl": _round(sum(pnl)),
        "gross_pnl": _round(sum(abs(value) for value in pnl)),
        "profit_factor": _round(gross_profit / gross_loss) if gross_loss else None,
        "expectancy": _round(mean(pnl)) if pnl else None,
        "win_rate": _round(len(wins) / len(pnl) * 100.0) if pnl else 0.0,
        "average_win": _round(average_win) if average_win is not None else None,
        "average_loss": _round(average_loss) if average_loss is not None else None,
        "payoff": _round(average_win / abs(average_loss)) if average_win is not None and average_loss else None,
        "max_drawdown": _round(_max_drawdown(pnl)),
        "max_consecutive_losses": _max_consecutive_losses(pnl),
        "worst_loss": _round(min(losses)) if losses else None,
        "loss_p05": _round(_percentile(losses, 0.05)) if losses else None,
        "loss_p10": _round(_percentile(losses, 0.10)) if losses else None,
        "loss_to_average_win_abs": _round(abs(average_loss) / average_win) if average_win and average_loss else None,
    }


def _heuristic_assessment(
    metrics: dict[str, Any],
    base_metrics: dict[str, Any],
    priced: int,
    total: int,
    heuristic: str,
) -> str:
    if heuristic == "always_base_management":
        return "official_baseline_reference"
    if total == 0 or priced / total < 0.30:
        return "too_little_coverage"
    delta_net = _float(metrics["net_pnl"]) - _float(base_metrics["net_pnl"])
    delta_avg_loss = _float(metrics["average_loss"]) - _float(base_metrics["average_loss"])
    win_rate_drop = _float(base_metrics["win_rate"]) - _float(metrics["win_rate"])
    pf = _float(metrics["profit_factor"])
    expectancy = _float(metrics["expectancy"])
    if delta_avg_loss > 0.0 and delta_net > 0.0 and (pf <= 1.0 or expectancy <= 0.0):
        return "diagnostic_loss_reduction_no_positive_edge"
    if priced / total < 0.50 and delta_net > 0.0 and delta_avg_loss > 0.0:
        return "thin_coverage_watchlist"
    if delta_net > 0.0 and delta_avg_loss >= 0.0 and pf > 1.0 and expectancy > 0.0 and win_rate_drop <= 5.0:
        return "promising"
    if delta_avg_loss > 0.0 and pf > 1.0 and expectancy > 0.0 and win_rate_drop <= 8.0:
        return "watchlist"
    if delta_avg_loss > 0.0 and delta_net < 0.0:
        return "loss_reduction_but_edge_costly"
    return "not_defendible"


def _best_series_for_event(
    event: AzirManagementEvent,
    exit_dt: datetime | None,
    series_by_source: list[PriceSeries],
) -> PriceSeries | None:
    if exit_dt is None:
        return None
    for series in series_by_source:
        if series.bars_after_fill_before_horizon(event.fill_timestamp, exit_dt):
            return series
    return None


def _threshold_points(event: AzirManagementEvent, spec: dict[str, Any], config: TradeProtectionConfig) -> float:
    if "threshold_atr" in spec:
        atr_points = _float(event.setup.get("atr_points")) or config.default_sl_points
        return max(1.0, atr_points * float(spec["threshold_atr"]))
    return float(spec.get("threshold_points", config.break_even_points[0]))


def _path_until(path: list[OhlcvBar], horizon: datetime, series: PriceSeries) -> list[OhlcvBar]:
    return [bar for bar in path if bar.open_time + timedelta(minutes=series.timeframe_minutes) <= horizon]


def _entry(event: AzirManagementEvent) -> float | None:
    return _to_float(event.trade.get("fill_price"))


def _event_exit_dt(event: AzirManagementEvent) -> datetime | None:
    text = str(event.trade.get("exit_timestamp", "") or "")
    if not text:
        return None
    parsed = _parse_timestamp(text)
    return parsed if parsed.year > 1 else None


def _event_key(event: AzirManagementEvent) -> str:
    return f"{event.setup_day}|{event.fill_timestamp.isoformat(sep=' ')}|{event.side}"


def _lot_size(event: AzirManagementEvent, config: TradeProtectionConfig) -> float:
    parsed = _to_float(event.setup.get("lot_size"))
    return parsed if parsed is not None and parsed > 0.0 else config.default_lot_size


def _pnl(side: str, entry: float, exit_price: float, lot_size: float, contract_size: float) -> float:
    direction = 1.0 if side == "buy" else -1.0
    return (exit_price - entry) * direction * lot_size * contract_size


def _bar_mfe_points(side: str, entry: float, bar: OhlcvBar, point: float) -> float:
    return max(0.0, (bar.high - entry) / point) if side == "buy" else max(0.0, (entry - bar.low) / point)


def _path_mfe_points(side: str, entry: float, path: list[OhlcvBar], point: float) -> float:
    if not path:
        return 0.0
    if side == "buy":
        return max(0.0, (max(bar.high for bar in path) - entry) / point)
    return max(0.0, (entry - min(bar.low for bar in path)) / point)


def _path_mae_points(side: str, entry: float, path: list[OhlcvBar], point: float) -> float:
    if not path:
        return 0.0
    if side == "buy":
        return max(0.0, (entry - min(bar.low for bar in path)) / point)
    return max(0.0, (max(bar.high for bar in path) - entry) / point)


def _favorable_points_from_price(side: str, entry: float, price: float, point: float) -> float:
    return (price - entry) / point if side == "buy" else (entry - price) / point


def _bar_hits_price(bar: OhlcvBar, price: float) -> bool:
    return bar.low <= price <= bar.high


def _direction_proxy(side: str, path: list[OhlcvBar]) -> float:
    if not path:
        return 0.0
    direction = path[-1].close - path[0].open
    return _round(direction if side == "buy" else -direction)


def _minutes_to_session_close(timestamp: datetime, close_hour: int) -> float:
    close_dt = timestamp.replace(hour=close_hour, minute=0, second=0, microsecond=0)
    return max(0.0, (close_dt - timestamp).total_seconds() / 60.0)


def _minutes_between(start: datetime, end: datetime | None) -> float:
    return 0.0 if end is None else (end - start).total_seconds() / 60.0


def _max_drawdown(pnl_values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in pnl_values:
        equity += value
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return max_dd


def _max_consecutive_losses(pnl_values: list[float]) -> int:
    current = 0
    max_seen = 0
    for value in pnl_values:
        if value < 0:
            current += 1
            max_seen = max(max_seen, current)
        else:
            current = 0
    return max_seen


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _float(value: Any) -> float:
    parsed = _to_float(value)
    return 0.0 if parsed is None else float(parsed)


def _delta(before: Any, after: Any) -> float | str:
    before_float = _to_float(before)
    after_float = _to_float(after)
    if before_float is None or after_float is None:
        return ""
    return _round(after_float - before_float)


def _filter_family(rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["family"] == family]


def family_label(spec: dict[str, Any]) -> str:
    return str(spec.get("family", "unknown"))


def _file_info(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}


def _dataset_schema_markdown() -> str:
    return (
        "# Post-Entry Dataset Schema\n\n"
        "## Unit\n\n"
        "One row is a causal snapshot of an already-filled Azir protected trade. "
        "Snapshots are taken only after the fill and only from bars closed at the snapshot timestamp.\n\n"
        "## Included Feature Groups\n\n"
        "- Trade state: side, entry, initial SL/TP points, distance to SL/TP, MFE/MAE so far, unrealized PnL so far, time to session close, trailing proxy state.\n"
        "- Market context: setup ATR, ATR-relative MFE/MAE, post-entry speed, M1/M5 close direction proxy, volume proxy, setup spread and spread/ATR.\n"
        "- Risk context: current policy constants from `risk_engine_azir_v1` that matter for interpretation.\n\n"
        "## Explicit Leakage Exclusions\n\n"
        "- Final `net_pnl` is not included.\n"
        "- Final `exit_reason` is not included.\n"
        "- Full-trade MFE/MAE after the snapshot is not included.\n"
        "- Any bar after the snapshot timestamp is excluded.\n"
        "- PPO/RL labels are not created in this sprint.\n"
    )


def _summary_markdown(report: dict[str, Any]) -> str:
    top = "\n".join(
        f"- `{row['heuristic']}` ({row['family']}): net={row['net_pnl']}, PF={row['profit_factor']}, "
        f"exp={row['expectancy']}, avg_loss={row['average_loss']}, "
        f"delta_net={row['delta_net_pnl_vs_base_same_coverage']}, assessment=`{row['assessment']}`."
        for row in report["heuristic_comparison"][:8]
    )
    family = "\n".join(
        f"- `{row['family']}`: best=`{row['best_heuristic']}`, delta_net={row['best_delta_net_pnl']}, "
        f"delta_avg_loss={row['best_delta_average_loss']}, assessment=`{row['best_assessment']}`."
        for row in report["family_summary"]
    )
    decision = report["decision"]
    return (
        "# Trade Protection Research For Azir V1\n\n"
        "## Executive Summary\n\n"
        f"- Protected Azir trades evaluated: {report['data_coverage']['protected_trade_events']}.\n"
        f"- Post-entry dataset rows: {report['post_entry_dataset']['rows']}.\n"
        f"- M1 coverage: {report['data_coverage']['m1_start']} -> {report['data_coverage']['m1_end']}.\n"
        f"- M5 coverage: {report['data_coverage']['m5_start']} -> {report['data_coverage']['m5_end']}.\n"
        f"- Tick file present: {report['data_coverage']['tick_level_file_present']}.\n"
        f"- Best defendible candidate: `{decision['best_defendible_candidate'] or 'none'}`.\n"
        f"- Trade-protection signal exists: {decision['trade_protection_signal_exists']}.\n\n"
        "## Top Heuristics\n\n"
        f"{top}\n\n"
        "## Family Summary\n\n"
        f"{family}\n\n"
        "## Decision\n\n"
        f"- {decision['reason']}\n"
        f"- Recommended next sprint: `{decision['recommended_next_sprint']}`.\n"
    )


def _assessment_markdown(report: dict[str, Any]) -> str:
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    decision = report["decision"]
    return (
        "# Protection Candidate Assessment\n\n"
        "## Decision\n\n"
        f"- Best loss-reduction candidate: `{decision['best_loss_reduction_candidate'] or 'none'}`.\n"
        f"- Best defendible candidate: `{decision['best_defendible_candidate'] or 'none'}`.\n"
        f"- Ready for supervised trade protection: {decision['ready_for_supervised_trade_protection_model_v1']}.\n"
        f"- Ready for management RL: {decision['ready_for_management_rl_env_reframed_v1']}.\n"
        f"- Reason: {decision['reason']}\n\n"
        "## Interpretation\n\n"
        "A heuristic is promising only if it improves same-coverage economics while keeping PF and expectancy positive "
        "and without destroying win rate. A heuristic that reduces average loss but gives back too much PnL remains diagnostic, not promotable.\n\n"
        "## Limitations\n\n"
        f"{limitations}\n"
    )


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "sprint": report["sprint"],
        "protected_trade_events": report["data_coverage"]["protected_trade_events"],
        "post_entry_dataset_rows": report["post_entry_dataset"]["rows"],
        "best_defendible_candidate": report["decision"]["best_defendible_candidate"],
        "trade_protection_signal_exists": report["decision"]["trade_protection_signal_exists"],
        "recommended_next_sprint": report["decision"]["recommended_next_sprint"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
