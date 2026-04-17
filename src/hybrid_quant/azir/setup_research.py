"""Controlled setup-level research for Azir.

This module deliberately does not change Azir, the protected benchmark, the Risk
Engine, or any RL layer. It runs small, interpretable setup variants through the
existing Python replica as a research proxy and compares them against the frozen
protected benchmark only as a reference anchor.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from .economic_audit import _round, _to_float, _write_csv
from .replica import AzirPythonReplica, AzirReplicaConfig, load_ohlcv_csv


DEFAULT_CONFIG = {
    "experiment": "setup_base_research_for_azir_v1",
    "symbol": "XAUUSD-STD",
    "benchmark": "baseline_azir_protected_economic_v1",
    "variants": [
        {
            "name": "azir_current_python_proxy",
            "family": "baseline_control",
            "description": "Current Python replica settings: rolling 10-bar swing and fixed 5-point offset.",
            "replica": {},
        },
        {"name": "swing_8_rolling", "family": "swing_definition", "replica": {"swing_bars": 8}},
        {"name": "swing_12_rolling", "family": "swing_definition", "replica": {"swing_bars": 12}},
        {"name": "swing_15_rolling", "family": "swing_definition", "replica": {"swing_bars": 15}},
        {
            "name": "swing_10_fractal",
            "family": "swing_definition",
            "replica": {"swing_bars": 10, "swing_definition": "fractal", "fractal_side_bars": 2},
        },
        {
            "name": "sell_only_diagnostic",
            "family": "buy_sell_asymmetry",
            "replica": {"allow_buys": False, "allow_sells": True},
        },
        {
            "name": "buy_only_diagnostic",
            "family": "buy_sell_asymmetry",
            "replica": {"allow_buys": True, "allow_sells": False},
        },
        {
            "name": "sell_friendly_swing_buy12_sell8",
            "family": "buy_sell_asymmetry",
            "replica": {"buy_swing_bars": 12, "sell_swing_bars": 8},
        },
        {
            "name": "sell_friendly_offset_buy7_sell3",
            "family": "buy_sell_asymmetry",
            "replica": {"buy_entry_offset_points": 7.0, "sell_entry_offset_points": 3.0},
        },
        {
            "name": "range_min_1_0_atr",
            "family": "range_quality",
            "replica": {"range_quality_enabled": True, "min_range_width_atr": 1.0},
        },
        {
            "name": "range_max_4_0_atr",
            "family": "range_quality",
            "replica": {"range_quality_enabled": True, "max_range_width_atr": 4.0},
        },
        {
            "name": "range_band_1_0_4_0_atr",
            "family": "range_quality",
            "replica": {"range_quality_enabled": True, "min_range_width_atr": 1.0, "max_range_width_atr": 4.0},
        },
        {
            "name": "compression_lte_4_0_atr",
            "family": "range_quality",
            "replica": {"range_quality_enabled": True, "max_compression_range_atr": 4.0},
        },
        {"name": "offset_3_points", "family": "entry_offset", "replica": {"entry_offset_points": 3.0}},
        {"name": "offset_8_points", "family": "entry_offset", "replica": {"entry_offset_points": 8.0}},
        {
            "name": "offset_atr_0_04",
            "family": "entry_offset",
            "replica": {"entry_offset_atr_fraction": 0.04},
        },
        {
            "name": "offset_sell3_buy7",
            "family": "entry_offset",
            "replica": {"buy_entry_offset_points": 7.0, "sell_entry_offset_points": 3.0},
        },
    ],
    "selection": {
        "min_delta_net_pnl": 25.0,
        "max_drawdown_worsening_factor": 1.1,
        "min_profit_factor": 1.05,
        "min_closed_trades_factor": 0.5,
    },
}


@dataclass(frozen=True)
class SetupVariant:
    name: str
    family: str
    description: str
    replica: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controlled Azir setup base research.")
    parser.add_argument("--m5-input-path", required=True, help="XAUUSD M5 OHLCV CSV using internal schema.")
    parser.add_argument("--protected-report-path", required=True, help="Frozen protected economic benchmark report.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-path", default="", help="Optional YAML variant matrix.")
    parser.add_argument("--symbol", default="XAUUSD-STD")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_setup_research(
        m5_input_path=Path(args.m5_input_path),
        protected_report_path=Path(args.protected_report_path),
        output_dir=Path(args.output_dir),
        config_path=Path(args.config_path) if args.config_path else None,
        symbol=args.symbol,
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_setup_research(
    *,
    m5_input_path: Path,
    protected_report_path: Path,
    output_dir: Path,
    config_path: Path | None = None,
    symbol: str = "XAUUSD-STD",
) -> dict[str, Any]:
    config = load_research_config(config_path)
    bars = load_ohlcv_csv(m5_input_path)
    if not bars:
        raise ValueError("M5 input has no bars.")
    protected_reference = load_protected_reference(protected_report_path)
    variants = parse_variants(config)
    if not variants:
        raise ValueError("Setup research config contains no variants.")

    variant_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    side_rows: list[dict[str, Any]] = []
    setup_rows: list[dict[str, Any]] = []
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    baseline_metrics: dict[str, Any] | None = None
    baseline_name = variants[0].name
    baseline_closed_trades = 0

    for variant in variants:
        events = run_variant(bars, variant, symbol)
        trades = extract_trade_rows(events)
        setups = extract_setup_rows(events, variant)
        metrics = compute_metrics(trades)
        if variant.name == baseline_name:
            baseline_metrics = metrics
            baseline_closed_trades = int(metrics["closed_trades"])
        if baseline_metrics is None:
            raise ValueError("First variant must be the baseline control.")
        row = build_variant_row(variant, metrics, baseline_metrics, baseline_closed_trades, protected_reference)
        variant_rows.append(row)
        family_rows[variant.family].append(row)
        setup_rows.extend(setups)
        yearly_rows.extend(breakdown_rows(variant, trades, "year"))
        side_rows.extend(breakdown_rows(variant, trades, "side"))

    ranked = rank_candidates(variant_rows, config.get("selection", {}))
    decision = build_decision(ranked, variant_rows)
    report = {
        "sprint": "setup_base_research_for_azir_v1",
        "methodology": {
            "scope": "setup-level research proxy",
            "mt5_rerun": False,
            "risk_engine_changed": False,
            "rl_used": False,
            "important_limitation": (
                "Counterfactual setup variants are priced with the bar-based Python replica, not with observed MT5 "
                "broker fills. A candidate must be re-exported/retested in MT5 before becoming a frozen benchmark."
            ),
        },
        "m5_input_path": str(m5_input_path),
        "m5_range": {"start": bars[0].open_time.isoformat(sep=" "), "end": bars[-1].open_time.isoformat(sep=" ")},
        "m5_rows": len(bars),
        "protected_report_path": str(protected_report_path),
        "protected_reference": protected_reference,
        "baseline_proxy_variant": baseline_name,
        "variants": [variant.__dict__ for variant in variants],
        "candidate_ranking": ranked,
        "decision": decision,
        "limitations": [
            "The current research is useful for screening setup ideas, not for freezing economic truth.",
            "The Python replica uses OHLC bar order assumptions for fills/exits and is not tick-perfect.",
            "Trailing remains the existing bar-based approximation inherited from the replica.",
            "Any positive candidate must be validated against MT5 logs/tick replay before replacing Azir setup logic.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(variant_rows, output_dir / "candidate_variants.csv")
    _write_csv(family_rows.get("swing_definition", []), output_dir / "swing_definition_comparison.csv")
    _write_csv(family_rows.get("buy_sell_asymmetry", []), output_dir / "buy_sell_asymmetry_comparison.csv")
    _write_csv(family_rows.get("range_quality", []), output_dir / "range_quality_filter_comparison.csv")
    _write_csv(family_rows.get("entry_offset", []), output_dir / "offset_comparison.csv")
    _write_csv(yearly_rows, output_dir / "yearly_variant_summary.csv")
    _write_csv(side_rows, output_dir / "side_variant_summary.csv")
    _write_csv(setup_rows, output_dir / "setup_variant_events.csv")
    (output_dir / "setup_research_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "setup_research_summary.md").write_text(summary_markdown(report, variant_rows), encoding="utf-8")
    return report


def load_research_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return dict(DEFAULT_CONFIG)
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    merged = dict(DEFAULT_CONFIG)
    for key, value in loaded.items():
        merged[key] = value
    return merged


def parse_variants(config: dict[str, Any]) -> list[SetupVariant]:
    variants: list[SetupVariant] = []
    for item in config.get("variants", []):
        variants.append(
            SetupVariant(
                name=str(item["name"]),
                family=str(item.get("family", "uncategorized")),
                description=str(item.get("description", "")),
                replica=dict(item.get("replica", {})),
            )
        )
    return variants


def load_protected_reference(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    metrics = report["metrics"]["azir_with_risk_engine_v1_forced_closes_revalued"]
    return {
        "benchmark_name": report.get("benchmark_name", "baseline_azir_protected_economic_v1"),
        "closed_trades": metrics.get("closed_trades"),
        "net_pnl": metrics.get("net_pnl"),
        "profit_factor": metrics.get("profit_factor"),
        "expectancy": metrics.get("expectancy"),
        "max_drawdown": metrics.get("max_drawdown_abs"),
        "max_consecutive_losses": metrics.get("max_consecutive_losses"),
        "note": "Protected observed/revalued benchmark; not directly comparable to counterfactual setup proxy.",
    }


def run_variant(bars: list[Any], variant: SetupVariant, symbol: str) -> list[dict[str, Any]]:
    fields = set(AzirReplicaConfig.__dataclass_fields__)
    overrides = {key: value for key, value in variant.replica.items() if key in fields}
    replica_config = AzirReplicaConfig(symbol=symbol, **overrides)
    return AzirPythonReplica(bars, replica_config).run()


def extract_trade_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in events:
        if row.get("event_type") != "exit":
            continue
        timestamp = str(row.get("timestamp", ""))
        setup_day = str(row.get("event_id", "")).split("_")[0] if row.get("event_id") else timestamp[:10].replace(".", "-")
        rows.append(
            {
                "has_exit": True,
                "setup_day": setup_day,
                "exit_timestamp": timestamp,
                "year": setup_day[:4],
                "side": row.get("fill_side", ""),
                "exit_reason": row.get("exit_reason", ""),
                "gross_pnl": _to_float(row.get("gross_pnl")) or 0.0,
                "net_pnl": _to_float(row.get("net_pnl")) or 0.0,
                "mfe_points": _to_float(row.get("mfe_points")) or 0.0,
                "mae_points": _to_float(row.get("mae_points")) or 0.0,
                "trailing_activated": str(row.get("trailing_activated", "")).lower() == "true",
            }
        )
    return rows


def extract_setup_rows(events: list[dict[str, Any]], variant: SetupVariant) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in events:
        if row.get("event_type") != "opportunity":
            continue
        rows.append(
            {
                "variant": variant.name,
                "family": variant.family,
                "timestamp": row.get("timestamp", ""),
                "buy_order_placed": row.get("buy_order_placed", ""),
                "sell_order_placed": row.get("sell_order_placed", ""),
                "swing_high": row.get("swing_high", ""),
                "swing_low": row.get("swing_low", ""),
                "buy_entry": row.get("buy_entry", ""),
                "sell_entry": row.get("sell_entry", ""),
                "pending_distance_points": row.get("pending_distance_points", ""),
                "range_width_atr": row.get("range_width_atr", ""),
                "compression_range_atr": row.get("compression_range_atr", ""),
                "range_quality_passed": row.get("range_quality_passed", ""),
                "atr_points": row.get("atr_points", ""),
                "prev_close_vs_ema20_points": row.get("prev_close_vs_ema20_points", ""),
            }
        )
    return rows


def compute_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    pnl = [_to_float(row.get("net_pnl")) or 0.0 for row in trades]
    gross = [_to_float(row.get("gross_pnl")) or 0.0 for row in trades]
    wins = [value for value in pnl if value > 0]
    losses = [value for value in pnl if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "closed_trades": len(trades),
        "gross_pnl": _round(sum(gross)),
        "net_pnl": _round(sum(pnl)),
        "win_rate": _round((len(wins) / len(pnl)) * 100.0) if pnl else 0.0,
        "average_win": _round(mean(wins)) if wins else None,
        "average_loss": _round(mean(losses)) if losses else None,
        "payoff": _round((mean(wins) / abs(mean(losses))) if wins and losses else 0.0) if wins and losses else None,
        "profit_factor": _round(gross_profit / gross_loss) if gross_loss else None,
        "expectancy": _round(mean(pnl)) if pnl else None,
        "max_drawdown": _round(max_drawdown(pnl)),
        "max_consecutive_losses": max_consecutive_losses(pnl),
        "positive_year_ratio": positive_period_ratio(trades, "year"),
    }


def build_variant_row(
    variant: SetupVariant,
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    baseline_closed_trades: int,
    protected: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "variant": variant.name,
        "family": variant.family,
        "description": variant.description,
        "replica_overrides": json.dumps(variant.replica, sort_keys=True),
    }
    row.update(metrics)
    row.update(
        {
            "delta_net_pnl_vs_python_baseline": _round((_to_float(metrics.get("net_pnl")) or 0.0) - (_to_float(baseline.get("net_pnl")) or 0.0)),
            "delta_expectancy_vs_python_baseline": _round((_to_float(metrics.get("expectancy")) or 0.0) - (_to_float(baseline.get("expectancy")) or 0.0)),
            "delta_max_drawdown_vs_python_baseline": _round((_to_float(metrics.get("max_drawdown")) or 0.0) - (_to_float(baseline.get("max_drawdown")) or 0.0)),
            "trade_count_ratio_vs_python_baseline": _round((metrics["closed_trades"] / baseline_closed_trades) if baseline_closed_trades else 0.0),
            "protected_reference_net_pnl": protected.get("net_pnl"),
            "protected_reference_profit_factor": protected.get("profit_factor"),
        }
    )
    return row


def breakdown_rows(variant: SetupVariant, trades: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trades:
        group_key = str(row.get(key, ""))
        grouped[group_key].append(row)
    rows: list[dict[str, Any]] = []
    for group_key, group_trades in sorted(grouped.items()):
        metrics = compute_metrics(group_trades)
        rows.append({"variant": variant.name, "family": variant.family, key: group_key, **metrics})
    return rows


def rank_candidates(rows: list[dict[str, Any]], selection: dict[str, Any]) -> list[dict[str, Any]]:
    if not rows:
        return []
    baseline = rows[0]
    min_delta = float(selection.get("min_delta_net_pnl", 25.0))
    max_dd_factor = float(selection.get("max_drawdown_worsening_factor", 1.1))
    min_pf = float(selection.get("min_profit_factor", 1.05))
    min_trade_factor = float(selection.get("min_closed_trades_factor", 0.5))
    baseline_dd = _to_float(baseline.get("max_drawdown")) or 0.0
    ranked: list[dict[str, Any]] = []
    for row in rows:
        pf = _to_float(row.get("profit_factor")) or 0.0
        delta = _to_float(row.get("delta_net_pnl_vs_python_baseline")) or 0.0
        dd = _to_float(row.get("max_drawdown")) or 0.0
        trade_ratio = _to_float(row.get("trade_count_ratio_vs_python_baseline")) or 0.0
        candidate = (
            row["variant"] != baseline["variant"]
            and delta >= min_delta
            and pf >= min_pf
            and (baseline_dd <= 0.0 or dd <= baseline_dd * max_dd_factor)
            and trade_ratio >= min_trade_factor
            and (_to_float(row.get("expectancy")) or 0.0) > 0.0
        )
        score = (
            (delta * 1.0)
            + ((_to_float(row.get("expectancy")) or 0.0) * 50.0)
            + (pf * 20.0)
            - (dd * 0.25)
            + ((_to_float(row.get("positive_year_ratio")) or 0.0) * 10.0)
        )
        ranked.append({**row, "candidate_pass": candidate, "selection_score": _round(score)})
    return sorted(ranked, key=lambda item: (bool(item["candidate_pass"]), _to_float(item.get("selection_score")) or 0.0), reverse=True)


def build_decision(ranked: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [row for row in ranked if row.get("candidate_pass")]
    best = passing[0] if passing else (ranked[0] if ranked else {})
    family_best: dict[str, dict[str, Any]] = {}
    for row in ranked:
        family_best.setdefault(str(row["family"]), row)
    return {
        "defensible_improvement_found": bool(passing),
        "recommended_variant": best.get("variant", ""),
        "recommended_family": best.get("family", ""),
        "most_promising_family_by_proxy": best.get("family", ""),
        "family_leaders": {
            family: {
                "variant": row.get("variant"),
                "net_pnl": row.get("net_pnl"),
                "profit_factor": row.get("profit_factor"),
                "expectancy": row.get("expectancy"),
                "max_drawdown": row.get("max_drawdown"),
                "delta_net_pnl_vs_python_baseline": row.get("delta_net_pnl_vs_python_baseline"),
                "candidate_pass": row.get("candidate_pass"),
            }
            for family, row in sorted(family_best.items())
        },
        "reason": (
            "At least one controlled setup variant improves the Python baseline proxy without breaking the risk/quality gates."
            if passing
            else "No controlled setup variant clears the defensible-improvement gate against the Python baseline proxy."
        ),
        "next_sprint": (
            "mt5_forward_parity_export_for_best_setup_candidate_v1"
            if passing
            else "setup_research_hypothesis_review_or_keep_azir_setup_frozen_v1"
        ),
    }


def positive_period_ratio(trades: list[dict[str, Any]], key: str) -> float:
    grouped: dict[str, float] = defaultdict(float)
    for row in trades:
        grouped[str(row.get(key, ""))] += _to_float(row.get("net_pnl")) or 0.0
    if not grouped:
        return 0.0
    return _round(len([value for value in grouped.values() if value > 0.0]) / len(grouped))


def max_drawdown(pnl: list[float]) -> float:
    peak = 0.0
    equity = 0.0
    worst = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst


def max_consecutive_losses(pnl: list[float]) -> int:
    current = 0
    worst = 0
    for value in pnl:
        if value < 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return worst


def summary_markdown(report: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    decision = report["decision"]
    protected = report["protected_reference"]
    top = report["candidate_ranking"][:8]
    table = "\n".join(
        "| {variant} | {family} | {closed_trades} | {net_pnl} | {profit_factor} | {expectancy} | {max_drawdown} | {delta} | {candidate} |".format(
            variant=row["variant"],
            family=row["family"],
            closed_trades=row["closed_trades"],
            net_pnl=row["net_pnl"],
            profit_factor=row["profit_factor"],
            expectancy=row["expectancy"],
            max_drawdown=row["max_drawdown"],
            delta=row["delta_net_pnl_vs_python_baseline"],
            candidate=row.get("candidate_pass"),
        )
        for row in top
    )
    return (
        "# Azir Setup Base Research V1\n\n"
        "## Executive Summary\n\n"
        "- Scope: controlled setup research only; no RL, no Risk Engine changes, no Azir MQL5 changes.\n"
        f"- M5 rows: {report['m5_rows']}; range: {report['m5_range']['start']} -> {report['m5_range']['end']}.\n"
        f"- Protected reference net/PF/exp/DD: {protected['net_pnl']} / {protected['profit_factor']} / "
        f"{protected['expectancy']} / {protected['max_drawdown']}.\n"
        f"- Defensible improvement found: {decision['defensible_improvement_found']}.\n"
        f"- Recommended variant: `{decision['recommended_variant']}`.\n"
        f"- Recommended next sprint: `{decision['next_sprint']}`.\n\n"
        "## Top Proxy Variants\n\n"
        "| variant | family | trades | net | PF | expectancy | max DD | delta vs baseline | pass |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        f"{table}\n\n"
        "## Interpretation Guardrail\n\n"
        "- These are counterfactual Python-replica results. A positive candidate is not a frozen benchmark.\n"
        "- The only frozen economic baseline remains `baseline_azir_protected_economic_v1` until MT5/tick replay validates a candidate.\n"
    )


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    decision = report["decision"]
    return {
        "sprint": report["sprint"],
        "m5_rows": report["m5_rows"],
        "m5_range": report["m5_range"],
        "protected_net_pnl": report["protected_reference"]["net_pnl"],
        "recommended_variant": decision["recommended_variant"],
        "defensible_improvement_found": decision["defensible_improvement_found"],
        "next_sprint": decision["next_sprint"],
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
