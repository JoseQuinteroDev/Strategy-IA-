"""Feature/label diagnostics for Azir post-entry trade protection.

This sprint deliberately stops before model training. Labels may use future
outcomes, but feature rows remain strictly causal at the observation timestamp.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from hybrid_quant.env.azir_management_env import AzirManagementEvent, build_azir_management_replay_dataset
from hybrid_quant.risk.azir_state import AzirRiskConfig

from .economic_audit import _parse_timestamp, _round, _to_float, _write_csv
from .replica import load_ohlcv_csv
from .trade_protection_research import (
    DEFAULT_M1_PATH,
    DEFAULT_M5_PATH,
    DEFAULT_MT5_LOG,
    DEFAULT_PROTECTED_REPORT,
    DEFAULT_TICK_PATH,
    PriceSeries,
    TradeProtectionConfig,
    build_post_entry_dataset,
)


DEFAULT_OUTPUT_DIR = Path("artifacts/azir-trade-protection-label-diagnostics-v1")
DEFAULT_RESEARCH_ARTIFACT_DIR = Path("artifacts/azir-trade-protection-research-v1")


@dataclass(frozen=True)
class LabelDiagnosticsConfig:
    materiality_pnl: float = 0.50
    deterioration_unrealized_pnl: float = -0.50
    deterioration_mae_atr_fraction: float = 0.50
    deterioration_max_mfe_atr_fraction: float = 0.25
    momentum_break_price_units: float = -0.25
    min_feature_auc_edge: float = 0.08
    min_label_positive_rate: float = 5.0
    max_label_positive_rate: float = 60.0
    min_labeled_snapshots: int = 250


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Azir post-entry labels/features before supervised/RL work.")
    parser.add_argument("--mt5-log-path", default=str(DEFAULT_MT5_LOG))
    parser.add_argument("--protected-report-path", default=str(DEFAULT_PROTECTED_REPORT))
    parser.add_argument("--m1-input-path", default=str(DEFAULT_M1_PATH))
    parser.add_argument("--m5-input-path", default=str(DEFAULT_M5_PATH))
    parser.add_argument("--tick-input-path", default=str(DEFAULT_TICK_PATH))
    parser.add_argument("--research-artifact-dir", default=str(DEFAULT_RESEARCH_ARTIFACT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbol", default="XAUUSD-STD")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_trade_protection_label_diagnostics(
        mt5_log_path=Path(args.mt5_log_path),
        protected_report_path=Path(args.protected_report_path),
        m1_input_path=Path(args.m1_input_path),
        m5_input_path=Path(args.m5_input_path),
        tick_input_path=Path(args.tick_input_path),
        research_artifact_dir=Path(args.research_artifact_dir),
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_trade_protection_label_diagnostics(
    *,
    mt5_log_path: Path,
    protected_report_path: Path,
    m1_input_path: Path,
    m5_input_path: Path,
    tick_input_path: Path,
    research_artifact_dir: Path,
    output_dir: Path,
    symbol: str = "XAUUSD-STD",
    protection_config: TradeProtectionConfig | None = None,
    label_config: LabelDiagnosticsConfig | None = None,
) -> dict[str, Any]:
    protection_config = protection_config or TradeProtectionConfig()
    label_config = label_config or LabelDiagnosticsConfig()
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
    post_entry_rows = build_post_entry_dataset(events, series_by_source, protection_config)
    labeled_rows = label_post_entry_rows(post_entry_rows, events, label_config)
    label_distribution_rows = build_label_distribution(labeled_rows)
    feature_rows = diagnose_features(labeled_rows)
    separability = build_separability_summary(feature_rows, label_distribution_rows, label_config)
    coverage_rows = build_coverage_bias_report(events, labeled_rows, protection_config)
    report = build_report(
        mt5_log_path=mt5_log_path,
        protected_report_path=protected_report_path,
        m1_input_path=m1_input_path,
        m5_input_path=m5_input_path,
        tick_input_path=tick_input_path,
        research_artifact_dir=research_artifact_dir,
        events=events,
        post_entry_rows=post_entry_rows,
        labeled_rows=labeled_rows,
        label_distribution_rows=label_distribution_rows,
        feature_rows=feature_rows,
        coverage_rows=coverage_rows,
        separability=separability,
        protection_config=protection_config,
        label_config=label_config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(labeled_rows, output_dir / "trade_protection_labeled_dataset.csv")
    _write_csv(labeled_rows[:100], output_dir / "trade_protection_sample_labeled_dataset.csv")
    _write_csv(label_distribution_rows, output_dir / "trade_protection_label_distribution.csv")
    _write_csv(feature_rows, output_dir / "trade_protection_feature_diagnostics.csv")
    _write_csv(coverage_rows, output_dir / "trade_protection_coverage_bias_report.csv")
    (output_dir / "trade_protection_label_definitions.md").write_text(label_definitions_markdown(label_config), encoding="utf-8")
    (output_dir / "trade_protection_separability_report.md").write_text(separability_markdown(report), encoding="utf-8")
    (output_dir / "trade_protection_label_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def label_post_entry_rows(
    post_entry_rows: list[dict[str, Any]],
    events: list[AzirManagementEvent],
    config: LabelDiagnosticsConfig,
) -> list[dict[str, Any]]:
    events_by_key = {_event_key(event): event for event in events}
    rows: list[dict[str, Any]] = []
    for source in post_entry_rows:
        row = dict(source)
        event = events_by_key.get(str(row.get("event_key", "")))
        if event is None:
            continue
        base_pnl = float(event.protected_net_pnl)
        early_close_pnl = _float(row.get("unrealized_pnl_so_far"))
        early_close_delta = early_close_pnl - base_pnl
        atr_points = max(_float(row.get("atr_points_setup")), 1.0)
        mfe = _float(row.get("mfe_points_so_far"))
        mae = _float(row.get("mae_points_so_far"))
        unrealized = _float(row.get("unrealized_pnl_so_far"))
        direction_proxy = _float(row.get("m1_m5_close_direction_proxy"))

        deteriorated = (
            unrealized <= config.deterioration_unrealized_pnl
            or (mae >= atr_points * config.deterioration_mae_atr_fraction and mfe <= atr_points * config.deterioration_max_mfe_atr_fraction)
        )
        momentum_break = direction_proxy <= config.momentum_break_price_units and unrealized <= 0.0
        early_helpful = early_close_delta >= config.materiality_pnl
        early_harmful = early_close_delta <= -config.materiality_pnl
        recoverable = (deteriorated or momentum_break) and (base_pnl - early_close_pnl >= config.materiality_pnl)
        false_deterioration = (deteriorated or momentum_break) and early_harmful

        row.update(
            {
                "label_deteriorated_trade": int(deteriorated),
                "label_early_close_helpful": int(early_helpful),
                "label_early_close_harmful": int(early_harmful),
                "label_recoverable_trade": int(recoverable),
                "label_momentum_break_true": int(momentum_break and early_helpful),
                "label_false_deterioration": int(false_deterioration),
                "label_base_final_net_pnl": _round(base_pnl),
                "label_early_close_proxy_pnl": _round(early_close_pnl),
                "label_early_close_delta_vs_base": _round(early_close_delta),
                "label_final_winner": int(base_pnl > 0.0),
                "label_final_loser": int(base_pnl < 0.0),
            }
        )
        rows.append(row)
    return rows


def build_label_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label in LABEL_COLUMNS:
        positives = sum(int(row.get(label, 0)) for row in rows)
        result.append(
            {
                "label": label,
                "rows": len(rows),
                "positive_rows": positives,
                "negative_rows": len(rows) - positives,
                "positive_pct": _round(positives / len(rows) * 100.0) if rows else 0.0,
            }
        )
    return result


def diagnose_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feature_names = [name for name in NUMERIC_FEATURES if any(str(row.get(name, "")).strip() != "" for row in rows)]
    result: list[dict[str, Any]] = []
    for feature in feature_names:
        values = [_optional_float(row.get(feature)) for row in rows]
        present = [value for value in values if value is not None and math.isfinite(value)]
        missing = len(values) - len(present)
        stats = _feature_stats(present)
        for label in LABEL_COLUMNS:
            y = [int(row.get(label, 0)) for row, value in zip(rows, values, strict=False) if value is not None and math.isfinite(value)]
            x = [float(value) for value in values if value is not None and math.isfinite(value)]
            auc = _auc(x, y) if x and len(set(y)) == 2 else None
            corr = _point_biserial(x, y) if x and len(set(y)) == 2 else None
            result.append(
                {
                    "feature": feature,
                    "label": label,
                    "rows": len(rows),
                    "present_rows": len(present),
                    "missing_rows": missing,
                    "missing_pct": _round(missing / len(rows) * 100.0) if rows else 0.0,
                    **stats,
                    "auc": "" if auc is None else _round(auc),
                    "auc_edge_abs": "" if auc is None else _round(abs(auc - 0.5)),
                    "point_biserial_corr": "" if corr is None else _round(corr),
                    "scale_warning": _scale_warning(stats),
                    "leakage_warning": "no" if feature not in FORBIDDEN_FEATURES else "yes",
                }
            )
    result.sort(key=lambda row: (_float(row.get("auc_edge_abs")), abs(_float(row.get("point_biserial_corr")))), reverse=True)
    return result


def build_coverage_bias_report(
    events: list[AzirManagementEvent],
    labeled_rows: list[dict[str, Any]],
    protection_config: TradeProtectionConfig,
) -> list[dict[str, Any]]:
    labeled_keys = {str(row["event_key"]) for row in labeled_rows}
    by_key_snapshots = Counter(str(row["event_key"]) for row in labeled_rows)
    rows: list[dict[str, Any]] = []
    for group_name, group_fn in {
        "year": lambda event: event.setup_day[:4],
        "side": lambda event: event.side,
        "winner_loser": lambda event: "winner" if event.protected_net_pnl > 0 else "loser" if event.protected_net_pnl < 0 else "flat",
        "duration_bucket": lambda event: _duration_bucket(event),
    }.items():
        grouped: dict[str, list[AzirManagementEvent]] = defaultdict(list)
        for event in events:
            grouped[group_fn(event)].append(event)
        for value, group_events in sorted(grouped.items()):
            included = [event for event in group_events if _event_key(event) in labeled_keys]
            excluded = len(group_events) - len(included)
            rows.append(
                {
                    "group": group_name,
                    "value": value,
                    "total_trades": len(group_events),
                    "included_trades": len(included),
                    "excluded_trades": excluded,
                    "included_pct": _round(len(included) / len(group_events) * 100.0) if group_events else 0.0,
                    "avg_base_pnl_all": _round(mean(event.protected_net_pnl for event in group_events)) if group_events else 0.0,
                    "avg_base_pnl_included": _round(mean(event.protected_net_pnl for event in included)) if included else "",
                    "snapshot_rows": sum(by_key_snapshots.get(_event_key(event), 0) for event in group_events),
                }
            )
    reason_counts: Counter[str] = Counter()
    min_snapshot = min(protection_config.snapshot_minutes)
    for event in events:
        if _event_key(event) in labeled_keys:
            continue
        duration = _duration_minutes(event)
        if duration is None:
            reason_counts["missing_exit_timestamp"] += 1
        elif duration < min_snapshot:
            reason_counts[f"duration_below_{min_snapshot}m_first_snapshot"] += 1
        else:
            reason_counts["no_causal_price_path_for_snapshot"] += 1
    for reason, count in sorted(reason_counts.items()):
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


LABEL_COLUMNS = (
    "label_deteriorated_trade",
    "label_early_close_helpful",
    "label_early_close_harmful",
    "label_recoverable_trade",
    "label_momentum_break_true",
    "label_false_deterioration",
)


NUMERIC_FEATURES = (
    "snapshot_minutes_after_fill",
    "entry_price",
    "initial_sl_points",
    "initial_tp_points",
    "trailing_start_points",
    "trailing_step_points",
    "distance_to_initial_sl_points",
    "distance_to_initial_tp_points",
    "mfe_points_so_far",
    "mae_points_so_far",
    "unrealized_pnl_so_far",
    "time_to_session_close_minutes",
    "trailing_modifications_so_far_proxy",
    "atr_points_setup",
    "atr_relative_mfe",
    "atr_relative_mae",
    "post_entry_speed_points_per_min",
    "m1_m5_close_direction_proxy",
    "volume_proxy_sum",
    "tick_count_so_far",
    "spread_points_current",
    "spread_points_mean",
    "spread_points_setup",
    "spread_to_atr",
    "fill_hour",
    "day_of_week",
    "risk_context_trades_per_day_policy",
    "risk_context_consecutive_losses_policy",
)


FORBIDDEN_FEATURES = {
    "label_base_final_net_pnl",
    "label_early_close_delta_vs_base",
    "label_final_winner",
    "label_final_loser",
}


def build_separability_summary(
    feature_rows: list[dict[str, Any]],
    label_distribution_rows: list[dict[str, Any]],
    config: LabelDiagnosticsConfig,
) -> dict[str, Any]:
    label_dist = {row["label"]: row for row in label_distribution_rows}
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        by_label[str(row["label"])].append(row)
    summaries: dict[str, Any] = {}
    for label, rows in sorted(by_label.items()):
        top = sorted(rows, key=lambda row: _float(row.get("auc_edge_abs")), reverse=True)[:8]
        positive_pct = _float(label_dist.get(label, {}).get("positive_pct"))
        best_auc_edge = _float(top[0].get("auc_edge_abs")) if top else 0.0
        usable = (
            len(rows) > 0
            and config.min_label_positive_rate <= positive_pct <= config.max_label_positive_rate
            and best_auc_edge >= config.min_feature_auc_edge
        )
        summaries[label] = {
            "positive_pct": positive_pct,
            "best_auc_edge_abs": _round(best_auc_edge),
            "top_features": [
                {
                    "feature": item["feature"],
                    "auc": item["auc"],
                    "auc_edge_abs": item["auc_edge_abs"],
                    "point_biserial_corr": item["point_biserial_corr"],
                    "scale_warning": item["scale_warning"],
                }
                for item in top
            ],
            "separability_assessment": "promising_for_diagnostics" if usable else "weak_or_unbalanced",
        }
    return summaries


def build_report(
    *,
    mt5_log_path: Path,
    protected_report_path: Path,
    m1_input_path: Path,
    m5_input_path: Path,
    tick_input_path: Path,
    research_artifact_dir: Path,
    events: list[AzirManagementEvent],
    post_entry_rows: list[dict[str, Any]],
    labeled_rows: list[dict[str, Any]],
    label_distribution_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    separability: dict[str, Any],
    protection_config: TradeProtectionConfig,
    label_config: LabelDiagnosticsConfig,
) -> dict[str, Any]:
    promising_labels = [
        label
        for label, payload in separability.items()
        if payload["separability_assessment"] == "promising_for_diagnostics"
    ]
    best_label = max(
        promising_labels,
        key=lambda label: _float(separability[label]["best_auc_edge_abs"]),
        default="",
    )
    severe_bias = _coverage_is_severely_biased(coverage_rows)
    ready = bool(best_label) and len(labeled_rows) >= label_config.min_labeled_snapshots and not severe_bias
    return {
        "sprint": "trade_protection_feature_label_diagnostics_v1",
        "source_paths": {
            "mt5_log_path": str(mt5_log_path),
            "protected_report_path": str(protected_report_path),
            "m1_input_path": str(m1_input_path),
            "m5_input_path": str(m5_input_path),
            "tick_input_path": str(tick_input_path),
            "trade_protection_research_artifact_dir": str(research_artifact_dir),
        },
        "inputs_found": {
            "mt5_log_path": mt5_log_path.exists(),
            "protected_report_path": protected_report_path.exists(),
            "m1_input_path": m1_input_path.exists(),
            "m5_input_path": m5_input_path.exists(),
            "tick_input_path": tick_input_path.exists(),
            "trade_protection_research_artifact_dir": research_artifact_dir.exists(),
        },
        "dataset": {
            "protected_trade_events": len(events),
            "post_entry_snapshot_rows": len(post_entry_rows),
            "labeled_rows": len(labeled_rows),
            "unique_labeled_trades": len({row["event_key"] for row in labeled_rows}),
            "feature_count_diagnosed": len(set(row["feature"] for row in feature_rows)),
            "label_count": len(LABEL_COLUMNS),
        },
        "protection_config": asdict(protection_config),
        "label_config": asdict(label_config),
        "label_distribution": label_distribution_rows,
        "separability": separability,
        "top_feature_diagnostics": feature_rows[:25],
        "coverage_bias": coverage_rows,
        "decision": {
            "best_label_candidate": best_label,
            "separability_real_enough_for_supervised_v1": ready,
            "coverage_bias_severe": severe_bias,
            "ready_for_ppo_or_rl": False,
            "recommended_next_sprint": "supervised_trade_protection_model_v1" if ready else "tick_level_trade_protection_label_replay_v1",
            "reason": (
                f"`{best_label}` has enough univariate separability and sample coverage to justify a small supervised diagnostic model."
                if ready
                else "Labels/features are useful diagnostically, but coverage bias or weak separability makes supervised/RL premature."
            ),
        },
        "limitations": [
            "Labels may use future protected outcomes; features do not.",
            "Early-close PnL is a snapshot close proxy, not a broker-confirmed tick fill.",
            "The labeled sample is limited to trades with enough post-entry path for at least one snapshot.",
            "This sprint performs univariate diagnostics only; no final supervised or RL model is trained.",
        ],
    }


def label_definitions_markdown(config: LabelDiagnosticsConfig) -> str:
    return (
        "# Trade Protection Label Definitions\n\n"
        "## Observation Timestamp\n\n"
        "Each row is observed at a post-fill snapshot timestamp. Features can only use bars closed between fill and that snapshot.\n\n"
        "## Labels\n\n"
        f"- `deteriorated_trade`: causal state label. True when unrealized PnL <= {config.deterioration_unrealized_pnl}, or MAE is at least {config.deterioration_mae_atr_fraction} ATR while MFE is no more than {config.deterioration_max_mfe_atr_fraction} ATR.\n"
        f"- `early_close_helpful`: outcome label. True when closing at the snapshot proxy would improve final protected PnL by at least {config.materiality_pnl}.\n"
        f"- `early_close_harmful`: outcome label. True when closing at the snapshot proxy would worsen final protected PnL by at least {config.materiality_pnl}.\n"
        "- `recoverable_trade`: true when the snapshot looks deteriorated or momentum-broken but the protected final outcome recovers materially versus snapshot close.\n"
        "- `momentum_break_true`: true when the causal momentum-break condition appears and early close would have helped materially.\n"
        "- `false_deterioration`: true when deterioration/momentum-break appears but early close would have been harmful.\n\n"
        "## Leakage Rules\n\n"
        "- Final PnL and final exit reason are labels only, not features.\n"
        "- Full-trade MFE/MAE after the snapshot are not features.\n"
        "- Bars after the snapshot timestamp are prohibited in features.\n"
    )


def separability_markdown(report: dict[str, Any]) -> str:
    lines = []
    for label, payload in report["separability"].items():
        top = ", ".join(f"`{item['feature']}` auc_edge={item['auc_edge_abs']}" for item in payload["top_features"][:4])
        lines.append(
            f"- `{label}`: positive={payload['positive_pct']}%, best_auc_edge={payload['best_auc_edge_abs']}, "
            f"assessment=`{payload['separability_assessment']}`. Top: {top}."
        )
    return (
        "# Trade Protection Separability Report\n\n"
        "## Summary\n\n"
        f"- Labeled rows: {report['dataset']['labeled_rows']}.\n"
        f"- Unique labeled trades: {report['dataset']['unique_labeled_trades']}.\n"
        f"- Best label candidate: `{report['decision']['best_label_candidate'] or 'none'}`.\n"
        f"- Ready for supervised v1: {report['decision']['separability_real_enough_for_supervised_v1']}.\n"
        f"- Reason: {report['decision']['reason']}\n\n"
        "## Label-Level Separability\n\n"
        + "\n".join(lines)
        + "\n\n## Important Caution\n\n"
        "This is not model validation. It is a pre-model diagnostic to decide whether labels are learnable enough to justify a small supervised sprint.\n"
    )


def _feature_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"min": "", "max": "", "mean": "", "std": "", "unique_values": 0}
    return {
        "min": _round(min(values)),
        "max": _round(max(values)),
        "mean": _round(mean(values)),
        "std": _round(pstdev(values)) if len(values) > 1 else 0.0,
        "unique_values": len(set(_round(value) for value in values)),
    }


def _auc(values: list[float], labels: list[int]) -> float | None:
    pairs = [(value, label) for value, label in zip(values, labels, strict=False)]
    positives = sum(label == 1 for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    sorted_pairs = sorted(pairs, key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(sorted_pairs):
        end = index + 1
        while end < len(sorted_pairs) and sorted_pairs[end][0] == sorted_pairs[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        rank_sum += avg_rank * sum(label == 1 for _, label in sorted_pairs[index:end])
        index = end
    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return max(auc, 1.0 - auc)


def _point_biserial(values: list[float], labels: list[int]) -> float | None:
    if not values or len(set(labels)) < 2:
        return None
    mean_x = mean(values)
    mean_y = mean(labels)
    denom_x = math.sqrt(sum((value - mean_x) ** 2 for value in values))
    denom_y = math.sqrt(sum((label - mean_y) ** 2 for label in labels))
    if denom_x == 0.0 or denom_y == 0.0:
        return None
    return sum((value - mean_x) * (label - mean_y) for value, label in zip(values, labels, strict=False)) / (denom_x * denom_y)


def _scale_warning(stats: dict[str, Any]) -> str:
    min_value = _to_float(stats.get("min"))
    max_value = _to_float(stats.get("max"))
    std = _to_float(stats.get("std"))
    unique = int(stats.get("unique_values") or 0)
    if unique <= 2:
        return "binary_or_near_constant"
    if std is not None and std == 0.0:
        return "constant"
    if min_value is not None and max_value is not None and abs(max_value - min_value) > 10000:
        return "large_scale_needs_normalization"
    return "ok"


def _coverage_is_severely_biased(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if row.get("group") == "winner_loser" and row.get("value") in {"winner", "loser"}:
            included = _to_float(row.get("included_pct"))
            if included is not None and included < 25.0:
                return True
    return False


def _duration_bucket(event: AzirManagementEvent) -> str:
    duration = _duration_minutes(event)
    if duration is None:
        return "missing"
    if duration < 5:
        return "<5m"
    if duration < 15:
        return "5-15m"
    if duration < 30:
        return "15-30m"
    if duration < 60:
        return "30-60m"
    return ">=60m"


def _duration_minutes(event: AzirManagementEvent) -> float | None:
    exit_dt = _event_exit_dt(event)
    if exit_dt is None:
        return None
    return (exit_dt - event.fill_timestamp).total_seconds() / 60.0


def _event_exit_dt(event: AzirManagementEvent) -> datetime | None:
    text = str(event.trade.get("exit_timestamp", "") or "")
    if not text:
        return None
    parsed = _parse_timestamp(text)
    return parsed if parsed.year > 1 else None


def _event_key(event: AzirManagementEvent) -> str:
    return f"{event.setup_day}|{event.fill_timestamp.isoformat(sep=' ')}|{event.side}"


def _optional_float(value: Any) -> float | None:
    parsed = _to_float(value)
    return None if parsed is None else float(parsed)


def _float(value: Any) -> float:
    parsed = _to_float(value)
    return 0.0 if parsed is None else float(parsed)


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "sprint": report["sprint"],
        "protected_trade_events": report["dataset"]["protected_trade_events"],
        "labeled_rows": report["dataset"]["labeled_rows"],
        "unique_labeled_trades": report["dataset"]["unique_labeled_trades"],
        "best_label_candidate": report["decision"]["best_label_candidate"],
        "ready_for_supervised_v1": report["decision"]["separability_real_enough_for_supervised_v1"],
        "recommended_next_sprint": report["decision"]["recommended_next_sprint"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
