"""Threshold refinement for Azir tick-level trade-protection labels.

This sprint compares a small set of interpretable early-deterioration rules on
the tick-level labeled dataset. It does not train a supervised model or RL
policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from .economic_audit import _round, _to_float, _write_csv
from .trade_protection_label_diagnostics import LABEL_COLUMNS, NUMERIC_FEATURES


DEFAULT_OUTPUT_DIR = Path("artifacts/azir-trade-protection-threshold-refinement-v1")
DEFAULT_TICK_LABEL_ARTIFACT_DIR = Path("artifacts/azir-tick-level-trade-protection-label-replay-v1")
DEFAULT_PREVIOUS_LABEL_ARTIFACT_DIR = Path("artifacts/azir-trade-protection-label-diagnostics-v1")
DEFAULT_MT5_LOG = Path(r"C:\Users\joseq\Documents\Playground\todos_los_ticks.csv")
DEFAULT_TICK_PATH = Path(r"C:\Users\joseq\Documents\tick_level.csv")
DEFAULT_M1_PATH = Path(r"C:\Users\joseq\Documents\xauusd_m1.csv")
DEFAULT_M5_PATH = Path(r"C:\Users\joseq\Documents\xauusd_m5.csv")
DEFAULT_PROTECTED_REPORT = Path("artifacts/azir-protected-economic-v1-freeze/forced_close_revaluation_report.json")


@dataclass(frozen=True)
class ThresholdRefinementConfig:
    helpful_materiality: float = 0.50
    harmful_materiality: float = 0.50
    analysis_years: tuple[str, ...] = ("2023", "2024", "2025")
    decision_years: tuple[str, ...] = ("2024", "2025")
    min_triggered_snapshots: int = 40
    min_helpful_precision: float = 35.0
    max_harmful_rate: float = 55.0
    max_side_precision_gap: float = 25.0


@dataclass(frozen=True)
class MomentumCandidate:
    name: str
    direction_max: float
    unrealized_max: float
    mae_atr_min: float = 0.0
    speed_max: float | None = None
    max_horizon_seconds: int | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine Azir trade-protection label thresholds without model training.")
    parser.add_argument("--tick-label-artifact-dir", default=str(DEFAULT_TICK_LABEL_ARTIFACT_DIR))
    parser.add_argument("--previous-label-artifact-dir", default=str(DEFAULT_PREVIOUS_LABEL_ARTIFACT_DIR))
    parser.add_argument("--mt5-log-path", default=str(DEFAULT_MT5_LOG))
    parser.add_argument("--tick-input-path", default=str(DEFAULT_TICK_PATH))
    parser.add_argument("--m1-input-path", default=str(DEFAULT_M1_PATH))
    parser.add_argument("--m5-input-path", default=str(DEFAULT_M5_PATH))
    parser.add_argument("--protected-report-path", default=str(DEFAULT_PROTECTED_REPORT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_threshold_refinement(
        tick_label_artifact_dir=Path(args.tick_label_artifact_dir),
        previous_label_artifact_dir=Path(args.previous_label_artifact_dir),
        mt5_log_path=Path(args.mt5_log_path),
        tick_input_path=Path(args.tick_input_path),
        m1_input_path=Path(args.m1_input_path),
        m5_input_path=Path(args.m5_input_path),
        protected_report_path=Path(args.protected_report_path),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_threshold_refinement(
    *,
    tick_label_artifact_dir: Path,
    previous_label_artifact_dir: Path,
    mt5_log_path: Path,
    tick_input_path: Path,
    m1_input_path: Path,
    m5_input_path: Path,
    protected_report_path: Path,
    output_dir: Path,
    config: ThresholdRefinementConfig | None = None,
) -> dict[str, Any]:
    config = config or ThresholdRefinementConfig()
    labeled_path = tick_label_artifact_dir / "tick_level_trade_protection_labeled_dataset.csv"
    rows = load_labeled_rows(labeled_path)
    analysis_rows = [row for row in rows if str(row.get("setup_day", ""))[:4] in config.analysis_years]
    candidates = momentum_candidates()
    momentum_rows = [evaluate_momentum_candidate(candidate, analysis_rows, config) for candidate in candidates]
    momentum_rows.sort(key=lambda row: (_float(row["score"]), _float(row["helpful_precision_pct"]), _float(row["avg_delta_pnl"])), reverse=True)
    early_rows = evaluate_early_close_helpful_thresholds(analysis_rows)
    best = momentum_rows[0] if momentum_rows else {}
    stability = {
        "year": stability_rows(best.get("candidate", ""), candidates, analysis_rows, "year", config),
        "side": stability_rows(best.get("candidate", ""), candidates, analysis_rows, "side", config),
        "horizon": stability_rows(best.get("candidate", ""), candidates, analysis_rows, "snapshot_seconds_after_fill", config),
        "source": stability_rows(best.get("candidate", ""), candidates, analysis_rows, "data_source", config),
    }
    report = build_report(
        rows=rows,
        analysis_rows=analysis_rows,
        momentum_rows=momentum_rows,
        early_rows=early_rows,
        stability=stability,
        config=config,
        paths={
            "tick_label_artifact_dir": tick_label_artifact_dir,
            "previous_label_artifact_dir": previous_label_artifact_dir,
            "mt5_log_path": mt5_log_path,
            "tick_input_path": tick_input_path,
            "m1_input_path": m1_input_path,
            "m5_input_path": m5_input_path,
            "protected_report_path": protected_report_path,
        },
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(momentum_rows, output_dir / "momentum_break_threshold_comparison.csv")
    _write_csv(early_rows, output_dir / "early_close_helpful_threshold_comparison.csv")
    _write_csv(stability["year"], output_dir / "protection_label_stability_by_year.csv")
    _write_csv(stability["side"], output_dir / "protection_label_stability_by_side.csv")
    _write_csv(stability["horizon"], output_dir / "protection_label_stability_by_horizon.csv")
    _write_csv(stability["source"], output_dir / "protection_label_stability_by_source.csv")
    (output_dir / "trade_protection_threshold_refinement_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "trade_protection_threshold_refinement_summary.md").write_text(summary_markdown(report), encoding="utf-8")
    (output_dir / "protection_threshold_candidate_assessment.md").write_text(assessment_markdown(report), encoding="utf-8")
    return report


def load_labeled_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Tick-level labeled dataset not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def momentum_candidates() -> list[MomentumCandidate]:
    return [
        MomentumCandidate("current_momentum_break", direction_max=-0.25, unrealized_max=0.0),
        MomentumCandidate("stricter_unrealized", direction_max=-0.25, unrealized_max=-0.50),
        MomentumCandidate("stronger_direction", direction_max=-0.50, unrealized_max=0.0),
        MomentumCandidate("require_mae_025atr", direction_max=-0.25, unrealized_max=0.0, mae_atr_min=0.25),
        MomentumCandidate("require_mae_050atr", direction_max=-0.25, unrealized_max=0.0, mae_atr_min=0.50),
        MomentumCandidate("early_horizon_60s", direction_max=-0.25, unrealized_max=0.0, max_horizon_seconds=60),
        MomentumCandidate("mid_horizon_120s", direction_max=-0.25, unrealized_max=0.0, max_horizon_seconds=120),
        MomentumCandidate("low_speed_break", direction_max=-0.25, unrealized_max=0.0, speed_max=30.0),
        MomentumCandidate("balanced_break", direction_max=-0.25, unrealized_max=-0.25, mae_atr_min=0.25),
    ]


def evaluate_momentum_candidate(
    candidate: MomentumCandidate,
    rows: list[dict[str, Any]],
    config: ThresholdRefinementConfig,
) -> dict[str, Any]:
    triggered = [row for row in rows if candidate_triggers(candidate, row)]
    helpful = [row for row in triggered if _float(row.get("label_early_close_delta_vs_base")) >= config.helpful_materiality]
    harmful = [row for row in triggered if _float(row.get("label_early_close_delta_vs_base")) <= -config.harmful_materiality]
    deltas = [_float(row.get("label_early_close_delta_vs_base")) for row in triggered]
    base = {
        "candidate": candidate.name,
        "direction_max": candidate.direction_max,
        "unrealized_max": candidate.unrealized_max,
        "mae_atr_min": candidate.mae_atr_min,
        "speed_max": "" if candidate.speed_max is None else candidate.speed_max,
        "max_horizon_seconds": "" if candidate.max_horizon_seconds is None else candidate.max_horizon_seconds,
        "rows": len(rows),
        "triggered_snapshots": len(triggered),
        "trigger_rate_pct": _pct(len(triggered), len(rows)),
        "helpful_count": len(helpful),
        "harmful_count": len(harmful),
        "helpful_precision_pct": _pct(len(helpful), len(triggered)),
        "harmful_rate_pct": _pct(len(harmful), len(triggered)),
        "avg_delta_pnl": _round(mean(deltas)) if deltas else "",
        "median_delta_pnl": _round(median(deltas)) if deltas else "",
        "sum_delta_pnl": _round(sum(deltas)) if deltas else "",
        "years_with_triggers": len({str(row.get("setup_day", ""))[:4] for row in triggered}),
        "buy_triggered": sum(1 for row in triggered if row.get("side") == "buy"),
        "sell_triggered": sum(1 for row in triggered if row.get("side") == "sell"),
        "tick_triggered": sum(1 for row in triggered if row.get("data_source") == "tick"),
        "m1_triggered": sum(1 for row in triggered if row.get("data_source") == "m1_fallback"),
    }
    subgroups = subgroup_quality(candidate.name, [candidate], rows, config)
    side_rates = [row["helpful_precision_pct"] for row in subgroups if row["group"] == "side" and row["triggered_snapshots"]]
    year_rates = [row["helpful_precision_pct"] for row in subgroups if row["group"] == "year" and row["triggered_snapshots"]]
    decision_year_rows = [
        row
        for row in subgroups
        if row["group"] == "year" and str(row["value"]) in config.decision_years and int(row["triggered_snapshots"]) > 0
    ]
    base["side_precision_gap_pct"] = _round(max(side_rates) - min(side_rates)) if len(side_rates) >= 2 else ""
    base["min_year_helpful_precision_pct"] = _round(min(year_rates)) if year_rates else ""
    base["decision_years_with_triggers"] = len(decision_year_rows)
    base["min_decision_year_triggered"] = min((int(row["triggered_snapshots"]) for row in decision_year_rows), default=0)
    base["min_decision_year_helpful_precision_pct"] = _round(
        min((float(row["helpful_precision_pct"]) for row in decision_year_rows), default=0.0)
    )
    base["score"] = candidate_score(base, config)
    base["assessment"] = candidate_assessment(base, config)
    return base


def evaluate_early_close_helpful_thresholds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for materiality in (0.25, 0.50, 1.00):
        label_name = f"early_close_helpful_delta_ge_{materiality:.2f}"
        labels = [int(_float(row.get("label_early_close_delta_vs_base")) >= materiality) for row in rows]
        positive = sum(labels)
        top_features = top_feature_auc(rows, labels)
        result.append(
            {
                "label_variant": label_name,
                "materiality_pnl": materiality,
                "rows": len(rows),
                "positive_rows": positive,
                "positive_pct": _pct(positive, len(rows)),
                "best_feature": top_features[0]["feature"] if top_features else "",
                "best_auc": top_features[0]["auc"] if top_features else "",
                "best_auc_edge_abs": top_features[0]["auc_edge_abs"] if top_features else "",
                "top_features": "|".join(f"{item['feature']}:{item['auc_edge_abs']}" for item in top_features[:5]),
            }
        )
    return result


def stability_rows(
    candidate_name: str,
    candidates: list[MomentumCandidate],
    rows: list[dict[str, Any]],
    field: str,
    config: ThresholdRefinementConfig,
) -> list[dict[str, Any]]:
    return subgroup_quality(candidate_name, candidates, rows, config, field)


def subgroup_quality(
    candidate_name: str,
    candidates: list[MomentumCandidate],
    rows: list[dict[str, Any]],
    config: ThresholdRefinementConfig,
    only_field: str | None = None,
) -> list[dict[str, Any]]:
    candidate = next((item for item in candidates if item.name == candidate_name), None)
    if candidate is None:
        return []
    fields = [only_field] if only_field else ["year", "side"]
    result: list[dict[str, Any]] = []
    for field in fields:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[group_value(row, field)].append(row)
        for value, group_rows in sorted(grouped.items()):
            triggered = [row for row in group_rows if candidate_triggers(candidate, row)]
            helpful = [row for row in triggered if _float(row.get("label_early_close_delta_vs_base")) >= config.helpful_materiality]
            harmful = [row for row in triggered if _float(row.get("label_early_close_delta_vs_base")) <= -config.harmful_materiality]
            deltas = [_float(row.get("label_early_close_delta_vs_base")) for row in triggered]
            result.append(
                {
                    "candidate": candidate.name,
                    "group": field,
                    "value": value,
                    "rows": len(group_rows),
                    "triggered_snapshots": len(triggered),
                    "trigger_rate_pct": _pct(len(triggered), len(group_rows)),
                    "helpful_count": len(helpful),
                    "harmful_count": len(harmful),
                    "helpful_precision_pct": _pct(len(helpful), len(triggered)),
                    "harmful_rate_pct": _pct(len(harmful), len(triggered)),
                    "avg_delta_pnl": _round(mean(deltas)) if deltas else "",
                    "sum_delta_pnl": _round(sum(deltas)) if deltas else "",
                }
            )
    return result


def candidate_triggers(candidate: MomentumCandidate, row: dict[str, Any]) -> bool:
    if candidate.max_horizon_seconds is not None and _float(row.get("snapshot_seconds_after_fill")) > candidate.max_horizon_seconds:
        return False
    if candidate.speed_max is not None and _float(row.get("post_entry_speed_points_per_min")) > candidate.speed_max:
        return False
    return (
        _float(row.get("m1_m5_close_direction_proxy")) <= candidate.direction_max
        and _float(row.get("unrealized_pnl_so_far")) <= candidate.unrealized_max
        and _float(row.get("atr_relative_mae")) >= candidate.mae_atr_min
    )


def top_feature_auc(rows: list[dict[str, Any]], labels: list[int]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for feature in NUMERIC_FEATURES:
        values: list[float] = []
        y: list[int] = []
        for row, label in zip(rows, labels, strict=False):
            parsed = _to_float(row.get(feature))
            if parsed is None or not math.isfinite(parsed):
                continue
            values.append(float(parsed))
            y.append(int(label))
        if not values or len(set(y)) < 2:
            continue
        auc = auc_score(values, y)
        corr = point_biserial(values, y)
        result.append(
            {
                "feature": feature,
                "auc": _round(auc),
                "auc_edge_abs": _round(abs(auc - 0.5)),
                "point_biserial_corr": "" if corr is None else _round(corr),
            }
        )
    result.sort(key=lambda row: _float(row["auc_edge_abs"]), reverse=True)
    return result


def auc_score(values: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(values, labels, strict=False), key=lambda item: item[0])
    positives = sum(label == 1 for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        rank_sum += avg_rank * sum(label == 1 for _, label in pairs[index:end])
        index = end
    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return max(auc, 1.0 - auc)


def point_biserial(values: list[float], labels: list[int]) -> float | None:
    if not values or len(set(labels)) < 2:
        return None
    mean_x = mean(values)
    mean_y = mean(labels)
    denom_x = math.sqrt(sum((value - mean_x) ** 2 for value in values))
    denom_y = math.sqrt(sum((label - mean_y) ** 2 for label in labels))
    if denom_x == 0.0 or denom_y == 0.0:
        return None
    return sum((value - mean_x) * (label - mean_y) for value, label in zip(values, labels, strict=False)) / (denom_x * denom_y)


def group_value(row: dict[str, Any], field: str) -> str:
    if field == "year":
        return str(row.get("setup_day", ""))[:4]
    if field == "snapshot_seconds_after_fill":
        return str(row.get("snapshot_seconds_after_fill", ""))
    return str(row.get(field, ""))


def candidate_score(row: dict[str, Any], config: ThresholdRefinementConfig) -> float:
    triggered = _float(row.get("triggered_snapshots"))
    precision = _float(row.get("helpful_precision_pct"))
    harmful = _float(row.get("harmful_rate_pct"))
    avg_delta = _float(row.get("avg_delta_pnl"))
    support_bonus = min(triggered / max(config.min_triggered_snapshots, 1), 2.0) * 5.0
    return _round(avg_delta * 10.0 + precision - harmful * 0.6 + support_bonus)


def candidate_assessment(row: dict[str, Any], config: ThresholdRefinementConfig) -> str:
    if _float(row.get("triggered_snapshots")) < config.min_triggered_snapshots:
        return "too_little_support"
    if _float(row.get("helpful_precision_pct")) < config.min_helpful_precision:
        return "weak_precision"
    if _float(row.get("harmful_rate_pct")) > config.max_harmful_rate:
        return "too_many_harmful_closes"
    side_gap = _to_float(row.get("side_precision_gap_pct"))
    if side_gap is not None and side_gap > config.max_side_precision_gap:
        return "side_concentrated"
    years = int(_float(row.get("years_with_triggers")))
    if years < 2:
        return "year_concentrated"
    if int(_float(row.get("decision_years_with_triggers"))) < len(config.decision_years):
        return "decision_year_missing"
    if int(_float(row.get("min_decision_year_triggered"))) < config.min_triggered_snapshots:
        return "decision_year_low_support"
    return "candidate_watchlist"


def build_report(
    *,
    rows: list[dict[str, Any]],
    analysis_rows: list[dict[str, Any]],
    momentum_rows: list[dict[str, Any]],
    early_rows: list[dict[str, Any]],
    stability: dict[str, list[dict[str, Any]]],
    config: ThresholdRefinementConfig,
    paths: dict[str, Path],
) -> dict[str, Any]:
    best = momentum_rows[0] if momentum_rows else {}
    watchlist = [row for row in momentum_rows if row.get("assessment") == "candidate_watchlist"]
    recommended = watchlist[0] if watchlist else best
    stable = bool(watchlist)
    return {
        "sprint": "tick_level_trade_protection_threshold_refinement_v1",
        "source_paths": {key: str(value) for key, value in paths.items()},
        "inputs_found": {key: value.exists() for key, value in paths.items()},
        "config": asdict(config),
        "dataset": {
            "all_rows": len(rows),
            "analysis_rows": len(analysis_rows),
            "unique_trades": len({row.get("event_key") for row in analysis_rows}),
            "years": sorted({str(row.get("setup_day", ""))[:4] for row in analysis_rows}),
            "tick_rows": sum(1 for row in analysis_rows if row.get("data_source") == "tick"),
            "m1_rows": sum(1 for row in analysis_rows if row.get("data_source") == "m1_fallback"),
        },
        "momentum_break_threshold_comparison": momentum_rows,
        "early_close_helpful_threshold_comparison": early_rows,
        "stability": stability,
        "decision": {
            "best_scoring_candidate": best.get("candidate", ""),
            "recommended_candidate": recommended.get("candidate", ""),
            "recommended_definition": candidate_definition(recommended),
            "label_momentum_break_true_still_best_decision_label": True,
            "stable_enough_for_supervised_trade_protection_model_v1": stable,
            "ready_for_ppo_or_rl": False,
            "recommended_next_sprint": "supervised_trade_protection_model_v1" if stable else "tick_level_trade_protection_more_history_or_walkforward_v1",
            "reason": (
                "A momentum-break threshold candidate has enough support, precision, and 2024-2025 subgroup stability for a small supervised model sprint."
                if stable
                else "Momentum-break remains useful, but no refined definition is stable enough across subgroups to justify supervised training yet."
            ),
        },
        "limitations": [
            "The analysis uses the tick-level labeled artifact, not a new tick scan.",
            "2021-2022 still have no tick-level early snapshots, so temporal robustness remains incomplete.",
            "Threshold refinement is intentionally small and interpretable, not a grid search.",
            "No PPO, RL, or final supervised model is trained in this sprint.",
        ],
    }


def candidate_definition(row: dict[str, Any]) -> str:
    if not row:
        return ""
    parts = [
        f"m1_m5_close_direction_proxy <= {row.get('direction_max')}",
        f"unrealized_pnl_so_far <= {row.get('unrealized_max')}",
    ]
    if _float(row.get("mae_atr_min")) > 0:
        parts.append(f"atr_relative_mae >= {row.get('mae_atr_min')}")
    if str(row.get("speed_max", "")):
        parts.append(f"post_entry_speed_points_per_min <= {row.get('speed_max')}")
    if str(row.get("max_horizon_seconds", "")):
        parts.append(f"snapshot_seconds_after_fill <= {row.get('max_horizon_seconds')}")
    parts.append("early_close_delta_vs_base >= 0.50 defines true positive")
    return " AND ".join(parts)


def summary_markdown(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    decision = report["decision"]
    momentum_rows = report["momentum_break_threshold_comparison"]
    early_rows = report["early_close_helpful_threshold_comparison"]
    best = momentum_rows[0] if momentum_rows else {}

    lines = [
        "# Tick-Level Trade Protection Threshold Refinement v1",
        "",
        "## Executive Summary",
        "",
        "- Scope: refine early post-entry deterioration labels using the existing tick-level labeled artifact.",
        f"- Analysis rows: {dataset['analysis_rows']} snapshots across {dataset['unique_trades']} trades.",
        f"- Years included: {', '.join(dataset['years'])}.",
        f"- Snapshot sources: {dataset['tick_rows']} tick rows and {dataset['m1_rows']} M1 fallback rows.",
        f"- Best scoring momentum candidate: `{decision['best_scoring_candidate']}`.",
        f"- Recommended candidate: `{decision['recommended_candidate']}`.",
        f"- Recommended definition: `{decision['recommended_definition']}`.",
        f"- Stable enough for supervised sprint: `{decision['stable_enough_for_supervised_trade_protection_model_v1']}`.",
        f"- Recommended next sprint: `{decision['recommended_next_sprint']}`.",
        "",
        "## Best Momentum-Break Candidate",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- Triggered snapshots: {best['triggered_snapshots']} ({best['trigger_rate_pct']}%).",
                f"- Helpful precision: {best['helpful_precision_pct']}%.",
                f"- Harmful rate: {best['harmful_rate_pct']}%.",
                f"- Average close-early delta vs base: {best['avg_delta_pnl']}.",
                f"- Assessment: `{best['assessment']}`.",
                "",
            ]
        )
    else:
        lines.extend(["- No momentum-break candidates were evaluated.", ""])

    lines.extend(
        [
            "## Early-Close Label Variants",
            "",
            "| Variant | Positives | Positive % | Best feature | Best AUC edge |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for row in early_rows:
        lines.append(
            f"| {row['label_variant']} | {row['positive_rows']} | {row['positive_pct']} | "
            f"{row['best_feature']} | {row['best_auc_edge_abs']} |"
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
            "## Decision",
            "",
            decision["reason"],
            "",
        ]
    )
    return "\n".join(lines)


def assessment_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    momentum_rows = report["momentum_break_threshold_comparison"]
    stability = report["stability"]
    lines = [
        "# Protection Threshold Candidate Assessment",
        "",
        "## Recommended Definition",
        "",
        f"`{decision['recommended_definition']}`",
        "",
        "## Momentum Candidate Ranking",
        "",
        "| Candidate | Score | Triggered | Helpful % | Harmful % | Avg Delta | Assessment |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in momentum_rows:
        lines.append(
            f"| {row['candidate']} | {row['score']} | {row['triggered_snapshots']} | "
            f"{row['helpful_precision_pct']} | {row['harmful_rate_pct']} | {row['avg_delta_pnl']} | {row['assessment']} |"
        )

    lines.extend(
        [
            "",
            "## Stability Snapshot",
            "",
            "### By Year",
            "",
            "| Year | Triggered | Helpful % | Harmful % | Avg Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in stability.get("year", []):
        lines.append(
            f"| {row['value']} | {row['triggered_snapshots']} | {row['helpful_precision_pct']} | "
            f"{row['harmful_rate_pct']} | {row['avg_delta_pnl']} |"
        )

    lines.extend(
        [
            "",
            "### By Side",
            "",
            "| Side | Triggered | Helpful % | Harmful % | Avg Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in stability.get("side", []):
        lines.append(
            f"| {row['value']} | {row['triggered_snapshots']} | {row['helpful_precision_pct']} | "
            f"{row['harmful_rate_pct']} | {row['avg_delta_pnl']} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Stable enough for supervised model sprint: `{decision['stable_enough_for_supervised_trade_protection_model_v1']}`.",
            f"- Ready for RL/PPO: `{decision['ready_for_ppo_or_rl']}`.",
            f"- Next sprint: `{decision['recommended_next_sprint']}`.",
            f"- Rationale: {decision['reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def _pct(numerator: int | float, denominator: int | float) -> float:
    denominator_float = float(denominator)
    if denominator_float == 0.0:
        return 0.0
    return _round(float(numerator) / denominator_float * 100.0)


def _float(value: Any) -> float:
    parsed = _to_float(value)
    if parsed is None or not math.isfinite(parsed):
        return 0.0
    return float(parsed)


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "sprint": report["sprint"],
        "analysis_rows": report["dataset"]["analysis_rows"],
        "unique_trades": report["dataset"]["unique_trades"],
        "recommended_candidate": report["decision"]["recommended_candidate"],
        "stable_enough_for_supervised_trade_protection_model_v1": report["decision"][
            "stable_enough_for_supervised_trade_protection_model_v1"
        ],
        "recommended_next_sprint": report["decision"]["recommended_next_sprint"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
