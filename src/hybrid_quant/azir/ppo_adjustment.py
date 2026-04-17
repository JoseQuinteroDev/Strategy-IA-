"""Reward/observation adjustment sprint for the Azir skip/take PPO."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

from hybrid_quant.env.azir_event_env import (
    ACTION_SKIP,
    ACTION_TAKE,
    OBSERVATION_FIELDS_V1,
    OBSERVATION_FIELDS_V2,
    AzirEventReplayEnvironment,
    AzirReplayEvent,
    build_azir_event_replay_dataset,
)
from hybrid_quant.risk.azir_state import AzirRiskConfig

from .train_ppo_skip_take import (
    AzirPPOConfig,
    load_config,
    make_env,
    run_train_first_ppo_skip_take,
    split_events,
)


PolicyFn = Callable[[AzirEventReplayEnvironment, dict[str, Any]], int]


RAW_ABSOLUTE_V1_FIELDS = {
    "swing_high",
    "swing_low",
    "buy_entry",
    "sell_entry",
    "ema20",
    "atr",
    "daily_realized_pnl",
    "daily_drawdown_abs",
    "total_drawdown_abs",
    "remaining_daily_loss",
}


TRANSFORM_NOTES = [
    {
        "before": "day_of_week, month",
        "after": "day_of_week_sin/cos, month_sin/cos",
        "reason": "Cyclical calendar encoding avoids ordinal jumps such as December to January.",
    },
    {
        "before": "swing_high, swing_low, buy_entry, sell_entry, ema20",
        "after": "distance-to-EMA and swing width normalized by ATR",
        "reason": "Absolute XAUUSD price levels are not stationarity-friendly for MLP PPO.",
    },
    {
        "before": "pending_distance_points, spread_points, prev_close_vs_ema20_points",
        "after": "ATR-normalized ratios",
        "reason": "Point distances become comparable across volatility regimes.",
    },
    {
        "before": "atr_points",
        "after": "atr_points_scaled",
        "reason": "Keeps volatility level information while reducing raw magnitude.",
    },
    {
        "before": "daily_realized_pnl, drawdowns, remaining_daily_loss",
        "after": "equity/loss-limit ratios",
        "reason": "Risk state is expressed on a stable account scale.",
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adjust Azir PPO observations/reward and retrain skip/take PPO.")
    parser.add_argument("--mt5-log-path", required=True)
    parser.add_argument("--protected-report-path", required=True)
    parser.add_argument("--previous-report-path", default="artifacts/azir-ppo-skip-take-v1/ppo_eval_report.json")
    parser.add_argument("--config-path", default="configs/experiments/azir_ppo_skip_take_adjusted_v1.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbol", default="XAUUSD-STD")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_reward_observation_adjustment(
        mt5_log_path=Path(args.mt5_log_path),
        protected_report_path=Path(args.protected_report_path),
        previous_report_path=Path(args.previous_report_path),
        config_path=Path(args.config_path),
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_reward_observation_adjustment(
    *,
    mt5_log_path: Path,
    protected_report_path: Path,
    previous_report_path: Path,
    config_path: Path,
    output_dir: Path,
    symbol: str = "XAUUSD-STD",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    events = build_azir_event_replay_dataset(
        mt5_log_path=mt5_log_path,
        protected_report_path=protected_report_path,
        symbol=symbol,
        risk_config=AzirRiskConfig(),
    )
    splits = split_events(events, config)

    observation_rows = observation_adjustment_report(splits, config)
    reward_rows = reward_component_adjustment_report(splits, config)
    _write_csv(observation_rows, output_dir / "observation_adjustment_report.csv")
    _write_csv(reward_rows, output_dir / "reward_component_adjustment_report.csv")
    (output_dir / "feature_transform_summary.md").write_text(
        feature_transform_markdown(observation_rows),
        encoding="utf-8",
    )
    (output_dir / "reward_adjustment_report.md").write_text(
        reward_adjustment_markdown(config, reward_rows),
        encoding="utf-8",
    )

    training_report = run_train_first_ppo_skip_take(
        mt5_log_path=mt5_log_path,
        protected_report_path=protected_report_path,
        output_dir=output_dir,
        symbol=symbol,
        config=config,
    )
    seed_rows = seed_stability_before_after(previous_report_path, training_report)
    decision = adjusted_decision(previous_report_path, training_report)
    training_report["sprint"] = "reward_observation_adjustment_for_ppo_v1"
    training_report["observation_adjustment"] = {
        "before_schema": list(OBSERVATION_FIELDS_V1),
        "after_schema": list(OBSERVATION_FIELDS_V2),
        "before_feature_count": len(OBSERVATION_FIELDS_V1),
        "after_feature_count": len(OBSERVATION_FIELDS_V2),
        "transform_notes": TRANSFORM_NOTES,
    }
    training_report["reward_adjustment"] = {
        "mode": config.reward_mode,
        "drawdown_penalty_weight": config.drawdown_penalty_weight,
        "risk_tension_penalty_weight": config.risk_tension_penalty_weight,
        "reward_pnl_scale": config.reward_pnl_scale,
        "skip_opportunity_cost_weight": config.skip_opportunity_cost_weight,
        "skip_opportunity_cost_cap": config.skip_opportunity_cost_cap,
    }
    training_report["adjustment_decision"] = decision

    _write_csv(seed_rows, output_dir / "seed_stability_before_after.csv")
    source_comparison = output_dir / "controls_vs_ppo.csv"
    if source_comparison.exists():
        shutil.copyfile(source_comparison, output_dir / "controls_vs_adjusted_ppo.csv")
    (output_dir / "adjusted_ppo_eval_report.json").write_text(
        json.dumps(training_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "adjusted_ppo_summary.md").write_text(
        adjusted_summary_markdown(training_report),
        encoding="utf-8",
    )
    return training_report


def observation_adjustment_report(splits: dict[str, list[AzirReplayEvent]], config: AzirPPOConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for version, label in (("v1", "before"), ("v2", "after")):
        version_config = replace(config, observation_version=version)
        for split_name, events in splits.items():
            feature_values: dict[str, list[float]] = {}
            for values in iter_observations(events, version_config):
                for feature, value in values.items():
                    feature_values.setdefault(feature, []).append(value)
            for feature, values in feature_values.items():
                finite_values = [value for value in values if math.isfinite(value)]
                feature_min = min(finite_values) if finite_values else 0.0
                feature_max = max(finite_values) if finite_values else 0.0
                abs_max = max(abs(feature_min), abs(feature_max))
                feature_mean = mean(finite_values) if finite_values else 0.0
                feature_std = pstdev(finite_values) if len(finite_values) > 1 else 0.0
                rows.append(
                    {
                        "phase": label,
                        "observation_version": version,
                        "split": split_name,
                        "feature": feature,
                        "count": len(values),
                        "non_finite_count": len(values) - len(finite_values),
                        "min": _round(feature_min),
                        "max": _round(feature_max),
                        "abs_max": _round(abs_max),
                        "mean": _round(feature_mean),
                        "std": _round(feature_std),
                        "zero_pct": _round(sum(1 for value in finite_values if abs(value) <= 1e-12) / len(finite_values)) if finite_values else 0.0,
                        "scale_flag": scale_flag(abs_max),
                        "raw_absolute_price_or_money": feature in RAW_ABSOLUTE_V1_FIELDS,
                    }
                )
    return rows


def iter_observations(events: list[AzirReplayEvent], config: AzirPPOConfig) -> list[dict[str, float]]:
    env = make_env(events, config)
    env.reset(seed=123)
    done = False
    rows: list[dict[str, float]] = []
    while not done:
        event = env.current_event()
        risk = env._evaluate_risk(event)
        rows.append(env._observation_dict(event, risk))
        action = ACTION_TAKE if ACTION_TAKE in env.valid_actions(event, risk) else ACTION_SKIP
        _, _, done, _, _ = env.step(action)
    return rows


def reward_component_adjustment_report(splits: dict[str, list[AzirReplayEvent]], config: AzirPPOConfig) -> list[dict[str, Any]]:
    before_config = replace(
        config,
        observation_version="v1",
        reward_mode="protected_net_pnl_minus_risk_penalties",
        drawdown_penalty_weight=0.05,
        reward_pnl_scale=1.0,
        skip_opportunity_cost_weight=0.0,
        skip_opportunity_cost_cap=1.0,
    )
    scenarios = [("before_v1_reward", before_config), ("after_v2_reward", config)]
    policies: list[tuple[str, PolicyFn]] = [
        ("skip_all", lambda env, info: ACTION_SKIP),
        ("take_all_valid", lambda env, info: ACTION_TAKE if ACTION_TAKE in info.get("valid_actions", ()) else ACTION_SKIP),
    ]
    rows: list[dict[str, Any]] = []
    for scenario, scenario_config in scenarios:
        for split_name, events in splits.items():
            for policy_name, policy in policies:
                rows.append(_reward_components_for_policy(scenario, split_name, policy_name, events, scenario_config, policy))
    return rows


def _reward_components_for_policy(
    scenario: str,
    split_name: str,
    policy_name: str,
    events: list[AzirReplayEvent],
    config: AzirPPOConfig,
    policy: PolicyFn,
) -> dict[str, Any]:
    env = make_env(events, config)
    _, info = env.reset(seed=123)
    done = False
    rewards: list[float] = []
    components: dict[str, list[float]] = {
        "protected_net_pnl": [],
        "protected_reward_pnl": [],
        "drawdown_penalty": [],
        "risk_tension_penalty": [],
        "invalid_action_penalty": [],
        "skip_opportunity_cost": [],
    }
    effect_counts: dict[str, int] = {}
    while not done:
        event = env.current_event()
        risk = env._evaluate_risk(event)
        info = env._info(event, risk, action_effect="pre_decision", valid_actions=env.valid_actions(event, risk))
        action = int(policy(env, info))
        _, reward, done, _, step_info = env.step(action)
        rewards.append(float(reward))
        effect = str(step_info["action_effect"])
        effect_counts[effect] = effect_counts.get(effect, 0) + 1
        breakdown = step_info.get("reward_breakdown", {})
        for key in components:
            components[key].append(_float(breakdown.get(key)))
    return {
        "scenario": scenario,
        "split": split_name,
        "policy": policy_name,
        "events": len(events),
        "total_reward": _round(sum(rewards)),
        "mean_reward": _round(mean(rewards)) if rewards else 0.0,
        "non_zero_reward_pct": _round(sum(1 for reward in rewards if abs(reward) > 1e-12) / len(rewards)) if rewards else 0.0,
        "take_effect_count": effect_counts.get("take", 0),
        "skip_effect_count": effect_counts.get("skip", 0),
        "risk_blocked_count": effect_counts.get("risk_blocked_take_transformed_to_skip", 0),
        "invalid_take_count": effect_counts.get("invalid_take_no_azir_order_transformed_to_skip", 0),
        **{f"sum_{key}": _round(sum(values)) for key, values in components.items()},
        **{f"mean_{key}": _round(mean(values)) if values else 0.0 for key, values in components.items()},
    }


def seed_stability_before_after(previous_report_path: Path, adjusted_report: dict[str, Any]) -> list[dict[str, Any]]:
    previous = _load_json(previous_report_path)
    rows: list[dict[str, Any]] = []
    rows.extend(_seed_rows_for_report("before", previous))
    rows.extend(_seed_rows_for_report("after", adjusted_report))
    rows.extend(_seed_summary_rows("before", previous))
    rows.extend(_seed_summary_rows("after", adjusted_report))
    return rows


def _seed_rows_for_report(label: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in report.get("ppo_seeds", []):
        rows.append(
            {
                "phase": label,
                "row_type": "seed",
                "seed": row.get("seed"),
                "net_pnl": row.get("net_pnl"),
                "profit_factor": row.get("profit_factor"),
                "expectancy": row.get("expectancy"),
                "max_drawdown": row.get("max_drawdown"),
                "trades_taken": row.get("trades_taken"),
                "take_rate": row.get("take_rate"),
                "collapsed_to_skip": _float(row.get("take_rate")) < 0.10,
            }
        )
    return rows


def _seed_summary_rows(label: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    seed_rows = list(report.get("ppo_seeds", []))
    if not seed_rows:
        return []
    net_values = [_float(row.get("net_pnl")) for row in seed_rows]
    take_all = next((row for row in report.get("controls", []) if row.get("policy") == "take_all_valid"), {})
    take_all_net = _float(take_all.get("net_pnl"))
    return [
        {
            "phase": label,
            "row_type": "summary",
            "seed": "mean",
            "net_pnl": _round(mean(net_values)),
            "net_pnl_std": _round(pstdev(net_values)) if len(net_values) > 1 else 0.0,
            "best_net_pnl": _round(max(net_values)),
            "worst_net_pnl": _round(min(net_values)),
            "take_all_valid_net_pnl": _round(take_all_net),
            "seeds_beating_take_all_valid": sum(1 for value in net_values if value > take_all_net),
            "collapsed_seed_count": sum(1 for row in seed_rows if _float(row.get("take_rate")) < 0.10),
        }
    ]


def adjusted_decision(previous_report_path: Path, adjusted_report: dict[str, Any]) -> dict[str, Any]:
    previous = _load_json(previous_report_path)
    before_summary = _seed_summary_rows("before", previous)[0] if _seed_summary_rows("before", previous) else {}
    after_summary = _seed_summary_rows("after", adjusted_report)[0] if _seed_summary_rows("after", adjusted_report) else {}
    controls = adjusted_report.get("controls", [])
    take_all = next((row for row in controls if row.get("policy") == "take_all_valid"), {})
    ppo_mean = next((row for row in adjusted_report.get("comparison", []) if row.get("policy") == "ppo_mean"), {})
    best_ppo = next((row for row in adjusted_report.get("comparison", []) if row.get("policy") == "ppo_best_seed"), {})
    after_std = _float(after_summary.get("net_pnl_std"))
    before_std = _float(before_summary.get("net_pnl_std"))
    take_all_net = _float(take_all.get("net_pnl"))
    ppo_mean_net = _float(ppo_mean.get("net_pnl"))
    best_ppo_net = _float(best_ppo.get("net_pnl"))
    seeds_beating = int(after_summary.get("seeds_beating_take_all_valid") or 0)
    clear_value = ppo_mean_net > take_all_net and seeds_beating >= 2
    return {
        "take_all_valid_net_pnl": _round(take_all_net),
        "adjusted_ppo_mean_net_pnl": _round(ppo_mean_net),
        "adjusted_ppo_best_seed_net_pnl": _round(best_ppo_net),
        "adjusted_seeds_beating_take_all_valid": seeds_beating,
        "before_seed_net_pnl_std": _round(before_std),
        "after_seed_net_pnl_std": _round(after_std),
        "seed_variance_improved": after_std < before_std if before_std > 0 else False,
        "ppo_adjusted_adds_clear_value": clear_value,
        "ppo_adjusted_beats_take_all_valid_on_mean": ppo_mean_net > take_all_net,
        "recommended_next_sprint": "robust_validation_for_ppo_v1" if clear_value else "reward_observation_adjustment_v2_or_policy_regularization",
        "reason": (
            "Adjusted PPO mean beats take_all_valid and at least two seeds agree; robustness validation is justified."
            if clear_value
            else "Adjusted PPO still does not beat the strong take_all_valid control in a stable way."
        ),
    }


def feature_transform_markdown(rows: list[dict[str, Any]]) -> str:
    before_large = _top_scale_rows(rows, "before")
    after_large = _top_scale_rows(rows, "after")
    notes = "\n".join(f"- `{item['before']}` -> `{item['after']}`: {item['reason']}" for item in TRANSFORM_NOTES)
    before_top = "\n".join(f"- `{row['feature']}` {row['split']}: abs_max={row['abs_max']} ({row['scale_flag']})" for row in before_large[:8])
    after_top = "\n".join(f"- `{row['feature']}` {row['split']}: abs_max={row['abs_max']} ({row['scale_flag']})" for row in after_large[:8])
    return (
        "# Azir PPO Feature Transform Summary\n\n"
        "## Schema Change\n\n"
        f"- V1 feature count: {len(OBSERVATION_FIELDS_V1)}.\n"
        f"- V2 feature count: {len(OBSERVATION_FIELDS_V2)}.\n"
        "- V2 keeps the action contract unchanged: `0=skip`, `1=take`.\n"
        "- The main change is scale hygiene, not adding future information.\n\n"
        "## Transformations\n\n"
        f"{notes}\n\n"
        "## Largest V1 Scales\n\n"
        f"{before_top or '- None.'}\n\n"
        "## Largest V2 Scales\n\n"
        f"{after_top or '- None.'}\n"
    )


def reward_adjustment_markdown(config: AzirPPOConfig, rows: list[dict[str, Any]]) -> str:
    sample = [row for row in rows if row["split"] == "test" and row["policy"] in {"skip_all", "take_all_valid"}]
    sample_lines = "\n".join(
        f"- `{row['scenario']}` / `{row['policy']}`: total_reward={row['total_reward']}, "
        f"non_zero_reward_pct={row['non_zero_reward_pct']}, skip_cost={row['sum_skip_opportunity_cost']}"
        for row in sample
    )
    return (
        "# Azir PPO Reward Adjustment Report\n\n"
        "## V2 Reward\n\n"
        f"- Mode: `{config.reward_mode}`.\n"
        f"- Protected PnL scale: `{config.reward_pnl_scale}`.\n"
        f"- Drawdown penalty weight: `{config.drawdown_penalty_weight}`.\n"
        f"- Risk tension penalty weight: `{config.risk_tension_penalty_weight}`.\n"
        f"- Skip opportunity cost: `{config.skip_opportunity_cost_weight}` capped at `{config.skip_opportunity_cost_cap}`.\n\n"
        "## Rationale\n\n"
        "- Keep reward interpretable and anchored to protected net PnL.\n"
        "- Reduce drawdown penalty weight slightly so rare losses do not dominate sparse trade feedback.\n"
        "- Add a small capped penalty only when skipping a valid setup that later had positive protected PnL; this discourages collapse-to-skip without rewarding blind aggression.\n\n"
        "## Test Split Component Snapshot\n\n"
        f"{sample_lines or '- No reward rows generated.'}\n"
    )


def adjusted_summary_markdown(report: dict[str, Any]) -> str:
    comparison = report.get("comparison", [])
    selected = [
        row for row in comparison
        if row.get("policy") in {"take_all_valid", "take_only_sell_valid", "random_valid_mean", "ppo_mean", "ppo_best_seed"}
    ]
    rows = "\n".join(
        f"- `{row['policy']}`: net={row.get('net_pnl')}, PF={row.get('profit_factor')}, "
        f"DD={row.get('max_drawdown')}, trades={row.get('trades_taken')}, take_rate={row.get('take_rate')}"
        for row in selected
    )
    decision = report["adjustment_decision"]
    return (
        "# Azir PPO Reward/Observation Adjustment Summary\n\n"
        "## Executive Summary\n\n"
        f"- Observation version: `{report['hparams']['observation_version']}`.\n"
        f"- Reward mode: `{report['hparams']['reward_mode']}`.\n"
        f"- PPO adjusted adds clear value: `{decision['ppo_adjusted_adds_clear_value']}`.\n"
        f"- Seed variance improved: `{decision['seed_variance_improved']}`.\n\n"
        "## Comparison\n\n"
        f"{rows}\n\n"
        "## Decision\n\n"
        f"- {decision['reason']}\n"
        f"- Recommended next sprint: `{decision['recommended_next_sprint']}`.\n"
    )


def scale_flag(abs_max: float) -> str:
    if abs_max > 1000.0:
        return "extreme_abs_scale"
    if abs_max > 100.0:
        return "large_abs_scale"
    if abs_max > 10.0:
        return "medium_abs_scale"
    return "unit_or_small_scale"


def _top_scale_rows(rows: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    split_rows = [row for row in rows if row["phase"] == phase and row["split"] == "train"]
    return sorted(split_rows, key=lambda row: _float(row.get("abs_max")), reverse=True)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "sprint": report["sprint"],
        "splits": report["splits"],
        "decision": report["adjustment_decision"],
        "output_metrics": {
            row["policy"]: {
                "net_pnl": row.get("net_pnl"),
                "profit_factor": row.get("profit_factor"),
                "max_drawdown": row.get("max_drawdown"),
                "trades_taken": row.get("trades_taken"),
                "take_rate": row.get("take_rate"),
            }
            for row in report["comparison"]
            if row["policy"] in {"take_all_valid", "take_only_sell_valid", "random_valid_mean", "ppo_mean", "ppo_best_seed"}
        },
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float:
    try:
        if value in {None, ""}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


if __name__ == "__main__":
    raise SystemExit(main())
