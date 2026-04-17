"""Reward, observation, seed, and checkpoint diagnostics for Azir PPO."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable

from hybrid_quant.env.azir_event_env import (
    ACTION_SKIP,
    ACTION_TAKE,
    OBSERVATION_FIELDS,
    AzirEventReplayEnvironment,
    AzirReplayEvent,
    build_azir_event_replay_dataset,
)
from hybrid_quant.risk.azir_state import AzirRiskConfig

from .train_ppo_skip_take import (
    AzirPPOConfig,
    evaluate_event_policy,
    load_config as load_ppo_config,
    random_valid_policy,
    skip_all_policy,
    split_events,
    take_all_valid_policy,
    take_only_sell_valid_policy,
    trade_metrics,
)


PolicyFn = Callable[[AzirEventReplayEnvironment, dict[str, Any]], int]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Azir PPO skip/take behavior.")
    parser.add_argument("--mt5-log-path", required=True)
    parser.add_argument("--protected-report-path", required=True)
    parser.add_argument("--ppo-output-dir", default="artifacts/azir-ppo-skip-take-v1")
    parser.add_argument("--config-path", default="configs/experiments/azir_ppo_skip_take.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbol", default="XAUUSD-STD")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_reward_observation_diagnostics(
        mt5_log_path=Path(args.mt5_log_path),
        protected_report_path=Path(args.protected_report_path),
        ppo_output_dir=Path(args.ppo_output_dir),
        config_path=Path(args.config_path) if args.config_path else None,
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_reward_observation_diagnostics(
    *,
    mt5_log_path: Path,
    protected_report_path: Path,
    ppo_output_dir: Path,
    output_dir: Path,
    config_path: Path | None = None,
    symbol: str = "XAUUSD-STD",
    ppo_config: AzirPPOConfig | None = None,
) -> dict[str, Any]:
    config = ppo_config or load_ppo_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    events = build_azir_event_replay_dataset(
        mt5_log_path=mt5_log_path,
        protected_report_path=protected_report_path,
        symbol=symbol,
        risk_config=AzirRiskConfig(),
    )
    splits = split_events(events, config)
    atr_threshold = atr_high_threshold(splits["train"])

    seed_rows, trace_rows = evaluate_saved_ppo_seeds(
        ppo_output_dir=ppo_output_dir,
        test_events=splits["test"],
        seeds=config.seeds,
    )
    checkpoint_rows = evaluate_checkpoints(
        ppo_output_dir=ppo_output_dir,
        validation_events=splits["validation"],
        test_events=splits["test"],
        seeds=config.seeds,
    )
    control_rows = evaluate_strong_controls(splits["test"], config, atr_threshold)
    reward_rows = reward_component_analysis(splits, ppo_output_dir, config.seeds, atr_threshold)
    observation_rows = observation_diagnostics(splits)
    scale_rows = feature_scale_report(observation_rows)
    decision = diagnostics_decision(control_rows, seed_rows, checkpoint_rows, reward_rows, observation_rows)

    _write_csv(seed_rows, output_dir / "ppo_seed_diagnostics.csv")
    _write_csv(reward_rows, output_dir / "reward_component_analysis.csv")
    _write_csv(observation_rows, output_dir / "observation_diagnostics.csv")
    _write_csv(scale_rows, output_dir / "feature_scale_report.csv")
    _write_csv(checkpoint_rows, output_dir / "checkpoint_comparison.csv")
    _write_csv(control_rows + seed_rows, output_dir / "heuristic_controls_vs_ppo.csv")
    _write_csv(trace_rows, output_dir / "ppo_policy_trace.csv")

    report = {
        "sprint": "reward_observation_diagnostics_for_ppo_v1",
        "benchmark": "baseline_azir_protected_economic_v1",
        "environment": "AzirEventReplayEnvironment",
        "actions": {"0": "skip", "1": "take"},
        "mt5_log_path": str(mt5_log_path),
        "protected_report_path": str(protected_report_path),
        "ppo_output_dir": str(ppo_output_dir),
        "splits": {name: split_metadata(items) for name, items in splits.items()},
        "atr_high_threshold_from_train_sell_valid": atr_threshold,
        "ppo_seed_diagnostics": seed_rows,
        "strong_controls": control_rows,
        "reward_component_analysis": reward_rows,
        "observation_diagnostics": observation_rows,
        "feature_scale_report": scale_rows,
        "checkpoint_comparison": checkpoint_rows,
        "decision": decision,
    }
    (output_dir / "diagnostics_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "diagnostics_summary.md").write_text(diagnostics_summary_markdown(report), encoding="utf-8")
    return report


def evaluate_saved_ppo_seeds(
    *,
    ppo_output_dir: Path,
    test_events: list[AzirReplayEvent],
    seeds: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from stable_baselines3 import PPO

    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for seed in seeds:
        model_path = ppo_output_dir / "ppo_models" / f"seed-{seed}" / "final_model.zip"
        if not model_path.exists():
            rows.append({"policy": f"ppo_seed_{seed}", "seed": seed, "model_status": "missing", "model_path": str(model_path)})
            continue
        model = PPO.load(str(model_path), env=make_env(test_events))
        row, trace = trace_sb3_model(f"ppo_seed_{seed}", model, test_events, seed=seed, split="test", model_label="final")
        row["model_path"] = str(model_path)
        row["model_status"] = "loaded"
        rows.append(row)
        trace_rows.extend(trace)
    return rows, trace_rows


def evaluate_checkpoints(
    *,
    ppo_output_dir: Path,
    validation_events: list[AzirReplayEvent],
    test_events: list[AzirReplayEvent],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    from stable_baselines3 import PPO

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for label, model_path, step in checkpoint_candidates(ppo_output_dir, seed):
            if not model_path.exists():
                continue
            validation_model = PPO.load(str(model_path), env=make_env(validation_events))
            validation_row, _ = trace_sb3_model(
                f"ppo_seed_{seed}_{label}", validation_model, validation_events, seed=seed, split="validation", model_label=label
            )
            test_model = PPO.load(str(model_path), env=make_env(test_events))
            test_row, _ = trace_sb3_model(
                f"ppo_seed_{seed}_{label}", test_model, test_events, seed=seed, split="test", model_label=label
            )
            rows.append(
                {
                    "seed": seed,
                    "checkpoint_label": label,
                    "training_step": step,
                    "model_path": str(model_path),
                    "validation_net_pnl": validation_row["net_pnl"],
                    "validation_profit_factor": validation_row["profit_factor"],
                    "validation_expectancy": validation_row["expectancy"],
                    "validation_max_drawdown": validation_row["max_drawdown"],
                    "validation_trades_taken": validation_row["trades_taken"],
                    "validation_take_rate": validation_row["take_rate"],
                    "test_net_pnl": test_row["net_pnl"],
                    "test_profit_factor": test_row["profit_factor"],
                    "test_expectancy": test_row["expectancy"],
                    "test_max_drawdown": test_row["max_drawdown"],
                    "test_trades_taken": test_row["trades_taken"],
                    "test_take_rate": test_row["take_rate"],
                    "is_best_by_validation_net_pnl": False,
                }
            )
    annotate_best_validation_checkpoints(rows)
    return rows


def checkpoint_candidates(ppo_output_dir: Path, seed: int) -> list[tuple[str, Path, int]]:
    candidates: list[tuple[str, Path, int]] = [
        ("final", ppo_output_dir / "ppo_models" / f"seed-{seed}" / "final_model.zip", 0),
        ("best_validation", ppo_output_dir / "ppo_models" / f"seed-{seed}" / "best_model" / "best_model.zip", 0),
    ]
    checkpoint_dir = ppo_output_dir / "ppo_checkpoints" / f"seed-{seed}"
    for path in sorted(checkpoint_dir.glob("*.zip"), key=_checkpoint_step):
        candidates.append((f"checkpoint_{_checkpoint_step(path)}", path, _checkpoint_step(path)))
    deduped: list[tuple[str, Path, int]] = []
    seen: set[Path] = set()
    for label, path, step in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append((label, path, step))
    return deduped


def annotate_best_validation_checkpoints(rows: list[dict[str, Any]]) -> None:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["seed"])].append(row)
    for seed_rows in grouped.values():
        best = max(seed_rows, key=lambda row: (_float(row["validation_net_pnl"]), -_float(row["validation_max_drawdown"])))
        best["is_best_by_validation_net_pnl"] = True


def trace_sb3_model(
    policy_name: str,
    model: Any,
    events: list[AzirReplayEvent],
    *,
    seed: int,
    split: str,
    model_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    def _policy(env: AzirEventReplayEnvironment, info: dict[str, Any]) -> int:
        observation = env._observation(env.current_event(), env._evaluate_risk(env.current_event()))
        action, _ = model.predict(observation, deterministic=True)
        return int(action)

    return trace_policy(policy_name, events, _policy, seed=seed, split=split, model_label=model_label)


def trace_policy(
    policy_name: str,
    events: list[AzirReplayEvent],
    policy: PolicyFn,
    *,
    seed: int,
    split: str,
    model_label: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = make_env(events)
    _, info = env.reset(seed=seed)
    done = False
    pnl_values: list[float] = []
    rewards: list[float] = []
    action_counts: Counter[int] = Counter()
    effect_counts: Counter[str] = Counter()
    by_year: dict[str, list[float]] = defaultdict(list)
    trace_rows: list[dict[str, Any]] = []
    take_attempts = valid_take_actions = blocked_attempts = invalid_attempts = 0
    buy_takes = sell_takes = 0

    while not done:
        event = env.current_event()
        risk = env._evaluate_risk(event)
        current_info = env._info(
            event,
            risk,
            action_effect="pre_decision",
            valid_actions=env.valid_actions(event, risk),
        )
        action = int(policy(env, current_info))
        action_counts[action] += 1
        if action == ACTION_TAKE:
            take_attempts += 1
        _, reward, done, _, info = env.step(action)
        rewards.append(float(reward))
        effect = str(info["action_effect"])
        effect_counts[effect] += 1
        pnl = 0.0
        if effect == "take":
            valid_take_actions += 1
            if info.get("has_protected_fill"):
                pnl = float(info["reward_breakdown"]["protected_net_pnl"])
                pnl_values.append(pnl)
                by_year[event.setup_day[:4]].append(pnl)
                if _is_sell_setup(event):
                    sell_takes += 1
                else:
                    buy_takes += 1
        elif effect == "risk_blocked_take_transformed_to_skip":
            blocked_attempts += 1
        elif effect == "invalid_take_no_azir_order_transformed_to_skip":
            invalid_attempts += 1
        trace_rows.append(
            {
                "policy": policy_name,
                "seed": seed,
                "split": split,
                "model_label": model_label,
                "setup_day": event.setup_day,
                "year": event.setup_day[:4],
                "action": action,
                "action_effect": effect,
                "side": _side(event),
                "order_placed": event.order_placed,
                "has_protected_fill": event.has_protected_fill,
                "protected_net_pnl": pnl,
                "reward": _round(float(reward)),
            }
        )

    metrics = trade_metrics(pnl_values)
    take_rate = take_attempts / len(events) if events else 0.0
    row = {
        "policy": policy_name,
        "seed": seed,
        "split": split,
        "model_label": model_label,
        "events": len(events),
        "total_reward": _round(sum(rewards)),
        "mean_reward": _round(mean(rewards)) if rewards else 0.0,
        "reward_std": _round(pstdev(rewards)) if len(rewards) > 1 else 0.0,
        "trades_taken": len(pnl_values),
        "take_attempts": take_attempts,
        "valid_take_actions": valid_take_actions,
        "take_rate": _round(take_rate),
        "valid_take_rate": _round(valid_take_actions / len(events)) if events else 0.0,
        "skip_rate": _round(action_counts[ACTION_SKIP] / len(events)) if events else 0.0,
        "blocked_attempts": blocked_attempts,
        "invalid_attempts": invalid_attempts,
        "buy_takes": buy_takes,
        "sell_takes": sell_takes,
        "action_skip_count": action_counts[ACTION_SKIP],
        "action_take_count": action_counts[ACTION_TAKE],
        "effect_skip_count": effect_counts["skip"],
        "effect_take_count": effect_counts["take"],
        "effect_risk_blocked_count": effect_counts["risk_blocked_take_transformed_to_skip"],
        "effect_invalid_take_count": effect_counts["invalid_take_no_azir_order_transformed_to_skip"],
        "collapsed_to_skip": take_rate < 0.10,
        "too_aggressive": take_rate > 0.85,
        "yearly_net_pnl": json.dumps({year: _round(sum(values)) for year, values in sorted(by_year.items())}),
        **metrics,
    }
    return row, trace_rows


def evaluate_strong_controls(
    test_events: list[AzirReplayEvent],
    config: AzirPPOConfig,
    atr_threshold: float,
) -> list[dict[str, Any]]:
    rows = [
        evaluate_event_policy("skip_all", test_events, skip_all_policy, seed=0),
        evaluate_event_policy("take_all_valid", test_events, take_all_valid_policy, seed=0),
        evaluate_event_policy("take_only_sell_valid", test_events, take_only_sell_valid_policy, seed=0),
        evaluate_event_policy(
            "take_sell_valid_atr_high",
            test_events,
            lambda env, info, rng: take_sell_valid_atr_high_policy(env, info, atr_threshold),
            seed=0,
        ),
    ]
    random_rows = [
        evaluate_event_policy("random_valid", test_events, random_valid_policy, seed=seed)
        for seed in config.random_valid_seeds
    ]
    rows.extend(random_rows)
    rows.append(_aggregate_control_rows("random_valid_mean", random_rows))
    return rows


def take_sell_valid_atr_high_policy(env: AzirEventReplayEnvironment, info: dict[str, Any], atr_threshold: float) -> int:
    event = env.current_event()
    if ACTION_TAKE in info.get("valid_actions", ()) and _is_sell_setup(event) and _float(event.setup.get("atr_points")) >= atr_threshold:
        return ACTION_TAKE
    return ACTION_SKIP


def atr_high_threshold(train_events: list[AzirReplayEvent]) -> float:
    values = [
        _float(event.setup.get("atr_points"))
        for event in train_events
        if event.order_placed and _is_sell_setup(event) and math.isfinite(_float(event.setup.get("atr_points")))
    ]
    return _round(median(values)) if values else 0.0


def reward_component_analysis(
    splits: dict[str, list[AzirReplayEvent]],
    ppo_output_dir: Path,
    seeds: tuple[int, ...],
    atr_threshold: float,
) -> list[dict[str, Any]]:
    from stable_baselines3 import PPO

    policies: list[tuple[str, PolicyFn]] = [
        ("take_all_valid", lambda env, info: ACTION_TAKE if ACTION_TAKE in info.get("valid_actions", ()) else ACTION_SKIP),
        ("take_only_sell_valid", lambda env, info: ACTION_TAKE if ACTION_TAKE in info.get("valid_actions", ()) and _is_sell_setup(env.current_event()) else ACTION_SKIP),
        ("take_sell_valid_atr_high", lambda env, info: take_sell_valid_atr_high_policy(env, info, atr_threshold)),
        ("take_every_event_diagnostic", lambda env, info: ACTION_TAKE),
    ]
    rows: list[dict[str, Any]] = []
    for split_name, events in splits.items():
        for policy_name, policy in policies:
            rows.append(_reward_components_for_policy(split_name, policy_name, events, policy))
        if split_name == "test":
            for seed in seeds:
                model_path = ppo_output_dir / "ppo_models" / f"seed-{seed}" / "final_model.zip"
                if model_path.exists():
                    model = PPO.load(str(model_path), env=make_env(events))
                    rows.append(
                        _reward_components_for_policy(
                            split_name,
                            f"ppo_seed_{seed}",
                            events,
                            lambda env, info, model=model: _predict_action(model, env),
                        )
                    )
    return rows


def _reward_components_for_policy(
    split_name: str,
    policy_name: str,
    events: list[AzirReplayEvent],
    policy: PolicyFn,
) -> dict[str, Any]:
    env = make_env(events)
    _, info = env.reset(seed=0)
    done = False
    components: dict[str, list[float]] = defaultdict(list)
    effects: Counter[str] = Counter()
    while not done:
        event = env.current_event()
        risk = env._evaluate_risk(event)
        current_info = env._info(
            event,
            risk,
            action_effect="pre_decision",
            valid_actions=env.valid_actions(event, risk),
        )
        action = int(policy(env, current_info))
        _, reward, done, _, info = env.step(action)
        effects[str(info["action_effect"])] += 1
        breakdown = info.get("reward_breakdown", {})
        components["reward"].append(float(reward))
        for key in ["protected_net_pnl", "drawdown_penalty", "risk_tension_penalty", "invalid_action_penalty"]:
            components[key].append(float(breakdown.get(key, 0.0) or 0.0))
    rewards = components["reward"]
    pnl = components["protected_net_pnl"]
    penalties = (
        sum(components["drawdown_penalty"])
        + sum(components["risk_tension_penalty"])
        + sum(components["invalid_action_penalty"])
    )
    return {
        "split": split_name,
        "policy": policy_name,
        "events": len(events),
        "take_events": effects["take"],
        "skip_events": effects["skip"],
        "risk_blocked_events": effects["risk_blocked_take_transformed_to_skip"],
        "invalid_take_events": effects["invalid_take_no_azir_order_transformed_to_skip"],
        "nonzero_reward_rate": _round(len([value for value in rewards if abs(value) > 1e-12]) / len(events)) if events else 0.0,
        "reward_sum": _round(sum(rewards)),
        "reward_mean": _round(mean(rewards)) if rewards else 0.0,
        "reward_abs_mean": _round(mean(abs(value) for value in rewards)) if rewards else 0.0,
        "protected_net_pnl_sum": _round(sum(pnl)),
        "protected_net_pnl_abs_mean": _round(mean(abs(value) for value in pnl)) if pnl else 0.0,
        "drawdown_penalty_sum": _round(sum(components["drawdown_penalty"])),
        "risk_tension_penalty_sum": _round(sum(components["risk_tension_penalty"])),
        "invalid_action_penalty_sum": _round(sum(components["invalid_action_penalty"])),
        "penalty_to_abs_pnl_ratio": _round(penalties / max(sum(abs(value) for value in pnl), 1e-12)),
    }


def observation_diagnostics(splits: dict[str, list[AzirReplayEvent]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, events in splits.items():
        obs_rows = observation_rows_for_split(events)
        pnl_values = [row["future_protected_net_pnl"] for row in obs_rows]
        fill_values = [row["future_has_protected_fill"] for row in obs_rows]
        for feature in OBSERVATION_FIELDS:
            values = [row[feature] for row in obs_rows]
            rows.append(
                {
                    "split": split_name,
                    "feature": feature,
                    "count": len(values),
                    "missing_count": 0,
                    "non_finite_count": len([value for value in values if not math.isfinite(value)]),
                    "min": _round(min(values)) if values else 0.0,
                    "max": _round(max(values)) if values else 0.0,
                    "mean": _round(mean(values)) if values else 0.0,
                    "std": _round(pstdev(values)) if len(values) > 1 else 0.0,
                    "abs_max": _round(max(abs(value) for value in values)) if values else 0.0,
                    "p95_abs": _round(percentile([abs(value) for value in values], 0.95)) if values else 0.0,
                    "zero_rate": _round(len([value for value in values if abs(value) <= 1e-12]) / len(values)) if values else 0.0,
                    "corr_with_future_pnl": _round(pearson(values, pnl_values)),
                    "corr_with_future_fill": _round(pearson(values, fill_values)),
                    "scale_flag": scale_flag(values),
                }
            )
    return rows


def observation_rows_for_split(events: list[AzirReplayEvent]) -> list[dict[str, Any]]:
    env = make_env(events)
    _, info = env.reset(seed=0)
    done = False
    rows: list[dict[str, Any]] = []
    while not done:
        event = env.current_event()
        risk = env._evaluate_risk(event)
        obs = env._observation_dict(event, risk)
        rows.append(
            {
                **obs,
                "setup_day": event.setup_day,
                "future_protected_net_pnl": event.protected_net_pnl if event.has_protected_fill else 0.0,
                "future_has_protected_fill": 1.0 if event.has_protected_fill else 0.0,
                "side": _side(event),
            }
        )
        _, _, done, _, info = env.step(ACTION_SKIP)
    return rows


def feature_scale_report(observation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observation_rows:
        by_feature[str(row["feature"])].append(row)
    for feature, feature_rows in sorted(by_feature.items()):
        max_abs = max(_float(row.get("abs_max")) for row in feature_rows)
        max_std = max(_float(row.get("std")) for row in feature_rows)
        max_corr = max(abs(_float(row.get("corr_with_future_pnl"))) for row in feature_rows)
        rows.append(
            {
                "feature": feature,
                "max_abs_across_splits": _round(max_abs),
                "max_std_across_splits": _round(max_std),
                "max_abs_corr_with_future_pnl": _round(max_corr),
                "recommended_action": recommended_feature_action(max_abs=max_abs, max_std=max_std, max_corr=max_corr),
            }
        )
    return rows


def diagnostics_decision(
    control_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
    reward_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    take_all = _row_by_policy(control_rows, "take_all_valid")
    take_sell = _row_by_policy(control_rows, "take_only_sell_valid")
    ppo_best = max(seed_rows, key=lambda row: _float(row.get("net_pnl"))) if seed_rows else {}
    net_values = [_float(row.get("net_pnl")) for row in seed_rows]
    ppo_mean_net = mean(net_values) if net_values else 0.0
    ppo_std_net = pstdev(net_values) if len(net_values) > 1 else 0.0
    seeds_beating_take_all = len([row for row in seed_rows if _float(row.get("net_pnl")) > _float(take_all.get("net_pnl"))])
    skip_collapses = [row["seed"] for row in seed_rows if row.get("collapsed_to_skip") is True]
    aggressive_seeds = [row["seed"] for row in seed_rows if row.get("too_aggressive") is True]
    best_checkpoint = max(checkpoint_rows, key=lambda row: _float(row.get("test_net_pnl"))) if checkpoint_rows else {}
    checkpoint_improves_final = bool(
        best_checkpoint
        and _float(best_checkpoint.get("test_net_pnl")) > _float(ppo_best.get("net_pnl"))
        and best_checkpoint.get("checkpoint_label") != "final"
    )
    large_scale_features = sorted(
        {
            row["feature"]
            for row in observation_rows
            if row.get("scale_flag") in {"large_raw_price_scale", "very_large_numeric_scale"}
        }
    )
    penalty_rows = [row for row in reward_rows if row.get("split") == "test" and row.get("policy") == "take_all_valid"]
    penalty_to_abs_pnl = _float(penalty_rows[0].get("penalty_to_abs_pnl_ratio")) if penalty_rows else 0.0
    ppo_promising = bool(
        ppo_best
        and _float(ppo_best.get("net_pnl")) > _float(take_all.get("net_pnl"))
        and _float(ppo_best.get("max_drawdown")) < _float(take_all.get("max_drawdown"))
    )
    ppo_robust_enough = bool(ppo_mean_net > _float(take_all.get("net_pnl")) and seeds_beating_take_all >= 2 and not skip_collapses)
    primary_issue = "training_variance_and_model_selection" if ppo_promising and not ppo_robust_enough else "ppo_not_beating_controls"
    if large_scale_features:
        primary_issue += "_plus_unscaled_observations"
    if ppo_robust_enough:
        reason = "PPO is stable enough across seeds and beats take_all_valid on mean test performance."
    elif ppo_promising:
        reason = "PPO has a promising best seed, but seed variance and/or unscaled observations make the result too unstable for robust validation."
    else:
        reason = "PPO does not beat the corrected take_all_valid control; seed variance and unscaled observations should be fixed before another PPO claim."
    return {
        "ppo_basic_still_promising": ppo_promising,
        "ppo_robust_enough_for_robust_validation": ppo_robust_enough,
        "ppo_best_seed": ppo_best.get("seed"),
        "ppo_best_seed_net_pnl": ppo_best.get("net_pnl"),
        "ppo_mean_net_pnl": _round(ppo_mean_net),
        "ppo_seed_net_pnl_std": _round(ppo_std_net),
        "seeds_beating_take_all_valid": seeds_beating_take_all,
        "take_all_valid_net_pnl": take_all.get("net_pnl"),
        "take_only_sell_valid_net_pnl": take_sell.get("net_pnl"),
        "collapsed_to_skip_seeds": skip_collapses,
        "too_aggressive_seeds": aggressive_seeds,
        "checkpoint_improves_best_final_on_test": checkpoint_improves_final,
        "best_checkpoint_by_test": {
            "seed": best_checkpoint.get("seed"),
            "label": best_checkpoint.get("checkpoint_label"),
            "test_net_pnl": best_checkpoint.get("test_net_pnl"),
            "validation_net_pnl": best_checkpoint.get("validation_net_pnl"),
        },
        "large_scale_observation_features": large_scale_features,
        "reward_penalty_to_abs_pnl_ratio_take_all_valid_test": penalty_to_abs_pnl,
        "primary_issue": primary_issue,
        "recommended_next_sprint": "robust_validation_for_ppo_v1" if ppo_robust_enough else "reward_observation_adjustment_for_ppo_v1",
        "reason": reason,
    }


def diagnostics_summary_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    seed_lines = "\n".join(
        f"- seed `{row['seed']}`: net={row.get('net_pnl')}, DD={row.get('max_drawdown')}, "
        f"trades={row.get('trades_taken')}, take_rate={row.get('take_rate')}, collapsed={row.get('collapsed_to_skip')}"
        for row in report["ppo_seed_diagnostics"]
    )
    control_lines = "\n".join(
        f"- `{row['policy']}`: net={row.get('net_pnl')}, PF={row.get('profit_factor')}, "
        f"DD={row.get('max_drawdown')}, trades={row.get('trades_taken')}, take_rate={row.get('take_rate')}"
        for row in report["strong_controls"]
        if row["policy"] in {"take_all_valid", "take_only_sell_valid", "take_sell_valid_atr_high", "random_valid_mean"}
    )
    best_checkpoint = decision["best_checkpoint_by_test"]
    top_scale = [
        row
        for row in report["feature_scale_report"]
        if row["recommended_action"] != "keep_as_is"
    ][:8]
    scale_lines = "\n".join(
        f"- `{row['feature']}`: max_abs={row['max_abs_across_splits']}, action={row['recommended_action']}"
        for row in top_scale
    ) or "- No obvious large-scale feature issue detected."
    return (
        "# Azir PPO Reward/Observation Diagnostics\n\n"
        "## Executive Summary\n\n"
        f"- PPO basic still promising: {decision['ppo_basic_still_promising']}.\n"
        f"- PPO robust enough for robust validation: {decision['ppo_robust_enough_for_robust_validation']}.\n"
        f"- Primary issue: `{decision['primary_issue']}`.\n"
        f"- Recommended next sprint: `{decision['recommended_next_sprint']}`.\n"
        f"- Reason: {decision['reason']}\n\n"
        "## Seed Diagnostics\n\n"
        f"{seed_lines}\n\n"
        "## Strong Controls\n\n"
        f"{control_lines}\n\n"
        "## Checkpoint Diagnostics\n\n"
        f"- Best checkpoint by test net PnL: seed={best_checkpoint.get('seed')}, label={best_checkpoint.get('label')}, "
        f"test_net={best_checkpoint.get('test_net_pnl')}, validation_net={best_checkpoint.get('validation_net_pnl')}.\n"
        f"- Checkpoint improves best final on test: {decision['checkpoint_improves_best_final_on_test']}.\n"
        f"- Checkpoints evaluated: {len(report['checkpoint_comparison'])}.\n\n"
        "## Reward Diagnostics\n\n"
        f"- Penalty-to-absolute-PnL ratio for `take_all_valid` on test: {decision['reward_penalty_to_abs_pnl_ratio_take_all_valid_test']}.\n"
        "- Reward is mostly driven by protected PnL; skip has zero reward, so sparse positive feedback can still let a seed collapse toward skip.\n\n"
        "## Observation Diagnostics\n\n"
        f"{scale_lines}\n\n"
        "## Decision\n\n"
        "- Do not add actions yet.\n"
        "- Do not move to PPO robust validation until seed instability is addressed or explained.\n"
        "- Next work should focus on observation scaling/normalization and reward calibration, then rerun the same controls.\n"
    )


def make_env(events: list[AzirReplayEvent]) -> AzirEventReplayEnvironment:
    return AzirEventReplayEnvironment(list(events))


def split_metadata(events: list[AzirReplayEvent]) -> dict[str, Any]:
    return {
        "events": len(events),
        "start_day": events[0].setup_day if events else None,
        "end_day": events[-1].setup_day if events else None,
        "protected_fill_events": len([event for event in events if event.has_protected_fill]),
        "order_placed_events": len([event for event in events if event.order_placed]),
        "sell_order_events": len([event for event in events if _is_sell_setup(event)]),
        "buy_order_events": len([event for event in events if _is_buy_setup(event)]),
    }


def _predict_action(model: Any, env: AzirEventReplayEnvironment) -> int:
    observation = env._observation(env.current_event(), env._evaluate_risk(env.current_event()))
    action, _ = model.predict(observation, deterministic=True)
    return int(action)


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    return int(match.group(1)) if match else 0


def _row_by_policy(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("policy") == policy), {})


def _is_sell_setup(event: AzirReplayEvent) -> bool:
    return str(event.setup.get("sell_order_placed", "")).strip().lower() == "true"


def _is_buy_setup(event: AzirReplayEvent) -> bool:
    return str(event.setup.get("buy_order_placed", "")).strip().lower() == "true"


def _side(event: AzirReplayEvent) -> str:
    if _is_sell_setup(event):
        return "sell"
    if _is_buy_setup(event):
        return "buy"
    return "none"


def _aggregate_control_rows(policy_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"policy": policy_name, "seed": "mean"}
    numeric_keys = [
        "events",
        "total_reward",
        "trades_taken",
        "take_attempts",
        "valid_take_actions",
        "take_rate",
        "valid_take_rate",
        "blocked_attempts",
        "invalid_attempts",
        "buy_takes",
        "sell_takes",
        "net_pnl",
        "profit_factor",
        "expectancy",
        "win_rate",
        "average_win",
        "average_loss",
        "payoff",
        "max_drawdown",
        "max_consecutive_losses",
    ]
    result: dict[str, Any] = {"policy": policy_name, "seed": "mean"}
    for key in numeric_keys:
        values = [_float(row.get(key)) for row in rows if row.get(key) not in {None, ""}]
        result[key] = _round(mean(values)) if values else ""
    return result


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * quantile))))
    return sorted_values[index]


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((value - x_mean) ** 2 for value in xs)
    y_var = sum((value - y_mean) ** 2 for value in ys)
    if x_var <= 0.0 or y_var <= 0.0:
        return 0.0
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    return cov / math.sqrt(x_var * y_var)


def scale_flag(values: list[float]) -> str:
    if not values:
        return "empty"
    abs_max = max(abs(value) for value in values)
    std = pstdev(values) if len(values) > 1 else 0.0
    if abs_max >= 10000.0:
        return "very_large_numeric_scale"
    if abs_max >= 1000.0:
        return "large_raw_price_scale"
    if std <= 1e-12 and abs_max > 0.0:
        return "constant_nonzero"
    if abs_max <= 1.0:
        return "bounded_or_binary"
    return "moderate_scale"


def recommended_feature_action(*, max_abs: float, max_std: float, max_corr: float) -> str:
    if max_abs >= 1000.0:
        return "normalize_or_express_relative_to_price"
    if max_std <= 1e-12:
        return "consider_dropping_if_constant"
    if max_corr < 0.01 and max_abs <= 1.0:
        return "low_signal_binary_review"
    return "keep_as_is"


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


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "sprint": report["sprint"],
        "splits": report["splits"],
        "decision": report["decision"],
        "artifacts": [
            "ppo_seed_diagnostics.csv",
            "reward_component_analysis.csv",
            "observation_diagnostics.csv",
            "feature_scale_report.csv",
            "checkpoint_comparison.csv",
            "heuristic_controls_vs_ppo.csv",
            "diagnostics_summary.md",
            "diagnostics_report.json",
        ],
    }


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
