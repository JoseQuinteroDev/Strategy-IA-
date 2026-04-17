"""Train the first skip/take PPO policy on AzirEventReplayEnvironment."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

import yaml

from hybrid_quant.env.azir_event_env import (
    ACTION_SKIP,
    ACTION_TAKE,
    AzirEventReplayEnvironment,
    AzirEventRewardConfig,
    build_azir_event_replay_dataset,
)
from hybrid_quant.risk.azir_state import AzirRiskConfig


DEFAULT_CONFIG = {
    "benchmark": "baseline_azir_protected_economic_v1",
    "risk_engine": "risk_engine_azir_v1",
    "splits": {"train_ratio": 0.60, "validation_ratio": 0.20, "test_ratio": 0.20},
    "env": {"observation_version": "v1"},
    "reward": {
        "mode": "protected_net_pnl_minus_risk_penalties",
        "drawdown_penalty_weight": 0.05,
        "risk_tension_penalty_weight": 0.25,
        "risk_blocked_penalty": 0.25,
        "invalid_take_penalty": 0.05,
        "reward_pnl_scale": 1.0,
        "skip_opportunity_cost_weight": 0.0,
        "skip_opportunity_cost_cap": 1.0,
    },
    "ppo": {
        "policy": "MlpPolicy",
        "total_timesteps": 5000,
        "learning_rate": 0.0003,
        "n_steps": 64,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "eval_freq": 1000,
        "checkpoint_freq": 1000,
        "device": "auto",
        "verbose": 0,
        "seeds": [7, 11, 17],
    },
    "controls": {"random_valid_seeds": [101, 202, 303, 404, 505]},
}


PolicyFn = Callable[[AzirEventReplayEnvironment, dict[str, Any], random.Random], int]


@dataclass(frozen=True)
class AzirPPOConfig:
    train_ratio: float = 0.60
    validation_ratio: float = 0.20
    test_ratio: float = 0.20
    policy: str = "MlpPolicy"
    total_timesteps: int = 5000
    learning_rate: float = 0.0003
    n_steps: int = 64
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    clip_range: float = 0.2
    eval_freq: int = 1000
    checkpoint_freq: int = 1000
    device: str = "auto"
    verbose: int = 0
    seeds: tuple[int, ...] = (7, 11, 17)
    random_valid_seeds: tuple[int, ...] = (101, 202, 303, 404, 505)
    observation_version: str = "v1"
    reward_mode: str = "protected_net_pnl_minus_risk_penalties"
    drawdown_penalty_weight: float = 0.05
    risk_tension_penalty_weight: float = 0.25
    risk_blocked_penalty: float = 0.25
    invalid_take_penalty: float = 0.05
    reward_pnl_scale: float = 1.0
    skip_opportunity_cost_weight: float = 0.0
    skip_opportunity_cost_cap: float = 1.0

    def to_hparams(self) -> dict[str, Any]:
        return {
            "observation_version": self.observation_version,
            "reward_mode": self.reward_mode,
            "drawdown_penalty_weight": self.drawdown_penalty_weight,
            "risk_tension_penalty_weight": self.risk_tension_penalty_weight,
            "risk_blocked_penalty": self.risk_blocked_penalty,
            "invalid_take_penalty": self.invalid_take_penalty,
            "reward_pnl_scale": self.reward_pnl_scale,
            "skip_opportunity_cost_weight": self.skip_opportunity_cost_weight,
            "skip_opportunity_cost_cap": self.skip_opportunity_cost_cap,
            "policy": self.policy,
            "total_timesteps": self.total_timesteps,
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ent_coef": self.ent_coef,
            "clip_range": self.clip_range,
            "eval_freq": self.eval_freq,
            "checkpoint_freq": self.checkpoint_freq,
            "device": self.device,
            "verbose": self.verbose,
            "seeds": list(self.seeds),
        }

    def to_reward_config(self) -> AzirEventRewardConfig:
        return AzirEventRewardConfig(
            mode=self.reward_mode,
            drawdown_penalty_weight=self.drawdown_penalty_weight,
            risk_tension_penalty_weight=self.risk_tension_penalty_weight,
            risk_blocked_penalty=self.risk_blocked_penalty,
            invalid_take_penalty=self.invalid_take_penalty,
            reward_pnl_scale=self.reward_pnl_scale,
            skip_opportunity_cost_weight=self.skip_opportunity_cost_weight,
            skip_opportunity_cost_cap=self.skip_opportunity_cost_cap,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train first Azir PPO skip/take policy.")
    parser.add_argument("--mt5-log-path", required=True)
    parser.add_argument("--protected-report-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-path", default="")
    parser.add_argument("--symbol", default="XAUUSD-STD")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seeds", default="", help="Comma-separated PPO seeds override.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(Path(args.config_path) if args.config_path else None)
    if args.total_timesteps is not None:
        config = _replace_config(config, total_timesteps=args.total_timesteps)
    if args.seeds:
        config = _replace_config(config, seeds=tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip()))
    report = run_train_first_ppo_skip_take(
        mt5_log_path=Path(args.mt5_log_path),
        protected_report_path=Path(args.protected_report_path),
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
        config=config,
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def load_config(path: Path | None) -> AzirPPOConfig:
    payload = DEFAULT_CONFIG if path is None else _merge_dicts(DEFAULT_CONFIG, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    ppo = payload.get("ppo", {})
    splits = payload.get("splits", {})
    controls = payload.get("controls", {})
    env = payload.get("env", {})
    reward = payload.get("reward", {})
    return AzirPPOConfig(
        train_ratio=float(splits.get("train_ratio", 0.60)),
        validation_ratio=float(splits.get("validation_ratio", 0.20)),
        test_ratio=float(splits.get("test_ratio", 0.20)),
        policy=str(ppo.get("policy", "MlpPolicy")),
        total_timesteps=int(ppo.get("total_timesteps", 5000)),
        learning_rate=float(ppo.get("learning_rate", 0.0003)),
        n_steps=int(ppo.get("n_steps", 64)),
        batch_size=int(ppo.get("batch_size", 64)),
        gamma=float(ppo.get("gamma", 0.99)),
        gae_lambda=float(ppo.get("gae_lambda", 0.95)),
        ent_coef=float(ppo.get("ent_coef", 0.01)),
        clip_range=float(ppo.get("clip_range", 0.2)),
        eval_freq=int(ppo.get("eval_freq", 1000)),
        checkpoint_freq=int(ppo.get("checkpoint_freq", 1000)),
        device=str(ppo.get("device", "auto")),
        verbose=int(ppo.get("verbose", 0)),
        seeds=tuple(int(seed) for seed in ppo.get("seeds", [7, 11, 17])),
        random_valid_seeds=tuple(int(seed) for seed in controls.get("random_valid_seeds", [101, 202, 303, 404, 505])),
        observation_version=str(env.get("observation_version", "v1")),
        reward_mode=str(reward.get("mode", "protected_net_pnl_minus_risk_penalties")),
        drawdown_penalty_weight=float(reward.get("drawdown_penalty_weight", 0.05)),
        risk_tension_penalty_weight=float(reward.get("risk_tension_penalty_weight", 0.25)),
        risk_blocked_penalty=float(reward.get("risk_blocked_penalty", 0.25)),
        invalid_take_penalty=float(reward.get("invalid_take_penalty", 0.05)),
        reward_pnl_scale=float(reward.get("reward_pnl_scale", 1.0)),
        skip_opportunity_cost_weight=float(reward.get("skip_opportunity_cost_weight", 0.0)),
        skip_opportunity_cost_cap=float(reward.get("skip_opportunity_cost_cap", 1.0)),
    )


def run_train_first_ppo_skip_take(
    *,
    mt5_log_path: Path,
    protected_report_path: Path,
    output_dir: Path,
    symbol: str = "XAUUSD-STD",
    config: AzirPPOConfig | None = None,
) -> dict[str, Any]:
    config = config or AzirPPOConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    events = build_azir_event_replay_dataset(
        mt5_log_path=mt5_log_path,
        protected_report_path=protected_report_path,
        symbol=symbol,
        risk_config=AzirRiskConfig(),
    )
    splits = split_events(events, config)
    _write_yaml(config.to_hparams(), output_dir / "ppo_hparams.yaml")

    control_rows = evaluate_controls(splits["test"], config)
    seed_rows, ppo_policy_rows = train_and_evaluate_ppo(splits, config, output_dir)
    all_rows = control_rows + ppo_policy_rows
    ranking = sorted(all_rows, key=lambda row: (_float(row["net_pnl"]), _float(row["profit_factor"]), -_float(row["max_drawdown"])), reverse=True)
    behavior_rows = policy_behavior_summary(all_rows)
    report = {
        "sprint": "train_first_ppo_skip_take",
        "benchmark": "baseline_azir_protected_economic_v1",
        "environment": "AzirEventReplayEnvironment",
        "actions": {"0": "skip", "1": "take"},
        "mt5_log_path": str(mt5_log_path),
        "protected_report_path": str(protected_report_path),
        "splits": {name: split_metadata(items) for name, items in splits.items()},
        "hparams": config.to_hparams(),
        "controls": control_rows,
        "ppo_seeds": seed_rows,
        "comparison": all_rows,
        "policy_behavior": behavior_rows,
        "decision": decision_summary(control_rows, seed_rows),
    }
    _write_csv(all_rows, output_dir / "controls_vs_ppo.csv")
    _write_csv(seed_rows, output_dir / "seed_comparison.csv")
    _write_csv(behavior_rows, output_dir / "policy_behavior_summary.csv")
    (output_dir / "ppo_eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "ppo_training_summary.md").write_text(summary_markdown(report), encoding="utf-8")
    return report


def split_events(events: list[Any], config: AzirPPOConfig) -> dict[str, list[Any]]:
    total = len(events)
    train_end = max(1, int(total * config.train_ratio))
    validation_end = max(train_end + 1, int(total * (config.train_ratio + config.validation_ratio)))
    validation_end = min(validation_end, total - 1)
    return {
        "train": events[:train_end],
        "validation": events[train_end:validation_end],
        "test": events[validation_end:],
    }


def evaluate_controls(test_events: list[Any], config: AzirPPOConfig) -> list[dict[str, Any]]:
    rows = [
        evaluate_event_policy("skip_all", test_events, skip_all_policy, seed=0, config=config),
        evaluate_event_policy("take_all_valid", test_events, take_all_valid_policy, seed=0, config=config),
        evaluate_event_policy("take_only_sell_valid", test_events, take_only_sell_valid_policy, seed=0, config=config),
    ]
    random_rows = [
        evaluate_event_policy("random_valid", test_events, random_valid_policy, seed=seed, config=config)
        for seed in config.random_valid_seeds
    ]
    rows.extend(random_rows)
    rows.append(_aggregate_rows("random_valid_mean", random_rows))
    return rows


def train_and_evaluate_ppo(
    splits: dict[str, list[Any]],
    config: AzirPPOConfig,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

    checkpoint_root = output_dir / "ppo_checkpoints"
    model_root = output_dir / "ppo_models"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []

    for seed in config.seeds:
        seed_dir = model_root / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        train_env = make_env(splits["train"], config)
        eval_env = make_env(splits["validation"], config)
        checkpoint_dir = checkpoint_root / f"seed-{seed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks = CallbackList(
            [
                CheckpointCallback(
                    save_freq=max(1, config.checkpoint_freq),
                    save_path=str(checkpoint_dir),
                    name_prefix=f"ppo_azir_seed_{seed}",
                ),
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(seed_dir / "best_model"),
                    log_path=str(seed_dir / "eval_logs"),
                    eval_freq=max(1, config.eval_freq),
                    deterministic=True,
                    n_eval_episodes=1,
                ),
            ]
        )
        model = PPO(
            config.policy,
            train_env,
            seed=seed,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            ent_coef=config.ent_coef,
            clip_range=config.clip_range,
            verbose=config.verbose,
            device=config.device,
        )
        model.learn(total_timesteps=config.total_timesteps, callback=callbacks)
        final_model_path = seed_dir / "final_model"
        model.save(str(final_model_path))
        seed_report = evaluate_sb3_model(f"ppo_seed_{seed}", model, splits["test"], seed=seed, config=config)
        seed_report["model_path"] = str(final_model_path.with_suffix(".zip"))
        seed_report["checkpoint_dir"] = str(checkpoint_dir)
        seed_rows.append(seed_report)
        policy_rows.append(seed_report)

    policy_rows.append(_aggregate_rows("ppo_mean", seed_rows))
    best_seed = max(seed_rows, key=lambda row: (_float(row["net_pnl"]), _float(row["profit_factor"]))) if seed_rows else {}
    if best_seed:
        best_row = dict(best_seed)
        best_row["policy"] = "ppo_best_seed"
        policy_rows.append(best_row)
    return seed_rows, policy_rows


def make_env(events: list[Any], config: AzirPPOConfig | None = None) -> AzirEventReplayEnvironment:
    config = config or AzirPPOConfig()
    return AzirEventReplayEnvironment(
        list(events),
        observation_version=config.observation_version,
        reward_config=config.to_reward_config(),
    )


def evaluate_event_policy(
    policy_name: str,
    events: list[Any],
    policy: PolicyFn,
    *,
    seed: int,
    config: AzirPPOConfig | None = None,
) -> dict[str, Any]:
    env = make_env(events, config)
    observation, info = env.reset(seed=seed)
    rng = random.Random(seed)
    done = False
    total_reward = 0.0
    pnl_values: list[float] = []
    take_attempts = 0
    valid_take_actions = 0
    blocked_attempts = 0
    invalid_attempts = 0
    buy_takes = 0
    sell_takes = 0

    while not done:
        current_event = env.current_event()
        current_risk = env._evaluate_risk(current_event)
        current_info = env._info(
            current_event,
            current_risk,
            action_effect="pre_decision",
            valid_actions=env.valid_actions(current_event, current_risk),
        )
        action = int(policy(env, current_info, rng))
        if action == ACTION_TAKE:
            take_attempts += 1
        observation, reward, done, _, info = env.step(action)
        total_reward += float(reward)
        effect = str(info["action_effect"])
        if effect == "take":
            valid_take_actions += 1
            if info.get("has_protected_fill"):
                pnl = float(info["reward_breakdown"]["protected_net_pnl"])
                pnl_values.append(pnl)
                if _is_sell_setup(current_event):
                    sell_takes += 1
                else:
                    buy_takes += 1
        elif effect == "risk_blocked_take_transformed_to_skip":
            blocked_attempts += 1
        elif effect == "invalid_take_no_azir_order_transformed_to_skip":
            invalid_attempts += 1

    metrics = trade_metrics(pnl_values)
    return {
        "policy": policy_name,
        "seed": seed,
        "events": len(events),
        "total_reward": _round(total_reward),
        "trades_taken": len(pnl_values),
        "take_attempts": take_attempts,
        "valid_take_actions": valid_take_actions,
        "take_rate": _round(take_attempts / len(events)) if events else 0.0,
        "valid_take_rate": _round(valid_take_actions / len(events)) if events else 0.0,
        "blocked_attempts": blocked_attempts,
        "invalid_attempts": invalid_attempts,
        "buy_takes": buy_takes,
        "sell_takes": sell_takes,
        **metrics,
    }


def evaluate_sb3_model(policy_name: str, model: Any, events: list[Any], *, seed: int, config: AzirPPOConfig | None = None) -> dict[str, Any]:
    def _policy(env: AzirEventReplayEnvironment, info: dict[str, Any], rng: random.Random) -> int:
        observation = env._observation(env.current_event(), env._evaluate_risk(env.current_event()))
        action, _ = model.predict(observation, deterministic=True)
        return int(action)

    return evaluate_event_policy(policy_name, events, _policy, seed=seed, config=config)


def skip_all_policy(env: AzirEventReplayEnvironment, info: dict[str, Any], rng: random.Random) -> int:
    return ACTION_SKIP


def take_all_valid_policy(env: AzirEventReplayEnvironment, info: dict[str, Any], rng: random.Random) -> int:
    return ACTION_TAKE if ACTION_TAKE in info.get("valid_actions", ()) else ACTION_SKIP


def random_valid_policy(env: AzirEventReplayEnvironment, info: dict[str, Any], rng: random.Random) -> int:
    valid = tuple(info.get("valid_actions", (ACTION_SKIP,)))
    if ACTION_TAKE not in valid:
        return ACTION_SKIP
    return rng.choice([ACTION_SKIP, ACTION_TAKE])


def take_only_sell_valid_policy(env: AzirEventReplayEnvironment, info: dict[str, Any], rng: random.Random) -> int:
    event = env.current_event()
    if ACTION_TAKE in info.get("valid_actions", ()) and _is_sell_setup(event):
        return ACTION_TAKE
    return ACTION_SKIP


def trade_metrics(pnl_values: list[float]) -> dict[str, Any]:
    wins = [pnl for pnl in pnl_values if pnl > 0.0]
    losses = [pnl for pnl in pnl_values if pnl < 0.0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "net_pnl": _round(sum(pnl_values)),
        "profit_factor": _round(gross_profit / gross_loss) if gross_loss else (None if not gross_profit else float("inf")),
        "expectancy": _round(mean(pnl_values)) if pnl_values else 0.0,
        "win_rate": _round(len(wins) / len(pnl_values)) if pnl_values else 0.0,
        "average_win": _round(mean(wins)) if wins else 0.0,
        "average_loss": _round(mean(losses)) if losses else 0.0,
        "payoff": _round((mean(wins) / abs(mean(losses))) if wins and losses else 0.0),
        "max_drawdown": _round(max_drawdown(pnl_values)),
        "max_consecutive_losses": max_consecutive_losses(pnl_values),
    }


def policy_behavior_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "policy": row["policy"],
            "seed": row.get("seed", ""),
            "events": row["events"],
            "take_attempts": row["take_attempts"],
            "valid_take_actions": row["valid_take_actions"],
            "trades_taken": row["trades_taken"],
            "take_rate": row["take_rate"],
            "valid_take_rate": row["valid_take_rate"],
            "blocked_attempts": row["blocked_attempts"],
            "invalid_attempts": row["invalid_attempts"],
            "buy_takes": row["buy_takes"],
            "sell_takes": row["sell_takes"],
        }
        for row in rows
    ]


def decision_summary(control_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    take_all = next(row for row in control_rows if row["policy"] == "take_all_valid")
    best_ppo = max(seed_rows, key=lambda row: _float(row["net_pnl"])) if seed_rows else None
    ppo_mean = _aggregate_rows("ppo_mean", seed_rows) if seed_rows else {}
    best_beats_take_all = bool(best_ppo and _float(best_ppo["net_pnl"]) > _float(take_all["net_pnl"]))
    mean_beats_take_all = bool(ppo_mean and _float(ppo_mean["net_pnl"]) > _float(take_all["net_pnl"]))
    improves_dd = bool(best_ppo and _float(best_ppo["max_drawdown"]) < _float(take_all["max_drawdown"]))
    seeds_beating_take_all = len([row for row in seed_rows if _float(row["net_pnl"]) > _float(take_all["net_pnl"])])
    net_values = [_float(row["net_pnl"]) for row in seed_rows]
    net_std = _round(pstdev(net_values)) if len(net_values) > 1 else 0.0
    stable_enough = bool(seed_rows and seeds_beating_take_all >= max(2, (len(seed_rows) // 2) + 1))
    clear_value = bool(mean_beats_take_all and improves_dd and stable_enough)
    if best_beats_take_all:
        no_value_reason = (
            "Best PPO seed is promising, but PPO is not stable enough across seeds or mean performance does not beat "
            "take_all_valid. Diagnose reward/observation before robust validation."
        )
    else:
        no_value_reason = (
            "PPO does not beat take_all_valid on best-seed net PnL, and mean performance is lower. "
            "Diagnose reward/observation before robust validation."
        )
    return {
        "ppo_best_seed_beats_take_all_valid_on_net_pnl": best_beats_take_all,
        "ppo_mean_beats_take_all_valid_on_net_pnl": mean_beats_take_all,
        "ppo_best_seed_reduces_drawdown_vs_take_all_valid": improves_dd,
        "ppo_seeds_beating_take_all_valid": seeds_beating_take_all,
        "ppo_seed_count": len(seed_rows),
        "ppo_seed_net_pnl_std": net_std,
        "ppo_mean_net_pnl": ppo_mean.get("net_pnl"),
        "take_all_valid_net_pnl": take_all.get("net_pnl"),
        "ppo_adds_clear_value": clear_value,
        "ready_for_robust_validation_for_ppo_v1": clear_value,
        "recommended_next_sprint": "robust_validation_for_ppo_v1" if clear_value else "reward_observation_diagnostics_for_ppo_v1",
        "reason": (
            "PPO mean beats take_all_valid with lower drawdown and enough seeds agree; validate robustness before adding actions."
            if clear_value
            else no_value_reason
        ),
    }


def split_metadata(events: list[Any]) -> dict[str, Any]:
    return {
        "events": len(events),
        "start_day": events[0].setup_day if events else None,
        "end_day": events[-1].setup_day if events else None,
        "protected_fill_events": len([event for event in events if event.has_protected_fill]),
    }


def max_drawdown(pnl_values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnl_values:
        equity += pnl
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return abs(max_dd)


def max_consecutive_losses(pnl_values: list[float]) -> int:
    current = 0
    best = 0
    for pnl in pnl_values:
        if pnl < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _aggregate_rows(policy_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"policy": policy_name}
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
        if key == "net_pnl" and len(values) > 1:
            result["net_pnl_std"] = _round(pstdev(values))
    return result


def _is_sell_setup(event: Any) -> bool:
    return str(event.setup.get("sell_order_placed", "")).strip().lower() == "true"


def _replace_config(config: AzirPPOConfig, **updates: Any) -> AzirPPOConfig:
    payload = asdict(config)
    payload.update(updates)
    if isinstance(payload.get("seeds"), list):
        payload["seeds"] = tuple(payload["seeds"])
    if isinstance(payload.get("random_valid_seeds"), list):
        payload["random_valid_seeds"] = tuple(payload["random_valid_seeds"])
    return AzirPPOConfig(**payload)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _write_yaml(payload: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


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


def summary_markdown(report: dict[str, Any]) -> str:
    rows = report["comparison"]
    take_all = next(row for row in rows if row["policy"] == "take_all_valid")
    ppo_best = next((row for row in rows if row["policy"] == "ppo_best_seed"), {})
    decision = report["decision"]
    return (
        "# Azir PPO Skip/Take Training Summary\n\n"
        "## Executive Summary\n\n"
        f"- Environment: `{report['environment']}`.\n"
        f"- Train events: {report['splits']['train']['events']}.\n"
        f"- Validation events: {report['splits']['validation']['events']}.\n"
        f"- Test events: {report['splits']['test']['events']}.\n"
        f"- take_all_valid net PnL: {take_all['net_pnl']}; PF: {take_all['profit_factor']}; DD: {take_all['max_drawdown']}.\n"
        f"- best PPO net PnL: {ppo_best.get('net_pnl')}; PF: {ppo_best.get('profit_factor')}; DD: {ppo_best.get('max_drawdown')}.\n"
        f"- PPO mean net PnL: {decision['ppo_mean_net_pnl']}; seed net PnL std: {decision['ppo_seed_net_pnl_std']}.\n"
        f"- PPO adds clear value: {decision['ppo_adds_clear_value']}.\n\n"
        "## Decision\n\n"
        f"- {decision['reason']}\n"
        f"- Recommended next sprint: `{decision['recommended_next_sprint']}`.\n\n"
        "## Guardrails\n\n"
        "- PPO actions remain limited to `skip/take`.\n"
        "- `risk_engine_azir_v1` cannot be overridden by the policy.\n"
        "- This sprint does not add sizing, close early, break even, or trailing alternatives.\n"
    )


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "sprint": report["sprint"],
        "splits": report["splits"],
        "decision": report["decision"],
        "output_metrics": {
            row["policy"]: {
                "net_pnl": row.get("net_pnl"),
                "profit_factor": row.get("profit_factor"),
                "max_drawdown": row.get("max_drawdown"),
                "trades_taken": row.get("trades_taken"),
            }
            for row in report["comparison"]
            if row["policy"] in {"skip_all", "take_all_valid", "take_only_sell_valid", "random_valid_mean", "ppo_mean", "ppo_best_seed"}
        },
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
