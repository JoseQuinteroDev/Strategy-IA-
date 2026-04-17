"""Valid-action masking and small regularization sweep for Azir PPO."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import yaml

from hybrid_quant.env.azir_event_env import (
    ACTION_TAKE,
    AzirEventReplayEnvironment,
    AzirReplayEvent,
    build_azir_event_replay_dataset,
)
from hybrid_quant.risk.azir_state import AzirRiskConfig

from .train_ppo_skip_take import (
    AzirPPOConfig,
    evaluate_event_policy,
    evaluate_sb3_model,
    load_config,
    make_env,
    random_valid_policy,
    skip_all_policy,
    split_events,
    take_all_valid_policy,
    take_only_sell_valid_policy,
)


@dataclass(frozen=True)
class RegularizationVariant:
    name: str
    ent_coef: float


@dataclass(frozen=True)
class MaskingExperimentConfig:
    ppo_config: AzirPPOConfig
    masking_mode: str
    regularization_variants: tuple[RegularizationVariant, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Azir PPO valid-action masking and regularization diagnostics.")
    parser.add_argument("--mt5-log-path", required=True)
    parser.add_argument("--protected-report-path", required=True)
    parser.add_argument("--config-path", default="configs/experiments/azir_ppo_masking_regularization_v1.yaml")
    parser.add_argument("--unmasked-report-path", default="artifacts/azir-ppo-reward-observation-adjusted-v1/adjusted_ppo_eval_report.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbol", default="XAUUSD-STD")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_masking_regularization(
        mt5_log_path=Path(args.mt5_log_path),
        protected_report_path=Path(args.protected_report_path),
        config_path=Path(args.config_path),
        unmasked_report_path=Path(args.unmasked_report_path),
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
    )
    print(json.dumps(_console_summary(report), indent=2, ensure_ascii=False))
    return 0


def run_masking_regularization(
    *,
    mt5_log_path: Path,
    protected_report_path: Path,
    config_path: Path,
    unmasked_report_path: Path,
    output_dir: Path,
    symbol: str = "XAUUSD-STD",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_config = load_masking_config(config_path)
    ppo_config = experiment_config.ppo_config
    events = build_azir_event_replay_dataset(
        mt5_log_path=mt5_log_path,
        protected_report_path=protected_report_path,
        symbol=symbol,
        risk_config=AzirRiskConfig(),
    )
    unmasked_splits = split_events(events, ppo_config)
    masked_splits = {
        split_name: eligible_only_events(split_events, ppo_config)
        for split_name, split_events in unmasked_splits.items()
    }
    masking_rows = masking_report_rows(unmasked_splits, masked_splits)
    controls = evaluate_masked_controls(masked_splits["test"], ppo_config)
    ppo_rows, checkpoint_rows = run_regularization_sweep(masked_splits, experiment_config, output_dir)
    comparison_rows = controls + ppo_rows
    sweep_rows = regularization_sweep_summary(checkpoint_rows, controls)
    masked_vs_unmasked_rows = masked_vs_unmasked_report(
        unmasked_report_path=unmasked_report_path,
        masked_controls=controls,
        masked_ppo_rows=ppo_rows,
    )
    decision = masking_decision(controls, checkpoint_rows)

    _write_csv(masked_vs_unmasked_rows, output_dir / "masked_vs_unmasked_ppo.csv")
    _write_csv(checkpoint_rows, output_dir / "checkpoint_final_vs_best.csv")
    _write_csv(sweep_rows, output_dir / "regularization_sweep_summary.csv")
    _write_csv(comparison_rows, output_dir / "controls_vs_masked_ppo.csv")
    _write_csv(masking_rows, output_dir / "masking_coverage.csv")
    (output_dir / "action_masking_report.md").write_text(
        action_masking_markdown(experiment_config, masking_rows, decision),
        encoding="utf-8",
    )

    report = {
        "sprint": "ppo_valid_action_masking_and_policy_regularization_v1",
        "benchmark": "baseline_azir_protected_economic_v1",
        "environment": "AzirEventReplayEnvironment",
        "masking_mode": experiment_config.masking_mode,
        "actions": {"0": "skip", "1": "take"},
        "mt5_log_path": str(mt5_log_path),
        "protected_report_path": str(protected_report_path),
        "unmasked_report_path": str(unmasked_report_path),
        "splits_unmasked": {name: split_metadata(items) for name, items in unmasked_splits.items()},
        "splits_masked": {name: split_metadata(items) for name, items in masked_splits.items()},
        "masking_coverage": masking_rows,
        "hparams_base": ppo_config.to_hparams(),
        "regularization_variants": [variant.__dict__ for variant in experiment_config.regularization_variants],
        "controls": controls,
        "ppo_rows": ppo_rows,
        "comparison": comparison_rows,
        "checkpoint_final_vs_best": checkpoint_rows,
        "regularization_sweep_summary": sweep_rows,
        "masked_vs_unmasked_ppo": masked_vs_unmasked_rows,
        "decision": decision,
    }
    (output_dir / "ppo_masking_diagnostics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def load_masking_config(path: Path) -> MaskingExperimentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ppo_config = load_config(path)
    masking = payload.get("masking", {})
    variants = tuple(
        RegularizationVariant(
            name=str(item.get("name", f"ent_{item.get('ent_coef', ppo_config.ent_coef)}")),
            ent_coef=float(item.get("ent_coef", ppo_config.ent_coef)),
        )
        for item in payload.get("regularization_variants", [])
    )
    return MaskingExperimentConfig(
        ppo_config=ppo_config,
        masking_mode=str(masking.get("mode", "eligible_only")),
        regularization_variants=variants or (RegularizationVariant(name=f"masked_ent_{ppo_config.ent_coef}", ent_coef=ppo_config.ent_coef),),
    )


def eligible_only_events(events: list[AzirReplayEvent], config: AzirPPOConfig) -> list[AzirReplayEvent]:
    """Keep only events where take is a genuinely valid action before policy choice."""

    eligible: list[AzirReplayEvent] = []
    for event in events:
        env = make_env([event], config)
        _, info = env.reset(seed=0)
        if ACTION_TAKE in info.get("valid_actions", ()):
            eligible.append(event)
    return eligible


def masking_report_rows(unmasked_splits: dict[str, list[AzirReplayEvent]], masked_splits: dict[str, list[AzirReplayEvent]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, events in unmasked_splits.items():
        masked = masked_splits[split_name]
        rows.append(
            {
                "split": split_name,
                "unmasked_events": len(events),
                "masked_eligible_events": len(masked),
                "removed_events": len(events) - len(masked),
                "eligible_event_pct": _round(len(masked) / len(events)) if events else 0.0,
                "unmasked_protected_fill_events": sum(1 for event in events if event.has_protected_fill),
                "masked_protected_fill_events": sum(1 for event in masked if event.has_protected_fill),
                "removed_no_order_or_risk_blocked_events": len(events) - len(masked),
            }
        )
    return rows


def evaluate_masked_controls(test_events: list[AzirReplayEvent], config: AzirPPOConfig) -> list[dict[str, Any]]:
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
    rows.append(aggregate_rows("random_valid_mean", random_rows))
    for row in rows:
        row["masking_mode"] = "eligible_only"
        row["model_label"] = "control"
    return rows


def run_regularization_sweep(
    splits: dict[str, list[AzirReplayEvent]],
    experiment_config: MaskingExperimentConfig,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

    ppo_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for variant in experiment_config.regularization_variants:
        config = replace(experiment_config.ppo_config, ent_coef=variant.ent_coef)
        variant_dir = output_dir / "ppo_models" / variant.name
        checkpoint_root = output_dir / "ppo_checkpoints" / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        seed_final_rows: list[dict[str, Any]] = []
        seed_best_rows: list[dict[str, Any]] = []
        for seed in config.seeds:
            seed_dir = variant_dir / f"seed-{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir = checkpoint_root / f"seed-{seed}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            train_env = make_env(splits["train"], config)
            eval_env = make_env(splits["validation"], config)
            callbacks = CallbackList(
                [
                    CheckpointCallback(
                        save_freq=max(1, config.checkpoint_freq),
                        save_path=str(checkpoint_dir),
                        name_prefix=f"ppo_masked_{variant.name}_seed_{seed}",
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
            final_test = evaluate_sb3_model(f"{variant.name}_final_seed_{seed}", model, splits["test"], seed=seed, config=config)
            final_validation = evaluate_sb3_model(
                f"{variant.name}_final_seed_{seed}_validation",
                model,
                splits["validation"],
                seed=seed,
                config=config,
            )
            best_model_path = seed_dir / "best_model" / "best_model.zip"
            if best_model_path.exists():
                best_model = PPO.load(str(best_model_path), env=make_env(splits["validation"], config))
                best_validation = evaluate_sb3_model(
                    f"{variant.name}_best_validation_seed_{seed}",
                    best_model,
                    splits["validation"],
                    seed=seed,
                    config=config,
                )
                best_test = evaluate_sb3_model(
                    f"{variant.name}_best_validation_seed_{seed}",
                    best_model,
                    splits["test"],
                    seed=seed,
                    config=config,
                )
            else:
                best_validation = final_validation
                best_test = final_test
                best_model_path = final_model_path.with_suffix(".zip")

            final_test = annotate_model_row(final_test, variant, seed, "final", final_model_path.with_suffix(".zip"))
            best_test = annotate_model_row(best_test, variant, seed, "best_validation", best_model_path)
            seed_final_rows.append(final_test)
            seed_best_rows.append(best_test)
            ppo_rows.extend([final_test, best_test])
            checkpoint_rows.append(checkpoint_comparison_row(variant, seed, final_validation, final_test, best_validation, best_test))

        ppo_rows.append(aggregate_model_rows(f"{variant.name}_final_mean", seed_final_rows, variant, "final"))
        ppo_rows.append(aggregate_model_rows(f"{variant.name}_best_validation_mean", seed_best_rows, variant, "best_validation"))
    return ppo_rows, checkpoint_rows


def annotate_model_row(
    row: dict[str, Any],
    variant: RegularizationVariant,
    seed: int,
    model_label: str,
    model_path: Path,
) -> dict[str, Any]:
    result = dict(row)
    result["variant"] = variant.name
    result["ent_coef"] = variant.ent_coef
    result["model_label"] = model_label
    result["seed"] = seed
    result["masking_mode"] = "eligible_only"
    result["model_path"] = str(model_path)
    return result


def checkpoint_comparison_row(
    variant: RegularizationVariant,
    seed: int,
    final_validation: dict[str, Any],
    final_test: dict[str, Any],
    best_validation: dict[str, Any],
    best_test: dict[str, Any],
) -> dict[str, Any]:
    return {
        "variant": variant.name,
        "ent_coef": variant.ent_coef,
        "seed": seed,
        "final_validation_net_pnl": final_validation.get("net_pnl"),
        "best_validation_net_pnl": best_validation.get("net_pnl"),
        "final_test_net_pnl": final_test.get("net_pnl"),
        "best_validation_test_net_pnl": best_test.get("net_pnl"),
        "final_test_profit_factor": final_test.get("profit_factor"),
        "best_validation_test_profit_factor": best_test.get("profit_factor"),
        "final_test_expectancy": final_test.get("expectancy"),
        "best_validation_test_expectancy": best_test.get("expectancy"),
        "final_test_max_drawdown": final_test.get("max_drawdown"),
        "best_validation_test_max_drawdown": best_test.get("max_drawdown"),
        "final_test_trades_taken": final_test.get("trades_taken"),
        "best_validation_test_trades_taken": best_test.get("trades_taken"),
        "final_test_take_rate": final_test.get("take_rate"),
        "best_validation_test_take_rate": best_test.get("take_rate"),
        "best_validation_improves_test_net_pnl": _float(best_test.get("net_pnl")) > _float(final_test.get("net_pnl")),
    }


def aggregate_model_rows(policy_name: str, rows: list[dict[str, Any]], variant: RegularizationVariant, model_label: str) -> dict[str, Any]:
    row = aggregate_rows(policy_name, rows)
    row["variant"] = variant.name
    row["ent_coef"] = variant.ent_coef
    row["model_label"] = model_label
    row["masking_mode"] = "eligible_only"
    return row


def regularization_sweep_summary(checkpoint_rows: list[dict[str, Any]], controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    take_all = next(row for row in controls if row["policy"] == "take_all_valid")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in checkpoint_rows:
        grouped.setdefault((str(row["variant"]), "final"), []).append(row)
        grouped.setdefault((str(row["variant"]), "best_validation"), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (variant, model_label), rows in sorted(grouped.items()):
        prefix = "final_test" if model_label == "final" else "best_validation_test"
        net_values = [_float(row.get(f"{prefix}_net_pnl")) for row in rows]
        dd_values = [_float(row.get(f"{prefix}_max_drawdown")) for row in rows]
        trade_values = [_float(row.get(f"{prefix}_trades_taken")) for row in rows]
        summary_rows.append(
            {
                "variant": variant,
                "model_label": model_label,
                "seed_count": len(rows),
                "mean_net_pnl": _round(mean(net_values)) if net_values else 0.0,
                "net_pnl_std": _round(pstdev(net_values)) if len(net_values) > 1 else 0.0,
                "best_net_pnl": _round(max(net_values)) if net_values else 0.0,
                "worst_net_pnl": _round(min(net_values)) if net_values else 0.0,
                "mean_max_drawdown": _round(mean(dd_values)) if dd_values else 0.0,
                "mean_trades_taken": _round(mean(trade_values)) if trade_values else 0.0,
                "take_all_valid_net_pnl": take_all.get("net_pnl"),
                "seeds_beating_take_all_valid": sum(1 for value in net_values if value > _float(take_all.get("net_pnl"))),
                "beats_take_all_valid_on_mean": _round(mean(net_values)) > _float(take_all.get("net_pnl")) if net_values else False,
            }
        )
    return sorted(summary_rows, key=lambda row: (_float(row["mean_net_pnl"]), -_float(row["mean_max_drawdown"])), reverse=True)


def masked_vs_unmasked_report(
    *,
    unmasked_report_path: Path,
    masked_controls: list[dict[str, Any]],
    masked_ppo_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unmasked = _load_json(unmasked_report_path)
    for row in unmasked.get("comparison", []):
        if row.get("policy") in {"take_all_valid", "take_only_sell_valid", "random_valid_mean", "ppo_mean", "ppo_best_seed"}:
            rows.append(flatten_comparison_row("unmasked_adjusted", row))
    for row in masked_controls + masked_ppo_rows:
        policy = str(row.get("policy", ""))
        if (
            policy in {"take_all_valid", "take_only_sell_valid", "random_valid_mean"}
            or policy.endswith("_final_mean")
            or policy.endswith("_best_validation_mean")
        ):
            rows.append(flatten_comparison_row("masked_eligible_only", row))
    return rows


def flatten_comparison_row(phase: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": phase,
        "policy": row.get("policy"),
        "variant": row.get("variant", ""),
        "model_label": row.get("model_label", ""),
        "seed": row.get("seed", ""),
        "net_pnl": row.get("net_pnl"),
        "profit_factor": row.get("profit_factor"),
        "expectancy": row.get("expectancy"),
        "max_drawdown": row.get("max_drawdown"),
        "trades_taken": row.get("trades_taken"),
        "take_rate": row.get("take_rate"),
        "valid_take_rate": row.get("valid_take_rate"),
        "blocked_attempts": row.get("blocked_attempts"),
        "invalid_attempts": row.get("invalid_attempts"),
    }


def masking_decision(controls: list[dict[str, Any]], checkpoint_rows: list[dict[str, Any]]) -> dict[str, Any]:
    take_all = next(row for row in controls if row["policy"] == "take_all_valid")
    take_all_net = _float(take_all.get("net_pnl"))
    best_row = max(
        checkpoint_rows,
        key=lambda row: (
            _float(row.get("best_validation_test_net_pnl")),
            _float(row.get("best_validation_test_profit_factor")),
            -_float(row.get("best_validation_test_max_drawdown")),
        ),
    ) if checkpoint_rows else {}
    final_best_row = max(
        checkpoint_rows,
        key=lambda row: (
            _float(row.get("final_test_net_pnl")),
            _float(row.get("final_test_profit_factor")),
            -_float(row.get("final_test_max_drawdown")),
        ),
    ) if checkpoint_rows else {}
    best_net = _float(best_row.get("best_validation_test_net_pnl"))
    final_best_net = _float(final_best_row.get("final_test_net_pnl"))
    best_models_beating = sum(1 for row in checkpoint_rows if _float(row.get("best_validation_test_net_pnl")) > take_all_net)
    final_models_beating = sum(1 for row in checkpoint_rows if _float(row.get("final_test_net_pnl")) > take_all_net)
    clear_value = best_net > take_all_net and best_models_beating >= 2
    return {
        "take_all_valid_net_pnl": _round(take_all_net),
        "best_masked_best_validation_variant": best_row.get("variant"),
        "best_masked_best_validation_seed": best_row.get("seed"),
        "best_masked_best_validation_test_net_pnl": _round(best_net),
        "best_masked_final_variant": final_best_row.get("variant"),
        "best_masked_final_seed": final_best_row.get("seed"),
        "best_masked_final_test_net_pnl": _round(final_best_net),
        "best_validation_models_beating_take_all_valid": best_models_beating,
        "final_models_beating_take_all_valid": final_models_beating,
        "ppo_skip_take_masked_adds_clear_value": clear_value,
        "ready_for_robust_validation_for_ppo_v1": clear_value,
        "recommended_next_sprint": "robust_validation_for_ppo_v1" if clear_value else "close_skip_take_or_design_management_actions_v1",
        "reason": (
            "Masked PPO beats take_all_valid across enough validation-selected models; robust validation is justified."
            if clear_value
            else "Masked PPO does not beat take_all_valid in a defendible way; skip/take alone is not adding marginal selection value."
        ),
    }


def action_masking_markdown(
    config: MaskingExperimentConfig,
    masking_rows: list[dict[str, Any]],
    decision: dict[str, Any],
) -> str:
    coverage = "\n".join(
        f"- `{row['split']}`: {row['masked_eligible_events']} / {row['unmasked_events']} events kept "
        f"({row['eligible_event_pct']}); removed={row['removed_events']}."
        for row in masking_rows
    )
    variants = "\n".join(f"- `{variant.name}`: ent_coef={variant.ent_coef}" for variant in config.regularization_variants)
    return (
        "# Azir PPO Valid-Action Masking Report\n\n"
        "## Masking Method\n\n"
        "- Masking mode: `eligible_only`.\n"
        "- The temporal split is performed first on all Azir setup events.\n"
        "- Inside each split, the replay keeps only events where `ACTION_TAKE` is genuinely valid before policy choice.\n"
        "- Removed events include days with no real Azir order or days blocked by the external Risk Engine before the agent acts.\n"
        "- This does not add actions and does not change Azir, trailing, or the Risk Engine.\n\n"
        "## Coverage\n\n"
        f"{coverage}\n\n"
        "## Regularization Variants\n\n"
        f"{variants}\n\n"
        "## Decision\n\n"
        f"- PPO skip/take masked adds clear value: `{decision['ppo_skip_take_masked_adds_clear_value']}`.\n"
        f"- Best validation-selected masked test net: `{decision['best_masked_best_validation_test_net_pnl']}`.\n"
        f"- take_all_valid net: `{decision['take_all_valid_net_pnl']}`.\n"
        f"- {decision['reason']}\n"
    )


def split_metadata(events: list[AzirReplayEvent]) -> dict[str, Any]:
    return {
        "events": len(events),
        "start_day": events[0].setup_day if events else None,
        "end_day": events[-1].setup_day if events else None,
        "protected_fill_events": len([event for event in events if event.has_protected_fill]),
    }


def aggregate_rows(policy_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    decision = report["decision"]
    best_rows = [
        row for row in report["regularization_sweep_summary"]
        if row.get("model_label") == "best_validation"
    ]
    return {
        "sprint": report["sprint"],
        "masking_mode": report["masking_mode"],
        "splits_masked": report["splits_masked"],
        "decision": decision,
        "best_validation_summaries": best_rows[:5],
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
