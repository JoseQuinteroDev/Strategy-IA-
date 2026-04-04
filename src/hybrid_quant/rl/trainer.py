from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Callable, Sequence

import numpy as np
import torch

from hybrid_quant.core import TrainingArtifact
from hybrid_quant.env import TradingEnvironment

from . import sb3_compat


class RLTrainer(ABC):
    @abstractmethod
    def fit(self, environment: TradingEnvironment) -> TrainingArtifact:
        """Train a policy against the provided environment."""


@dataclass(slots=True)
class PPOSeedArtifact:
    seed: int
    seed_dir: Path
    last_model_path: Path
    best_model_path: Path
    checkpoint_dir: Path
    eval_log_dir: Path
    tensorboard_log_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in [
            "seed_dir",
            "last_model_path",
            "best_model_path",
            "checkpoint_dir",
            "eval_log_dir",
            "tensorboard_log_dir",
        ]:
            payload[key] = str(payload[key])
        return payload


@dataclass(slots=True)
class PPOTrainer(RLTrainer):
    algorithm: str
    total_timesteps: int
    enabled: bool = False
    checkpoint_dir: str = "artifacts/rl"
    seeds: list[int] = field(default_factory=lambda: [7, 11])
    policy: str = "MlpPolicy"
    learning_rate: float = 0.0003
    n_steps: int = 128
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    clip_range: float = 0.2
    eval_freq: int = 5000
    checkpoint_freq: int = 5000
    n_eval_episodes: int = 1
    device: str = "auto"
    verbose: int = 1
    tensorboard_log_dir: str = "artifacts/rl/tensorboard"

    def fit(self, environment: TradingEnvironment) -> TrainingArtifact:
        if not self.enabled:
            return TrainingArtifact(
                algorithm=self.algorithm,
                status="disabled",
                metadata={
                    "enabled": self.enabled,
                    "total_timesteps": self.total_timesteps,
                    "reason": "PPO trainer is configured but disabled in settings.",
                    "environment": environment.__class__.__name__,
                },
            )

        output_dir = Path(self.checkpoint_dir) / "adhoc-fit"
        artifact = self.train(
            train_env_factory=lambda: environment,
            eval_env_factory=lambda: environment,
            output_dir=output_dir,
            seeds=self.seeds[:1],
        )
        return artifact

    def train(
        self,
        *,
        train_env_factory: Callable[[], TradingEnvironment],
        eval_env_factory: Callable[[], TradingEnvironment] | None,
        output_dir: str | Path,
        seeds: Sequence[int] | None = None,
    ) -> TrainingArtifact:
        sb3_compat.require_sb3()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        selected_seeds = list(seeds or self.seeds)
        seed_artifacts: list[PPOSeedArtifact] = []

        for seed in selected_seeds:
            seed_artifacts.append(
                self._train_single_seed(
                    seed=seed,
                    train_env_factory=train_env_factory,
                    eval_env_factory=eval_env_factory,
                    output_dir=output_path,
                )
            )

        payload = {
            "algorithm": self.algorithm,
            "enabled": self.enabled,
            "total_timesteps": self.total_timesteps,
            "seeds": selected_seeds,
            "seed_artifacts": [artifact.to_dict() for artifact in seed_artifacts],
        }
        report_path = output_path / "training_artifact.json"
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return TrainingArtifact(
            algorithm=self.algorithm,
            status="trained",
            metadata={
                "output_dir": str(output_path),
                "report_path": str(report_path),
                "seeds": selected_seeds,
                "seed_artifacts": [artifact.to_dict() for artifact in seed_artifacts],
            },
        )

    def load_model(self, model_path: str | Path, environment: TradingEnvironment | None = None):
        sb3_compat.require_sb3()
        model_cls = sb3_compat.PPO
        return model_cls.load(str(model_path), env=environment)

    def _train_single_seed(
        self,
        *,
        seed: int,
        train_env_factory: Callable[[], TradingEnvironment],
        eval_env_factory: Callable[[], TradingEnvironment] | None,
        output_dir: Path,
    ) -> PPOSeedArtifact:
        seed_dir = output_dir / f"seed-{seed}"
        checkpoint_dir = seed_dir / "checkpoints"
        best_model_dir = seed_dir / "best_model"
        eval_log_dir = seed_dir / "eval_logs"
        tensorboard_log_dir = Path(self.tensorboard_log_dir) / f"seed-{seed}"
        for directory in [seed_dir, checkpoint_dir, best_model_dir, eval_log_dir, tensorboard_log_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self._seed_everything(seed)
        train_env = train_env_factory()
        if sb3_compat.check_env is not None:
            sb3_compat.check_env(train_env, warn=True)

        eval_env = eval_env_factory() if eval_env_factory is not None else None
        model = sb3_compat.PPO(
            self.policy,
            train_env,
            seed=seed,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            ent_coef=self.ent_coef,
            clip_range=self.clip_range,
            verbose=self.verbose,
            tensorboard_log=str(tensorboard_log_dir),
            device=self.device,
        )

        callbacks = []
        if self.checkpoint_freq > 0 and sb3_compat.CheckpointCallback is not None:
            callbacks.append(
                sb3_compat.CheckpointCallback(
                    save_freq=self.checkpoint_freq,
                    save_path=str(checkpoint_dir),
                    name_prefix=f"ppo-seed-{seed}",
                )
            )
        if eval_env is not None and self.eval_freq > 0 and sb3_compat.EvalCallback is not None:
            callbacks.append(
                sb3_compat.EvalCallback(
                    eval_env,
                    best_model_save_path=str(best_model_dir),
                    log_path=str(eval_log_dir),
                    eval_freq=self.eval_freq,
                    deterministic=True,
                    n_eval_episodes=self.n_eval_episodes,
                )
            )

        callback = None
        if len(callbacks) == 1:
            callback = callbacks[0]
        elif len(callbacks) > 1 and sb3_compat.CallbackList is not None:
            callback = sb3_compat.CallbackList(callbacks)

        model.learn(total_timesteps=self.total_timesteps, callback=callback)

        last_model_path = seed_dir / "last_model"
        model.save(str(last_model_path))
        best_model_path = self._resolve_best_model_path(best_model_dir, last_model_path)

        metadata = {
            "seed": seed,
            "total_timesteps": self.total_timesteps,
            "policy": self.policy,
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
        }
        (seed_dir / "seed_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return PPOSeedArtifact(
            seed=seed,
            seed_dir=seed_dir,
            last_model_path=last_model_path,
            best_model_path=best_model_path,
            checkpoint_dir=checkpoint_dir,
            eval_log_dir=eval_log_dir,
            tensorboard_log_dir=tensorboard_log_dir,
            metadata=metadata,
        )

    def _resolve_best_model_path(self, best_model_dir: Path, last_model_path: Path) -> Path:
        candidates = sorted(best_model_dir.glob("*.zip"))
        if candidates:
            return candidates[0]
        last_model_zip = last_model_path.with_suffix(".zip")
        if last_model_zip.exists():
            return last_model_zip
        return last_model_path

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
