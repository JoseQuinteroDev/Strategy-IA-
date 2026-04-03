from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from hybrid_quant.core import TrainingArtifact
from hybrid_quant.env import TradingEnvironment


class RLTrainer(ABC):
    @abstractmethod
    def fit(self, environment: TradingEnvironment) -> TrainingArtifact:
        """Train a policy against the provided environment."""


@dataclass(slots=True)
class DeferredPPOTrainer(RLTrainer):
    algorithm: str
    total_timesteps: int
    enabled: bool = False

    def fit(self, environment: TradingEnvironment) -> TrainingArtifact:
        return TrainingArtifact(
            algorithm=self.algorithm,
            status="deferred",
            metadata={
                "enabled": self.enabled,
                "total_timesteps": self.total_timesteps,
                "reason": "RL implementation intentionally postponed until the base stack is stable.",
                "environment": environment.__class__.__name__,
            },
        )

