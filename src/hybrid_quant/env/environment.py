from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from hybrid_quant.core import EnvObservation, EnvTransition, FeatureSnapshot


class TradingEnvironment(ABC):
    @abstractmethod
    def reset(self) -> EnvObservation:
        """Reset environment state and return initial observation."""

    @abstractmethod
    def step(self, action: float) -> EnvTransition:
        """Apply an action and return the next transition."""


@dataclass(slots=True)
class HybridTradingEnvironment(TradingEnvironment):
    observation_window: int
    max_steps: int
    reward_mode: str
    features: Sequence[FeatureSnapshot] = field(default_factory=tuple)
    _cursor: int = 0
    _position: float = 0.0

    def attach_features(self, features: Sequence[FeatureSnapshot]) -> None:
        self.features = tuple(features)

    def reset(self) -> EnvObservation:
        self._cursor = 0
        self._position = 0.0
        return self._build_observation()

    def step(self, action: float) -> EnvTransition:
        self._position = action
        self._cursor += 1
        terminated = bool(self.features) and self._cursor >= len(self.features)
        truncated = self._cursor >= self.max_steps
        observation = self._build_observation()
        return EnvTransition(
            observation=observation,
            reward=0.0,
            terminated=terminated,
            truncated=truncated,
            info={"reward_mode": self.reward_mode, "scaffold": True},
        )

    def _build_observation(self) -> EnvObservation:
        if not self.features:
            return EnvObservation(
                timestamp=None,
                features={},
                position=self._position,
                metadata={"scaffold": True},
            )

        index = min(self._cursor, len(self.features) - 1)
        snapshot = self.features[index]
        return EnvObservation(
            timestamp=snapshot.timestamp,
            features=dict(snapshot.values),
            position=self._position,
            metadata={
                "cursor": self._cursor,
                "window": self.observation_window,
                "scaffold": True,
            },
        )

