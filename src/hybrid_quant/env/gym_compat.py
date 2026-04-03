from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised when gymnasium is installed
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback used in this local environment
    GYMNASIUM_AVAILABLE = False

    class _BaseEnv:
        metadata: dict[str, object] = {}

        def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None) -> None:
            self.np_random = np.random.default_rng(seed)
            return None

    class _Discrete:
        def __init__(self, n: int) -> None:
            self.n = int(n)

        def contains(self, value: object) -> bool:
            return isinstance(value, (int, np.integer)) and 0 <= int(value) < self.n

    class _Box:
        def __init__(
            self,
            low: float | np.ndarray,
            high: float | np.ndarray,
            shape: tuple[int, ...] | None = None,
            dtype: type[np.floating] | np.dtype = np.float32,
        ) -> None:
            self.dtype = np.dtype(dtype)
            if shape is None:
                self.low = np.asarray(low, dtype=self.dtype)
                self.high = np.asarray(high, dtype=self.dtype)
            else:
                self.low = np.full(shape, low, dtype=self.dtype)
                self.high = np.full(shape, high, dtype=self.dtype)
            self.shape = self.low.shape

        def contains(self, value: object) -> bool:
            array = np.asarray(value, dtype=self.dtype)
            return array.shape == self.shape

    class _SpacesModule:
        Box = _Box
        Discrete = _Discrete

    class _GymModule:
        Env = _BaseEnv
        spaces = _SpacesModule()

    gym = _GymModule()
    spaces = gym.spaces


__all__ = ["GYMNASIUM_AVAILABLE", "gym", "spaces"]
