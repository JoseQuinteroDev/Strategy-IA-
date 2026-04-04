from __future__ import annotations

try:  # pragma: no cover - exercised when dependencies are installed
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    SB3_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback for environments without SB3
    PPO = None
    CallbackList = None
    CheckpointCallback = None
    EvalCallback = None
    DummyVecEnv = None
    VecMonitor = None
    check_env = None
    SB3_AVAILABLE = False


def require_sb3() -> None:
    if not SB3_AVAILABLE:
        raise RuntimeError(
            "stable-baselines3 is required for PPO training. Install project dependencies first."
        )


__all__ = [
    "SB3_AVAILABLE",
    "PPO",
    "CallbackList",
    "CheckpointCallback",
    "DummyVecEnv",
    "EvalCallback",
    "VecMonitor",
    "check_env",
    "require_sb3",
]
