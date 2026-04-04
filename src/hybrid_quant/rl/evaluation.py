from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from hybrid_quant.env import HybridTradingEnvironment


class PolicyModel(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[int, Any]:
        """Return an action compatible with the environment."""


PolicyFn = Callable[[np.ndarray, dict[str, Any], random.Random], int]


@dataclass(slots=True)
class EvaluationSummary:
    policy_name: str
    episodes: int
    mean_reward: float
    net_pnl: float
    win_rate: float
    max_drawdown: float
    total_return: float
    number_of_trades: int
    blocked_by_risk: int
    terminated_by_risk_limit: int
    truncated_by_max_steps: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def baseline_policy(observation: np.ndarray, info: dict[str, Any], rng: random.Random) -> int:
    if info.get("candidate_actionable") and info.get("risk_approved"):
        return HybridTradingEnvironment.ACTION_TAKE_TRADE
    return HybridTradingEnvironment.ACTION_SKIP


def random_policy(observation: np.ndarray, info: dict[str, Any], rng: random.Random) -> int:
    return rng.choice(
        [
            HybridTradingEnvironment.ACTION_SKIP,
            HybridTradingEnvironment.ACTION_TAKE_TRADE,
            HybridTradingEnvironment.ACTION_CLOSE_EARLY,
        ]
    )


def sb3_policy(model: PolicyModel) -> PolicyFn:
    def _policy(observation: np.ndarray, info: dict[str, Any], rng: random.Random) -> int:
        action, _ = model.predict(observation, deterministic=True)
        return int(action)

    return _policy


def evaluate_policy(
    *,
    policy_name: str,
    env_factory: Callable[[], HybridTradingEnvironment],
    action_fn: PolicyFn,
    seeds: Sequence[int],
) -> EvaluationSummary:
    episode_rewards: list[float] = []
    episode_pnls: list[float] = []
    episode_returns: list[float] = []
    max_drawdowns: list[float] = []
    trade_pnls: list[float] = []
    blocked_by_risk = 0
    terminated_by_risk_limit = 0
    truncated_by_max_steps = 0

    for episode_seed in seeds:
        env = env_factory()
        observation, info = env.reset(seed=episode_seed)
        rng = random.Random(episode_seed)
        done = False
        truncated = False
        total_reward = 0.0
        initial_equity = float(info["portfolio"]["equity"])
        final_equity = initial_equity
        episode_max_drawdown = 0.0

        while not done and not truncated:
            action = action_fn(observation, info, rng)
            observation, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            final_equity = float(info["portfolio"]["equity"])
            episode_max_drawdown = max(episode_max_drawdown, float(info["portfolio"]["total_drawdown_pct"]))
            blocked_by_risk += int(bool(info.get("blocked_attempt")))
            for trade in info.get("closed_trades_detail", []):
                trade_pnls.append(float(trade["net_pnl"]))

        if info.get("terminated_reason") == "risk_limit":
            terminated_by_risk_limit += 1
        if info.get("truncated_reason") == "max_steps":
            truncated_by_max_steps += 1

        episode_rewards.append(total_reward)
        episode_pnls.append(final_equity - initial_equity)
        episode_returns.append((final_equity - initial_equity) / initial_equity if initial_equity else 0.0)
        max_drawdowns.append(episode_max_drawdown)

    wins = [pnl for pnl in trade_pnls if pnl > 0.0]
    win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
    return EvaluationSummary(
        policy_name=policy_name,
        episodes=len(seeds),
        mean_reward=float(sum(episode_rewards) / max(len(episode_rewards), 1)),
        net_pnl=float(sum(episode_pnls) / max(len(episode_pnls), 1)),
        win_rate=float(win_rate),
        max_drawdown=float(max(max_drawdowns) if max_drawdowns else 0.0),
        total_return=float(sum(episode_returns) / max(len(episode_returns), 1)),
        number_of_trades=len(trade_pnls),
        blocked_by_risk=blocked_by_risk,
        terminated_by_risk_limit=terminated_by_risk_limit,
        truncated_by_max_steps=truncated_by_max_steps,
        metadata={
            "episode_rewards": episode_rewards,
            "episode_pnls": episode_pnls,
            "episode_returns": episode_returns,
            "episode_max_drawdowns": max_drawdowns,
        },
    )
