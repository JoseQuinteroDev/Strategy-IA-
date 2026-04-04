from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pandas as pd

from hybrid_quant.rl import PPOTrainingRunner
from hybrid_quant.rl.dataset import RLEpisodeBuilder
from hybrid_quant.rl.evaluation import baseline_policy, evaluate_policy, random_policy, sb3_policy
from hybrid_quant.core import SignalSide, StrategySignal


class _FakeCheckpointCallback:
    def __init__(self, *, save_freq: int, save_path: str, name_prefix: str) -> None:
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix


class _FakeEvalCallback:
    def __init__(
        self,
        eval_env,
        *,
        best_model_save_path: str,
        log_path: str,
        eval_freq: int,
        deterministic: bool,
        n_eval_episodes: int,
    ) -> None:
        self.eval_env = eval_env
        self.best_model_save_path = Path(best_model_save_path)
        self.log_path = Path(log_path)
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.n_eval_episodes = n_eval_episodes


class _FakeCallbackList:
    def __init__(self, callbacks) -> None:
        self.callbacks = list(callbacks)


class _FakePPO:
    def __init__(self, policy, env, **kwargs) -> None:
        self.policy = policy
        self.env = env
        self.kwargs = kwargs

    def learn(self, total_timesteps: int, callback=None):
        for cb in _flatten_callbacks(callback):
            if isinstance(cb, _FakeCheckpointCallback):
                cb.save_path.mkdir(parents=True, exist_ok=True)
                (cb.save_path / f"{cb.name_prefix}_{total_timesteps}_steps.zip").write_text("checkpoint", encoding="utf-8")
            if isinstance(cb, _FakeEvalCallback):
                cb.best_model_save_path.mkdir(parents=True, exist_ok=True)
                cb.log_path.mkdir(parents=True, exist_ok=True)
                (cb.best_model_save_path / "best_model.zip").write_text("best-model", encoding="utf-8")
                (cb.log_path / "evaluations.json").write_text("[]", encoding="utf-8")
        return self

    def save(self, path: str) -> None:
        Path(path).with_suffix(".zip").write_text("last-model", encoding="utf-8")

    @classmethod
    def load(cls, path: str, env=None):
        return cls("MlpPolicy", env, loaded_from=path)

    def predict(self, observation, deterministic: bool = True):
        return 1, None


def _flatten_callbacks(callback):
    if callback is None:
        return []
    if isinstance(callback, _FakeCallbackList):
        return list(callback.callbacks)
    return [callback]


def _fake_check_env(env, warn: bool = True) -> None:
    observation, _ = env.reset(seed=0)
    assert env.observation_space.contains(observation)


def _frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01T00:00:00Z", periods=60, freq="5min", tz="UTC")
    frame = pd.DataFrame(index=index)
    close = pd.Series([100.0 + (idx * 0.2) for idx in range(len(index))], index=index)
    frame["open"] = close - 0.1
    frame["high"] = close + 0.8
    frame["low"] = close - 0.3
    frame["close"] = close
    frame["volume"] = 100.0
    return frame


def _synthetic_signals(self, bars, features):
    signals = []
    for index, bar in enumerate(bars):
        if index % 4 == 0 and index < len(bars) - 1:
            signals.append(
                StrategySignal(
                    symbol="BTCUSDT",
                    timestamp=bar.timestamp,
                    side=SignalSide.LONG,
                    strength=1.0,
                    rationale="synthetic rl candidate",
                    entry_price=bar.close,
                    stop_price=bar.close - 0.5,
                    target_price=bar.close + 0.5,
                    time_stop_bars=12,
                    close_on_session_end=True,
                    entry_reason="synthetic rl candidate",
                )
            )
        else:
            signals.append(
                StrategySignal(
                    symbol="BTCUSDT",
                    timestamp=bar.timestamp,
                    side=SignalSide.FLAT,
                    strength=0.0,
                    rationale="hold",
                )
            )
    return signals


class PPOTrainingRunnerTests(unittest.TestCase):
    def _build_runner(self) -> PPOTrainingRunner:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        runner = PPOTrainingRunner.from_config(config_dir)
        runner.application.settings.backtest.fee_bps = 0.0
        runner.application.settings.backtest.slippage_bps = 0.0
        runner.application.rl_trainer.total_timesteps = 16
        runner.application.rl_trainer.eval_freq = 4
        runner.application.rl_trainer.checkpoint_freq = 4
        runner.application.rl_trainer.seeds = [3]
        return runner

    @contextlib.contextmanager
    def _patched_sb3_backend(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(RLEpisodeBuilder, "_generate_signals", _synthetic_signals))
            stack.enter_context(mock.patch("hybrid_quant.rl.sb3_compat.SB3_AVAILABLE", True))
            stack.enter_context(mock.patch("hybrid_quant.rl.sb3_compat.PPO", _FakePPO))
            stack.enter_context(mock.patch("hybrid_quant.rl.sb3_compat.CheckpointCallback", _FakeCheckpointCallback))
            stack.enter_context(mock.patch("hybrid_quant.rl.sb3_compat.EvalCallback", _FakeEvalCallback))
            stack.enter_context(mock.patch("hybrid_quant.rl.sb3_compat.CallbackList", _FakeCallbackList))
            stack.enter_context(mock.patch("hybrid_quant.rl.sb3_compat.check_env", _fake_check_env))
            yield

    def test_runner_smoke_generates_training_and_evaluation_artifacts(self) -> None:
        runner = self._build_runner()

        with tempfile.TemporaryDirectory() as tmp_dir, self._patched_sb3_backend():
            artifacts = runner.run(output_dir=tmp_dir, input_frame=_frame(), seeds=[3])

            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.comparison_path.exists())
            self.assertTrue(artifacts.summary_path.exists())

            report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["algorithm"], "PPO")
            self.assertEqual(report["seeds"], [3])
            self.assertIn("comparison", report)
            self.assertIn("ppo_trained", report["comparison"])

            best_model = Path(tmp_dir) / "models" / "seed-3" / "best_model" / "best_model.zip"
            last_model = Path(tmp_dir) / "models" / "seed-3" / "last_model.zip"
            training_artifact = Path(tmp_dir) / "models" / "training_artifact.json"
            self.assertTrue(best_model.exists())
            self.assertTrue(last_model.exists())
            self.assertTrue(training_artifact.exists())

    def test_trainer_fit_can_load_best_model(self) -> None:
        runner = self._build_runner()
        trainer = replace(
            runner.application.rl_trainer,
            enabled=True,
            total_timesteps=12,
            eval_freq=4,
            checkpoint_freq=4,
            seeds=[5],
        )

        with tempfile.TemporaryDirectory() as tmp_dir, self._patched_sb3_backend():
            frame = runner.episode_builder.prepare_frame(input_frame=_frame())
            dataset = runner.episode_builder.build_dataset(frame)
            env = runner._make_env_factory(dataset.get("train"))()
            artifact = trainer.fit(env, output_dir=Path(tmp_dir) / "fit-models", seeds=[5])

            self.assertEqual(artifact.status, "trained")
            best_model = Path(tmp_dir) / "fit-models" / "seed-5" / "best_model" / "best_model.zip"
            self.assertTrue(best_model.exists())

            model = trainer.load_model(best_model, environment=env)
            observation, _ = env.reset(seed=5)
            action, _ = model.predict(observation, deterministic=True)
            self.assertEqual(int(action), 1)

    def test_evaluation_pipeline_reports_all_policy_metrics(self) -> None:
        runner = self._build_runner()

        with self._patched_sb3_backend():
            frame = runner.episode_builder.prepare_frame(input_frame=_frame())
            dataset = runner.episode_builder.build_dataset(frame)
            env_factory = runner._make_env_factory(dataset.get("test"))
            fake_model = _FakePPO.load("fake-model.zip", env=env_factory())
            summaries = {
                "baseline_without_rl": evaluate_policy(
                    policy_name="baseline_without_rl",
                    env_factory=env_factory,
                    action_fn=baseline_policy,
                    seeds=[3],
                ),
                "random_policy": evaluate_policy(
                    policy_name="random_policy",
                    env_factory=env_factory,
                    action_fn=random_policy,
                    seeds=[3],
                ),
                "ppo_trained": evaluate_policy(
                    policy_name="ppo_trained",
                    env_factory=env_factory,
                    action_fn=sb3_policy(fake_model),
                    seeds=[3],
                ),
            }

        for label, summary in summaries.items():
            payload = summary.to_dict()
            self.assertEqual(payload["policy_name"], label)
            self.assertEqual(payload["episodes"], 1)
            self.assertIn("mean_reward", payload)
            self.assertIn("net_pnl", payload)
            self.assertIn("win_rate", payload)
            self.assertIn("max_drawdown", payload)
            self.assertIn("total_return", payload)
            self.assertIn("number_of_trades", payload)
            self.assertIn("blocked_by_risk", payload)
            self.assertIn("terminated_by_risk_limit", payload)
            self.assertIn("truncated_by_max_steps", payload)
