from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import pandas as pd

from hybrid_quant.core import Settings, load_settings
from hybrid_quant.data import (
    BinanceHistoricalDownloader,
    DownloadRequest,
    HistoricalDataIngestionService,
    ParquetDatasetStore,
)
from hybrid_quant.env import HybridTradingEnvironment

from .dataset import EpisodeData, RLEpisodeBuilder
from .evaluation import EvaluationSummary, baseline_policy, evaluate_policy, random_policy, sb3_policy
from .trainer import PPOTrainer

if TYPE_CHECKING:
    from hybrid_quant.bootstrap import TradingApplication


@dataclass(slots=True)
class PPOTrainingArtifacts:
    output_dir: Path
    report_path: Path
    comparison_path: Path
    summary_path: Path
    training_artifact: Any


@dataclass(slots=True)
class PPOTrainingRunner:
    application: TradingApplication
    data_service: HistoricalDataIngestionService
    episode_builder: RLEpisodeBuilder

    @classmethod
    def from_config(cls, config_dir: str | Path) -> "PPOTrainingRunner":
        settings = load_settings(config_dir)
        from hybrid_quant.bootstrap import build_application

        application = build_application(config_dir)
        data_service = HistoricalDataIngestionService(
            downloader=BinanceHistoricalDownloader(
                base_url=settings.data.historical_api_url,
                timeout_seconds=settings.data.request_timeout_seconds,
            ),
            store=ParquetDatasetStore(
                compression=settings.data.parquet_compression,
                engine=settings.data.parquet_engine,
            ),
        )
        episode_builder = RLEpisodeBuilder(application=application, data_service=data_service)
        return cls(application=application, data_service=data_service, episode_builder=episode_builder)

    def run(
        self,
        *,
        output_dir: str | Path,
        request: DownloadRequest | None = None,
        input_frame: pd.DataFrame | None = None,
        allow_gaps: bool = False,
        seeds: Sequence[int] | None = None,
    ) -> PPOTrainingArtifacts:
        settings = self.application.settings
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        frame = self.episode_builder.prepare_frame(
            request=request,
            input_frame=input_frame,
            allow_gaps=allow_gaps,
        )
        dataset = self.episode_builder.build_dataset(frame)
        train_episode = dataset.get(settings.rl.train_split)
        eval_episode = dataset.get(settings.rl.eval_split)
        test_episode = dataset.get(settings.rl.test_split)

        selected_seeds = list(seeds or settings.rl.seeds)
        trainer = replace(self.application.rl_trainer, enabled=True)
        models_dir = output_path / "models"
        training_artifact = trainer.train(
            train_env_factory=self._make_env_factory(train_episode),
            eval_env_factory=self._make_env_factory(eval_episode),
            output_dir=models_dir,
            seeds=selected_seeds,
        )

        comparison = self._build_comparison(
            trainer=trainer,
            test_episode=test_episode,
            selected_seeds=selected_seeds,
            training_artifact=training_artifact,
        )
        comparison_path = output_path / "comparison.json"
        comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

        report_payload = {
            "algorithm": settings.rl.algorithm,
            "symbol": settings.market.symbol,
            "execution_timeframe": settings.market.execution_timeframe,
            "filter_timeframe": settings.market.filter_timeframe,
            "train_split": self._episode_metadata(train_episode),
            "eval_split": self._episode_metadata(eval_episode),
            "test_split": self._episode_metadata(test_episode),
            "seeds": selected_seeds,
            "training": training_artifact.metadata,
            "comparison": comparison,
        }
        report_path = output_path / "report.json"
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

        summary_path = output_path / "summary.md"
        summary_path.write_text(self._build_summary_markdown(comparison), encoding="utf-8")

        return PPOTrainingArtifacts(
            output_dir=output_path,
            report_path=report_path,
            comparison_path=comparison_path,
            summary_path=summary_path,
            training_artifact=training_artifact,
        )

    def _make_env_factory(self, episode: EpisodeData) -> Callable[[], HybridTradingEnvironment]:
        def _factory() -> HybridTradingEnvironment:
            environment = HybridTradingEnvironment(
                observation_window=self.application.settings.env.observation_window,
                max_steps=self.application.settings.env.max_steps,
                reward_mode=self.application.settings.env.reward_mode,
                strategy=self.application.strategy,
                risk_engine=self.application.risk_engine,
                initial_capital=self.application.settings.backtest.initial_capital,
                fee_bps=self.application.settings.backtest.fee_bps,
                slippage_bps=self.application.settings.backtest.slippage_bps,
                intrabar_exit_policy=self.application.settings.backtest.intrabar_exit_policy,
                symbol=self.application.settings.market.symbol,
                execution_timeframe=self.application.settings.market.execution_timeframe,
                filter_timeframe=self.application.settings.market.filter_timeframe,
            )
            environment.attach_market_data(
                episode.bars,
                episode.features,
                candidate_signals=episode.candidate_signals,
                symbol=self.application.settings.market.symbol,
                execution_timeframe=self.application.settings.market.execution_timeframe,
                filter_timeframe=self.application.settings.market.filter_timeframe,
            )
            return environment

        return _factory

    def _build_comparison(
        self,
        *,
        trainer: PPOTrainer,
        test_episode: EpisodeData,
        selected_seeds: Sequence[int],
        training_artifact: Any,
    ) -> dict[str, Any]:
        env_factory = self._make_env_factory(test_episode)
        heuristic_result = evaluate_policy(
            policy_name="baseline_without_rl",
            env_factory=env_factory,
            action_fn=baseline_policy,
            seeds=selected_seeds,
        )
        random_result = evaluate_policy(
            policy_name="random_policy",
            env_factory=env_factory,
            action_fn=random_policy,
            seeds=selected_seeds,
        )

        ppo_seed_results: list[EvaluationSummary] = []
        for seed_artifact in training_artifact.metadata["seed_artifacts"]:
            model = trainer.load_model(seed_artifact["best_model_path"], environment=env_factory())
            ppo_seed_results.append(
                evaluate_policy(
                    policy_name=f"ppo_seed_{seed_artifact['seed']}",
                    env_factory=env_factory,
                    action_fn=sb3_policy(model),
                    seeds=[seed_artifact["seed"]],
                )
            )

        return {
            "baseline_without_rl": heuristic_result.to_dict(),
            "random_policy": random_result.to_dict(),
            "ppo_trained": self._aggregate_summaries("ppo_trained", ppo_seed_results).to_dict(),
            "ppo_per_seed": [summary.to_dict() for summary in ppo_seed_results],
        }

    def _aggregate_summaries(self, policy_name: str, summaries: Sequence[EvaluationSummary]) -> EvaluationSummary:
        if not summaries:
            return EvaluationSummary(
                policy_name=policy_name,
                episodes=0,
                mean_reward=0.0,
                net_pnl=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                total_return=0.0,
                number_of_trades=0,
                blocked_by_risk=0,
                terminated_by_risk_limit=0,
                truncated_by_max_steps=0,
                metadata={"seed_results": []},
            )

        count = len(summaries)
        return EvaluationSummary(
            policy_name=policy_name,
            episodes=sum(summary.episodes for summary in summaries),
            mean_reward=float(sum(summary.mean_reward for summary in summaries) / count),
            net_pnl=float(sum(summary.net_pnl for summary in summaries) / count),
            win_rate=float(sum(summary.win_rate for summary in summaries) / count),
            max_drawdown=float(max(summary.max_drawdown for summary in summaries)),
            total_return=float(sum(summary.total_return for summary in summaries) / count),
            number_of_trades=int(sum(summary.number_of_trades for summary in summaries)),
            blocked_by_risk=int(sum(summary.blocked_by_risk for summary in summaries)),
            terminated_by_risk_limit=int(sum(summary.terminated_by_risk_limit for summary in summaries)),
            truncated_by_max_steps=int(sum(summary.truncated_by_max_steps for summary in summaries)),
            metadata={"seed_results": [summary.to_dict() for summary in summaries]},
        )

    def _episode_metadata(self, episode: EpisodeData) -> dict[str, Any]:
        return {
            "name": episode.name,
            "rows": len(episode.frame),
            "start": episode.start.isoformat() if episode.start else None,
            "end": episode.end.isoformat() if episode.end else None,
        }

    def _build_summary_markdown(self, comparison: dict[str, Any]) -> str:
        lines = ["# PPO Training Summary", ""]
        for label in ["baseline_without_rl", "random_policy", "ppo_trained"]:
            metrics = comparison[label]
            lines.extend(
                [
                    f"## {label}",
                    f"- Mean reward: `{metrics['mean_reward']}`",
                    f"- Net PnL: `{metrics['net_pnl']}`",
                    f"- Win rate: `{metrics['win_rate']}`",
                    f"- Max drawdown: `{metrics['max_drawdown']}`",
                    f"- Total return: `{metrics['total_return']}`",
                    f"- Trades: `{metrics['number_of_trades']}`",
                    f"- Risk blocked attempts: `{metrics['blocked_by_risk']}`",
                    f"- Risk-limit terminations: `{metrics['terminated_by_risk_limit']}`",
                    f"- Max-step truncations: `{metrics['truncated_by_max_steps']}`",
                    "",
                ]
            )
        return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate a PPO baseline.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output-dir")
    parser.add_argument("--input-path")
    parser.add_argument("--allow-gaps", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = load_settings(args.config_dir)
    runner = PPOTrainingRunner.from_config(args.config_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(settings.rl.checkpoint_dir) / "ppo-baseline"

    if args.input_path:
        frame = _read_input_frame(args.input_path)
        artifacts = runner.run(
            output_dir=output_dir,
            input_frame=frame,
            allow_gaps=args.allow_gaps,
            seeds=args.seeds,
        )
    else:
        start = _parse_datetime(args.start or settings.data.default_start)
        end = _resolve_end_datetime(args.end, settings, start)
        request = DownloadRequest(
            symbol=settings.market.symbol,
            interval=settings.market.execution_timeframe,
            start=start,
            end=end,
            limit=settings.data.request_limit,
        )
        artifacts = runner.run(
            output_dir=output_dir,
            request=request,
            allow_gaps=args.allow_gaps or settings.data.allow_gaps,
            seeds=args.seeds,
        )

    print(f"PPO report written to {artifacts.report_path}")
    print(f"Comparison written to {artifacts.comparison_path}")
    print(f"Summary written to {artifacts.summary_path}")
    return 0


def _read_input_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    frame = pd.read_csv(source, parse_dates=["open_time"], index_col="open_time")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


def _parse_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _resolve_end_datetime(raw_end: str | None, settings: Settings, start: datetime) -> datetime:
    if raw_end:
        return _parse_datetime(raw_end)
    if settings.data.default_end:
        return _parse_datetime(settings.data.default_end)
    return start + timedelta(days=90)
