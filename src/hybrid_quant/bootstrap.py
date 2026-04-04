from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backtest import IntradayBacktestEngine
from .core import Settings, load_settings
from .data import InMemoryDataSource
from .env import HybridTradingEnvironment
from .features import FeaturePipeline
from .paper import PaperTradingRunner
from .risk import PropFirmRiskEngine
from .rl.trainer import PPOTrainer
from .strategy import Strategy, build_strategy
from .validation import WalkForwardValidator


@dataclass(slots=True)
class TradingApplication:
    settings: Settings
    data_source: InMemoryDataSource
    feature_pipeline: FeaturePipeline
    strategy: Strategy
    risk_engine: PropFirmRiskEngine
    backtest_engine: IntradayBacktestEngine
    environment: HybridTradingEnvironment
    rl_trainer: PPOTrainer
    validator: WalkForwardValidator
    paper_runner: PaperTradingRunner

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.settings.app.name,
            "symbol": self.settings.market.symbol,
            "execution_timeframe": self.settings.market.execution_timeframe,
            "filter_timeframe": self.settings.market.filter_timeframe,
            "strategy": self.settings.strategy.name,
            "risk_mode": "prop_firm" if self.settings.risk.prop_firm_mode else "standard",
            "rl_algorithm": self.settings.rl.algorithm,
            "rl_enabled": self.settings.rl.enabled,
        }


def build_application(config_dir: str | Path) -> TradingApplication:
    settings = load_settings(config_dir)
    return build_application_from_settings(settings)


def build_application_from_settings(settings: Settings) -> TradingApplication:
    estimated_round_trip_cost_bps = settings.strategy.estimated_round_trip_cost_bps or (
        2.0 * (settings.backtest.fee_bps + settings.backtest.slippage_bps)
    )

    data_source = InMemoryDataSource(
        symbol=settings.market.symbol,
        execution_timeframe=settings.market.execution_timeframe,
        provider_name=settings.data.provider,
    )
    feature_pipeline = FeaturePipeline(
        feature_names=settings.features.enabled,
        lookback_window=settings.features.lookback_window,
        regime_window=settings.features.regime_window,
        normalize=settings.features.normalize,
    )
    strategy = build_strategy(
        settings,
        estimated_round_trip_cost_bps=estimated_round_trip_cost_bps,
    )
    risk_engine = PropFirmRiskEngine(
        max_risk_per_trade=settings.risk.max_risk_per_trade,
        max_daily_loss=settings.risk.max_daily_loss,
        max_total_drawdown=settings.risk.max_total_drawdown,
        daily_kill_switch=settings.risk.daily_kill_switch,
        max_trades_per_day=settings.risk.max_trades_per_day,
        max_open_positions=settings.risk.max_open_positions,
        max_leverage=settings.risk.max_leverage,
        block_outside_session=settings.risk.block_outside_session,
        session_start_hour_utc=settings.risk.session_start_hour_utc,
        session_start_minute_utc=settings.risk.session_start_minute_utc,
        session_end_hour_utc=settings.risk.session_end_hour_utc,
        session_end_minute_utc=settings.risk.session_end_minute_utc,
        require_stop_loss=settings.risk.require_stop_loss,
        prop_firm_mode=settings.risk.prop_firm_mode,
    )
    backtest_engine = IntradayBacktestEngine(
        initial_capital=settings.backtest.initial_capital,
        fee_bps=settings.backtest.fee_bps,
        slippage_bps=settings.backtest.slippage_bps,
        latency_ms=settings.backtest.latency_ms,
        intrabar_exit_policy=settings.backtest.intrabar_exit_policy,
    )
    environment = HybridTradingEnvironment(
        state_context_bars=settings.env.effective_state_context_bars,
        max_steps=settings.env.max_steps,
        reward_mode=settings.env.reward_mode,
        strategy=strategy,
        risk_engine=risk_engine,
        initial_capital=settings.backtest.initial_capital,
        fee_bps=settings.backtest.fee_bps,
        slippage_bps=settings.backtest.slippage_bps,
        intrabar_exit_policy=settings.backtest.intrabar_exit_policy,
        symbol=settings.market.symbol,
        execution_timeframe=settings.market.execution_timeframe,
        filter_timeframe=settings.market.filter_timeframe,
    )
    rl_trainer = PPOTrainer(
        algorithm=settings.rl.algorithm,
        checkpoint_dir=settings.rl.checkpoint_dir,
        total_timesteps=settings.rl.total_timesteps,
        enabled=settings.rl.enabled,
        seeds=settings.rl.seeds,
        policy=settings.rl.policy,
        learning_rate=settings.rl.learning_rate,
        n_steps=settings.rl.n_steps,
        batch_size=settings.rl.batch_size,
        gamma=settings.rl.gamma,
        gae_lambda=settings.rl.gae_lambda,
        ent_coef=settings.rl.ent_coef,
        clip_range=settings.rl.clip_range,
        eval_freq=settings.rl.eval_freq,
        checkpoint_freq=settings.rl.checkpoint_freq,
        n_eval_episodes=settings.rl.n_eval_episodes,
        device=settings.rl.device,
        verbose=settings.rl.verbose,
        tensorboard_log_dir=settings.rl.tensorboard_log_dir,
    )
    validator = WalkForwardValidator(
        walk_forward_splits=settings.validation.walk_forward_splits,
        min_trades=settings.validation.min_trades,
        max_drawdown_limit=settings.validation.max_drawdown_limit,
        sharpe_floor=settings.validation.sharpe_floor,
    )
    paper_runner = PaperTradingRunner(
        venue=settings.paper.broker,
        dry_run=settings.paper.dry_run,
        heartbeat_seconds=settings.paper.heartbeat_seconds,
    )

    return TradingApplication(
        settings=settings,
        data_source=data_source,
        feature_pipeline=feature_pipeline,
        strategy=strategy,
        risk_engine=risk_engine,
        backtest_engine=backtest_engine,
        environment=environment,
        rl_trainer=rl_trainer,
        validator=validator,
        paper_runner=paper_runner,
    )
