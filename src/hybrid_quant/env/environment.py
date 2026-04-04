from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from hybrid_quant.core import FeatureSnapshot, MarketBar, PortfolioState, SignalSide, StrategyContext, StrategySignal
from hybrid_quant.execution import PortfolioSimulator, is_within_session, signal_has_executable_levels
from hybrid_quant.risk import PropFirmRiskEngine
from hybrid_quant.strategy import Strategy

from .gym_compat import gym, spaces


class TradingEnvironment(gym.Env, ABC):
    metadata = {"render_modes": []}

    @abstractmethod
    def attach_market_data(
        self,
        bars: Sequence[MarketBar],
        features: Sequence[FeatureSnapshot],
        *,
        candidate_signals: Sequence[StrategySignal] | None = None,
        symbol: str | None = None,
        execution_timeframe: str | None = None,
        filter_timeframe: str | None = None,
    ) -> None:
        """Load an episode into the environment."""


@dataclass(slots=True)
class EpisodeStatus:
    terminated: bool = False
    truncated: bool = False
    terminated_reason: str | None = None
    truncated_reason: str | None = None


@dataclass(slots=True)
class RewardBreakdown:
    mode: str
    equity_delta: float
    fees_penalty: float
    daily_drawdown_penalty: float
    total_drawdown_penalty: float
    overtrading_penalty: float
    violation_penalty: float

    def total(self) -> float:
        if self.mode == "pnl_only":
            return float(self.equity_delta)

        reward = self.equity_delta
        reward -= self.fees_penalty * 0.5
        reward -= self.daily_drawdown_penalty
        reward -= self.total_drawdown_penalty
        reward -= self.overtrading_penalty
        reward -= self.violation_penalty
        return float(reward)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "mode": self.mode,
            "equity_delta": float(self.equity_delta),
            "fees_penalty": float(self.fees_penalty),
            "daily_drawdown_penalty": float(self.daily_drawdown_penalty),
            "total_drawdown_penalty": float(self.total_drawdown_penalty),
            "overtrading_penalty": float(self.overtrading_penalty),
            "violation_penalty": float(self.violation_penalty),
            "reward": self.total(),
        }


class HybridTradingEnvironment(TradingEnvironment):
    """Gymnasium environment over strategy-generated candidate trades.

    Semantics:
    - The observation is a single-bar snapshot: current market features plus portfolio/risk state.
    - `ACTION_TAKE_TRADE` acts on the candidate trade attached to the current bar and queues execution for the next bar.
    - `ACTION_CLOSE_EARLY` force-closes an open position on the current bar.
    - Every `step()` advances the environment by at most one bar and lets the shared `PortfolioSimulator`
      process intrabar exits, time stop, and session close using the same execution semantics as the baseline.
    - The `PropFirmRiskEngine` is evaluated against the current candidate and current portfolio state before the
      simulator advances to the next bar.

    `state_context_bars` is the canonical configuration name for the environment context budget. The legacy
    `observation_window` argument is still accepted for backward compatibility, but the environment currently
    exposes a single-bar observation rather than a stacked temporal tensor.
    """

    ACTION_SKIP = 0
    ACTION_TAKE_TRADE = 1
    ACTION_CLOSE_EARLY = 2
    OBSERVATION_MODE = "single_bar_features_plus_portfolio_state"

    def __init__(
        self,
        observation_window: int | None = None,
        max_steps: int = 3000,
        reward_mode: str = "risk_adjusted",
        *,
        state_context_bars: int | None = None,
        strategy: Strategy | None = None,
        risk_engine: PropFirmRiskEngine | None = None,
        initial_capital: float = 100000.0,
        fee_bps: float = 4.0,
        slippage_bps: float = 2.0,
        intrabar_exit_policy: str = "conservative",
        symbol: str = "BTCUSDT",
        execution_timeframe: str = "5m",
        filter_timeframe: str = "1H",
    ) -> None:
        resolved_context_bars = state_context_bars if state_context_bars is not None else observation_window
        self.state_context_bars = max(1, int(resolved_context_bars or 1))
        self.observation_window = self.state_context_bars
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.strategy = strategy
        self.risk_engine = risk_engine
        self.initial_capital = initial_capital
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.intrabar_exit_policy = intrabar_exit_policy
        self.symbol = symbol
        self.execution_timeframe = execution_timeframe
        self.filter_timeframe = filter_timeframe

        self.action_space = spaces.Discrete(3)
        self._state_feature_names = [
            "position",
            "cash_ratio",
            "equity_ratio",
            "daily_pnl_pct",
            "total_drawdown_pct",
            "trades_today_ratio",
            "daily_kill_switch_active",
            "session_allowed",
            "remaining_daily_loss",
            "remaining_total_drawdown",
            "remaining_trade_slots",
            "remaining_position_slots",
            "candidate_actionable",
            "candidate_side",
        ]
        self._feature_names: list[str] = []
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self._state_feature_names),), dtype=np.float32)

        self._bars: tuple[MarketBar, ...] = ()
        self._features: tuple[FeatureSnapshot, ...] = ()
        self._candidate_signals: tuple[StrategySignal, ...] = ()
        self._simulator: PortfolioSimulator | None = None
        self._cursor = 0
        self._episode_steps = 0
        self._current_day = None
        self._day_start_equity = initial_capital
        self._peak_equity = initial_capital
        self._trades_today = 0
        self._daily_kill_switch_active = False
        self._last_equity = initial_capital

    def attach_market_data(
        self,
        bars: Sequence[MarketBar],
        features: Sequence[FeatureSnapshot],
        *,
        candidate_signals: Sequence[StrategySignal] | None = None,
        symbol: str | None = None,
        execution_timeframe: str | None = None,
        filter_timeframe: str | None = None,
    ) -> None:
        """Attach one full episode worth of bars, features, and optional precomputed candidate signals."""
        if len(bars) != len(features):
            raise ValueError("Bars and features must have the same length for the environment.")
        if not bars:
            raise ValueError("The environment requires at least one bar to build an episode.")

        self._bars = tuple(bars)
        self._features = tuple(features)
        self.symbol = symbol or self.symbol
        self.execution_timeframe = execution_timeframe or self.execution_timeframe
        self.filter_timeframe = filter_timeframe or self.filter_timeframe
        self._feature_names = list(self._features[0].values.keys()) if self._features else []
        if candidate_signals is None:
            self._candidate_signals = tuple(self._build_candidate_signals())
        else:
            if len(candidate_signals) != len(self._bars):
                raise ValueError("Candidate signals must match the number of bars.")
            self._candidate_signals = tuple(candidate_signals)

        total_dims = len(self._feature_names) + len(self._state_feature_names)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dims,), dtype=np.float32)

    def attach_features(self, features: Sequence[FeatureSnapshot]) -> None:
        synthetic_bars = [
            MarketBar(
                timestamp=feature.timestamp,
                open=0.0,
                high=0.0,
                low=0.0,
                close=0.0,
                volume=0.0,
            )
            for feature in features
        ]
        flat_signals = [
            StrategySignal(
                symbol=self.symbol,
                timestamp=feature.timestamp,
                side=SignalSide.FLAT,
                strength=0.0,
                rationale="No candidate signal loaded for this feature set.",
            )
            for feature in features
        ]
        self.attach_market_data(synthetic_bars, features, candidate_signals=flat_signals)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the episode and return the first observation for the current bar."""
        if not self._bars or not self._features or not self._candidate_signals:
            raise ValueError("Attach bars, features, and candidate signals before calling reset().")

        try:
            super().reset(seed=seed, options=options)
        except TypeError:
            super().reset(seed=seed)

        self._simulator = PortfolioSimulator(
            initial_capital=self.initial_capital,
            fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps,
            intrabar_exit_policy=self.intrabar_exit_policy,
        )
        self._cursor = 0
        self._episode_steps = 0
        self._current_day = self._bars[0].timestamp.date()
        self._day_start_equity = self.initial_capital
        self._peak_equity = self.initial_capital
        self._trades_today = 0
        self._daily_kill_switch_active = False
        self._last_equity = self.initial_capital

        observation = self._build_observation(self._cursor)
        info = self._build_info(
            reward=0.0,
            action=self.ACTION_SKIP,
            candidate_signal=self._candidate_signals[self._cursor],
            risk_decision=self._evaluate_risk(self._candidate_signals[self._cursor], self._bars[self._cursor]),
            portfolio_state=self._portfolio_state(self._bars[self._cursor]),
            trades_closed=(),
            queued_trade=False,
            action_effect="reset",
            reward_breakdown=RewardBreakdown(
                mode=self.reward_mode,
                equity_delta=0.0,
                fees_penalty=0.0,
                daily_drawdown_penalty=0.0,
                total_drawdown_penalty=0.0,
                overtrading_penalty=0.0,
                violation_penalty=0.0,
            ),
        )
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply an action to the current candidate trade and advance the simulation by one bar.

        Action timing:
        - The action is evaluated against the candidate trade on `self._cursor`.
        - `take_trade` queues the candidate for execution on the next bar if risk allows it.
        - The simulator then advances to the next bar, applying entries/exits with the shared execution semantics.
        - Reward is computed from the realized equity change and prop-firm penalties for that step.
        """
        if self._simulator is None:
            raise RuntimeError("Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        current_bar = self._bars[self._cursor]
        candidate_signal = self._candidate_signals[self._cursor]
        start_equity = self._last_equity
        risk_decision = self._evaluate_risk(candidate_signal, current_bar)
        trades_closed, queued_trade, action_effect, blocked_attempt = self._apply_action(
            action=action,
            current_bar=current_bar,
            candidate_signal=candidate_signal,
            risk_decision=risk_decision,
        )
        episode_status = self._advance_one_bar(current_bar=current_bar, trades_closed=trades_closed)

        self._episode_steps += 1
        current_bar = self._bars[self._cursor]
        portfolio_state = self._portfolio_state(current_bar)

        if self._episode_steps >= self.max_steps and not episode_status.terminated:
            truncation_trade = self._simulator.force_close(bar=current_bar, index=self._cursor, exit_reason="max_steps")
            if truncation_trade is not None:
                trades_closed.append(truncation_trade)
                self._trades_today += 1
                portfolio_state = self._portfolio_state(current_bar)
            episode_status.truncated = True

        if self.risk_engine and portfolio_state.total_drawdown_pct >= self.risk_engine.max_total_drawdown:
            episode_status.terminated = True

        if episode_status.terminated:
            episode_status.terminated_reason = (
                "risk_limit" if portfolio_state.total_drawdown_pct >= self._max_total_drawdown_limit() else "end_of_data"
            )
        if episode_status.truncated:
            episode_status.truncated_reason = "max_steps"

        reward_breakdown = self._compute_reward(
            start_equity=start_equity,
            end_equity=portfolio_state.equity,
            portfolio_state=portfolio_state,
            trades_closed=tuple(trades_closed),
            blocked_attempt=blocked_attempt,
        )
        reward = reward_breakdown.total()
        self._last_equity = portfolio_state.equity

        observation = self._build_observation(self._cursor)
        info = self._build_info(
            reward=reward,
            action=action,
            candidate_signal=candidate_signal,
            risk_decision=risk_decision,
            portfolio_state=portfolio_state,
            trades_closed=tuple(trades_closed),
            queued_trade=queued_trade,
            action_effect=action_effect,
            reward_breakdown=reward_breakdown,
        )
        if episode_status.terminated_reason is not None:
            info["terminated_reason"] = episode_status.terminated_reason
        if episode_status.truncated_reason is not None:
            info["truncated_reason"] = episode_status.truncated_reason
        return observation, reward, episode_status.terminated, episode_status.truncated, info

    def _build_candidate_signals(self) -> list[StrategySignal]:
        if self.strategy is None:
            return [
                StrategySignal(
                    symbol=self.symbol,
                    timestamp=bar.timestamp,
                    side=SignalSide.FLAT,
                    strength=0.0,
                    rationale="No strategy attached to the environment.",
                )
                for bar in self._bars
            ]

        signals: list[StrategySignal] = []
        for bar, feature in zip(self._bars, self._features, strict=True):
            adx = feature.values.get("adx_1h")
            regime = (
                "trend"
                if adx is not None
                and math.isfinite(float(adx))
                and float(adx) > getattr(self.strategy, "adx_threshold", 25.0)
                else "range"
            )
            signals.append(
                self.strategy.generate(
                    StrategyContext(
                        symbol=self.symbol,
                        execution_timeframe=self.execution_timeframe,
                        filter_timeframe=self.filter_timeframe,
                        bars=[bar],
                        features=[feature],
                        regime=regime,
                    )
                )
            )
        return signals

    def _build_observation(self, index: int) -> np.ndarray:
        bar = self._bars[index]
        feature_snapshot = self._features[index]
        candidate_signal = self._candidate_signals[index]
        portfolio_state = self._portfolio_state(bar)
        observation_values = self._feature_vector(feature_snapshot) + self._state_vector(
            portfolio_state=portfolio_state,
            candidate_signal=candidate_signal,
        )
        return np.asarray(observation_values, dtype=np.float32)

    def _build_info(
        self,
        *,
        reward: float,
        action: int,
        candidate_signal: StrategySignal,
        risk_decision,
        portfolio_state: PortfolioState,
        trades_closed: Sequence[Any],
        queued_trade: bool,
        action_effect: str,
        reward_breakdown: RewardBreakdown,
    ) -> dict[str, Any]:
        return {
            "timestamp": self._bars[self._cursor].timestamp.isoformat(),
            "cursor": self._cursor,
            "reward_mode": self.reward_mode,
            "observation_mode": self.OBSERVATION_MODE,
            "state_context_bars": self.state_context_bars,
            "reward": reward,
            "reward_breakdown": reward_breakdown.to_dict(),
            "action": int(action),
            "action_effect": action_effect,
            "candidate_side": candidate_signal.side.value,
            "candidate_actionable": self._candidate_is_actionable(candidate_signal),
            "candidate_evaluated_at": candidate_signal.timestamp.isoformat(),
            "risk_approved": risk_decision.approved,
            "risk_reason_code": risk_decision.reason_code,
            "risk_blocked_by": list(risk_decision.blocked_by),
            "risk_rationale": risk_decision.rationale,
            "queued_trade": queued_trade,
            "blocked_attempt": action_effect == "trade_blocked",
            "trades_closed": [trade.exit_reason for trade in trades_closed],
            "closed_trades_detail": [
                {
                    "exit_reason": trade.exit_reason,
                    "net_pnl": trade.net_pnl,
                    "gross_pnl": trade.gross_pnl,
                    "fees_paid": trade.fees_paid,
                    "return_pct": trade.return_pct,
                }
                for trade in trades_closed
            ],
            "net_pnl_step": float(sum(getattr(trade, "net_pnl", 0.0) for trade in trades_closed)),
            "fees_step": float(sum(getattr(trade, "fees_paid", 0.0) for trade in trades_closed)),
            "portfolio": {
                "cash": portfolio_state.cash,
                "equity": portfolio_state.equity,
                "daily_pnl_pct": portfolio_state.daily_pnl_pct,
                "total_drawdown_pct": portfolio_state.total_drawdown_pct,
                "trades_today": portfolio_state.trades_today,
                "daily_kill_switch_active": portfolio_state.daily_kill_switch_active,
                "session_allowed": portfolio_state.session_allowed,
                "open_positions": portfolio_state.open_positions,
            },
            "observation_keys": self._observation_keys(),
        }

    def _apply_action(self, *, action: int, current_bar: MarketBar, candidate_signal: StrategySignal, risk_decision):
        simulator = self._require_simulator()
        trades_closed = []
        queued_trade = False
        action_effect = "skip"
        blocked_attempt = False

        if action == self.ACTION_CLOSE_EARLY and simulator.position is not None:
            forced_trade = simulator.force_close(bar=current_bar, index=self._cursor, exit_reason="agent_close_early")
            if forced_trade is not None:
                trades_closed.append(forced_trade)
                self._trades_today += 1
                action_effect = "closed_early"
            return trades_closed, queued_trade, action_effect, blocked_attempt

        if action == self.ACTION_CLOSE_EARLY:
            return trades_closed, queued_trade, "no_position_to_close", blocked_attempt

        if action != self.ACTION_TAKE_TRADE:
            return trades_closed, queued_trade, action_effect, blocked_attempt

        if (
            risk_decision.approved
            and signal_has_executable_levels(candidate_signal)
            and simulator.position is None
            and simulator.pending_entry is None
            and self._cursor < len(self._bars) - 1
        ):
            queued_trade = simulator.queue_signal(
                signal=candidate_signal,
                index=self._cursor,
                size_fraction=risk_decision.size_fraction,
                max_leverage=risk_decision.max_leverage,
            )
            return trades_closed, queued_trade, ("trade_queued" if queued_trade else "queue_rejected"), blocked_attempt

        blocked_attempt = self._candidate_is_actionable(candidate_signal)
        action_effect = "trade_blocked" if blocked_attempt else "no_candidate_trade"
        return trades_closed, queued_trade, action_effect, blocked_attempt

    def _advance_one_bar(self, *, current_bar: MarketBar, trades_closed: list[Any]) -> EpisodeStatus:
        simulator = self._require_simulator()
        next_index = self._cursor + 1
        if next_index < len(self._bars):
            self._sync_day_boundary(self._bars[next_index])
            next_trade = simulator.step(
                index=next_index,
                bar=self._bars[next_index],
                feature=self._features[next_index],
                exit_zscore_threshold=self._exit_zscore_threshold(),
                session_close_hour_utc=self._session_close_hour(),
                session_close_minute_utc=self._session_close_minute(),
            )
            if next_trade is not None:
                trades_closed.append(next_trade)
                self._trades_today += 1
            self._cursor = next_index
            return EpisodeStatus()

        final_trade = simulator.force_close(bar=current_bar, index=self._cursor, exit_reason="end_of_data")
        if final_trade is not None:
            trades_closed.append(final_trade)
            self._trades_today += 1
        return EpisodeStatus(terminated=True)

    def _evaluate_risk(self, signal: StrategySignal, bar: MarketBar):
        if self.risk_engine is None:
            return _FallbackRiskDecision.approved()
        return self.risk_engine.evaluate(signal, self._portfolio_state(bar))

    def _portfolio_state(self, bar: MarketBar) -> PortfolioState:
        simulator = self._require_simulator()
        current_equity = simulator.equity(bar.close)
        self._peak_equity = max(self._peak_equity, current_equity)
        daily_pnl_pct = ((current_equity - self._day_start_equity) / self._day_start_equity) if self._day_start_equity > 0.0 else 0.0
        total_drawdown_pct = ((self._peak_equity - current_equity) / self._peak_equity) if self._peak_equity > 0.0 else 0.0
        if self.risk_engine is not None and self.risk_engine.daily_kill_switch and daily_pnl_pct <= -self.risk_engine.max_daily_loss:
            self._daily_kill_switch_active = True

        open_positions = int(simulator.position is not None) + int(simulator.pending_entry is not None)
        gross_exposure = (simulator.position.quantity * bar.close) if simulator.position is not None else 0.0
        return PortfolioState(
            equity=current_equity,
            cash=simulator.cash,
            daily_pnl_pct=daily_pnl_pct,
            open_positions=open_positions,
            gross_exposure=gross_exposure,
            peak_equity=self._peak_equity,
            total_drawdown_pct=total_drawdown_pct,
            trades_today=self._trades_today,
            daily_kill_switch_active=self._daily_kill_switch_active,
            session_allowed=is_within_session(
                bar.timestamp,
                start_hour_utc=self._session_start_hour(),
                start_minute_utc=self._session_start_minute(),
                end_hour_utc=self._session_end_hour(),
                end_minute_utc=self._session_end_minute(),
            ),
            timestamp=bar.timestamp,
        )

    def _compute_reward(
        self,
        *,
        start_equity: float,
        end_equity: float,
        portfolio_state: PortfolioState,
        trades_closed: Sequence[Any],
        blocked_attempt: bool,
    ) -> RewardBreakdown:
        equity_delta = (end_equity - start_equity) / self.initial_capital
        fees_step = sum(getattr(trade, "fees_paid", 0.0) for trade in trades_closed) / self.initial_capital
        daily_drawdown_penalty = max(0.0, -portfolio_state.daily_pnl_pct) * 0.25
        total_drawdown_penalty = max(0.0, portfolio_state.total_drawdown_pct) * 0.15
        overtrading_penalty = 0.0005 * max(0.0, float(self._trades_today - max(1, self._max_trades_per_day_limit() // 2)))
        violation_penalty = 0.0
        if blocked_attempt:
            violation_penalty += 0.01
        if self.risk_engine is not None and portfolio_state.daily_pnl_pct <= -self.risk_engine.max_daily_loss:
            violation_penalty += 0.02
        if portfolio_state.total_drawdown_pct >= self._max_total_drawdown_limit():
            violation_penalty += 0.05

        return RewardBreakdown(
            mode=self.reward_mode,
            equity_delta=float(equity_delta),
            fees_penalty=float(fees_step),
            daily_drawdown_penalty=float(daily_drawdown_penalty),
            total_drawdown_penalty=float(total_drawdown_penalty),
            overtrading_penalty=float(overtrading_penalty),
            violation_penalty=float(violation_penalty),
        )

    def _sync_day_boundary(self, next_bar: MarketBar) -> None:
        simulator = self._require_simulator()
        next_day = next_bar.timestamp.date()
        if self._current_day != next_day:
            self._current_day = next_day
            self._day_start_equity = simulator.equity(next_bar.open)
            self._trades_today = 0
            self._daily_kill_switch_active = False

    def _exit_zscore_threshold(self) -> float | None:
        return getattr(self.strategy, "exit_zscore", None) if self.strategy is not None else None

    def _session_close_hour(self) -> int:
        return self.strategy.session_close_hour_utc if self.strategy is not None else 23

    def _session_close_minute(self) -> int:
        return self.strategy.session_close_minute_utc if self.strategy is not None else 55

    def _session_start_hour(self) -> int:
        return self.risk_engine.session_start_hour_utc if self.risk_engine is not None else 0

    def _session_start_minute(self) -> int:
        return self.risk_engine.session_start_minute_utc if self.risk_engine is not None else 0

    def _session_end_hour(self) -> int:
        return self.risk_engine.session_end_hour_utc if self.risk_engine is not None else 23

    def _session_end_minute(self) -> int:
        return self.risk_engine.session_end_minute_utc if self.risk_engine is not None else 55

    def _max_trades_per_day_limit(self) -> int:
        return self.risk_engine.max_trades_per_day if self.risk_engine is not None else 999999

    def _max_open_positions_limit(self) -> int:
        return self.risk_engine.max_open_positions if self.risk_engine is not None else 1

    def _max_total_drawdown_limit(self) -> float:
        return self.risk_engine.max_total_drawdown if self.risk_engine is not None else 1.0

    def _remaining_daily_loss(self, portfolio_state: PortfolioState) -> float:
        if self.risk_engine is None:
            return 1.0
        return float(max(0.0, self.risk_engine.max_daily_loss + portfolio_state.daily_pnl_pct))

    def _remaining_total_drawdown(self, portfolio_state: PortfolioState) -> float:
        return float(max(0.0, self._max_total_drawdown_limit() - portfolio_state.total_drawdown_pct))

    def _remaining_trade_slots(self) -> float:
        limit = self._max_trades_per_day_limit()
        return float(max(0.0, limit - self._trades_today))

    def _remaining_position_slots(self) -> float:
        simulator = self._require_simulator()
        open_positions = int(simulator.position is not None) + int(simulator.pending_entry is not None)
        return float(max(0.0, self._max_open_positions_limit() - open_positions))

    def _feature_vector(self, feature_snapshot: FeatureSnapshot) -> list[float]:
        return [float(feature_snapshot.values.get(name, 0.0)) for name in self._feature_names]

    def _state_vector(self, *, portfolio_state: PortfolioState, candidate_signal: StrategySignal) -> list[float]:
        position = 0.0
        if self._simulator is not None and self._simulator.position is not None:
            position = 1.0 if self._simulator.position.side == SignalSide.LONG else -1.0

        return [
            position,
            portfolio_state.cash / self.initial_capital,
            portfolio_state.equity / self.initial_capital,
            portfolio_state.daily_pnl_pct,
            portfolio_state.total_drawdown_pct,
            self._safe_divide(self._trades_today, max(self._max_trades_per_day_limit(), 1)),
            float(portfolio_state.daily_kill_switch_active),
            float(portfolio_state.session_allowed),
            self._remaining_daily_loss(portfolio_state),
            self._remaining_total_drawdown(portfolio_state),
            self._remaining_trade_slots(),
            self._remaining_position_slots(),
            float(self._candidate_is_actionable(candidate_signal)),
            self._candidate_side_value(candidate_signal),
        ]

    def _observation_keys(self) -> list[str]:
        return self._feature_names + self._state_feature_names

    def _candidate_is_actionable(self, candidate_signal: StrategySignal) -> bool:
        return candidate_signal.side in {SignalSide.LONG, SignalSide.SHORT}

    def _candidate_side_value(self, candidate_signal: StrategySignal) -> float:
        if candidate_signal.side == SignalSide.LONG:
            return 1.0
        if candidate_signal.side == SignalSide.SHORT:
            return -1.0
        return 0.0

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        if denominator <= 0.0:
            return 0.0
        return float(numerator / denominator)

    def _require_simulator(self) -> PortfolioSimulator:
        if self._simulator is None:
            raise RuntimeError("The environment simulator is not initialized. Call reset() first.")
        return self._simulator


class _FallbackRiskDecision:
    def __init__(self) -> None:
        self.approved = True
        self.size_fraction = 0.0025
        self.max_leverage = 1.0
        self.rationale = "No risk engine attached."
        self.reason_code = "no_risk_engine"
        self.blocked_by: tuple[str, ...] = tuple()
        self.metadata: dict[str, Any] = {}

    @classmethod
    def approved(cls) -> "_FallbackRiskDecision":
        return cls()
