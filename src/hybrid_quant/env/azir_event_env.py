"""Event-based RL environment contract for Azir.

The agent does not create signals. Azir creates one daily setup event, the
Risk Engine applies hard guards, and the agent can only skip or take.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from hybrid_quant.azir.economic_audit import (
    _event_day_from_text,
    _is_true,
    _parse_timestamp,
    _read_raw_event_log,
    _round,
    _to_float,
    _write_csv,
    reconstruct_lifecycles,
)
from hybrid_quant.azir.protected_benchmark_freeze import revalue_forced_close
from hybrid_quant.azir.replica import load_ohlcv_csv
from hybrid_quant.azir.risk_reaudit import apply_risk_engine_to_lifecycle
from hybrid_quant.risk.azir_engine import AzirRiskEngine
from hybrid_quant.risk.azir_state import AzirRiskConfig, AzirRiskDecision, AzirRiskState

try:  # pragma: no cover - normal project environments should have numpy/gym_compat.
    import numpy as np

    from .gym_compat import gym, spaces
except ModuleNotFoundError:  # pragma: no cover - local minimal Python fallback.
    np = None  # type: ignore[assignment]

    class _SimpleArray(list):
        @property
        def shape(self) -> tuple[int, ...]:
            return (len(self),)

    class _SimpleNP:
        float32 = float
        inf = float("inf")

        @staticmethod
        def asarray(values: list[float], dtype: Any = None) -> _SimpleArray:
            return _SimpleArray(float(value) for value in values)

        @staticmethod
        def zeros(length: int, dtype: Any = None) -> _SimpleArray:
            return _SimpleArray([0.0] * int(length))

    class _BaseEnv:
        metadata: dict[str, object] = {}

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
            return None

    class _Discrete:
        def __init__(self, n: int) -> None:
            self.n = int(n)

        def contains(self, value: object) -> bool:
            return isinstance(value, int) and 0 <= value < self.n

    class _Box:
        def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: Any = float) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Spaces:
        Discrete = _Discrete
        Box = _Box

    class _Gym:
        Env = _BaseEnv

    np = _SimpleNP()  # type: ignore[assignment]
    gym = _Gym()
    spaces = _Spaces()


ACTION_SKIP = 0
ACTION_TAKE = 1
DEFAULT_SYMBOL = "XAUUSD-STD"


OBSERVATION_FIELDS: tuple[str, ...] = (
    "setup_hour",
    "day_of_week",
    "month",
    "is_friday",
    "buy_order_placed",
    "sell_order_placed",
    "buy_allowed_by_trend",
    "sell_allowed_by_trend",
    "swing_high",
    "swing_low",
    "buy_entry",
    "sell_entry",
    "pending_distance_points",
    "spread_points",
    "ema20",
    "prev_close_vs_ema20_points",
    "atr",
    "atr_points",
    "rsi",
    "trend_filter_enabled",
    "atr_filter_enabled",
    "atr_filter_passed",
    "rsi_gate_enabled",
    "rsi_gate_required",
    "prior_exposure_flag",
    "cleanup_issue_before_risk",
    "daily_realized_pnl",
    "daily_drawdown_abs",
    "total_drawdown_abs",
    "consecutive_losses_today",
    "trades_today",
    "remaining_daily_loss",
    "risk_tension_ratio",
    "risk_engine_approved",
    "risk_blocked_flag",
)


FORBIDDEN_OBSERVATION_FIELDS: tuple[str, ...] = (
    "protected_net_pnl",
    "protected_gross_pnl",
    "observed_net_pnl",
    "reward",
    "exit_reason",
    "fill_timestamp",
    "exit_timestamp",
)


@dataclass(frozen=True)
class AzirEventRewardConfig:
    mode: str = "protected_net_pnl_minus_risk_penalties"
    drawdown_penalty_weight: float = 0.05
    risk_tension_penalty_weight: float = 0.25
    risk_blocked_penalty: float = 0.25
    invalid_take_penalty: float = 0.05


@dataclass(frozen=True)
class AzirReplayEvent:
    setup_day: str
    timestamp: datetime
    setup: dict[str, Any]
    outcome: dict[str, Any] = field(default_factory=dict)
    lifecycle: dict[str, Any] = field(default_factory=dict)

    @property
    def order_placed(self) -> bool:
        return _is_true(self.setup.get("buy_order_placed")) or _is_true(self.setup.get("sell_order_placed"))

    @property
    def protected_net_pnl(self) -> float:
        return _to_float(self.outcome.get("protected_net_pnl")) or 0.0

    @property
    def protected_gross_pnl(self) -> float:
        return _to_float(self.outcome.get("protected_gross_pnl")) or self.protected_net_pnl

    @property
    def has_protected_fill(self) -> bool:
        return bool(self.outcome.get("has_protected_fill"))


@dataclass
class AzirEnvAccountState:
    initial_equity: float = 10000.0
    equity: float = 10000.0
    peak_equity: float = 10000.0
    current_day: str = ""
    day_start_equity: float = 10000.0
    daily_realized_pnl: float = 0.0
    consecutive_losses_today: int = 0
    trades_today: int = 0
    total_trades_taken: int = 0


class AzirEventReplayEnvironment(gym.Env):
    """Gym-compatible Azir replay environment over daily setup events."""

    metadata = {"render_modes": []}
    ACTION_SKIP = ACTION_SKIP
    ACTION_TAKE = ACTION_TAKE

    def __init__(
        self,
        events: list[AzirReplayEvent],
        *,
        risk_config: AzirRiskConfig | None = None,
        reward_config: AzirEventRewardConfig | None = None,
        initial_equity: float = 10000.0,
    ) -> None:
        if not events:
            raise ValueError("AzirEventReplayEnvironment requires at least one replay event.")
        _validate_observation_schema()
        self.events = sorted(events, key=lambda event: event.timestamp)
        self.risk_config = risk_config or AzirRiskConfig(starting_equity=initial_equity)
        self.reward_config = reward_config or AzirEventRewardConfig()
        self.risk_engine = AzirRiskEngine(self.risk_config)
        self.initial_equity = float(initial_equity)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(OBSERVATION_FIELDS),), dtype=np.float32)
        self._cursor = 0
        self._account = AzirEnvAccountState(initial_equity=initial_equity, equity=initial_equity, peak_equity=initial_equity)

    @property
    def observation_fields(self) -> tuple[str, ...]:
        return OBSERVATION_FIELDS

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            super().reset(seed=seed, options=options)
        except TypeError:
            super().reset(seed=seed)
        self._cursor = 0
        first_day = self.events[0].setup_day
        self._account = AzirEnvAccountState(
            initial_equity=self.initial_equity,
            equity=self.initial_equity,
            peak_equity=self.initial_equity,
            current_day=first_day,
            day_start_equity=self.initial_equity,
        )
        event = self.events[self._cursor]
        risk_decision = self._evaluate_risk(event)
        return self._observation(event, risk_decision), self._info(event, risk_decision, action_effect="reset")

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid Azir action: {action}")
        event = self.events[self._cursor]
        self._roll_day_if_needed(event)
        risk_decision = self._evaluate_risk(event)
        reward, breakdown, action_effect = self._apply_action(event, int(action), risk_decision)
        self._cursor += 1
        terminated = self._cursor >= len(self.events)
        if terminated:
            observation = np.zeros(len(OBSERVATION_FIELDS), dtype=np.float32)
        else:
            next_event = self.events[self._cursor]
            self._roll_day_if_needed(next_event)
            next_risk = self._evaluate_risk(next_event)
            observation = self._observation(next_event, next_risk)
        info = self._info(
            event,
            risk_decision,
            action=int(action),
            action_effect=action_effect,
            reward_breakdown=breakdown,
            valid_actions=self.valid_actions(event, risk_decision),
        )
        return observation, float(reward), terminated, False, info

    def valid_actions(self, event: AzirReplayEvent | None = None, risk_decision: AzirRiskDecision | None = None) -> tuple[int, ...]:
        event = event or self.events[self._cursor]
        risk_decision = risk_decision or self._evaluate_risk(event)
        if event.order_placed and risk_decision.approved:
            return (ACTION_SKIP, ACTION_TAKE)
        return (ACTION_SKIP,)

    def current_event(self) -> AzirReplayEvent:
        return self.events[self._cursor]

    def _apply_action(
        self,
        event: AzirReplayEvent,
        action: int,
        risk_decision: AzirRiskDecision,
    ) -> tuple[float, dict[str, Any], str]:
        risk_tension = self._risk_tension_ratio()
        if action == ACTION_SKIP:
            return 0.0, self._reward_breakdown(0.0, 0.0, 0.0, 0.0), "skip"
        if not risk_decision.approved:
            penalty = self.reward_config.risk_blocked_penalty
            return -penalty, self._reward_breakdown(0.0, 0.0, risk_tension, penalty), "risk_blocked_take_transformed_to_skip"
        if not event.order_placed:
            penalty = self.reward_config.invalid_take_penalty
            return -penalty, self._reward_breakdown(0.0, 0.0, risk_tension, penalty), "invalid_take_no_azir_order_transformed_to_skip"

        pnl = event.protected_net_pnl if event.has_protected_fill else 0.0
        self._account.equity += pnl
        self._account.peak_equity = max(self._account.peak_equity, self._account.equity)
        self._account.daily_realized_pnl += pnl
        self._account.trades_today += 1 if event.has_protected_fill else 0
        self._account.total_trades_taken += 1
        self._account.consecutive_losses_today = self._account.consecutive_losses_today + 1 if pnl < 0 else 0
        daily_dd = max(0.0, self._account.day_start_equity - self._account.equity)
        total_dd = max(0.0, self._account.peak_equity - self._account.equity)
        dd_penalty = self.reward_config.drawdown_penalty_weight * (daily_dd + total_dd)
        tension_penalty = self.reward_config.risk_tension_penalty_weight * risk_tension
        reward = pnl - dd_penalty - tension_penalty
        return reward, self._reward_breakdown(pnl, dd_penalty, risk_tension, 0.0), "take"

    def _observation(self, event: AzirReplayEvent, risk_decision: AzirRiskDecision) -> np.ndarray:
        values = self._observation_dict(event, risk_decision)
        return np.asarray([values[name] for name in OBSERVATION_FIELDS], dtype=np.float32)

    def _observation_dict(self, event: AzirReplayEvent, risk_decision: AzirRiskDecision) -> dict[str, float]:
        setup = event.setup
        daily_dd = max(0.0, self._account.day_start_equity - self._account.equity)
        total_dd = max(0.0, self._account.peak_equity - self._account.equity)
        max_daily_loss = abs(self.risk_config.max_daily_loss)
        remaining_daily_loss = max(0.0, max_daily_loss + self._account.daily_realized_pnl)
        prior_exposure = bool(event.lifecycle.get("survived_change_of_day")) or int(event.lifecycle.get("out_of_window_fill_count") or 0) > 0
        cleanup_issue = bool(
            event.lifecycle.get("order_placed")
            and event.lifecycle.get("cleanup_count", 0) == 0
            and (
                event.lifecycle.get("lifecycle_status") == "missing_cleanup_or_unresolved"
                or prior_exposure
            )
        )
        return {
            "setup_hour": float(event.timestamp.hour),
            "day_of_week": _float_or_zero(setup.get("day_of_week")),
            "month": float(event.timestamp.month),
            "is_friday": _bool_float(_is_true(setup.get("is_friday")) or setup.get("event_type") == "blocked_friday"),
            "buy_order_placed": _bool_float(_is_true(setup.get("buy_order_placed"))),
            "sell_order_placed": _bool_float(_is_true(setup.get("sell_order_placed"))),
            "buy_allowed_by_trend": _bool_float(_is_true(setup.get("buy_allowed_by_trend"))),
            "sell_allowed_by_trend": _bool_float(_is_true(setup.get("sell_allowed_by_trend"))),
            "swing_high": _float_or_zero(setup.get("swing_high")),
            "swing_low": _float_or_zero(setup.get("swing_low")),
            "buy_entry": _float_or_zero(setup.get("buy_entry")),
            "sell_entry": _float_or_zero(setup.get("sell_entry")),
            "pending_distance_points": _float_or_zero(setup.get("pending_distance_points")),
            "spread_points": _float_or_zero(setup.get("spread_points")),
            "ema20": _float_or_zero(setup.get("ema20")),
            "prev_close_vs_ema20_points": _float_or_zero(setup.get("prev_close_vs_ema20_points")),
            "atr": _float_or_zero(setup.get("atr")),
            "atr_points": _float_or_zero(setup.get("atr_points")),
            "rsi": _float_or_zero(setup.get("rsi")),
            "trend_filter_enabled": _bool_float(_is_true(setup.get("trend_filter_enabled"))),
            "atr_filter_enabled": _bool_float(_is_true(setup.get("atr_filter_enabled"))),
            "atr_filter_passed": _bool_float(_is_true(setup.get("atr_filter_passed"))),
            "rsi_gate_enabled": _bool_float(_is_true(setup.get("rsi_gate_enabled"))),
            "rsi_gate_required": _bool_float(_is_true(setup.get("rsi_gate_required"))),
            "prior_exposure_flag": _bool_float(prior_exposure),
            "cleanup_issue_before_risk": _bool_float(cleanup_issue),
            "daily_realized_pnl": float(self._account.daily_realized_pnl),
            "daily_drawdown_abs": float(daily_dd),
            "total_drawdown_abs": float(total_dd),
            "consecutive_losses_today": float(self._account.consecutive_losses_today),
            "trades_today": float(self._account.trades_today),
            "remaining_daily_loss": float(remaining_daily_loss),
            "risk_tension_ratio": float(self._risk_tension_ratio()),
            "risk_engine_approved": _bool_float(risk_decision.approved),
            "risk_blocked_flag": _bool_float(not risk_decision.approved),
        }

    def _evaluate_risk(self, event: AzirReplayEvent) -> AzirRiskDecision:
        state = AzirRiskState(
            timestamp=event.timestamp,
            pending_orders=0,
            open_positions=0,
            daily_realized_pnl=self._account.daily_realized_pnl,
            trades_today=self._account.trades_today,
            consecutive_losses_today=self._account.consecutive_losses_today,
            spread_points=_to_float(event.setup.get("spread_points")),
            reconciled=True,
        )
        return self.risk_engine.evaluate(state, context="setup_attempt")

    def _roll_day_if_needed(self, event: AzirReplayEvent) -> None:
        if self._account.current_day == event.setup_day:
            return
        self._account.current_day = event.setup_day
        self._account.day_start_equity = self._account.equity
        self._account.daily_realized_pnl = 0.0
        self._account.consecutive_losses_today = 0
        self._account.trades_today = 0

    def _risk_tension_ratio(self) -> float:
        limit = abs(self.risk_config.max_daily_loss)
        if limit <= 0.0:
            return 0.0
        used = max(0.0, -self._account.daily_realized_pnl)
        return min(1.0, used / limit)

    def _reward_breakdown(
        self,
        pnl: float,
        drawdown_penalty: float,
        risk_tension: float,
        invalid_penalty: float,
    ) -> dict[str, float | str]:
        tension_penalty = self.reward_config.risk_tension_penalty_weight * risk_tension
        return {
            "mode": self.reward_config.mode,
            "protected_net_pnl": float(pnl),
            "drawdown_penalty": float(drawdown_penalty),
            "risk_tension_penalty": float(tension_penalty),
            "invalid_action_penalty": float(invalid_penalty),
            "reward": float(pnl - drawdown_penalty - tension_penalty - invalid_penalty),
        }

    def _info(
        self,
        event: AzirReplayEvent,
        risk_decision: AzirRiskDecision,
        *,
        action: int = ACTION_SKIP,
        action_effect: str,
        reward_breakdown: dict[str, Any] | None = None,
        valid_actions: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        return {
            "setup_day": event.setup_day,
            "timestamp": event.timestamp.isoformat(sep=" "),
            "action": action,
            "action_effect": action_effect,
            "valid_actions": tuple(valid_actions or self.valid_actions(event, risk_decision)),
            "risk_approved": risk_decision.approved,
            "risk_reason_code": risk_decision.reason_code,
            "risk_blocked_by": risk_decision.blocked_by,
            "risk_actions": risk_decision.actions,
            "order_placed": event.order_placed,
            "has_protected_fill": event.has_protected_fill,
            "reward_breakdown": reward_breakdown or {},
            "equity": self._account.equity,
            "observation_schema": OBSERVATION_FIELDS,
        }


def build_azir_event_replay_dataset(
    *,
    mt5_log_path: Path,
    protected_report_path: Path | None = None,
    m1_input_path: Path | None = None,
    symbol: str = DEFAULT_SYMBOL,
    risk_config: AzirRiskConfig | None = None,
) -> list[AzirReplayEvent]:
    risk_config = risk_config or AzirRiskConfig()
    rows = _read_raw_event_log(mt5_log_path, symbol)
    reconstruction = reconstruct_lifecycles(
        rows,
        session_start_hour=risk_config.session_fill_start_hour,
        session_end_hour=risk_config.session_fill_end_hour,
        close_hour=risk_config.close_hour,
    )
    simulation = apply_risk_engine_to_lifecycle(
        rows=rows,
        lifecycle_rows=reconstruction["lifecycles"],
        trade_rows=reconstruction["trades"],
        config=risk_config,
    )
    outcomes = _protected_outcomes_by_day(simulation["protected_trades"])
    forced_cases = _load_forced_cases(protected_report_path)
    if forced_cases:
        _merge_forced_case_outcomes(outcomes, forced_cases)
    elif m1_input_path is not None:
        bars = load_ohlcv_csv(m1_input_path)
        forced_decisions = [row for row in simulation["trade_decisions"] if row["risk_status"] == "forced_close_unpriced"]
        cases = [
            revalue_forced_close(
                decision=row,
                bars=bars,
                config=risk_config,
                lot_size=0.10,
                contract_size=100.0,
            )
            for row in forced_decisions
        ]
        _merge_forced_case_outcomes(outcomes, cases)

    setup_rows = _canonical_setup_rows(rows)
    lifecycles_by_day = {row["setup_day"]: row for row in reconstruction["lifecycles"]}
    events = [
        AzirReplayEvent(
            setup_day=day,
            timestamp=_parse_timestamp(setup.get("timestamp")),
            setup=setup,
            outcome=outcomes.get(day, {}),
            lifecycle=lifecycles_by_day.get(day, {}),
        )
        for day, setup in sorted(setup_rows.items())
    ]
    return events


def write_azir_env_inspection_artifacts(
    *,
    env: AzirEventReplayEnvironment,
    output_dir: Path,
    sample_events: int = 20,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    observation, info = env.reset(seed=123)
    schema = {
        "environment": "AzirEventReplayEnvironment",
        "unit": "daily_azir_setup_event",
        "actions": {"0": "skip", "1": "take"},
        "observation_fields": list(OBSERVATION_FIELDS),
        "forbidden_observation_fields": list(FORBIDDEN_OBSERVATION_FIELDS),
        "initial_observation_shape": list(observation.shape),
        "notes": [
            "Future PnL/outcome fields are intentionally excluded from observations.",
            "Risk Engine approval flags are observable because live risk state is part of the control layer.",
            "Action 1 is transformed to skip when risk_engine_azir_v1 blocks a setup.",
        ],
    }
    observation_rows = []
    valid_action_rows = []
    episode_rows = []
    for _ in range(min(sample_events, len(env.events))):
        event = env.current_event()
        risk = env._evaluate_risk(event)
        obs_dict = env._observation_dict(event, risk)
        observation_rows.append({"setup_day": event.setup_day, **obs_dict})
        valid_action_rows.append(
            {
                "setup_day": event.setup_day,
                "valid_actions": "|".join(str(action) for action in env.valid_actions(event, risk)),
                "risk_approved": risk.approved,
                "risk_reason_code": risk.reason_code,
                "order_placed": event.order_placed,
            }
        )
        action = ACTION_TAKE if ACTION_TAKE in env.valid_actions(event, risk) else ACTION_SKIP
        _, reward, terminated, _, step_info = env.step(action)
        episode_rows.append(
            {
                "setup_day": event.setup_day,
                "action": action,
                "reward": _round(reward),
                "action_effect": step_info["action_effect"],
                "risk_approved": step_info["risk_approved"],
                "equity": _round(step_info["equity"]),
            }
        )
        if terminated:
            break
    (output_dir / "azir_rl_observation_schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(observation_rows, output_dir / "azir_rl_sample_observations.csv")
    _write_csv(valid_action_rows, output_dir / "azir_rl_valid_actions.csv")
    (output_dir / "azir_rl_sample_episode.json").write_text(json.dumps(episode_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"schema": schema, "initial_info": info, "sample_steps": episode_rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the Azir event-based RL environment contract.")
    parser.add_argument("--mt5-log-path", required=True)
    parser.add_argument("--protected-report-path", default="")
    parser.add_argument("--m1-input-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--sample-events", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    events = build_azir_event_replay_dataset(
        mt5_log_path=Path(args.mt5_log_path),
        protected_report_path=Path(args.protected_report_path) if args.protected_report_path else None,
        m1_input_path=Path(args.m1_input_path) if args.m1_input_path else None,
        symbol=args.symbol,
    )
    env = AzirEventReplayEnvironment(events)
    artifacts = write_azir_env_inspection_artifacts(env=env, output_dir=Path(args.output_dir), sample_events=args.sample_events)
    print(
        json.dumps(
            {
                "environment": "AzirEventReplayEnvironment",
                "events": len(events),
                "actions": {"0": "skip", "1": "take"},
                "observation_fields": len(OBSERVATION_FIELDS),
                "output_dir": args.output_dir,
                "first_sample_reward": artifacts["sample_steps"][0]["reward"] if artifacts["sample_steps"] else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _canonical_setup_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("event_type") not in {"opportunity", "blocked_friday"}:
            continue
        grouped.setdefault(str(row["_event_day"]), []).append(row)
    return {day: _canonical_setup_row(day_rows) for day, day_rows in grouped.items()}


def _canonical_setup_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    opportunities = [row for row in rows if row.get("event_type") == "opportunity"]
    if opportunities:
        placed = [row for row in opportunities if _is_true(row.get("buy_order_placed")) or _is_true(row.get("sell_order_placed"))]
        return placed[-1] if placed else opportunities[0]
    return rows[0]


def _protected_outcomes_by_day(protected_trades: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    outcomes: dict[str, dict[str, Any]] = {}
    for trade in protected_trades:
        day = str(trade.get("setup_day", ""))
        outcome = outcomes.setdefault(day, {"protected_net_pnl": 0.0, "protected_gross_pnl": 0.0, "has_protected_fill": False})
        outcome["protected_net_pnl"] += _to_float(trade.get("net_pnl")) or 0.0
        outcome["protected_gross_pnl"] += _to_float(trade.get("gross_pnl")) or 0.0
        outcome["has_protected_fill"] = True
        outcome["exit_reason"] = trade.get("exit_reason", "")
    return outcomes


def _load_forced_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    report = json.loads(path.read_text(encoding="utf-8"))
    return list(report.get("forced_close_cases", []))


def _merge_forced_case_outcomes(outcomes: dict[str, dict[str, Any]], cases: list[dict[str, Any]]) -> None:
    for case in cases:
        if case.get("revaluation_status") != "priced_with_m1_proxy":
            continue
        day = str(case.get("setup_day", ""))
        outcome = outcomes.setdefault(day, {"protected_net_pnl": 0.0, "protected_gross_pnl": 0.0, "has_protected_fill": False})
        outcome["protected_net_pnl"] += _to_float(case.get("revalued_net_pnl")) or 0.0
        outcome["protected_gross_pnl"] += _to_float(case.get("revalued_gross_pnl")) or 0.0
        outcome["has_protected_fill"] = True
        outcome["exit_reason"] = "risk_engine_forced_close_revalued"


def _validate_observation_schema() -> None:
    leaked = set(OBSERVATION_FIELDS) & set(FORBIDDEN_OBSERVATION_FIELDS)
    if leaked:
        raise ValueError(f"Observation schema contains future/outcome fields: {sorted(leaked)}")


def _float_or_zero(value: Any) -> float:
    parsed = _to_float(value)
    return 0.0 if parsed is None else float(parsed)


def _bool_float(value: bool) -> float:
    return 1.0 if value else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
