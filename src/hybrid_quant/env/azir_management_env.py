"""Management-action RL contract for Azir protected trades.

This environment is intentionally not a PPO training sprint.  It defines the
next contract after skip/take failed to add value: Azir + Risk Engine take the
trade, then an agent may choose a discrete management policy.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from hybrid_quant.azir.economic_audit import (
    _is_true,
    _parse_timestamp,
    _read_raw_event_log,
    _round,
    _to_float,
    _write_csv,
    reconstruct_lifecycles,
)
from hybrid_quant.azir.protected_benchmark_freeze import _build_revalued_trades
from hybrid_quant.azir.risk_reaudit import apply_risk_engine_to_lifecycle
from hybrid_quant.risk.azir_state import AzirRiskConfig

try:  # pragma: no cover - project/runtime path normally has numpy and gymnasium.
    import numpy as np

    from .gym_compat import gym, spaces
except ModuleNotFoundError:  # pragma: no cover - minimal local fallback.
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


ACTION_BASE_MANAGEMENT = 0
ACTION_CLOSE_EARLY = 1
ACTION_MOVE_TO_BREAK_EVEN = 2
ACTION_TRAILING_CONSERVATIVE = 3
ACTION_TRAILING_AGGRESSIVE = 4


MANAGEMENT_ACTIONS: dict[int, str] = {
    ACTION_BASE_MANAGEMENT: "base_management",
    ACTION_CLOSE_EARLY: "close_early",
    ACTION_MOVE_TO_BREAK_EVEN: "move_to_break_even",
    ACTION_TRAILING_CONSERVATIVE: "trailing_conservative",
    ACTION_TRAILING_AGGRESSIVE: "trailing_aggressive",
}


MANAGEMENT_OBSERVATION_FIELDS: tuple[str, ...] = (
    "side_buy",
    "side_sell",
    "fill_hour_sin",
    "fill_hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "duration_to_fill_minutes_scaled",
    "fill_price_vs_buy_entry_atr",
    "fill_price_vs_sell_entry_atr",
    "pending_distance_atr",
    "spread_atr",
    "swing_width_atr",
    "prev_close_vs_ema_atr",
    "atr_points_scaled",
    "rsi_centered",
    "trend_filter_enabled",
    "atr_filter_passed",
    "rsi_gate_required",
    "buy_allowed_by_trend",
    "sell_allowed_by_trend",
    "trailing_start_atr",
    "trailing_step_atr",
)


FORBIDDEN_MANAGEMENT_OBSERVATION_FIELDS: tuple[str, ...] = (
    "protected_net_pnl",
    "gross_pnl",
    "net_pnl",
    "exit_reason",
    "exit_timestamp",
    "mfe_points",
    "mae_points",
    "trailing_activated",
    "trailing_modifications",
)


@dataclass(frozen=True)
class AzirManagementRewardConfig:
    """Counterfactual management reward proxy.

    The default mode is deliberately labelled as a proxy.  It is useful for
    contract tests and heuristic research, but it is not a frozen economic
    benchmark until tick/M1 price replay prices each management action.
    """

    mode: str = "observational_proxy_v1"
    break_even_activation_points: float = 90.0
    conservative_trailing_activation_points: float = 120.0
    aggressive_trailing_activation_points: float = 60.0
    conservative_profit_haircut: float = 0.15
    aggressive_profit_haircut: float = 0.35
    close_early_profit_fraction: float = 0.35
    close_early_loss_fraction: float = 0.50
    counterfactual_penalty: float = 0.0


@dataclass(frozen=True)
class AzirManagementEvent:
    setup_day: str
    fill_timestamp: datetime
    setup: dict[str, Any]
    trade: dict[str, Any]
    lifecycle: dict[str, Any]

    @property
    def protected_net_pnl(self) -> float:
        return _to_float(self.trade.get("net_pnl")) or 0.0

    @property
    def side(self) -> str:
        return str(self.trade.get("fill_side", "")).strip().lower()

    @property
    def mfe_points(self) -> float:
        return _to_float(self.trade.get("mfe_points")) or 0.0

    @property
    def mae_points(self) -> float:
        return _to_float(self.trade.get("mae_points")) or 0.0


class AzirManagementReplayEnvironment(gym.Env):
    """One-step replay environment over protected Azir trades."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        events: list[AzirManagementEvent],
        *,
        reward_config: AzirManagementRewardConfig | None = None,
    ) -> None:
        if not events:
            raise ValueError("AzirManagementReplayEnvironment requires at least one management event.")
        self.events = sorted(events, key=lambda event: event.fill_timestamp)
        self.reward_config = reward_config or AzirManagementRewardConfig()
        self.action_space = spaces.Discrete(len(MANAGEMENT_ACTIONS))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(MANAGEMENT_OBSERVATION_FIELDS),),
            dtype=np.float32,
        )
        self._cursor = 0

    @property
    def observation_fields(self) -> tuple[str, ...]:
        return MANAGEMENT_OBSERVATION_FIELDS

    @property
    def action_labels(self) -> dict[int, str]:
        return MANAGEMENT_ACTIONS

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            super().reset(seed=seed, options=options)
        except TypeError:
            super().reset(seed=seed)
        self._cursor = 0
        event = self.events[self._cursor]
        return self._observation(event), self._info(event, action_effect="reset")

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid Azir management action: {action}")
        event = self.events[self._cursor]
        reward, breakdown, effect = self._apply_management_action(event, int(action))
        self._cursor += 1
        terminated = self._cursor >= len(self.events)
        observation = (
            np.zeros(len(MANAGEMENT_OBSERVATION_FIELDS), dtype=np.float32)
            if terminated
            else self._observation(self.events[self._cursor])
        )
        info = self._info(event, action=int(action), action_effect=effect, reward_breakdown=breakdown)
        return observation, float(reward), terminated, False, info

    def current_event(self) -> AzirManagementEvent:
        return self.events[self._cursor]

    def valid_actions(self, event: AzirManagementEvent | None = None) -> tuple[int, ...]:
        event = event or self.events[self._cursor]
        if event.protected_net_pnl == 0.0 and event.mfe_points == 0.0 and event.mae_points == 0.0:
            return (ACTION_BASE_MANAGEMENT,)
        return tuple(MANAGEMENT_ACTIONS)

    def _observation(self, event: AzirManagementEvent) -> np.ndarray:
        values = self._observation_dict(event)
        return np.asarray([values[name] for name in MANAGEMENT_OBSERVATION_FIELDS], dtype=np.float32)

    def _observation_dict(self, event: AzirManagementEvent) -> dict[str, float]:
        setup = event.setup
        trade = event.trade
        atr_points = max(_float_or_zero(setup.get("atr_points")), 1e-9)
        atr = max(_float_or_zero(setup.get("atr")), 1e-9)
        swing_high = _float_or_zero(setup.get("swing_high"))
        swing_low = _float_or_zero(setup.get("swing_low"))
        fill_price = _float_or_zero(trade.get("fill_price"))
        day_of_week = _float_or_zero(setup.get("day_of_week"))
        return {
            "side_buy": _bool_float(event.side == "buy"),
            "side_sell": _bool_float(event.side == "sell"),
            "fill_hour_sin": _cyclical_sin(float(event.fill_timestamp.hour), 24.0),
            "fill_hour_cos": _cyclical_cos(float(event.fill_timestamp.hour), 24.0),
            "day_of_week_sin": _cyclical_sin(day_of_week, 7.0),
            "day_of_week_cos": _cyclical_cos(day_of_week, 7.0),
            "month_sin": _cyclical_sin(float(event.fill_timestamp.month), 12.0),
            "month_cos": _cyclical_cos(float(event.fill_timestamp.month), 12.0),
            "duration_to_fill_minutes_scaled": _safe_ratio(_float_or_zero(trade.get("duration_to_fill_seconds")), 3600.0),
            "fill_price_vs_buy_entry_atr": _safe_ratio(fill_price - _float_or_zero(setup.get("buy_entry")), atr),
            "fill_price_vs_sell_entry_atr": _safe_ratio(fill_price - _float_or_zero(setup.get("sell_entry")), atr),
            "pending_distance_atr": _safe_ratio(_float_or_zero(setup.get("pending_distance_points")), atr_points),
            "spread_atr": _safe_ratio(_float_or_zero(setup.get("spread_points")), atr_points),
            "swing_width_atr": _safe_ratio(swing_high - swing_low, atr),
            "prev_close_vs_ema_atr": _safe_ratio(_float_or_zero(setup.get("prev_close_vs_ema20_points")), atr_points),
            "atr_points_scaled": _safe_ratio(atr_points, 1000.0),
            "rsi_centered": _safe_ratio(_float_or_zero(setup.get("rsi")) - 50.0, 50.0),
            "trend_filter_enabled": _bool_float(_is_true(setup.get("trend_filter_enabled"))),
            "atr_filter_passed": _bool_float(_is_true(setup.get("atr_filter_passed"))),
            "rsi_gate_required": _bool_float(_is_true(setup.get("rsi_gate_required"))),
            "buy_allowed_by_trend": _bool_float(_is_true(setup.get("buy_allowed_by_trend"))),
            "sell_allowed_by_trend": _bool_float(_is_true(setup.get("sell_allowed_by_trend"))),
            "trailing_start_atr": _safe_ratio(_float_or_zero(setup.get("trailing_start_points")), atr_points),
            "trailing_step_atr": _safe_ratio(_float_or_zero(setup.get("trailing_step_points")), atr_points),
        }

    def _apply_management_action(self, event: AzirManagementEvent, action: int) -> tuple[float, dict[str, Any], str]:
        base_pnl = event.protected_net_pnl
        proxy_pnl = base_pnl
        action_label = MANAGEMENT_ACTIONS[action]
        pricing_confidence = "observed" if action == ACTION_BASE_MANAGEMENT else "proxy_requires_price_replay"
        if action == ACTION_CLOSE_EARLY:
            proxy_pnl = self._close_early_proxy(base_pnl)
        elif action == ACTION_MOVE_TO_BREAK_EVEN:
            proxy_pnl = self._break_even_proxy(event, base_pnl)
        elif action == ACTION_TRAILING_CONSERVATIVE:
            proxy_pnl = self._trailing_proxy(
                event,
                base_pnl,
                activation_points=self.reward_config.conservative_trailing_activation_points,
                profit_haircut=self.reward_config.conservative_profit_haircut,
            )
        elif action == ACTION_TRAILING_AGGRESSIVE:
            proxy_pnl = self._trailing_proxy(
                event,
                base_pnl,
                activation_points=self.reward_config.aggressive_trailing_activation_points,
                profit_haircut=self.reward_config.aggressive_profit_haircut,
            )
        reward = proxy_pnl - (0.0 if action == ACTION_BASE_MANAGEMENT else self.reward_config.counterfactual_penalty)
        return reward, {
            "mode": self.reward_config.mode,
            "action_label": action_label,
            "base_protected_net_pnl": float(base_pnl),
            "management_proxy_net_pnl": float(proxy_pnl),
            "counterfactual_penalty": 0.0 if action == ACTION_BASE_MANAGEMENT else float(self.reward_config.counterfactual_penalty),
            "reward": float(reward),
            "pricing_confidence": pricing_confidence,
            "mfe_points_used_for_proxy": float(event.mfe_points),
            "mae_points_used_for_proxy": float(event.mae_points),
        }, action_label

    def _close_early_proxy(self, base_pnl: float) -> float:
        if base_pnl >= 0.0:
            return base_pnl * self.reward_config.close_early_profit_fraction
        return base_pnl * self.reward_config.close_early_loss_fraction

    def _break_even_proxy(self, event: AzirManagementEvent, base_pnl: float) -> float:
        if event.mfe_points >= self.reward_config.break_even_activation_points and base_pnl < 0.0:
            return 0.0
        return base_pnl

    def _trailing_proxy(self, event: AzirManagementEvent, base_pnl: float, *, activation_points: float, profit_haircut: float) -> float:
        if event.mfe_points < activation_points:
            return base_pnl
        if base_pnl < 0.0:
            return 0.0
        return base_pnl * (1.0 - profit_haircut)

    def _info(
        self,
        event: AzirManagementEvent,
        *,
        action: int = ACTION_BASE_MANAGEMENT,
        action_effect: str,
        reward_breakdown: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "setup_day": event.setup_day,
            "fill_timestamp": event.fill_timestamp.isoformat(sep=" "),
            "action": action,
            "action_label": MANAGEMENT_ACTIONS.get(action, "unknown"),
            "action_effect": action_effect,
            "valid_actions": self.valid_actions(event),
            "side": event.side,
            "observation_schema": MANAGEMENT_OBSERVATION_FIELDS,
            "reward_breakdown": reward_breakdown or {},
        }


def build_azir_management_replay_dataset(
    *,
    mt5_log_path: Path,
    protected_report_path: Path | None = None,
    symbol: str = "XAUUSD-STD",
    risk_config: AzirRiskConfig | None = None,
) -> list[AzirManagementEvent]:
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
    protected_trades = list(simulation["protected_trades"])
    forced_cases = _load_forced_cases(protected_report_path)
    if forced_cases:
        protected_trades = _build_revalued_trades(protected_trades, reconstruction["trades"], forced_cases)
    setup_rows = _canonical_setup_rows(rows)
    lifecycles_by_day = {row["setup_day"]: row for row in reconstruction["lifecycles"]}
    events: list[AzirManagementEvent] = []
    for trade in protected_trades:
        setup_day = str(trade.get("setup_day", ""))
        fill_timestamp = str(trade.get("fill_timestamp", ""))
        setup = setup_rows.get(setup_day)
        if not setup or not fill_timestamp:
            continue
        events.append(
            AzirManagementEvent(
                setup_day=setup_day,
                fill_timestamp=_parse_timestamp(fill_timestamp),
                setup=setup,
                trade=trade,
                lifecycle=lifecycles_by_day.get(setup_day, {}),
            )
        )
    return sorted(events, key=lambda event: event.fill_timestamp)


def write_azir_management_env_artifacts(
    *,
    env: AzirManagementReplayEnvironment,
    output_dir: Path,
    sample_events: int = 20,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    observation, info = env.reset(seed=123)
    schema = {
        "environment": "AzirManagementReplayEnvironment",
        "sprint": "design_management_actions_for_azir_v1",
        "unit": "one protected Azir trade after fill",
        "decision_timing": "first post-fill management checkpoint; exact tick-level checkpoint must be priced in the next sprint",
        "actions": {str(key): value for key, value in MANAGEMENT_ACTIONS.items()},
        "observation_fields": list(env.observation_fields),
        "forbidden_observation_fields": list(FORBIDDEN_MANAGEMENT_OBSERVATION_FIELDS),
        "initial_observation_shape": list(observation.shape),
        "reward_mode": env.reward_config.mode,
        "reward_status": "proxy_not_frozen_economic_benchmark",
    }
    observation_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for _ in range(min(sample_events, len(env.events))):
        event = env.current_event()
        observation_rows.append({"setup_day": event.setup_day, "fill_timestamp": event.fill_timestamp.isoformat(sep=" "), **env._observation_dict(event)})
        _, reward, terminated, _, step_info = env.step(ACTION_BASE_MANAGEMENT)
        episode_rows.append(
            {
                "setup_day": event.setup_day,
                "fill_timestamp": event.fill_timestamp.isoformat(sep=" "),
                "action": ACTION_BASE_MANAGEMENT,
                "action_label": MANAGEMENT_ACTIONS[ACTION_BASE_MANAGEMENT],
                "reward": _round(reward),
                "pricing_confidence": step_info["reward_breakdown"].get("pricing_confidence", ""),
                "base_protected_net_pnl": step_info["reward_breakdown"].get("base_protected_net_pnl", ""),
            }
        )
        if terminated:
            break
    (output_dir / "azir_management_action_contract.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(observation_rows, output_dir / "azir_management_sample_observations.csv")
    _write_csv(episode_rows, output_dir / "azir_management_sample_episode.csv")
    (output_dir / "azir_management_env_summary.md").write_text(_summary_markdown(schema, len(env.events)), encoding="utf-8")
    return {"schema": schema, "initial_info": info, "sample_steps": episode_rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect Azir management-action RL environment contract.")
    parser.add_argument("--mt5-log-path", required=True)
    parser.add_argument("--protected-report-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--symbol", default="XAUUSD-STD")
    parser.add_argument("--sample-events", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    events = build_azir_management_replay_dataset(
        mt5_log_path=Path(args.mt5_log_path),
        protected_report_path=Path(args.protected_report_path) if args.protected_report_path else None,
        symbol=args.symbol,
    )
    env = AzirManagementReplayEnvironment(events)
    artifacts = write_azir_management_env_artifacts(env=env, output_dir=Path(args.output_dir), sample_events=args.sample_events)
    print(
        json.dumps(
            {
                "environment": "AzirManagementReplayEnvironment",
                "events": len(events),
                "actions": {str(key): value for key, value in MANAGEMENT_ACTIONS.items()},
                "observation_fields": len(env.observation_fields),
                "reward_status": artifacts["schema"]["reward_status"],
                "output_dir": args.output_dir,
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


def _load_forced_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    report = json.loads(path.read_text(encoding="utf-8"))
    return list(report.get("forced_close_cases", []))


def _summary_markdown(schema: dict[str, Any], events: int) -> str:
    actions = "\n".join(f"- `{key}` = `{label}`" for key, label in schema["actions"].items())
    return (
        "# Azir Management RL Environment Contract\n\n"
        "## Summary\n\n"
        f"- Environment: `{schema['environment']}`.\n"
        f"- Replay events: {events} protected trades.\n"
        f"- Unit: {schema['unit']}.\n"
        f"- Reward status: `{schema['reward_status']}`.\n\n"
        "## Actions\n\n"
        f"{actions}\n\n"
        "## Important Limitation\n\n"
        "Only `base_management` uses observed protected benchmark PnL. All alternative management actions "
        "currently use an observational proxy and must be repriced with M1/tick replay before PPO training.\n"
    )


def _float_or_zero(value: Any) -> float:
    parsed = _to_float(value)
    return 0.0 if parsed is None else float(parsed)


def _bool_float(value: bool) -> float:
    return 1.0 if value else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def _cyclical_sin(value: float, period: float) -> float:
    if period <= 0.0:
        return 0.0
    return float(math.sin(2.0 * math.pi * value / period))


def _cyclical_cos(value: float, period: float) -> float:
    if period <= 0.0:
        return 0.0
    return float(math.cos(2.0 * math.pi * value / period))


if __name__ == "__main__":
    raise SystemExit(main())
