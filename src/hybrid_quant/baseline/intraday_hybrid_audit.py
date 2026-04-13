from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from hybrid_quant.core import Settings, apply_settings_overrides
from hybrid_quant.data import read_ohlcv_frame

from .diagnostics import BaselineDiagnosticsRunner
from .intraday_hybrid_research import load_intraday_hybrid_research_config
from .orb_intraday_active_research import _build_runner_from_settings, _filter_frame_by_range, _sanitize_value
from .variants import load_variant_settings, variant_summary_payload


@dataclass(slots=True)
class AuditArtifacts:
    output_dir: Path
    report_path: Path
    summary_path: Path
    split_metrics_path: Path
    sensitivity_path: Path
    ablation_path: Path
    risk_path_path: Path
    equity_curve_path: Path
    trades_path: Path


def run_audit(
    *,
    config_dir: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    variant: str = "baseline_intraday_hybrid",
    experiment_config: str | Path = "configs/experiments/intraday_hybrid_research.yaml",
    allow_gaps: bool = False,
    include_ablations: bool = True,
) -> AuditArtifacts:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_dir)
    settings = load_variant_settings(config_path, variant)
    frame = read_ohlcv_frame(input_path)
    frame = _filter_frame_by_range(frame, start=None, end=None)

    runner = _build_runner_from_settings(settings)
    baseline_dir = output_path / "baseline"
    baseline_artifacts = runner.run(
        output_dir=baseline_dir,
        input_frame=frame,
        allow_gaps=allow_gaps or settings.data.allow_gaps,
    )

    diagnostics_dir = output_path / "diagnostics"
    diagnostics_runner = BaselineDiagnosticsRunner(runner.application)
    diagnostics_artifacts = diagnostics_runner.run(
        artifact_dir=baseline_dir,
        output_dir=diagnostics_dir,
        include_variants=False,
        include_risk_replay=True,
    )

    report_payload = json.loads(baseline_artifacts.report_path.read_text(encoding="utf-8"))
    diagnostics_payload = json.loads(diagnostics_artifacts.diagnostics_path.read_text(encoding="utf-8"))
    enriched_trades = pd.read_csv(diagnostics_dir / "enriched_trades.csv", parse_dates=["entry_timestamp", "exit_timestamp", "signal_time"])
    enriched_trades["entry_timestamp"] = pd.to_datetime(enriched_trades["entry_timestamp"], utc=True)
    enriched_trades["exit_timestamp"] = pd.to_datetime(enriched_trades["exit_timestamp"], utc=True)
    if "signal_time" in enriched_trades:
        enriched_trades["signal_time"] = pd.to_datetime(enriched_trades["signal_time"], utc=True)

    split_ranges = _split_ranges(frame, experiment_config)
    enriched_trades = _assign_splits_to_trades(enriched_trades, split_ranges)
    split_metrics = _split_metrics(enriched_trades, split_ranges, initial_capital=settings.backtest.initial_capital)

    equity_curve = _equity_curve_from_result(baseline_artifacts.result)
    equity_curve_path = output_path / "equity_curve.csv"
    equity_curve.to_csv(equity_curve_path, index=False)

    trades_path = output_path / "trades_full.csv"
    enriched_trades.to_csv(trades_path, index=False)

    _copy_temporal_artifacts(diagnostics_dir, output_path)

    split_metrics_path = output_path / "split_metrics.csv"
    split_metrics.to_csv(split_metrics_path, index=False)

    audit_risk = _risk_and_drawdown_audit(
        equity_curve=equity_curve,
        trades=enriched_trades,
        initial_capital=settings.backtest.initial_capital,
    )
    risk_path_path = output_path / "risk_path_stats.json"
    risk_path_path.write_text(json.dumps(_sanitize_value(audit_risk), indent=2), encoding="utf-8")

    sensitivity = _cost_slippage_sensitivity(
        diagnostics_runner=diagnostics_runner,
        baseline_dir=baseline_dir,
        settings=settings,
    )
    sensitivity_path = output_path / "cost_slippage_sensitivity.csv"
    sensitivity.to_csv(sensitivity_path, index=False)

    ablation = (
        _run_ablation_suite(
            config_dir=config_path,
            base_settings=settings,
            frame=frame,
            output_dir=output_path / "ablations",
            allow_gaps=allow_gaps or settings.data.allow_gaps,
        )
        if include_ablations
        else pd.DataFrame()
    )
    ablation_path = output_path / "filter_ablation.csv"
    ablation.to_csv(ablation_path, index=False)
    component_ablation_path = output_path / "component_ablation.csv"
    ablation.to_csv(component_ablation_path, index=False)

    strategy_payload = variant_summary_payload(settings)
    strategy_payload["session_close_utc"] = (
        f"{settings.strategy.session_close_hour_utc:02d}:{settings.strategy.session_close_minute_utc:02d}"
    )

    audit_report = {
        "variant": variant,
        "input_path": str(input_path),
        "period": {
            "start": frame.index[0].isoformat() if not frame.empty else None,
            "end": frame.index[-1].isoformat() if not frame.empty else None,
            "bars": int(len(frame)),
            "timezone": "UTC",
        },
        "splits": split_metrics.to_dict(orient="records"),
        "settings": {
            "strategy": strategy_payload,
            "risk": {
                "initial_capital": settings.backtest.initial_capital,
                "risk_per_trade": settings.risk.max_risk_per_trade,
                "max_leverage": settings.risk.max_leverage,
                "max_trades_per_day": settings.risk.max_trades_per_day,
                "block_outside_session": settings.risk.block_outside_session,
                "risk_session_start_utc": f"{settings.risk.session_start_hour_utc:02d}:{settings.risk.session_start_minute_utc:02d}",
                "risk_session_end_utc": f"{settings.risk.session_end_hour_utc:02d}:{settings.risk.session_end_minute_utc:02d}",
            },
            "backtest": {
                "fee_bps": settings.backtest.fee_bps,
                "slippage_bps": settings.backtest.slippage_bps,
                "intrabar_exit_policy": settings.backtest.intrabar_exit_policy,
                "latency_ms": settings.backtest.latency_ms,
            },
        },
        "baseline_report": report_payload,
        "baseline_metrics": diagnostics_payload["baseline_metrics"],
        "risk_and_drawdown_audit": audit_risk,
        "cost_slippage_sensitivity": sensitivity.to_dict(orient="records"),
        "filter_ablation": ablation.to_dict(orient="records"),
        "execution_model": _execution_model_description(settings),
        "leakage_review": _leakage_review(),
        "ppo_readiness": _ppo_readiness(diagnostics_payload["baseline_metrics"], split_metrics, sensitivity, ablation),
        "artifacts": {
            "baseline_dir": str(baseline_dir),
            "diagnostics_dir": str(diagnostics_dir),
            "trades_full_csv": str(trades_path),
            "equity_curve_csv": str(equity_curve_path),
            "split_metrics_csv": str(split_metrics_path),
            "cost_slippage_sensitivity_csv": str(sensitivity_path),
            "filter_ablation_csv": str(ablation_path),
            "component_ablation_csv": str(component_ablation_path),
            "risk_path_stats_json": str(risk_path_path),
        },
    }

    report_path = output_path / "audit_report.json"
    summary_path = output_path / "audit_summary.md"
    report_path.write_text(json.dumps(_sanitize_value(audit_report), indent=2), encoding="utf-8")
    summary_path.write_text(_build_summary(audit_report), encoding="utf-8")
    return AuditArtifacts(
        output_dir=output_path,
        report_path=report_path,
        summary_path=summary_path,
        split_metrics_path=split_metrics_path,
        sensitivity_path=sensitivity_path,
        ablation_path=ablation_path,
        risk_path_path=risk_path_path,
        equity_curve_path=equity_curve_path,
        trades_path=trades_path,
    )


def _copy_temporal_artifacts(diagnostics_dir: Path, output_dir: Path) -> None:
    for filename in [
        "yearly_breakdown.csv",
        "quarterly_breakdown.csv",
        "monthly_breakdown.csv",
        "weekday_breakdown.csv",
        "hourly_breakdown.csv",
        "side_breakdown.csv",
        "exit_reason_breakdown.csv",
        "risk_execution_breakdown.csv",
    ]:
        source = diagnostics_dir / filename
        if source.exists():
            shutil.copyfile(source, output_dir / filename)


def _split_ranges(frame: pd.DataFrame, experiment_config: str | Path) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    index = pd.to_datetime(frame.index, utc=True)
    start = index.min()
    end = index.max()
    train_ratio = 0.60
    validation_ratio = 0.20
    config_path = Path(experiment_config)
    if config_path.exists():
        experiment = load_intraday_hybrid_research_config(config_path)
        train_ratio = experiment.temporal_splits.train_ratio
        validation_ratio = experiment.temporal_splits.validation_ratio
    total_seconds = max((end - start).total_seconds(), 1.0)
    train_end = start + pd.Timedelta(seconds=total_seconds * train_ratio)
    validation_end = train_end + pd.Timedelta(seconds=total_seconds * validation_ratio)
    return [
        ("train", start, train_end),
        ("validation", train_end, validation_end),
        ("test", validation_end, end),
    ]


def _assign_splits_to_trades(
    trades: pd.DataFrame,
    split_ranges: list[tuple[str, pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    if trades.empty or "exit_timestamp" not in trades.columns:
        frame = trades.copy()
        frame["audit_split"] = pd.Series(dtype="object")
        return frame
    frame = trades.copy()
    exits = pd.to_datetime(frame["exit_timestamp"], utc=True, errors="coerce")
    frame["audit_split"] = "unassigned"
    for index, (split_name, start, end) in enumerate(split_ranges):
        is_last = index == len(split_ranges) - 1
        if is_last:
            mask = (exits >= start) & (exits <= end)
        else:
            mask = (exits >= start) & (exits < end)
        frame.loc[mask, "audit_split"] = split_name
    return frame


def _split_metrics(
    trades: pd.DataFrame,
    split_ranges: list[tuple[str, pd.Timestamp, pd.Timestamp]],
    *,
    initial_capital: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    exits = pd.to_datetime(trades["exit_timestamp"], utc=True, errors="coerce") if not trades.empty else pd.Series(dtype="datetime64[ns, UTC]")
    for index, (split_name, start, end) in enumerate(split_ranges):
        is_last = index == len(split_ranges) - 1
        if trades.empty:
            subset = trades.copy()
        elif is_last:
            subset = trades.loc[(exits >= start) & (exits <= end)].copy()
        else:
            subset = trades.loc[(exits >= start) & (exits < end)].copy()
        row = _metrics_from_trade_frame(subset, initial_capital=initial_capital)
        row.update({"split": split_name, "start": start.isoformat(), "end": end.isoformat()})
        rows.append(row)
    columns = ["split", "start", "end"] + [key for key in rows[0] if key not in {"split", "start", "end"}]
    return pd.DataFrame(rows)[columns]


def _metrics_from_trade_frame(trades: pd.DataFrame, *, initial_capital: float) -> dict[str, Any]:
    if trades.empty:
        return {
            "number_of_trades": 0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "fees_paid": 0.0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "payoff": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_consecutive_losses": 0,
            "average_losing_streak": 0.0,
            "max_daily_net_loss": 0.0,
            "max_daily_net_loss_pct": 0.0,
        }
    wins = trades.loc[trades["net_pnl"] > 0.0, "net_pnl"]
    losses = trades.loc[trades["net_pnl"] <= 0.0, "net_pnl"]
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses.sum())) if not losses.empty else 0.0
    max_daily_loss = _max_period_loss(trades, "exit_timestamp", "D")
    streaks = _losing_streaks(trades["net_pnl"].tolist())
    return {
        "number_of_trades": int(len(trades)),
        "gross_pnl": float(trades["gross_pnl"].sum()),
        "net_pnl": float(trades["net_pnl"].sum()),
        "fees_paid": float(trades["fees_paid"].sum()),
        "win_rate": float((trades["net_pnl"] > 0.0).mean()),
        "average_win": float(wins.mean()) if not wins.empty else 0.0,
        "average_loss": float(losses.mean()) if not losses.empty else 0.0,
        "payoff": (float(wins.mean()) / abs(float(losses.mean()))) if not wins.empty and not losses.empty and float(losses.mean()) < 0.0 else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else 0.0,
        "expectancy": float(trades["net_pnl"].mean()),
        "max_consecutive_losses": max(streaks) if streaks else 0,
        "average_losing_streak": float(sum(streaks) / len(streaks)) if streaks else 0.0,
        "max_daily_net_loss": max_daily_loss,
        "max_daily_net_loss_pct": abs(max_daily_loss) / initial_capital if initial_capital > 0.0 else 0.0,
    }


def _equity_curve_from_result(result: Any) -> pd.DataFrame:
    points = result.metadata.get("equity_curve", [])
    frame = pd.DataFrame(points)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "drawdown"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce")
    frame["running_peak"] = frame["equity"].cummax()
    frame["drawdown"] = (frame["equity"] / frame["running_peak"]) - 1.0
    return frame


def _risk_and_drawdown_audit(
    *,
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float,
) -> dict[str, Any]:
    streaks = _losing_streaks(trades["net_pnl"].tolist()) if not trades.empty else []
    daily_loss = _max_period_loss(trades, "exit_timestamp", "D") if not trades.empty else 0.0
    weekly_loss = _max_period_loss(trades, "exit_timestamp", "W-SUN") if not trades.empty else 0.0
    daily_dd = _worst_group_drawdown(equity_curve, "D")
    weekly_dd = _worst_group_drawdown(equity_curve, "W-SUN")
    dd_event = _max_drawdown_event(equity_curve)
    return {
        "max_losing_streak": max(streaks) if streaks else 0,
        "average_losing_streak": float(sum(streaks) / len(streaks)) if streaks else 0.0,
        "losing_streak_count": len(streaks),
        "worst_daily_net_loss": daily_loss,
        "worst_daily_net_loss_pct": abs(daily_loss) / initial_capital if initial_capital > 0.0 else 0.0,
        "worst_weekly_net_loss": weekly_loss,
        "worst_weekly_net_loss_pct": abs(weekly_loss) / initial_capital if initial_capital > 0.0 else 0.0,
        "worst_daily_drawdown": daily_dd,
        "worst_weekly_drawdown": weekly_dd,
        **dd_event,
    }


def _losing_streaks(pnls: Sequence[float]) -> list[int]:
    streaks: list[int] = []
    current = 0
    for pnl in pnls:
        if float(pnl) <= 0.0:
            current += 1
        elif current:
            streaks.append(current)
            current = 0
    if current:
        streaks.append(current)
    return streaks


def _max_period_loss(trades: pd.DataFrame, timestamp_column: str, frequency: str) -> float:
    if trades.empty:
        return 0.0
    frame = trades.copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    grouped = frame.groupby(pd.Grouper(key=timestamp_column, freq=frequency), observed=False)["net_pnl"].sum()
    return float(grouped.min()) if not grouped.empty else 0.0


def _worst_group_drawdown(equity_curve: pd.DataFrame, frequency: str) -> float:
    if equity_curve.empty:
        return 0.0
    frame = equity_curve[["timestamp", "equity"]].copy()
    grouped = frame.groupby(pd.Grouper(key="timestamp", freq=frequency), observed=False)
    worst = 0.0
    for _, group in grouped:
        if group.empty:
            continue
        dd = (group["equity"] / group["equity"].cummax()) - 1.0
        worst = max(worst, abs(float(dd.min())))
    return worst


def _max_drawdown_event(equity_curve: pd.DataFrame) -> dict[str, Any]:
    if equity_curve.empty:
        return {
            "max_drawdown_start": None,
            "max_drawdown_trough": None,
            "max_drawdown_recovery": None,
            "max_drawdown_duration_days": 0.0,
            "recovery_time_days": None,
            "recovered": True,
        }
    frame = equity_curve.copy()
    trough_index = frame["drawdown"].idxmin()
    trough = frame.loc[trough_index]
    prior = frame.loc[:trough_index].copy()
    peak_equity = float(prior["equity"].cummax().iloc[-1])
    peak_rows = prior.loc[prior["equity"] >= peak_equity]
    peak = peak_rows.iloc[-1]
    after = frame.loc[trough_index:].copy()
    recovered_rows = after.loc[after["equity"] >= peak_equity]
    recovery = recovered_rows.iloc[0] if not recovered_rows.empty else None
    peak_time = pd.Timestamp(peak["timestamp"])
    trough_time = pd.Timestamp(trough["timestamp"])
    recovery_time = pd.Timestamp(recovery["timestamp"]) if recovery is not None else None
    return {
        "max_drawdown_start": peak_time.isoformat(),
        "max_drawdown_trough": trough_time.isoformat(),
        "max_drawdown_recovery": recovery_time.isoformat() if recovery_time is not None else None,
        "max_drawdown_duration_days": (trough_time - peak_time).total_seconds() / 86_400.0,
        "recovery_time_days": ((recovery_time - trough_time).total_seconds() / 86_400.0) if recovery_time is not None else None,
        "recovered": recovery_time is not None,
    }


def _cost_slippage_sensitivity(
    *,
    diagnostics_runner: BaselineDiagnosticsRunner,
    baseline_dir: Path,
    settings: Settings,
) -> pd.DataFrame:
    artifact_set = diagnostics_runner._load_artifact_set(baseline_dir)
    base_fee = settings.backtest.fee_bps
    base_slippage = settings.backtest.slippage_bps
    scenarios = [
        ("base", 1.0, 1.0),
        ("fees_x1_5", 1.5, 1.0),
        ("fees_x2", 2.0, 1.0),
        ("fees_x3", 3.0, 1.0),
        ("slippage_x1_5", 1.0, 1.5),
        ("slippage_x2", 1.0, 2.0),
        ("slippage_x3", 1.0, 3.0),
        ("costs_x1_5", 1.5, 1.5),
        ("costs_x2", 2.0, 2.0),
        ("costs_x3", 3.0, 3.0),
    ]
    rows: list[dict[str, Any]] = []
    baseline_net = None
    for name, fee_multiplier, slippage_multiplier in scenarios:
        result = diagnostics_runner._run_backtest_replay(
            bars=artifact_set.bars,
            features=artifact_set.features,
            signals=artifact_set.signals,
            fee_bps=base_fee * fee_multiplier,
            slippage_bps=base_slippage * slippage_multiplier,
        )
        trade_frame = _trades_to_frame(result.trade_records)
        metrics = _metrics_from_trade_frame(trade_frame, initial_capital=settings.backtest.initial_capital)
        if baseline_net is None:
            baseline_net = float(result.pnl_net)
        rows.append(
            {
                "scenario": name,
                "fee_multiplier": fee_multiplier,
                "slippage_multiplier": slippage_multiplier,
                "fee_bps": base_fee * fee_multiplier,
                "slippage_bps": base_slippage * slippage_multiplier,
                "number_of_trades": int(result.trades),
                "net_pnl": float(result.pnl_net),
                "gross_pnl": metrics["gross_pnl"],
                "fees_paid": metrics["fees_paid"],
                "profit_factor": metrics["profit_factor"],
                "expectancy": float(result.expectancy),
                "max_drawdown": float(result.max_drawdown),
                "delta_net_pnl_vs_base": float(result.pnl_net - baseline_net),
            }
        )
    return pd.DataFrame(rows)


def _run_ablation_suite(
    *,
    config_dir: Path,
    base_settings: Settings,
    frame: pd.DataFrame,
    output_dir: Path,
    allow_gaps: bool,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    ablations: list[tuple[str, str, dict[str, Any]]] = [
        ("baseline", "Current baseline with all mandatory filters.", {}),
        (
            "no_htf_filter",
            "Disable EMA200/slope and macro-bias gate.",
            {"strategy": {"use_ema_200_1h_trend_filter": False, "use_ema_200_1h_slope": False, "use_macro_bias_filter": False}},
        ),
        (
            "no_vwap_filter",
            "Disable VWAP structure in feature context and strategy anchor.",
            {"strategy": {"use_intraday_vwap_filter": False, "require_context_vwap_structure": False}},
        ),
        ("no_rvol_filter", "Disable minimum relative-volume gate.", {"strategy": {"minimum_relative_volume": 0.0}}),
        (
            "no_time_filter",
            "Disable strategy entry window and risk outside-session block.",
            {
                "strategy": {"enforce_entry_session": False, "allowed_hours_utc": []},
                "risk": {"block_outside_session": False, "session_start_hour_utc": 0, "session_start_minute_utc": 0, "session_end_hour_utc": 23, "session_end_minute_utc": 55},
            },
        ),
        (
            "no_extension_limiter",
            "Disable anti-chase and pullback-depth extension limits.",
            {"strategy": {"max_breakout_distance_atr": 0.0, "maximum_pullback_depth_atr": 0.0}},
        ),
    ]
    rows: list[dict[str, Any]] = []
    baseline_net: float | None = None
    for name, description, overrides in ablations:
        settings = apply_settings_overrides(base_settings, overrides)
        settings = apply_settings_overrides(settings, {"strategy": {"variant_name": f"audit_{name}"}})
        runner = _build_runner_from_settings(settings)
        artifact_dir = output_dir / name
        artifacts = runner.run(output_dir=artifact_dir, input_frame=frame, allow_gaps=allow_gaps)
        report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
        trades = pd.read_csv(artifacts.trades_path, parse_dates=["entry_timestamp", "exit_timestamp"])
        metrics = _metrics_from_trade_frame(trades, initial_capital=settings.backtest.initial_capital)
        if baseline_net is None:
            baseline_net = float(report["pnl_net"])
        rows.append(
            {
                "variant": name,
                "description": description,
                "number_of_trades": int(report["number_of_trades"]),
                "net_pnl": float(report["pnl_net"]),
                "gross_pnl": metrics["gross_pnl"],
                "fees_paid": metrics["fees_paid"],
                "win_rate": float(report["win_rate"]),
                "average_win": metrics["average_win"],
                "average_loss": metrics["average_loss"],
                "payoff": metrics["payoff"],
                "profit_factor": metrics["profit_factor"],
                "expectancy": float(report["expectancy"]),
                "max_drawdown": float(report["max_drawdown"]),
                "sharpe": float(report["sharpe"]),
                "sortino": float(report["sortino"]),
                "calmar": float(report["calmar"]),
                "delta_net_pnl_vs_baseline": float(report["pnl_net"] - baseline_net),
                "artifact_dir": str(artifact_dir),
            }
        )
    return pd.DataFrame(rows)


def _trades_to_frame(trades: Sequence[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        rows.append(
            {
                "symbol": trade.symbol,
                "side": trade.side.value,
                "entry_timestamp": trade.entry_timestamp,
                "exit_timestamp": trade.exit_timestamp,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "gross_pnl": trade.gross_pnl,
                "net_pnl": trade.net_pnl,
                "fees_paid": trade.fees_paid,
                "return_pct": trade.return_pct,
                "bars_held": trade.bars_held,
                "exit_reason": trade.exit_reason,
                "entry_reason": trade.entry_reason,
            }
        )
    return pd.DataFrame(rows)


def _execution_model_description(settings: Settings) -> dict[str, Any]:
    return {
        "entry_timing": "Signals are generated from the current completed 5m bar and queued for the next bar open.",
        "entry_slippage": "Long entries pay open * (1 + slippage_bps/10000); short entries receive open * (1 - slippage_bps/10000).",
        "exit_slippage": "Long exits receive exit * (1 - slippage_bps/10000); short exits pay exit * (1 + slippage_bps/10000).",
        "intrabar_policy": settings.backtest.intrabar_exit_policy,
        "intrabar_policy_detail": "With 'conservative', if SL and TP are touched in the same candle, the worse outcome for the open position is selected.",
        "gap_model": "No separate gap fill model is implemented. Pending entries fill at the next bar open with slippage; SL/TP are evaluated against each bar high/low.",
        "fees": "Fees are charged on notional at entry and exit.",
        "fee_bps": settings.backtest.fee_bps,
        "slippage_bps": settings.backtest.slippage_bps,
        "initial_capital": settings.backtest.initial_capital,
        "risk_per_trade": settings.risk.max_risk_per_trade,
        "max_leverage": settings.risk.max_leverage,
        "sizing": "Quantity = min((cash * risk_per_trade) / signal_stop_distance, (cash * max_leverage) / slipped_entry_price).",
        "pnl_units": "PnL is absolute account currency units, not percentage points. Percentage metrics divide by equity/capital where explicitly named.",
    }


def _leakage_review() -> dict[str, Any]:
    return {
        "overall_assessment": "No obvious direct look-ahead was found in the baseline execution path, but the timestamp convention must be interpreted carefully.",
        "bar_timestamp_convention": "Rows are indexed by open_time. The strategy uses the row OHLCV as a completed signal bar and enters on the next bar open, so it does not enter inside the same bar whose close/high/low formed the signal.",
        "htf_context": "1H features are resampled with label='right', closed='left' and forward-filled, so the 1H value at a 5m timestamp represents the last completed hourly bucket.",
        "rolling_features": "Rolling breakout ranges use shift(1); relative volume uses shift(1) by session slot; opening-range values are exposed only after the configured range is complete.",
        "normalization": "The current deterministic pipeline does not fit a global scaler in this path; features are raw deterministic values despite normalize metadata.",
        "split": "Train/validation/test here is a temporal reporting split, not a walk-forward re-training split. It is suitable for audit evidence, not sufficient as final anti-overfit validation.",
        "residual_risk": "Because strategy parameters were selected after prior research on the same dataset, the remaining overfitting risk is material until a true walk-forward or out-of-sample validation is run.",
    }


def _ppo_readiness(
    metrics: dict[str, Any],
    split_metrics: pd.DataFrame,
    sensitivity: pd.DataFrame,
    ablation: pd.DataFrame,
) -> dict[str, Any]:
    positive_splits = int((split_metrics["net_pnl"] > 0.0).sum()) if not split_metrics.empty else 0
    base_pf = float(metrics.get("profit_factor", 0.0))
    costs_x2 = sensitivity.loc[sensitivity["scenario"] == "costs_x2"]
    survives_costs_x2 = bool(not costs_x2.empty and float(costs_x2.iloc[0]["net_pnl"]) > 0.0)
    verdict = "NOT_READY_FOR_PPO"
    reasons = [
        "Run a dedicated walk-forward robustness phase before PPO.",
        "Current audit is still based on one historical dataset and a reporting split.",
    ]
    if float(metrics.get("net_pnl", 0.0)) > 0.0 and base_pf > 1.10 and positive_splits >= 2 and survives_costs_x2:
        verdict = "GO_WITH_CAUTION_AFTER_WALK_FORWARD"
        reasons = ["Economics are positive, but PPO should still wait for walk-forward confirmation."]
    elif float(metrics.get("net_pnl", 0.0)) > 0.0:
        verdict = "NEEDS_ROBUSTNESS_BEFORE_PPO"
        reasons = [
            "Positive net PnL is not enough: PF/cost stress/split stability are not strong enough for PPO yet.",
            "A PPO agent trained now may learn dataset-specific artifacts rather than robust trade selection.",
        ]
    return {
        "verdict": verdict,
        "positive_split_count": positive_splits,
        "split_count": int(len(split_metrics)),
        "base_profit_factor": base_pf,
        "survives_costs_x2": survives_costs_x2,
        "ablation_rows": int(len(ablation)),
        "reasons": reasons,
    }


def _build_summary(report: dict[str, Any]) -> str:
    metrics = report["baseline_metrics"]
    risk = report["risk_and_drawdown_audit"]
    ppo = report["ppo_readiness"]
    execution = report["execution_model"]
    strategy = report["settings"]["strategy"]
    risk_settings = report["settings"]["risk"]
    backtest = report["settings"]["backtest"]
    sensitivity_rows = report.get("cost_slippage_sensitivity", [])
    ablation_rows = report.get("filter_ablation", [])
    lines = [
        "# Baseline Intraday Hybrid Audit",
        "",
        "## Direct Answer",
        f"- Period: `{report['period']['start']}` -> `{report['period']['end']}` UTC, bars `{report['period']['bars']}`.",
        f"- Trades: `{metrics['number_of_trades']}` | net `{metrics['net_pnl']:.2f}` | PF `{metrics['profit_factor']:.4f}` | expectancy `{metrics['expectancy']:.2f}` | DD `{metrics['max_drawdown']:.4f}`.",
        f"- PPO readiness: `{ppo['verdict']}`.",
        "- Drawdown fields are fractions of equity: `0.029` means `2.9%`.",
        "",
        "## PnL And Execution Model",
        f"- Initial capital: `{execution['initial_capital']:.2f}`; risk per trade: `{execution['risk_per_trade']:.4f}`; max leverage: `{execution['max_leverage']:.2f}`.",
        f"- Base costs: fee `{backtest['fee_bps']}` bps per side, slippage `{backtest['slippage_bps']}` bps per entry/exit, intrabar policy `{backtest['intrabar_exit_policy']}`.",
        f"- Sizing: {execution['sizing']}",
        f"- PnL units: {execution['pnl_units']}",
        f"- Entry timing: {execution['entry_timing']}",
        f"- Intrabar: {execution['intrabar_policy_detail']}",
        f"- Gaps: {execution['gap_model']}",
        "",
        "## Time Filter",
        f"- Strategy entry window: `{strategy['entry_session_start_utc']}` -> `{strategy['entry_session_end_utc']}` UTC; enforce_entry_session `{strategy['enforce_entry_session']}`.",
        f"- Risk session block: `{risk_settings['risk_session_start_utc']}` -> `{risk_settings['risk_session_end_utc']}` UTC; block_outside_session `{risk_settings['block_outside_session']}`.",
        f"- Open positions at end of execution session: close_on_session_end `{strategy['close_on_session_end']}`, session close `{strategy['session_close_utc']}` UTC.",
        "",
        "## Splits",
    ]
    for row in report["splits"]:
        lines.append(
            f"- `{row['split']}` `{row['start']}` -> `{row['end']}`: trades `{row['number_of_trades']}`, net `{row['net_pnl']:.2f}`, PF `{row['profit_factor']:.4f}`."
        )
    lines.extend(
        [
            "",
            "## Risk Details",
            f"- Max losing streak: `{risk['max_losing_streak']}`; average losing streak: `{risk['average_losing_streak']:.2f}`.",
            f"- Worst daily drawdown: `{risk['worst_daily_drawdown']:.4f}`; worst weekly drawdown: `{risk['worst_weekly_drawdown']:.4f}`.",
            f"- Worst daily net loss: `{risk['worst_daily_net_loss']:.2f}` (`{risk['worst_daily_net_loss_pct']:.4f}` of initial capital).",
            f"- Worst weekly net loss: `{risk['worst_weekly_net_loss']:.2f}` (`{risk['worst_weekly_net_loss_pct']:.4f}` of initial capital).",
            f"- Max DD start/trough/recovery: `{risk['max_drawdown_start']}` -> `{risk['max_drawdown_trough']}` -> `{risk['max_drawdown_recovery']}`.",
            f"- Max DD duration days: `{risk['max_drawdown_duration_days']:.2f}`; recovery time days: `{risk['recovery_time_days']}`.",
            "",
            "## Cost And Slippage Sensitivity",
        ]
    )
    for row in sensitivity_rows:
        lines.append(
            f"- `{row['scenario']}` fee `{row['fee_bps']:.4f}` bps slip `{row['slippage_bps']:.4f}` bps: net `{row['net_pnl']:.2f}`, PF `{row['profit_factor']:.4f}`, DD `{row['max_drawdown']:.4f}`."
        )
    lines.extend(
        [
            "",
            "## Component Ablation",
        ]
    )
    for row in ablation_rows:
        lines.append(
            f"- `{row['variant']}`: trades `{row['number_of_trades']}`, net `{row['net_pnl']:.2f}`, PF `{row['profit_factor']:.4f}`, DD `{row['max_drawdown']:.4f}`, delta net `{row['delta_net_pnl_vs_baseline']:.2f}`."
        )
    lines.extend(
        [
            "",
            "## Leakage Review",
            f"- {report['leakage_review']['overall_assessment']}",
            f"- HTF context: {report['leakage_review']['htf_context']}",
            f"- Residual risk: {report['leakage_review']['residual_risk']}",
            "",
            "## Artifacts",
        ]
    )
    for label, path in report["artifacts"].items():
        lines.append(f"- `{label}`: `{path}`")
    lines.extend(["", "## Honest Conclusion"])
    lines.extend(f"- {reason}" for reason in ppo["reasons"])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the final baseline_intraday_hybrid before PPO.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--variant", default="baseline_intraday_hybrid")
    parser.add_argument("--experiment-config", default="configs/experiments/intraday_hybrid_research.yaml")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-gaps", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    artifacts = run_audit(
        config_dir=args.config_dir,
        input_path=args.input_path,
        output_dir=args.output_dir,
        variant=args.variant,
        experiment_config=args.experiment_config,
        allow_gaps=args.allow_gaps,
        include_ablations=not args.skip_ablations,
    )
    print(f"Audit report: {artifacts.report_path}")
    print(f"Audit summary: {artifacts.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
