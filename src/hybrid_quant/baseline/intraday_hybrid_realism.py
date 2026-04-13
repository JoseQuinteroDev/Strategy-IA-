from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from hybrid_quant.core import Settings, apply_settings_overrides
from hybrid_quant.data import read_ohlcv_frame
from hybrid_quant.validation.walk_forward import build_rolling_windows

from .intraday_hybrid_audit import (
    _equity_curve_from_result,
    _metrics_from_trade_frame,
    _risk_and_drawdown_audit,
)
from .intraday_hybrid_research import load_intraday_hybrid_research_config
from .orb_intraday_active_research import _build_runner_from_settings, _filter_frame_by_range, _sanitize_value
from .variants import load_variant_settings


@dataclass(slots=True)
class ScenarioConfig:
    name: str
    label: str
    overrides: dict[str, Any]


@dataclass(slots=True)
class WalkForwardConfig:
    enabled: bool = True
    splits: int = 3
    train_ratio: float = 0.50
    validation_ratio: float = 0.20
    test_ratio: float = 0.10


@dataclass(slots=True)
class QualityThresholds:
    minimum_profit_factor: float = 1.10
    minimum_expectancy: float = 0.0
    maximum_drawdown: float = 0.05
    minimum_positive_walk_forward_test_ratio: float = 0.50
    require_costs_x2_positive: bool = True


@dataclass(slots=True)
class VariantComparisonConfig:
    enabled: bool = True
    experiment_config: str = "configs/experiments/intraday_hybrid_research.yaml"
    selected_variants: tuple[str, ...] = ()


@dataclass(slots=True)
class RealismExperimentConfig:
    name: str
    base_variant: str
    instrument_scenarios: tuple[ScenarioConfig, ...]
    timezone_scenarios: tuple[ScenarioConfig, ...]
    cost_scenarios: tuple[ScenarioConfig, ...]
    walk_forward: WalkForwardConfig
    variant_comparison: VariantComparisonConfig
    quality: QualityThresholds


@dataclass(slots=True)
class RealismArtifacts:
    output_dir: Path
    report_path: Path
    summary_path: Path
    dataset_audit_path: Path
    instrument_scenarios_path: Path
    timezone_scenarios_path: Path
    cost_sensitivity_path: Path
    walk_forward_path: Path
    variant_comparison_path: Path


def run_realism_sprint(
    *,
    config_dir: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    experiment_config: str | Path = "configs/experiments/intraday_hybrid_realism.yaml",
    variant: str | None = None,
    allow_gaps: bool = False,
    start: datetime | None = None,
    end: datetime | None = None,
    skip_walk_forward: bool = False,
    skip_variants: bool = False,
) -> RealismArtifacts:
    config_path = Path(config_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    experiment = load_intraday_hybrid_realism_config(experiment_config)
    base_variant = variant or experiment.base_variant
    base_settings = load_variant_settings(config_path, base_variant)
    frame = _filter_frame_by_range(read_ohlcv_frame(input_path), start=start, end=end)

    dataset_audit = _audit_dataset(frame=frame, input_path=input_path)
    dataset_audit_path = output_path / "dataset_audit.json"
    dataset_audit_path.write_text(json.dumps(_sanitize_value(dataset_audit), indent=2), encoding="utf-8")

    realistic_settings = _realistic_base_settings(base_settings)
    baseline_row, baseline_artifacts = _run_settings_scenario(
        name="baseline_realistic",
        label="Frozen baseline_intraday_hybrid with realistic MNQ execution.",
        settings=realistic_settings,
        frame=frame,
        output_dir=output_path / "baseline_realistic",
        allow_gaps=allow_gaps or realistic_settings.data.allow_gaps,
    )
    baseline_equity = _equity_curve_from_result(baseline_artifacts.result)
    baseline_trades = _read_trades(baseline_artifacts.trades_path)
    baseline_equity_path = output_path / "equity_curve_realistic.csv"
    baseline_trades_path = output_path / "trades_realistic.csv"
    baseline_equity.to_csv(baseline_equity_path, index=False)
    baseline_trades.to_csv(baseline_trades_path, index=False)

    risk_path_stats = _risk_and_drawdown_audit(
        equity_curve=baseline_equity,
        trades=baseline_trades,
        initial_capital=realistic_settings.backtest.initial_capital,
    )
    risk_path = output_path / "risk_path_realistic.json"
    risk_path.write_text(json.dumps(_sanitize_value(risk_path_stats), indent=2), encoding="utf-8")

    instrument_rows = _run_scenario_group(
        group_name="instrument_scenarios",
        base_settings=base_settings,
        scenarios=experiment.instrument_scenarios,
        frame=frame,
        output_dir=output_path / "instrument_scenarios",
        allow_gaps=allow_gaps,
    )
    instrument_scenarios_path = output_path / "instrument_scenarios.csv"
    instrument_rows.to_csv(instrument_scenarios_path, index=False)

    timezone_rows = _run_scenario_group(
        group_name="timezone_scenarios",
        base_settings=realistic_settings,
        scenarios=experiment.timezone_scenarios,
        frame=frame,
        output_dir=output_path / "timezone_scenarios",
        allow_gaps=allow_gaps,
    )
    timezone_scenarios_path = output_path / "timezone_scenarios.csv"
    timezone_rows.to_csv(timezone_scenarios_path, index=False)

    cost_rows = _run_scenario_group(
        group_name="execution_cost_sensitivity",
        base_settings=realistic_settings,
        scenarios=experiment.cost_scenarios,
        frame=frame,
        output_dir=output_path / "cost_sensitivity",
        allow_gaps=allow_gaps,
    )
    cost_sensitivity_path = output_path / "execution_cost_sensitivity.csv"
    cost_rows.to_csv(cost_sensitivity_path, index=False)

    walk_forward_rows = (
        _run_walk_forward(
            settings=realistic_settings,
            frame=frame,
            output_dir=output_path / "walk_forward",
            config=experiment.walk_forward,
            allow_gaps=allow_gaps,
        )
        if experiment.walk_forward.enabled and not skip_walk_forward
        else pd.DataFrame()
    )
    walk_forward_path = output_path / "walk_forward_results.csv"
    walk_forward_rows.to_csv(walk_forward_path, index=False)

    variant_rows = (
        _run_variant_comparison(
            config_dir=config_path,
            base_settings=realistic_settings,
            frame=frame,
            output_dir=output_path / "variant_comparison_realistic",
            config=experiment.variant_comparison,
            allow_gaps=allow_gaps,
        )
        if experiment.variant_comparison.enabled and not skip_variants
        else pd.DataFrame()
    )
    variant_comparison_path = output_path / "variant_comparison_realistic.csv"
    variant_rows.to_csv(variant_comparison_path, index=False)

    conclusion = _build_conclusion(
        baseline=baseline_row,
        cost_rows=cost_rows,
        walk_forward_rows=walk_forward_rows,
        quality=experiment.quality,
    )
    report = _build_report(
        experiment=experiment,
        base_variant=base_variant,
        input_path=input_path,
        dataset_audit=dataset_audit,
        baseline_row=baseline_row,
        instrument_rows=instrument_rows,
        timezone_rows=timezone_rows,
        cost_rows=cost_rows,
        walk_forward_rows=walk_forward_rows,
        variant_rows=variant_rows,
        risk_path_stats=risk_path_stats,
        realistic_settings=realistic_settings,
        conclusion=conclusion,
        artifact_paths={
            "dataset_audit_json": dataset_audit_path,
            "baseline_dir": baseline_artifacts.output_dir,
            "trades_realistic_csv": baseline_trades_path,
            "equity_curve_realistic_csv": baseline_equity_path,
            "risk_path_realistic_json": risk_path,
            "instrument_scenarios_csv": instrument_scenarios_path,
            "timezone_scenarios_csv": timezone_scenarios_path,
            "execution_cost_sensitivity_csv": cost_sensitivity_path,
            "walk_forward_results_csv": walk_forward_path,
            "variant_comparison_realistic_csv": variant_comparison_path,
        },
    )

    report_path = output_path / "realism_report.json"
    summary_path = output_path / "realism_summary.md"
    report_path.write_text(json.dumps(_sanitize_value(report), indent=2), encoding="utf-8")
    summary_path.write_text(_build_summary(report), encoding="utf-8")
    return RealismArtifacts(
        output_dir=output_path,
        report_path=report_path,
        summary_path=summary_path,
        dataset_audit_path=dataset_audit_path,
        instrument_scenarios_path=instrument_scenarios_path,
        timezone_scenarios_path=timezone_scenarios_path,
        cost_sensitivity_path=cost_sensitivity_path,
        walk_forward_path=walk_forward_path,
        variant_comparison_path=variant_comparison_path,
    )


def load_intraday_hybrid_realism_config(path: str | Path) -> RealismExperimentConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Intraday hybrid realism config file not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Intraday hybrid realism config must be a mapping: {config_path}")

    def scenario_list(key: str) -> tuple[ScenarioConfig, ...]:
        raw_items = payload.get(key, []) or []
        if not isinstance(raw_items, list):
            raise ValueError(f"'{key}' must be a list of scenario mappings.")
        scenarios: list[ScenarioConfig] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise ValueError(f"Each item in '{key}' must be a mapping.")
            overrides = raw.get("overrides", {}) or {}
            if not isinstance(overrides, dict):
                raise ValueError(f"Scenario '{raw.get('name')}' overrides must be a mapping.")
            scenarios.append(
                ScenarioConfig(
                    name=str(raw["name"]),
                    label=str(raw.get("label", raw["name"])),
                    overrides=overrides,
                )
            )
        return tuple(scenarios)

    walk_payload = payload.get("walk_forward", {}) or {}
    variant_payload = payload.get("variant_comparison", {}) or {}
    quality_payload = payload.get("quality", {}) or {}
    return RealismExperimentConfig(
        name=str(payload.get("name", "intraday_hybrid_realism")),
        base_variant=str(payload.get("base_variant", "baseline_intraday_hybrid")),
        instrument_scenarios=scenario_list("instrument_scenarios"),
        timezone_scenarios=scenario_list("timezone_scenarios"),
        cost_scenarios=scenario_list("cost_scenarios"),
        walk_forward=WalkForwardConfig(
            enabled=bool(walk_payload.get("enabled", True)),
            splits=int(walk_payload.get("splits", 3)),
            train_ratio=float(walk_payload.get("train_ratio", 0.50)),
            validation_ratio=float(walk_payload.get("validation_ratio", 0.20)),
            test_ratio=float(walk_payload.get("test_ratio", 0.10)),
        ),
        variant_comparison=VariantComparisonConfig(
            enabled=bool(variant_payload.get("enabled", True)),
            experiment_config=str(
                variant_payload.get("experiment_config", "configs/experiments/intraday_hybrid_research.yaml")
            ),
            selected_variants=tuple(str(item) for item in variant_payload.get("selected_variants", []) or []),
        ),
        quality=QualityThresholds(
            minimum_profit_factor=float(quality_payload.get("minimum_profit_factor", 1.10)),
            minimum_expectancy=float(quality_payload.get("minimum_expectancy", 0.0)),
            maximum_drawdown=float(quality_payload.get("maximum_drawdown", 0.05)),
            minimum_positive_walk_forward_test_ratio=float(
                quality_payload.get("minimum_positive_walk_forward_test_ratio", 0.50)
            ),
            require_costs_x2_positive=bool(quality_payload.get("require_costs_x2_positive", True)),
        ),
    )


def _realistic_base_settings(settings: Settings) -> Settings:
    return apply_settings_overrides(
        settings,
        {
            "market": {"symbol": "MNQ", "venue": "cme"},
            "strategy": {
                "enforce_entry_session": True,
                "entry_session_start_hour_utc": 14,
                "entry_session_start_minute_utc": 0,
                "entry_session_end_hour_utc": 19,
                "entry_session_end_minute_utc": 0,
                "entry_session_timezone": "UTC",
                "allowed_hours_utc": [14, 15, 16, 17, 18],
                "close_on_session_end": True,
                "session_close_hour_utc": 20,
                "session_close_minute_utc": 55,
                "session_close_timezone": "UTC",
            },
            "risk": {
                "block_outside_session": True,
                "session_start_hour_utc": 14,
                "session_start_minute_utc": 0,
                "session_end_hour_utc": 19,
                "session_end_minute_utc": 0,
                "session_timezone": "UTC",
            },
            "backtest": {
                "fee_bps": 0.0,
                "slippage_bps": 0.0,
                "fee_per_contract_per_side": 1.0,
                "slippage_points": 0.25,
                "point_value": 2.0,
                "contract_step": 1.0,
                "min_contracts": 1.0,
                "max_contracts": 20.0,
                "intrabar_exit_policy": "conservative",
                "gap_exit_policy": "open",
            },
        },
    )


def _run_scenario_group(
    *,
    group_name: str,
    base_settings: Settings,
    scenarios: Sequence[ScenarioConfig],
    frame: pd.DataFrame,
    output_dir: Path,
    allow_gaps: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        settings = apply_settings_overrides(base_settings, scenario.overrides)
        settings = apply_settings_overrides(settings, {"strategy": {"variant_name": scenario.name}})
        row, _ = _run_settings_scenario(
            name=scenario.name,
            label=scenario.label,
            settings=settings,
            frame=frame,
            output_dir=output_dir / scenario.name,
            allow_gaps=allow_gaps or settings.data.allow_gaps,
        )
        row["group"] = group_name
        rows.append(row)
    return pd.DataFrame(rows)


def _run_settings_scenario(
    *,
    name: str,
    label: str,
    settings: Settings,
    frame: pd.DataFrame,
    output_dir: Path,
    allow_gaps: bool,
) -> tuple[dict[str, Any], Any]:
    runner = _build_runner_from_settings(settings)
    artifacts = runner.run(output_dir=output_dir, input_frame=frame, allow_gaps=allow_gaps)
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    trades = _read_trades(artifacts.trades_path)
    metrics = _metrics_from_trade_frame(trades, initial_capital=settings.backtest.initial_capital)
    row = {
        "scenario": name,
        "label": label,
        "symbol": settings.market.symbol,
        "number_of_trades": int(report["number_of_trades"]),
        "gross_pnl": metrics["gross_pnl"],
        "net_pnl": float(report["pnl_net"]),
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
        "max_consecutive_losses": metrics["max_consecutive_losses"],
        **_period_activity_metrics(
            trades=trades,
            start=pd.Timestamp(report["start"]) if report.get("start") else frame.index.min(),
            end=pd.Timestamp(report["end"]) if report.get("end") else frame.index.max(),
        ),
        "point_value": settings.backtest.point_value,
        "contract_step": settings.backtest.contract_step,
        "min_contracts": settings.backtest.min_contracts,
        "max_contracts": settings.backtest.max_contracts,
        "fee_bps": settings.backtest.fee_bps,
        "slippage_bps": settings.backtest.slippage_bps,
        "fee_per_contract_per_side": settings.backtest.fee_per_contract_per_side,
        "slippage_points": settings.backtest.slippage_points,
        "intrabar_exit_policy": settings.backtest.intrabar_exit_policy,
        "gap_exit_policy": settings.backtest.gap_exit_policy,
        "entry_session_timezone": settings.strategy.entry_session_timezone,
        "entry_session_start": _time_label(
            settings.strategy.entry_session_start_hour_utc,
            settings.strategy.entry_session_start_minute_utc,
        ),
        "entry_session_end": _time_label(
            settings.strategy.entry_session_end_hour_utc,
            settings.strategy.entry_session_end_minute_utc,
        ),
        "risk_session_timezone": settings.risk.session_timezone,
        "session_close_timezone": settings.strategy.session_close_timezone,
        "session_close": _time_label(settings.strategy.session_close_hour_utc, settings.strategy.session_close_minute_utc),
        "allowed_hours_utc": ",".join(str(hour) for hour in settings.strategy.allowed_hours_utc),
        "artifact_dir": str(output_dir),
    }
    return row, artifacts


def _run_walk_forward(
    *,
    settings: Settings,
    frame: pd.DataFrame,
    output_dir: Path,
    config: WalkForwardConfig,
    allow_gaps: bool,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = build_rolling_windows(
        total_bars=len(frame),
        splits=config.splits,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
    )
    rows: list[dict[str, Any]] = []
    for window in windows:
        phases = [
            ("train", window.train_start, window.train_end),
            ("validation", window.validation_start, window.validation_end),
            ("test", window.test_start, window.test_end),
        ]
        for phase, start_index, end_index in phases:
            subset = frame.iloc[start_index:end_index].copy()
            if subset.empty:
                continue
            phase_settings = apply_settings_overrides(
                settings,
                {"strategy": {"variant_name": f"walk_forward_{window.window_id}_{phase}"}},
            )
            row, _ = _run_settings_scenario(
                name=f"wf_{window.window_id}_{phase}",
                label=f"Walk-forward window {window.window_id} {phase}.",
                settings=phase_settings,
                frame=subset,
                output_dir=output_dir / f"window_{window.window_id}" / phase,
                allow_gaps=allow_gaps or phase_settings.data.allow_gaps,
            )
            row.update(
                {
                    "window_id": window.window_id,
                    "phase": phase,
                    "start": subset.index.min().isoformat(),
                    "end": subset.index.max().isoformat(),
                    "bars": int(len(subset)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _run_variant_comparison(
    *,
    config_dir: Path,
    base_settings: Settings,
    frame: pd.DataFrame,
    output_dir: Path,
    config: VariantComparisonConfig,
    allow_gaps: bool,
) -> pd.DataFrame:
    experiment = load_intraday_hybrid_research_config(config.experiment_config)
    selected = set(config.selected_variants)
    rows: list[dict[str, Any]] = []
    for variant in experiment.variants:
        if selected and variant.name not in selected:
            continue
        source_variant = variant.source_variant or experiment.base_variant
        source_settings = (
            base_settings
            if source_variant == experiment.base_variant
            else load_variant_settings(config_dir, source_variant)
        )
        settings = apply_settings_overrides(source_settings, variant.overrides)
        settings = _realistic_base_settings(settings)
        settings = apply_settings_overrides(settings, {"strategy": {"variant_name": variant.name}})
        row, _ = _run_settings_scenario(
            name=variant.name,
            label=variant.label,
            settings=settings,
            frame=frame,
            output_dir=output_dir / variant.name,
            allow_gaps=allow_gaps or settings.data.allow_gaps,
        )
        row.update({"candidate": variant.candidate, "source_variant": source_variant})
        rows.append(row)
    return pd.DataFrame(rows)


def _audit_dataset(*, frame: pd.DataFrame, input_path: str | Path) -> dict[str, Any]:
    index = pd.to_datetime(frame.index, utc=True)
    diffs = index.to_series().diff().dropna()
    dominant_interval = diffs.mode().iloc[0] if not diffs.empty else pd.NaT
    expected = pd.Timedelta(minutes=5)
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame else pd.Series(dtype=float)
    normalized_schema = set(frame.columns) == {"open", "high", "low", "close", "volume"}
    looks_like_real_scale = bool(close.median() > 1000.0 and close.max() > 5000.0)
    return {
        "input_path": str(input_path),
        "columns_after_load": list(frame.columns),
        "schema_normalized_to_internal_ohlcv": normalized_schema,
        "start": index.min().isoformat() if len(index) else None,
        "end": index.max().isoformat() if len(index) else None,
        "bars": int(len(frame)),
        "monotonic_increasing": bool(index.is_monotonic_increasing),
        "duplicate_timestamps": int(index.duplicated().sum()),
        "dominant_interval": str(dominant_interval),
        "dominant_interval_is_5m": bool(dominant_interval == expected),
        "gap_count_gt_5m": int((diffs > expected).sum()) if not diffs.empty else 0,
        "largest_gap": str(diffs.max()) if not diffs.empty else None,
        "price_min": float(close.min()),
        "price_median": float(close.median()),
        "price_max": float(close.max()),
        "volume_min": float(volume.min()) if not volume.empty else None,
        "volume_median": float(volume.median()) if not volume.empty else None,
        "volume_max": float(volume.max()) if not volume.empty else None,
        "statistical_price_normalization_detected": not looks_like_real_scale,
        "price_scale_assessment": (
            "Prices are in NQ/MNQ-like index-point scale; no z-score/min-max price normalization is evident."
            if looks_like_real_scale
            else "Prices do not look like NQ/MNQ index-point scale; verify export provenance before trusting dollar PnL."
        ),
        "raw_back_adjusted_or_stitched_status": (
            "Unknown from CSV contents. The file name and OHLCV schema do not prove whether futures rolls were "
            "back-adjusted, stitched, broker-continuous, or raw front-month data."
        ),
        "economic_pnl_validity": (
            "Legacy unit PnL is not economically sufficient for futures. Dollar PnL is meaningful only after applying "
            "point_value, contract sizing, fixed per-contract fees, and slippage in points."
        ),
    }


def _read_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "entry_timestamp" in frame:
        frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True, errors="coerce")
    if "exit_timestamp" in frame:
        frame["exit_timestamp"] = pd.to_datetime(frame["exit_timestamp"], utc=True, errors="coerce")
    return frame


def _period_activity_metrics(*, trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    start_ts = _utc_timestamp(start)
    end_ts = _utc_timestamp(end)
    days = max((end_ts - start_ts).total_seconds() / 86_400.0, 1.0)
    years = days / 365.25
    weeks = days / 7.0
    months = days / 30.4375
    count = int(len(trades))
    if trades.empty or "entry_timestamp" not in trades:
        pct_weeks = 0.0
        pct_days = 0.0
    else:
        entries = pd.to_datetime(trades["entry_timestamp"], utc=True, errors="coerce")
        traded_days = entries.dt.floor("D").nunique()
        traded_weeks = entries.dt.tz_convert(None).dt.to_period("W-SUN").nunique()
        pct_days = traded_days / max(int(days), 1)
        pct_weeks = traded_weeks / max(int(weeks), 1)
    return {
        "trades_per_year": count / years if years > 0.0 else 0.0,
        "trades_per_month": count / months if months > 0.0 else 0.0,
        "trades_per_week_avg": count / weeks if weeks > 0.0 else 0.0,
        "percentage_of_days_with_trade": float(pct_days),
        "percentage_of_weeks_with_trade": float(pct_weeks),
    }


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _build_report(
    *,
    experiment: RealismExperimentConfig,
    base_variant: str,
    input_path: str | Path,
    dataset_audit: dict[str, Any],
    baseline_row: dict[str, Any],
    instrument_rows: pd.DataFrame,
    timezone_rows: pd.DataFrame,
    cost_rows: pd.DataFrame,
    walk_forward_rows: pd.DataFrame,
    variant_rows: pd.DataFrame,
    risk_path_stats: dict[str, Any],
    realistic_settings: Settings,
    conclusion: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "experiment_name": experiment.name,
        "base_variant": base_variant,
        "input_path": str(input_path),
        "dataset_audit": dataset_audit,
        "realistic_baseline": baseline_row,
        "instrument_scenarios": instrument_rows.to_dict(orient="records"),
        "timezone_scenarios": timezone_rows.to_dict(orient="records"),
        "execution_cost_sensitivity": cost_rows.to_dict(orient="records"),
        "walk_forward": walk_forward_rows.to_dict(orient="records"),
        "variant_comparison_realistic": variant_rows.to_dict(orient="records"),
        "risk_path_realistic": risk_path_stats,
        "execution_model": _execution_model_description(realistic_settings),
        "instrument_model": _instrument_model_description(realistic_settings),
        "timezone_model": _timezone_model_description(realistic_settings),
        "conclusion": conclusion,
        "artifacts": {key: str(value) for key, value in artifact_paths.items()},
    }


def _build_conclusion(
    *,
    baseline: dict[str, Any],
    cost_rows: pd.DataFrame,
    walk_forward_rows: pd.DataFrame,
    quality: QualityThresholds,
) -> dict[str, Any]:
    baseline_net = float(baseline.get("net_pnl", 0.0))
    baseline_pf = float(baseline.get("profit_factor", 0.0))
    baseline_expectancy = float(baseline.get("expectancy", 0.0))
    baseline_dd = float(baseline.get("max_drawdown", 1.0))
    costs_x2 = (
        cost_rows.loc[cost_rows["scenario"].astype(str).str.contains("costs_x2", na=False)]
        if not cost_rows.empty and "scenario" in cost_rows
        else pd.DataFrame()
    )
    survives_costs_x2 = bool(not costs_x2.empty and float(costs_x2.iloc[0]["net_pnl"]) > 0.0)
    wf_tests = (
        walk_forward_rows.loc[walk_forward_rows["phase"] == "test"]
        if not walk_forward_rows.empty and "phase" in walk_forward_rows
        else pd.DataFrame()
    )
    positive_wf_test_ratio = float((wf_tests["net_pnl"] > 0.0).mean()) if not wf_tests.empty else 0.0

    if baseline_net <= 0.0 or baseline_expectancy <= quality.minimum_expectancy:
        classification = "EDGE_FAILS_NOT_READY_FOR_PPO"
        reasons = [
            "The realistic baseline is not profitable after instrument-aware execution.",
            "Do not train PPO on this baseline yet; it would learn from weak or negative trade proposals.",
        ]
    elif (
        baseline_pf >= quality.minimum_profit_factor
        and baseline_dd <= quality.maximum_drawdown
        and positive_wf_test_ratio >= quality.minimum_positive_walk_forward_test_ratio
        and (survives_costs_x2 or not quality.require_costs_x2_positive)
    ):
        classification = "EDGE_SURVIVES_BUT_STILL_WALK_FORWARD_BEFORE_PPO"
        reasons = [
            "The realistic baseline remains economically positive under the configured quality gates.",
            "PPO can be considered only after reviewing window-level stability and cost stress in detail.",
        ]
    else:
        classification = "EDGE_FRAGILE_NOT_READY_FOR_PPO"
        reasons = [
            "The baseline remains alive but does not pass enough robustness or cost-stress checks.",
            "The next step should be additional robustness or strategy cleanup, not PPO.",
        ]
    return {
        "classification": classification,
        "baseline_net_pnl": baseline_net,
        "baseline_profit_factor": baseline_pf,
        "baseline_expectancy": baseline_expectancy,
        "baseline_max_drawdown": baseline_dd,
        "survives_costs_x2": survives_costs_x2,
        "positive_walk_forward_test_ratio": positive_wf_test_ratio,
        "reasons": reasons,
    }


def _instrument_model_description(settings: Settings) -> dict[str, Any]:
    return {
        "symbol": settings.market.symbol,
        "point_value": settings.backtest.point_value,
        "contract_step": settings.backtest.contract_step,
        "min_contracts": settings.backtest.min_contracts,
        "max_contracts": settings.backtest.max_contracts,
        "risk_per_trade_fraction": settings.risk.max_risk_per_trade,
        "initial_capital": settings.backtest.initial_capital,
        "sizing_formula": (
            "contracts = floor(min(equity*risk_per_trade / (stop_points*point_value), "
            "equity*max_leverage / (entry_price*point_value)) / contract_step) * contract_step"
        ),
        "mnq_reference": "MNQ is modeled with point_value=2 USD/point.",
        "nq_reference": "NQ is modeled in scenarios with point_value=20 USD/point.",
    }


def _execution_model_description(settings: Settings) -> dict[str, Any]:
    return {
        "entry_timing": "A signal uses a completed bar and queues entry for the next bar open.",
        "slippage_model": (
            "Entry and exit apply directional bps slippage plus fixed slippage_points. "
            "For MNQ base, slippage_points=0.25 means one tick per side."
        ),
        "fee_model": "Costs combine notional fee_bps and fixed fee_per_contract_per_side on entry and exit.",
        "intrabar_exit_policy": settings.backtest.intrabar_exit_policy,
        "intrabar_exit_policy_detail": (
            "Conservative mode chooses the worse outcome when stop and target are both touched inside one candle."
        ),
        "gap_exit_policy": settings.backtest.gap_exit_policy,
        "gap_exit_policy_detail": (
            "With gap_exit_policy=open, an existing position that opens beyond stop/target exits at that bar open."
        ),
        "pnl_units": "PnL is reported in account dollars after point_value and contract quantity.",
    }


def _timezone_model_description(settings: Settings) -> dict[str, Any]:
    return {
        "entry_session_timezone": settings.strategy.entry_session_timezone,
        "risk_session_timezone": settings.risk.session_timezone,
        "session_close_timezone": settings.strategy.session_close_timezone,
        "dst_support": "Implemented through zoneinfo/pandas timezone conversion for named zones.",
        "entry_window": (
            f"{_time_label(settings.strategy.entry_session_start_hour_utc, settings.strategy.entry_session_start_minute_utc)} "
            f"to {_time_label(settings.strategy.entry_session_end_hour_utc, settings.strategy.entry_session_end_minute_utc)} "
            f"in {settings.strategy.entry_session_timezone}."
        ),
        "outside_session_behavior": "Strategy and RiskEngine both block candidate entries outside the configured window.",
        "open_position_behavior": (
            "Open positions remain until SL/TP/time-stop unless close_on_session_end=true, which forces closure at "
            "the configured session close."
        ),
    }


def _build_summary(report: dict[str, Any]) -> str:
    baseline = report["realistic_baseline"]
    conclusion = report["conclusion"]
    dataset = report["dataset_audit"]
    lines = [
        "# Intraday Hybrid Realism And Robustness Sprint",
        "",
        "## Direct Answer",
        f"- Classification: `{conclusion['classification']}`.",
        f"- Realistic baseline: trades `{baseline['number_of_trades']}`, net `${baseline['net_pnl']:.2f}`, PF `{baseline['profit_factor']:.4f}`, expectancy `${baseline['expectancy']:.2f}`, DD `{baseline['max_drawdown']:.4f}`.",
        "- Drawdown is a fraction of equity: `0.024` means `2.4%`.",
        "",
        "## Dataset Audit",
        f"- File: `{report['input_path']}`.",
        f"- Period: `{dataset['start']}` -> `{dataset['end']}`, bars `{dataset['bars']}`.",
        f"- Dominant interval: `{dataset['dominant_interval']}`; 5m compatible `{dataset['dominant_interval_is_5m']}`; gaps > 5m `{dataset['gap_count_gt_5m']}`.",
        f"- Price scale: {dataset['price_scale_assessment']}",
        f"- Futures provenance: {dataset['raw_back_adjusted_or_stitched_status']}",
        f"- PnL validity: {dataset['economic_pnl_validity']}",
        "",
        "## Instrument And Execution",
        f"- Symbol `{report['instrument_model']['symbol']}`, point value `{report['instrument_model']['point_value']}`, contract step `{report['instrument_model']['contract_step']}`.",
        f"- Base fees: `${baseline['fee_per_contract_per_side']:.2f}` per contract per side; slippage `{baseline['slippage_points']}` points per side.",
        f"- Intrabar policy `{baseline['intrabar_exit_policy']}`; gap policy `{baseline['gap_exit_policy']}`.",
        f"- Time filter: `{baseline['entry_session_start']}` -> `{baseline['entry_session_end']}` in `{baseline['entry_session_timezone']}`; RiskEngine timezone `{baseline['risk_session_timezone']}`.",
        "",
        "## Cost Stress",
    ]
    for row in report["execution_cost_sensitivity"]:
        lines.append(
            f"- `{row['scenario']}`: trades `{row['number_of_trades']}`, net `${row['net_pnl']:.2f}`, PF `{row['profit_factor']:.4f}`, DD `{row['max_drawdown']:.4f}`."
        )
    lines.extend(["", "## Timezone Scenarios"])
    for row in report["timezone_scenarios"]:
        lines.append(
            f"- `{row['scenario']}` {row['entry_session_timezone']} `{row['entry_session_start']}`-`{row['entry_session_end']}`: trades `{row['number_of_trades']}`, net `${row['net_pnl']:.2f}`, PF `{row['profit_factor']:.4f}`."
        )
    lines.extend(["", "## Walk-Forward"])
    if report["walk_forward"]:
        for row in report["walk_forward"]:
            if row.get("phase") == "test":
                lines.append(
                    f"- Window `{row['window_id']}` test `{row['start']}` -> `{row['end']}`: trades `{row['number_of_trades']}`, net `${row['net_pnl']:.2f}`, PF `{row['profit_factor']:.4f}`."
                )
    else:
        lines.append("- Walk-forward was skipped or disabled.")
    lines.extend(["", "## Variant Comparison Under Realism"])
    if report["variant_comparison_realistic"]:
        ordered = sorted(
            report["variant_comparison_realistic"],
            key=lambda item: (float(item.get("net_pnl", 0.0)), float(item.get("profit_factor", 0.0))),
            reverse=True,
        )
        for row in ordered[:8]:
            lines.append(
                f"- `{row['scenario']}`: net `${row['net_pnl']:.2f}`, PF `{row['profit_factor']:.4f}`, expectancy `${row['expectancy']:.2f}`, DD `{row['max_drawdown']:.4f}`, trades `{row['number_of_trades']}`."
            )
    else:
        lines.append("- Variant comparison was skipped or disabled.")
    lines.extend(["", "## Honest Conclusion"])
    lines.extend(f"- {reason}" for reason in conclusion["reasons"])
    lines.extend(["", "## Artifacts"])
    for label, path in report["artifacts"].items():
        lines.append(f"- `{label}`: `{path}`")
    return "\n".join(lines) + "\n"


def _time_label(hour: int, minute: int) -> str:
    return f"{int(hour):02d}:{int(minute):02d}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run realism and robustness checks for baseline_intraday_hybrid.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--experiment-config", default="configs/experiments/intraday_hybrid_realism.yaml")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-gaps", action="store_true")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--skip-walk-forward", action="store_true")
    parser.add_argument("--skip-variants", action="store_true")
    return parser


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return pd.Timestamp(value).to_pydatetime()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    artifacts = run_realism_sprint(
        config_dir=args.config_dir,
        input_path=args.input_path,
        output_dir=args.output_dir,
        experiment_config=args.experiment_config,
        variant=args.variant,
        allow_gaps=args.allow_gaps,
        start=_parse_datetime(args.start),
        end=_parse_datetime(args.end),
        skip_walk_forward=args.skip_walk_forward,
        skip_variants=args.skip_variants,
    )
    payload = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    print(f"Realism report: {artifacts.report_path}")
    print(f"Realism summary: {artifacts.summary_path}")
    print(f"classification={payload['conclusion']['classification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
