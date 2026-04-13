from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import pandas as pd

from hybrid_quant.core import BacktestRequest, BacktestResult, ExecutedTrade, MarketBar, SignalSide, StrategySignal
from hybrid_quant.execution import PortfolioSimulator, resolve_intrabar_policy, signal_has_executable_levels


class BacktestEngine(ABC):
    @abstractmethod
    def run(self, request: BacktestRequest) -> BacktestResult:
        """Execute an offline simulation."""


@dataclass(slots=True)
class IntradayBacktestEngine(BacktestEngine):
    """Intraday baseline backtester.

    Intrabar ambiguity policy:
    when the same bar touches both the stop and the target, execution is resolved by
    `intrabar_exit_policy`:

    - `stop_first`: assume the stop is reached before the target.
    - `target_first`: assume the target is reached before the stop.
    - `conservative`: choose the worse PnL outcome for the open position.
    """

    initial_capital: float
    fee_bps: float
    slippage_bps: float
    latency_ms: int
    intrabar_exit_policy: str = "conservative"
    gap_exit_policy: str = "level"
    point_value: float = 1.0
    contract_step: float = 0.0
    min_contracts: float = 0.0
    max_contracts: float | None = None
    fee_per_contract_per_side: float = 0.0
    slippage_points: float = 0.0

    def run(self, request: BacktestRequest) -> BacktestResult:
        intrabar_policy = resolve_intrabar_policy(request.intrabar_exit_policy, self.intrabar_exit_policy)
        bars = list(request.bars)
        features = list(request.features)
        signals = self._normalize_signals(request, bars)

        if len(features) != len(bars):
            raise ValueError("Bars and features must have the same length for backtesting.")
        if len(signals) != len(bars):
            raise ValueError("Bars and signals must have the same length for backtesting.")

        initial_capital = request.initial_capital or self.initial_capital
        if not bars:
            return BacktestResult(
                start=None,
                end=None,
                trades=0,
                total_return=0.0,
                max_drawdown=0.0,
                pnl_net=0.0,
                equity_final=initial_capital,
                metadata={"mode": "baseline", "bars": 0, "features": 0},
            )

        simulator = PortfolioSimulator(
            initial_capital=initial_capital,
            fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps,
            intrabar_exit_policy=intrabar_policy,
            gap_exit_policy=request.gap_exit_policy or self.gap_exit_policy,
            point_value=self.point_value,
            contract_step=self.contract_step,
            min_contracts=self.min_contracts,
            max_contracts=self.max_contracts,
            fee_per_contract_per_side=self.fee_per_contract_per_side,
            slippage_points=self.slippage_points,
        )
        trades: list[ExecutedTrade] = []
        equity_points: list[tuple[pd.Timestamp, float]] = []
        cooldown_until_index = -1

        for index, bar in enumerate(bars):
            timestamp = pd.Timestamp(bar.timestamp)
            trade = simulator.step(
                index=index,
                bar=bar,
                feature=features[index],
                exit_zscore_threshold=request.exit_zscore_threshold,
                session_close_hour_utc=request.session_close_hour_utc,
                session_close_minute_utc=request.session_close_minute_utc,
                session_close_timezone=request.session_close_timezone,
                session_close_windows=request.session_close_windows,
            )
            if trade is not None:
                trades.append(trade)
                cooldown_until_index = index + request.signal_cooldown_bars

            signal = signals[index]
            can_queue = (
                simulator.position is None
                and simulator.pending_entry is None
                and index < len(bars) - 1
                and index > cooldown_until_index
                and signal_has_executable_levels(signal)
            )
            if can_queue:
                size_fraction = float(signal.metadata.get("risk_size_fraction", request.risk_per_trade_fraction))
                max_leverage = float(signal.metadata.get("risk_max_leverage", request.max_leverage))
                simulator.queue_signal(
                    signal=signal,
                    index=index,
                    size_fraction=size_fraction,
                    max_leverage=max_leverage,
                )

            equity_points.append((timestamp, simulator.equity(bar.close)))

        final_trade = simulator.force_close(bar=bars[-1], index=len(bars) - 1, exit_reason="end_of_data")
        if final_trade is not None:
            trades.append(final_trade)
            equity_points[-1] = (pd.Timestamp(bars[-1].timestamp), simulator.cash)

        equity_series = pd.Series(
            data=[equity for _, equity in equity_points],
            index=pd.DatetimeIndex([timestamp for timestamp, _ in equity_points]),
            dtype=float,
        )
        metrics = self._compute_metrics(trades=trades, equity_series=equity_series, initial_capital=initial_capital)

        return BacktestResult(
            start=bars[0].timestamp,
            end=bars[-1].timestamp,
            trades=len(trades),
            total_return=metrics["total_return"],
            max_drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
            payoff=metrics["payoff"],
            expectancy=metrics["expectancy"],
            sharpe=metrics["sharpe"],
            sortino=metrics["sortino"],
            calmar=metrics["calmar"],
            pnl_net=metrics["pnl_net"],
            equity_final=metrics["equity_final"],
            trade_records=tuple(trades),
            metadata={
                "bars": len(bars),
                "features": len(features),
                "initial_capital": initial_capital,
                "fees_bps": self.fee_bps,
                "slippage_bps": self.slippage_bps,
                "fee_per_contract_per_side": self.fee_per_contract_per_side,
                "slippage_points": self.slippage_points,
                "point_value": self.point_value,
                "contract_step": self.contract_step,
                "min_contracts": self.min_contracts,
                "max_contracts": self.max_contracts,
                "latency_ms": self.latency_ms,
                "mode": "baseline",
                "intrabar_exit_policy": intrabar_policy,
                "gap_exit_policy": request.gap_exit_policy or self.gap_exit_policy,
                "equity_curve": [
                    {"timestamp": timestamp.isoformat(), "equity": equity}
                    for timestamp, equity in equity_points
                ],
                "average_win": metrics["average_win"],
                "average_loss": metrics["average_loss"],
                "exit_reason_counts": metrics["exit_reason_counts"],
            },
        )

    def _normalize_signals(self, request: BacktestRequest, bars: list[MarketBar]) -> list[StrategySignal]:
        if request.signals:
            return list(request.signals)
        if request.signal is not None:
            return [request.signal for _ in bars]
        return [
            StrategySignal(
                symbol=bar_symbol if (bar_symbol := getattr(request, "symbol", None)) else "UNKNOWN",
                timestamp=bar.timestamp,
                side=SignalSide.FLAT,
                strength=0.0,
                rationale="No signal provided for this bar.",
            )
            for bar in bars
        ]

    def _compute_metrics(
        self,
        *,
        trades: list[ExecutedTrade],
        equity_series: pd.Series,
        initial_capital: float,
    ) -> dict[str, float | dict[str, int]]:
        equity_final = float(equity_series.iloc[-1]) if not equity_series.empty else initial_capital
        pnl_net = equity_final - initial_capital
        total_return = pnl_net / initial_capital if initial_capital else 0.0

        drawdown = (equity_series / equity_series.cummax()) - 1.0 if not equity_series.empty else pd.Series(dtype=float)
        max_drawdown = float(abs(drawdown.min())) if not drawdown.empty else 0.0

        net_pnls = [trade.net_pnl for trade in trades]
        wins = [trade.net_pnl for trade in trades if trade.net_pnl > 0.0]
        losses = [trade.net_pnl for trade in trades if trade.net_pnl <= 0.0]
        win_rate = len(wins) / len(trades) if trades else 0.0
        average_win = sum(wins) / len(wins) if wins else 0.0
        average_loss = sum(losses) / len(losses) if losses else 0.0
        payoff = average_win / abs(average_loss) if losses and average_loss != 0.0 else (float("inf") if wins else 0.0)
        expectancy = sum(net_pnls) / len(net_pnls) if net_pnls else 0.0

        returns = equity_series.pct_change().dropna()
        bars_per_year = self._bars_per_year(equity_series.index)
        sharpe = 0.0
        sortino = 0.0
        if not returns.empty and bars_per_year > 0:
            volatility = float(returns.std(ddof=0))
            if volatility > 0.0:
                sharpe = float((returns.mean() / volatility) * math.sqrt(bars_per_year))

            downside = returns[returns < 0.0]
            downside_volatility = float(downside.std(ddof=0)) if not downside.empty else 0.0
            if downside_volatility > 0.0:
                sortino = float((returns.mean() / downside_volatility) * math.sqrt(bars_per_year))

        annualized_return = 0.0
        if len(equity_series) > 1 and bars_per_year > 0 and equity_final > 0.0 and initial_capital > 0.0:
            periods = len(equity_series) - 1
            annualized_return = float((equity_final / initial_capital) ** (bars_per_year / periods) - 1.0)
        calmar = annualized_return / max_drawdown if max_drawdown > 0.0 else 0.0

        exit_reason_counts: dict[str, int] = {}
        for trade in trades:
            exit_reason_counts[trade.exit_reason] = exit_reason_counts.get(trade.exit_reason, 0) + 1

        return {
            "equity_final": equity_final,
            "pnl_net": pnl_net,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "payoff": payoff,
            "expectancy": expectancy,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "average_win": average_win,
            "average_loss": average_loss,
            "exit_reason_counts": exit_reason_counts,
        }

    def _bars_per_year(self, index: pd.DatetimeIndex) -> float:
        if len(index) < 2:
            return 0.0
        deltas = index.to_series().diff().dropna().dt.total_seconds()
        if deltas.empty:
            return 0.0
        step_seconds = float(deltas.median())
        if step_seconds <= 0.0:
            return 0.0
        return (365.0 * 24.0 * 60.0 * 60.0) / step_seconds
