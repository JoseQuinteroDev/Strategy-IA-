from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import time
import math

import pandas as pd

from hybrid_quant.core import (
    BacktestRequest,
    BacktestResult,
    ExecutedTrade,
    FeatureSnapshot,
    MarketBar,
    SignalSide,
    StrategySignal,
)


class BacktestEngine(ABC):
    @abstractmethod
    def run(self, request: BacktestRequest) -> BacktestResult:
        """Execute an offline simulation."""


@dataclass(slots=True)
class _PendingEntry:
    signal: StrategySignal
    generated_index: int
    stop_distance: float
    target_distance: float
    size_fraction: float
    max_leverage: float


@dataclass(slots=True)
class _OpenPosition:
    symbol: str
    side: SignalSide
    entry_timestamp: pd.Timestamp
    entry_index: int
    entry_price: float
    stop_price: float
    target_price: float
    quantity: float
    entry_fee: float
    time_stop_bars: int | None
    close_on_session_end: bool
    entry_reason: str | None
    signal_metadata: dict[str, object]


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

    def run(self, request: BacktestRequest) -> BacktestResult:
        intrabar_policy = self._resolve_intrabar_policy(request.intrabar_exit_policy)
        bars = list(request.bars)
        features = list(request.features)
        signals = self._normalize_signals(request, bars)

        if len(features) != len(bars):
            raise ValueError("Bars and features must have the same length for backtesting.")
        if len(signals) != len(bars):
            raise ValueError("Bars and signals must have the same length for backtesting.")

        if not bars:
            return BacktestResult(
                start=None,
                end=None,
                trades=0,
                total_return=0.0,
                max_drawdown=0.0,
                pnl_net=0.0,
                equity_final=request.initial_capital or self.initial_capital,
                metadata={"mode": "baseline", "bars": 0, "features": 0},
            )

        cash = request.initial_capital or self.initial_capital
        fee_rate = self.fee_bps / 10000.0
        trades: list[ExecutedTrade] = []
        equity_points: list[tuple[pd.Timestamp, float]] = []
        pending_entry: _PendingEntry | None = None
        position: _OpenPosition | None = None
        cooldown_until_index = -1

        for index, bar in enumerate(bars):
            timestamp = pd.Timestamp(bar.timestamp)

            if pending_entry is not None and index > pending_entry.generated_index and position is None:
                position, cash = self._open_position(
                    pending=pending_entry,
                    entry_bar=bar,
                    index=index,
                    cash=cash,
                    request=request,
                    fee_rate=fee_rate,
                )
                pending_entry = None

            if position is not None:
                trade, cash, position = self._evaluate_open_position(
                    position=position,
                    bar=bar,
                    feature=features[index],
                    index=index,
                    cash=cash,
                    fee_rate=fee_rate,
                    request=request,
                    intrabar_policy=intrabar_policy,
                )
                if trade is not None:
                    trades.append(trade)
                    cooldown_until_index = index + request.signal_cooldown_bars

            signal = signals[index]
            if (
                position is None
                and pending_entry is None
                and index < len(bars) - 1
                and index > cooldown_until_index
                and signal.side in {SignalSide.LONG, SignalSide.SHORT}
                and signal.entry_price is not None
                and signal.stop_price is not None
                and signal.target_price is not None
            ):
                stop_distance = abs(signal.entry_price - signal.stop_price)
                target_distance = abs(signal.target_price - signal.entry_price)
                if stop_distance > 0.0 and target_distance > 0.0:
                    size_fraction = float(signal.metadata.get("risk_size_fraction", request.risk_per_trade_fraction))
                    max_leverage = float(signal.metadata.get("risk_max_leverage", request.max_leverage))
                    pending_entry = _PendingEntry(
                        signal=signal,
                        generated_index=index,
                        stop_distance=stop_distance,
                        target_distance=target_distance,
                        size_fraction=size_fraction,
                        max_leverage=max_leverage,
                    )

            equity_points.append((timestamp, self._equity_value(cash, position, bar.close)))

        if position is not None:
            last_bar = bars[-1]
            final_trade, cash = self._close_position(
                position=position,
                exit_bar=last_bar,
                exit_price=last_bar.close,
                exit_reason="end_of_data",
                cash=cash,
                fee_rate=fee_rate,
                index=len(bars) - 1,
            )
            trades.append(final_trade)
            equity_points[-1] = (pd.Timestamp(last_bar.timestamp), cash)

        equity_series = pd.Series(
            data=[equity for _, equity in equity_points],
            index=pd.DatetimeIndex([timestamp for timestamp, _ in equity_points]),
            dtype=float,
        )
        metrics = self._compute_metrics(
            trades=trades,
            equity_series=equity_series,
            initial_capital=request.initial_capital or self.initial_capital,
        )

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
                "initial_capital": request.initial_capital or self.initial_capital,
                "fees_bps": self.fee_bps,
                "slippage_bps": self.slippage_bps,
                "latency_ms": self.latency_ms,
                "mode": "baseline",
                "intrabar_exit_policy": intrabar_policy,
                "equity_curve": [
                    {"timestamp": timestamp.isoformat(), "equity": equity}
                    for timestamp, equity in equity_points
                ],
                "average_win": metrics["average_win"],
                "average_loss": metrics["average_loss"],
                "exit_reason_counts": metrics["exit_reason_counts"],
            },
        )

    def _normalize_signals(
        self,
        request: BacktestRequest,
        bars: list[MarketBar],
    ) -> list[StrategySignal]:
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

    def _open_position(
        self,
        *,
        pending: _PendingEntry,
        entry_bar: MarketBar,
        index: int,
        cash: float,
        request: BacktestRequest,
        fee_rate: float,
    ) -> tuple[_OpenPosition | None, float]:
        raw_entry_price = self._apply_entry_slippage(pending.signal.side, entry_bar.open)
        risk_budget = cash * pending.size_fraction
        quantity_from_risk = risk_budget / pending.stop_distance if pending.stop_distance > 0.0 else 0.0
        quantity_from_leverage = (cash * pending.max_leverage) / raw_entry_price if raw_entry_price > 0.0 else 0.0
        quantity = min(quantity_from_risk, quantity_from_leverage)
        if quantity <= 0.0 or not math.isfinite(quantity):
            return None, cash

        entry_fee = raw_entry_price * quantity * fee_rate
        cash_after_entry_fee = cash - entry_fee
        if pending.signal.side == SignalSide.LONG:
            stop_price = raw_entry_price - pending.stop_distance
            target_price = raw_entry_price + pending.target_distance
        else:
            stop_price = raw_entry_price + pending.stop_distance
            target_price = raw_entry_price - pending.target_distance

        position = _OpenPosition(
            symbol=pending.signal.symbol,
            side=pending.signal.side,
            entry_timestamp=pd.Timestamp(entry_bar.timestamp),
            entry_index=index,
            entry_price=raw_entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            entry_fee=entry_fee,
            time_stop_bars=pending.signal.time_stop_bars,
            close_on_session_end=pending.signal.close_on_session_end,
            entry_reason=pending.signal.entry_reason,
            signal_metadata=dict(pending.signal.metadata),
        )
        return position, cash_after_entry_fee

    def _evaluate_open_position(
        self,
        *,
        position: _OpenPosition,
        bar: MarketBar,
        feature: FeatureSnapshot,
        index: int,
        cash: float,
        fee_rate: float,
        request: BacktestRequest,
        intrabar_policy: str,
    ) -> tuple[ExecutedTrade | None, float, _OpenPosition | None]:
        stop_hit, target_hit = self._intrabar_exit_flags(position, bar)
        exit_decision = self._resolve_intrabar_exit(
            position=position,
            stop_hit=stop_hit,
            target_hit=target_hit,
            intrabar_policy=intrabar_policy,
        )
        if exit_decision is not None:
            exit_reason, exit_price = exit_decision
            trade, updated_cash = self._close_position(
                position=position,
                exit_bar=bar,
                exit_price=exit_price,
                exit_reason=exit_reason,
                cash=cash,
                fee_rate=fee_rate,
                index=index,
            )
            return trade, updated_cash, None

        if position.close_on_session_end and self._is_session_close(
            bar.timestamp,
            request.session_close_hour_utc,
            request.session_close_minute_utc,
        ):
            trade, updated_cash = self._close_position(
                position=position,
                exit_bar=bar,
                exit_price=bar.close,
                exit_reason="session_close",
                cash=cash,
                fee_rate=fee_rate,
                index=index,
            )
            return trade, updated_cash, None

        bars_held = (index - position.entry_index) + 1
        if position.time_stop_bars is not None and bars_held >= position.time_stop_bars:
            trade, updated_cash = self._close_position(
                position=position,
                exit_bar=bar,
                exit_price=bar.close,
                exit_reason="time_stop",
                cash=cash,
                fee_rate=fee_rate,
                index=index,
            )
            return trade, updated_cash, None

        if request.exit_zscore_threshold is not None:
            zscore = feature.values.get("zscore_distance_to_mean")
            if zscore is not None and math.isfinite(float(zscore)) and abs(float(zscore)) <= request.exit_zscore_threshold:
                trade, updated_cash = self._close_position(
                    position=position,
                    exit_bar=bar,
                    exit_price=bar.close,
                    exit_reason="mean_reversion_exit",
                    cash=cash,
                    fee_rate=fee_rate,
                    index=index,
                )
                return trade, updated_cash, None

        return None, cash, position

    def _close_position(
        self,
        *,
        position: _OpenPosition,
        exit_bar: MarketBar,
        exit_price: float,
        exit_reason: str,
        cash: float,
        fee_rate: float,
        index: int,
    ) -> tuple[ExecutedTrade, float]:
        slipped_exit_price = self._apply_exit_slippage(position.side, exit_price)
        direction = 1.0 if position.side == SignalSide.LONG else -1.0
        gross_pnl = direction * (slipped_exit_price - position.entry_price) * position.quantity
        exit_fee = slipped_exit_price * position.quantity * fee_rate
        cash_after_exit = cash + gross_pnl - exit_fee
        total_fees = position.entry_fee + exit_fee
        net_pnl = gross_pnl - total_fees

        trade = ExecutedTrade(
            symbol=position.symbol,
            side=position.side,
            entry_timestamp=position.entry_timestamp.to_pydatetime(),
            exit_timestamp=exit_bar.timestamp,
            entry_price=position.entry_price,
            exit_price=slipped_exit_price,
            quantity=position.quantity,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            fees_paid=total_fees,
            return_pct=net_pnl / max(position.entry_price * position.quantity, 1e-12),
            bars_held=(index - position.entry_index) + 1,
            exit_reason=exit_reason,
            entry_reason=position.entry_reason,
            metadata=dict(position.signal_metadata),
        )
        return trade, cash_after_exit

    def _intrabar_exit_flags(self, position: _OpenPosition, bar: MarketBar) -> tuple[bool, bool]:
        if position.side == SignalSide.LONG:
            return bar.low <= position.stop_price, bar.high >= position.target_price
        return bar.high >= position.stop_price, bar.low <= position.target_price

    def _resolve_intrabar_exit(
        self,
        *,
        position: _OpenPosition,
        stop_hit: bool,
        target_hit: bool,
        intrabar_policy: str,
    ) -> tuple[str, float] | None:
        if stop_hit and target_hit:
            if intrabar_policy == "stop_first":
                return "stop_loss", position.stop_price
            if intrabar_policy == "target_first":
                return "take_profit", position.target_price
            return self._conservative_intrabar_exit(position)

        if stop_hit:
            return "stop_loss", position.stop_price
        if target_hit:
            return "take_profit", position.target_price
        return None

    def _conservative_intrabar_exit(self, position: _OpenPosition) -> tuple[str, float]:
        if position.side == SignalSide.LONG:
            stop_pnl = position.stop_price - position.entry_price
            target_pnl = position.target_price - position.entry_price
        else:
            stop_pnl = position.entry_price - position.stop_price
            target_pnl = position.entry_price - position.target_price

        if stop_pnl <= target_pnl:
            return "stop_loss", position.stop_price
        return "take_profit", position.target_price

    def _equity_value(self, cash: float, position: _OpenPosition | None, close_price: float) -> float:
        if position is None:
            return cash
        direction = 1.0 if position.side == SignalSide.LONG else -1.0
        unrealized = direction * (close_price - position.entry_price) * position.quantity
        return cash + unrealized

    def _apply_entry_slippage(self, side: SignalSide, price: float) -> float:
        slip = self.slippage_bps / 10000.0
        if side == SignalSide.LONG:
            return price * (1.0 + slip)
        return price * (1.0 - slip)

    def _apply_exit_slippage(self, side: SignalSide, price: float) -> float:
        slip = self.slippage_bps / 10000.0
        if side == SignalSide.LONG:
            return price * (1.0 - slip)
        return price * (1.0 + slip)

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

    def _resolve_intrabar_policy(self, override: str | None) -> str:
        policy = override or self.intrabar_exit_policy
        allowed = {"stop_first", "target_first", "conservative"}
        if policy not in allowed:
            raise ValueError(f"Unsupported intrabar exit policy: {policy}")
        return policy

    def _is_session_close(self, timestamp: object, hour: int, minute: int) -> bool:
        normalized = pd.Timestamp(timestamp).tz_convert("UTC")
        session_close = time(hour, minute)
        return normalized.time() >= session_close
