"""Faithful-enough Python replica of the AzirIA MT5 baseline.

The MQL5 EA remains the source of truth. This module intentionally mirrors the
observed implementation, including quirks such as the hardcoded 5-point entry
offset and the RSI gate only activating when both pending orders exist.
"""

from __future__ import annotations

import csv
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from .event_log import AZIR_EVENT_COLUMNS


@dataclass(frozen=True)
class OhlcvBar:
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class AzirReplicaConfig:
    symbol: str = "XAUUSD"
    magic: int = 123456321
    lot_size: float = 0.10
    sl_points: int = 500
    tp_points: int = 500
    trailing_start_points: int = 90
    trailing_step_points: int = 50
    point: float = 0.01
    contract_size: float = 100.0
    timeframe: str = "M5"
    ny_open_hour: int = 16
    ny_open_minute: int = 30
    close_hour: int = 22
    swing_bars: int = 10
    buy_swing_bars: int | None = None
    sell_swing_bars: int | None = None
    swing_definition: str = "rolling"
    fractal_side_bars: int = 2
    entry_offset_points: float = 5.0
    buy_entry_offset_points: float | None = None
    sell_entry_offset_points: float | None = None
    entry_offset_atr_fraction: float | None = None
    buy_entry_offset_atr_fraction: float | None = None
    sell_entry_offset_atr_fraction: float | None = None
    range_quality_enabled: bool = False
    min_range_width_atr: float | None = None
    max_range_width_atr: float | None = None
    compression_lookback_bars: int = 20
    max_compression_range_atr: float | None = None
    allow_buys: bool = True
    allow_sells: bool = True
    allow_trend_filter: bool = True
    allow_atr_filter: bool = True
    atr_period: int = 14
    atr_minimum: float = 100.0
    allow_rsi_filter: bool = True
    rsi_period: int = 14
    rsi_bullish_threshold: float = 70.0
    rsi_sell_threshold: float = 30.0
    min_distance_between_pendings: float = 200.0
    no_trade_fridays: bool = True
    simulated_order_retcode: int = 10008


def load_ohlcv_csv(path: str | Path) -> list[OhlcvBar]:
    """Load OHLCV data using the internal schema from the import/audit sprint."""

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"OHLCV file does not exist: {input_path}")

    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
        reader = csv.DictReader(handle, dialect=dialect)
        required = {"open_time", "open", "high", "low", "close", "volume"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"OHLCV file is missing required columns: {sorted(missing)}")
        rows: list[OhlcvBar] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                rows.append(
                    OhlcvBar(
                        open_time=_parse_datetime(row["open_time"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                )
            except Exception as exc:  # pragma: no cover - includes source context.
                raise ValueError(f"Invalid OHLCV row at line {line_number}: {row}") from exc
    rows.sort(key=lambda bar: bar.open_time)
    return rows


def _parse_datetime(value: str) -> datetime:
    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y.%m.%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return datetime.fromisoformat(value)


def ema(values: list[float], period: int) -> list[float | None]:
    alpha = 2.0 / (period + 1.0)
    result: list[float | None] = []
    current: float | None = None
    for value in values:
        current = value if current is None else (value * alpha + current * (1.0 - alpha))
        result.append(current)
    return result


def atr(bars: list[OhlcvBar], period: int) -> list[float | None]:
    """Replicate MT5 iATR as observed in Azir logs.

    Empirical parity against `todos_los_ticks.csv` showed that MT5's value in
    this setup matches a simple moving average of True Range over the last
    `period` closed bars, read with shift 1 at setup time.
    """

    true_ranges: list[float] = []
    for index, bar in enumerate(bars):
        if index == 0:
            true_ranges.append(bar.high - bar.low)
            continue
        previous_close = bars[index - 1].close
        true_ranges.append(
            max(
                bar.high - bar.low,
                abs(bar.high - previous_close),
                abs(bar.low - previous_close),
            )
        )

    result: list[float | None] = [None] * len(bars)
    for index in range(period - 1, len(true_ranges)):
        current = sum(true_ranges[index - period + 1 : index + 1]) / period
        result[index] = current
    return result


def rsi(bars: list[OhlcvBar], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(bars)
    if len(bars) <= period:
        return result

    gains: list[float] = []
    losses: list[float] = []
    for index in range(1, period + 1):
        change = bars[index].close - bars[index - 1].close
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    result[period] = _rsi_from_averages(avg_gain, avg_loss)

    for index in range(period + 1, len(bars)):
        change = bars[index].close - bars[index - 1].close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        result[index] = _rsi_from_averages(avg_gain, avg_loss)

    return result


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


class AzirPythonReplica:
    """Bar-based Azir replica for research parity checks."""

    def __init__(
        self,
        bars: list[OhlcvBar],
        config: AzirReplicaConfig | None = None,
        rsi_bars: list[OhlcvBar] | None = None,
    ) -> None:
        if not bars:
            raise ValueError("Azir replica requires at least one OHLCV bar")
        self.bars = sorted(bars, key=lambda bar: bar.open_time)
        self.rsi_bars = sorted(rsi_bars or bars, key=lambda bar: bar.open_time)
        self.config = config or AzirReplicaConfig()
        self._ema20 = ema([bar.close for bar in self.bars], 20)
        self._atr = atr(self.bars, self.config.atr_period)
        self._rsi = rsi(self.rsi_bars, self.config.rsi_period)
        self._rsi_times = [bar.open_time for bar in self.rsi_bars]

    def run(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        setup_time = time(self.config.ny_open_hour, self.config.ny_open_minute)
        seen_dates: set[date] = set()

        for setup_index, bar in enumerate(self.bars):
            current_date = bar.open_time.date()
            if current_date in seen_dates or bar.open_time.time() != setup_time:
                continue
            seen_dates.add(current_date)
            rows.extend(self._evaluate_day(setup_index))
        return rows

    def _evaluate_day(self, setup_index: int) -> list[dict[str, Any]]:
        cfg = self.config
        setup_bar = self.bars[setup_index]
        dt = setup_bar.open_time
        day_of_week = _mql_day_of_week(dt)
        is_friday = day_of_week == 5

        if cfg.no_trade_fridays and is_friday:
            return [
                self._event(
                    dt,
                    "blocked_friday",
                    day_of_week=day_of_week,
                    is_friday=True,
                    notes="NoTradeFridays blocked the daily opportunity before order evaluation.",
                )
            ]

        buy_swing_bars = cfg.buy_swing_bars or cfg.swing_bars
        sell_swing_bars = cfg.sell_swing_bars or cfg.swing_bars
        required_history = max(
            cfg.swing_bars,
            buy_swing_bars,
            sell_swing_bars,
            cfg.atr_period,
            cfg.compression_lookback_bars,
            20,
        )
        if setup_index < required_history + 1:
            return []

        previous_index = setup_index - 1
        swing_high = self._swing_high(setup_index, buy_swing_bars)
        swing_low = self._swing_low(setup_index, sell_swing_bars)
        prev_close = self.bars[previous_index].close
        ema20 = self._ema20[previous_index]
        atr_value = self._atr[previous_index]
        if ema20 is None or atr_value is None:
            return []

        atr_points = atr_value / cfg.point
        buy_offset_points = self._entry_offset_points("buy", atr_points)
        sell_offset_points = self._entry_offset_points("sell", atr_points)
        buy_entry = swing_high + buy_offset_points * cfg.point
        sell_entry = swing_low - sell_offset_points * cfg.point
        pending_distance_points = (buy_entry - sell_entry) / cfg.point
        swing_range_points = (swing_high - swing_low) / cfg.point
        range_width_atr = swing_range_points / atr_points if atr_points else 0.0
        compression_range_atr = self._compression_range_atr(setup_index, atr_points)
        rsi_setup = self._closed_rsi_before(dt)
        buy_allowed_by_trend = (not cfg.allow_trend_filter or prev_close > ema20) and cfg.allow_buys
        sell_allowed_by_trend = (not cfg.allow_trend_filter or prev_close < ema20) and cfg.allow_sells
        spread_points = 0.0

        common = {
            "day_of_week": day_of_week,
            "is_friday": is_friday,
            "timeframe": cfg.timeframe,
            "ny_open_hour": cfg.ny_open_hour,
            "ny_open_minute": cfg.ny_open_minute,
            "close_hour": cfg.close_hour,
            "swing_bars": cfg.swing_bars,
            "buy_swing_bars": buy_swing_bars,
            "sell_swing_bars": sell_swing_bars,
            "swing_definition": cfg.swing_definition,
            "entry_offset_points": cfg.entry_offset_points,
            "buy_entry_offset_points": buy_offset_points,
            "sell_entry_offset_points": sell_offset_points,
            "lot_size": cfg.lot_size,
            "sl_points": cfg.sl_points,
            "tp_points": cfg.tp_points,
            "trailing_start_points": cfg.trailing_start_points,
            "trailing_step_points": cfg.trailing_step_points,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "buy_entry": buy_entry,
            "sell_entry": sell_entry,
            "pending_distance_points": pending_distance_points,
            "swing_range_points": swing_range_points,
            "range_width_atr": range_width_atr,
            "compression_range_atr": compression_range_atr,
            "range_quality_enabled": cfg.range_quality_enabled,
            "spread_points": spread_points,
            "ema20": ema20,
            "prev_close": prev_close,
            "prev_close_vs_ema20_points": (prev_close - ema20) / cfg.point,
            "prev_close_above_ema20": prev_close > ema20,
            "atr": atr_value,
            "atr_points": atr_points,
            "atr_filter_enabled": cfg.allow_atr_filter,
            "atr_minimum": cfg.atr_minimum,
            "rsi": rsi_setup,
            "rsi_gate_enabled": cfg.allow_rsi_filter,
            "rsi_bullish_threshold": cfg.rsi_bullish_threshold,
            "rsi_sell_threshold": cfg.rsi_sell_threshold,
            "allow_buys": cfg.allow_buys,
            "allow_sells": cfg.allow_sells,
            "trend_filter_enabled": cfg.allow_trend_filter,
            "buy_allowed_by_trend": buy_allowed_by_trend,
            "sell_allowed_by_trend": sell_allowed_by_trend,
        }

        if cfg.allow_atr_filter and atr_points < cfg.atr_minimum:
            return [
                self._event(
                    dt,
                    "opportunity",
                    **common,
                    atr_filter_passed=False,
                    rsi_gate_required=False,
                    rsi_gate_passed=True,
                    buy_order_placed=False,
                    sell_order_placed=False,
                    buy_retcode=0,
                    sell_retcode=0,
                    notes="ATR filter failed; no orders were sent.",
                )
            ]

        range_quality_passed = self._range_quality_passed(range_width_atr, compression_range_atr)
        if cfg.range_quality_enabled and not range_quality_passed:
            return [
                self._event(
                    dt,
                    "opportunity",
                    **common,
                    atr_filter_passed=True,
                    range_quality_passed=False,
                    rsi_gate_required=False,
                    rsi_gate_passed=True,
                    buy_order_placed=False,
                    sell_order_placed=False,
                    buy_retcode=0,
                    sell_retcode=0,
                    notes="Range quality filter failed; no orders were sent.",
                )
            ]

        buy_placed = False
        sell_placed = False
        if cfg.allow_trend_filter:
            if prev_close > ema20 and cfg.allow_buys:
                buy_placed = True
            elif prev_close < ema20 and cfg.allow_sells:
                sell_placed = True
        else:
            buy_placed = cfg.allow_buys
            sell_placed = cfg.allow_sells

        rsi_gate_required = (
            cfg.allow_rsi_filter
            and buy_placed
            and sell_placed
            and pending_distance_points >= cfg.min_distance_between_pendings
        )
        if not buy_placed and not sell_placed:
            notes = "No pending order placed after trend/direction/order-send evaluation."
        elif rsi_gate_required:
            notes = (
                "Pending order placement evaluated; RSI gate activated because both sides "
                "were placed and distance threshold passed."
            )
        else:
            notes = "Pending order placement evaluated; RSI gate inactive for this opportunity."

        rows = [
            self._event(
                dt,
                "opportunity",
                **common,
                atr_filter_passed=True,
                range_quality_passed=True,
                rsi_gate_required=rsi_gate_required,
                rsi_gate_passed=True,
                buy_order_placed=buy_placed,
                sell_order_placed=sell_placed,
                buy_retcode=cfg.simulated_order_retcode if buy_placed else 0,
                sell_retcode=cfg.simulated_order_retcode if sell_placed else 0,
                notes=notes,
            )
        ]
        if buy_placed or sell_placed:
            rows.extend(
                self._simulate_orders(
                    setup_index=setup_index,
                    buy_entry=buy_entry,
                    sell_entry=sell_entry,
                    buy_placed=buy_placed,
                    sell_placed=sell_placed,
                    rsi_gate_required=rsi_gate_required,
                )
            )
        return rows

    def _simulate_orders(
        self,
        *,
        setup_index: int,
        buy_entry: float,
        sell_entry: float,
        buy_placed: bool,
        sell_placed: bool,
        rsi_gate_required: bool,
    ) -> list[dict[str, Any]]:
        cfg = self.config
        setup_dt = self.bars[setup_index].open_time
        close_index = self._close_index_for_day(setup_index)
        if close_index is None:
            close_index = len(self.bars) - 1

        for index in range(setup_index, close_index):
            bar = self.bars[index]
            hit_buy = buy_placed and bar.high >= buy_entry
            hit_sell = sell_placed and bar.low <= sell_entry
            if not hit_buy and not hit_sell:
                continue

            side = self._resolve_fill_side(bar, hit_buy, hit_sell, buy_entry, sell_entry)
            entry = buy_entry if side == "buy" else sell_entry
            fill_rsi = self._rsi_at_or_before(bar.open_time)
            rsi_pass = True
            if rsi_gate_required:
                if side == "buy":
                    rsi_pass = fill_rsi is not None and fill_rsi >= cfg.rsi_bullish_threshold
                else:
                    rsi_pass = fill_rsi is not None and fill_rsi <= cfg.rsi_sell_threshold

            rows = [
                self._event(
                    bar.open_time,
                    "fill",
                    fill_side=side,
                    fill_price=entry,
                    duration_to_fill_seconds=int((bar.open_time - setup_dt).total_seconds()),
                    rsi=fill_rsi,
                    rsi_gate_enabled=cfg.allow_rsi_filter,
                    rsi_gate_required=rsi_gate_required,
                    rsi_gate_passed=rsi_pass,
                    slippage_points=0.0,
                    notes="Simulated OHLC fill; tick ordering is not available in the CSV.",
                )
            ]
            if rsi_gate_required and not rsi_pass:
                rows.append(
                    self._exit_event(
                        timestamp=bar.open_time,
                        side=side,
                        entry=entry,
                        exit_price=entry,
                        exit_reason="rsi_gate_rejected",
                        mfe_points=0.0,
                        mae_points=0.0,
                        trailing_activated=False,
                        trailing_modifications=0,
                        opposite_order_cancelled=False,
                    )
                )
                return rows
            if rsi_gate_required and buy_placed and sell_placed:
                rows.append(
                    self._event(
                        bar.open_time,
                        "opposite_pending_cancelled",
                        fill_side=side,
                        fill_price=entry,
                        opposite_order_cancelled=True,
                        notes="RSI gate passed; simulated cancellation of opposite pending.",
                    )
                )
            rows.extend(self._simulate_exit(index, close_index, side, entry))
            return rows

        return [
            self._event(
                self.bars[close_index].open_time,
                "no_fill_close_cleanup",
                fill_side="no_fill",
                exit_reason="close_hour_cleanup",
                gross_pnl=0.0,
                net_pnl=0.0,
                notes="Close hour reached with pending orders but no fill.",
            )
        ]

    def _simulate_exit(
        self,
        fill_index: int,
        close_index: int,
        side: str,
        entry: float,
    ) -> list[dict[str, Any]]:
        cfg = self.config
        is_buy = side == "buy"
        stop = entry - cfg.sl_points * cfg.point if is_buy else entry + cfg.sl_points * cfg.point
        target = entry + cfg.tp_points * cfg.point if is_buy else entry - cfg.tp_points * cfg.point
        mfe_points = 0.0
        mae_points = 0.0
        trailing_activated = False
        trailing_modifications = 0
        rows: list[dict[str, Any]] = []

        for index in range(fill_index, close_index):
            bar = self.bars[index]
            favorable = (bar.high - entry) / cfg.point if is_buy else (entry - bar.low) / cfg.point
            adverse = (entry - bar.low) / cfg.point if is_buy else (bar.high - entry) / cfg.point
            mfe_points = max(mfe_points, favorable)
            mae_points = max(mae_points, adverse)

            if is_buy:
                if bar.low <= stop:
                    rows.append(
                        self._exit_event(
                            timestamp=bar.open_time,
                            side=side,
                            entry=entry,
                            exit_price=stop,
                            exit_reason="stop_loss_or_trailing_stop",
                            mfe_points=mfe_points,
                            mae_points=mae_points,
                            trailing_activated=trailing_activated,
                            trailing_modifications=trailing_modifications,
                            opposite_order_cancelled=False,
                        )
                    )
                    return rows
                if bar.high >= target:
                    rows.append(
                        self._exit_event(
                            timestamp=bar.open_time,
                            side=side,
                            entry=entry,
                            exit_price=target,
                            exit_reason="take_profit",
                            mfe_points=mfe_points,
                            mae_points=mae_points,
                            trailing_activated=trailing_activated,
                            trailing_modifications=trailing_modifications,
                            opposite_order_cancelled=False,
                        )
                    )
                    return rows
                if favorable >= cfg.trailing_start_points:
                    new_stop = bar.high - cfg.trailing_step_points * cfg.point
                    if new_stop > stop:
                        stop = new_stop
                        trailing_activated = True
                        trailing_modifications += 1
                        rows.append(self._trailing_event(bar.open_time, side, entry, mfe_points, mae_points))
            else:
                if bar.high >= stop:
                    rows.append(
                        self._exit_event(
                            timestamp=bar.open_time,
                            side=side,
                            entry=entry,
                            exit_price=stop,
                            exit_reason="stop_loss_or_trailing_stop",
                            mfe_points=mfe_points,
                            mae_points=mae_points,
                            trailing_activated=trailing_activated,
                            trailing_modifications=trailing_modifications,
                            opposite_order_cancelled=False,
                        )
                    )
                    return rows
                if bar.low <= target:
                    rows.append(
                        self._exit_event(
                            timestamp=bar.open_time,
                            side=side,
                            entry=entry,
                            exit_price=target,
                            exit_reason="take_profit",
                            mfe_points=mfe_points,
                            mae_points=mae_points,
                            trailing_activated=trailing_activated,
                            trailing_modifications=trailing_modifications,
                            opposite_order_cancelled=False,
                        )
                    )
                    return rows
                if favorable >= cfg.trailing_start_points:
                    new_stop = bar.low + cfg.trailing_step_points * cfg.point
                    if new_stop < stop:
                        stop = new_stop
                        trailing_activated = True
                        trailing_modifications += 1
                        rows.append(self._trailing_event(bar.open_time, side, entry, mfe_points, mae_points))

        close_bar = self.bars[close_index]
        rows.append(
            self._exit_event(
                timestamp=close_bar.open_time,
                side=side,
                entry=entry,
                exit_price=close_bar.open,
                exit_reason="expert_close_or_session_close",
                mfe_points=mfe_points,
                mae_points=mae_points,
                trailing_activated=trailing_activated,
                trailing_modifications=trailing_modifications,
                opposite_order_cancelled=False,
            )
        )
        return rows

    def _trailing_event(
        self,
        timestamp: datetime,
        side: str,
        entry: float,
        mfe_points: float,
        mae_points: float,
    ) -> dict[str, Any]:
        return self._event(
            timestamp,
            "trailing_modified",
            fill_side=side,
            fill_price=entry,
            mfe_points=mfe_points,
            mae_points=mae_points,
            trailing_activated=True,
            trailing_modifications=1,
            trailing_outcome="sl_modified",
            notes="Bar-based trailing approximation from OHLC data.",
        )

    def _exit_event(
        self,
        *,
        timestamp: datetime,
        side: str,
        entry: float,
        exit_price: float,
        exit_reason: str,
        mfe_points: float,
        mae_points: float,
        trailing_activated: bool,
        trailing_modifications: int,
        opposite_order_cancelled: bool,
    ) -> dict[str, Any]:
        cfg = self.config
        gross_pnl = self._gross_pnl(side, entry, exit_price)
        return self._event(
            timestamp,
            "exit",
            fill_side=side,
            fill_price=entry,
            mfe_points=mfe_points,
            mae_points=mae_points,
            exit_reason=exit_reason,
            gross_pnl=gross_pnl,
            net_pnl=gross_pnl,
            commission=0.0,
            swap=0.0,
            trailing_activated=trailing_activated,
            trailing_modifications=trailing_modifications,
            trailing_outcome="exit_after_trailing" if trailing_activated else "",
            opposite_order_cancelled=opposite_order_cancelled,
            notes=(
                "Simulated PnL uses lot_size * contract_size and zero commission; "
                "MT5 broker deal history remains empirical source of truth."
            ),
        )

    def _event(self, timestamp: datetime, event_type: str, **values: Any) -> dict[str, Any]:
        cfg = self.config
        row: dict[str, Any] = {column: "" for column in AZIR_EVENT_COLUMNS}
        row.update(
            {
                "timestamp": _format_dt(timestamp),
                "event_id": f"{timestamp.date().isoformat()}_{cfg.symbol}_{cfg.magic}",
                "event_type": event_type,
                "symbol": cfg.symbol,
                "magic": cfg.magic,
                "day_of_week": values.pop("day_of_week", _mql_day_of_week(timestamp)),
                "is_friday": values.pop("is_friday", _mql_day_of_week(timestamp) == 5),
                "server_time": _format_dt(timestamp),
                "timeframe": cfg.timeframe,
                "ny_open_hour": cfg.ny_open_hour,
                "ny_open_minute": cfg.ny_open_minute,
                "close_hour": cfg.close_hour,
                "swing_bars": cfg.swing_bars,
                "lot_size": cfg.lot_size,
                "sl_points": cfg.sl_points,
                "tp_points": cfg.tp_points,
                "trailing_start_points": cfg.trailing_start_points,
                "trailing_step_points": cfg.trailing_step_points,
                "atr_filter_enabled": cfg.allow_atr_filter,
                "atr_minimum": cfg.atr_minimum,
                "rsi_gate_enabled": cfg.allow_rsi_filter,
                "rsi_bullish_threshold": cfg.rsi_bullish_threshold,
                "rsi_sell_threshold": cfg.rsi_sell_threshold,
                "allow_buys": cfg.allow_buys,
                "allow_sells": cfg.allow_sells,
                "trend_filter_enabled": cfg.allow_trend_filter,
            }
        )
        row.update(values)
        return row

    def _closed_rsi_before(self, timestamp: datetime) -> float | None:
        index = bisect_left(self._rsi_times, timestamp) - 1
        if index < 0:
            return None
        return self._rsi[index]

    def _rsi_at_or_before(self, timestamp: datetime) -> float | None:
        index = bisect_right(self._rsi_times, timestamp) - 1
        if index < 0:
            return None
        return self._rsi[index]

    def _close_index_for_day(self, setup_index: int) -> int | None:
        setup_date = self.bars[setup_index].open_time.date()
        close_time = time(self.config.close_hour, 0)
        for index in range(setup_index, len(self.bars)):
            bar = self.bars[index]
            if bar.open_time.date() != setup_date:
                return index - 1
            if bar.open_time.time() == close_time:
                return index
        return None

    def _resolve_fill_side(
        self,
        bar: OhlcvBar,
        hit_buy: bool,
        hit_sell: bool,
        buy_entry: float,
        sell_entry: float,
    ) -> str:
        if hit_buy and not hit_sell:
            return "buy"
        if hit_sell and not hit_buy:
            return "sell"
        return "buy" if abs(bar.open - buy_entry) <= abs(bar.open - sell_entry) else "sell"

    def _gross_pnl(self, side: str, entry: float, exit_price: float) -> float:
        direction = 1.0 if side == "buy" else -1.0
        return (exit_price - entry) * direction * self.config.lot_size * self.config.contract_size

    def _swing_high(self, setup_index: int, lookback: int) -> float:
        window = self.bars[setup_index - lookback : setup_index]
        if self.config.swing_definition == "fractal":
            pivot = self._last_confirmed_pivot_high(setup_index, lookback)
            if pivot is not None:
                return pivot
        return max(bar.high for bar in window)

    def _swing_low(self, setup_index: int, lookback: int) -> float:
        window = self.bars[setup_index - lookback : setup_index]
        if self.config.swing_definition == "fractal":
            pivot = self._last_confirmed_pivot_low(setup_index, lookback)
            if pivot is not None:
                return pivot
        return min(bar.low for bar in window)

    def _last_confirmed_pivot_high(self, setup_index: int, lookback: int) -> float | None:
        side = max(1, self.config.fractal_side_bars)
        start = max(0, setup_index - lookback)
        end = setup_index
        for index in range(end - side - 1, start + side - 1, -1):
            candidate = self.bars[index].high
            left = [bar.high for bar in self.bars[index - side : index]]
            right = [bar.high for bar in self.bars[index + 1 : index + side + 1]]
            if left and right and candidate > max(left) and candidate > max(right):
                return candidate
        return None

    def _last_confirmed_pivot_low(self, setup_index: int, lookback: int) -> float | None:
        side = max(1, self.config.fractal_side_bars)
        start = max(0, setup_index - lookback)
        end = setup_index
        for index in range(end - side - 1, start + side - 1, -1):
            candidate = self.bars[index].low
            left = [bar.low for bar in self.bars[index - side : index]]
            right = [bar.low for bar in self.bars[index + 1 : index + side + 1]]
            if left and right and candidate < min(left) and candidate < min(right):
                return candidate
        return None

    def _entry_offset_points(self, side: str, atr_points: float) -> float:
        cfg = self.config
        fixed = cfg.buy_entry_offset_points if side == "buy" else cfg.sell_entry_offset_points
        atr_fraction = cfg.buy_entry_offset_atr_fraction if side == "buy" else cfg.sell_entry_offset_atr_fraction
        if fixed is None:
            fixed = cfg.entry_offset_points
        if atr_fraction is None:
            atr_fraction = cfg.entry_offset_atr_fraction
        if atr_fraction is not None:
            return atr_points * atr_fraction
        return fixed

    def _compression_range_atr(self, setup_index: int, atr_points: float) -> float:
        lookback = max(1, self.config.compression_lookback_bars)
        if setup_index < lookback or atr_points <= 0.0:
            return 0.0
        window = self.bars[setup_index - lookback : setup_index]
        return ((max(bar.high for bar in window) - min(bar.low for bar in window)) / self.config.point) / atr_points

    def _range_quality_passed(self, range_width_atr: float, compression_range_atr: float) -> bool:
        cfg = self.config
        if cfg.min_range_width_atr is not None and range_width_atr < cfg.min_range_width_atr:
            return False
        if cfg.max_range_width_atr is not None and range_width_atr > cfg.max_range_width_atr:
            return False
        if cfg.max_compression_range_atr is not None and compression_range_atr > cfg.max_compression_range_atr:
            return False
        return True


def _mql_day_of_week(value: datetime) -> int:
    # Python: Monday=0..Sunday=6. MQL: Sunday=0..Saturday=6.
    return (value.weekday() + 1) % 7


def _format_dt(value: datetime) -> str:
    return value.strftime("%Y.%m.%d %H:%M:%S")
