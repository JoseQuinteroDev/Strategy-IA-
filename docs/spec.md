# Azir Hybrid Quant Framework - Formal Strategy Specification

Sprint 0/1 source of truth: `C:\Users\joseq\Documents\Playground\Azir.mq5`.

This document describes the behavior observed in the MQL5 source code, not a
desired redesign. If comments and executable logic differ, the executable logic
wins and the discrepancy is documented here.

## Scope

- Official baseline: AzirIA MT5.
- EA file: `Azir.mq5`, version string `2.2`.
- Main market: `XAUUSD` / gold, according to the EA header and the audited CSV.
- Default operating timeframe: `PERIOD_M5`.
- No PPO, LSTM, Transformers, or strategy redesign in this sprint.
- Goal of this sprint: formalize, log, and prepare a faithful Python replica.

## Dataset Audit

Audited file:

- `C:\Users\joseq\Documents\xauusd_m5.csv`

Observed schema:

- `open_time`
- `open`
- `high`
- `low`
- `close`
- `volume`

Observed coverage and quality:

- First timestamp: `2021-01-04 01:00:00`
- Last timestamp: `2025-12-31 23:55:00`
- Total rows: `354482`
- Effective dominant timeframe: `5 minutes`
- Bad timestamps: `0`
- Adjacent duplicate timestamps: `0`
- Out-of-order adjacent rows: `0`
- Dominant adjacent delta: `00:05:00`, count `353186`
- Common expected gaps: daily maintenance `23:55 -> 01:00` and weekend gaps
- Timezone: not encoded in the file; apparent broker/server session time.

Assessment:

- The file is suitable for a first M5 bar-based replica/backtest of Azir.
- Timezone/server-time alignment must be frozen before comparing Python results
  against MT5 logs.
- The CSV does not prove whether prices are broker raw, back-adjusted, or
  transformed; it only looks like direct OHLCV export.

## Inputs and Defaults

Core execution inputs observed in `Azir.mq5`:

- `LotSize = 0.1`
- `SL_Points = 500`
- `TP_Points = 500`
- `Trailing_Start_Points = 90`
- `Trailing_Step_Points = 50`
- `MagicNumber = 123456321`
- `Timeframe = PERIOD_M5`
- `NY_Open_Hour = 16`
- `NY_Open_Minute = 30`
- `Close_Hour = 22`
- `SwingBars = 10`
- `AllowBuys = true`
- `AllowSells = true`
- `AllowTrendFilter = true`
- `AllowAtrFilter = true`
- `ATR_Timeframe = PERIOD_M5`
- `ATR_Period = 14`
- `ATR_Minimum = 100`
- `AllowRsiFilter = true`
- `RSI_Timeframe = PERIOD_M1`
- `RSI_Period = 14`
- `RSI_Bullish_Threshold = 70`
- `RSI_Sell_Threshold = 30`
- `MinDistanceBetweenPendings = 200`
- `NoTradeFridays = true`

## Timezone and Session Dependency

The EA uses `TimeCurrent()` and `TimeToStruct()` directly. Therefore:

- All scheduling is in MT5 broker/server time.
- The input comments mention GMT+3 for the New York open and close settings.
- There is no explicit timezone conversion or DST handling in the EA.
- Python replication must use the same broker/server timestamp convention.

## Daily Reset

On every tick, the EA checks:

- `new_day = iTime(_Symbol, PERIOD_D1, 0)`

When the day changes, it resets:

- `orders_placed_today`
- `rsi_gate_required_today`
- `last_buy_entry_price`
- `last_sell_entry_price`
- current day state
- logger daily state

## Friday Filter

If `NoTradeFridays` is true and `dt.day_of_week == 5`:

- the EA prints that it does not trade Fridays,
- updates the HUD,
- returns before setup/order logic.

The new logger records a `blocked_friday` event only at the configured setup
minute to avoid logging every tick.

## Setup Time

The EA evaluates daily pending order placement only when:

- `dt.hour == NY_Open_Hour`
- `dt.min == NY_Open_Minute`
- `orders_placed_today == false`

Default setup time:

- `16:30` broker/server time.

## Swing High / Low

At setup time, the EA calculates:

- swing high: highest high over the last `SwingBars` closed M5 bars, shift `1`
- swing low: lowest low over the last `SwingBars` closed M5 bars, shift `1`

Default:

- `SwingBars = 10`

This avoids using the currently forming bar.

## Entry Offset

The entry offset is not an input. It is hardcoded:

- `offset_points_value = 5`
- `entry_offset = offset_points_value * point`

Entries:

- `buy_entry = swing_high + entry_offset`
- `sell_entry = swing_low - entry_offset`

## EMA20 Trend Filter

The EA creates:

- `EMA20 = iMA(_Symbol, Timeframe, 20, 0, MODE_EMA, PRICE_CLOSE)`

At setup time it reads the closed bar value with shift `1`.

It also reads:

- `prev_close = iClose(_Symbol, Timeframe, 1)`

If `AllowTrendFilter` is true:

- if `prev_close > EMA20` and buys are allowed, it sends a BuyStop only.
- else if `prev_close < EMA20` and sells are allowed, it sends a SellStop only.
- if equal, it sends no order.

If `AllowTrendFilter` is false:

- it may send both BuyStop and SellStop, depending on `AllowBuys` and
  `AllowSells`.

## ATR Filter

The EA creates:

- `ATR = iATR(_Symbol, ATR_Timeframe, ATR_Period)`

At setup time it reads the closed bar value with shift `1`.

It computes:

- `atr_points = atr / point`

If `AllowAtrFilter` is true and `atr_points < ATR_Minimum`:

- no orders are sent for the day.
- the logger records an `opportunity` row with `atr_filter_passed=false`.

## Pending Orders

Order type:

- BuyStop above swing high.
- SellStop below swing low.

Order time:

- `ORDER_TIME_GTC`

Stop/target:

- Buy SL: `buy_entry - SL_Points * point`
- Buy TP: `buy_entry + TP_Points * point`
- Sell SL: `sell_entry + SL_Points * point`
- Sell TP: `sell_entry - TP_Points * point`

The day is marked as having placed orders only if at least one order send
returns success.

## RSI Gate

The EA creates:

- `RSI = iRSI(_Symbol, RSI_Timeframe, RSI_Period, PRICE_CLOSE)`

The gate activates only if all conditions are true:

- `AllowRsiFilter == true`
- both BuyStop and SellStop were placed
- distance between pending entries is at least `MinDistanceBetweenPendings`

If active, on a trade transaction:

- BUY passes if current RSI is `>= RSI_Bullish_Threshold`
- SELL passes if current RSI is `<= RSI_Sell_Threshold`
- if the gate fails, the EA closes the position.
- if the gate passes, the EA cancels the opposite pending order.

Important observed behavior:

- With default `AllowTrendFilter=true`, the EA sends only one side because of an
  `if / else if` branch. Therefore the RSI gate normally cannot activate.

## Opposite Pending Cancellation

The code cancels the opposite pending order only inside the RSI gate pass branch.

Therefore:

- If the RSI gate is inactive, the transaction handler returns before opposite
  cancellation.
- With `AllowTrendFilter=false` and RSI gate inactive, both pending orders can
  remain unless later closed by the end-of-session cleanup.

## Trailing

For each open EA position on every tick:

- profit in points is calculated from current bid/ask vs open price.
- if profit points are at least `Trailing_Start_Points`, the EA attempts to move
  SL by `Trailing_Step_Points`.
- it only improves the stop and respects `SYMBOL_TRADE_STOPS_LEVEL`.

The logger records each successful trailing modification.

## End-of-Session Close / Cleanup

At:

- `dt.hour == Close_Hour`
- `dt.min == 0`

Default:

- `22:00` broker/server time.

The EA:

- closes all EA positions for the current symbol and magic,
- deletes all EA pending orders for the current symbol and magic.

This is minute-specific and depends on receiving a tick during that minute.

## Exported Event Log

The EA now includes `AzirEventLogger.mqh`.

Inputs:

- `EnableEventLogging = true`
- `EventLogFileName = ""`
- `EventLogUseCommonFolder = true`

Default output file:

- `azir_events_<symbol>_<magic>.csv`

Default location when common folder is enabled:

- `Terminal common data path\Files\`

## Known Discrepancies and Risks

- The strategy refers to New York open, but the EA uses raw broker/server time.
- The comment says GMT+3; the code does not enforce GMT+3.
- Entry offset is hardcoded to 5 points, not configurable.
- RSI gate is effectively disabled under default trend-filter behavior because
  only one pending side is placed.
- Opposite pending cancellation only occurs after RSI gate pass, not after every
  fill.
- The transaction handler does not explicitly filter `DEAL_ENTRY` before RSI
  gate logic; exit deals can enter the same branch if the gate is active.
- End-of-session cleanup only executes during `Close_Hour:00`, so it depends on
  a tick arriving in that minute.

## Next Sprint Readiness

The replica phase now consumes:

- this formal spec,
- the audited M5 CSV,
- the MT5 event log produced by Azir.

The replica acceptance test should compare daily opportunity rows first, then
fills/exits, before any RL environment is reintroduced.

## Python Replica Status

Implemented module:

- `src/hybrid_quant/azir/replica.py`

Runner:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.azir `
  --input-path "C:\Users\joseq\Documents\xauusd_m5.csv" `
  --mt5-log-path "C:\path\to\azir_events_XAUUSD_123456321.csv" `
  --output-dir "artifacts\azir-replica"
```

Faithfully reproduced from the MQL5 source:

- setup time in raw server time,
- Friday blocking,
- swing high/low from the last closed bars,
- hardcoded 5-point entry offset,
- EMA20 `if / else if` trend branch,
- ATR minimum filter,
- default RSI gate activation quirk,
- end-of-session cleanup time,
- opposite pending cancellation only after RSI gate pass.

Approximate in Python because the historical file is M5 OHLCV, not MT5 ticks:

- pending fill ordering inside one M5 candle,
- trailing stop movement from live bid/ask ticks,
- same-bar SL/TP ambiguity,
- broker commission/swap/slippage,
- M1 RSI gate unless a separate M1 file is provided to the runner.
