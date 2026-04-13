# baseline_azir_setup_frozen_v1

Status: FROZEN for setup/filtro parity only.

## Sources of Truth

- `C:\Users\joseq\Documents\Playground\Azir.mq5`
- `C:\Users\joseq\Documents\Playground\todos_los_ticks.csv`
- `artifacts\azir-parity-final-setup-m5`

## Scope

- Symbol: `XAUUSD-STD`
- Timeframe: M5 setup, M1 RSI gate state when available in MT5
- Server time: MT5 broker/server time; no timezone conversion in EA.
- First event timestamp: `2021-01-04 16:30:00`
- Last event timestamp: `2025-12-30 16:30:45`
- First setup day: `2021-01-04`
- Last setup day: `2025-12-30`

## Inputs Frozen

- Setup time: 16:30 broker/server time
- Close hour: 22 broker/server time
- Swing bars: 10
- Entry offset: 5 points
- EMA filter: EMA20 on M5 closed bar shift 1
- ATR filter: SMA true range parity with MT5 iATR observed value; 14 closed M5 bars shift 1
- RSI gate: Required only when both pendings are placed and distance >= minimum threshold.
- Friday filter: NoTradeFridays blocks setup on Friday.

## Parity Evidence

- Setup day match: 100.0%
- Setup field match: 99.94%
- ATR parity: 100.0%
- Fill count match: 99.5249%
- Remaining setup divergence days: 8

## Frozen Scope

- daily setup day presence
- swing high/low and pending levels
- EMA/ATR setup filters
- RSI gate required flag at setup
- buy/sell order placement intent after canonical 16:30 deduplication

## Explicitly Not Frozen

- tick-level trailing modifications
- MFE/MAE exact path
- PnL equivalence in the Python replica
- broker order-send failures not inferable from OHLC bars

## Limitations

- MT5 logs repeated opportunities inside 16:30; the benchmark uses a canonical daily setup row.
- Economic results are empirical MT5 log evidence, not yet a Python execution benchmark.
- Trailing and exact exit path need tick replay or explicit approximation rules before Risk Engine/PPO.
