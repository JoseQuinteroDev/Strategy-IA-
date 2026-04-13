# Azir Hybrid Quant Framework - Architecture

This project now treats AzirIA MT5 as the official operational baseline. Older
Python baselines remain useful as legacy research artifacts, but they are not the
main strategy source of truth for the hybrid roadmap.

## Roadmap Order

The architecture follows the Azir roadmap in this order:

1. Specification and logging.
2. Faithful Python replica of Azir.
3. Serious backtester.
4. Audit and realism checks.
5. Risk Engine.
6. Frozen benchmark.
7. RL environment.
8. PPO.
9. Robustness and anti-overfit validation.
10. Trailing/exit research.
11. Sequential models.
12. Paper trading.

Sprint 0/1 covers step 1. Sprint 1 adds the first Python replica and MT5
parity runner, but still does not change Azir trading logic.

## Source of Truth Boundary

Operational truth:

- `C:\Users\joseq\Documents\Playground\Azir.mq5`

Support code added in this sprint:

- `C:\Users\joseq\Documents\Playground\AzirEventLogger.mqh`
- `src/hybrid_quant/azir/event_log.py`
- `src/hybrid_quant/azir/replica.py`
- `src/hybrid_quant/azir/comparison.py`
- `src/hybrid_quant/azir/cli.py`

Rules:

- MQL5 strategy logic is not rewritten in this sprint.
- Python helpers validate exported events; they do not define trading behavior.
- Documentation must describe observed MQL5 behavior, including defects.

## Current Data Flow

```text
MT5 Azir EA
  -> AzirEventLogger.mqh
  -> azir_events_<symbol>_<magic>.csv
  -> Python schema validation
  -> future faithful Python replica
```

Historical OHLCV flow for the next sprint:

```text
xauusd_m5.csv
  -> CSV audit
  -> Python replica input
  -> replica-vs-MT5 event comparison
```

## Azir Replica Runner

Run the Python replica on the audited XAUUSD M5 CSV:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.azir `
  --input-path "C:\Users\joseq\Documents\xauusd_m5.csv" `
  --output-dir "artifacts\azir-replica"
```

Run parity when a real MT5 log exists:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.azir `
  --input-path "C:\Users\joseq\Documents\xauusd_m5.csv" `
  --mt5-log-path "C:\path\to\azir_events_XAUUSD_123456321.csv" `
  --output-dir "artifacts\azir-replica"
```

Expected artifacts:

- `python_events.csv`
- `replica_run_metadata.json`
- `parity_report.json`
- `parity_summary.md`
- `discrepancies.csv`

If no MT5 log is provided, the runner still generates Python events and writes a
blocked parity report instead of inventing a match percentage.

## Event Logging Layer

The logger is append-only CSV. It is intentionally simple so exports can be read
by Python, Excel, or a database loader.

Logged event types:

- `opportunity`
- `blocked_friday`
- `fill`
- `trailing_modified`
- `opposite_pending_cancelled`
- `no_fill_close_cleanup`
- `exit`

The logger records both decision context and outcomes:

- daily swing levels and pending prices,
- EMA20, ATR, RSI, spread,
- filters and direction permissions,
- order placement return codes,
- fills and duration to fill,
- MFE/MAE points while the EA is live,
- trailing activation/modification,
- exit reason and PnL fields from MT5 history.

## Python Schema Layer

`hybrid_quant.azir.event_log` contains:

- canonical column order,
- required field validation,
- CSV writer for deterministic test fixtures.

This is deliberately small. The replica sprint adds a reader/parser and
comparison reports, but empirical parity still requires a real MT5 log.

## CSV Audit Layer

The initial historical file found for Azir is:

- `C:\Users\joseq\Documents\xauusd_m5.csv`

It covers `2021-01-04 01:00:00` through `2025-12-31 23:55:00` with dominant M5
spacing. Timestamps are naive and likely broker/server time. The Python replica
must not silently reinterpret them as UTC.

## Backtester/Risk/RL Boundary

Existing modules under `src/hybrid_quant/backtest`, `risk`, `validation`, and
`rl` are left untouched in Sprint 0/1 and the replica sprint.

The next phases should integrate Azir in layers:

- first reproduce daily opportunities without risk intervention,
- then reproduce order/fill/exit behavior with realistic assumptions,
- then add a prop-firm Risk Engine around the frozen replica,
- only after benchmark stability, expose observations/actions to PPO.

## Replica Fidelity Boundary

The Python replica can reproduce deterministic daily setup logic from M5 bars:

- setup timing in broker/server time,
- Friday blocking,
- swing high/low,
- hardcoded 5-point offset,
- EMA20 trend branch,
- ATR minimum filter,
- pending order intent,
- default RSI gate activation behavior.

Parts that are intentionally approximate until tick/broker logs are available:

- intrabar pending fill order,
- trailing updates from live bid/ask ticks,
- same-bar stop/target sequencing,
- broker commission/swap/slippage,
- M1 RSI gate if only M5 historical data is available.

## Non-Goals in Sprint 0/1 and Replica Sprint

- No PPO.
- No model training.
- No strategy improvement.
- No parameter optimization.
- No multi-asset expansion.
- No correction of suspicious Azir behavior unless a later ticket explicitly
  requests it.

## Acceptance State After Sprint 0/1 and Replica Sprint

The project is ready for parity-driven backtester work when:

- the MQL5 strategy is documented in `docs/spec.md`,
- the EA can export daily opportunities and outcomes,
- the event CSV schema is test-covered,
- the historical CSV coverage is known and caveated,
- known MQL5 discrepancies are preserved rather than hidden,
- the Python replica can emit `python_events.csv`,
- a real MT5 `azir_events_*.csv` is available for empirical parity.
