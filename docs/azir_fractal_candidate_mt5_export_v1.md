# Azir Fractal Candidate MT5 Export V1

This sprint keeps `baseline_azir_setup_frozen_v1` and
`baseline_azir_protected_economic_v1` unchanged. The goal is to move
`baseline_azir_setup_candidate_fractal_v1` closer to MT5 evidence by producing
a candidate event log that can be compared with the observed Azir log.

## Candidate Definition

- Variant name: `swing_10_fractal`.
- Setup time: unchanged Azir setup time, `16:30` server time.
- Bars: last 10 fully closed M5 candles before the setup bar.
- Pivot high: high greater than the 2 closed bars to the left and the 2 closed
  bars to the right inside the closed-bar window.
- Pivot low: low lower than the 2 closed bars to the left and the 2 closed bars
  to the right inside the closed-bar window.
- `swing_high`: most recent confirmed pivot high, with rolling 10-bar high as
  fallback if no pivot exists.
- `swing_low`: most recent confirmed pivot low, with rolling 10-bar low as
  fallback if no pivot exists.
- Entry offset: unchanged Azir hardcoded 5 points.
- Unchanged logic: EMA20 filter, ATR filter, RSI gate, Friday filter, SL/TP,
  trailing and external Risk Engine.

## MT5 Export Path

The auxiliary script is:

`mql5/AzirFractalCandidateEventExport.mq5`

It is not a replacement EA. It writes canonical Azir event-log columns for the
fractal candidate so Python can compare:

- setup calendar,
- buy/sell intent,
- swing levels,
- pending entry levels,
- EMA/ATR/RSI setup fields,
- basic order-placement intent.

Fill, trailing and final PnL remain out of scope unless the candidate is later
run as a true EA or replayed with broker/tick lifecycle evidence.

## Python Comparison Path

The comparison runner is:

`python -m hybrid_quant.azir.fractal_candidate_export`

If a real candidate MT5 log is passed via `--candidate-log-path`, the runner
compares that file against `todos_los_ticks.csv`. If no candidate log is passed,
it writes a Python-equivalent `fractal_candidate_event_log.csv` so the artifact
contract is reproducible, but it marks the evidence as pending MT5 execution.

## Promotion Guardrail

This sprint can formalize the export process. It cannot freeze a new economic
benchmark unless a real MT5 candidate event log exists and then survives a
protected economic audit.
