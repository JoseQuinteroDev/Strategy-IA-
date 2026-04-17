# Azir Best Setup Candidate: swing_10_fractal

This document defines the controlled candidate produced by
`setup_base_research_for_azir_v1`. It does not replace the official Azir MQL5
baseline and does not freeze a new benchmark.

## Candidate Name

`baseline_azir_setup_candidate_fractal_v1`

## Exact Definition

- Setup time, filters, offset, SL, TP, trailing, Friday behavior and Risk Engine
  stay unchanged.
- The only changed setup component is swing high / swing low selection.
- At the 16:30 server-time setup, inspect the last 10 fully closed M5 bars.
- A pivot high is confirmed when a bar high is greater than the highs of the 2
  bars to its left and the 2 bars to its right, all inside already closed data.
- A pivot low is confirmed when a bar low is lower than the lows of the 2 bars
  to its left and the 2 bars to its right.
- `swing_high` is the most recent confirmed pivot high inside the 10-bar
  window. If no pivot high exists, fallback to the current rolling max.
- `swing_low` is the most recent confirmed pivot low inside the 10-bar window.
  If no pivot low exists, fallback to the current rolling min.
- Entry levels remain Azir-style:
  `buy_entry = swing_high + 5 * Point`
  `sell_entry = swing_low - 5 * Point`

## Validation Status

The candidate has positive Python-replica proxy evidence, but it does not yet
have its own MT5 event log. It may be treated as a formal research candidate
only after MT5/export validation, not as a frozen benchmark.

## Run

```powershell
$env:PYTHONPATH='src'
python -m hybrid_quant.azir.best_setup_candidate `
  --mt5-log-path "C:\Users\joseq\Documents\Playground\todos_los_ticks.csv" `
  --m5-input-path "C:\Users\joseq\Documents\xauusd_m5.csv" `
  --protected-report-path "artifacts\azir-protected-economic-v1-freeze\forced_close_revaluation_report.json" `
  --setup-research-report-path "artifacts\azir-setup-base-research-v1\setup_research_report.json" `
  --output-dir "artifacts\azir-best-setup-candidate-fractal-v1" `
  --symbol XAUUSD-STD
```
