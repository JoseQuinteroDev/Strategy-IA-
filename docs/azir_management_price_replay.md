# management_price_replay_and_heuristic_backtest_v1

## Purpose

This sprint prices Azir management actions before training any management PPO.
The signal remains Azir `take_all_valid` protected by `risk_engine_azir_v1`.
The replay layer only asks: if a protected trade already exists, would a simple
management rule have improved the outcome?

## Data Sources

- Base economic benchmark: `baseline_azir_protected_economic_v1`.
- Event source: canonical Azir MT5 event log.
- Price source for counterfactual management: XAUUSD M1 OHLCV where available.

## Actions Priced

- `base_management`: uses observed/revalued protected benchmark net PnL.
- `close_early`: exits at the close of the first configured closed M1 bar after fill.
- `move_to_break_even`: activates a BE stop after favorable M1 movement reaches the configured threshold.
- `trailing_conservative`: M1-based trailing with a wider activation/step.
- `trailing_aggressive`: M1-based trailing with a faster activation/step.

## Conservative Replay Rules

- The replay uses only closed M1 bars.
- The first partial fill minute is not used for stop/trailing path decisions.
- If one M1 candle contains both stop and target, the stop is selected first.
- If M1 coverage is unavailable, the action is marked unpriced and excluded from
  same-coverage heuristic deltas.
- Alternative actions are not a frozen benchmark until tick/broker execution can
  validate the counterfactual fills.

## Heuristic Comparison

The runner compares each heuristic against `base_management` over the same priced
subset. This avoids declaring victory just because an action has less coverage.

Default heuristics:

- `always_base_management`
- `always_close_early`
- `move_to_be_after_mfe_threshold`
- `conservative_trailing_after_mfe_threshold`
- `aggressive_trailing_after_mfe_threshold`
- `sell_only_conservative_trailing_after_mfe_threshold`
- `buy_only_conservative_trailing_after_mfe_threshold`

## Command

```powershell
$env:PYTHONPATH='src'
python -m hybrid_quant.azir.management_replay `
  --mt5-log-path "C:\Users\joseq\Documents\Playground\todos_los_ticks.csv" `
  --protected-report-path "C:\Users\joseq\Documents\Playground\hybrid-quant-framework\artifacts\azir-protected-economic-v1-freeze\forced_close_revaluation_report.json" `
  --m1-input-path "C:\Users\joseq\Documents\xauusd_m1.csv" `
  --config-path "configs\experiments\azir_management_price_replay_v1.yaml" `
  --output-dir "artifacts\azir-management-price-replay-v1" `
  --symbol XAUUSD-STD
```

## Artifacts

- `management_replay_summary.md`
- `management_replay_report.json`
- `management_heuristics_comparison.csv`
- `management_price_replay_cases.csv`
- `management_exit_distribution.csv`
- `management_limitations.md`

## Decision Gate

This sprint can justify heuristic screening. It should not train management PPO
unless action pricing coverage and counterfactual execution fidelity are deemed
sufficient.
