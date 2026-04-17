# management_price_replay_v2_with_tick_or_broker_execution

## Purpose

This sprint upgrades Azir management replay from M1-only evidence to a
tick-first valuation. It still does not train PPO and does not change Azir or
`risk_engine_azir_v1`.

## Pricing Priority

1. `base_management`: frozen `baseline_azir_protected_economic_v1`.
2. Alternative actions: real tick replay when the tick CSV covers the protected
   trade lifecycle.
3. M1 fallback when tick evidence cannot price the action.
4. `unpriced` when neither source is sufficient.

## Tick Execution Convention

- Long exits use bid.
- Short exits use ask.
- `close_early` exits at the first tick at or after fill + 60 seconds.
- `move_to_break_even` activates only after chronological tick movement reaches
  the configured favorable threshold.
- Conservative/aggressive trailing ratchets from chronological tick prices after
  activation.
- If the rule never activates, the protected base result is retained.
- Broker queue priority, latency, and partial fills are not simulated.

## Command

```powershell
$env:PYTHONPATH='src'
python -m hybrid_quant.azir.management_replay_v2 `
  --mt5-log-path "C:\Users\joseq\Documents\Playground\todos_los_ticks.csv" `
  --protected-report-path "C:\Users\joseq\Documents\Playground\hybrid-quant-framework\artifacts\azir-protected-economic-v1-freeze\forced_close_revaluation_report.json" `
  --m1-input-path "C:\Users\joseq\Documents\xauusd_m1.csv" `
  --tick-input-path "C:\Users\joseq\Documents\tick_level.csv" `
  --config-path "configs\experiments\azir_management_price_replay_v2.yaml" `
  --output-dir "artifacts\azir-management-price-replay-v2" `
  --symbol XAUUSD-STD
```

## Artifacts

- `management_replay_v2_summary.md`
- `management_replay_v2_report.json`
- `management_heuristics_comparison_v2.csv`
- `management_tick_coverage_report.csv`
- `management_tick_replay_cases.csv`
- `management_exit_distribution_v2.csv`
- `management_limitations_v2.md`
- `heuristics_vs_base_same_coverage_v2.csv`

## Decision Gate

Management PPO should only start if a tick-priced or mostly tick-priced
heuristic shows material improvement over `base_management` on same-coverage
comparison. A small uplift on a partial subset is not enough.
