# Azir Setup Base Research V1

This sprint studies small setup-level changes around Azir without changing the
MQL5 EA, the frozen protected benchmark, the Risk Engine, or any RL component.

## Scope

- Benchmark anchor: `baseline_azir_protected_economic_v1`.
- Research engine: Python Azir replica using M5 OHLCV.
- Purpose: screen setup hypotheses before any MT5/tick-level validation.
- Non-goal: freeze a new economic benchmark from counterfactual Python-only
  results.

## Families

- `swing_definition`: rolling lookback changes and a simple confirmed-pivot
  swing definition.
- `buy_sell_asymmetry`: buy-only/sell-only diagnostics plus side-specific swing
  and offset variants.
- `range_quality`: filters based on swing range relative to ATR and simple
  prior compression.
- `entry_offset`: fixed and ATR-relative entry-offset changes.

## Run

```powershell
$env:PYTHONPATH='src'
python -m hybrid_quant.azir.setup_research `
  --m5-input-path "C:\Users\joseq\Documents\xauusd_m5.csv" `
  --protected-report-path "artifacts\azir-protected-economic-v1-freeze\forced_close_revaluation_report.json" `
  --config-path "configs\experiments\azir_setup_base_research_v1.yaml" `
  --output-dir "artifacts\azir-setup-base-research-v1" `
  --symbol XAUUSD-STD
```

## Artifacts

- `setup_research_summary.md`
- `setup_research_report.json`
- `candidate_variants.csv`
- `swing_definition_comparison.csv`
- `buy_sell_asymmetry_comparison.csv`
- `range_quality_filter_comparison.csv`
- `offset_comparison.csv`
- `yearly_variant_summary.csv`
- `side_variant_summary.csv`
- `setup_variant_events.csv`

## Interpretation

The frozen benchmark remains `baseline_azir_protected_economic_v1`. A setup
variant can only become a serious candidate if it improves the Python proxy and
then survives MT5/tick-level validation. Positive Python-only results are a
research lead, not operational truth.
