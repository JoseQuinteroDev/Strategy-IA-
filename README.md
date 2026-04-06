# Hybrid Quant Framework

Framework cuantitativo modular en Python para research intradía. El proyecto mantiene la infraestructura de datos, features, backtest, risk, diagnostics y RL, pero la baseline principal de Nasdaq pasa ahora a ser una estrategia clara de `Opening Range Breakout (ORB)` sobre `NQ`.

## Estado actual

Implementado:

- ingesta OHLCV, limpieza temporal, validación y exportación a `CSV / Parquet`
- features deterministas multi-timeframe
- baseline reproducible con `BaselineRunner`
- `PropFirmRiskEngine`
- backtest con costes, slippage, `SL / TP`, `time stop` y cierre forzoso de sesión
- diagnostics detallados y artifacts exportables
- entorno Gymnasium y PPO ya cableados, pero no son la prioridad actual

Prioridad actual:

- validar una baseline Nasdaq seria antes de volver a RL
- usar `baseline_nq_orb` como baseline principal de Nasdaq

## Baseline principal de Nasdaq

La baseline principal de Nasdaq es:

- `baseline_nq_orb`

Familia:

- `opening_range_breakout`

Lógica base:

- construye un `opening range` configurable de `15m` o `30m`
- calcula `opening_range_high`, `opening_range_low`, `opening_range_width` y `opening_range_width_atr`
- entra solo en ruptura del opening range
- filtra largos con `close > EMA200 1H` y `EMA200 1H slope > 0`
- filtra cortos con `close < EMA200 1H` y `EMA200 1H slope < 0`
- aplica filtros mínimos de calidad:
  - expansión mínima de la vela
  - momentum mínimo
  - ancho mínimo y máximo del opening range relativo a ATR
  - `relative_volume` mínimo
  - filtro anti-chase por `breakout_distance_atr`
- soporta dos modos de entrada:
  - `breakout_close_entry`
  - `breakout_retest_entry`

La configuración vive en:

- [configs/variants/baseline_nq_orb.yaml](configs/variants/baseline_nq_orb.yaml)

## Qué queda como legacy

La familia anterior `trend_breakout` no desaparece, pero deja de ser la baseline principal de Nasdaq.

Se conserva solo por compatibilidad y comparación histórica:

- [configs/variants/baseline_trend_nasdaq.yaml](configs/variants/baseline_trend_nasdaq.yaml)
- [src/hybrid_quant/strategy/trend_breakout.py](src/hybrid_quant/strategy/trend_breakout.py)

Las variantes experimentales de esa familia deben tratarse como `legacy / deprecated`, no como flujo principal.

## Arquitectura

```text
hybrid-quant-framework/
|-- configs/
|-- docs/
|-- scripts/
|-- src/
|   `-- hybrid_quant/
|       |-- backtest/
|       |-- baseline/
|       |-- core/
|       |-- data/
|       |-- env/
|       |-- execution/
|       |-- features/
|       |-- paper/
|       |-- risk/
|       |-- rl/
|       |-- strategy/
|       `-- validation/
`-- tests/
```

Módulos que tocan la baseline ORB:

- `features`: cálculo de opening range, slope y `relative_volume`
- `strategy`: nueva familia `opening_range_breakout`
- `baseline`: runner y diagnostics
- `configs/variants`: baseline principal Nasdaq y legacy

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Importar un dataset externo de NQ/MNQ

Formatos aceptados:

- `CSV`
- `Parquet`

Esquema interno objetivo:

- `open_time`
- `open`
- `high`
- `low`
- `close`
- `volume`

Ejemplo:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.data import `
  --config-dir configs `
  --input-path "C:\ruta\a\tu\export_externo_nq_5m.csv" `
  --output-path "data/raw/NQ/5m/ohlcv-normalized.csv" `
  --interval 5m `
  --allow-gaps
```

## Ejecutar la baseline ORB de Nasdaq

Backtest directo sobre un dataset local normalizado:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline `
  --config-dir configs `
  --variant baseline_nq_orb `
  --input-path "data/raw/NQ/5m/ohlcv-normalized.csv" `
  --output-dir "artifacts/baseline-nq-orb" `
  --allow-gaps
```

Artefactos generados:

- `ohlcv.csv`
- `features.csv`
- `signals.csv`
- `trades.csv`
- `risk_decisions.csv`
- `risk.log`
- `report.json`
- `summary.md`

También puedes limitar el rango temporal desde CLI sin tocar código, usando un dataset descargado o la infraestructura de datos ya existente:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline `
  --config-dir configs `
  --variant baseline_nq_orb `
  --start 2024-01-01T00:00:00+00:00 `
  --end 2024-12-31T23:55:00+00:00 `
  --output-dir "artifacts/baseline-nq-orb-2024"
```

## Ejecutar diagnóstico de la baseline ORB

Flujo de un solo paso:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline.analyze `
  --config-dir configs `
  --variant baseline_nq_orb `
  --input-path "data/raw/NQ/5m/ohlcv-normalized.csv" `
  --output-dir "artifacts/baseline-nq-orb-analysis" `
  --allow-gaps
```

Diagnostics ahora incluye, además de lo anterior:

- resultados por año
- resultados por hora
- resultados por día de la semana
- `profit factor`
- primera ruptura del día vs siguientes
- `MFE / MAE`
- `breakout_distance_atr`
- distribución por ancho del opening range

## Ejecutar ablaciones ORB

La matriz reproducible de ablaciones vive en:

- `configs/experiments/orb_ablation.yaml`

La matriz por defecto compara:

- opening range `15m` vs `30m`
- `breakout_close_entry` vs `breakout_retest_entry`
- solo primera ruptura del día vs múltiples rupturas
- con slope de `EMA200 1H` vs sin slope
- con filtro de `RVOL` vs sin filtro de `RVOL`

Runner reproducible:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline.orb_ablation `
  --config-dir configs `
  --experiment-config "configs/experiments/orb_ablation.yaml" `
  --input-path "data/raw/NQ/5m/ohlcv-normalized.csv" `
  --output-dir "artifacts/orb-ablation" `
  --allow-gaps
```

También puedes recortar el rango temporal desde CLI:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline.orb_ablation `
  --config-dir configs `
  --experiment-config "configs/experiments/orb_ablation.yaml" `
  --input-path "data/raw/NQ/5m/ohlcv-normalized.csv" `
  --start 2024-01-01T00:00:00+00:00 `
  --end 2024-12-31T23:55:00+00:00 `
  --output-dir "artifacts/orb-ablation-2024" `
  --allow-gaps
```

Artifacts principales de la ablación:

- `orb_ablation_comparison.json`
- `orb_ablation_results.csv`
- `orb_ablation_summary.md`
- `opening_range_summary.csv`
- `entry_mode_summary.csv`
- `breakout_budget_summary.csv`
- `ema_slope_summary.csv`
- `relative_volume_summary.csv`

Cada variante también deja su propia carpeta:

- `variants/<variant>/baseline`
- `variants/<variant>/diagnostics`

## Ejecutar validación focalizada de la subfamilia ORB ganadora

La sensibilidad local alrededor de la subfamilia ganadora vive en:

- `configs/experiments/orb_focus_validation.yaml`

Esta fase no hace un grid search agresivo. Solo compara ajustes pequeños y trazables alrededor de:

- `orb30_close_multi_no_slope_no_rvol`

Runner reproducible:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline.orb_focus_validation `
  --config-dir configs `
  --experiment-config "configs/experiments/orb_focus_validation.yaml" `
  --input-path "data/raw/NQ/5m/ohlcv-normalized.csv" `
  --output-dir "artifacts/orb-focus-validation-full" `
  --allow-gaps
```

Artifacts principales:

- `orb_focus_validation_comparison.json`
- `orb_focus_validation_results.csv`
- `orb_focus_validation_summary.md`
- `temporal_block_results.csv`
- `yearly_variant_summary.csv`
- `quarterly_variant_summary.csv`
- `yearly_equity_curve_summary.csv`

Cada variante también deja su propia carpeta:

- `variants/<variant>/baseline`
- `variants/<variant>/diagnostics`

## Ejecutar expansión controlada de frecuencia alrededor de width_wider

La sensibilidad local orientada a frecuencia vive en:

- `configs/experiments/orb_frequency_expansion.yaml`

La referencia central es:

- `orb30_close_multi_no_slope_no_rvol_width_wider`

Esta fase no hace un grid search agresivo. Solo abre localmente la frecuencia alrededor de `width_wider` y exige guard rails explícitos sobre:

- `expectancy`
- `profit_factor`
- `max_drawdown`
- estabilidad temporal básica

Runner reproducible:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline.orb_frequency_expansion `
  --config-dir configs `
  --experiment-config "configs/experiments/orb_frequency_expansion.yaml" `
  --input-path "data/raw/NQ/5m/ohlcv-normalized.csv" `
  --output-dir "artifacts/orb-frequency-expansion-full" `
  --allow-gaps
```

Artifacts principales:

- `orb_frequency_expansion_comparison.json`
- `orb_frequency_expansion_results.csv`
- `orb_frequency_expansion_summary.md`
- `activity_summary.csv`
- `temporal_block_results.csv`
- `yearly_variant_summary.csv`
- `quarterly_variant_summary.csv`
- `yearly_equity_curve_summary.csv`

Cada variante también deja su propia carpeta:

- `variants/<variant>/baseline`
- `variants/<variant>/diagnostics`

## Ejecutar frequency push alrededor de las candidatas ORB

La fase de empuje de frecuencia vive en:

- `configs/experiments/orb_frequency_push.yaml`

Toma como base:

- `extension_laxer`
- `width_laxer_extension_laxer`
- `width_wider` como control conservador

No hace un grid search masivo. Solo intenta comprobar si esta subfamilia puede acercarse a una zona más práctica de actividad, con guard rails explícitos sobre:

- `profit_factor`
- `expectancy`
- `max_drawdown`
- estabilidad temporal por año y trimestre

Runner reproducible:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline.orb_frequency_push `
  --config-dir configs `
  --experiment-config "configs/experiments/orb_frequency_push.yaml" `
  --input-path "data/raw/NQ/5m/ohlcv-normalized.csv" `
  --output-dir "artifacts/orb-frequency-push-full" `
  --allow-gaps
```

Artifacts principales:

- `orb_frequency_push_comparison.json`
- `orb_frequency_push_results.csv`
- `orb_frequency_push_summary.md`
- `activity_summary.csv`
- `yearly_variant_summary.csv`
- `quarterly_variant_summary.csv`
- `candidate_ranking.csv`

## Ejecutar la nueva familia ORB intradía activa

La nueva dirección de research vive en:

- `configs/variants/baseline_nq_intraday_orb_active.yaml`
- `configs/experiments/orb_intraday_active_research.yaml`

Esta familia no reutiliza la ORB clásica como lógica principal. Sigue anclada al opening range, pero busca más oportunidades intradía con tres setups explícitos:

- `breakout_continuation`
- `first_pullback_after_breakout`
- `reclaim_acceptance`

La referencia histórica anterior sigue disponible como control:

- `baseline_nq_orb`

Runner reproducible:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline.orb_intraday_active_research `
  --config-dir configs `
  --experiment-config "configs/experiments/orb_intraday_active_research.yaml" `
  --input-path "data/raw/NQ/5m/ohlcv-normalized.csv" `
  --output-dir "artifacts/orb-intraday-active-full" `
  --allow-gaps
```

Artifacts principales:

- `orb_intraday_active_comparison.json`
- `orb_intraday_active_results.csv`
- `orb_intraday_active_summary.md`
- `activity_summary.csv`
- `yearly_variant_summary.csv`
- `quarterly_variant_summary.csv`
- `candidate_ranking.csv`

Cada variante deja además:

- `variants/<variant>/baseline`
- `variants/<variant>/diagnostics`

## Tests

Suite completa:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Tests ORB más relevantes:

```powershell
$env:PYTHONPATH = "src"
python -m unittest `
  tests.unit.test_features_deterministic `
  tests.unit.test_strategy_opening_range_breakout `
  tests.unit.test_strategy_orb_intraday_active `
  tests.integration.test_baseline_nq_orb_flow `
  tests.integration.test_orb_intraday_active_research_flow -v
```

## Nueva familia contextual intradia

La nueva direccion de research intradia para Nasdaq vive en:

- `configs/variants/baseline_nq_intraday_contextual.yaml`
- `configs/experiments/intraday_nasdaq_contextual_research.yaml`

Esta familia usa el opening range como contexto, pero ya no depende solo de una ruptura rigida. Compara tres setups:

- `context_pullback_continuation`
- `vwap_reclaim_acceptance`
- `session_trend_continuation`

Runner reproducible:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline.intraday_nasdaq_contextual_research `
  --config-dir configs `
  --experiment-config "configs/experiments/intraday_nasdaq_contextual_research.yaml" `
  --input-path "data/raw/NQ/5m/ohlcv-normalized.csv" `
  --output-dir "artifacts/intraday-nasdaq-contextual-full" `
  --allow-gaps
```

Artifacts principales:

- `intraday_nasdaq_contextual_comparison.json`
- `intraday_nasdaq_contextual_results.csv`
- `intraday_nasdaq_contextual_summary.md`
- `activity_summary.csv`
- `yearly_variant_summary.csv`
- `quarterly_variant_summary.csv`
- `candidate_ranking.csv`

Tests mas relevantes para esta fase:

```powershell
$env:PYTHONPATH = "src"
python -m unittest `
  tests.unit.test_strategy_intraday_nasdaq_contextual `
  tests.integration.test_intraday_nasdaq_contextual_research_flow -v
```

## Ejecutar el zoom sobre session_trend_30m

La fase focalizada sobre `session_trend_30m` vive en:

- `configs/variants/session_trend_30m.yaml`
- `configs/experiments/session_trend_30m_zoom.yaml`
- `src/hybrid_quant/baseline/session_trend_30m_zoom.py`

Esta fase no cambia la naturaleza de `session_trend_30m`. Solo compara combinaciones controladas de:

- filtro HTF (`EMA200 1H` y slope)
- contexto intradía (`VWAP`, `EMA20/EMA50`, `OR midpoint`)
- horas operativas
- filtro de extensión / momentum
- confirmaciones secundarias de continuación
- cap diario de oportunidades
- sesgo direccional

Runner reproducible:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline.session_trend_30m_zoom `
  --config-dir configs `
  --experiment-config "configs/experiments/session_trend_30m_zoom.yaml" `
  --input-path "data/raw/NQ/5m/ohlcv-normalized.csv" `
  --output-dir "artifacts/session-trend-30m-zoom-full" `
  --allow-gaps
```

Artifacts principales:

- `session_trend_30m_zoom_comparison.json`
- `session_trend_30m_zoom_results.csv`
- `session_trend_30m_zoom_summary.md`
- `candidate_ranking.csv`
- `hourly_variant_summary.csv`
- `side_variant_summary.csv`
- `temporal_variant_summary.csv`
- `filter_ablation_summary.csv`

Convención importante:

- `max_drawdown` se guarda como fracción de equity. Por ejemplo, `0.0272` significa `2.72%`.

## Ejecutar la validacion extendida de shorts_strict_clean_hours

La fase extendida alrededor de `shorts_strict_clean_hours` vive en:

- `configs/variants/shorts_strict_clean_hours.yaml`
- `configs/variants/long_only_clean_hours.yaml`
- `configs/experiments/shorts_strict_clean_hours_extended.yaml`
- `src/hybrid_quant/baseline/shorts_strict_clean_hours_extended.py`

La prioridad de lectura aqui es:

- rentabilidad
- drawdown
- frecuencia

Convencion importante:

- `max_drawdown` se guarda como fraccion de equity. Ejemplo: `0.02448 = 2.448%`.

Primero importa el CSV real nuevo a una ruta reproducible del repo:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.data import `
  --config-dir configs `
  --input-path "C:\Users\joseq\Documents\nuevos_datos.csv" `
  --output-path "data/raw/NQ/5m/ohlcv-normalized-extended.csv" `
  --interval 5m `
  --allow-gaps
```

Runner reproducible:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline.shorts_strict_clean_hours_extended `
  --config-dir configs `
  --experiment-config "configs/experiments/shorts_strict_clean_hours_extended.yaml" `
  --input-path "data/raw/NQ/5m/ohlcv-normalized-extended.csv" `
  --output-dir "artifacts/shorts-strict-clean-hours-extended" `
  --allow-gaps
```

Artifacts principales:

- `shorts_strict_clean_hours_extended_comparison.json`
- `shorts_strict_clean_hours_extended_results.csv`
- `shorts_strict_clean_hours_extended_summary.md`
- `candidate_ranking.csv`
- `comparison_summary.csv`
- `yearly_variant_summary.csv`
- `quarterly_variant_summary.csv`
- `hourly_variant_summary.csv`
- `side_variant_summary.csv`
- `dataset_coverage.json`

## RL

El entorno Gymnasium y PPO siguen en el repo, pero no se han tocado en este sprint. La secuencia recomendada ahora es:

1. importar dataset real de `NQ 5m`
2. correr `baseline_nq_orb`
3. diagnosticarla
4. solo después decidir si merece validación robusta y, más adelante, PPO

## TODO razonable

- añadir modo de stop estructural sobre el nivel roto además del stop en ATR
- enriquecer más el diagnóstico ORB con breakdowns específicos de retest vs close entry
- decidir más adelante si la familia `trend_breakout` legacy se elimina del todo o se conserva únicamente para comparativa histórica
