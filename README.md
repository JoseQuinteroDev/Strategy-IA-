# Hybrid Quant Framework

Framework cuantitativo modular en Python para `BTCUSDT` intradia, con baseline rule-based reproducible, Risk Engine prop-firm friendly, entorno RL compatible con Gymnasium y primer pipeline PPO ya conectado.

## Estado real del repo

Ya implementado:

- ingesta OHLCV historica desde Binance REST
- limpieza temporal, deduplicacion, validacion de indices y exportacion a Parquet
- split temporal `train / validation / test`
- features deterministas multi-timeframe
- estrategia intradia de mean reversion con `EMA200 1H` y `ADX 1H`
- nueva familia de estrategia intradia `trend-following / breakout` para `NQ`
- backtest con costes, slippage, `SL / TP`, `time stop` y cierre de sesion
- `PropFirmRiskEngine` integrado en el baseline runner
- capa compartida de ejecucion con `PortfolioSimulator`
- entorno RL `HybridTradingEnvironment`
- integracion PPO con `stable-baselines3`
- diagnostico detallado del baseline y comparacion entre variantes
- tests unitarios e integracion sobre data, baseline, risk, env y RL

Pendiente:

- validacion robusta out-of-sample
- entrenamiento PPO largo y evaluacion estadistica seria
- SAC / TD3
- LSTM / Transformer
- multiactivo
- live trading / MQL5

## Universo actual

- activo: `BTCUSDT`
- timeframe operativo: `5m`
- timeframe filtro: `1H`
- estilo: `intradia`
- baseline rule-based: mean reversion con filtro de tendencia y regimen
- RL actual: decision discreta sobre candidate trades generados por la estrategia

Tambien existe ya una familia separada `trend_breakout` orientada a Nasdaq (`NQ`) usando el mismo backtester, Risk Engine y pipeline de features. De momento esta variante esta pensada para ejecutarse con OHLCV local via `--input-path`, porque la infraestructura de descarga automatica sigue siendo crypto-centric.

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

Modulos clave:

- `data`: descarga, limpieza, validacion temporal, parquet y splits.
- `features`: calculo determinista de indicadores y contexto de mercado.
- `strategy`: candidate trades del baseline rule-based.
- `execution`: simulacion compartida de portfolio y semantica de ejecucion.
- `risk`: guardrails prop-firm friendly.
- `backtest`: simulacion determinista del baseline.
- `baseline`: pipeline reproducible, diagnostico y comparacion entre variantes.
- `env`: entorno RL compatible con Gymnasium.
- `rl`: dataset builder, trainer PPO, evaluacion y runner reproducible.

## Baselines disponibles

### `baseline_v1`

Es la referencia historica del proyecto y la que se usa para mantener continuidad con el diagnostico existente. Para comparaciones apples-to-apples se ejecuta en modo directo `strategy + backtest`, sin el filtrado adicional del `BaselineRunner`.

### `baseline_v2`

Es una variante mas selectiva y cost-aware definida en [configs/variants/baseline_v2.yaml](configs/variants/baseline_v2.yaml). Introduce cambios pequenos, interpretables y defendibles:

- bloqueo de horas toxicas detectadas en el diagnostico
- exclusion de fines de semana
- cooldown mas estricto
- `ADX` mas selectivo para el filtro de regimen
- `TP` mas amplio para mejorar estructura economica
- filtro simple de `target_to_cost_ratio` para evitar setups que no compensan fees + slippage

### `baseline_v3`

Es un refinement quirurgico sobre `baseline_v2`. Mantiene la misma logica base y el mismo quality gate cost-aware, pero excluye tambien una segunda franja de horas donde `baseline_v2` seguia destruyendose en neto. La intencion es muy simple:

- seguir con frecuencia baja
- reducir aun mas fees
- evitar ventanas de baja calidad despues del filtro cost-aware
- intentar cruzar break-even neto sin aumentar drawdown

### `baseline_trend_nasdaq`

Es una familia nueva, separada de mean reversion. Usa un enfoque interpretable de `trend-following / breakout` intradia sobre `NQ`:

- filtro de tendencia con `EMA200 1H`
- filtro de regimen con `ADX 1H`
- ruptura del rango previo con confirmacion de momentum
- expansion minima de volatilidad en la vela de ruptura
- quality gate cost-aware para evitar breakouts demasiado pequenos
- `SL / TP / time stop / close on session end` reutilizando la misma semantica de ejecucion del framework

La configuracion vive en [configs/variants/baseline_trend_nasdaq.yaml](configs/variants/baseline_trend_nasdaq.yaml).

## Entorno RL

El entorno actual es `HybridTradingEnvironment`. La semantica completa esta documentada en [docs/rl_environment.md](docs/rl_environment.md).

Resumen corto:

- observacion: features de la barra actual + estado del portfolio + estado de riesgo + candidate trade actual
- acciones:
  - `0 = skip`
  - `1 = take_trade`
  - `2 = close_early`
- la estrategia sigue generando candidate trades
- el agente no inventa setups todavia
- `PortfolioSimulator` resuelve ejecucion y cierres
- `PropFirmRiskEngine` aprueba o bloquea entradas

Nota de configuracion:

- `state_context_bars` es el nombre canonico
- `observation_window` queda como alias compatible
- hoy la observacion sigue siendo single-bar, no una ventana temporal apilada real

## PPO actual

La integracion PPO vive en:

- [src/hybrid_quant/rl/trainer.py](src/hybrid_quant/rl/trainer.py)
- [src/hybrid_quant/rl/runner.py](src/hybrid_quant/rl/runner.py)
- [src/hybrid_quant/rl/evaluation.py](src/hybrid_quant/rl/evaluation.py)
- [src/hybrid_quant/rl/dataset.py](src/hybrid_quant/rl/dataset.py)

Flujo:

1. preparar OHLCV y features
2. construir splits temporales `train / validation / test`
3. entrenar PPO por multiples seeds
4. guardar checkpoints y mejor modelo
5. evaluar `baseline_without_rl`, `random_policy` y `ppo_trained`

## Instalacion

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Tests

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Ingesta de datos

Descarga OHLCV:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.data download `
  --config-dir configs `
  --start 2024-01-01T00:00:00+00:00 `
  --end 2024-02-01T00:00:00+00:00
```

Split de un dataset existente:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.data split `
  --config-dir configs `
  --input-path data/raw/BTCUSDT/5m/ohlcv.parquet
```

## Baseline runner con Risk Engine

Este pipeline genera:

- `ohlcv.csv`
- `features.csv`
- `signals.csv`
- `trades.csv`
- `risk_decisions.csv`
- `risk.log`
- `report.json`
- `summary.md`

Ejecucion:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline `
  --config-dir configs `
  --start 2024-01-01T00:00:00+00:00 `
  --end 2024-03-31T23:55:00+00:00 `
  --output-dir artifacts/baseline-q1-2024
```

Ejemplo para la familia nueva `baseline_trend_nasdaq` usando un CSV/Parquet local de `NQ 5m`:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline `
  --config-dir configs `
  --variant baseline_trend_nasdaq `
  --input-path data/raw/NQ/5m/ohlcv.csv `
  --output-dir artifacts/baseline-trend-nasdaq `
  --allow-gaps
```

## Diagnostico del baseline

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline.diagnostics `
  --config-dir configs `
  --artifact-dir artifacts/baseline-q1-2024 `
  --output-dir artifacts/baseline-q1-2024-diagnostics
```

## Comparacion baseline_v1 vs baseline_v2 vs baseline_v3

La comparacion usa el mismo `ohlcv.csv` para las variantes seleccionadas y genera artefactos raiz como:

- `baseline_comparison.json`
- `baseline_comparison_summary.md`
- `baseline_v1_report.json`
- `baseline_v2_report.json`
- `baseline_v3_report.json`
- `baseline_v1_v2_v3_comparison.json`
- `baseline_v1_v2_v3_summary.md`
- `baseline_v2_monthly_breakdown.csv`
- `baseline_v2_hourly_breakdown.csv`
- `baseline_v2_exit_reason_breakdown.csv`
- `baseline_v2_side_breakdown.csv`
- `baseline_v3_monthly_breakdown.csv`
- `baseline_v3_hourly_breakdown.csv`

Ejemplo reproducible sobre el baseline historico Q1:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline.comparison `
  --config-dir configs `
  --input-path artifacts/baseline-q1-2024/ohlcv.csv `
  --output-dir artifacts/baseline-v1-v2-v3-q1-2024 `
  --oos-start 2024-03-01T00:00:00+00:00 `
  --oos-end 2024-03-31T23:55:00+00:00
```

Tambien puedes usar el wrapper:

```powershell
$env:PYTHONPATH = "src"
python scripts/compare_baselines.py `
  --config-dir configs `
  --input-path artifacts/baseline-q1-2024/ohlcv.csv `
  --output-dir artifacts/baseline-v1-v2-v3-q1-2024
```

Comparacion generica de variantes, incluyendo la nueva familia `trend_breakout`:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline.comparison `
  --config-dir configs `
  --input-path data/raw/NQ/5m/ohlcv.csv `
  --output-dir artifacts/comparison-trend-nasdaq `
  --variant baseline_trend_nasdaq `
  --variant baseline_v3 `
  --allow-gaps
```

## Entrenamiento PPO

Con entrypoint del proyecto:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.rl.runner `
  --config-dir configs `
  --input-path data/raw/BTCUSDT/5m/ohlcv.parquet `
  --output-dir artifacts/ppo-baseline
```

O con script:

```powershell
$env:PYTHONPATH = "src"
python scripts/train_ppo.py `
  --config-dir configs `
  --input-path data/raw/BTCUSDT/5m/ohlcv.parquet `
  --output-dir artifacts/ppo-baseline
```

## Configuracion relevante

- [configs/backtest.yaml](configs/backtest.yaml): costes, slippage y politica intrabar.
- [configs/risk.yaml](configs/risk.yaml): limites prop-firm y sesion.
- [configs/strategy.yaml](configs/strategy.yaml): baseline historica `baseline_v1`.
- [configs/variants/baseline_v2.yaml](configs/variants/baseline_v2.yaml): override selectivo y cost-aware.
- [configs/variants/baseline_v3.yaml](configs/variants/baseline_v3.yaml): refinement quirurgico de ventanas horarias sobre `baseline_v2`.
- [configs/variants/baseline_trend_nasdaq.yaml](configs/variants/baseline_trend_nasdaq.yaml): nueva familia `trend_breakout` para Nasdaq con filtros de ruptura y momentum.
- [configs/env.yaml](configs/env.yaml): `max_steps`, `reward_mode`, `state_context_bars`.
- [configs/rl.yaml](configs/rl.yaml): seeds, PPO, checkpoints y splits de entrenamiento.

## Estado practico

El proyecto esta listo para seguir iterando con PPO a nivel de arquitectura, pero la decision correcta sigue siendo endurecer primero la baseline rule-based y su validacion out-of-sample. `baseline_v1` queda como control historico, `baseline_v2` como mejora selectiva cost-aware y `baseline_v3` como refinement quirurgico antes del sprint de validacion robusta.

## Validacion robusta de baseline_v3

La validacion robusta vive en [robustness.py](src/hybrid_quant/validation/robustness.py) y no toca PPO. Su objetivo es responder si `baseline_v3` merece pasar al siguiente nivel de investigacion.

Incluye:

- walk-forward rolling real con ventanas `train / validation / test`
- bloques temporales mensuales para comprobar consistencia fuera del tramo principal
- Monte Carlo por reordenacion de trades con seed fija
- sensibilidad a fees y slippage
- clasificacion final `GO`, `GO WITH CAUTION` o `NO-GO`

Artefactos principales:

- `robustness_report.json`
- `robustness_summary.md`
- `walk_forward_results.csv`
- `temporal_block_results.csv`
- `monte_carlo_summary.json`
- `cost_sensitivity.csv`
- aliases opcionales como `robustness_report_extended.json` si usas `--artifact-suffix`

Ejemplo de ejecucion sobre un dataset local:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.validation `
  --config-dir configs `
  --input-path artifacts/baseline-q1-2024/ohlcv.csv `
  --output-dir artifacts/baseline-v3-robustness-q1-2024 `
  --variant baseline_v3
```

Tambien puede descargar y validar directamente desde la infraestructura de datos ya existente:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.validation `
  --config-dir configs `
  --start 2023-01-01T00:00:00+00:00 `
  --end 2024-12-31T23:55:00+00:00 `
  --output-dir artifacts/baseline-v3-robustness-extended `
  --variant baseline_v3 `
  --allow-gaps `
  --artifact-suffix extended
```

Comparacion entre una validacion Q1 y una validacion extendida:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.validation.comparison `
  --baseline-report artifacts/baseline-v3-robustness-q1-2024/robustness_report.json `
  --extended-report artifacts/baseline-v3-robustness-extended/robustness_report_extended.json `
  --output-dir artifacts/baseline-v3-robustness-comparison `
  --baseline-label q1_2024 `
  --extended-label extended_2023_2024
```

Como interpretar la clasificacion:

- `GO`: la baseline supera de forma suficientemente limpia los checks temporales, de drawdown y de sensibilidad.
- `GO WITH CAUTION`: la baseline es prometedora, pero aun necesita mas evidencia temporal antes de volver a PPO.
- `NO-GO`: la baseline todavia no es lo bastante robusta y no merece pasar al siguiente nivel de investigacion.
