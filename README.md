# Hybrid Quant Framework

Framework cuantitativo modular en Python orientado a intradia y preparado para evolucionar hacia un stack hibrido rule-based + RL, con foco inicial en un baseline reproducible y prop-firm friendly.

## Estado actual

Lo que ya esta implementado:

- ingesta OHLCV historica desde Binance REST
- limpieza temporal, deduplicacion y exportacion
- features deterministas multi-timeframe
- estrategia base intradia de mean reversion con filtros `EMA200 1H` y `ADX 1H`
- backtest baseline con costes, slippage, `SL/TP`, `time stop` y cierre de sesion
- runner reproducible de baseline con artefactos y reporte
- tests unitarios e integracion para wiring y contratos principales

Lo que aun no esta implementado:

- Risk Engine completo con guardrails diarios y reglas prop-firm mas duras
- paper/live execution real
- entorno RL entrenable
- PPO, SAC, LSTM o cualquier entrenamiento de agentes

## Universo MVP

- Activo inicial: `BTCUSDT`
- Timeframe operativo: `5m`
- Timeframe filtro: `1H`
- Estilo: `intradia`
- Estrategia base: mean reversion con filtro de tendencia y regimen

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
|       |-- features/
|       |-- paper/
|       |-- risk/
|       |-- rl/
|       |-- strategy/
|       `-- validation/
`-- tests/
```

Responsabilidades:

- `data`: descarga, limpieza, validacion temporal, parquet y splits.
- `features`: calculo determinista de senales e indicadores.
- `strategy`: decision rule-based para el baseline intradia.
- `backtest`: simulacion baseline y metricas.
- `baseline`: pipeline reproducible end-to-end.
- `risk`: base de integracion para el siguiente sprint.
- `env` y `rl`: placeholders estructurales, sin entrenamiento todavia.

## Features actuales

`build_features(df)` en [deterministic.py](C:/Users/joseq/Documents/Playground/hybrid-quant-framework/src/hybrid_quant/features/deterministic.py) calcula:

- retornos logaritmicos
- `ATR`
- `EMA200 1H`
- `ADX 1H`
- `VWAP` intradia
- `EMA50` intradia
- volatilidad realizada
- rango de vela
- z-score de distancia a media
- variables temporales de hora, dia y sesion

## Estrategia baseline

La estrategia base usa:

- filtro de tendencia con `EMA200 1H`
- filtro de regimen con `ADX 1H`
- mean reversion contra `VWAP` o `EMA50`
- `SL = 1 ATR`
- `TP = 1 ATR`
- `time stop`
- cierre al final de sesion

Cada senal devuelve:

- direccion
- precio de entrada
- stop
- target
- `time_stop_bars`
- motivo de entrada

## Politica intrabar

Cuando una misma vela toca `stop` y `target`, el backtest usa una politica configurable en `configs/backtest.yaml` mediante `intrabar_exit_policy`:

- `stop_first`: asume que el stop se ejecuta antes que el target
- `target_first`: asume que el target se ejecuta antes que el stop
- `conservative`: elige el peor resultado economico para la posicion abierta

El valor por defecto actual es `conservative`.

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Tests:

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

## Baseline reproducible

Runner completo:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.baseline `
  --config-dir configs `
  --start 2024-01-01T00:00:00+00:00 `
  --end 2024-03-31T23:55:00+00:00 `
  --output-dir artifacts/baseline-q1-2024
```

Alternativa:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_baseline.py `
  --config-dir configs `
  --input-path data/raw/BTCUSDT/5m/ohlcv.parquet `
  --output-dir artifacts/baseline-local
```

Artefactos generados:

- `ohlcv.csv`
- `features.csv`
- `signals.csv`
- `trades.csv`
- `report.json`
- `summary.md`

## Situacion actual del baseline

El baseline ya es reproducible y auditable, pero todavia debe endurecerse antes de usarlo como base para RL:

- la integracion `data -> features -> strategy -> backtest` ya corre end-to-end
- el baseline produce trades y metricas reales
- la capa de riesgo todavia es basica dentro del backtest
- el objetivo inmediato es sanear y estabilizar el baseline antes del siguiente sprint de `Risk Engine`
