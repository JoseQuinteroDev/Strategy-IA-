# Hybrid Quant Framework

Base profesional y extensible para un framework de trading cuantitativo hibrido orientado a prop firms, construido en Python.

## Objetivo de esta fase

Esta primera fase prepara la arquitectura y el repositorio para el MVP inicial, sin introducir todavia logica compleja de trading ni entrenamiento RL.

Scope actual:

- Activo inicial: `BTCUSDT`
- Timeframe operativo: `5m`
- Timeframe de filtro: `1H`
- Estilo: `intradia`
- Estrategia base objetivo: `mean reversion` con filtro de tendencia y regimen
- Orden de construccion: estrategia base -> risk engine -> entorno RL -> PPO

## Principios de arquitectura

- `src/` layout para separar codigo de aplicacion y entorno de trabajo.
- Configuracion declarativa por YAML en `configs/`.
- Modulos desacoplados por responsabilidad: `data`, `features`, `strategy`, `risk`, `backtest`, `env`, `rl`, `validation`, `paper`.
- Contratos compartidos en `hybrid_quant.core`.
- Implementaciones scaffold que permiten cableado temprano y evolucion incremental.

## Estructura

```text
hybrid-quant-framework/
|-- configs/
|-- docs/
|   `-- spec.md
|-- src/
|   `-- hybrid_quant/
|       |-- backtest/
|       |-- core/
|       |-- data/
|       |-- env/
|       |-- features/
|       |-- paper/
|       |-- risk/
|       |-- rl/
|       |-- strategy/
|       |-- validation/
|       `-- bootstrap.py
|-- tests/
|   |-- integration/
|   `-- unit/
`-- pyproject.toml
```

## Modulos del MVP

- `data`: contratos de carga y abstracciones para historico, cache y futuras fuentes exchange.
- `features`: pipeline de features desacoplado del origen de datos.
- `strategy`: capa de decision para estrategias rule-based y futuras politicas hibridas.
- `risk`: control de sizing, limites diarios, exposure y reglas prop-firm friendly.
- `backtest`: simulacion offline y reportes reproducibles.
- `env`: entorno compatible con futuras integraciones RL.
- `rl`: scaffolding para entrenamiento futuro, sin implementar PPO todavia.
- `validation`: walk-forward, filtros de robustez y metricas de aceptacion.
- `paper`: ejecucion simulada para paper trading y dry runs.

## Configuracion

La configuracion se distribuye por archivos YAML:

- `configs/base.yaml`
- `configs/data.yaml`
- `configs/features.yaml`
- `configs/strategy.yaml`
- `configs/risk.yaml`
- `configs/backtest.yaml`
- `configs/env.yaml`
- `configs/rl.yaml`
- `configs/validation.yaml`
- `configs/paper.yaml`

El cargador fusiona los archivos en orden y genera un objeto tipado de settings.

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Si vas a ejecutar exportacion Parquet fuera de este entorno, instala tambien las dependencias de datos definidas en `pyproject.toml`.

Para ejecutar los tests base con librerias de la stdlib:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Ingestion de datos

La fase de datos ya incluye:

- descarga historica OHLCV desde Binance REST
- limpieza de duplicados y orden temporal
- validacion de indices de tiempo por timeframe
- exportacion a Parquet
- split cronologico `train/validation/test`
- CLI para descargar y partir datasets

Ejemplo de descarga e ingesta completa:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.data download `
  --config-dir configs `
  --start 2024-01-01T00:00:00+00:00 `
  --end 2024-02-01T00:00:00+00:00
```

Ejemplo de split sobre un Parquet ya exportado:

```powershell
$env:PYTHONPATH = "src"
python -m hybrid_quant.data split `
  --config-dir configs `
  --input-path data/raw/BTCUSDT/5m/ohlcv.parquet
```

Tambien puedes usar los entry points:

- `hybrid-quant-data download ...`
- `hq-data-download ...`
- `hq-data-split ...`

## Estado actual

Este scaffold ya deja resuelto:

- layout profesional del repositorio
- contratos y placeholders por modulo
- configuracion YAML tipada
- bootstrap de la aplicacion
- documentacion tecnica del MVP
- tests base para wiring y contratos

No incluye todavia:

- logica real de ingestion de datos desde exchange
- calculo real de indicadores y features
- reglas operativas completas de la estrategia
- engine realista de ejecucion o fill simulation
- entrenamiento PPO ni pipelines RL

## Roadmap de desarrollo

1. Implementar la estrategia base de mean reversion con filtros de tendencia y regimen.
2. Construir el risk engine con sizing, daily loss guardrails y restricciones prop firm.
3. Endurecer el backtest con costes, slippage y validacion walk-forward.
4. Implementar el entorno RL sobre la misma semantica de estados, acciones y recompensas.
5. Incorporar PPO cuando la capa rule-based y el risk engine esten estabilizados.
