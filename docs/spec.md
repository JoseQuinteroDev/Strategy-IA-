# MVP Technical Specification

## 1. Vision

Construir un framework de trading cuantitativo hibrido, modular y prop-firm friendly para evolucionar desde una estrategia base rule-based hacia una capa RL posterior sin reescribir el stack completo.

## 2. Alcance del MVP

- Instrumento: `BTCUSDT`
- Timeframe operativo: `5m`
- Timeframe de filtro: `1H`
- Estilo: `intradia`
- Estrategia objetivo: `mean reversion` con filtros de tendencia y regimen
- Prioridad de ejecucion:
  1. estrategia base
  2. risk engine
  3. entorno RL
  4. PPO

## 3. No objetivos de esta fase

- No entrenar agentes RL.
- No implementar aun PPO.
- No integrar exchange real.
- No modelar ejecucion avanzada ni microestructura.
- No optimizar hiperparametros.

## 4. Requisitos de arquitectura

- Python `3.11+`
- `src/` layout
- Configuracion centralizada por YAML
- Separacion fuerte por bounded contexts funcionales
- Contratos de dominio compartidos y tipados
- Bootstrap unico para ensamblar el sistema
- Tests base desde el primer commit

## 5. Modulos

### 5.1 data

Responsabilidades:

- definir contratos de peticion y respuesta de mercado
- abstraer proveedores historicos, cache y futuras conexiones live
- entregar batches coherentes por simbolo y timeframe

Salida esperada:

- `MarketDataBatch`

### 5.2 features

Responsabilidades:

- transformar barras en snapshots de features
- desacoplar calculo de features del proveedor de datos
- preparar entradas reutilizables por estrategia, backtest y RL

Salida esperada:

- `FeatureSnapshot[]`

### 5.3 strategy

Responsabilidades:

- encapsular la estrategia base rule-based
- exponer una interfaz estable de generacion de senales
- permitir sustitucion futura por politicas hibridas

Salida esperada:

- `StrategySignal`

### 5.4 risk

Responsabilidades:

- decidir si una senal puede ejecutarse
- aplicar limites diarios, sizing y restricciones prop-firm
- producir una decision auditable independiente de la estrategia

Salida esperada:

- `RiskDecision`

### 5.5 backtest

Responsabilidades:

- orquestar simulaciones reproducibles
- registrar metrica base del experimento
- permanecer compatible con strategy y risk

Salida esperada:

- `BacktestResult`

### 5.6 env

Responsabilidades:

- ofrecer un entorno RL-compatible que reutilice el mismo modelo de datos
- exponer observaciones, transiciones y estado de posicion
- servir como puente entre backtesting y entrenamiento

Salida esperada:

- `EnvObservation`, `EnvTransition`

### 5.7 rl

Responsabilidades:

- reservar contratos para entrenamiento futuro
- mantener el punto de integracion con el entorno
- preparar el camino hacia PPO sin acoplar la fase 1

Salida esperada:

- `TrainingArtifact`

### 5.8 validation

Responsabilidades:

- validar robustez minima del sistema
- consolidar checks de walk-forward y metricas
- producir un reporte binario y auditable

Salida esperada:

- `ValidationReport`

### 5.9 paper

Responsabilidades:

- simular ejecucion live sin dinero real
- validar handoff entre strategy, risk y ejecucion
- soportar dry runs y futuras integraciones broker/exchange

Salida esperada:

- `PaperExecution`

## 6. Flujo previsto

```text
data -> features -> strategy -> risk -> backtest
                                 |-> validation
                                 |-> paper

data + features + portfolio state -> env -> rl
```

## 7. Configuracion del MVP

El sistema debe cargar configuracion desde `configs/` y fusionarla en un objeto tipado.

Secciones requeridas:

- `app`
- `market`
- `storage`
- `data`
- `features`
- `strategy`
- `risk`
- `backtest`
- `env`
- `rl`
- `validation`
- `paper`

## 8. Reglas iniciales del dominio

- El activo inicial es exclusivamente `BTCUSDT`.
- El sistema opera sobre `5m` y consulta filtro superior `1H`.
- La arquitectura debe asumir intradia y dejar espacio para limites diarios.
- El risk layer debe pensarse desde el principio como componente central, no accesorio.
- La capa RL debe depender del mismo contrato de observaciones y acciones que el sistema clasico.

## 9. Criterios de aceptacion de esta fase

- Existe una estructura de repo limpia y profesional.
- Hay README y documentacion tecnica del MVP.
- Todos los modulos requeridos existen y exponen contratos claros.
- La configuracion YAML se puede cargar en settings tipados.
- Hay tests base que ejercitan el wiring principal.
- No se ha introducido logica compleja de trading ni entrenamiento RL.

## 10. Proxima implementacion recomendada

1. Implementar calculo real de features para mean reversion, tendencia y regimen.
2. Activar la estrategia base sobre esos features y definir senales auditablemente.
3. Endurecer el risk engine para restricciones tipo prop firm.
4. Enriquecer el backtest antes de abrir la fase de entorno RL.

