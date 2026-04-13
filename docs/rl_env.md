# design_rl_env_for_azir_v1

## Status

- Environment: `AzirEventReplayEnvironment`.
- Unit of decision: one Azir daily setup event.
- Signal source: AzirIA MT5 / frozen setup benchmark.
- Risk layer: `risk_engine_azir_v1`.
- Economic source for rewards: `baseline_azir_protected_economic_v1`.
- PPO training: not part of this sprint.

## Philosophy

The agent does not generate trades from raw market bars. Azir proposes a daily
setup, the Risk Engine applies hard lifecycle/risk constraints, and the policy
only chooses between:

- `0 = skip`
- `1 = take`

If the Risk Engine blocks a setup, action `take` is transformed into `skip` with
a small invalid-action penalty. The agent cannot override hard risk rules.

## Pipeline

1. Load the MT5 event log.
2. Build one canonical setup event per Azir setup day.
3. Apply `risk_engine_azir_v1` lifecycle simulation.
4. Attach protected economic outcomes from `baseline_azir_protected_economic_v1`.
5. At each step, build observation from setup/context/risk state only.
6. Apply action `skip` or `take`.
7. Compute reward from protected benchmark PnL and risk penalties.
8. Advance to the next Azir setup event.

## Observation

The observation is a flat numeric vector. It intentionally contains no future
outcome, no PnL target, and no exit information.

Included setup fields:

- setup hour
- day of week
- month
- Friday flag
- buy/sell order placement flags
- buy/sell trend-allowed flags
- swing high / swing low
- buy entry / sell entry
- pending distance
- spread points
- EMA20
- previous close vs EMA20 in points
- ATR and ATR points
- RSI
- trend filter enabled
- ATR filter enabled / passed
- RSI gate enabled / required

Included operational/risk fields:

- prior exposure flag
- cleanup issue flag before risk layer
- daily realized PnL
- daily drawdown
- total drawdown
- consecutive losses today
- trades today
- remaining daily loss
- risk tension ratio
- Risk Engine approved flag
- Risk Engine blocked flag

Explicitly excluded:

- protected net/gross PnL
- observed net/gross PnL
- reward
- exit reason
- fill timestamp
- exit timestamp
- MFE/MAE
- trailing outcome

## Reward

Initial reward mode:

```text
reward = protected_net_pnl
         - drawdown_penalty
         - risk_tension_penalty
         - invalid_action_penalty
```

Where:

- `protected_net_pnl` comes from `baseline_azir_protected_economic_v1` only when
  the action is valid and taken.
- `drawdown_penalty` is proportional to current daily and total drawdown.
- `risk_tension_penalty` increases as daily loss approaches the configured limit.
- `invalid_action_penalty` applies when the agent attempts to take a setup with
  no Azir order or a setup blocked by the Risk Engine.

Skipping receives zero reward in this MVP. There is no opportunity-cost reward
yet; that is intentionally left for a later reward-shaping sprint.

## Artifacts

The inspection CLI writes:

- `azir_rl_observation_schema.json`
- `azir_rl_sample_observations.csv`
- `azir_rl_valid_actions.csv`
- `azir_rl_sample_episode.json`

## CLI

```powershell
$env:PYTHONPATH='src'
python -m hybrid_quant.env.azir_event_env `
  --mt5-log-path "C:\Users\joseq\Documents\Playground\todos_los_ticks.csv" `
  --protected-report-path "C:\Users\joseq\Documents\Playground\hybrid-quant-framework\artifacts\azir-protected-economic-v1-freeze\forced_close_revaluation_report.json" `
  --output-dir "C:\Users\joseq\Documents\Playground\hybrid-quant-framework\artifacts\azir-rl-env-v1" `
  --symbol XAUUSD-STD
```

## Current Decision

This environment is ready for the next sprint, `train_first_ppo_skip_take`, only
as an interface contract and replay substrate. PPO training should still be
treated as a separate sprint with its own checks, baselines, seeds and reporting.
