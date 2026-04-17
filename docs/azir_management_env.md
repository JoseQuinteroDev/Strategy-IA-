# design_management_actions_for_azir_v1

## Status

- Environment: `AzirManagementReplayEnvironment`.
- Unit of decision: one protected Azir trade after fill.
- Signal source: `take_all_valid` over `baseline_azir_setup_frozen_v1`.
- Hard guardrail source: `risk_engine_azir_v1`.
- Economic benchmark source for base action: `baseline_azir_protected_economic_v1`.
- PPO training: not part of this sprint.

## Why This Exists

The skip/take line is closed as the main research path. With valid-action
masking and small regularization, PPO converged to the same behavior as
`take_all_valid`. The next potentially useful RL layer is therefore management,
not signal selection.

The agent does not create entries and does not decide whether Azir should take
a valid setup. Azir plus the Risk Engine create the protected trade. The agent
only decides how that trade should be managed.

## Decision Timing

The first version defines a single management decision point:

- after a protected trade is filled,
- before the final exit is known to the agent,
- at a conceptual first post-fill management checkpoint.

This is intentionally conservative as a contract. Exact tick-level timing for
close-early, break-even or alternative trailing must be priced in a later replay
sprint before training PPO.

## Actions

The discrete action space is:

- `0 = base_management`
- `1 = close_early`
- `2 = move_to_break_even`
- `3 = trailing_conservative`
- `4 = trailing_aggressive`

The recommended first comparison baselines for the next research phase are:

- always `base_management`
- always `close_early`
- always `move_to_break_even`
- always `trailing_conservative`
- always `trailing_aggressive`
- side-aware variants such as sell-only conservative trailing if the audit still
  shows side asymmetry

## Observations

Observation fields are numeric and intentionally exclude future trade outcome
fields. They include:

- side flags,
- fill time cyclic encoding,
- calendar cyclic encoding,
- duration to fill,
- fill price versus Azir entry levels normalized by ATR,
- pending distance, spread and swing width normalized by ATR,
- previous close versus EMA20 normalized by ATR,
- ATR level scaled down,
- RSI centered around 50,
- trend/ATR/RSI gate flags,
- original Azir trailing start/step normalized by ATR.

Explicitly excluded:

- protected net/gross PnL,
- observed net/gross PnL,
- exit timestamp,
- exit reason,
- MFE/MAE,
- trailing activation/modifications,
- reward.

## Reward

The implemented reward mode is `observational_proxy_v1`.

Only `base_management` uses observed protected benchmark PnL directly. The
alternative actions use simple MFE/MAE-derived proxy rules so the contract is
testable:

- `close_early` takes a fraction of observed win/loss.
- `move_to_break_even` clips losses to zero only when MFE suggests BE could
  plausibly have activated.
- `trailing_conservative` and `trailing_aggressive` clip losses after activation
  thresholds and haircut winners to represent earlier exits.

This is not a frozen economic benchmark. It is a design scaffold. Before PPO
training, the project should replace or validate these proxies with M1/tick
price replay.

## Risk Engine Integration

The environment consumes only trades that survived the protected economic
benchmark. That means:

- Azir generated the opportunity.
- `risk_engine_azir_v1` allowed the lifecycle.
- forced closes already handled by the protected benchmark are included only
  when revalued.

The management agent cannot resurrect prevented trades, create new trades,
increase size, or violate lifecycle guards.

## CLI

```powershell
$env:PYTHONPATH='src'
python -m hybrid_quant.env.azir_management_env `
  --mt5-log-path "C:\Users\joseq\Documents\Playground\todos_los_ticks.csv" `
  --protected-report-path "C:\Users\joseq\Documents\Playground\hybrid-quant-framework\artifacts\azir-protected-economic-v1-freeze\forced_close_revaluation_report.json" `
  --output-dir "C:\Users\joseq\Documents\Playground\hybrid-quant-framework\artifacts\azir-management-env-v1" `
  --symbol XAUUSD-STD
```

Artifacts:

- `azir_management_action_contract.json`
- `azir_management_sample_observations.csv`
- `azir_management_sample_episode.csv`
- `azir_management_env_summary.md`

## Decision

This sprint defines the management environment contract. It does not justify
training PPO yet. The recommended next sprint is:

`management_price_replay_and_heuristic_backtest_v1`

That sprint should price each management action with M1/tick replay or explicitly
mark which actions cannot be economically evaluated.
