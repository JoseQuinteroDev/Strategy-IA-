# RL Environment Semantics

## Overview

`HybridTradingEnvironment` is a Gymnasium-compatible intraday trading environment built on top of the same execution and risk stack used by the deterministic baseline.

The environment does not invent setups from scratch. The rule-based strategy still generates one candidate trade per bar, and the agent decides what to do with that candidate.

## Observation

Each observation is a single-bar snapshot composed of:

- current deterministic market features for the active bar
- current position direction
- cash ratio
- equity ratio
- daily pnl percentage
- total drawdown percentage
- trades used today
- daily kill switch state
- session allowed flag
- remaining distance to the main risk limits
- candidate trade flags:
  - whether the current bar has an actionable setup
  - candidate side encoded as `-1 / 0 / 1`

Current observation mode:

- `single_bar_features_plus_portfolio_state`

Canonical context parameter:

- `state_context_bars`

Backward-compatible alias:

- `observation_window`

Important:

- `state_context_bars` is currently a context budget hint and compatibility field.
- The environment still emits a single-bar observation, not a stacked temporal tensor.

## Actions

- `0 = skip`
- `1 = take_trade`
- `2 = close_early`

Meaning:

- `skip`: do nothing on the current candidate trade.
- `take_trade`: ask the environment to take the candidate trade attached to the current bar.
- `close_early`: if there is an open position, force-close it on the current bar.

## Step Semantics

`step(action)` follows this sequence:

1. Read the current bar at `cursor`.
2. Read the candidate trade for that same bar.
3. Evaluate the candidate with `PropFirmRiskEngine` against the current `PortfolioState`.
4. Apply the agent action.
5. Advance the shared `PortfolioSimulator` by one bar.
6. Recompute portfolio state and reward.
7. Return the next observation plus `info`.

Timing details:

- The candidate trade is always evaluated on the current bar.
- `take_trade` queues the trade for execution on the next bar.
- The simulator then processes the next bar using the same semantics as the baseline:
  - slippage
  - fees
  - intrabar exit policy
  - stop loss / target
  - time stop
  - session close

Bar advance:

- Every successful call to `step()` advances the episode by at most one bar.
- If the episode is already on the final bar, the environment force-closes any remaining position with `end_of_data`.

## Reward

Reward is based on step-level equity change plus penalties:

- equity delta normalized by initial capital
- fee penalty
- daily drawdown penalty
- total drawdown penalty
- overtrading penalty
- violation penalty for blocked attempts or breached prop-firm limits

Reward modes:

- `risk_adjusted`: full reward with penalties
- `pnl_only`: only normalized equity delta

The environment returns a reward breakdown in `info["reward_breakdown"]`.

## Execution and Risk Interaction

`PortfolioSimulator` is responsible for:

- pending entry queue
- opening and closing positions
- intrabar exit resolution
- session close exits
- time stop exits
- equity and cash updates

`PropFirmRiskEngine` is responsible for:

- approving or blocking the current candidate trade
- size fraction
- leverage cap
- session gating
- stop-loss enforcement
- daily and total risk guardrails

Interaction model:

- risk is checked before a new trade is queued
- execution is handled by the simulator after the action is chosen
- reward and terminal conditions are derived from the post-step portfolio state

## Info Payload

The environment exposes a stable `info` payload including:

- action and action effect
- candidate trade metadata
- risk approval and block reasons
- reward breakdown
- closed trades on the step
- portfolio snapshot
- observation keys
- termination or truncation reason when applicable
