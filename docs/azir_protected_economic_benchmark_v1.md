# baseline_azir_protected_economic_v1

## Status

- Status: FROZEN as protected economic benchmark v1.
- Base strategy: AzirIA MT5 unchanged.
- Setup/filter benchmark dependency: `baseline_azir_setup_frozen_v1`.
- Protection layer: `risk_engine_azir_v1`.
- RL/PPO status: not trained, not enabled.

## Sources Of Truth

- EA source: `C:\Users\joseq\Documents\Playground\Azir.mq5`.
- MT5 empirical event log: `C:\Users\joseq\Documents\Playground\todos_los_ticks.csv`.
- M1 price source for forced-close repricing: `C:\Users\joseq\Documents\xauusd_m1.csv`.
- Risk policy: `risk_engine_azir_v1`.

## Pricing Convention

The protected benchmark applies Risk Engine lifecycle guards to the observed MT5
event log. Observed PnL is retained only when the protected lifecycle would keep
the same exit. For the two forced-close counterfactuals, PnL is repriced with
M1 data:

- Base price: close of the latest fully closed M1 bar before the configured
  Risk Engine close hour, 22:00 server time.
- Sensitivity: selected M1 bar OHLC plus the first available post-close M1
  open/close when present.
- Commission/swap: 0.00 for the two repriced cases, matching the observed MT5
  close rows.
- PnL formula: `(exit_price - entry_price) * direction * lot_size * contract_size`,
  with `lot_size=0.10`, `contract_size=100.0`, `direction=1` for buys and `-1`
  for sells.

## Final Metrics

- Closed trades: 836.
- Net PnL: 376.08.
- Win rate: 85.0478%.
- Average win: 1.4154.
- Average loss: -5.0424.
- Payoff: 0.2807.
- Profit factor: 1.5967.
- Expectancy: 0.4499.
- Max drawdown abs: 39.56.
- Max consecutive losses: 4.

## Forced-Close Repricing

- 2024-06-19 buy: entry 2331.48, selected M1 close 2330.03, revalued PnL -14.50.
- 2024-09-02 sell: entry 2498.42, selected M1 close 2499.95, revalued PnL -15.30.

## Freeze Decision

The protected benchmark can be frozen because:

- lifecycle anomalies are controlled by `risk_engine_azir_v1`;
- all previously unpriced forced-close cases are repriced with a deterministic
  M1 convention;
- the protected/revalued economics remain positive after conservative repricing;
- the uncertainty band does not change the strategic decision that Azir retains
  observable protected edge.

## Limitations

- This benchmark is not a tick-perfect MT5 trailing/execution replica.
- The two repriced days have no M1 bar exactly at 22:00; the benchmark uses the
  latest closed M1 quote before that target.
- Future PPO should use this as the protected economic baseline, not as proof
  that trailing internals are fully replicated tick-by-tick.
