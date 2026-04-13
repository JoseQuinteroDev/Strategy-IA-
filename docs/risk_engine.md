# Azir Risk Engine V1

`risk_engine_azir_v1` is the first hard risk layer around AzirIA MT5. It does
not replace Azir, does not generate entries and does not modify the EA's
trading rules. It is a lifecycle guard that decides whether a setup can proceed
and which operational cleanup actions must be executed.

## Source Context

The risk layer is based on:

- `baseline_azir_setup_frozen_v1`
- `C:\Users\joseq\Documents\Playground\Azir.mq5`
- `C:\Users\joseq\Documents\Playground\todos_los_ticks.csv`
- Economic audit artifacts under `artifacts\azir-economic-audit`

The economic audit found:

- 6 multi-exit days
- 5 fills outside the intended 16-21 server-time fill window
- 2 Friday exit events despite Friday setup blocking
- 8 cleanup/persistence issues

## Design

The engine observes broker/account state:

- timestamp in MT5 server time
- live pending order count
- open position count
- daily realized PnL
- trades today
- consecutive losses today
- spread when available
- reconciliation status
- trailing audit flags

It emits:

- `approved` or blocked setup decision
- `blocked_by` rule codes
- lifecycle actions such as `cancel_all_pendings` or `close_open_positions`
- warnings for audit-only cases such as trailing state

The integration layer is responsible for executing actions in MT5 or Python.
The Risk Engine remains policy, not broker plumbing.

## Implemented Rules

`hard_cancel_all_pendings_at_close`

- At/after close hour, cancel all pending orders.
- If configured, close open positions as well.
- Targets stale GTC exposure and out-of-window fills.

`block_new_setups_if_any_position_or_pending_exists`

- Blocks a new 16:30 setup if there is any live pending or open position.
- Targets overlapping lifecycle state and multi-exit days.

`force_reconcile_orders_positions_before_setup`

- Requires clean broker/account state before setup.
- With `block_and_cleanup`, emits cancel/close cleanup actions and blocks setup.

`friday_no_new_trade_plus_close_or_cancel_prior_exposure`

- Blocks new Friday setup.
- If prior exposure exists, emits cancel/close actions.
- This extends Azir's existing Friday block, which only blocks new setup.

`daily_max_loss_guard`

- Blocks new setup and activates kill-switch after configured daily loss.

`consecutive_losses_kill_switch`

- Blocks after configured consecutive daily losses.

`max_trades_per_day`

- Caps Azir activity per server-time day.

`spread_guard_if_available`

- Blocks setup if spread is available and above threshold.
- Emits an audit warning if spread is missing.

`trailing_guardrails`

- Does not modify trailing.
- Audits whether trailing is expected/active for open positions.

`cancel_remaining_pendings_after_fill`

- Emits cleanup after a fill if pending orders remain live.
- This is a protective lifecycle rule, not an entry-rule change.

## Evaluation Against Economic Audit Anomalies

Run:

```powershell
$env:PYTHONPATH='src'
python -m hybrid_quant.risk.azir_engine `
  --mt5-log-path "C:\Users\joseq\Documents\Playground\todos_los_ticks.csv" `
  --output-dir "artifacts\azir-risk-engine-v1" `
  --symbol "XAUUSD-STD"
```

Generated artifacts:

- `azir_risk_engine_report.json`
- `azir_risk_engine_summary.md`
- `azir_risk_anomaly_mitigation.csv`

## Important Limitation

The evaluation estimates policy impact against logged anomalies. It is not a
new MT5 backtest and not a frozen economic benchmark. The next required step is
to re-audit Azir with this risk layer active or simulated in the lifecycle
reconstruction.
