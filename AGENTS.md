# Agent Context Notes

## RL Objective (current default)
- `core/rl_training_pipeline.py` is tuned for **average travel time** optimization.
- Deadline values are still generated and tracked, but treated as **diagnostics** (`on_time_diag`, `avg_tardy_diag`) rather than primary reward drivers.

## Robustness / Performance choices
- The training step loop avoids redundant decision-context construction for vehicles with pending decisions.
- TraCI access paths catch `TraCIException` around volatile vehicle state accesses so "vehicle is not known" events do not break loop progress.
- Route-application attempts are guarded by the step-local active-vehicle set to reduce stale-ID operations without extra `getIDList()` calls.

## Logging conventions
- Step logs report `ontime_diag`.
- Episode summary reports `avg_travel` and `avg_tardy_diag`.
