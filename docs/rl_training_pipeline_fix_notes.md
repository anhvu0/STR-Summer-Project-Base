# RL Training Pipeline Fix Notes

Historical note: this file describes the structural cleanup that happened before the MAPPO migration. The current training path is fully on-policy and no longer uses replay backups.

This document summarizes the structural fixes applied to `core/rl_training_pipeline.py` in response to training-pathology feedback.

## 1) Reward now scales with elapsed simulation time

Previously, transition rewards were charged once per decision transition regardless of how long it took in simulation.

### Changes
- Each open transition now stores its `decision_step`.
- On transition close, `delta_t = current_step - decision_step` is computed.
- `compute_reward(...)` now accepts `delta_t`, and the base time + congestion penalties are scaled by elapsed time.

## 2) Historical: target backup respected valid-next-action constraints

Previously, action masking was used for behavior policy only; TD backup used unmasked max over all actions.

### Changes
- Replay buffer now stores `next_valid_actions` per transition.
- The replay implementation used masked next-action backup:
  - the online network selected the best valid next action,
  - the target network evaluated that selected action.

## 3) Reduced heuristic takeover in execution policy

Previously, the planner used RL only for the first move and then greedily filled the remaining horizon.

### Changes
- `build_decision_list(...)` now executes a single RL-selected direction during training.
- This keeps training and execution closer to a true learned policy.

## 4) Planner/reward objective coefficient alignment

Planner and reward used very different scales.

### Changes
- `_score_next_edge(...)` now reuses reward-side coefficient scales (`deadline_deficit_scale`, `system_congestion_scale`, `distance_tiebreak_scale`) instead of hardcoded extreme constants.

## 5) Non-global arrivals no longer receive destination success reward

Previously, terminal transitions triggered with `arrived=True` regardless of whether global destination was reached.

### Changes
- Arrival closeout now passes `reached_global_destination` into `compute_reward(...)`.
- Destination reward / on-time bonus are only granted for global-destination arrivals.
- Non-global exits incur an early-exit penalty.

## 6) Earlier decisions and route-intent action space

The prior policy decided late and only from lane-feasible actions.

### Changes
- Decision trigger is moved earlier on edges (higher adaptive threshold).
- Action selection now uses edge-feasible moves (route intent), while lane alignment is handled as lower-level control (`ensure_lane_for_direction`).

## 7) State representation improvements

The prior state had raw edge IDs and full-network density vector.

### Changes
- Raw edge IDs replaced by fixed-size edge embeddings (`current` + `destination`).
- Full density vector replaced by a compact local congestion summary:
  - current density,
  - outgoing mean/max/min density,
  - delta from global mean,
  - global density std.

These changes reduce fake ordinal bias and lower high-dimensional noise in the state.
