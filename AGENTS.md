# Project Agent Notes

## RL objective context (updated April 2026)

- The RL training objective in `core/rl_training_pipeline.py` is **average travel-time minimization**, not deadline-feasibility maximization.
- Reward priorities are:
  1. Per-step travel-time cost (dominant),
  2. Completion/arrival reward,
  3. Congestion and progress shaping,
  4. Safety/control penalties (loops, route mismatch, failed route application, unreachable transitions, teleports).
- Episode metrics are travel-time centric:
  - `completion_rate`
  - `avg_travel_time`
  - `p50_travel_time`
  - `p90_travel_time`
  - plus existing safety/decision telemetry.

## Compatibility note

- Vehicle deadlines are still present in generated vehicle objects for backward compatibility with old scenarios, but training logic should not prioritize on-time-arrival metrics unless explicitly reintroduced.

## Routing stability context (updated April 14, 2026)

- The prior route-mismatch failure was fixed by unified route application through shared `setRoute(...)` helpers (training + inference).
- The dominant remaining failure mode is now timeout-at-step-cap from overlong pending decisions and repeated lane-change deferrals.
- Current objective: reduce defer/pending loops while preserving strict SUMO route continuity/alignment.
- Key telemetry to track in each run:
  - `pending_decision_timeouts`
  - `fallback_to_lane_feasible_now`
  - `mean_pending_age`
  - `deferred_lane_change_actions`
  - `decision_pending_at_episode_end`
  - `fail_timeout`

### Files changed in this pass

- `core/junction_decision_engine.py`
- `core/rl_training_pipeline.py`
- `controller/QLearningController.py`
- `AGENTS.md`

### Next things to inspect

- Correlation between `pending_decision_timeouts` and `completion_rate` over rolling windows (ensure timeout recovery helps finish rate).
- Whether `lane_change_defer_limit` and `pending_timeout_steps` need per-network tuning for high-speed edges.
- Distribution of fallback actions to confirm controller is not over-collapsing to a single lane-feasible direction.

## Execution-aware pending/fallback policy (updated April 14, 2026)

- Strategic choice opening remains broad in `build_context()`:
  - `available_actions`, `forced_action`, `branch_with_choice`, and `is_decision_open()` keep branch points visible to policy selection.
- Conservatism is applied **after** action selection:
  - Policy still selects over `context.available_actions`.
  - Execution-aware fallback/cancel only triggers when a chosen action is lane-change constrained or pending execution stalls.
- Pending cancellation is based on **non-progress signals**, not age alone:
  - same `decision_edge`,
  - minimum pending age,
  - action still not lane-feasible-now,
  - no lane-alignment improvement,
  - shrinking remaining distance reduces maneuver executability.
- Lower defer/timeout behavior:
  - `lane_change_defer_limit` default lowered to `1`.
  - `pending_timeout_steps` lowered and made distance/speed aware through effective timeout logic.
- Reward/learning rationale:
  - Keep replay based on actual execution outcomes.
  - Add explicit penalties for non-lane-feasible selections, fallback after defer/failure, and non-progress pending cancellation.
- Post-run telemetry to monitor:
  - `pending_decision_timeouts`
  - `fallback_to_lane_feasible_now`
  - `mean_pending_age`
  - `deferred_lane_change_actions`
  - `decision_pending_at_episode_end`
  - `fail_timeout`
  - fallback distribution by original action vs fallback action.
  - `loop_events` / `dead_end_reentry_events` / `safety_overrides` to watch local oscillation control.
