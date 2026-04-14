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

## Tail-vehicle loop/dead-end analysis context (updated April 14, 2026)

- Recent patches added:
  - pre-commit loop/trap filtering (`prefilter_action_for_loops`)
  - ranked fallbacks (`ranked_fallback_actions`)
  - lane-change observe/cooldown flow
  - lane-change attempt/success/fail telemetry wiring in training
- Interpretation guidance:
  - These changes **mitigate** high loop/dead-end reentry pressure but do **not** guarantee elimination.
  - If episodes still show high `loop_events`, `dead_end_reentry_events`, or very high `override_ratio`, treat this as fallback-churn / local-minima behavior rather than route-apply mismatch.
- Metrics that should be reviewed together for tail failures:
  - `loop_events`
  - `short_cycle_events`
  - `aba_bounce_events`
  - `dead_end_reentry_events`
  - `loop_override_count`
  - `dead_end_reentry_override_count`
  - `fallback_overrides`
  - `fallback_to_lane_feasible_now`
  - `cooldown_replans_blocked`
  - `fail_timeout`
- Known caveat from run logs:
  - Some episodes still end with a few vehicles near step-cap despite good early completion, indicating long-tail policy instability under congestion.
- See `docs/loop_deadend_tail_analysis.md` for analysis checklist and reporting template.
