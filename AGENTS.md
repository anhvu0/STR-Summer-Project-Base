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

## Execution-aware routing policy (updated April 14, 2026)

- Lane-feasible-now execution is now dominant near commit windows and under pressure:
  - If close to junction commit distance, lane alignment is not improving, or edge density is high, prefer `lane_feasible_now_actions`.
  - Keep strategic spreading where possible, but do not wait through long lane-change-required pending loops.
- Defaults were tightened to reduce dead waiting:
  - `lane_change_defer_limit = 1`
  - `pending_timeout_steps = 6`
- Early pending cancellation/replan is enabled:
  - If a pending decision stays on the same edge for several steps without lane-alignment progress and road is running out, cancel early and replan/fallback.
  - Inputs include pending age, lane index/required shift, `dist_to_end`, and lane-feasible alternatives.
- Reward/learning rationale:
  - Training and inference both keep execution-aware fallback so policy learns realistic, executable decisions.
  - Training explicitly penalizes non-`lane_feasible_now` selections and lane-change fallback cases so the model learns travel-time gains that are physically executable.
- Telemetry expectations after each run:
  - Must track `pending_decision_timeouts`, `fallback_to_lane_feasible_now`, `mean_pending_age`, `deferred_lane_change_actions`, `decision_pending_at_episode_end`, `fail_timeout`.
  - Inspect fallback distribution by original action -> fallback action (not just total fallback count) to catch collapse to a single direction.
