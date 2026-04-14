# Loop / Dead-end Tail Failure Analysis Notes

Updated: April 14, 2026

## Scope

Use this note when analyzing runs where:

- completion drops after early progress,
- a few vehicles take very long to finish,
- or `fail_timeout` grows while many decisions are still being opened.

This note is context for training + inference behavior in:

- `core/junction_decision_engine.py`
- `core/rl_training_pipeline.py`
- `controller/QLearningController.py`

## What changed recently

Recent routing-control patches introduced:

1. Lane-change observe/cooldown state flow (observe first, commit later).
2. Pre-commit loop/trap filter (`prefilter_action_for_loops`).
3. Ranked fallback selection (`ranked_fallback_actions`) instead of simple first-available fallback.
4. Additional metrics for observe, fallback, cooldown, and loop/dead-end overrides.
5. Lane-change attempt/success/fail counters wired in training logs.

## Important interpretation

These changes are **mitigations**, not a formal proof that loops/dead-end reentry are solved.

- If `loop_events` / `dead_end_reentry_events` remain high, that indicates persistent local-minima behavior.
- High `override_ratio` with high `fallback_overrides` usually means the policy is repeatedly being corrected away from unsafe or low-quality decisions.
- Good early completion with bad tail completion usually indicates a small subset of vehicles trapped in recurrent fallback/cooldown patterns.

## Minimum metric set to inspect together

Track these per episode and rolling windows:

- `completion_rate`
- `avg_travel_time`, `p50_travel_time`, `p90_travel_time`
- `loop_events`
- `short_cycle_events`
- `aba_bounce_events`
- `dead_end_reentry_events`
- `loop_override_count`
- `dead_end_reentry_override_count`
- `fallback_overrides`
- `fallback_to_lane_feasible_now`
- `cooldown_replans_blocked`
- `pending_decision_timeouts`
- `decision_pending_at_episode_end`
- `fail_timeout`
- `teleported_controlled`

## Quick diagnosis patterns

### Pattern A: High overrides + high fallback + high dead_end_reentry

Likely cause:

- fallback ranking still collapses to locally safe but globally poor continuation under congestion.

Check:

- whether fallback candidate diversity is low,
- whether same edge repeats dominate recent history for tail vehicles.

### Pattern B: High pending timeout + low lane_change_success

Likely cause:

- observe windows are too permissive for low-speed/high-density areas or commit windows are reached too late.

Check:

- `lane_change_observe_abort_no_progress`
- `lane_change_observe_abort_commit_window`
- `same_edge_pending_released_no_progress`

### Pattern C: Early completion strong, tail fails at step cap

Likely cause:

- policy works for easy flows but oscillates for hard congestion pockets.

Check:

- episodes with high `cooldown_replans_blocked` and repeated overrides near the end.

## Suggested reporting template

When posting run analysis, include:

1. Episode range and seeds.
2. Rolling metrics (`completion_rate`, `avg_travel_time`, `fail_timeout`).
3. Tail-focused safety metrics (loop/dead-end/fallback/cooldown).
4. One sentence diagnosis:
   - "policy under-explores",
   - "fallback churn dominates",
   - or "lane-change observe too slow/too lenient".
5. Next tuning hypothesis with expected metric movement.

## Reminder for future patches

- Keep training and inference semantics aligned for observe/fallback/cooldown.
- Do not report dead-end problem as "fixed" unless both:
  - dead-end telemetry stabilizes at low values across rolling windows, and
  - tail timeouts (`fail_timeout`) remain low under varied seeds.
