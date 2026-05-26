# Telemetry Metrics Reference (Training + Inference)

Updated: April 21, 2026

This document explains the telemetry that matters most after the April 2026 routing-stability and frozen-evaluation updates.

## 1) Source of truth

Primary producers:
- training: `core/rl_training_pipeline.py`
- inference/runtime controller: `controller/MAPPOController.py`
- runtime simulation wrapper: `core/STR_SUMO.py`

The two main CSV outputs now serve different purposes:
- `rl_episode_metrics.csv`: training-rollout metrics with online replay still active inside the episode.
- `rl_frozen_eval_metrics.csv`: held-out frozen evaluation metrics with no online learning.

## 2) Decision lifecycle metrics

Core counters:
- `decisions_opened`
- `decisions_finalized`
- `decisions_skipped`
- `forced_actions`

Useful derived ratios:
- `override_ratio`
- `actionable_skip_ratio`
- `pending_resolution_success_rate`
- `fallback_rate_per_opened_decision`

Interpretation:
- rising `decisions_opened` with flat `decisions_finalized` usually means pending churn is growing,
- high `actionable_skip_ratio` with good social-choice quality often means decision availability, not policy ranking, is the bottleneck.

## 3) Loop and fallback diagnostics

Key loop metrics:
- `loop_events`
- `short_cycle_events`
- `aba_bounce_events`
- `dead_end_reentry_events`
- `loop_override_count`
- `dead_end_reentry_override_count`
- `loop_after_fallback_rate`

Key fallback metrics:
- `fallback_selected_total`
- `fallback_selected_lane_now`
- `fallback_overrides`
- `fallback_to_lane_feasible_now`

Interpretation:
- if loop signals rise together with fallback signals, fallback churn is likely dominating tail failures,
- modest lane-now fallback growth can be healthy if it replaces long proactive stalls.

## 4) Pending-decision and timeout metrics

Key counters:
- `pending_decision_timeouts`
- `pending_release_route_no_progress_abort`
- `pending_release_route_stall_timeout`
- `pending_release_route_hard_timeout`
- `deferred_lane_change_actions`
- `cooldown_replans_blocked`
- `decision_pending_at_episode_end`
- `fail_timeout`
- `timeout_rate`

Interpretation:
- these are the first metrics to check when vehicles survive too long near step cap,
- use them together with completion rate and tail travel-time metrics,
- high timeout metrics with low loop metrics usually points to stale pending management rather than explicit loop traps.

## 5) Emergency-brake metrics

Tracked metrics:
- `emergency_brake_events`
- `emergency_brake_due_to_leader`
- `emergency_brake_due_to_congestion`
- `emergency_brake_near_junction`
- `emergency_brake_other_reason`
- `emergency_brake_after_fallback`
- `emergency_brake_after_proactive`
- `emergency_brake_after_lane_now`
- `emergency_brake_without_recent_decision`

How to interpret them safely:
- these are heuristic stress signals, not collision counts,
- a spike means the policy or the surrounding traffic created harsher-than-desired local braking,
- `after_fallback`, `after_proactive`, and `after_lane_now` help locate whether the stress comes from fallback recovery, proactive maneuvers, or ordinary lane-now commitments,
- compare them with teleport and completion metrics before concluding the policy is unsafe.

## 6) Teleport and tail metrics

Key metrics:
- `teleport_inferred_jam`
- `teleport_inferred_yield_or_deadlock`
- `controlled_teleport_rate`
- `tail_vehicles_over_p90_count`
- `tail_completion_gap_steps`
- `p95_to_p50_travel_ratio`
- `delay_fairness_gini`

Interpretation:
- teleports indicate severe local failures or deadlocks,
- tail metrics show whether only a few vehicles are collapsing while averages still look acceptable,
- use `gini`, `p95_to_p50`, and `tail_completion_gap_steps` together to avoid being fooled by a good mean travel time.

## 7) Frozen evaluation metrics

`rl_frozen_eval_metrics.csv` contains aggregate held-out inference results.
Current fields:
- `episode`
- `seed_count`
- `seed_list`
- `spawn_interval`
- `completion_rate_mean`
- `avg_travel_time_mean`
- `p50_travel_time_mean`
- `p90_travel_time_mean`
- `timeout_rate_mean`
- `tail_completion_gap_steps_mean`
- `p95_to_p50_travel_ratio_mean`
- `deadlines_missed_mean`
- `vehicles_reached_destination_mean`
- `controlled_vehicle_count_mean`
- `best_checkpoint_updated`
- `score_key`

Checkpoint ranking priority:
1. higher completion rate,
2. lower timeout rate,
3. lower average travel time,
4. lower `p90` travel time,
5. lower tail completion gap,
6. lower `p95`/`p50` travel ratio,
7. lower deadline misses.

This ranking is intentionally deployment-oriented.

## 8) Practical reading order

When a run looks bad, inspect in this order:
1. `completion_rate`, `avg_travel_time`, `p90_travel_time`, `timeout_rate`
2. `pending_decision_timeouts`, `deferred_lane_change_actions`, `fail_timeout`
3. loop and fallback metrics
4. emergency-brake metrics
5. teleport and tail metrics

## 9) Bottom line

Training metrics tell you how learning is progressing.
Frozen-eval metrics tell you how the saved checkpoint behaves when deployed.
Use both, but do not treat them as interchangeable.
