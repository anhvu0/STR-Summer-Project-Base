# Telemetry Metrics Reference (Training + Inference)

This document is the canonical reference for telemetry semantics after the April 2026 metric-cleanup pass.
It explains **what each metric means**, **how it is computed**, and **how to interpret it safely**.

---

## 1) Scope and Source of Truth

Primary producers:
- Training: `core/rl_training_pipeline.py`
- Inference/runtime controller: `controller/QLearningController.py`
- Decision context semantics (skip reasons, commit window, lane-feasible actions): `core/junction_decision_engine.py`

Telemetry goals:
1. Preserve routing behavior while making diagnostics truthful.
2. Keep training and inference semantics aligned for comparable debugging.
3. Remove dead/stale metrics that have no live producer.

---

## 2) Decision Lifecycle Metrics

These metrics track the control loop lifecycle per episode.

- `decisions_opened`
  - Incremented when a pending decision is started (route pending or observe flow).
- `decisions_finalized`
  - Incremented when pending decision transitions are finalized.
- `decisions_skipped`
  - Incremented when a context is skipped (including skip-reason branch and pending-gated skip).
- `forced_actions`
  - Incremented when `context.forced_action` is used.

### Derived lifecycle ratios

- `override_ratio`
  - Formula:
    - `(safety_overrides + fallback_overrides + route_apply_fail) / max(decisions_opened, 1)`
- `skipped_to_finalized_ratio`
  - Formula:
    - `decisions_skipped / max(decisions_finalized, 1)`
- `skipped_minus_finalized`
  - Formula:
    - `decisions_skipped - decisions_finalized`
- `skipped_significantly_gt_finalized`
  - Formula:
    - `1 if decisions_skipped >= 1.25 * max(decisions_finalized, 1) else 0`

---

## 3) Skip-Reason Metrics (Single-Count Semantics)

Skip reasons originate from `JunctionDecisionEngine.build_context()` and are counted **exactly once** only when the context is actually resolved as forced/skipped.

- `skip_reason_forced_by_lane_commit`
- `skip_reason_too_late_or_unreachable`
- `skip_reason_forced_single_path`
- `skip_reason_no_branch`

### Counting rule

For any given context:
- If `forced_action` branch executes and `context.skip_reason` exists: increment once.
- Else if skip branch executes (`elif context.skip_reason`): increment once.
- No unconditional pre-counting at context creation time.

Interpretation:
- These metrics now represent **actual resolved skip/forced outcomes**, not merely context flags observed in passing.

---

## 4) Lane-Change Metrics

### Raw counters

- `lane_change_attempts`
- `lane_change_success`
- `lane_change_fail`
- `lane_change_observe_started`
- `lane_change_observe_success`
- `lane_change_observe_abort_no_progress`
- `lane_change_observe_abort_commit_window`
- `observe_abort_low_speed`
- `deferred_lane_change_actions`

### Exported request-based rates

- `lane_change_request_success_rate`
  - Formula:
    - `lane_change_success / max(lane_change_attempts, 1)`
- `lane_change_request_failure_rate`
  - Formula:
    - `lane_change_fail / max(lane_change_attempts, 1)`

> Note: names explicitly indicate request-denominator semantics; formulas are unchanged from prior implementation.

---

## 5) Fallback Metrics (Truthful Split)

Fallback is now measured with two explicit counters:

- `fallback_selected_total`
  - Increment every time a fallback action is chosen.
- `fallback_selected_lane_now`
  - Increment only when the chosen fallback action is in `context.lane_feasible_now_actions` (or observe-context equivalent).

Also tracked:
- `fallback_overrides` (override events using fallback)
- `cooldown_fallback_overrides`
- `observe_abort_fallback_overrides`
- `loop_prefilter_overrides`
- `fallback_finalized` (used in loop-after-fallback diagnostics)

### Invariant

- `fallback_selected_lane_now <= fallback_selected_total` must always hold.

### Where fallback-selected counters are incremented

Training paths:
1. observe-abort fallback
2. loop-prefilter fallback
3. cooldown fallback

Inference paths mirror the same three semantic cases.

---

## 6) Policy Candidate / Action-Space Diagnostics

These metrics evaluate whether policy candidate construction collapses to lane-now only despite broader feasible options.

- `policy_candidates_with_broader_available`
  - Increment when:
    - `len(set(filtered_available_actions)) > len(set(safe_lane_now_actions))`
- `policy_candidates_collapsed_to_lane_now_only`
  - Increment when:
    - `policy_set == lane_now_set`
    - AND `len(policy_set) < len(broader_available_set)`

### Important semantic note

Comparison baseline is now **filtered broader actions** (`filtered_available_actions`) rather than raw `available_actions`.
This prevents false collapse flags caused by actions already excluded by safety filtering.

---

## 7) Soft-Commit Window Admission Metric

- `soft_commit_window_admissions`

### Meaning
Counts one-lane proactive actions that survive filtering while in commit window.

### Increment rule
Increment if all are true:
1. `context.commit_window`
2. `required_shift == 1`
3. `action in filtered_available_actions`

This counts truthful *admission* under soft-commit semantics even if action is not finally selected.

---

## 8) Reachability/Lane-Change Availability Diagnostics

- `reachable_lane_change_nonempty`
  - Increment when reachable-with-lane-change set is non-empty.
- `reachable_lane_change_excluded_any`
  - Increment when reachable set is not subset of available action set.
- `reachable_lane_change_excluded_all`
  - Increment when reachable set and available set are disjoint.

Derived rate:
- `reachable_lane_change_excluded_any_rate`
  - Formula:
    - `reachable_lane_change_excluded_any / max(reachable_lane_change_nonempty, 1)`

---

## 9) Pending-Decision and Timeout Metrics

- `pending_decision_timeouts`
- `same_edge_pending_released_no_progress`
- `pending_resolved_success`
- `pending_resolved_timeout`
- `pending_resolved_abort_no_progress`
- `pending_commit_window_grace_kept`

Derived:
- `pending_resolution_success_rate`
  - Formula:
    - `pending_resolved_success / max(pending_resolved_success + pending_resolved_timeout + pending_resolved_abort_no_progress, 1)`
- `timeout_rate`
  - Formula:
    - `alive_at_step_cap / max(total_controlled, 1)`

---

## 10) Terminal Outcome / Failure Metrics

Live terminal outcomes are:
- `global_arrival`
- `teleport`
- `removed_nonarrival`

Related exported failure counters:
- `fail_teleport`
- `fail_timeout`
- `fail_removed_non_destination`
- `fail_unreachable_transition`
- `fail_dead_end_no_outgoing`

### Reward alignment note

No `non_global_arrival` terminal branch is used in current terminal classification.
Reward terminal handling is aligned to global-arrival vs non-global arrival fallback penalty path.

---

## 11) Social-Choice / Fairness / Tail Metrics

- `social_regret_mean`
  - Mean sampled decision regret vs local best feasible social proxy.
- `social_regret_p90`
  - P90 of sampled social regret.
- `social_best_action_chosen_rate`
  - Formula:
    - `social_best_action_chosen / max(social_regret_count, 1)`
- `delay_fairness_gini`
  - Gini coefficient over completed controlled travel times.
- `p95_to_p50_travel_ratio`
  - Formula:
    - `p95(completed_tt) / max(p50_tt, 1e-6)`
- `tail_vehicles_over_p90_count`
  - Completed vehicles in >=p90 tail + unfinished controlled vehicles.
- `tail_completion_gap_steps`
  - Tail reference travel time minus p50 travel time.

Loop/fallback coupling:
- `loop_after_fallback_rate`
  - Formula:
    - `loop_after_fallback_events / max(fallback_finalized, 1)`

---

## 12) Congestion, Safety, and Environment Diagnostics

- `mean_network_density`
- `p95_network_density`
- `congestion_high_pressure_steps`
- `emergency_brake_events`
- `emergency_brake_due_to_leader`
- `emergency_brake_due_to_congestion`
- `emergency_brake_near_junction`
- `emergency_brake_other_reason`
- `teleport_inferred_jam`
- `teleport_inferred_yield_or_deadlock`
- `controlled_teleport_rate`
  - Formula:
    - `teleported_controlled / max(total_controlled, 1)`

---

## 13) Deprecated / Removed Metrics

These are intentionally removed from CSV export because they had no live producer path in current logic:

- `decisions_superseded`
- `batched_policy_calls`
- `arrived_non_global_target`
- `distance_worsening_overrides`

Do **not** reuse these names without adding a clear producer and schema update.

---

## 14) Training vs Inference Alignment Checklist

When changing telemetry semantics, keep both files aligned:
1. `core/rl_training_pipeline.py`
2. `controller/QLearningController.py`

Minimum alignment set:
- policy collapse comparison baseline (filtered broader set)
- soft-commit admission counting rule
- fallback selected counters and lane-now truth condition

---

## 15) Quick Sanity Checks

Recommended checks after telemetry changes:

1. `fallback_selected_lane_now <= fallback_selected_total` for every run.
2. Skip-reason counts move with real skip/forced branches (no pre-count drift).
3. `soft_commit_window_admissions` can become non-zero when commit-window one-lane proactive actions survive filtering.
4. CSV row keys exactly match CSV header keys.
5. No stale metric names remain in code or docs.
