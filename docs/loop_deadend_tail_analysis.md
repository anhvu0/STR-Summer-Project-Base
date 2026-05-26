# Loop / Dead-end / Tail Failure Analysis Notes

Updated: April 21, 2026

## Scope

Use this note when analyzing runs where:
- completion drops after strong early progress,
- a small tail of vehicles takes extremely long to finish,
- loop and dead-end signals stay high,
- or emergency-brake / teleport signals spike together with pending churn.

This note applies to:
- `core/junction_decision_engine.py`
- `core/shared_decision_policy.py`
- `core/rl_training_pipeline.py`
- `controller/MAPPOController.py`

## What changed

The current mitigation stack is built from several layers rather than one single fix.

Loop and trap mitigation:
1. `prefilter_action_for_loops` blocks obviously bad short-cycle, dead-end-reentry, and trap-like actions before they are committed.
2. `ranked_fallback_actions` replaces first-available fallback behavior with a more stable ranked fallback search.
3. observe/cooldown flow limits repeated proactive lane-change chasing on the same edge.
4. active vs passive pending handling stops healthy lane-now queueing from being miscounted as a stale failed decision.

Inference-alignment mitigation:
1. unified route application keeps training and inference on the same SUMO route-commit semantics.
2. `MAPPOController.should_control_vehicle(...)` now wakes inference on the same structural `forced` and `open` decision cases that training evaluates.
3. held-out frozen evaluation now measures deployment-style behavior directly during training.

These changes reduce fallback churn and tail instability, but they are still mitigations rather than a proof that loops are impossible.

## How to read the failure modes

### Pattern A: high loop events plus high fallback churn

Likely cause:
- the policy is repeatedly entering a locally safe but globally poor continuation under congestion.

Inspect together:
- `loop_events`
- `short_cycle_events`
- `aba_bounce_events`
- `dead_end_reentry_events`
- `fallback_overrides`
- `fallback_to_lane_feasible_now`
- `loop_after_fallback_rate`

### Pattern B: high pending timeout plus low completion

Likely cause:
- proactive decisions are being opened, deferred, or released too often without enough real progress.

Inspect together:
- `pending_decision_timeouts`
- `pending_release_route_no_progress_abort`
- `pending_release_route_stall_timeout`
- `deferred_lane_change_actions`
- `cooldown_replans_blocked`
- `fail_timeout`

### Pattern C: emergency-brake spike without matching collision/teleport spike

Likely cause:
- the controller is still finishing routes, but the local control sequence is creating stress near merges, leader interactions, or commit windows.

Inspect together:
- `emergency_brake_events`
- `emergency_brake_due_to_leader`
- `emergency_brake_due_to_congestion`
- `emergency_brake_near_junction`
- `emergency_brake_after_fallback`
- `emergency_brake_after_proactive`
- `emergency_brake_after_lane_now`

Important guardrail:
- treat emergency-brake metrics as stress indicators, not as a direct crash count.

## Training-vs-inference guardrail

Do not diagnose deployment quality from `rl_episode_metrics.csv` alone.
Late training episodes can look better than deployment because training still performs replay updates during the episode.
Use held-out frozen evaluation instead:
- `rl_frozen_eval_metrics.csv`
- best checkpoint `<model-output>.best.pt`
- best-checkpoint metadata `<model-output>.best.pt.meta.json`

If training looks strong but frozen eval is weak, the issue is generalization or rollout mismatch, not that inference should keep learning.

## Minimum metric set to inspect together

- `completion_rate`
- `avg_travel_time`
- `p50_travel_time`
- `p90_travel_time`
- `timeout_rate`
- `pending_decision_timeouts`
- `deferred_lane_change_actions`
- `loop_events`
- `dead_end_reentry_events`
- `fallback_to_lane_feasible_now`
- `loop_after_fallback_rate`
- `emergency_brake_events`
- `fail_timeout`
- `controlled_teleport_rate`

## Reporting template

When posting run analysis, include:
1. episode range or evaluation seeds,
2. whether the numbers come from training rollout or frozen eval,
3. the minimum metric set above,
4. one-sentence diagnosis,
5. the next tuning hypothesis.

## Bottom line

The current loop fix is not a single if-statement.
It is the combination of:
- safer action prefiltering,
- better ranked fallbacks,
- stricter pending lifecycle handling,
- and closer training/inference control alignment.

Judge success on lower travel time and higher completion without worsening the tail, timeout, and emergency-brake stress signals.
