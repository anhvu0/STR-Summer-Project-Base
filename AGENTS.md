# Project Agent Notes


## RL objective context (updated April 2026)

- The RL training objective in `core/rl_training_pipeline.py` is average travel-time minimization, not deadline-feasibility maximization.
- Reward priorities are:
  1. per-step travel-time cost (dominant),
  2. completion / arrival reward,
  3. congestion and progress shaping,
  4. safety / control penalties (loops, route mismatch, failed route application, unreachable transitions, teleports).
- Episode metrics are travel-time centric:
  - `completion_rate`
  - `avg_travel_time`
  - `p50_travel_time`
  - `p90_travel_time`
  - plus safety and decision telemetry.

## Compatibility note

- Vehicle deadlines are still present in generated vehicle objects for backward compatibility with old scenarios, but training logic should not prioritize on-time-arrival metrics unless explicitly reintroduced.

## Routing stability context (updated April 21, 2026)

- Unified route application fixed the earlier route-mismatch failure mode across training and inference.
- The remaining dominant failure mode is tail collapse from overlong pending decisions, repeated lane-change deferrals, and fallback churn under congestion.
- Recent stability work includes:
  - pre-commit loop / trap filtering (`prefilter_action_for_loops`)
  - ranked fallback selection (`ranked_fallback_actions`)
  - observe / cooldown flow for proactive lane changes
  - active vs passive pending handling for truthful same-edge monitoring
  - inference wake-up alignment on structural `forced` and `open` decisions in `controller/MAPPOController.py`

### Key telemetry to track

- `pending_decision_timeouts`
- `fallback_to_lane_feasible_now`
- `mean_pending_age`
- `deferred_lane_change_actions`
- `decision_pending_at_episode_end`
- `fail_timeout`
- `loop_after_fallback_rate`
- `emergency_brake_events`

## Frozen evaluation context (updated April 21, 2026)

- `rl_episode_metrics.csv` is a training-rollout log, not a pure frozen deployment metric.
- Training episodes are now fixed-policy MAPPO rollouts with updates applied after each episode, so they are cleaner than the older replay-updated rollouts but still not a substitute for held-out frozen inference.
- `core/rl_training_pipeline.py` now supports held-out frozen evaluation with:
  - `eval_every`
  - `frozen_eval_seeds`
  - `eval_spawn_interval`
  - `best_model_output_path`
- Deployment-quality outputs are:
  - `rl_frozen_eval_metrics.csv`
- `<model-output>.best.pt`
- `<model-output>.best.pt.meta.json`
- Best-checkpoint ranking priority is:
  1. completion rate,
  2. timeout rate,
  3. average travel time,
  4. `p90` travel time,
  5. tail completion gap,
  6. `p95`/`p50` travel ratio,
  7. deadline misses.

## Hard-brake interpretation context

- `emergency_brake_*` metrics are heuristic stress signals, not collision counts.
- Read them together with completion, timeout, teleport, and fallback metrics before concluding the policy is unsafe.
- Attribution splits (`due_to_leader`, `due_to_congestion`, `near_junction`, `after_fallback`, `after_proactive`, `after_lane_now`) are intended to localize maneuver quality problems.

## Files changed in this pass

- `core/rl_training_pipeline.py`
- `core/STR_SUMO.py`
- `controller/MAPPOController.py`
- `train_rl.py`
- `main.py`
- `README.md`
- `docs/loop_deadend_tail_analysis.md`
- `docs/telemetry_metrics_reference.md`
- `docs/inference_log_analysis.md`
- `docs/rl_pipeline_workflow_and_guards.md`
- `AGENTS.md`

## Next things to inspect

- Whether held-out frozen eval tracks improvement consistently across rolling windows.
- Whether `lane_change_defer_limit` and `pending_timeout_steps` still need per-network tuning.
- Whether hard-brake stress falls when loop / fallback churn falls, or whether those now need separate tuning.
