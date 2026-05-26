# RL Pipeline Workflow and Guard Rails

This document is a practical guide to the RL routing pipeline.

It explains:
- what the pipeline is trying to do,
- how a vehicle moves through the control flow,
- how training and inference share logic,
- what each pending state means,
- which guard rails prevent unsafe or stale decisions,
- and how frozen evaluation now fits into the deployment workflow.

The goal is to make the pipeline easy to follow for someone reading the code for the first time while still keeping enough detail to support debugging and future changes.

## What the Pipeline Is Optimizing

The current RL objective is average travel-time minimization.
That means the system mainly tries to:
- get vehicles to their destinations,
- keep travel time low,
- avoid congestion-amplifying choices,
- and avoid pathological behaviors like loops, dead ends, route mismatch, or stale pending decisions.

Vehicle deadlines may still appear in data structures for backward compatibility, but they are not the main target of the current training objective.

## Where the Logic Lives

Main files:
- `core/STR_SUMO.py`: runtime loop that decides when controlled vehicles are handed to a controller.
- `core/junction_decision_engine.py`: lane feasibility, commit-window logic, route application, and progress checks.
- `core/shared_decision_policy.py`: shared decision classification and pending lifecycle logic.
- `core/rl_training_pipeline.py`: training controller plus held-out frozen evaluation and checkpoint selection.
- `controller/MAPPOController.py`: frozen inference controller.

Mental model:
- `STR_SUMO` decides when a vehicle is eligible for control.
- `junction_decision_engine` decides what is physically and topologically possible.
- `shared_decision_policy` decides how to interpret the decision lifecycle.
- the training or inference controller decides what action to take.

## End-to-end guard rails

### 1. Context feasibility guard

Implemented mainly in `build_context(...)`.
Protects against:
- impossible directions,
- impossible lane changes,
- too-late proactive maneuvers.

### 2. Forced / skip guard

Implemented mainly in `classify_decision(...)`.
Protects against:
- counting fake decisions,
- blaming the model when the road already forced the outcome.

### 3. Loop / trap guard

Implemented mainly through:
- `prefilter_action_for_loops`,
- ranked fallback selection,
- dead-end-reentry checks,
- recent-edge history.

Protects against:
- short cycles,
- local oscillation,
- dead-end reentry,
- fallback churn that keeps reopening the same bad neighborhood.

### 4. Pending lifecycle guard

Implemented mainly through observe pending, route pending, progress checks, release rules, and cooldown.
Protects against:
- stale same-edge commitments,
- repeated proactive retries,
- timing out healthy lane-now queueing as if it were a routing error.

### 5. Route-application guard

Training and inference now share the same route-application helpers.
This protects against:
- training on route decisions that SUMO never really accepted,
- route-mismatch drift between training and inference.

## Active vs passive pending

This is one of the most important stability ideas in the current codebase.

Active pending:
- observe-lane-change pending,
- proactive route pending that still needs same-edge monitoring.

Passive pending:
- a healthy lane-now commitment that is already correctly aligned.

Why it matters:
- active pending may legitimately timeout or abort,
- passive lane-now queueing should not be treated as a routing failure just because the vehicle is waiting at a light or in traffic.

## Loop mitigation summary

The loop fix is a layered mitigation:
1. reject obviously bad loop/trap actions before commitment,
2. choose safer ranked fallbacks when the primary action is not usable,
3. avoid immediate same-edge re-chasing with cooldown,
4. keep pending semantics truthful so lane-now queueing is not misread as a loop.

This improves tail stability without claiming loops are impossible.

## Hard-brake interpretation

Hard-brake telemetry is diagnostic, not the objective by itself.
Use it to answer questions like:
- is the policy causing local stress near the junction?
- did fallback recovery create a harsh merge?
- are proactive maneuvers getting committed too aggressively?

Read hard-brake metrics together with:
- completion rate,
- timeout rate,
- teleports,
- fallback metrics,
- and `p90` travel time.

A hard-brake spike does not automatically mean the policy is unusable, but it does mean the local maneuver quality is worse than desired.

## Training vs inference

### What training does that inference does not

Training:
- acts,
- stores transitions,
- runs replay updates,
- changes the model during the overall training process.

Inference:
- loads a checkpoint,
- chooses actions from that checkpoint,
- does not replay,
- does not update weights.

### Important consequence

A late training episode can look better than the same checkpoint behaves later in deployment because the training rollout is not the same thing as a fully frozen evaluation.

## Frozen evaluation workflow

To close that gap, `RLTrainingPipeline` now supports held-out frozen evaluation.

CLI flags:
- `--eval-every`
- `--eval-seeds`
- `--eval-spawn-interval`
- `--best-model-output`

What happens during frozen evaluation:
1. the current training checkpoint is saved to a temporary eval model,
2. held-out seeds are generated with the chosen evaluation spawn interval,
3. the real inference controller (`MAPPOController`) is run through `STR_SUMO`,
4. aggregate metrics are written to `rl_frozen_eval_metrics.csv`,
5. the best checkpoint is updated if the new frozen-eval score is better.

Checkpoint ranking priority:
1. completion rate,
2. timeout rate,
3. average travel time,
4. `p90` travel time,
5. tail completion gap,
6. `p95`/`p50` travel ratio,
7. deadline misses.

This is the deployment-quality metric path.

## Inference cadence alignment

`MAPPOController.should_control_vehicle(...)` now wakes the controller on the same structural `forced` and `open` decision cases that training evaluates, while still preserving edge-change and active-pending monitoring.

This reduces one of the main rollout mismatches that previously made `main.py` look much worse than training suggested.

## Practical debugging checklist

When a run looks unhealthy, check these in order:
1. Are travel time and completion good in frozen eval, not just training rollout?
2. Are pending timeouts and lane-change deferrals growing?
3. Are loop and fallback metrics rising together?
4. Are hard-brake signals clustering after fallback or proactive actions?
5. Is inference using matching `spawn_interval` and seed settings?

## Bottom line

The pipeline is designed to make fewer but more truthful routing decisions.
The most important current stability gains come from:
- safer loop-aware action filtering,
- stricter pending handling,
- aligned route application,
- closer training/inference wake-up semantics,
- and held-out frozen evaluation for checkpoint selection.
