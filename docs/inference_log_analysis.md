# Inference Quality Analysis and Fix Summary

Updated: April 21, 2026

## Problem

The late rows in `rl_episode_metrics.csv` can look strong, but `main.py` inference can still perform much worse:
- more teleports,
- more hard braking,
- more timeouts at step cap,
- worse average and tail travel time.

## Root cause

The main issue was not that inference should keep learning.
The issue was that training and deployment were being measured under different conditions.

### 1. Training CSV was optimistic

`rl_episode_metrics.csv` comes from training rollouts.
During those episodes the training loop is still performing replay updates, so the policy is not fully frozen while the episode is being measured.
That means late training rows can look better than the saved checkpoint behaves when reused later in deployment.

### 2. Inference rollout cadence was more restrictive

Training evaluates decision opportunities aggressively over the controlled set each step.
Inference used a stricter wake-up rule and could skip some structural decision points that training would still process.
That gap made deployment behavior worse even when the model weights themselves matched.

### 3. Traffic generation had to be matched

If inference uses a different `spawn_interval` or seed regime than training, congestion severity changes and the policy may be judged on a harder distribution than the one it was trained on.

## Fixes now reflected in the repo

### Inference-side parity fixes

`main.py`
- added `--spawn-interval`
- added `--seed`

This lets inference reproduce the same vehicle-generation settings used in training.

`controller/QLearningController.py`
- `should_control_vehicle(...)` now wakes on the same structural `forced` and `open` decision cases that training evaluates.
- active pending and edge-change handling remain intact.

This reduces training-vs-inference decision-cadence mismatch.

### Training-side deployment-quality fixes

`core/rl_training_pipeline.py`
- added configurable held-out frozen evaluation (`eval_every`, `eval_seeds`, `eval_spawn_interval`),
- added `rl_frozen_eval_metrics.csv`,
- added best-checkpoint saving to `<model-output>.best.h5`,
- added metadata export to `<model-output>.best.h5.meta.json`.

Frozen evaluation runs the current checkpoint without online learning and scores it on held-out seeds using the real inference controller.

## What to trust now

Use these outputs for different questions:
- `rl_episode_metrics.csv`: "Is training improving?"
- `rl_frozen_eval_metrics.csv`: "How good is the saved checkpoint when deployed?"
- `<model-output>.best.h5`: "Which checkpoint should I actually use for inference?"

## Recommended workflow

1. Train with held-out frozen evaluation enabled.
2. Match inference `spawn_interval` and `seed` to the scenario you want to compare.
3. Choose the best held-out checkpoint, not automatically the final checkpoint.
4. Judge success by completion, timeout rate, average travel time, and `p90` travel time together.

## Bottom line

Inference does not learn online in the normal deployment path, and it should not be expected to.
The correct way to make inference better is to train better, evaluate checkpoints in a truly frozen way, and deploy the checkpoint that already works well under frozen held-out evaluation.
