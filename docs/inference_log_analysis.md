# Inference Log Analysis: Dijkstra vs Q-Learning (main.py)

## What the log shows

The log appears to include two back-to-back inference runs:

1. **Dijkstra route controller** summary (first block).
2. **Q-Learning route controller** summary (second block, beginning at `Testing Q Learning Route Controller`).

## High-level comparison

### Dijkstra (first run)
- Vehicles reached destination: **10/10**.
- Average timespan: **70.7**.
- Deadlines missed: **0**.
- Per-vehicle completion lines indicate all vehicles `40..49` finished successfully.

### Q-Learning (second run)
- Vehicles reached destination: **9/10**.
- Average timespan: **75.22222222222223**.
- Deadlines missed: **0**.
- `Vehicle 49 reaches the destination: False` indicates one failure.

## Performance interpretation

Compared with Dijkstra, Q-Learning is currently:
- **Less reliable** on this scenario (90% vs 100% success).
- **Slightly slower** for successful trips on average (75.22 vs 70.7).

This suggests the Q policy is not yet stable/robust enough for inference-only deployment in this map/traffic setup.

## Behavioral diagnosis from route-choice traces

The Q-Learning section repeatedly logs `[OVERRIDE] ... chosen=... best_dir=...` immediately followed by
`Choice ... is: <best_dir>`.

This indicates the inference pipeline includes a **distance-based safety override** that corrects poor policy actions.

Observed issues:
- **Oscillation / ping-pong behavior** at local edge pairs, especially:
  - Vehicle 49 around `44884821#2` and `-44884821#2`, with visit counts climbing to 11.
  - Vehicle 44 around `597602756#6/#7/#8` and reverse edges.
  - Vehicle 46 around `597602756#5/#6/#7/#8` and reverse edges.
- Frequent overrides where proposed action increases distance (`prop_d`) while `best_dir` has lower distance (`best_d`).

These patterns are consistent with either:
- insufficient state representation (cannot disambiguate near-symmetric local states),
- Q-values not fully converged in these intersections,
- or action-value ties/noise causing unstable turn decisions.

## Why Vehicle 49 failed

Vehicle 49 shows repeated loop-like transitions with increasing `visit=` counts in the same local area before final failure line.
This strongly indicates the controller got trapped in a local oscillation and did not make net progress to destination.

## GPU/NUMA messages

TensorFlow NUMA warnings are informational in this environment:
- `could not open file to read NUMA node`
- `defaulting to 0`

The model still initializes on GPU (`Created device ... NVIDIA GeForce RTX 3070`), so these messages are likely not the cause of routing quality problems.

## Recommended next steps

1. **Add anti-loop penalties** during training and/or inference:
   - penalize revisits to recent edges,
   - cap repeated traversals in a sliding window.
2. **Strengthen progress-based shaping**:
   - reward reduction in shortest-path distance to destination each step,
   - penalize action when `prop_d > best_d`.
3. **Use deterministic tie-breaking** for near-equal Q-values to avoid oscillation.
4. **Keep override telemetry** and add per-vehicle counters (override ratio, unique edges visited, loop score).
5. **Regression target** for this map: recover at least Dijkstra-level reliability (10/10) before optimizing travel time.

## Bottom line

From this log alone, Dijkstra is currently the better inference controller for this scenario.
Q-Learning is close on average delay but still exhibits local cycling and one hard failure, which is unacceptable for parity with deterministic shortest-path routing.
