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
