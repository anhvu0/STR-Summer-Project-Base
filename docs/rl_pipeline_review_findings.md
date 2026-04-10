# RL/SUMO Pipeline Review Notes

This document captures a senior-level review of the DQN routing pipeline in `core/rl_training_pipeline.py` and related routing/deadline utilities.

Key findings include:
- DQN bootstrapping instability (no target network, max-over-same-network targets).
- Policy-execution mismatch where heuristics partially override RL actions.
- Weak selfless objective alignment in reward shaping.
- Terminal-transition leakage when episodes end with open transitions.
- Lane-feasibility and action masking mismatch around decision points.
- Deadline semantics mismatch between scheduled depart and realized depart.

See the assistant response for the fully ranked list with fixes.
