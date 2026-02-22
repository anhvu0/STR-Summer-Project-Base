# STR SUMO RL Training Improvement Playbook

This project already tracks useful online metrics:
- `teleport_events/ep`
- `teleported_controlled/ep`
- `completion_before_deadline`
- `avg_return`

(Printed in `core/rl_training_pipeline.py`.)

## 1) Optimize for the right objective first

Treat **`completion_before_deadline`** as the primary KPI and `avg_return` as a secondary signal.

Why: reward can improve without true task success if reward shaping overweights terms that do not directly increase on-time arrivals.

## 2) Run controlled A/B sweeps (one variable at a time)

The highest-impact knobs in this codebase:

- Exploration schedule in `DQNTrainer` (`epsilon_decay`, `epsilon_min`).
- Q-learning discount (`gamma`).
- Replay behavior (`batch_size`, replay frequency).
- Reward shaping constants (`destination_reward`, `deadline_penalty`, progress/loop penalties).
- Teleport terminal penalty (currently hardcoded to `-200.0`).

Recommended sweep order:
1. `epsilon_decay`: try `0.995`, `0.997`, `0.999`.
2. `epsilon_min`: try `0.10`, `0.15`.
3. `gamma`: try `0.97` and `0.99`.
4. Teleport penalty: try `-120`, `-150`, `-200`.
5. `TRAIN_EVERY` / `GRAD_STEPS`: try `(5,2)` and `(10,2)`.

## 3) Improve metric stability before changing architecture

The run is noisy. Use robust comparison rules:

- Compare means over fixed windows (e.g., last 100 episodes).
- Report at least 3 random seeds.
- Select config by highest `completion_before_deadline`, then lowest `teleported_controlled/ep`, then best `avg_return`.

## 4) Reward-shaping guidance for this implementation

Current shaping includes:
- travel-time and congestion penalties,
- progress reward,
- loop/dead-end penalties,
- deadline penalty,
- destination bonus,
- hard teleport penalty.

### FAQ: what happens if a vehicle takes a lane/route that cannot reach the destination?

Yes—this implementation penalizes that behavior so the policy can learn to avoid it over time.

- If a vehicle moves onto an edge where the destination becomes unreachable (`curr_distance = inf`), reward gets an additional `-100` and the transition ends.
- If the vehicle enters a true dead-end (no outgoing edges and not at destination), it gets another terminal `-50` dead-end penalty.
- If the vehicle teleports (often a symptom of bad routing/gridlock), the trainer stores a terminal transition with `teleport_penalty` (default `-150`).

Together, those terminal negatives push Q-values down for actions that lead into disconnected or trapping regions.

Common failure mode: huge negative events dominate learning and hide incremental progress.

Practical adjustments to test:
- Reduce teleport penalty magnitude slightly (`-200 -> -120/-150`) if policy becomes too risk-averse or unstable.
- Increase positive signal for deadline-respecting arrivals (destination reward or deadline bonus).
- Keep loop/dead-end penalties, but avoid stacking too many strong negatives at once.

## 5) Training-system hygiene

- Start updates only after replay buffer warmup (e.g., 2k-5k transitions).
- Use a target-network update cadence if not already present.
- Consider gradient clipping to reduce instability from rare high-magnitude transitions.
- Save/evaluate checkpoints every N episodes and stop early if KPI plateaus.

## 6) Interpreting your current logs

If `avg_return` improves while `completion_before_deadline` stays flat, you are likely optimizing shaping terms more than mission success. In that case, prioritize reward edits and KPI-driven selection over pure return improvements.
