# DQN Target Network: Use and Data Flow

This project uses two neural networks inside `DQNTrainer`:

- **online model** (`self.model`): updated every replay step.
- **target model** (`self.target_model`): a delayed snapshot of the online model.

The target model exists to stabilize Q-learning targets.

---

## Why a target model is used

If you bootstrap with the same model you are updating, target values move every gradient step.
That can cause oscillation/divergence because both sides of the Bellman update shift together.

A target model slows that movement:

1. Keep target weights fixed for several updates.
2. Train online model against those semi-stable targets.
3. Periodically copy online weights into target weights.

---

## Initialization flow

During trainer construction:

1. Build online model.
2. Build target model with identical architecture.
3. Copy online weights into target model once.
4. Set `target_update_every` and reset `train_steps`.

This means both models are equal at step 0.

---

## Replay/training flow (one minibatch)

For each replay minibatch:

1. Predict `q = online(states)`.
2. Predict `q_next = target(next_states)`.
3. Build Bellman targets per sample:
   - `target[action] = reward + (1-done) * gamma * max(q_next)`
4. Train online model on `(states, target)`.
5. Increment `train_steps`.
6. If `train_steps % target_update_every == 0`, copy online -> target.

So the online network learns continuously, while the target network changes only on sync steps.

---

## What this implementation is (and is not)

- ✅ **Implemented**: DQN with a target network (hard periodic updates).
- ❌ **Not yet implemented**: Double DQN.

Double DQN would pick the next action using the online model but evaluate that action using the target model.
Current logic uses `max_a Q_target(next_state, a)` directly.

---

## Main knobs you can tune

- `target_update_every`:
  - smaller: faster adaptation, potentially noisier targets;
  - larger: more stable targets, potentially slower learning.
- `gamma`: higher values weight long-term outcomes more.
- replay settings (`replay_warmup`, `batch_size`, `train_every`, `grad_steps`) affect update stability and sample efficiency.

---

## Quick troubleshooting checklist

If training is unstable:

- increase `target_update_every` (less frequent syncing),
- increase replay warmup,
- reduce reward magnitude extremes (already helped by reward clipping),
- monitor KPI: `completion_before_deadline` rather than only `avg_return`.

If learning is too slow:

- decrease `target_update_every`,
- increase `grad_steps` moderately,
- verify exploration schedule (`epsilon_decay`, `epsilon_min`) is not collapsing too early.
