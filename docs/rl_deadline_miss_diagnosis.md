# Why Dijkstra currently beats RL on missed deadlines

This note ties directly to the current implementation.

## Why Dijkstra is stronger right now

Dijkstra computes a deterministic shortest path using known edge lengths at decision time.
In this codebase, that means every decision is destination-directed and globally consistent for the current map graph.

By contrast, the RL policy must learn this behavior from sparse/indirect reward and noisy transitions,
so it can still oscillate or choose actions that need to be overridden.

## Core issues in the RL setup

### 1) Reward objective mismatch for deadline success

- Deadline failure is mostly a terminal cliff (`reward -= deadline_penalty` when `step > deadline`) instead of a dense signal.
- There is a near-deadline shaping term, but its magnitude is small compared with other terms.
- The reward mixes several goals (time, local congestion, system congestion, progress, loop penalties), so maximizing return can diverge from maximizing on-time arrivals.

Net effect: the agent can improve average return without reliably improving `completion_before_deadline`.

### 2) Very large negative events can dominate learning

The training loop stacks strong negatives for bad trajectories:
- missed deadline,
- no-path transitions,
- dead-ends,
- teleports.

When large terminal negatives dominate replay samples, value learning becomes unstable and the policy may become over-averse or brittle around branching points.

### 3) Policy/execution mismatch (training vs inference-time behavior)

Inference contains explicit override logic that can replace model actions using shortest-path heuristics,
loop checks, and feasibility checks. This means the deployed behavior is partly rule-based rather than purely learned.

If a policy often needs override, its learned Q-values are not robust enough by themselves; this often correlates with occasional failures and deadline misses in edge cases.

### 4) State encoding likely limits generalization

The state includes raw edge indices as scalar numeric features and a full density vector. Raw indices impose arbitrary ordinal structure,
which neural nets can overfit to instead of learning transferable topology semantics.

This can hurt policy stability on less frequent intersections and contribute to local cycling.

### 5) Decision timing and credit assignment remain hard

Decisions are only made near junctions and transitions are closed later at subsequent decision points.
This is generally sensible, but it increases delayed credit assignment complexity when combined with long horizons and traffic interactions.

## Practical fixes most likely to reduce deadline misses

1. Make deadline urgency denser and more dominant in per-step shaping (not just terminal).
2. Rebalance penalty magnitudes so teleport/dead-end/no-path events do not swamp progress learning.
3. Reduce dependence on inference overrides; treat high override ratio as a training failure metric.
4. Improve state representation (topology-aware features/embeddings instead of raw edge IDs).
5. Tune with KPI-first selection: optimize `completion_before_deadline` before `avg_return`.
