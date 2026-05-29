# Selfless Routing: Problem, Diagnosis, Fix, and the Open Split

_Last updated: 2026-05-28._

This document records the problem we are trying to solve, why the original setup
could not solve it, what was changed, what the changes achieved, and — most
importantly — **the unresolved split** that the next round of work has to choose
between. It is written so a future contributor (or reviewer) can pick up the
thread without re-deriving everything.

---

## 1. The problem we are trying to solve

**Research goal:** train a MAPPO routing policy in which controlled vehicles
*sometimes accept a personal ETA sacrifice in order to reduce congestion for the
whole fleet* ("selfless routing"). The fleet operates on a SUMO NYC grid.

This is a **social-dilemma / system-optimal routing** problem. The interesting
behavior is a vehicle choosing a route that is slower for itself because doing so
lowers total fleet travel time (relieves a shared bottleneck).

**Symptom that started the investigation:** the trained policy collapses to
ordinary shortest-path routing. In held-out evaluation the deployed policy chose
a non-shortest route only ~0.9% of the time and beat a Dijkstra baseline by ~1.4s
— i.e. it learned to be selfish, and the "selfless" behavior seen early in
training (~40%) was just exploration noise that annealed away.

---

## 2. Diagnosis — why the original setup could not produce selflessness

The collapse is **not** primarily a tuning bug. It is the policy behaving
*rationally* given the reward and the traffic regime. Each cause below was
verified by running instrumented episodes, not just by reading code.

### 2.1 The regime had essentially no congestion (the binding constraint)

**Summary:** the network is large relative to the demand. A few hundred vehicles
with dispersed origins and destinations occupy only a small fraction of road
capacity, so congestion rarely forms. With no congestion, selfish routing imposes
no cost on the rest of the fleet, and a selfless policy has no benefit available
to capture. ("Price of anarchy ≈ 1" states that selfish behavior is already close
to the best achievable coordinated outcome.)

Technical detail:
- Network: 132 passenger edges / 360 lanes / 27,236 lane-meters ≈ **3,890-vehicle**
  bumper-to-bumper capacity.
- Demand: 150 controlled + 150 background = 300 vehicles, dispersed origins *and*
  destinations (`target_pattern=3`). Even if all were present at once that is
  ~1.1 veh/100m/lane — **3–8% of capacity**. Teleports stayed at 0 even at 4×
  spawn rate.
- Consequence: the **price of anarchy ≈ 1** — selfish routing is already almost
  system-optimal, so there is nothing for a selfless policy to gain. No reward
  change can manufacture a benefit that the traffic does not contain.

### 2.2 The selfless signal was the wrong quantity, in the wrong units

**Summary:** the reward defined selflessness using the wrong quantity. It rewarded
an abstract "congestion relief" value (a unit-less density difference), whereas the
cost of detouring is measured in seconds of additional travel. Combining the two
requires an arbitrary conversion coefficient with no principled value, so the
trade-off was never well-defined. The quantity that should have been used — the
travel time saved for other vehicles, in seconds — was never computed.

Technical detail:
- The reward rewarded **dimensionless "density relief"** (~0.01–0.28) and traded
  it against an **ETA sacrifice measured in seconds**. Bridging the two requires
  an arbitrary coefficient (`route_balance_reward_scale`), and there is no
  principled value for it.
- The commensurate quantity — *fleet travel-time saved for others, in seconds* —
  was never measured.

### 2.3 Credit assignment buried even the signal that existed

**Summary:** the signal indicating whether a route choice was good was dominated by
noise. Each route decision is followed by a stretch of driving whose entire travel
cost was attributed to that single decision, even though most of the variation
comes from traffic the policy cannot control or predict rather than from the route
choice itself. The selfless component of the signal was roughly a quarter of that
noise, so it was difficult to learn. Separately, an existing penalty increases
whenever a trip runs long — the direct result of a detour — so the reward
penalized selflessness. Finally, each vehicle was rewarded only for its own trip,
which drives the policy toward the selfish equilibrium; the centralized critic
improves training stability but does not change the objective being optimized.

Technical detail:
- A macro-route decision is followed by ~5 edges (~72 steps) of driving, and the
  realized per-step travel cost of that window (mean −13, **std 8.75**) was bundled
  into one policy transition. The selfless signal's std was ~2.26 — **25% of the
  total** — and 97% of windows were cost-dominated. The cost variance comes from
  exogenous traffic the critic cannot predict from the decision state.
- The per-step cost decomposes to: base time 0.070/step (44%), **tail-delay
  0.049/step (31%)**, congestion-marginal 0.022 (14%), externality 0.013 (8%),
  density 0.004 (3%). The "refund" that was supposed to make selfless detours
  cost-neutral only covered the 0.070 base term.
- The **tail-delay penalty** ([`tail_delay_linear_penalty`](../core/rl_training_pipeline.py#L225))
  is *structurally anti-selfless*: it escalates when a vehicle exceeds ~1.35× its
  nominal ETA, which is exactly what a detour causes.
- Rewards were **purely individual** → the learned equilibrium is the selfish
  (Wardrop) user-equilibrium. The centralized critic gives the MAPPO *architecture*
  but not a cooperative *objective*; it only reduces variance.

### 2.4 Proof that tuning is not enough

**Summary:** re-tuning reward coefficients alone does not resolve the problem.
Applying every calibration fix to the original setup simultaneously still produced
a collapse to shortest-path routing. The limiting factors are the two structural
issues above — the absence of congestion and the purely individual objective —
rather than the parameter values. These are addressed in §3.

Technical detail:
A/B over 18 episodes in the light regime: applying every calibration fix at once
(full-rate refund + tail-delay penalty zeroed + 4× relief boost + entropy
annealing off) only moved the non-baseline rate from 1.8% to 7.4% — still a
collapse. A clean structural fix (decoupled credit + relief weighted 18×) also
collapsed (→3.6%), because at that weight only the top ~decile of detours are
net-positive, and those are rare in light traffic. **Conclusion: the regime and
the objective are the binding problems, not calibration.**

> Architecture note: the actor (per-candidate scorer) and critic (centralized)
> are **sound** — they were not changed. The fixes are in the regime, the
> objective, and (optionally) the credit path.

---

## 3. What was implemented

Two changes in [core/rl_training_pipeline.py](../core/rl_training_pipeline.py),
both **non-breaking** (defaults reproduce the legacy behavior exactly), plus CLI
wiring in [train_rl.py](../train_rl.py).

### 3.1 Regime fix — make congestion exist

**Summary:** the original experiment placed too few vehicles on a large network,
so congestion rarely formed, and without congestion there is no benefit to selfless
routing. The change directs many vehicles toward a **shared destination**,
concentrating flow into a corridor that congests. The vehicle count is tuned to
produce sustained congestion without full gridlock — excessive demand locks the
network and prevents vehicles from completing their trips.

Technical detail:
- Parametrized `num_target_vehicles` / `num_random_vehicles` / `target_pattern`
  in the pipeline constructor; used by
  [`generate_episode_vehicles`](../core/rl_training_pipeline.py#L2314).
- The selfless-routing study uses **`target_pattern=2`** (ranged origins → one
  shared destination), which concentrates flow into a corridor and creates real
  congestion on the *existing* network — no new map needed.
- Calibrated load (probe results):

  | demand | p95 edge density | teleports | completion | avg TT |
  |---|---|---|---|---|
  | pattern 3, 150+150 (legacy) | 2.5 | 0 | 100% | 265s |
  | **pattern 2, 350+150, spawn 0.5** | **7.0** | 0 | 100% | ~318s |
  | pattern 2, 500+200, spawn 0.5 | 13.3 | 84 | 90% | 1161s (gridlock) |

  → pattern 2 at ~350 controlled vehicles is the clean dilemma; ~500 over-saturates.

### 3.2 Objective fix — a shared fleet-delay reward in time units

**Summary:** originally each vehicle was evaluated only on its own trip, so it had
no incentive to help others, and the selfless term was a small, ad-hoc bonus. The
change adds a **shared cost reflecting the current fleet-wide congestion** that
every vehicle pays a share of. That cost is measured in **seconds of delay** — the
same unit as the vehicle's own travel time — so when a vehicle's choice reduces
congestion, every vehicle's reward (including its own) improves. A single
parameter, `team_reward_alpha`, sets how much the fleet is weighted: **0 = own
trip only (selfish), 1 = fleet delay counted equally with own delay**. Because all
terms are in seconds, the policy can weigh additional personal delay directly
against travel time saved for the fleet, with no arbitrary conversion factor, which
makes accepting a sacrifice rational only when it is a net benefit.

Technical detail:
- `team_reward_alpha` ∈ [0,1] (constructor + `--team-reward-alpha`):
  [set here](../core/rl_training_pipeline.py#L149). 0 = legacy individual
  objective; >0 internalizes a share of fleet congestion into every agent's reward.
- Each step caches a **fleet-delay rate** = mean normalized slowness of the live
  controlled fleet (`1 - speed/v_norm`, free-flow→0, gridlock→1):
  [computed here](../core/rl_training_pipeline.py#L2964).
- [`_team_congestion_cost`](../core/rl_training_pipeline.py#L1857) applies
  `alpha · team_reward_scale · fleet_delay_rate · elapsed` and is subtracted in
  both reward paths ([compute_reward](../core/rl_training_pipeline.py#L1926),
  [compute_pending_step_reward](../core/rl_training_pipeline.py#L2013)).
- Because the same global signal is paid by every agent and *drops when any agent
  relieves the jam*, parameter-shared MAPPO can learn cooperative routing from it.
  It is in travel-time units, so the ETA-vs-fleet tradeoff needs no arbitrary
  cross-unit coefficient.

### 3.3 Reproduce the selfless setup from the CLI
```bash
python train_rl.py --target-pattern 2 --num-target-vehicles 350 \
    --num-random-vehicles 150 --spawn-interval 0.5 \
    --team-reward-alpha 1.0 --episodes 60 --eval-every 20
```

---

## 4. Results

### 4.0 The three arms (A / B / C), explained

All three arms train the **same** MAPPO policy on the **same** congested regime
(pattern 2, 350+150 vehicles). They differ **only in the reward objective** — what
each vehicle is rewarded for. The progression runs from a purely self-interested
objective to a strongly team-oriented one:

- **Arm A — selfish (control), `team_reward_alpha = 0`.**
  Each vehicle is rewarded only for its *own* outcome (own travel time, progress,
  completion). This is the legacy objective and the experimental control: it
  characterizes fleet behavior under a purely self-interested objective. The
  expected outcome is the selfish (user) equilibrium, in which each vehicle takes
  whatever route is fastest for itself.

- **Arm B — team-oriented (the primary change), `team_reward_alpha = 1`.**
  In addition to its own outcome, every vehicle pays a share of a **fleet-wide
  congestion cost** measured in travel-time units (the `_team_congestion_cost`
  term, §3.2). When a vehicle's choice reduces shared congestion, every vehicle's
  reward improves, giving the policy an incentive to relieve congestion rather than
  only minimize its own time. Nothing else changes from Arm A, so any difference is
  attributable to this term.

- **Arm C — team-oriented with full authority.**
  Arm B, with the components of the *legacy* reward that oppose selflessness
  removed and the team term strengthened:
  1. **Stronger fleet term** (`team_reward_scale` 0.12 → **0.30**): the shared
     congestion cost carries more weight, strengthening the team incentive.
  2. **Anti-selfless penalty removed** (`--disable-tail-delay-penalty`): the legacy
     "tail-delay" penalty increases whenever a vehicle's trip runs long — the direct
     result of a helpful detour — so it penalized selflessness; it is set to zero.
  3. **Legacy proxy removed** (`--disable-route-balance`): an older, ad-hoc
     "congestion relief" bonus is disabled so the principled team term is the sole
     driver of selfless behavior, avoiding confounding signals.

  In summary: **A = selfish baseline; B = add a clean team incentive; C = give that
  incentive full authority by also removing the legacy terms that opposed it.** The
  A→B→C progression varies how strongly, and how cleanly, the objective values the
  fleet relative to the individual.

**Metrics used in the tables below:**
- **fleet avg TT** — average travel time across all vehicles in the episode
  (lower = the whole fleet gets where it's going faster). This is the headline
  fleet-welfare number.
- **eta-sacrifice** — how many seconds *slower* than the shortest available route
  the vehicles' chosen routes are, on average. This is the direct measure of
  "selflessness": >0 means vehicles are accepting personal delay.
- **non-baseline route rate** — fraction of route decisions where the vehicle
  picked something *other* than the shortest route (a proxy for "is it detouring
  at all").
- **completion** — fraction of vehicles that reached their destination (a sanity
  check that selflessness isn't achieved by stranding vehicles).

### 4.1 Training rollouts (stochastic policy, congested regime)
The regime fix alone restored meaningful route choice: even at α=0 the policy
keeps ~39% non-baseline routing (vs ~2% in the light regime — no collapse).

Paired 18-episode test (same seed/demand per episode), converged (last-8) means:

| arm | objective | fleet avg TT | eta-sacrifice | non-baseline | completion |
|---|---|---|---|---|---|
| A | individual (α=0) | 541s | 16.2s | 30% | 99.5% |
| B | team (α=1) | 485s | 16.8s | 31% | 99.7% |
| C | team (α=1) + full authority¹ | **465s** | **19.3s** | **35%** | 99.8% |

¹ stronger fleet term (scale 0.30), old hand-crafted proxy off, tail-delay off.

- Selfless behavior *emerged*: more self-sacrifice and route diversity as the
  objective shifts individual→team.
- It *helped the fleet*, concentrated in the congested episodes (e.g. one
  near-gridlock episode: 1661s → 1065s, −36%); neutral on light episodes — correct.
- Effect is monotonic in α/authority; B (moderate) is more consistent than C.

### 4.2 Frozen held-out evaluation (deterministic policy, 8 unseen seeds 7000–7007)

| metric | shortest-path (Dijkstra) | A: α=0 | B: α=1 |
|---|---|---|---|
| fleet avg travel time | 423.4s | 405.4s | **391.5s** |
| vs Dijkstra | — | −18.0s | **−31.9s** |
| fleet p90 | — | 760.7s | **743.8s** |
| win-rate vs Dijkstra | — | 0.875 | 0.875 |
| completion | — | 100% | 100% |
| eta-sacrifice (deterministic) | — | 0.5s | **0.4s** |
| non-baseline route rate | — | 3.9% | 3.4% |

- The team-reward policy generalizes: **−3.4% fleet TT vs the individual control,
  −7.5% vs shortest-path**, held out, 100% completion, better p90. The clean
  attribution is B vs A (identical except α). *(Caveat: see §4.3 — at 16 seeds this
  −3.4% is inside the paired noise band and does not replicate.)*

### 4.3 Higher-power eval: arm C (full authority) vs control, 16 seeds + 95% CIs

**Summary:** the strongest selfless configuration was evaluated across 16 held-out
traffic scenarios, with averages reported with 95% confidence intervals. The
result has two parts. First, the stronger objective made the policy selfless under
greedy (deterministic) deployment, not only during training: route choices carry
about 14s of self-imposed delay versus about 6s for the selfish control, and the
non-baseline route rate roughly doubles. The degree of selflessness is therefore a
controllable setting. Second, the additional selflessness produced no measurable
improvement in fleet travel time: the strongly-selfless and selfish policies
finished in essentially the same total time, both about 9.5% faster than naive
shortest-path routing. The reason is that the selfish policy already performs
congestion-aware routing, and on a grid with many parallel paths that is close to
optimal. The team-reward setting is thus an effective control over how much
vehicles sacrifice, but on this network that sacrifice does not reduce fleet
travel time.

Technical detail — the (a) decision fork (§5.3) run: arm C vs the α=0 control,
**deterministic** (greedy) deployment, 16 held-out seeds, shared Dijkstra
baseline, 95% CIs.

| metric (greedy, n=16) | Dijkstra | A: α=0 (selfish) | C: full authority |
|---|---|---|---|
| fleet avg TT (s) | 442.3 ± 85.4 | **399.8 ± 82.2** | **400.5 ± 82.1** |
| eta-sacrifice (s) | — | 5.8 ± 1.6 | **14.1 ± 3.0** |
| non-baseline rate | — | 0.13 ± 0.03 | **0.27 ± 0.05** |
| completion | — | 100% | 100% |
| fleet TT vs Dijkstra | — | −9.6% | −9.5% |

Paired (C − A, same demand per seed): **+0.7 ± 13.1 s** (C faster on only 6/16
seeds). Two clear, and partly opposing, conclusions:

1. **The stronger objective DID make the *greedy* policy genuinely selfless.**
   ETA-sacrifice 5.8s → 14.1s and non-baseline rate 13% → 27%, both with
   non-overlapping CIs. So **option (a) succeeded at closing the train-vs-deploy
   gap** — the deployed argmax policy now visibly sacrifices, not just the sampled
   one. (The earlier §4.2 greedy numbers of ~0.4s/3.4% were a weaker-objective,
   smaller-sample run; with arm C the greedy policy is selfless.)
2. **But the extra selflessness produced NO additional fleet benefit.** Arm C
   (14s sacrifice) and arm A (5.8s) have statistically identical fleet travel time
   (paired Δ ≈ 0 ± 13s ≈ ±3%). Both beat naive Dijkstra by ~9.5% — and that ~9.5%
   is captured by *congestion-aware* routing that even the **selfish** α=0 policy
   learns. Arm C's additional detouring is real sacrifice that the fleet does not
   benefit from.

**Why (the deeper finding):** in a dense grid, "selfish-but-congestion-aware"
routing already approximates the system optimum — there are many near-equal
alternative paths, so self-interested load-balancing spreads traffic well on its
own. The price of anarchy stays low *even after we created congestion* (§2.1), so
selflessness has almost no margin to improve on congestion-aware selfish routing.
The team reward is an effective, controllable **parameter governing how much
vehicles sacrifice**, but on this network that sacrifice is fleet-neutral rather
than fleet-improving.

---

## 5. The split (the key open issue)

> ### 5.0 Update — what the 16-seed (a) run resolved
> **Summary:** the policy can be made reliably selfless, but on this network
> selfless routing does not improve fleet travel time relative to a competent
> selfish policy — once vehicles route around congestion on their own, little
> further gain remains. The next step is therefore not a further policy change but
> evaluation on a network with a high price of anarchy, where the selfish
> equilibrium is known to be substantially worse than the optimum. Detail below.
>
> The decision fork below (§5.3) was run as **option (a)**, and the result (§4.3)
> reframes the whole split:
> - **The behavioral gap is closed.** A stronger objective (arm C) makes the
>   *greedy/deployed* policy genuinely selfless (14s sacrifice, 27% detours, CIs
>   clear of the control). Making the deterministic policy selfless is therefore a
>   solved, controllable problem, governed by `team_reward_alpha` together with
>   removing the anti-selfless terms.
> - **The efficiency claim does not hold on this network.** The additional
>   selflessness yields **no** fleet-travel-time improvement over a competent
>   *selfish* policy (paired
>   Δ ≈ 0 ± 3%); both already sit ~9.5% under naive shortest-path. On a dense grid,
>   selfish congestion-aware routing ≈ system optimum (low price of anarchy), so
>   there is almost nothing left for selflessness to win.
> - **Therefore the real open question is the regime, not the policy.** To show
>   selflessness *adds value*, the network must have a high price of anarchy — a
>   Braess/Pigou-style bottleneck where selfish equilibrium is provably far from
>   optimal. This is the original "controlled bottleneck" recommendation, now
>   strongly motivated by data. §§5.1–5.3 below are the pre-run framing, kept for
>   context.

The frozen result exposes a **gap between how the policy behaves when sampled vs.
when greedy**, and that gap forces a decision.

### 5.1 The behavioral split: stochastic training vs. deterministic deployment
- **Sampled (training):** ~30% non-baseline routes, 16–19s ETA sacrifice —
  visibly selfless.
- **Greedy/argmax (deployment):** ~3.4% non-baseline, ~0.4s sacrifice — almost
  pure shortest-path.

The team reward shifted *which* route wins at the margin (enough for the −3.4%
fleet gain), but the greedy policy is **not** a bold detourer. The visible
selflessness lives mostly in the exploration distribution, not the deployed
argmax policy.

### 5.2 The interpretation split: two different "successes"
- **As a fleet-efficiency result:** ✅ solid and defensible. A cooperative
  objective measurably and reproducibly beats both the selfish policy and
  shortest-path on held-out traffic, with no completion cost.
- **As a "watch vehicles sacrifice ETA" demonstration:** ⚠️ muted at deployment.
  The gain comes from subtle, well-targeted adjustments, not dramatic detours.

Which one is "the answer" depends on what the project is ultimately claiming.

### 5.3 The decision fork for the next round
- **(a) Push for *visible* selflessness + tighter confidence.** Frozen-eval the
  arm-C config (team_scale 0.30, tail-delay penalty off, old proxy off — 19s
  sacrifice in training) and widen to ~16 seeds for real confidence intervals.
  Tests whether a stronger objective makes the *greedy* policy genuinely selfless
  while keeping the fleet win.
- **(b) Accept stochastic deployment.** Evaluate the *sampled* policy (which does
  detour ~30%) and report the fleet benefit under that deployment mode — more
  faithful to how the policy actually behaves, and where the visible selflessness
  already exists.

- **In other words, you have 2 deployment methods:**

  (a) → greedy/deterministic deployment (the policy always picks its single highest-scored route, argmax).
  (b) → stochastic deployment (the policy samples a route from its probability distribution, so it sometimes takes alternatives).
  But the two options aren't just "pick a deployment mode" — that's the part to be careful about:

  (a) is "fix the policy so greedy works." If I just deployed the current policy greedily, it fails — that's exactly the collapse we saw (3.4% detour, 0.4s sacrifice). So (a) retrains with a stronger objective (arm-C: bigger team term, anti-selfless tail-delay penalty removed) to try to make the greedy/argmax policy itself genuinely selfless. Deployment mode stays greedy; the lever is the objective. That's what the 16-seed run is testing right now.

  (b) is "keep this policy, deploy it by sampling." No retraining — just acknowledge the selflessness already lives in the sampled distribution (~30% detour, 16s sacrifice) and report the fleet benefit under stochastic deployment.

  And here's the subtlety worth flagging: for a congestion/routing problem, stochastic deployment isn't just the "lenient" option — it may be the more correct one. Route diversity across the fleet is the actual mechanism that spreads load. If every vehicle in the same state acts greedily-identically, they all pile onto the same "best" route and re-congest it (a herding failure). A sampled policy naturally distributes vehicles across alternatives — which is what you want for load balancing. So (b) has a real principled justification, not just convenience.

  So the honest framing is:

  (a) = can a stronger objective make the deterministic policy selfless? (the harder, more "standard RL deployment" bar)
  (b) = deploy the policy the way it naturally load-balances (sampled), and measure the benefit there.


  Recommendation on file: **(a)** — it directly targets the deployment gap and
  yields publication-grade CIs. (b) is the faster, more lenient framing.

### 5.4 Caveats to carry forward
- 8–18 episodes/seeds is modest; some of the averaged gain is driven by rare
  severe-congestion episodes (which is arguably the point, but inflates means).
- The deterministic policy nearly reverting to shortest-path echoes the original
  failure mode — the difference now is that it *still* beats the selfish control
  and shortest-path on held-out traffic. Stronger objective authority and/or
  longer training are the levers to make the greedy behavior itself selfless.
- This is training-rollout + single-checkpoint frozen eval, not a multi-seed
  multi-checkpoint study. Treat the numbers as directionally strong, not final.

---

## 6. Pointers
- Diagnosis experiments and reward decomposition: reconstructable from
  `compute_reward` / `compute_pending_step_reward` and the per-step cost terms.
- Team-reward implementation: §3.2 line references above.
- Frozen-eval harness: [`_run_frozen_inference_eval`](../core/rl_training_pipeline.py#L2147)
  (runs the policy *and* a Dijkstra baseline on held-out seeds; inherits the
  congested regime automatically).
- Related prior docs: [training-improvement-playbook.md](training-improvement-playbook.md)
  (travel-time tuning; predates the cooperation objective),
  [rl_training_pipeline_fix_notes.md](rl_training_pipeline_fix_notes.md).
