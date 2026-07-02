# Saturation-aware coordination: Layer A detour throttle + Layer B reservation field + Layer C lane-control throttle

> Layers A/B (§§1–7) fix **routing** pile-ons at saturation. Layer C (§9) fixes a separate,
> larger driver of the greedy inference losses: forced lane-change holds that stall flow on
> seeds where the route is already identical to Dijkstra's. Start at §9 if that is your symptom.

## 1. The problem this fixes

On the NYC grid, MAPPO beats Dijkstra at moderate congestion but **loses on the most
saturated seeds** (e.g. seed 4010: +128s vs Dijkstra; with more aggressive
coordination-driven detouring the gap widened to +274s). The mechanism, verified with a
clean per-process tripinfo harness (`diag_clean.py`), is a **price-of-anarchy /
coordination failure**:

- Each detour looks locally beneficial — the policy sees positive predicted density
  relief for that one vehicle.
- But the actor scores each route candidate **independently** from a **pre-decision
  density snapshot**, and every vehicle deciding in the same window sees the same
  "alternative X is empty right now" and piles onto it.
- At saturation the alternatives are themselves near capacity, so the detouring fleet
  just spreads the jam onto them → **more** total congestion. Detouring adds distance
  *and* amplifies delay (4010: routeLength +39%, timeLoss +81s).
- When nothing has slack, shortest-path is optimal (PoA ≈ 1) — which is exactly what
  Dijkstra does. So miscoordinated "selfless" detours at saturation are worse than
  everyone taking the shortest path.

Root cause in code: both `get_candidates(...)` call sites scored candidates against the
raw instantaneous `_edge_density`, and the existing reservation state (`reserved_agents`)
never reached the route-candidate features. See the `selfless-routing-diagnosis` notes
and [`selfless_routing_analysis.md`](selfless_routing_analysis.md) §5.5.

Two cooperating fixes were added, both in
[`core/coordination_throttle.py`](../core/coordination_throttle.py).

---

## 2. Layer A — spare-capacity veto (deterministic guardrail)

**What.** After the policy picks a route, if the choice is a **detour** (candidate index
≠ 0) onto an alternative that is itself near capacity, revert to the shortest-path
**baseline** (candidate index 0).

**The key idea — read absolute density, not the relative relief.** The signal that pulls
the policy into a detour is the *relief* features (feature 9 = `baseline_mean_density −
alt_mean_density`, feature 10 = the first-edge version). At saturation the baseline is
jammed **and** the alternative is jammed, so relief can read positive (the alt is
*slightly* less jammed) even though the alternative has **zero real headroom**. The gate
therefore reads the alternative's **absolute** density (feature 3 = `max_density`,
feature 4 = `first_edge_density`), not the relief.

**Gate logic** (`detour_should_fallback`):

```
veto a detour  ⇔  idx != 0
                 AND (alt.max_density >= JAM  OR  alt.first_edge_density >= JAM)
                 AND (network_p95 >= NET_TRIGGER  OR  blended_relief < RELIEF_DEADBAND)
```

The first conjunct makes it a no-op on the baseline; the second requires the alternative
to lack absolute headroom; the third fires only when the network is genuinely saturated
(the regime where shortest-path is optimal) **or** the claimed relief is within snapshot
noise (so the detour buys nothing but distance).

**Why it is safe — a no-op on the wins.** The failure is a U-shape in traffic level. At
moderate congestion (where MAPPO wins) the alternatives have slack → `max_density` /
`first_edge_density` sit below `JAM` → the gate never fires → the useful detour is kept.
It can only ever revert to the baseline, and only at saturation.

**Where.** Inference path only, at the route-selection point in
[`controller/MAPPOController.py`](../controller/MAPPOController.py) (right after
`_act_route`, before `setRoute`). It is a deterministic post-policy guardrail and adds no
RNG, so greedy deployment stays reproducible.

**Why inference-only (not training).** Layer A overrides the *executed* route. If it ran
during training, the PPO importance ratio uses the log-prob of the action the policy
**sampled**, while the reward would reflect the **overridden** route — an off-policy
credit mismatch. Keeping Layer A at deployment matches the "deployment-only, no retrain"
framing: you can A/B it against an existing checkpoint immediately. (If you later want it
in training, record the executed action and recompute its log-prob rather than the
sampled one.)

---

## 3. Layer B — anticipatory reservation field

**What.** When a vehicle commits to a route, its **leading edges** are "booked" in a
decaying edge-level field. The route-candidate generator then scores against an
**effective density** = live count + reservations, so a vehicle deciding later in the
same window sees an alternative's relief already eroded by the detours that earlier
vehicles committed to. Simultaneous independent picks become a **damped sequential
best-response** instead of a pile-on.

**Mechanism** (`ReservationField`):

- **Seed** on commit: book the first `route_horizon` edges after the current edge,
  weighting edge *i* by `route_decay ** i` (the immediate diverted edge — where the
  pile-on happens — gets full weight). The current edge (index 0) is skipped because it
  is already in the live density.
- **Effective density**: `_effective_edge_density(e) = _edge_density(e) +
  density_weight · reserved(e) · density_scale / lane_meters(e)` — the reservation is
  added in the same units as live density. Only the candidate generator uses it; rewards,
  base state, and the central observation keep the **true** live density (so the learning
  signal and reported metrics stay honest).
- **Decay** once per simulation step by `time_decay` (and prune below `min_keep`): a
  booking fades as the booked vehicle actually enters the live density, which prevents
  double-counting.

**Where.** Both inference
([`controller/MAPPOController.py`](../controller/MAPPOController.py)) and training
([`core/rl_training_pipeline.py`](../core/rl_training_pipeline.py)). It only changes the
candidate **features** (an observation), not the action, so there is no off-policy issue —
and the policy should learn against the anticipatory density, so it is on by default in
both paths. The field is cleared per episode in `_reset_episode_density_state`.

---

## 4. How they compose

- **Layer A** catches *"the alternative is already full."*
- **Layer B** prevents *"the alternative will be full because we are all about to choose it."*

With both on, the effective density that Layer A inspects already reflects in-window
bookings, so the two reinforce: as reservations accumulate on an over-chosen alternative,
its `max_density` rises toward `JAM` and Layer A starts vetoing further detours onto it.

---

## 5. Configuration & CLI

Defaults live in `DetourThrottleConfig` and `ReservationFieldConfig`
([`core/coordination_throttle.py`](../core/coordination_throttle.py)):

| Knob | Default | Meaning |
|---|---|---|
| `jam_density` | 0.50 | alt near capacity if max/first-edge density ≥ this |
| `relief_deadband` | 0.01 | blended relief below this is treated as noise |
| `network_p95_trigger` | 0.30 | network "saturated" when occupied-edge p95 ≥ this |
| `route_horizon` | 4 | leading edges of a committed route to book |
| `route_decay` | 0.7 | weight of booked edge *i* = `route_decay ** i` |
| `time_decay` | 0.85 | per-step decay of every reservation |
| `density_weight` | 0.5 | scale of a reservation's contribution to density |
| `min_keep` | 0.02 | prune reservations below this after decay |

Both layers default **on**. Toggle for A/B:

- Inference ([`main.py`](../main.py)): `--disable-detour-throttle` (Layer A),
  `--disable-route-reservations` (Layer B).
- Training ([`train_rl.py`](../train_rl.py)): `--disable-route-reservations` (Layer B).

New telemetry counters: `detour_throttle_fallbacks` (Layer A reverts) and
`route_reservations_seeded` (Layer B bookings).

---

## 6. How to measure

Layer A needs **no retraining** — validate it against an existing checkpoint:

1. Run the saturated seed 4010 (plus a couple of U-shape losers and a couple of wins),
   greedy, via `diag_clean.py` / `main.py`, with `--disable-detour-throttle` (off) vs
   default (on). Expectation: 4010's +128s collapses toward Dijkstra; the wins stay flat
   (the gate does not fire where alternatives have slack).
2. Then enable Layer B in **training** (default) so the policy adapts to the anticipatory
   density, and re-run the frozen greedy eval.

Use a **fresh `ConnectionInfo` per run** (never share `conn` across the Dijkstra and
MAPPO sims in one process) — a shared object corrupts MAPPO's route-candidate features
and produces bogus deltas.

---

## 7. Limitations / honest framing

- On the NYC grid the price of anarchy stays low even when congested (many near-equal
  alternatives → selfish load-balancing ≈ system optimum). The 16-seed frozen eval
  showed extra selflessness gives ~0 fleet-TT benefit there. So these layers are a
  **robustness fix** — "never worse than shortest-path, sometimes better," recovering the
  saturated-seed losses — **not** a performance unlock. Demonstrating that selflessness
  *adds value* still needs a high-PoA (Braess/Pigou) bottleneck regime.
- Layer B seeds in vehicle-iteration order within a step (the route-epoch decisions are
  not priority-sorted), so the best-response is approximate. A future refinement could
  order route decisions by a coordination priority before seeding.
- Layer A is inference-only by design (§2); the policy is not trained against it.

---

## 8. Code map

| Piece | Location |
|---|---|
| Pure gate + reservation logic | [`core/coordination_throttle.py`](../core/coordination_throttle.py) |
| Layer A veto + Layer B seeding (inference) | [`controller/MAPPOController.py`](../controller/MAPPOController.py) — `make_decisions`, `_effective_edge_density` |
| Layer B effective density + seeding + decay (training) | [`core/rl_training_pipeline.py`](../core/rl_training_pipeline.py) — `_effective_edge_density`, step loop, `_reset_episode_density_state` |
| Layer C lane-control throttle | [`core/junction_decision_engine.py`](../core/junction_decision_engine.py) — `try_request_lane_change`; wired in [`controller/MAPPOController.py`](../controller/MAPPOController.py) `__init__` |
| CLI flags | [`main.py`](../main.py), [`train_rl.py`](../train_rl.py) |
| Unit tests | [`test/test_coordination_throttle.py`](../test/test_coordination_throttle.py) |

---

## 9. Layer C — lane-control throttle (the loss is the lane layer, not routing) — 2026-06-10

### 9.1 What this fixes (a different failure mode from A/B)

Layers A/B address **routing** pile-ons at saturation. Layer C addresses a failure mode that
has **nothing to do with route choice**: on several mid-congestion seeds the greedy policy
loses to Dijkstra *while choosing the shortest-path baseline on 100% of decisions*
(`route_actor … nonzero=0.0%`).

The diagnosis (reproduced per-seed with the deployed checkpoint, greedy):

- **Same routes.** At 0% detour, MAPPO's per-vehicle `routeLength` matches Dijkstra within
  ±1.4% — it is *not* taking longer paths (unlike the §1 saturation pile-on, which inflated
  routeLength +39%).
- **The gap is pure speed.** Driving the same routes, MAPPO vehicles move ~20% **slower** on
  the loss seeds and ~20% **faster** on the wins:

  | seed | result | MAPPO speed | Dijkstra speed | routeLength Δ |
  |---|---|---|---|---|
  | 4015 | loss | 3.16 m/s | 3.93 m/s | −0.2% |
  | 4001 | win  | 5.12 m/s | 4.37 m/s | +1.4% |
  | 4028 | win  | 4.34 m/s | 3.62 m/s | −0.4% |

- **The only motion command MAPPO issues that Dijkstra doesn't** is the forced
  `traci.vehicle.changeLane(vid, target, 70)` in
  [`try_request_lane_change`](../core/junction_decision_engine.py) (there is no `setSpeed`/
  `slowDown` anywhere in the controller). A **70-second** forced hold makes the vehicle insist
  on the target lane — braking and waiting for a gap. When that gap never opens under
  congestion, the vehicle **stalls itself and its followers** for the whole window → the ~20%
  flow loss. On the wins the change completes quickly and *helps* (better lane discipline →
  higher speed), which is most of why MAPPO beats Dijkstra at 0% detour in the first place.

So the forced lane change is a **double-edged sword**: net-positive on most seeds, net-negative
on the ones where it can't complete. Note `pending hard-timeouts` correlate with the losses
(r≈+0.47 among congested seeds) but are a **symptom** of stuck vehicles, not the lever — the
lever is the changeLane hold.

### 9.2 The fix

Shorten the forced hold and skip it while stalled, both in
[`try_request_lane_change`](../core/junction_decision_engine.py):

- `lane_change_hold_steps` (legacy **70** → throttled **55**): a change that can't complete
  releases its hold sooner and stops blocking; SUMO's native lane-change model resumes. **55
  is not arbitrary** — it is the welfare-optimal point of a hold sweep (§9.3): shorter holds
  (45/35/25) recover the loss seeds *more* but gridlock seed 4012 and destabilize 4010, while
  55 keeps every win, recovers the saturated 4010, and minimizes total fleet delay.
- `lane_change_skip_when_stalled_mps` (legacy **0.0** = off → throttled **1.0**): below 1 m/s
  the controller does not issue a forced change at all (no point fighting for a lane while
  stalled), deferring to the native model. Telemetry: `short_holds`, `stalled_skips`.

**Where / safety.** Inference-only, like Layer A. The defaults on the shared
`JunctionDecisionEngine` reproduce legacy behavior, so **training is unchanged**
(it builds its own engine and keeps hold=70); the inference controller lowers the values in
`MAPPOController.__init__` when `lane_control_throttle=True` (default). It changes only how a
chosen lane change is *executed*, not which action/route is chosen, so greedy stays
reproducible and there is no off-policy issue. Toggle with `--disable-lane-control-throttle`.

### 9.3 Calibration — the hold sweep (deployed checkpoint, greedy, Δ vs Dijkstra)

A *fixed* hold is the right lever but no single value is a guaranteed no-op — the loss seeds
disagree. Sweeping the hold on the discriminating seeds:

| seed | h70 (off) | h55 | h45 | h35 | shape |
|---|---|---|---|---|---|
| 4015 | +273 | +12 | +8 | −16 | monotone ↓ (shorter helps) |
| 4013 | +36 | +15 | +10 | +11 | helped |
| 4040 | +8 | +5 | −11 | −31 | helped |
| 4010 | +406 | **−40** | +141 | +791 | non-monotone; **best at 55**, gridlock at 35 |
| 4012 | +179 | +344 | +395 | +412 | monotone ↑ (**shorter hurts** — needs the long hold) |
| wins (4001/4028/4024) | big win | big win | big win | big win | insensitive |

`h55` recovers 4010+4015+4013+4040 and keeps the wins; the price is 4012. Shorter than 55 buys
a little more on 4015 but tips 4010 and 4012 into gridlock — hence the default.

### 9.4 A/B at the shipped default (hold=55, greedy, 14 seeds = 9 losses + 5 wins)

| seed | Dijkstra | off (h70) | on (h55) | off → on (Δ vs Dijkstra) |
|---|---|---|---|---|
| 4010 (saturated) | 588.4 | 994.0 | 547.4 | +405.6 → **−41.0** ✅ flips to win |
| 4015 | 456.6 | 730.1 | 468.5 | +273.5 → **+11.9** ≈ tie |
| 4013 | 463.2 | 498.9 | 478.5 | +35.7 → +15.3 |
| 4049 | 257.8 | 281.7 | 260.2 | +23.9 → +2.3 |
| 4040 | 410.3 | 418.7 | 414.9 | +8.4 → +4.5 |
| 4012 | 442.8 | 621.8 | 786.5 | +179.0 → **+343.7** ❌ regression |
| 4001 / 4017 / 4024 / 4028 / 4033 (wins) | — | −99 … −235 | −76 … −253 | all wins kept |

Across this (deliberately loss-heavy) set the mean delta moves from **+25.7 s (losing to
Dijkstra) to −18.5 s (beating it)** — a +44 s/seed swing, completion stays 100%, all five wins
hold. **Seed 4012 is the one regression** (+179 → +344): its vehicles need the full hold to
complete their changes, so any shortening costs it. It is a loss either way, not a flipped win,
and is plausibly amplified by the timid deployed checkpoint (0% detour; the log checkpoint had
4012 at ≈+13). Reproduce: `main.py --seeds … ` with and without `--disable-lane-control-throttle`.

### 9.5 Limitations / honest framing

- **Not a guaranteed no-op** (unlike Layer A). The lever is genuine — the loss is the forced
  changeLane hold, not routing — but the optimal hold is seed-dependent and non-monotone
  (4010), and one seed (4012) regresses at any shortening. `55` is the net-positive compromise;
  re-tune the two engine attributes per scenario, or disable, if a deployment is 4012-like.
- **Symptom vs lever.** `pending hard-timeouts` correlate with the losses (r≈+0.47 among
  congested seeds) but are a *symptom* of stuck vehicles; the lever is the hold itself.
- A natural next step is an **adaptive** hold — release a forced change only once it has made
  no lane progress for *k* steps (keeps 4012's completing changes, frees 4015's stuck ones) —
  rather than a single global value.
- Like Layer A, it is deployment-only; the policy is not trained against it. Lowering the
  engine defaults in training too is a possible future consistency step.
