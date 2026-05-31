# Saturation-aware detour coordination: Layer A throttle + Layer B reservation field

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
| CLI flags | [`main.py`](../main.py), [`train_rl.py`](../train_rl.py) |
| Unit tests | [`test/test_coordination_throttle.py`](../test/test_coordination_throttle.py) |
