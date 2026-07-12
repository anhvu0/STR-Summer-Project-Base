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
- The conclusion originally drawn from this — "when nothing has slack, shortest-path is
  optimal (PoA ≈ 1), so veto detours whenever the network is saturated" — turned out to be
  **regime-specific and is retired** (2026-07-12). At 450/150 target-pattern 2 (the current
  default regime) the Phase 0 forced-detour probe measured the opposite: relieving detours
  help **most** under saturation (a catastrophic-congestion seed went 1236s → 636s once
  detours were allowed). What holds in every regime is the *pile-on* failure above — a
  detour onto an alternative that is itself full and buys no measurable relief. The gate in
  §2 was recalibrated to key on exactly that, not on network-wide saturation.

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

**The key idea — absolute density as the precondition, illusory relief as the trigger.**
The signal that pulls the policy into a detour is the *relief* features (feature 9 =
`baseline_mean_density − alt_mean_density`, feature 10 = the first-edge version). At
saturation the baseline is jammed **and** the alternative is jammed, so relief can read
positive (the alt is *slightly* less jammed) even though the alternative has **zero real
headroom**. The gate therefore first checks the alternative's **absolute** density
(feature 3 = `max_density`, feature 4 = `first_edge_density`); only for a near-capacity
alternative does it then ask whether the claimed relief is real. A near-capacity detour
with genuine measured relief is **allowed through** — the Phase 0 probe showed those are
precisely the detours that pay off under congestion.

**Gate logic** (`detour_should_fallback`, recalibrated 2026-07-12):

```
veto a detour  ⇔  idx != 0
                 AND (alt.max_density >= JAM  OR  alt.first_edge_density >= JAM)
                 AND blended_relief < RELIEF_DEADBAND
```

The first conjunct makes it a no-op on the baseline; the second requires the alternative
to lack absolute headroom; the third fires only when the claimed relief is within
snapshot noise — a *pointless / pile-on* detour that buys nothing but distance.

**History — the retired saturation trigger.** The original gate had a third disjunct,
`network_p95 >= NET_TRIGGER` ("network saturated → veto"), built on the PoA ≈ 1 premise
of §1. At 450/150 it vetoed essentially **100% of detours** (on one measured seed the
policy wanted to detour 382 times; all were vetoed; `route_choice_nonzero_rate` = 0.000),
which made greedy and stochastic eval byte-identical and completely hid the learned
routing. Removing it (while keeping the relief-noise test) was validated on the existing
checkpoint with **no retraining**: −58s mean vs the old gate, wins 9/11 held-out seeds,
and beats blanket Layer-A-off. `network_p95_trigger` is retained in the config/signature
for compatibility but is **deprecated and unused**.

**Why it is safe — a no-op on the wins.** At moderate congestion (where MAPPO wins) the
alternatives have slack → `max_density` / `first_edge_density` sit below `JAM` → the gate
never fires → the useful detour is kept. It can only ever revert to the baseline, and
only when the chosen near-capacity detour shows no measurable relief.

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
| `network_p95_trigger` | 0.30 | **deprecated, unused** — the retired saturation trigger (§2); kept for signature compatibility only |
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

- The original "robustness fix, not a performance unlock" framing was measured at the
  earlier 350/150 regime (16-seed frozen eval, ~0 fleet-TT benefit) and **does not carry
  over to 450/150 target-pattern 2**: there, the recalibrated Layer A plus the Phase 2
  congestion-gated-reward retrain measured **−27% avg / −24% p90 vs Dijkstra on 30
  held-out seeds** (phase 2b), and the gain is concentrated in the congested tail exactly
  as the Phase 0 probe predicted. What remains true is that the *saturation-veto* variant
  of Layer A nullifies the policy (§2 history) — the layers help only in their
  recalibrated form.
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
