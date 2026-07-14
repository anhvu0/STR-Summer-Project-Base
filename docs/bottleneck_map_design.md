# Bottleneck map: a high-price-of-anarchy regime where selfless routing can win

_Created 2026-07-01. Companion to [selfless_routing_analysis.md](selfless_routing_analysis.md) §5.0._

## 1. Why this map exists

The NYC-grid study reached a firm negative result (§4.3/§5.0 of the analysis doc):
the team reward reliably makes the policy *selfless* (14.1s ETA sacrifice, 27%
non-baseline routes, CIs clear of the selfish control), but that sacrifice buys the
fleet **nothing** — on a dense grid with many near-equal parallel paths, selfish
congestion-aware routing already approximates the system optimum (price of anarchy
≈ 1). No reward design can conjure a benefit the traffic does not contain.

The standing recommendation was a **Braess/Pigou-style bottleneck network** where
the selfish equilibrium is provably far from optimal. This map is that network,
plus the demand pattern, defaults, and probe harness to run the experiment on it.

## 2. Topology

```
in1 \
in2 == nS ==stage== nF ==( A: 1 lane,  980m,  ~71s ff )== nM ==out(5 lanes)== nOUT
in3 /                 \\==( B: 2 lanes, 1333m, ~96s ff )==//
                       \==( C: 2 lanes, 1424m, ~103s ff )=/
```

All edges one-way west→east at 13.89 m/s. Sources: `in1..in3` (1 lane each);
sink: `out`. Built from
[configurations/maps/bottleneck/](../configurations/maps/bottleneck/)
(`*.nod.xml`, `*.edg.xml`, `*.con.xml`) via netconvert `--no-turnarounds`;
output at [configurations/maps/bottleneck.net.xml](../configurations/maps/bottleneck.net.xml).

Design decisions that carry the experiment:

- **Path A is individually optimal for everyone** (shortest path from every source
  runs through `a1 a2`) but has ~0.5 veh/s saturation (1 lane). B and C cost
  +25s / +32s at free flow but carry ~1.0 veh/s each. At the default demand of
  ~1 veh/s, selfish herding overloads A by 2x while the network as a whole has
  ~2.5 veh/s of slack — the textbook Pigou dilemma.
- **Every merge is lane-preserving** (`bottleneck.con.xml`: 3x1→3 at nS,
  1+2+2→5 at nM, disjoint target lanes). Without this, netconvert makes `a2` a
  minor link yielding to both detours and the *merge* becomes the bottleneck,
  masking the designed lane-drop. (Empirically this flipped the probe from
  PoA 0.74 — coordination *hurting* — to 1.49.)
- **One decision per vehicle carries the whole outcome**: the only route choice is
  at the fork `nF`, reached from the staging edge. This collapses the NYC grid's
  credit-assignment problem (§2.3 of the analysis doc) — a detour's consequences
  are not buried under 5 edges of exogenous noise.
- **No lane-control confound**: path A is single-lane, detours are uncongested at
  optimum, so the forced-lane-change layer (§9, Layer C) has almost nothing to do.
  Wins/losses vs Dijkstra are attributable to routing.

## 3. Measured price of anarchy (diag_bottleneck_poa.py)

Pure-SUMO probe, no RL stack: identical demand run once with everyone on A
(what static shortest-path assigns = Dijkstra baseline = selfish herding) and once
with a fixed capacity-proportional split (40% A / 30% B / 30% C ≈ system optimum).

| demand | selfish avg TT | coordinated avg TT | PoA lower bound |
|---|---|---|---|
| 300 veh @ 1.00s (default) | 287.3s | **192.3s** | **1.49** |
| 300 veh @ 1.25s | 267.4s | 178.6s | 1.50 |
| 300 veh @ 0.75s | 312.8s | 212.6s | 1.47 |
| 200 veh @ 1.00s | 260.8s | 185.1s | 1.41 |

~95s/vehicle (33%) is on the table for coordination at the default demand, robust
across demand levels, with 100% completion and 0 teleports everywhere. Contrast the
NYC grid, where the best learned policy beat Dijkstra by ~50s and *selflessness
contributed ~0 of it*. Reproduce:
`SUMO_HOME=... .venv/bin/python diag_bottleneck_poa.py 300 1.0`.

Reference points at the default demand (300 controlled + 100 background):

- Dijkstra / all-on-A: **~287s** (the selfish floor the policy must beat)
- untrained policy, ~31% random detours: ~210s (one-episode smoke)
- fixed capacity-proportional split: **~192s** (the coordination target)

## 4. What was changed to run the experiment (methods)

1. **Demand pattern 4** (`core/target_vehicles_generation_protocols.py`):
   every *source* edge (no incoming connections) → the single *sink* edge (no
   outgoing). Purely topology-derived, no hardcoded IDs; errors out on strongly
   connected maps (NYC), where patterns 1–3 remain the right tool. Also added
   `--validate` to the randomTrips call so background-vehicle counts don't
   silently collapse on directed maps (no-op on the NYC grid). Note: background
   yield on the funnel is still ~50% of the requested count (random O/D pairs get
   resampled but short intra-corridor trips dominate); treat `--num-random-vehicles`
   as an upper bound here.
2. **Reroute cadence** (`reroute_epoch_edges`, now a constructor/CLI parameter in
   both `core/rl_training_pipeline.py` and `controller/MAPPOController.py`,
   default **2** in the CLIs): the edge counter starts at 1 on the spawn edge, so
   with the legacy hardcoded 5 the first route decision fired on the 5th edge — on
   this 5-edge map that is *after* the fork (or never, on path A). The policy
   literally could not make the fork choice. With 2, the decision fires exactly on
   the staging edge, where the candidate generator offers all three routes
   (guarded by `test/test_bottleneck_map.py::test_fork_offers_all_three_routes_to_the_policy`).
   The NYC study's behavior is recovered with `--reroute-epoch-edges 5`.
3. **Selfless-ready defaults** (`train_rl.py`, `main.py`): bottleneck sumocfg,
   pattern 4, 300+100 vehicles, spawn 1.0s, arm-C objective ON by default
   (`team_reward_alpha=1.0`, `team_reward_scale=0.30`, difference mode, tail-delay
   penalty and route_balance proxy zeroed — re-enable with
   `--enable-tail-delay-penalty` / `--enable-route-balance` for A/B), greedy frozen
   eval, `--min-transitions-per-update 768` (~1 PPO update/episode here),
   checkpoints under `configurations/model/mappo_policy_bottleneck*`.
   `main.py` gained `--sumocfg`; the legacy NYC workflow is
   `--sumocfg ./configurations/myconfig.sumocfg` (checkpoints `mappo_policy_nyc*`,
   `--pattern 2 --reroute-epoch-edges 5`). NYC training CSVs preserved in
   [configurations/model/nyc_legacy_rl/](../configurations/model/nyc_legacy_rl/).

## 5. Running the experiment

```bash
# B/C arm (team objective; these are all defaults, shown for clarity)
SUMO_HOME=.venv/lib/python3.14/site-packages/sumo .venv/bin/python train_rl.py \
    --episodes 120 --team-reward-alpha 1.0 --eval-every 20

# A arm (selfish control): identical except the objective
... train_rl.py --episodes 120 --team-reward-alpha 0.0 --eval-every 20 \
    --model-output ./configurations/model/mappo_policy_bottleneck_selfish.pt

# Deployment comparison on held-out seeds (defaults match training demand)
... .venv/bin/python main.py --seeds 7000,7001,...   # add --model-path for the A arm
```

**Success criteria** (all on held-out seeds, greedy deployment):

1. *Selfless behavior*: non-baseline route rate well above the selfish arm's, with
   positive mean ETA sacrifice — on this map a system-optimal policy should detour
   ~50–60% of vehicles (~26–32s sacrifice each).
2. *Selflessness pays*: fleet avg TT of the α=1 arm below the α=0 arm's — the claim
   the NYC grid could not support. The available spread is ~95s/vehicle; even a
   fraction of it clears the ±3% noise band seen in prior evals.
3. *Anchors*: both arms ≤ Dijkstra (~287s at default demand); the coordinated split
   (~192s) is the effective optimum — report where the policy lands between them.

**Failure modes to watch**: greedy herding (all vehicles pick the *same* detour and
re-congest it — Layer B reservations and the fat-tail discussion in §5.5 of the
analysis doc are directly relevant); entropy-anneal collapse back to 100% baseline
(if the eval nonzero rate hits 0, the map cannot be blamed — check
`route_choice_nonzero_rate` in `rl_frozen_eval_metrics.csv` first, it was the
smoking gun on NYC).
