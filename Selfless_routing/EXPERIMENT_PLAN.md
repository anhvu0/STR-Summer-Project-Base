# Experiment plan — selfless-routing revision (2026-07-12)

Goal: repair the three fatal review issues (baseline attribution confound, untested
PoA thesis, unsupported −11.2% headline) and align the paper with its own title by
(a) measuring PoA instead of asserting it, (b) isolating the detour effect from the
replanning effect, and (c) demonstrating sacrifice at the vehicle level.

## 0. The organizing measurement: a three-level decomposition

Every claim in the paper reduces to where fleet travel time sits between three
reference points, per network and demand level:

```
T_static  >=  T_DUE  >=  T_SO
   |            |          |
   |            |          system optimum (coordinated assignment)
   |            dynamic user equilibrium (selfish, congestion-aware)
   static shortest path (selfish, free-flow weights = our Dijkstra baseline)
```

- **Information gap** = T_static / T_DUE — capturable by *any* congestion-aware
  replanner. Not selfless. Hypothesis: guard-on lives here.
- **Coordination gap** = T_DUE / T_SO — the *true price of anarchy*. Only
  coordination/sacrifice can capture it. Hypothesis: guard-off detours live here.

The paper's thesis, restated measurably: *guard-on captures the information gap;
guard-off additionally captures part of the coordination gap; the value of
selflessness is the size of the coordination gap, which grows with congestion.*

**Critical recalibration this framework forces**: the bottleneck map's recorded
"PoA 1.49" (diag_bottleneck_poa.py: all-on-A 287 s vs 40/30/30 split 192 s) is
T_static/T_split — almost all *information* gap. The UE-trap diagnosis showed the
exploring fleet equalizes at ~205 s vs SO ~192 s, i.e. the true coordination gap
is only ~1.07. If that holds up under measurement (E2), a dynamic replanner will
capture 287→205 on the bottleneck too, and the selfless headroom is ~13 s/veh,
not ~95. Measure before retraining (gate G2 below); redesign the map if needed.

## E0. Prerequisites (before any runs)

1. **Branch consolidation.** The bottleneck infra (map, pattern 4,
   `--reroute-epoch-edges`, diag_bottleneck_poa.py, docs/bottleneck_map_design.md,
   dead checkpoints) lives on `mappo_benchmark_map` (d7457c9, e2654b7); the paper
   harness (Selfless_routing/reproduce/) lives here on `test_branch_mappo`. Merge
   `mappo_benchmark_map` into a new `revision-2026-07` branch off
   `test_branch_mappo` so one tree has both.
2. **New arm: `dijkstra_dynamic`.** In the reproduce harness (extend
   `penetration_sweep.py`'s arm system or a sibling `demand_sweep.py` sharing its
   helpers): re-plan shortest path against *live* edge travel times at the **same
   decision cadence as MAPPO** (reroute_epoch_edges = 5 on NYC, 2 on bottleneck)
   using the same live-weight source the policy observes. Information parity is
   the point — a reviewer must not be able to say the baselines see less.
   Optional second variant: SUMO `device.rerouting` (period ≈ cadence-equivalent)
   as a citable off-the-shelf reference.
3. **Instrumentation, landed BEFORE the sweep so nothing is rerun.**
   - Per-run detour-event log: (vehicle id, sim time, edge chosen, shortest edge,
     policy ETA delta at decision). tripinfo already gives per-vehicle times.
   - Saturation diagnostics per run: teleports, max/mean depart delay (insertion
     backlog), unfinished-vehicle count. Any demand level that teleports or
     leaves vehicles uninserted is out of scope (or re-specced) — see E3.
4. **Keep the harness gotchas** (reproduce/README): fresh ConnectionInfo per run;
   libsumo shim installed before importing core/controller; routes/cfgs
   pre-generated sequentially, only simulations in parallel.

## E1. Baseline attribution (kills or confirms the confound) — NYC, eval-only

Context from the 07-06 internal review: guard-on executes **zero** detours on all
20 seeds (the coordination throttle's `relief < deadband` OR-branch reverts every
actor detour, 417–675 per scenario). So guard-on = candidate-0 + live replanning
+ reservations — possibly zero learning contribution. The arms below decompose
that stack layer by layer:

- Arms: `dijkstra` (static, congestion-blind — paper baseline),
  `dijkstra_dynamic` (E0.2 — separates the *information* advantage),
  `mappo_index0` (full deployment stack, always pick candidate 0 — separates the
  *engineered stack*: replanning + reservations, no RL),
  `mappo_untrained` (random-init weights, guard on — separates *training* from
  architecture), `mappo_on`, `mappo_off`. 100 % penetration; paper demand
  (450/150, pattern 2, spawn 0.5 s); seeds 4010–4029. 120 runs — still trivial
  next to the 300-run penetration sweep (~5 min / 10 workers).
- Analysis: per-seed paired deltas; Wilcoxon (primary, rank-based) + exact sign
  test per contrast. The contrasts that decide the paper:
  **guard_on − dijkstra_dynamic** and **guard_on − mappo_index0**.
- **Gate G1**: if both CIs include 0, guard-on's −6.3 % is replanning +
  engineering, not selflessness and not learning → the paper's selfless content
  is entirely in the detours, and the framing centers on guard-off-vs-guard-on
  (already the clean detour contrast, immune to this confound). If guard-on
  beats both, the learned policy has value beyond replanning and keeps a
  co-headline.
- Either outcome is reportable; Table 2 gains `dijkstra_dynamic` and
  `mappo_index0` columns no matter what.

## E2. Measure PoA (stop stating it blindly) — both networks

The paper currently *asserts* "low PoA" for NYC (docs and discussion do too) and
carries an unmeasured 1.49 for the bottleneck. Replace both with measurements.
All totals use tripinfo `duration + departDelay` (insertion delay counts), summed
over the fleet; report mean/veh alongside.

### NYC (per demand level of E3)

- **T_static**: existing `dijkstra` arm.
- **T_DUE**: `duaIterate.py` (in the venv:
  `.venv/lib/python3.14/site-packages/sumo/tools/assign/duaIterate.py`) on the
  450-corridor demand with the 150 background vehicles held fixed as additional
  traffic. ~50 iterations; verify convergence (relative gap / travel-time
  stability over last iterations, `duaIterate_analysis.py` helps).
- **T_SO point estimate**: `duaIterate.py --marginal-cost` (confirmed available),
  same convergence checks.
- **T_SO bracket** (since marginal-cost DUA is approximate):
  upper bound = min total TT over *everything* evaluated (all arms, all
  assignments — any realized routing bounds SO from above); lower bound =
  sum of free-flow shortest-path times. Report **PoA as an interval**
  [T_DUE/T_upper, T_DUE/T_lower] with the marginal-cost estimate as the point.
- Deliverables per demand level: information ratio T_static/T_DUE, coordination
  ratio (true PoA) T_DUE/T_SO. These two columns replace every "low PoA"
  hand-wave in the paper, and the abstract's r = −0.48 correlation gets
  superseded by E3's designed trend.

### Bottleneck

- **T_SO**: replace the hand-picked 40/30/30 with a grid search over
  (fA, fB, fC) splits in diag_bottleneck_poa.py (5 % steps ⇒ 231 combos, ~1 min
  each, embarrassingly parallel) → tight upper bound on T_SO for fixed-split
  assignments.
- **T_DUE**: duaIterate on the tiny map (fast); cross-check against the observed
  exploring-fleet equalization (~205 s) and against a `dijkstra_dynamic` fleet
  run — all three should agree, which doubles as validation of the
  dijkstra_dynamic arm.
- **Gate G2**: if measured T_DUE/T_SO < ~1.15, the map cannot show "large,
  consistent guard-off wins" and must be redesigned *before* E5 — widen the
  coordination gap by making path A's latency more sharply convex near capacity
  (classic Pigou needs convex latency; a hard bottleneck at near-saturation
  demand, or a Braess link, both work) and re-probe. Do not spend training
  compute on a map whose true PoA is 1.07.

## E3. Congestion dose–response (demand sweep) — NYC

Converts "guard-off shines in high traffic" from a post-hoc subgroup on 20 seeds
into a designed, pre-specified test.

- **Demand levels**: fixed 75/25 mix — 300/100, 450/150 (paper), 600/200,
  750/250. Each level must pass the saturation gate (0 teleports, bounded
  insertion backlog; the repo history flags saturation problems near/above
  600 total with pattern 2). If upper levels saturate, either re-spec via spawn
  interval (0.5 → 0.4/0.35 s at fixed counts) or drop the level and say so.
- **Seeds**: fresh confirmatory block, n = 50 (5010–5059) if the runs stay this
  cheap, n = 20 minimum — the guard-on paired-t vs Wilcoxon discrepancy
  (p ≈ 0.34 vs 0.003) shows n = 20 is marginal for anything but rank stats. The
  hypothesis was formed on 4010–4029 — state that in the paper (exploratory vs
  confirmatory split; cheap credibility).
- **Arms**: `dijkstra`, `dijkstra_dynamic`, `mappo_on`, `mappo_off` (index0 and
  untrained only at the paper demand level, in E1) ⇒ 4 levels × 50 seeds × 4
  arms = 800 runs (penetration sweep did 300 in ~5 min on 10 workers; resumable
  CSV like penetration_sweep.csv).
- **Pre-specified primary endpoint**: per-seed paired **guard_off − guard_on**
  fleet-TT delta (the detour effect, confound-free), per level; median +
  bootstrap CI; monotone trend test across levels (regression of paired delta on
  demand, or Page/Jonckheere).
- Secondary: each arm vs dijkstra_dynamic; win rates; p95; worst case; deadline
  misses; non-shortest rate; saturation diagnostics.
- **Thesis figure**: x = measured coordination ratio T_DUE/T_SO at that level
  (from E2), y = paired detour gain. If both rise together, the title's question
  has a quantitative answer.

## E4. Sacrifice accounting (does "selfless" hold at the vehicle level?)

Pure post-processing of E3's logs — no extra runs if E0.3 landed first.

- Same-seed vehicle-level pairing between guard_on and guard_off runs: for
  vehicles that executed ≥1 detour in the guard-off run, distribution of own
  ΔTT (off − on); same for never-detoured vehicles.
- Headline numbers: "detourers paid median +X s each; non-detourers gained
  median −Y s; net fleet −Z s" and **altruism efficiency** = fleet-seconds saved
  per detourer-second paid.
- Caveats to handle in analysis: detourer selection bias (they sit in congested
  regions — compare against matched non-detourers by depart time/region);
  guard-off chaos (pool across 20 seeds, medians + IQR, not means).
- Both outcomes serve the paper: individual cost + system gain = demonstrated
  sacrifice (signature measurement); no individual cost = no self/system tension
  at this PoA, feeding the regime story and pointing at the bottleneck.

## E5. Bottleneck training rescue (the high-PoA demonstration) — gated on G2

Two runs are already diagnosed dead: difference mode is zero-sum across the
fleet (invisible fleet improvement), shared mode dilutes the externality 1/N
under large common-mode noise. Do not rerun either.

- **Agreed fix**: explicit marginal-cost externality pricing — per-step reward
  charge while on a1/a2 proportional to (vehicles queued behind × ~2 s service),
  time units, zero at free flow, per-agent attributable, O(10–60 s) ≫ noise.
  Plus: skip route epochs with ≤1 feasible candidate (currently ~half the batch
  is zero-gradient transitions skewing advantage stats).
- **Training gates** (from the two dead runs): approx_kl ≥ 3e-3 by ep20 (else
  actor-lr 1e-3 / update-epochs 8); training avg_tt 210 → ≤195; frozen-eval
  nonzero-route ≫ 0.45 %.
- **Success bar, restated post-decomposition**: eval delta vs
  **dijkstra_dynamic** (not static Dijkstra) approaching −(T_DUE − T_SO)/veh.
  Beating static Dijkstra by 80 s while matching dynamic Dijkstra proves nothing
  selfless.
- **Final eval**: 4 arms × 20 seeds on the bottleneck. The expected shape of the
  result table *is* the thesis in one row: dijkstra 287 ≫ dijkstra_dynamic ≈
  guard_on ≈ T_DUE > guard_off → T_SO.
- **Timebox it.** This has genuine research risk (third attempt at the credit-
  assignment problem). The paper survives on E1–E4 alone (regime story with a
  measured-PoA null on NYC + designed dose–response); E5 is the strongest figure
  if it lands. Start it early in parallel, cap the attempts.

## E6. Statistics and paper repairs (no compute)

- Table 2: add `dijkstra_dynamic` + `mappo_index0` columns; rank-based stats as
  primary (Wilcoxon + exact sign test; 16/20 → p ≈ 0.012 stated explicitly,
  12/20 → p ≈ 0.25); bootstrap CIs on paired deltas; un-bold −11.2 %, caption it
  "higher mean, outlier-driven, not a reliable win". Generate all shipped
  numbers from the CSVs via a macro file (`numbers.tex`) so reruns can't desync
  the prose from the results.
- Add: p95/p50 worsens under guard-on (2.17 → 2.24) — one honest sentence;
  deadline-miss counts are means over seeds (note non-integer); 450/150 = 75 %
  penetration base rate and its interaction with the sweep; replace abstract's
  r = −0.48 with the E3 trend result.
- Reframe: the guard is a *selfishness knob*, not an ablation; define selfless =
  individually costly ∧ socially beneficial, then let E4 test the definition.
- Page budget (7 pp, full): cut the checkpoint-selection evaluation; compress the
  penetration sweep to one figure + two sentences; the E2 PoA table and E3
  dose–response figure take their place.

## Order, gates, decision tree

```
Week 1   E0 (branch merge + dijkstra_dynamic + logging)
         E1 (80 runs, one afternoon)            -> Gate G1: attribution
         E2-bottleneck probe upgrade            -> Gate G2: map viable?
Week 1-2 E2-NYC (duaIterate UE/SO per level)
Week 2   E3 + E4 (one instrumented sweep, 320 runs + analysis)
Week 2-4 E5 in parallel iff G2 (timeboxed; map redesign first if G2 fails)
Then     E6 + rewrite
```

| G1 (guard_on vs dyn-Dijkstra) | E5 outcome | Paper framing |
|---|---|---|
| no diff | wins big | Strongest: "replanning explains reliable gains; sacrifice pays only above measured PoA ≈ X" — detours carry the title via bottleneck + dose–response |
| no diff | fails/timeout | Honest regime paper: measured PoA ≈ 1 on NYC caps all routing gains; detours = variance; high-PoA demonstration = future work *with the measurement apparatus built* |
| guard_on wins | wins big | Two-tier result: policy beats equal-information replanning AND detours capture coordination gap |
| guard_on wins | fails | Learned-policy paper with regime analysis; selfless framing softened to the E4 accounting |

Every cell is publishable; the cells differ in venue ambition. What no cell
contains anymore: an attribution confound, an asserted PoA, or a headline that
fails a sign test.
