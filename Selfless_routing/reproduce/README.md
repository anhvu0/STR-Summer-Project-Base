# Reproducing the paper's experiments

Scripts that regenerate the numbers and figures added for the 2026-07 revision
(professor's review of the 7/8/2026 draft), plus the raw results they produced.

## Environment

Run everything from the repo root with the project venv (SUMO is installed as a
pip package inside it; the scripts set `SUMO_HOME` themselves if unset):

```bash
.venv/bin/python Selfless_routing/reproduce/penetration_sweep.py --phase all
.venv/bin/python Selfless_routing/reproduce/make_figures.py
```

## Controlled-vehicle penetration sweep

`penetration_sweep.py` implements the professor's requested experiment: vary the
percentage of controlled vehicles (10% -> 100%) and observe the impact on the
approach's effectiveness, holding congestion fixed.

Design (see the paper's Experimental Setup section for the prose version):

- **Demand is identical to the paper at every sweep point.** Per seed
  (4010-4029), the paper's scenario is generated: 450 corridor vehicles to one
  shared destination + 150 random background vehicles, pattern 2, 0.5 s spawn
  interval.
- **Penetration selects who is controlled, not how many vehicles exist.** A
  seeded shuffle of the 450 corridor vehicles defines nested subsets (the 10%
  subset is contained in the 25% subset, etc.). Non-controlled corridor
  vehicles get a fixed free-flow shortest path written into the route file
  (the generator only writes a start edge, so without this they would stop
  immediately) — they emulate unequipped traffic.
- **Arms:** `dijkstra`, `mappo_on` (deployed policy, detour guard active),
  `mappo_off` (guard ablation). Model (default):
  `configurations/model/mappo_policy_nyc_phase2b.best.pt` (the paper's Phase 2b
  episode-74 checkpoint: recalibrated Layer A + congestion-gated reward);
  override with `--model`.
- **Metrics:** fleet metrics come from SUMO tripinfo over all 450 corridor
  vehicles, so they are comparable across penetration levels. The StrSumo
  controller-side stats are recorded too (`strsumo_*` columns); at 100%
  penetration these reproduce the paper's Table 2 (e.g. seed 4010:
  Dijkstra 1585.0 s, MAPPO guard-on 1281.6 s).
- 20 seeds x 5 levels x 3 arms = 300 runs, ~5 min on 10 workers.

Outputs land in `results/`:

- `penetration_sweep.csv` — one row per (seed, level, arm); resumable (finished
  rows are skipped on rerun).
- `scenarios/`, `routes/`, `cfg/`, `tripinfo/` — per-run inputs/outputs; all
  regenerable, kept for auditability.
- `sweep_log.txt` — run log of the full sweep.

Headline numbers (median paired per-scenario delta vs Dijkstra, fleet metric;
Phase 2b model):

| penetration | 10% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| guard on, median delta (s) | -4.1 | -18.1 | -20.2 | -32.9 | -33.0 |
| guard on, wins | 11/20 | 16/20 | 15/20 | 16/20 | 15/20 |
| guard on, Wilcoxon p | 0.43 | 0.019 | 0.145 | 0.006 | 0.019 |
| guard on, non-shortest rate | 0.48 | 0.47 | 0.44 | 0.41 | 0.38 |
| guard off, non-shortest rate | 0.50 | 0.48 | 0.46 | 0.43 | 0.40 |

At 100% penetration the deployment comparison is: Dijkstra 596.6 s, MAPPO
guard-on 534.3 s (-10.4%, 15/20, Wilcoxon p=0.019), guard-off 516.9 s (-13.4%,
p=0.008); all arms complete 100% of vehicles.

`make_figures.py` regenerates `../Images/penetration_sweep.pdf` (paper Fig. 5),
`make_deployment_figure.py` regenerates `../Images/deployment.pdf` (Fig. 4), and
`make_training_figure.py` regenerates `../Images/training_dynamics.pdf` (Fig. 2)
from the CSV / episode log.

## Harness notes (learned the hard way)

- **Fresh `ConnectionInfo` per run.** Sharing one across runs leaks
  `edge_vehicle_count` state between simulations and corrupts results
  (`main.py` shares one across its seed loop — do not copy that pattern).
- Workers install the **libsumo shim before importing core/controller**
  (same trick as `train_rl.py`): in-process SUMO, no TraCI port conflicts,
  safe across worker processes.
- Route files and sumocfgs are **pre-generated sequentially**; only the
  simulations run in parallel, so nothing races on `configurations/`.
- Guard-on runs are deterministic and reproduce the paper's Table 2 exactly.
  Guard-off runs are mildly chaotic (near-tie argmax choices amplified by
  congestion), so their mean can wobble by a few seconds across machines; the
  qualitative story (guard-off a few percent better, concentrated at high
  congestion) is stable. The shipped Table 2 and figures are regenerated from
  this CSV (`penetration_sweep.csv`); the prior-model runs are archived as
  `penetration_sweep.pre_phase2b.csv` and `penetration_sweep.phase1.csv`.
