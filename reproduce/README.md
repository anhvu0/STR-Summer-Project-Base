# Reproducing the results

Everything needed to regenerate the paper's chained-Braess results, in dependency order.
Each stage has a commented wrapper (`run_*.sh`) that sets the environment and calls the
underlying script; the canonical commands are also listed below so you can run them by hand.

The wrappers keep the experiment scripts where they live (repo root and `scratch_braess/`)
because those scripts assume `PYTHONPATH=.`. Run the wrappers from anywhere — each one
`cd`s to the repo root itself.

---

## 0. Environment

- Python **3.14**, dependencies in [`requirements.txt`](../requirements.txt) (torch, numpy,
  and the SUMO python client — `traci`/`sumolib`/`libsumo` — plus `pytest`). SUMO is a pip
  package inside the venv; no system SUMO install is required.

  ```bash
  python3.14 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  ```

- **Determinism (mandatory).** Every command runs with:

  ```
  PYTHONHASHSEED=0  SUMO_HOME=.venv/lib/python3.14/site-packages/sumo  PYTHONPATH=.
  ```

  `PYTHONHASHSEED=0` pins the controller's route-commit order. Without it, near-gridlock
  seeds are chaotically sensitive and two runs of the same seed can differ by hundreds of
  seconds on the one or two scenarios that then dominate a 20-seed mean. All of this is
  wired up once in [`env.sh`](env.sh), which every wrapper sources.

- **Never run training and an eval concurrently** — both regenerate the shared route files
  (`configurations/rou/...`, `trips.trips.xml`) and corrupt each other. Sequence them.

---

## The pipeline

| # | Stage | Wrapper | Underlying script(s) |
|---|-------|---------|----------------------|
| 1 | Measured price of anarchy | [`run_poa.sh`](run_poa.sh) | `diag_braess_due_so.py` |
| 2 | Train production policy | [`run_train.sh`](run_train.sh) | `train_rl.py` |
| 3 | Held-out eval / head-to-head | [`run_eval.sh`](run_eval.sh) | `eval_braess_inference.py` |
| 4 | Learning-necessity baselines | [`run_baselines.sh`](run_baselines.sh) | `eval_braess_baselines.py` |
| 5 | Attribution ablations | [`run_ablations.sh`](run_ablations.sh) | `eval_braess_ablations.py` |
| 6 | Equilibrium / deviation audits | [`run_audits.sh`](run_audits.sh) | `scratch_braess/braess_*.py`, `due_*.py` |
| 7 | Figures & paper numbers | [`run_figures.sh`](run_figures.sh) | `Selfless_routing/reproduce/*.py` |

### 1. Measured price of anarchy

Is there a coordination gap to solve? Reports `T_static`, `T_DUE` (congestion-aware selfish
equilibrium) and `T_SO` (system optimum); the coordination gap is `T_DUE/T_SO` (~1.34 at the
training demand). Pure SUMO, no RL stack (~2-4 min).

```bash
reproduce/run_poa.sh 240 1.5          # N_VEHICLES SPAWN_INTERVAL
```

### 2. Train the production policy

MAPPO with the marginal-cost externality reward. Best checkpoint emerges early (~ep9) then
drifts, so early-stop on the frozen eval is essential (the wrapper keeps `--best-model-output`).
The frozen eval runs at the paradox demand (spawn 1.5); the CLI default 0.5 gridlocks. ~1-2 h
on CPU. Outputs `configurations/model/mappo_policy_braess.best.pt` (= the paper model).

```bash
reproduce/run_train.sh                 # exact recipe is in the wrapper header
```

### 3. Held-out evaluation

Does the policy capture the gap or herd like the selfish equilibrium? Production MAPPO plus
both Dijkstra arms (static + dynamic/live-traveltime) through the SAME harness on held-out
seeds 7000+, seed-pinned, paired one-sided Wilcoxon. Primary metric: tripinfo
duration+departDelay over all road users. ~10-20 min. Also writes per-arm tripinfo that the
`sacrifice` audit reads.

```bash
reproduce/run_eval.sh configurations/model/mappo_policy_braess.best.pt 20   # MODEL N_SEEDS
```

Expected (20 seeds 7000-7019, spawn 1.5, median): the marginal-cost policy reaches ~345 s,
capturing ~59% of the coordination gap and beating `dijkstra_dynamic` (~423 s ≈ static) on
19/20 seeds, `p < 1e-3`.

### 4. Learning-necessity baselines

Production MAPPO vs non-learning controllers, identical protocol to stage 3. ~10-20 min.

```bash
reproduce/run_baselines.sh 1.5 20      # SPAWN N_SEEDS
```

### 5. Attribution ablations

The Braess analogue of the NYC attribution table; isolates which components carry the gain.
~15-30 min.

```bash
reproduce/run_ablations.sh 20          # N_SEEDS
```

### 6. Equilibrium / deviation audits

One-vehicle best-response checks. `mappo-regret` asks whether the MAPPO outcome is an
approximate equilibrium, plain suboptimality, or sustained by individually costly holds; the
`due-*`/`br-due` audits check whether the DUE references are actually equilibria.

```bash
reproduce/run_audits.sh mappo-regret summarize 20   # round-2 headline (fast; cached results)
reproduce/run_audits.sh br-due 30 036               # best-response-dynamics T_DUE (heavy)
reproduce/run_audits.sh due-iter 039 40             # iterated-DUE residual
reproduce/run_audits.sh due-device 240 1.5 40       # rerouting-device DUE residual
reproduce/run_audits.sh sacrifice                   # per-vehicle sacrifice vs Dijkstra-dynamic
```

### 7. Figures & paper numbers

Regenerates `Selfless_routing/numbers.tex` and the figures. The generators live inside
`Selfless_routing/reproduce/`, which is the LaTeX paper working directory and is **gitignored**
(ships separately from the code); the wrapper warns and exits cleanly if it is absent. Most
generators read cached CSVs, so they run in seconds.

```bash
reproduce/run_figures.sh
```

---

## Auxiliary / legacy

- **Bottleneck map** (the alternate high-PoA case): `diag_bottleneck_poa.py`,
  `scratch_braess/eval_bottleneck_*.py`, `scratch_braess/so_grid_fine.py`,
  `bottleneck_ue_from_grid.py`. Same env prefix; see `docs/bottleneck_map_design.md`.
- **NYC penetration / deployment** (superseded phase-2 line): the runnable sweep and its
  figures live under `Selfless_routing/reproduce/` (`penetration_rep.py`,
  `attribution_eval.py`, `measure_poa.py`, `run_deterministic.sh`). Superseded scripts and
  logs were moved to [`../archive/`](../archive/).
- **Design docs**: `docs/braess_map_design.md` (map + PoA table), `BRAESS_HANDOFF.md`
  (current-state handoff), `Selfless_routing/EXPERIMENT_PLAN.md` (gates and arm definitions).
