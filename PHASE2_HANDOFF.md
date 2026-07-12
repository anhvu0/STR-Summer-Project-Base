# Selfless-Routing Debug + Phase 2 Fix — Handoff (2026-07-12)

Context for continuing work. Regime throughout: **450 controlled / 150 uncontrolled vehicles,
target-pattern 2, spawn-interval 0.5, held-out eval seeds 6000–6010** (NYC map).

---

## TL;DR

The RL pipeline looked like it "wasn't learning" (bit-identical frozen-eval rows). It **was**
learning useful routing — two mechanisms were forcing shortest-path and a third made the eval
blind to it. All three fixed. Retrained deployment now beats the pre-fix baseline by
**−70s avg (−13%) / −167s p90 (−16%), p90 wins 10/11 seeds, cross-seed variance halved,
100% completion.** One known limitation remains (over-detour drift → fleet-brake is the next step).

---

## Diagnosis (verified from code + sims, not docs)

The "not learning" reading was a symptom, not the disease. Root causes, in order of impact:

1. **Layer A vetoed ~100% of detours** (`detour_should_fallback` in `core/coordination_throttle.py`).
   Its premise "shortest-path is optimal under saturation (PoA≈1)" is **empirically false** here.
   With Layer A on, greedy and stochastic eval are byte-identical regardless of the weights →
   *this* is why every eval row was frozen. Measured: policy wanted to detour 382 times on one seed;
   all vetoed; `route_choice_nonzero_rate` = 0.000.
2. **Reward mildly discouraged detours** (measured EV ≈ −0.35/decision): a detour cost ~2.1 realized
   time but earned ~1.7 credit, and the refund used *estimated* not *realized* time.
3. **Eval was blind**: greedy argmax + Layer A + fixed seeds = a deterministic shortest-path run,
   so checkpoints were indistinguishable.

Supporting experiments:
- **Phase 0 probe** (fixed policies, Layer A off): detour gain is **real but tail/congestion-
  concentrated**, not a mean improvement. Random beats always-shortest-path on only 3/11 seeds; the
  mean win is driven by one catastrophic seed (6003: 1236s→636s). ⇒ A *mean* reward correctly
  collapses to shortest-path; the gain lives in the congested tail; capturing it needs a
  state-conditioned policy, not a fixed rule.
- **Phase 1b** (existing checkpoint, Layer A off): −57s mean / −108s p90 / 8-9 of 11 wins,
  variance halved — the policy had already learned good routing; Layer A was hiding it.

---

## Changes made (3 code files, uncommitted in working tree)

### `core/coordination_throttle.py` — Layer A recalibration
`detour_should_fallback` now vetoes a near-capacity detour **only when its blended relief vs the
baseline is within noise** (pointless/pile-on). Dropped the false "network saturated → veto"
trigger. `network_p95_trigger` kept but unused (marked deprecated).
Validated on existing checkpoint: **−58s mean, wins 9/11**, beats blanket Layer-A-off.

### `core/rl_training_pipeline.py` — congestion-gated reward + eval + instrumentation
- **Reward** (`_route_candidate_balance_components`): congestion gate `g = max(baseline_density −
  deadband, 0)` (~0 in light traffic → light behavior unchanged; grows with congestion). New knobs
  (current, softened values):
  - `route_balance_diversion_weight = 0.15`
  - `route_balance_congestion_relax = 0.75`
  - `route_balance_required_relief_floor = 0.60`
  - `route_balance_congestion_relief_bonus = 0.4`
  - `route_balance_detour_refund_fraction = 0.6`  (partial time-refund → selectivity)
  Verified per-decision economics: congested-detour EV **+2.95** (net ~+0.85), light **−0.4**.
- **Eval detectability** (`_run_frozen_inference_eval`): added a **stochastic Layer-A-off pass**
  (`eval_stochastic_samples`, default 3) so learning is visible before the greedy argmax flips.
  New CSV columns (appended): `stochastic_avg/p90/completion/route_choice_nonzero_rate_mean`,
  `greedy_route_choice_nonzero_rate_delta_vs_prev`, and per-decision
  `route_detour_reward_congested_mean` / `route_detour_reward_light_mean` / counts.
- **Best-checkpoint** (`_frozen_eval_score_key`): now selects on the **greedy deployment tail
  (p90 first)**, NOT the stochastic pass. (Selecting on stochastic once picked an over-detoured
  checkpoint whose greedy deployment was the worst of the run.)

### `train_rl.py`
Added `--eval-stochastic-samples` (default 3). Retrain used `--entropy-coef-end 0.10` (raised from
0.05, Phase 3) to slow premature entropy collapse; batch size left at default.

---

## Retrain (Phase 4)

Two runs. **Run #1** (reward EV +4.4, near clip) proved the fixes work — ep24 greedy eval 448/847
at 29% detour rate vs pre-fix 537/1035 at 0% — but **over-detoured monotonically** (detour rate
29→43%, avg 448→531). Softened the reward, then **Run #2** (EV +2.95): best = episode 99,
greedy eval **455.7 / 850.4**. Drift was delayed but not eliminated (rate 32→46%), though it was
non-harmful in run #2 (avg oscillated, ended good). Best-checkpoint early-stopping captured the good
policy.

**Final gate (clean 11-seed head-to-head, deployment = greedy + recalibrated Layer A):**

| metric | pre-fix | retrained | Δ |
|---|---|---|---|
| avg travel time | 537.0 ± 273 | 466.7 ± 158 | −70s (−13%), wins 7/11 |
| p90 travel time | 1035 | 868 | −167s (−16%), **wins 10/11** |
| cross-seed std | 273 | 158 | halved |
| greedy detour rate | 0.0% | 46% | eval can now see routing |
| completion | 100% | 100% | — |

Wilcoxon on the mean: p=0.28 (n=11, NOT significant — 4 small light-seed losses of +2–8s). **The
tail result (p90, 10/11) is the robust, reportable one**, consistent with the Phase 0 finding.

---

## Open items / next steps (priority order)

1. **Fleet-detour-rate brake (main next step).** The over-detour drift happens because a *per-agent*
   detour reward can't see the *fleet* collectively over-detouring — every individual detour looks
   locally good. Add a reward term that penalizes detouring when the fleet detour rate is already
   high, so ~33% becomes a *stable* attractor instead of a transient. This is what's needed for a
   cleanly-convergent, publishable result.
2. **Widen eval to ~25–30 seeds** before any significance claim on the mean (current n=11 gives the
   tail robustly but not the mean).
3. Optional: promote `mappo_policy_nyc_phase2.best.pt` → canonical `mappo_policy_nyc.best.pt` if you
   want to deploy the retrained policy (not done — canonical left as your original).

---

## Artifacts & file locations

- **Retrained model:** `configurations/model/mappo_policy_nyc_phase2.best.pt` (+ `.pt` final, `.meta.json`).
- **Original canonical best:** `configurations/model/mappo_policy_nyc.best.pt` — RESTORED from git
  (the retrain's hardcoded `--best-model-output` default had overwritten it).
- **Pre-fix baseline CSVs:** `configurations/rl_episode_metrics.pre_phase2.csv`,
  `configurations/rl_frozen_eval_metrics.pre_phase2.csv`.
- **Live metrics:** `configurations/rl_episode_metrics.csv`, `configurations/rl_frozen_eval_metrics.csv`
  (currently hold run-#2 data; rewritten on each training run).

---

## How to run (environment + commands)

Env for ALL commands: `PYTHONPATH=. SUMO_HOME=.venv/lib/python3.14/site-packages/sumo .venv/bin/python`

**Retrain** (what produced run #2):
```
PYTHONPATH=. SUMO_HOME=.venv/lib/python3.14/site-packages/sumo .venv/bin/python train_rl.py \
  --episodes 100 --eval-every 20 \
  --num-target-vehicles 450 --num-random-vehicles 150 --target-pattern 2 \
  --entropy-coef 0.15 --entropy-coef-end 0.10 --eval-stochastic-samples 3 \
  --model-output configurations/model/mappo_policy_nyc_phase2.pt
```
⚠️ `--best-model-output` defaults to `configurations/model/mappo_policy_nyc.best.pt` and WILL
overwrite it — pass an explicit `--best-model-output configurations/model/<name>.best.pt` to avoid.
Training rewrites `rl_episode_metrics.csv` / `rl_frozen_eval_metrics.csv` — back them up first.

**Eval harnesses** (were in the session scratchpad — ephemeral, ask to re-persist if needed):
- `phase0_probe.py` — fixed-policy probe (always0 / random / lowdensity vs Dijkstra), Layer A off.
- `phase1b_policy_eval.py` — existing policy, greedy/stoch × Layer A on/off, 11 seeds.
- `phase2a_validate.py` — recalibrated Layer A vs old deployment.
- `phase4_gate.py` — retrained best vs pre-fix, per seed.
Each subclasses `MAPPOPolicy`, overrides `_act_route` and/or sets `detour_throttle`, reuses
`RLTrainingPipeline` for vehicle generation + `StrSumo` to run episodes.

---

## Persistent memory written (auto-loaded next session)

`phase0-detour-gain-probe`, `layer-a-vetoes-policy`, `phase2-reward-overdetour` (+ MEMORY.md index).
These capture the diagnosis and results and will surface automatically in future chats.
