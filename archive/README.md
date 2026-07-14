# Archive

Superseded material from the earlier NYC penetration / Phase-2 deployment line, kept for
reference (some backs the older penetration/deployment figures). None of it is on the current
chained-Braess reproduce path — see [`../reproduce/README.md`](../reproduce/README.md) for the
live pipeline.

- `main.py` — original StrSumo demo entry point (shared one `ConnectionInfo` across a seed
  loop; the reproduce harness deliberately does not).
- `phase2c_stochastic_deploy.py`, `phase2c_stochastic.log` — Phase-2c stochastic-deployment
  eval (ruled out: loses to greedy).
- `diag_eval.py` — per-seed checkpoint comparison diagnostic for the Phase-2 NYC model.
- `inf_greedy_team_mode_{on,off}.txt` — old inference dumps.
- `phase2b_train.log` — Phase-2b training log.
- `PHASE2_HANDOFF.md` — Phase-2 handoff notes.
