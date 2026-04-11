# HANDOFF.md

This file is for the **next fresh assistant instance** to continue work without prior chat context.

## 1) Current status

- Core RL loop has been updated recently in `core/rl_training_pipeline.py` for:
  - n-step replay horizon handling,
  - cached path-based ETA estimation,
  - tighter action validity,
  - safer commitment application,
  - improved arrival/removal accounting,
  - clearer lateness/exit metrics.
- Docs were refreshed in `README.md` to reflect current architecture and run flow.

## 2) Open challenges

1. **Training objective tuning**:
   - Externality penalty may still be weak compared with deficit/arrival terms.
2. **ETA realism**:
   - ETA is structurally aligned with deadline generation but still not fully live-traffic aware.
3. **Metric semantics**:
   - `exit_without_destination_rate` is still a mixed bucket (kept for compatibility).
   - A fully disjoint failure taxonomy would improve downstream analysis.
4. **Performance**:
   - Candidate deficit calculations do multiple `estimate_eta` calls per step/vehicle.

## 3) Next steps (suggested order)

1. Add/expand tests around:
   - arrival vs wrong-target classification,
   - duplicate removal-cause prevention,
   - unresolved lateness using `final_step`.
2. Add metric decomposition fields (disjoint buckets) while preserving legacy fields.
3. Profile ETA and decision-path hot spots; optimize only if needed.
4. Run longer training smoke tests and compare metrics drift before/after the recent logic updates.

## 4) Paths, artifacts, datasets

- RL pipeline: `core/rl_training_pipeline.py`
- Vehicle/deadline generation: `core/target_vehicles_generation_protocols.py`
- RL entrypoint: `train_rl.py`
- Baseline entrypoint: `main.py`
- SUMO config examples:
  - `configurations/myconfig.sumocfg`
  - `test/myconfig.sumocfg`
- Common artifacts:
  - `configurations/rl_model_4corners.h5`
  - `configurations/rl_model.h5`
  - `test/rl_model.h5`
  - `trips.trips.xml` outputs in config/test paths

## 5) Recent test results and logs

Most recent checks run in this environment:

```bash
python -m py_compile core/rl_training_pipeline.py core/target_vehicles_generation_protocols.py train_rl.py
```

- Result: passed (no syntax errors).
- Note: no full SUMO episode benchmark suite was executed in this handoff pass.

## 6) Schemas/contracts and expected outputs

### RL pipeline `run()` return

`RLTrainingPipeline.run()` returns `metrics_history` where each item is a dict including (current expected keys):

- `episode`
- `true_destination_arrival_rate`
- `completion_before_deadline_rate`
- `wrong_target_arrival_rate`
- `exit_without_destination_rate`
- `non_true_destination_non_teleport_rate`
- `teleport_rate`
- `average_deadline_lateness`
- `average_deadline_lateness_late_only`
- `average_deficit_improvement`
- `lane_execution_failure_rate`
- `loop_oscillation_rate`
- `replay_td_error_stats`
- `removals_by_cause`
- `shared_bonus_triggered`
- `episode_return`

### Replay transition contract

Transitions stored in replay are expected to include:

- `state`, `action`, `reward`, `next_state`, `done`, `next_mask`
- optional: `horizon` (defaults to 1 if missing)

## 7) Exact environment and package notes

### Python/env notes

- Active runtime used by this agent session: `Python 3.10.19`.
- Repo also contains a checked-in Windows-style `pyvenv.cfg` referencing Python 3.11.4 paths from another machine.

### Important: avoid duplicate env creation

- Do **not** create a second environment unless explicitly requested.
- Use the current active environment and install from:

```bash
python -m pip install -r requirements.txt
```

- Keep versions as declared in `requirements.txt` (do not upgrade package versions unless asked).

### Core pinned deps currently declared

- `tensorflow==2.11.1`
- `tensorflow-estimator==2.11.0`
- `tensorboard==2.11`
- `Keras==2.11.0`
- `h5py==3.1.0`
- `scipy==1.10.0`

## 8) Notes for continuation

- Preserve localized edits and traceability in `core/rl_training_pipeline.py`.
- Be careful with SUMO step semantics:
  - pre-step vehicle scan != post-step arrival/removal.
- If touching metrics, document whether change is:
  - behavior/training-impacting, or
  - reporting-only.
