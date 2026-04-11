# STR-SUMO (Selfless Traffic Routing on SUMO)

STR-SUMO is a simulation/testbed for **deadline-aware traffic routing** where a subset of vehicles is controlled by a routing policy and evaluated on deadline, congestion, and reliability outcomes.

## Purpose and Scope

- Simulate controlled + uncontrolled traffic on SUMO road networks.
- Assign controlled vehicles a `(start_edge, destination_edge, release_time, deadline)` contract.
- Train/evaluate routing policies (Dijkstra baseline and RL-based).
- Measure on-time completion, wrong-target arrivals, teleports/disappearances, and lateness-related metrics.

## Brief Architecture

- `main.py`: baseline simulation entrypoint (non-RL).
- `train_rl.py`: RL training entrypoint.
- `core/rl_training_pipeline.py`: DQN training loop, reward shaping, replay, n-step handling, SUMO integration.
- `core/target_vehicles_generation_protocols.py`: controlled vehicle generation + deadline generation.
- `controller/`: routing policies (`RouteController`, `DijkstraController`, `QLearningController`).
- `configurations/`: SUMO configs, route files, network files, pre-trained model artifacts.
- `test/`: unit/integration-style scripts and small fixture assets.

## Stack and Versions

### Runtime used in this repo work

- Python: `3.10.19` (current execution runtime)
- SUMO: required (install separately and set `SUMO_HOME`)

### Python packages

Use `requirements.txt` as the source of truth. Key pinned versions currently in use:

- `tensorflow==2.11.1`
- `tensorflow-estimator==2.11.0`
- `tensorboard==2.11`
- `Keras==2.11.0`
- `h5py==3.1.0`
- `scipy==1.10.0`

Other dependencies are listed in `requirements.txt` (mixed exact pins and minimum bounds).

## Setup

1. Install SUMO: https://sumo.dlr.de/docs/Installing/index.html
2. Ensure `SUMO_HOME` is set.
3. Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## How to Run

### Baseline (Dijkstra) run

```bash
python main.py
```

### RL training

Default config:

```bash
python train_rl.py
```

Explicit config/model path:

```bash
python train_rl.py \
  --sumocfg ./configurations/myconfig.sumocfg \
  --model-output ./configurations/rl_model_4corners.h5 \
  --episodes 10 \
  --spawn-interval 4.0
```

### Reproducibility / seeds

- The RL pipeline currently seeds by episode index (`seed_with_episode=True` in `RLTrainingPipeline`), so repeated runs with the same episode count/config are deterministic at the Python RNG level per episode.
- Vehicle generation also accepts a seed and is fed episode seed by the pipeline.

## Recent Decisions (changelog-lite)

Recent RL training-loop decisions in `core/rl_training_pipeline.py`:

1. Replay now supports per-transition `horizon` for n-step-consistent bootstrapping.
2. ETA estimation is now path-based and cached (aligned with deadline model shape).
3. Candidate masking requires lane feasibility.
4. Commitments are recorded only when route application succeeds.
5. Terminal handling avoids duplicate counting via `terminal_step`.
6. Arrival accounting uses post-step arrival lists and a pre-step destination snapshot.
7. Unresolved-vehicle lateness uses `final_step` (actual episode end), not fixed max-step.
8. Metrics now expose:
   - `shared_bonus_triggered`
   - `average_deadline_lateness_late_only`
   - `non_true_destination_non_teleport_rate`

## Contribution Guidance

- Keep logic localized and debugger-friendly (avoid unnecessary abstraction layers).
- Add focused tests for changed logic where possible.
- Prefer small, traceable edits in existing functions.
