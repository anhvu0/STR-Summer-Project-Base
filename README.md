***Selfless Traffic Routing testbed based on SUMO (STR-SUMO)***

This project uses SUMO ([https://sumo.dlr.de/docs/index.html#introduction](https://sumo.dlr.de/docs/index.html#introduction)) as the traffic simulation backend.
The goal of STR-SUMO is to benchmark routing policies for a subset of controlled vehicles while preserving realistic map, lane, and route-continuity constraints.

The current RL objective is travel-time centric:
- lower average travel time,
- better tail travel-time behavior (`p50`, `p90`),
- stronger completion rate,
- fewer routing pathologies such as loops, route mismatches, teleports, and long pending-decision stalls.

Vehicle deadlines may still exist in legacy vehicle objects for compatibility, but they are not the primary optimization target in the current RL pipeline.

***Pre-requisites***

Use Python 3.x and install the dependencies from `requirements.txt`:

```bash
pip3 install -r requirements.txt
```

You also need a working SUMO installation:
[https://sumo.dlr.de/docs/Installing/index.html](https://sumo.dlr.de/docs/Installing/index.html)

***Repository layout***

`main.py`
- Main benchmark entry point.
- Runs Dijkstra and the trained Q-learning controller on the configured SUMO scenario.
- Inference now supports `--spawn-interval` and `--seed` so you can match training-style generation when comparing controllers.

Example:

```bash
python3 main.py --spawn-interval 2.0 --seed 42
```

`configurations`
- SUMO config files, network files, route files, trained models, and episode metrics.

`core`
- `Util.py`: data structures for vehicles and network information.
- `target_vehicles_generation_protocols.py`: route and vehicle generation utilities.
- `STR_SUMO.py`: SUMO runtime wrapper used by evaluation/inference.
- `junction_decision_engine.py`: lane-feasibility, commit-window, and route-application logic.
- `shared_decision_policy.py`: shared decision lifecycle logic used by both training and inference.
- `rl_training_pipeline.py`: DQN training pipeline plus held-out frozen evaluation and best-checkpoint selection.

`controller`
- `RouteController.py`: base controller interface.
- `DijkstraController.py`: shortest-path baseline.
- `QLearningController.py`: trained-policy inference controller.

`docs`
- Analysis and operational notes for loop mitigation, telemetry interpretation, and training/inference workflow.

***Training the RL policy***

Basic training:

```bash
python3 train_rl.py
```

Useful options:

```bash
python3 train_rl.py   --sumocfg ./configurations/myconfig.sumocfg   --model-output ./configurations/model/rl_model_map.pt   --episodes 500   --spawn-interval 2.0   --eval-every 25   --eval-seeds 1001,1002,1003   --eval-spawn-interval 2.0
```

What the outputs mean:
- `rl_episode_metrics.csv`: training-rollout metrics. These runs still include replay updates during the episode, so they are useful for training trends but are not a pure deployment-quality inference measure.
- `rl_frozen_eval_metrics.csv`: held-out frozen evaluation metrics. These runs use the saved checkpoint with no online learning and average results across held-out seeds.
- `<model-output>`: the final checkpoint at the end of training.
- `<model-output>.best.pt`: the best held-out frozen-eval checkpoint, selected by completion rate first, then timeout rate, average travel time, `p90` travel time, tail gap, tail spread ratio, and deadline misses.
- `<model-output>.best.pt.meta.json`: aggregate and per-seed metadata for the best held-out checkpoint.

This workflow is the recommended way to choose a deployment checkpoint.
Inference itself does not learn; it only applies the checkpoint you trained.
`main.py` now prefers the best frozen-eval checkpoint automatically and falls back to the final checkpoint if no best checkpoint exists yet.

***Recent stability fixes reflected in the codebase***

Loop and dead-end mitigation:
- pre-commit loop/trap filtering,
- ranked fallback selection instead of first-available fallback,
- observe/cooldown flow for proactive lane changes,
- active vs passive pending handling to avoid timing out healthy lane-now queueing.

Hard-brake diagnostics:
- emergency-brake telemetry is tracked as a stress/safety signal,
- attribution is split across leader, congestion, near-junction, and residual cases,
- recent decision attribution helps distinguish policy-caused stress from background traffic noise.

Inference parity improvements:
- `main.py` can now match training generation via `--spawn-interval` and `--seed`.
- `QLearningController.should_control_vehicle(...)` now wakes the inference controller on the same structural `forced` and `open` decision cases that training evaluates, instead of relying on an extra near-junction heuristic gate.
- Training now supports held-out frozen evaluation so checkpoint selection is based on deployment-style behavior instead of optimistic in-training rollouts.

***Contribution guidance***

Code:
- Use pydoc-style function docstrings where appropriate.
- Prefer concise comments for non-obvious logic.
- Follow `lowercase_with_underscores` naming.

Tests:
- Add focused tests for important functions when feasible.
- Put test files in the `test` directory.
- For RL changes, prefer short smoke checks plus telemetry-based validation over long ad hoc debugging runs.
