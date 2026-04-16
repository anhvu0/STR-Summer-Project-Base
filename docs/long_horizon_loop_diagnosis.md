# Long-horizon loop diagnosis (April 2026 patch)

## Symbol/path trace

- Training pipeline: `core/rl_training_pipeline.py` (`RLTrainingPipeline`).
- Shared decision engine: `core/junction_decision_engine.py` (`JunctionDecisionEngine`).
- Inference controller: `controller/QLearningController.py` (`QLearningPolicy`).
- Loop-safety helpers/signals: `core/route_loop_safety.py` (`transition_signal`, long-horizon/revisit detectors).

## Where each concern currently lives

- **Long-horizon loop risk detected**
  - `core/route_loop_safety.py::transition_signal(...)` emits `long_horizon_loop` and `revisit_without_progress`.
  - `core/junction_decision_engine.py::prefilter_action_for_loops(...)` computes these signals per candidate action.

- **Where it is blocked**
  - `prefilter_action_for_loops(...)` blocks short-cycle/dead-end/trap/severe-distance always.
  - Long-horizon/revisit are blocked when `hard_block_long_horizon_loop` and `hard_block_revisit_without_progress` are enabled.

- **Where it is merely penalized**
  - `core/rl_training_pipeline.py::compute_reward(...)` applies loop-related penalties.
  - This patch splits repeated-edge, long-horizon-loop, and revisit-without-progress penalties.

- **Where fallback actions are ranked**
  - `core/junction_decision_engine.py::ranked_fallback_actions(...)`.
  - This patch adds explicit heavy penalties for long-horizon/revisit/severe-distance and clean-first ordering.

- **Where debug rows are written**
  - Schema + writer helpers in `core/rl_training_pipeline.py`:
    - `_decision_debug_fields`
    - `_build_decision_debug_row(...)`
    - `_append_decision_debug_row(s)`

- **Where inference may diverge from training**
  - `controller/QLearningController.py` constructs its own `JunctionDecisionEngine`.
  - This patch aligns defaults by enabling long-horizon/revisit hard blocking in inference constructor defaults.

## Root myopia diagnosis

The key failure mode is no-progress long-horizon revisits that are not short cycles. Before this patch, training still allowed these actions by default (hard-block flags off), fallback ranking did not explicitly demote them, and override-learning used mostly flat penalties that did not teach cause-specific avoidance. As a result, policy selection could remain locally selfish while relying on post-hoc fallback guards.
