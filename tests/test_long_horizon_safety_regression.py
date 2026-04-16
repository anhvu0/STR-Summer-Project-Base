import os
import sys
import types
import importlib
from collections import deque

sys.path.insert(0, os.getcwd())


def _install_stubs():
    os.environ.setdefault("SUMO_HOME", "/tmp")

    np_mod = types.ModuleType("numpy")
    np_mod.clip = lambda x, lo, hi: lo if x < lo else hi if x > hi else x
    np_mod.mean = lambda arr: (sum(arr) / len(arr)) if arr else 0.0
    np_mod.std = lambda arr: 0.0
    np_mod.array = lambda arr, dtype=None: list(arr)
    np_mod.zeros = lambda n, dtype=None: [0.0 for _ in range(n)]
    np_mod.float32 = float
    np_mod.random = types.SimpleNamespace(default_rng=lambda seed=None: types.SimpleNamespace(normal=lambda **kwargs: [0.0] * kwargs.get("size", 1)))

    keras_mod = types.ModuleType("keras")
    keras_layers = types.ModuleType("keras.layers")
    keras_models = types.ModuleType("keras.models")
    keras_losses = types.ModuleType("keras.losses")
    keras_optimizers = types.ModuleType("keras.optimizers")
    keras_layers.Dense = object
    keras_models.Sequential = object
    keras_models.clone_model = lambda model: model
    keras_losses.Huber = object
    keras_optimizers.Adam = object

    sumolib_mod = types.ModuleType("sumolib")
    sumolib_mod.checkBinary = lambda _: "sumo"
    sumolib_mod.net = types.SimpleNamespace(readNet=lambda *_args, **_kwargs: None)

    traci_mod = types.ModuleType("traci")

    sys.modules.setdefault("numpy", np_mod)
    sys.modules.setdefault("keras", keras_mod)
    sys.modules.setdefault("keras.layers", keras_layers)
    sys.modules.setdefault("keras.models", keras_models)
    sys.modules.setdefault("keras.losses", keras_losses)
    sys.modules.setdefault("keras.optimizers", keras_optimizers)
    sys.modules.setdefault("sumolib", sumolib_mod)
    sys.modules.setdefault("traci", traci_mod)


def test_prefilter_blocks_long_horizon_when_hard_block_enabled():
    _install_stubs()
    jde = importlib.import_module("core.junction_decision_engine")

    class CI:
        outgoing_edges_dict = {"e0": {"S": "e1", "L": "e2"}}
        lane_outgoing_edges_dict = {"lane0": {"S": "e1", "L": "e2"}}
        edge_lane_ids = {"e0": ["lane0"]}

    engine = jde.JunctionDecisionEngine(
        connection_info=CI(),
        net=types.SimpleNamespace(getEdge=lambda _: types.SimpleNamespace(allows=lambda _: True)),
        direction_choices=["S", "L"],
        hard_block_long_horizon_loop=True,
        hard_block_revisit_without_progress=False,
    )
    context = jde.DecisionContext(
        vehicle_id="v1", edge_id="e0", destination="dest", step=1, speed=5.0, lane_id="lane0",
        lane_index=0, lane_count=1, dist_to_end=100.0, edge_valid_actions=[0, 1], lane_feasible_now_actions=[0, 1],
        reachable_with_lane_change_actions=[0, 1], available_actions=[0, 1], required_lane_shift={0: 0, 1: 0},
    )

    original_transition_signal = jde.transition_signal
    jde.transition_signal = lambda *args, **kwargs: {
        "aba_bounce": False,
        "short_cycle": False,
        "dead_end_reentry": False,
        "forced_corridor": False,
        "long_horizon_loop": True,
        "revisit_without_progress": False,
    }
    ok, details = engine.prefilter_action_for_loops(
        context=context,
        action_idx=0,
        destination="dest",
        recent_history=deque(["x", "y", "z"], maxlen=12),
        distance_fn=lambda *_: 10.0,
    )
    jde.transition_signal = original_transition_signal
    assert details.get("long_horizon_loop") is True
    assert ok is False


def test_prefilter_blocks_revisit_without_progress_when_enabled():
    _install_stubs()
    jde = importlib.import_module("core.junction_decision_engine")

    class CI:
        outgoing_edges_dict = {"e0": {"S": "e1"}}
        lane_outgoing_edges_dict = {"lane0": {"S": "e1"}}
        edge_lane_ids = {"e0": ["lane0"]}

    engine = jde.JunctionDecisionEngine(
        connection_info=CI(),
        net=types.SimpleNamespace(getEdge=lambda _: types.SimpleNamespace(allows=lambda _: True)),
        direction_choices=["S"],
        hard_block_long_horizon_loop=False,
        hard_block_revisit_without_progress=True,
    )
    context = jde.DecisionContext(
        vehicle_id="v1", edge_id="e0", destination="dest", step=1, speed=5.0, lane_id="lane0",
        lane_index=0, lane_count=1, dist_to_end=100.0, edge_valid_actions=[0], lane_feasible_now_actions=[0],
        reachable_with_lane_change_actions=[0], available_actions=[0], required_lane_shift={0: 0},
    )

    original_transition_signal = jde.transition_signal
    jde.transition_signal = lambda *args, **kwargs: {
        "aba_bounce": False,
        "short_cycle": False,
        "dead_end_reentry": False,
        "forced_corridor": False,
        "long_horizon_loop": False,
        "revisit_without_progress": True,
    }
    ok, details = engine.prefilter_action_for_loops(
        context=context,
        action_idx=0,
        destination="dest",
        recent_history=deque(["x", "y", "z"], maxlen=12),
        distance_fn=lambda edge, _dest: 100.0 if edge == "x" else 200.0,
    )
    jde.transition_signal = original_transition_signal
    assert details.get("revisit_without_progress") is True
    assert ok is False


def test_ranked_fallback_prefers_clean_over_long_horizon_risky():
    _install_stubs()
    jde = importlib.import_module("core.junction_decision_engine")

    class CI:
        outgoing_edges_dict = {"e0": {"A": "e1", "B": "e2"}}
        lane_outgoing_edges_dict = {"lane0": {"A": "e1", "B": "e2"}}
        edge_lane_ids = {"e0": ["lane0"]}

    engine = jde.JunctionDecisionEngine(
        connection_info=CI(),
        net=types.SimpleNamespace(getEdge=lambda _: types.SimpleNamespace(allows=lambda _: True)),
        direction_choices=["A", "B"],
        hard_block_long_horizon_loop=False,
        hard_block_revisit_without_progress=False,
    )
    context = jde.DecisionContext(
        vehicle_id="v1", edge_id="e0", destination="dest", step=1, speed=5.0, lane_id="lane0",
        lane_index=0, lane_count=1, dist_to_end=100.0, edge_valid_actions=[0, 1], lane_feasible_now_actions=[0, 1],
        reachable_with_lane_change_actions=[0, 1], available_actions=[0, 1], required_lane_shift={0: 0, 1: 0},
    )

    def fake_prefilter(context, action_idx, destination, recent_history, distance_fn=None, distance_slack=None):
        if action_idx == 0:
            return True, {"long_horizon_loop": True, "revisit_without_progress": False}
        return True, {"long_horizon_loop": False, "revisit_without_progress": False}

    engine.prefilter_action_for_loops = fake_prefilter
    ranked = engine.ranked_fallback_actions(context, "dest", [], None, distance_fn=lambda *_: 1.0)
    assert ranked[0] == 1


def test_compute_reward_revisit_without_progress_worse_than_without():
    _install_stubs()
    rlp = importlib.import_module("core.rl_training_pipeline")

    pipeline = rlp.RLTrainingPipeline.__new__(rlp.RLTrainingPipeline)
    pipeline.travel_time_penalty = 0.05
    pipeline.system_congestion_scale = 0.015
    pipeline.eta_progress_scale = 0.65
    pipeline.distance_tiebreak_scale = 0.06
    pipeline.progress_reward_scale = 1.0
    pipeline.loop_repeat_penalty = 1.8
    pipeline.long_horizon_loop_penalty = 6.0
    pipeline.revisit_without_progress_penalty = 7.0
    pipeline.destination_reward = 50.0
    pipeline.non_global_arrival_penalty = -8.0
    pipeline.reward_clip_low = -20.0
    pipeline.reward_clip_high = 20.0
    pipeline._density_vec = [0.1]
    pipeline.connection_info = types.SimpleNamespace(
        edge_vehicle_count={"a": 1, "b": 1},
        edge_length_dict={"a": 100.0, "b": 100.0},
        outgoing_edges_dict={"b": {"S": "c"}},
    )
    pipeline._get_route_difficulty_scale = lambda *_: 1.0
    pipeline.get_distance_to_destination = lambda edge, dest: {"a": 100.0, "b": 90.0}.get(edge, 100.0)
    pipeline._estimate_remaining_eta = lambda edge, dest: {"a": 12.5, "b": 11.0}.get(edge, 12.5)
    pipeline._clip_reward = lambda x: max(min(x, pipeline.reward_clip_high), pipeline.reward_clip_low)

    vehicle = types.SimpleNamespace(destination="dest", start_time=0)
    base_reward, _ = pipeline.compute_reward(vehicle, "a", "b", 10, arrived=False, revisit_without_progress=False)
    revisit_reward, _ = pipeline.compute_reward(vehicle, "a", "b", 10, arrived=False, revisit_without_progress=True)
    assert revisit_reward < base_reward


def test_debug_field_names_present_in_source():
    source = open("core/rl_training_pipeline.py", "r", encoding="utf-8").read()
    required = [
        "selected_action",
        "executed_action",
        "override_type",
        "override_cause",
        "fallback_action",
        "prefilter_long_horizon_loop",
        "prefilter_revisit_without_progress",
        "reward_loop_component",
        "reward_terminal_component",
    ]
    for field in required:
        assert f'"{field}"' in source
