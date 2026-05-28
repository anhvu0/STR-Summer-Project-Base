import numpy as np
import os
import sys
import math
import csv
import json
import copy

from xml.dom.minidom import parse
from collections import Counter, defaultdict, deque
from dataclasses import replace
import random
from controller.DijkstraController import DijkstraPolicy
from controller.RouteController import RouteController
from controller.MAPPOController import MAPPOPolicy
from core.STR_SUMO import StrSumo, build_runtime_sumocfg
from core.mappo import MAPPOConfig, MAPPOTrainer, action_mask_from_valid_actions
from core.junction_decision_engine import JunctionDecisionEngine, VehicleSnapshot
from core.shared_decision_policy import SharedDecisionPolicy
from core.Util import ConnectionInfo
from core.target_vehicles_generation_protocols import target_vehicles_generator
from core.route_loop_safety import transition_signal
from core.route_candidate_generator import (
    ROUTE_FEATURE_DIM,
    RouteCandidateGenerator,
    filter_candidates_by_first_edges,
    pack_route_candidate_features,
)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")

from sumolib import checkBinary
import traci
from traci import constants as tc
import sumolib

"""Training pipeline for the shared MAPPO traffic-routing policy."""

MAX_SIMULATION_STEPS = 4000 # This is the limit for each episode. Because vehicle might be stuck in infinite loop

# Compact metric glossary used by training CSV + logs.
# type ∈ {event_count, gauge, per_episode_aggregate, cumulative_counter, ratio}
METRIC_DOCS = {
    "episode_return_total": {"description": "Sum of all rewards in the episode.", "type": "per_episode_aggregate"},
    "avg_return_per_vehicle": {"description": "episode_return_total / controlled_vehicle_count.", "type": "ratio", "numerator": "episode_return_total", "denominator": "total_controlled"},
    "policy_loss": {"description": "Mean clipped-policy loss from the latest MAPPO update.", "type": "per_episode_aggregate"},
    "value_loss": {"description": "Mean critic regression loss from the latest MAPPO update.", "type": "per_episode_aggregate"},
    "entropy": {"description": "Mean action-distribution entropy from the latest MAPPO update.", "type": "gauge"},
    "approx_kl": {"description": "Approximate KL divergence between old and new MAPPO policies.", "type": "gauge"},
    "clip_fraction": {"description": "Share of samples clipped by the PPO ratio constraint.", "type": "ratio"},
    "decisions_opened": {"description": "Unique strategic decisions opened (one count per decision_id).", "type": "event_count", "mutually_exclusive_with_siblings": False},
    "decisions_finalized": {"description": "Strategic decisions that resolved/finalized; excludes synthetic terminal finalizations.", "type": "event_count"},
    "synthetic_terminal_finalizations": {"description": "Terminal transitions emitted when no pending strategic decision exists.", "type": "event_count"},
    "actionable_skip_ratio": {"description": "actionable_skips / actionable_decision_points.", "type": "ratio", "numerator": "actionable_skips", "denominator": "actionable_decision_points"},
    "pending_release_route_no_progress_abort": {"description": "Pending route releases aborted after no progress before timeout threshold.", "type": "event_count"},
}

class TrainingRouteHelper(RouteController):
    """
    Helper class to reuse compute_local_target during RL training and use connection_info
    """

    def __init__(self, connection_info):
        super().__init__(connection_info)

    def make_decisions(self, vehicles, connection_info):
        return {}
            
class RLTrainingPipeline:
    """
    Pipeline for training a routing policy with MAPPO.
    """

    def __init__(
        self,
        sumocfg_path,
        model_output_path,
        best_model_output_path=None,
        episodes=10,
        spawn_interval=4.0,
        seed_with_episode=True,
        destination_reward=50.0,
        teleport_penalty=-40.0,
        mappo_config=None,
        rolling_window=100,
        target_pattern=3,
        debug_exit_diagnostics=False,
        debug_exit_diagnostics_limit=20,
        step_log_every=100,
        density_refresh_every=4,
        normalize_per_step_cost_by_route_difficulty=False,
        route_difficulty_eta_floor=60.0,
        route_difficulty_scale_min=0.35,
        route_difficulty_scale_max=1.0,
        decision_debug_csv_path=None,
        fast_training_profile=False,
        eval_every=0,
        frozen_eval_seeds=None,
        eval_spawn_interval=None,
    ):
        """
        Args:
            sumocfg_path: SUMO config file path.
            model_output_path: Path to save the final trained model.
            best_model_output_path: Optional path for the best held-out frozen-eval checkpoint.
            episodes: Number of training episodes.
            spawn_interval: Interval between vehicle spawns.
            seed_with_episode: Whether to use the episode number as random seed.
            destination_reward: Reward when reaching the destination.
            teleport_penalty: Terminal penalty for teleport events.
            mappo_config: Optional MAPPOConfig override for policy/value updates.
            target_pattern: Vehicle generation pattern. 2 means varied origins
                and one shared destination (helps controlled travel-time comparison).
            normalize_per_step_cost_by_route_difficulty: If True, scales only the
                per-step travel-time cost by estimated O-D ETA so very long routes
                are not structurally over-penalized.
        """
        self.sumocfg_path = sumocfg_path
        self.model_output_path = model_output_path
        self.best_model_output_path = best_model_output_path or self._default_best_model_output_path(model_output_path)
        self.episodes = episodes
        self.spawn_interval = float(spawn_interval)
        self.eval_every = max(int(eval_every), 0)
        default_eval_seeds = self._default_frozen_eval_seeds() if self.eval_every > 0 else []
        self.frozen_eval_seeds = [int(seed) for seed in (frozen_eval_seeds or default_eval_seeds)]
        self.eval_spawn_interval = float(eval_spawn_interval) if eval_spawn_interval is not None else float(self.spawn_interval)
        self._best_frozen_eval_key = None
        self._best_frozen_eval_summary = None
        self.seed_with_episode = seed_with_episode
        self.destination_reward = destination_reward
        self.teleport_penalty = teleport_penalty
        self.mappo_config = mappo_config or MAPPOConfig()
        self.rolling_window = rolling_window
        self.target_pattern = target_pattern
        self.debug_exit_diagnostics = debug_exit_diagnostics
        self.debug_exit_diagnostics_limit = max(int(debug_exit_diagnostics_limit), 0)
        self.step_log_every = max(int(step_log_every), 1)
        self.density_refresh_every = max(int(density_refresh_every), 1)
        self.normalize_per_step_cost_by_route_difficulty = bool(normalize_per_step_cost_by_route_difficulty)
        self.route_difficulty_eta_floor = max(float(route_difficulty_eta_floor), 1.0)
        self.route_difficulty_scale_min = float(np.clip(route_difficulty_scale_min, 0.05, 1.0))
        self.route_difficulty_scale_max = float(np.clip(route_difficulty_scale_max, self.route_difficulty_scale_min, 1.0))
        self.fast_training_profile = bool(fast_training_profile)
        self.decision_debug_csv_path = decision_debug_csv_path
        if self.fast_training_profile and self.decision_debug_csv_path:
            self.decision_debug_csv_path = None
        if self.fast_training_profile:
            self.density_refresh_every = max(self.density_refresh_every, 4)
        self.runtime_sumocfg_path = build_runtime_sumocfg(self.sumocfg_path, fast_mode=self.fast_training_profile)
        self._distance_cache = {}
        self._cache_metrics = defaultdict(float)
        self.progress_reward_scale = 1.00
        self.system_congestion_scale = 0.04
        self.selfless_reward_scale = 0.35
        self.selfless_reward_clip = 3.0
        self.loop_window = 12
        self.loop_repeat_penalty = 1.5
        # Objective priority:
        # 1) minimize travel time (dominant)
        # 2) congestion externality (secondary)
        # 3) shortest-path distance as tie-breaker
        self.travel_time_penalty = 0.07
        self.eta_progress_scale = 0.30
        self.distance_tiebreak_scale = 0.02
        self.coordination_pressure_penalty = 0.35
        self.pending_coordination_penalty = 0.04
        self.score_slack = 30.0
        self.reward_clip_low = -20.0
        self.reward_clip_high = 20.0
        # Arrival transitions get a wider ceiling so destination_reward isn't uniformly
        # saturated. All non-terminal step rewards still use reward_clip_high.
        self.terminal_reward_clip_high = max(float(destination_reward) + 5.0, self.reward_clip_high)
        self.pending_timeout_penalty = -18.0
        self.pending_latency_penalty_per_step = 0.008
        self.pending_replan_penalty = -0.4
        self.stale_disappeared_penalty = -14.0
        self.observe_no_progress_penalty = -0.5
        self.observe_low_speed_penalty = -0.4
        self.observe_commit_window_miss_penalty = -0.7
        self.same_edge_repeat_chase_penalty = -0.5
        self.fallback_missed_lane_penalty = -0.4
        self.loop_trap_override_penalty = -1.4
        self.hard_brake_event_penalty = 0.35
        self.tail_delay_threshold_eta_mult = 1.35
        self.tail_delay_threshold_min_steps = 180.0
        self.tail_delay_threshold_max_steps = 320.0
        self.tail_delay_linear_penalty = 0.075
        self.tail_delay_quadratic_penalty = 0.00016
        self.tail_arrival_penalty_per_25_steps = 0.90
        self.tail_arrival_penalty_cap = 8.0
        self.hard_brake_attribution_window_steps = 24

        self.sumocfg_dir = os.path.dirname(sumocfg_path)
        self.net_file, self.route_file = self.parse_sumocfg(sumocfg_path)

        self.net = sumolib.net.readNet(os.path.join(self.sumocfg_dir, self.net_file))

        self.connection_info = ConnectionInfo(os.path.join(self.sumocfg_dir, self.net_file))
        self._edge_list = tuple(self.connection_info.edge_list)
        self._edge_lane_meters_cache = {
            edge_id: max(float(self.connection_info.edge_length_dict.get(edge_id, 5.0)), 5.0)
            * float(max(len(self.connection_info.edge_lane_ids.get(edge_id, [])), 1))
            for edge_id in self._edge_list
        }
        self._edge_lane_meters_vec = np.array(
            [self._edge_lane_meters_cache[edge_id] for edge_id in self._edge_list],
            dtype=np.float32,
        )
        self.route_helper = TrainingRouteHelper(self.connection_info)
        self.decision_engine = JunctionDecisionEngine(
            self.connection_info,
            self.net,
            self.route_helper.direction_choices,
        )

        # state = [edge_embedding, destination_embedding]
        #         + edge/lane/reachable/available feasibility masks (4*6)
        #         + commit flag + 3 lane features + 3 travel-time features
        #         + local congestion summary
        #         + per-action branch features (6 actions * 5 features)
        # NOTE: compact-state size changed; retraining is required.
        self.edge_embedding_dim = 8
        self.shared_policy = SharedDecisionPolicy(
            self.connection_info,
            self.decision_engine,
            self.route_helper.direction_choices,
            edge_embedding_dim=self.edge_embedding_dim,
            density_scale_m=100.0,
            max_simulation_steps=MAX_SIMULATION_STEPS,
        )
        self.local_congestion_k = self.shared_policy.local_congestion_k
        self._init_edge_embeddings(seed=1337)
        self.route_k = 4                           # number of candidate routes offered to policy
        self.route_feature_dim = ROUTE_FEATURE_DIM
        self.route_obs_dim = self.route_k * self.route_feature_dim
        self.reroute_epoch_edges = 5               # re-query policy every N completed edges
        self.state_size = self.shared_policy.compact_state_size + self.route_obs_dim
        self.central_observation_size = 18
        self.action_size = self.route_k            # policy picks a route index, not a direction
        if int(getattr(self.mappo_config, "route_candidate_feature_dim", 0)) != self.route_feature_dim:
            self.mappo_config = replace(
                self.mappo_config,
                route_candidate_feature_dim=self.route_feature_dim,
            )
        self.route_generator = RouteCandidateGenerator(
            connection_info=self.connection_info,
            net=self.net,
            k_routes=self.route_k,
            oversample=max(self.route_k * 2, 8),
            max_route_length_m=8000.0,
            lru_maxsize=2048,
        )
        self._episode_route_obs: dict = {}         # vehicle_id -> route obs array, reset each episode
        self.metrics_csv_path = os.path.join(self.sumocfg_dir, "rl_episode_metrics.csv")
        self.frozen_eval_metrics_csv_path = os.path.join(self.sumocfg_dir, "rl_frozen_eval_metrics.csv")
        self.best_model_metadata_path = self.best_model_output_path + ".meta.json"
        self._frozen_eval_model_path = self._default_best_model_output_path(self.model_output_path).replace(".best", ".frozen_eval_current")
        self._frozen_eval_baseline_cache = {}
        self._density_vec = np.zeros(len(self.connection_info.edge_list), dtype=np.float32)
        self._density_mean = 0.0
        self._density_std = 0.0
        self._density_p95 = 0.0
        # Density is now vehicles per 100m per lane; feature scale changed, retraining is required.
        self.density_scale_m = 100.0
        self._last_density_step = -10**9
        self._lane_length_cache = {}
        self._passenger_edge_set = set(self.connection_info.edge_list)
        self._vehicle_subscription_vars = (
            tc.VAR_ROAD_ID,
            tc.VAR_LANE_ID,
            tc.VAR_LANE_INDEX,
            tc.VAR_LANEPOSITION,
            tc.VAR_SPEED,
            tc.VAR_WAITING_TIME,
        )
        self._edge_subscription_vars = (tc.LAST_STEP_VEHICLE_NUMBER,)
        self._active_vehicle_subscriptions = set()
        self._step_vehicle_results = {}
        self._step_vehicle_wait_cache = {}
        self._step_lane_occupancy_cache = {}
        self._step_lane_halting_cache = {}
        self.congestion_density_threshold = 0.30
        self.congestion_low_speed_threshold = 2.0
        self.emergency_decel_threshold = 4.5
        self.teleport_jam_density_threshold = 0.55
        self.trainer = MAPPOTrainer(
            self.state_size,
            self.central_observation_size,
            self.action_size,
            config=self.mappo_config,
        )
        # Maps vehicle_id -> (buffer_index, buffer_generation) for terminal credit patching.
        self._vehicle_last_buffer_pos: dict = {}
        self._decision_debug_fields = [
            "episode", "step", "vehicle_id", "decision_edge", "action", "action_source", "available_actions",
            "forced_action", "lane_feasible_now_actions", "reachable_with_lane_change_actions",
            "commit_window", "dist_to_end", "intended_next_edge", "actual_next_edge", "finalized",
            "finalize_delay_steps", "reward", "done", "route_mismatch", "teleported",
            "reached_global_destination", "prev_eta", "curr_eta", "prev_distance", "curr_distance",
            "edge_density", "mean_density", "externality_penalty", "marginal_pressure",
            "terminal_outcome", "last_confirmed_edge", "destination", "in_arrived_ids", "in_teleport_ids",
        ]

    def _ensure_decision_debug_csv_header(self):
        if not self.decision_debug_csv_path:
            return
        if not os.path.isabs(self.decision_debug_csv_path):
            self.decision_debug_csv_path = os.path.join(self.sumocfg_dir, self.decision_debug_csv_path)
        if not os.path.exists(self.decision_debug_csv_path):
            with open(self.decision_debug_csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._decision_debug_fields).writeheader()

    def _append_decision_debug_row(self, row):
        if not self.decision_debug_csv_path:
            return
        with open(self.decision_debug_csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._decision_debug_fields).writerow(row)

    def _append_decision_debug_rows(self, rows):
        if not self.decision_debug_csv_path or not rows:
            return
        with open(self.decision_debug_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._decision_debug_fields)
            writer.writerows(rows)

    def _zero_central_observation(self):
        return np.zeros((1, self.central_observation_size), dtype=np.float32)

    def _action_mask_vector(self, valid_actions):
        return action_mask_from_valid_actions(self.action_size, valid_actions)

    def _route_logit_margin(self, selection, valid_route_indices):
        logits = np.asarray(getattr(selection, "masked_logits", []), dtype=np.float32).reshape(-1)
        valid = [idx for idx in valid_route_indices if 0 <= int(idx) < len(logits)]
        if len(valid) <= 1:
            return 0.0
        valid_logits = np.sort(logits[valid])
        return float(valid_logits[-1] - valid_logits[-2])

    def _record_route_actor_choice(self, decision_metrics, selection, feasible_candidates, chosen_idx):
        valid_count = len(feasible_candidates)
        decision_metrics[f"route_valid_candidates_{valid_count}"] += 1
        decision_metrics["route_valid_candidate_sum"] += valid_count
        decision_metrics[f"route_choice_idx_{chosen_idx}"] += 1
        if int(chosen_idx) != 0:
            decision_metrics["route_choice_nonzero_count"] += 1

        decision_metrics["route_logit_margin_sum"] += self._route_logit_margin(
            selection,
            range(valid_count),
        )
        decision_metrics["route_logit_margin_count"] += 1

        features = np.asarray(feasible_candidates[int(chosen_idx)].features, dtype=np.float32)
        if features.size >= ROUTE_FEATURE_DIM:
            decision_metrics["route_chosen_length_norm_sum"] += float(features[0])
            decision_metrics["route_chosen_eta_norm_sum"] += float(features[1])
            decision_metrics["route_chosen_density_sum"] += float(features[2])
            decision_metrics["route_chosen_first_density_sum"] += float(features[4])

    def _build_central_observation(
        self,
        *,
        step,
        total_controlled,
        arrived_ids,
        step_snapshots,
        vehicles,
        pending_decisions,
        coordination_state,
        open_decision_count,
    ):
        observation = np.zeros(self.central_observation_size, dtype=np.float32)
        total_controlled = max(int(total_controlled), 1)
        live_count = int(len(step_snapshots))
        arrived_count = int(len(arrived_ids))
        pending_summary = self._summarize_pending_backlog(pending_decisions, step)

        speeds = [float(snapshot.speed) for snapshot in step_snapshots.values()]
        waits = [self._vehicle_wait_time(vehicle_id) for vehicle_id in step_snapshots.keys()]
        remaining_etas = []
        for vehicle_id, snapshot in step_snapshots.items():
            vehicle = vehicles.get(vehicle_id)
            if vehicle is None:
                continue
            eta = self._estimate_remaining_eta(snapshot.edge_id, vehicle.destination)
            if math.isfinite(eta):
                remaining_etas.append(float(eta))

        observation[0] = min(float(step) / float(MAX_SIMULATION_STEPS), 1.0)
        observation[1] = float(arrived_count) / float(total_controlled)
        observation[2] = float(live_count) / float(total_controlled)
        observation[3] = float(pending_summary["total_open"]) / float(total_controlled)
        observation[4] = float(max(int(open_decision_count), 0)) / float(total_controlled)
        observation[5] = min(float(np.mean(speeds)) / 20.0, 1.0) if speeds else 0.0
        observation[6] = min(float(np.std(speeds)) / 10.0, 1.0) if len(speeds) > 1 else 0.0
        observation[7] = (
            min(float(np.mean(remaining_etas)) / float(MAX_SIMULATION_STEPS), 1.0)
            if remaining_etas else 0.0
        )
        observation[8] = (
            min(float(max(remaining_etas)) / float(MAX_SIMULATION_STEPS), 1.0)
            if remaining_etas else 0.0
        )
        observation[9] = min(float(np.mean(waits)) / 120.0, 1.0) if waits else 0.0
        observation[10] = float(np.clip(self._density_mean, 0.0, 1.0))
        observation[11] = float(np.clip(self._density_std, 0.0, 1.0))
        observation[12] = float(np.clip(self._density_p95, 0.0, 1.0))
        if coordination_state is not None:
            observation[13] = float(
                np.clip(
                    float(coordination_state.reserved_agents)
                    / max(float(self.shared_policy.coordination_reserved_agents_cap), 1.0),
                    0.0,
                    1.0,
                )
            )
        observation[14] = min(
            float(pending_summary["mean_age_all"]) / float(self.decision_engine.route_pending_hard_timeout_steps),
            1.0,
        )
        observation[15] = min(
            float(pending_summary["max_age_all"]) / float(self.decision_engine.route_pending_hard_timeout_steps),
            1.0,
        )
        observation[16] = float(pending_summary["active_monitoring_open"]) / float(total_controlled)
        observation[17] = 1.0 if (
            live_count > 0
            and float(np.mean([self._edge_density(snapshot.edge_id) for snapshot in step_snapshots.values()])) >= self.congestion_density_threshold
            and float(np.mean(speeds)) <= self.congestion_low_speed_threshold
        ) else 0.0
        return observation.reshape(1, -1)

    def _build_mappo_trace(self, state, central_observation, selection):
        return {
            "mappo_training": True,
            "mappo_initial_state": np.asarray(state, dtype=np.float32).reshape(1, -1).copy(),
            "mappo_initial_central_observation": np.asarray(central_observation, dtype=np.float32).reshape(1, -1).copy(),
            "mappo_log_prob": float(selection.log_prob),
            "mappo_value": float(selection.value),
            "mappo_action_mask": np.asarray(selection.action_mask, dtype=np.float32).reshape(-1).copy(),
            "mappo_reward_accumulator": 0.0,
        }

    def _accumulate_mappo_reward(self, metadata, reward):
        if not isinstance(metadata, dict) or not metadata.get("mappo_training", False):
            return
        metadata["mappo_reward_accumulator"] = float(metadata.get("mappo_reward_accumulator", 0.0)) + float(reward)

    def _accumulate_route_trace_reward(self, vehicle_route_trace, vehicle_id, reward):
        trace = vehicle_route_trace.get(vehicle_id) if isinstance(vehicle_route_trace, dict) else None
        if trace is not None:
            self._accumulate_mappo_reward(trace, reward)

    def _accumulate_route_epoch_step_reward(
        self,
        vehicle_route_trace,
        vehicle_id,
        vehicle,
        edge_id,
        step,
        coordination_pressure=0.0,
    ):
        trace = vehicle_route_trace.get(vehicle_id) if isinstance(vehicle_route_trace, dict) else None
        if trace is None:
            return 0.0
        try:
            last_credit_step = int(trace.get("route_last_credit_step", step))
        except (TypeError, ValueError):
            last_credit_step = int(step)
        elapsed = max(int(step) - int(last_credit_step), 0)
        trace["route_last_credit_step"] = int(step)
        if elapsed <= 0:
            return 0.0

        reward = self.compute_pending_step_reward(
            vehicle,
            edge_id,
            elapsed=elapsed,
            step=step,
            coordination_pressure=coordination_pressure,
        )
        self._accumulate_mappo_reward(trace, reward)
        trace["route_elapsed_steps"] = int(trace.get("route_elapsed_steps", 0)) + int(elapsed)
        return float(reward)

    def _reset_episode_density_state(self):
        for edge_id in self._edge_list:
            self.connection_info.edge_vehicle_count[edge_id] = 0
        self._density_vec = np.zeros(len(self._edge_list), dtype=np.float32)
        self._density_mean = 0.0
        self._density_std = 0.0
        self._density_p95 = 0.0
        self._last_density_step = -10**9

    def _record_immediate_mappo_transition(
        self,
        trace,
        *,
        action,
        reward,
        next_state,
        next_central_observation,
        done,
        discount_steps=1,
        metadata=None,
    ):
        if not isinstance(trace, dict) or not trace.get("mappo_training", False):
            return False
        total_reward = float(trace.get("mappo_reward_accumulator", 0.0)) + float(reward)
        stored_metadata = dict(metadata or {})
        vid = trace.get("vehicle_id")
        if vid is not None:
            stored_metadata["vehicle_id"] = vid
        self.trainer.record_transition(
            observation=trace["mappo_initial_state"],
            central_observation=trace["mappo_initial_central_observation"],
            action=int(action),
            action_mask=trace["mappo_action_mask"],
            log_prob=float(trace["mappo_log_prob"]),
            value=float(trace["mappo_value"]),
            reward=total_reward,
            next_observation=np.asarray(next_state, dtype=np.float32).reshape(1, -1),
            next_central_observation=np.asarray(next_central_observation, dtype=np.float32).reshape(1, -1),
            done=bool(done),
            discount_steps=max(int(discount_steps), 1),
            metadata=stored_metadata,
            critic_only=bool(trace.get("mappo_critic_only", False)),
        )
        if vid is not None:
            self._vehicle_last_buffer_pos[vid] = (len(self.trainer.buffer) - 1, self.trainer.buffer_generation)
        trace["mappo_transition_recorded"] = True
        return True

    def _record_pending_mappo_transition(
        self,
        pending,
        *,
        reward,
        next_state,
        next_central_observation,
        done,
        discount_steps=1,
        metadata=None,
    ):
        if pending is None or not isinstance(getattr(pending, "metadata", None), dict):
            return False
        return self._record_immediate_mappo_transition(
            pending.metadata,
            action=int(pending.intended_action),
            reward=reward,
            next_state=next_state,
            next_central_observation=next_central_observation,
            done=done,
            discount_steps=discount_steps,
            metadata=metadata,
        )

    def _format_top_counts(self, counts, limit=3):
        if not counts:
            return ""
        items = []
        for key, value in counts.items():
            try:
                numeric_value = float(value)
            except Exception:
                continue
            if numeric_value <= 0.0:
                continue
            items.append((str(key), numeric_value))
        if not items:
            return ""
        items.sort(key=lambda item: (-item[1], item[0]))
        formatted = []
        for key, numeric_value in items[:max(int(limit), 1)]:
            if abs(numeric_value - round(numeric_value)) <= 1e-9:
                value_str = str(int(round(numeric_value)))
            else:
                value_str = f"{numeric_value:.2f}"
            formatted.append(f"{key}:{value_str}")
        return "|".join(formatted)

    def _format_pending_descriptor(self, descriptor):
        if not descriptor:
            return ""
        return (
            f"{descriptor['vehicle_id']}@{descriptor['edge']}:"
            f"age={descriptor['age']},stall={descriptor['stall_age']},"
            f"phase={descriptor['phase']},mode={descriptor['resolution_mode']},"
            f"shift={descriptor['current_shift']}"
        )

    def _summarize_pending_backlog(self, pending_decisions, step):
        step = max(int(step), 0)
        summary = {
            "total_open": 0,
            "observe_open": 0,
            "route_open": 0,
            "lane_now_open": 0,
            "proactive_open": 0,
            "active_monitoring_open": 0,
            "mean_age_all": 0.0,
            "max_age_all": 0.0,
            "mean_age_active": 0.0,
            "max_age_active": 0.0,
            "mean_age_lane_now": 0.0,
            "max_age_lane_now": 0.0,
            "mean_stall_age": 0.0,
            "max_stall_age": 0.0,
            "oldest_descriptor": None,
            "hot_edges": "",
        }
        if not pending_decisions:
            return summary

        age_all = []
        age_active = []
        age_lane_now = []
        stall_ages = []
        hot_edges = Counter()
        oldest_descriptor = None

        for vehicle_id, pending in pending_decisions.items():
            metadata = pending.metadata if isinstance(pending.metadata, dict) else {}
            phase = self.shared_policy.pending_phase(pending)
            resolution_mode = self.shared_policy.pending_resolution_mode(pending)
            active_monitoring = self.shared_policy.pending_requires_active_same_edge_monitoring(pending)
            age = max(step - int(pending.decision_step), 0)
            last_progress_step = int(metadata.get("last_progress_step", pending.decision_step))
            stall_age = max(step - last_progress_step, 0)
            current_shift = int(metadata.get("last_required_shift", metadata.get("observe_last_required_shift", 99)))
            descriptor = {
                "vehicle_id": str(vehicle_id),
                "edge": str(pending.decision_edge),
                "age": int(age),
                "stall_age": int(stall_age),
                "phase": str(phase),
                "resolution_mode": str(resolution_mode),
                "current_shift": int(current_shift),
            }
            if oldest_descriptor is None or (descriptor["age"], descriptor["stall_age"], descriptor["vehicle_id"]) > (
                oldest_descriptor["age"],
                oldest_descriptor["stall_age"],
                oldest_descriptor["vehicle_id"],
            ):
                oldest_descriptor = descriptor

            summary["total_open"] += 1
            hot_edges[str(pending.decision_edge)] += 1
            age_all.append(float(age))
            stall_ages.append(float(stall_age))
            if phase == "observe_lane_change":
                summary["observe_open"] += 1
            else:
                summary["route_open"] += 1
            if active_monitoring:
                summary["active_monitoring_open"] += 1
                age_active.append(float(age))
            if resolution_mode == "lane_now":
                summary["lane_now_open"] += 1
                age_lane_now.append(float(age))
            else:
                summary["proactive_open"] += 1

        summary["mean_age_all"] = float(np.mean(age_all)) if age_all else 0.0
        summary["max_age_all"] = float(max(age_all)) if age_all else 0.0
        summary["mean_age_active"] = float(np.mean(age_active)) if age_active else 0.0
        summary["max_age_active"] = float(max(age_active)) if age_active else 0.0
        summary["mean_age_lane_now"] = float(np.mean(age_lane_now)) if age_lane_now else 0.0
        summary["max_age_lane_now"] = float(max(age_lane_now)) if age_lane_now else 0.0
        summary["mean_stall_age"] = float(np.mean(stall_ages)) if stall_ages else 0.0
        summary["max_stall_age"] = float(max(stall_ages)) if stall_ages else 0.0
        summary["oldest_descriptor"] = oldest_descriptor
        summary["hot_edges"] = self._format_top_counts(hot_edges)
        return summary

    def _build_decision_debug_row(
        self,
        episode,
        step,
        pending,
        actual_next_edge,
        finalized,
        finalize_delay_steps,
        reward="",
        done="",
        route_mismatch="",
        teleported=0,
        reached_global_destination=0,
        prev_eta="",
        curr_eta="",
        prev_distance="",
        curr_distance="",
        edge_density="",
        mean_density="",
        externality_penalty="",
        marginal_pressure="",
        terminal_outcome="",
        last_confirmed_edge="",
        destination="",
        in_arrived_ids="",
        in_teleport_ids="",
    ):
        context = pending.context
        return {
            "episode": episode,
            "step": step,
            "vehicle_id": context.vehicle_id,
            "decision_edge": pending.decision_edge,
            "action": pending.intended_action,
            "action_source": pending.metadata.get("action_source", ""),
            "available_actions": json.dumps(context.available_actions),
            "forced_action": context.forced_action if context.forced_action is not None else "",
            "lane_feasible_now_actions": json.dumps(context.lane_feasible_now_actions),
            "reachable_with_lane_change_actions": json.dumps(context.reachable_with_lane_change_actions),
            "commit_window": int(bool(context.commit_window)),
            "dist_to_end": float(context.dist_to_end),
            "intended_next_edge": pending.intended_next_edge,
            "actual_next_edge": actual_next_edge,
            "finalized": int(bool(finalized)),
            "finalize_delay_steps": finalize_delay_steps,
            "reward": reward,
            "done": done,
            "route_mismatch": route_mismatch,
            "teleported": teleported,
            "reached_global_destination": reached_global_destination,
            "prev_eta": prev_eta,
            "curr_eta": curr_eta,
            "prev_distance": prev_distance,
            "curr_distance": curr_distance,
            "edge_density": edge_density,
            "mean_density": mean_density,
            "externality_penalty": externality_penalty,
            "marginal_pressure": marginal_pressure,
            "terminal_outcome": terminal_outcome,
            "last_confirmed_edge": last_confirmed_edge,
            "destination": destination,
            "in_arrived_ids": in_arrived_ids,
            "in_teleport_ids": in_teleport_ids,
        }

    def _get_route_difficulty_scale(self, vehicle, reference_edge):
        """
        Return a multiplicative scale in [route_difficulty_scale_min, route_difficulty_scale_max]
        used for the per-step travel-time component only.
        """
        if not self.normalize_per_step_cost_by_route_difficulty:
            return 1.0

        cached_scale = getattr(vehicle, "route_difficulty_scale", None)
        if cached_scale is not None:
            return float(cached_scale)

        eta = self._estimate_remaining_eta(reference_edge, vehicle.destination)
        if not math.isfinite(eta):
            scale = 1.0
        else:
            # Harder/longer O-D pairs (larger ETA) receive a smaller per-step weight.
            normalized = self.route_difficulty_eta_floor / max(float(eta), self.route_difficulty_eta_floor)
            scale = float(np.clip(normalized, self.route_difficulty_scale_min, self.route_difficulty_scale_max))

        vehicle.route_difficulty_scale = scale
        return scale

    def _print_step_progress(
        self,
        episode,
        step,
        total_controlled,
        arrived_ids,
        decision_metrics,
        mean_density_samples=None,
        congestion_high_pressure_steps=0,
    ):
        override_total = decision_metrics["override_events_total"]
        completion = (len(arrived_ids) / float(total_controlled)) if total_controlled > 0 else 0.0
        failed = max(total_controlled - len(arrived_ids), 0)
        print(
            "[EP {:03d} | STEP {:04d}] rollout={} updates={} policy_loss={} "
            "done={}/{} fail={} open/final/skip={:.0f}/{:.0f}/{:.0f} forced={:.0f}".format(
                episode,
                step,
                len(self.trainer.memory),
                self.trainer.train_steps,
                "n/a" if self.trainer.last_loss is None else f"{self.trainer.last_loss:.4f}",
                len(arrived_ids),
                total_controlled,
                failed,
                decision_metrics["decisions_opened"],
                decision_metrics["decisions_finalized"],
                decision_metrics["decisions_skipped"],
                decision_metrics["forced_actions"],
            )
        )
        print(
            "  rates: completion={:.1%} override={:.1%} mismatch={} teleports={}".format(
                completion,
                override_total / max(decision_metrics["decisions_opened"], 1.0),
                int(decision_metrics["route_mismatch"]),
                int(decision_metrics["teleports"]),
            )
        )
        mean_density = (
            float(np.mean(mean_density_samples))
            if mean_density_samples else 0.0
        )
        decisions_skipped_actionable = max(
            float(decision_metrics["decisions_skipped"]) - float(decision_metrics["skipped_pending_hold"]),
            0.0,
        )
        aggregate_actionable_skip_to_final = decisions_skipped_actionable / max(
            float(decision_metrics["decisions_finalized"]),
            1.0,
        )
        print(
            "  diagnostics: mean_density={:.4f} congested_steps={} pending_timeout={} "
            "lane_change(a/s/f)={:.0f}/{:.0f}/{:.0f} emergency_brake={} teleport(jam/yield)={:.0f}/{:.0f} "
            "actionable_skip_to_finalized={:.2f} skipped_pending_hold={:.0f}".format(
                mean_density,
                int(congestion_high_pressure_steps),
                int(decision_metrics["pending_decision_timeouts"]),
                decision_metrics["lane_change_attempts"],
                decision_metrics["lane_change_success"],
                decision_metrics["lane_change_fail"],
                int(decision_metrics["emergency_brake_events"]),
                decision_metrics["teleport_inferred_jam"],
                decision_metrics["teleport_inferred_yield_or_deadlock"],
                aggregate_actionable_skip_to_final,
                decision_metrics["skipped_pending_hold"],
            )
        )

    def _record_skip(self, decision_metrics, skip_bucket):
        key_map = {
            "pending_hold": "skipped_pending_hold",
            "structural_no_branch": "skipped_structural_no_branch",
            "structural_forced_single_path": "skipped_structural_forced_single_path",
            "structural_forced_by_lane_commit": "skipped_structural_forced_by_lane_commit",
            "structural_too_late_or_unreachable": "skipped_structural_too_late_or_unreachable",
            "actionable_no_candidate": "skipped_actionable_no_candidate",
        }
        key = key_map.get(skip_bucket, "skipped_other")
        decision_metrics[key] += 1
        decision_metrics["decisions_skipped"] = (
            decision_metrics["skipped_pending_hold"]
            + decision_metrics["skipped_structural_no_branch"]
            + decision_metrics["skipped_structural_forced_single_path"]
            + decision_metrics["skipped_structural_forced_by_lane_commit"]
            + decision_metrics["skipped_structural_too_late_or_unreachable"]
            + decision_metrics["skipped_actionable_no_candidate"]
            + decision_metrics["skipped_other"]
        )

    def _record_override_event(self, decision_metrics, subtype):
        key_map = {
            "loop_prefilter": "override_event_loop_prefilter",
            "cooldown_fallback": "override_event_cooldown_fallback",
            "observe_abort_fallback": "override_event_observe_abort_fallback",
            "route_apply_fail": "override_event_route_apply_fail",
            "invalid_action": "override_event_invalid_action",
        }
        decision_metrics["override_events_total"] += 1
        subtype_key = key_map.get(subtype)
        if subtype_key:
            decision_metrics[subtype_key] += 1

    def _next_decision_id(self, decision_metrics):
        decision_metrics["decision_id_sequence"] += 1
        return f"ep_dec_{int(decision_metrics['decision_id_sequence'])}"

    def _register_decision_open(self, decision_metrics, pending):
        """
        Count one unique strategic decision open per decision_id.
        Phase transitions (observe -> route_pending) must not re-open.
        """
        if getattr(pending, "decision_open_recorded", False):
            return
        pending.decision_open_recorded = True
        pending.metadata["decision_open_recorded"] = True
        decision_metrics["decisions_opened"] += 1
        origin_mode = str(getattr(pending, "decision_origin_mode", "") or pending.metadata.get("decision_origin_mode", "lane_now"))
        if origin_mode == "proactive":
            decision_metrics["proactive_decisions_opened"] += 1
        else:
            decision_metrics["lane_now_decisions_opened"] += 1

    def _register_decision_finalized(self, decision_metrics, pending):
        """
        Count finalized strategic decisions only (synthetic terminal transitions excluded).
        """
        decision_metrics["decisions_finalized"] += 1
        origin_mode = str(getattr(pending, "decision_origin_mode", "") or pending.metadata.get("decision_origin_mode", "lane_now"))
        if origin_mode == "proactive":
            decision_metrics["proactive_decisions_finalized"] += 1
        else:
            decision_metrics["lane_now_decisions_finalized"] += 1

    def _record_pending_release(self, decision_metrics, release_reason):
        key_map = {
            "observe_abort_no_progress": "pending_release_observe_abort_no_progress",
            "observe_abort_commit_window": "pending_release_observe_abort_commit_window",
            "observe_abort_low_speed": "pending_release_observe_abort_low_speed",
            "wrong_lane_commit": "pending_release_wrong_lane_commit",
            "route_no_progress_abort": "pending_release_route_no_progress_abort",
            "route_stall_timeout": "pending_release_route_stall_timeout",
            "route_hard_timeout": "pending_release_route_hard_timeout",
        }
        key = key_map.get(release_reason)
        if key:
            decision_metrics[key] += 1
        decision_metrics["pending_release_events_total"] += 1
        if release_reason in {"route_stall_timeout", "route_hard_timeout"}:
            decision_metrics["pending_release_timeout_events_total"] += 1
        else:
            decision_metrics["pending_release_abort_events_total"] += 1

    def _action_social_cost_proxy(self, current_edge, action_idx, destination):
        """
        Lower is better for "selfless" local decisions.
        Proxy combines short-horizon corridor pressure + residual distance.
        """
        stats = self.shared_policy.action_corridor_stats(
            edge_id=current_edge,
            action_idx=action_idx,
            destination=destination,
            edge_density_fn=self._edge_density,
            distance_fn=self.get_distance_to_destination,
            eta_fn=self._estimate_remaining_eta,
        )
        if stats is None:
            return float("inf")
        current_distance = self.get_distance_to_destination(current_edge, destination)
        cost = float(stats.score)
        if math.isfinite(current_distance) and math.isfinite(stats.next_distance):
            if stats.next_distance >= (current_distance - 1.0):
                cost += 0.45
            loop_distance_slack = float(getattr(self.decision_engine, "loop_distance_slack", 30.0))
            if stats.best_distance <= (current_distance - max(8.0, 0.25 * loop_distance_slack)):
                cost -= 0.10
        return float(max(cost, 0.0))

    def _edge_lane_count(self, edge_id):
        return max(len(self.connection_info.edge_lane_ids.get(edge_id, [])), 1)

    def _edge_lane_meters(self, edge_id):
        cached = self._edge_lane_meters_cache.get(edge_id) if hasattr(self, "_edge_lane_meters_cache") else None
        if cached is not None:
            return float(cached)
        edge_len = max(float(self.connection_info.edge_length_dict.get(edge_id, 5.0)), 5.0)
        return edge_len * float(self._edge_lane_count(edge_id))

    def _edge_density(self, edge_id, count=None):
        if count is None:
            count = self.connection_info.edge_vehicle_count.get(edge_id, 0)
        return (float(count) * float(self.density_scale_m)) / max(self._edge_lane_meters(edge_id), 5.0)

    def _occupied_density_p95(self, density_vec):
        occupied = density_vec[density_vec > 0.0]
        if occupied.size == 0:
            return 0.0
        return float(np.percentile(occupied, 95))

    def _init_edge_embeddings(self, seed=1337):
        """
        Fixed edge embeddings avoid fake ordinal structure from raw edge indices.
        """
        rng = np.random.default_rng(seed)
        self._edge_embeddings = {}
        for edge_id in self.connection_info.edge_list:
            emb = rng.normal(loc=0.0, scale=0.1, size=self.edge_embedding_dim).astype(np.float32)
            self._edge_embeddings[edge_id] = emb

    def _get_edge_embedding(self, edge_id):
        return self._edge_embeddings.get(
            edge_id,
            np.zeros(self.edge_embedding_dim, dtype=np.float32),
        )

    def parse_sumocfg(self, sumocfg_path):
        """
        Parse the SUMO config file and return net and route filenames.
        """
        dom = parse(sumocfg_path)
        net_file_node = dom.getElementsByTagName('net-file')
        route_file_node = dom.getElementsByTagName('route-files')
        net_file = net_file_node[0].attributes['value'].nodeValue
        route_file = route_file_node[0].attributes['value'].nodeValue
        return net_file, route_file

    def encode_state(self, vehicle_id, edge_id, destination_edge, context=None, vehicle=None, step=None, snapshot=None, coordination_state=None):
        """
        Build a state vector for the given edge using cached per-step densities.
        Appends route-candidate features from self._episode_route_obs (set at reroute epochs).
        """
        if context is None:
            if snapshot is not None:
                self._cache_metrics["snapshot_cache_hits"] += 1
            context = self.decision_engine.build_context(
                vehicle_id,
                edge_id,
                destination_edge,
                int(step or 0),
                snapshot=snapshot,
            )
        if vehicle is not None and step is None:
            step = int(snapshot.step) if snapshot is not None else 0
        base_state = self.shared_policy.encode_state(
            edge_id=edge_id,
            destination_edge=destination_edge,
            context=context,
            edge_embedding_fn=self._get_edge_embedding,
            edge_density_fn=self._edge_density,
            eta_fn=self._estimate_remaining_eta,
            social_cost_fn=self._action_social_cost_proxy,
            global_density_stats=(float(self._density_mean), float(self._density_std)),
            step=step,
            vehicle_start_time=(float(vehicle.start_time) if vehicle is not None else None),
            vehicle_wait_time_fn=self._vehicle_wait_time,
            lane_halting_density_fn=self._lane_halting_density,
            lane_occupancy_fn=self._lane_occupancy,
            include_coordination=True,
            coordination_state=coordination_state,
        )
        route_obs = self._episode_route_obs.get(vehicle_id)
        if route_obs is None:
            route_obs = np.zeros(self.route_obs_dim, dtype=np.float32)
        base_flat = np.asarray(base_state, dtype=np.float32).reshape(-1)
        return np.concatenate([base_flat, route_obs]).reshape(1, -1)
    
    def valid_actions_for_vehicle(self, vehicle_id, edge_id, destination_edge, step):
        context = self.decision_engine.build_context(vehicle_id, edge_id, destination_edge, step)
        return context.available_actions

    def _policy_action_candidates(
        self,
        context,
        recent_history,
        cooldown_active,
        destination,
        decision_metrics=None,
        coordination_state=None,
    ):
        actions = self.shared_policy.policy_action_candidates(
            context,
            recent_history=recent_history,
            cooldown_active=cooldown_active,
            destination=destination,
            distance_fn=self.get_distance_to_destination,
            edge_density_fn=self._edge_density,
            metrics=decision_metrics,
            distance_slack=self.score_slack,
            coordination_state=coordination_state,
        )
        return self.shared_policy.rank_policy_actions(
            context=context,
            actions=actions,
            destination=destination,
            distance_fn=self.get_distance_to_destination,
            edge_density_fn=self._edge_density,
            recent_history=recent_history,
            coordination_state=coordination_state,
        )

    def _select_fallback_action(self, context, blocked_action, destination, recent_history, lane_now_only=False, coordination_state=None):
        return self.shared_policy.select_fallback_action(
            context,
            blocked_action=blocked_action,
            destination=destination,
            recent_history=recent_history,
            distance_fn=self.get_distance_to_destination,
            edge_density_fn=self._edge_density,
            lane_now_only=lane_now_only,
            coordination_state=coordination_state,
        )

    def _record_recent_decision_attribution(self, decision_attribution_by_vehicle, vehicle_id, step, action_source, resolution_mode):
        decision_attribution_by_vehicle[str(vehicle_id)] = {
            "step": int(step),
            "action_source": str(action_source or ""),
            "resolution_mode": str(resolution_mode or "lane_now"),
            "is_fallback": ("fallback" in str(action_source or "")),
        }

    def _attribute_hard_brake_event(self, decision_metrics, decision_attribution_by_vehicle, vehicle_id, step):
        record = decision_attribution_by_vehicle.get(str(vehicle_id))
        if not record:
            decision_metrics["emergency_brake_without_recent_decision"] += 1
            return

        age = max(int(step) - int(record.get("step", step)), 0)
        if age > int(self.hard_brake_attribution_window_steps):
            decision_metrics["emergency_brake_without_recent_decision"] += 1
            return

        if bool(record.get("is_fallback", False)):
            decision_metrics["emergency_brake_after_fallback"] += 1
            return

        if str(record.get("resolution_mode", "lane_now")) == "proactive":
            decision_metrics["emergency_brake_after_proactive"] += 1
            return

        decision_metrics["emergency_brake_after_lane_now"] += 1

    def _consume_hard_brake_events(self, hard_brake_counts_by_vehicle, vehicle_id):
        vehicle_key = str(vehicle_id)
        count = int(hard_brake_counts_by_vehicle.get(vehicle_key, 0))
        if count > 0:
            hard_brake_counts_by_vehicle[vehicle_key] = 0
        return max(count, 0)
    def dist_to_end(self, vehicle_id, snapshot=None):
        """
        Distance (meters) from the vehicle to the end of its current lane.
        """
        if snapshot is not None:
            return max(float(snapshot.dist_to_end), 0.0)
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        lane_len = traci.lane.getLength(lane_id)
        lane_pos = traci.vehicle.getLanePosition(vehicle_id)
        return max(lane_len - lane_pos, 0.0)


    # =========================
    # Teleport detection helpers
    # =========================
    def get_teleport_ids(self):
        """
        Return a set of vehicle IDs that teleported this step.

        SUMO/TraCI API differs by version, so we try multiple methods.
        """
        teleported = set()

        # Most common in many SUMO versions:
        try:
            teleported.update(traci.simulation.getStartingTeleportIDList())
        except Exception:
            pass
        try:
            teleported.update(traci.simulation.getEndingTeleportIDList())
        except Exception:
            pass

        # Some versions expose a vehicle-level list:
        try:
            teleported.update(traci.vehicle.getTeleportingList())
        except Exception:
            pass

        return teleported

    def _lane_length(self, lane_id):
        if lane_id in self._lane_length_cache:
            return self._lane_length_cache[lane_id]
        lane_len = float(traci.lane.getLength(lane_id))
        self._lane_length_cache[lane_id] = lane_len
        return lane_len

    def _vehicle_wait_time(self, vehicle_id):
        vehicle_id = str(vehicle_id)
        cached = self._step_vehicle_wait_cache.get(vehicle_id)
        if cached is not None:
            return float(cached)
        if vehicle_id == "__terminal__":
            self._step_vehicle_wait_cache[vehicle_id] = 0.0
            return 0.0
        result = self._step_vehicle_results.get(vehicle_id) or {}
        wait_time = result.get(tc.VAR_WAITING_TIME)
        if wait_time is None:
            wait_time = 0.0
        wait_time = max(float(wait_time), 0.0)
        self._step_vehicle_wait_cache[vehicle_id] = wait_time
        return wait_time

    def _lane_occupancy(self, lane_id):
        cached = self._step_lane_occupancy_cache.get(lane_id)
        if cached is not None:
            return float(cached)
        try:
            occupancy = float(traci.lane.getLastStepOccupancy(lane_id))
        except Exception:
            occupancy = 0.0
        if occupancy > 1.0:
            occupancy /= 100.0
        occupancy = float(np.clip(occupancy, 0.0, 1.0))
        self._step_lane_occupancy_cache[lane_id] = occupancy
        return occupancy

    def _lane_halting_density(self, lane_id):
        cached = self._step_lane_halting_cache.get(lane_id)
        if cached is not None:
            return float(cached)
        try:
            halting = float(traci.lane.getLastStepHaltingNumber(lane_id))
        except Exception:
            halting = 0.0
        density = (halting * float(self.density_scale_m)) / max(self._lane_length(lane_id), 5.0)
        density = float(np.clip(density, 0.0, 1.0))
        self._step_lane_halting_cache[lane_id] = density
        return density

    def _initialize_edge_subscriptions(self):
        for edge_id in self._edge_list:
            traci.edge.subscribe(edge_id, self._edge_subscription_vars)

    def _ensure_vehicle_subscriptions(self, vehicle_ids):
        for vehicle_id in vehicle_ids:
            if vehicle_id in self._active_vehicle_subscriptions:
                continue
            try:
                traci.vehicle.subscribe(vehicle_id, self._vehicle_subscription_vars)
                self._active_vehicle_subscriptions.add(vehicle_id)
            except traci.TraCIException:
                continue

    def collect_vehicle_snapshots(self, vehicle_ids, step, vehicle_results=None):
        snapshots = {}
        edge_lane_count = self._edge_lane_count
        vehicle_results = vehicle_results or {}
        for vehicle_id in vehicle_ids:
            result = vehicle_results.get(vehicle_id) or {}
            try:
                edge_id = result.get(tc.VAR_ROAD_ID)
                if edge_id is None:
                    edge_id = traci.vehicle.getRoadID(vehicle_id)
            except traci.TraCIException:
                continue
            if edge_id not in self._passenger_edge_set:
                continue
            try:
                lane_id = result.get(tc.VAR_LANE_ID)
                if lane_id is None:
                    lane_id = traci.vehicle.getLaneID(vehicle_id)
                lane_index = result.get(tc.VAR_LANE_INDEX)
                if lane_index is None:
                    lane_index = traci.vehicle.getLaneIndex(vehicle_id)
                lane_position = result.get(tc.VAR_LANEPOSITION)
                if lane_position is None:
                    lane_position = traci.vehicle.getLanePosition(vehicle_id)
                speed = result.get(tc.VAR_SPEED)
                if speed is None:
                    speed = traci.vehicle.getSpeed(vehicle_id)
                lane_index = int(lane_index)
                lane_position = float(lane_position)
                speed = max(float(speed), 0.0)
            except traci.TraCIException:
                continue
            lane_length = self._lane_length(lane_id)
            lane_count = edge_lane_count(edge_id)
            dist_to_end = max(lane_length - lane_position, 0.0)
            snapshots[vehicle_id] = VehicleSnapshot(
                vehicle_id=vehicle_id,
                step=int(step),
                edge_id=edge_id,
                lane_id=lane_id,
                lane_index=lane_index,
                lane_count=lane_count,
                lane_position=lane_position,
                lane_length=lane_length,
                dist_to_end=dist_to_end,
                speed=speed,
            )
        return snapshots

    def _get_or_build_step_context(
        self,
        context_cache,
        vehicle_id,
        edge_id,
        destination_edge,
        step,
        snapshot=None,
    ):
        key = (str(vehicle_id), edge_id, destination_edge, int(step))
        cached = context_cache.get(key)
        if cached is not None:
            self._cache_metrics["snapshot_cache_hits"] += 1
            return cached
        context = self.decision_engine.build_context(
            str(vehicle_id),
            edge_id,
            destination_edge,
            int(step),
            snapshot=snapshot,
        )
        context_cache[key] = context
        return context

    def _get_or_encode_step_state(
        self,
        state_cache,
        context_cache,
        vehicle_id,
        vehicle,
        step,
        snapshot,
        context=None,
        coordination_state=None,
    ):
        coord_key = None
        if coordination_state is not None:
            coord_key = (
                round(float(coordination_state.reserved_agents), 3),
                tuple(sorted((str(edge), float(load)) for edge, load in coordination_state.next_edge_loads.items())),
                tuple(sorted((str(edge), float(load)) for edge, load in coordination_state.corridor_edge_loads.items())),
                tuple(sorted((str(dest), float(load)) for dest, load in coordination_state.destination_loads.items())),
            )
        key = (str(vehicle_id), snapshot.edge_id, vehicle.destination, int(step), coord_key)
        cached = state_cache.get(key)
        if cached is not None:
            self._cache_metrics["snapshot_cache_hits"] += 1
            return cached
        if context is None:
            context = self._get_or_build_step_context(
                context_cache,
                vehicle_id,
                snapshot.edge_id,
                vehicle.destination,
                step,
                snapshot=snapshot,
            )
        state = self.encode_state(
            vehicle_id,
            snapshot.edge_id,
            vehicle.destination,
            context=context,
            vehicle=vehicle,
            step=step,
            snapshot=snapshot,
            coordination_state=coordination_state,
        )
        state_cache[key] = state
        return state

    def cleanup_vehicle_state(
        self,
        vehicle_id,
        pending_decisions,
        prev_edge_by_vehicle,
        last_seen_edge_by_vehicle,
        last_planned_terminal_edge_by_vehicle,
        recent_edge_history,
        last_snapshot_by_vehicle,
    ):
        pending_decisions.pop(vehicle_id, None)
        prev_edge_by_vehicle.pop(vehicle_id, None)
        last_seen_edge_by_vehicle.pop(vehicle_id, None)
        last_planned_terminal_edge_by_vehicle.pop(vehicle_id, None)
        self._episode_route_obs.pop(vehicle_id, None)
        last_snapshot_by_vehicle.pop(vehicle_id, None)
        recent_edge_history.pop(vehicle_id, None)

    def make_terminal_next_state_from_snapshot(self, snapshot, destination_edge, vehicle=None, step=None):
        if snapshot is None:
            return self.make_terminal_next_state_from_edge(None, destination_edge, vehicle=vehicle, step=step)
        return self.encode_state(
            snapshot.vehicle_id,
            snapshot.edge_id,
            destination_edge,
            context=self.decision_engine.build_context(
                snapshot.vehicle_id,
                snapshot.edge_id,
                destination_edge,
                int(step if step is not None else snapshot.step),
                snapshot=snapshot,
            ),
            vehicle=vehicle,
            step=step if step is not None else snapshot.step,
            snapshot=snapshot,
        )

    def make_terminal_next_state_from_edge(self, edge_id, destination_edge, vehicle=None, step=None):
        if (
            edge_id in self.connection_info.edge_index_dict
            and destination_edge in self.connection_info.edge_index_dict
        ):
            terminal_context = self.decision_engine.build_context(
                vehicle_id="__terminal__",
                edge_id=edge_id,
                destination=destination_edge,
                step=int(step or 0),
                snapshot=VehicleSnapshot(
                    vehicle_id="__terminal__",
                    step=int(step or 0),
                    edge_id=edge_id,
                    lane_id=self.connection_info.edge_lane_ids.get(edge_id, [""])[0] if self.connection_info.edge_lane_ids.get(edge_id) else "",
                    lane_index=0,
                    lane_count=max(len(self.connection_info.edge_lane_ids.get(edge_id, [])), 1),
                    lane_position=0.0,
                    lane_length=float(self.connection_info.edge_length_dict.get(edge_id, 0.0)),
                    dist_to_end=float(self.connection_info.edge_length_dict.get(edge_id, 0.0)),
                    speed=0.0,
                ),
            )
            return self.encode_state(
                "__terminal__",
                edge_id,
                destination_edge,
                context=terminal_context,
                vehicle=vehicle,
                step=step,
            )
        return np.zeros((1, self.state_size), dtype=np.float32)

    def _classify_terminal_outcome(self, vehicle_id, arrived_ids, teleport_ids, is_live):
        """
        Classify terminal outcome for a controlled vehicle removed from the live set.
        Arrival classification is strictly based on SUMO arrived ids.
        """
        if vehicle_id in arrived_ids:
            return "global_arrival"
        if (vehicle_id in teleport_ids) and (not is_live):
            return "teleport"
        return "removed_nonarrival"

    def _finalize_terminal_transition(
        self,
        vehicle_id,
        vehicle,
        outcome,
        step,
        pending_decisions,
        terminal_recorded_ids,
        last_snapshot_by_vehicle,
        last_seen_edge_by_vehicle,
        decision_metrics,
        decision_latency_steps,
        finalized_decision_rewards,
        decision_debug_rows,
        episode,
        hard_brake_counts_by_vehicle,
        in_arrived_ids=False,
        in_teleport_ids=False,
        ever_teleported=False,
        next_central_observation=None,
    ):
        if vehicle_id in terminal_recorded_ids:
            return 0.0
        pending = pending_decisions.get(vehicle_id)
        hard_brake_events = self._consume_hard_brake_events(hard_brake_counts_by_vehicle, vehicle_id)
        if pending is None:
            last_confirmed_edge = last_seen_edge_by_vehicle.get(vehicle_id, vehicle.destination)
            snapshot = last_snapshot_by_vehicle.get(vehicle_id)
            base_state = self.make_terminal_next_state_from_snapshot(
                snapshot,
                vehicle.destination,
                vehicle=vehicle,
                step=step,
            )
            if outcome == "global_arrival":
                reward, done = self.compute_reward(
                    vehicle,
                    vehicle.destination,
                    vehicle.destination,
                    step,
                    arrived=True,
                    reached_global_destination=True,
                    hard_brake_events=hard_brake_events,
                    terminal_outcome=outcome,
                )
                if ever_teleported:
                    teleport_assisted_arrival_penalty = 8.0
                    reward = self._clip_reward(reward - teleport_assisted_arrival_penalty)
                next_state = self.make_terminal_next_state_from_edge(
                    vehicle.destination,
                    vehicle.destination,
                    vehicle=vehicle,
                    step=step,
                )
            elif outcome == "teleport":
                reward = self._clip_reward(self.teleport_penalty)
                done = True
                next_state = base_state
                decision_metrics["fail_teleport"] += 1
            elif outcome == "timeout":
                reward = self._clip_reward(self.pending_timeout_penalty)
                done = True
                next_state = base_state
                decision_metrics["fail_timeout"] += 1
            elif outcome == "removed_nonarrival":
                reward = self._clip_reward(self.stale_disappeared_penalty)
                done = True
                next_state = base_state
                decision_metrics["fail_removed_non_destination"] += 1
            else:
                raise ValueError(f"Unknown terminal outcome: {outcome}")
            decision_metrics["synthetic_terminal_finalizations"] += 1
            # Patch the last recorded MAPPO transition for this vehicle with the terminal
            # reward and done=True. When pending is None, the vehicle arrived (or ended)
            # after its last pending decision was already finalized, so the arrival signal
            # would otherwise be lost from the policy's learning buffer.
            entry = self._vehicle_last_buffer_pos.pop(vehicle_id, None)
            if entry is not None:
                buf_idx, gen = entry
                if gen == self.trainer.buffer_generation and buf_idx < len(self.trainer.buffer):
                    t = self.trainer.buffer[buf_idx]
                    t.reward = self._clip_reward(t.reward + reward)
                    t.done = True
            decision_debug_rows.append({
                "episode": episode,
                "step": step,
                "vehicle_id": vehicle_id,
                "decision_edge": last_confirmed_edge,
                "action": 0,
                "action_source": "synthetic_terminal",
                "available_actions": "[]",
                "forced_action": "",
                "lane_feasible_now_actions": "[]",
                "reachable_with_lane_change_actions": "[]",
                "commit_window": "",
                "dist_to_end": "",
                "intended_next_edge": "",
                "actual_next_edge": vehicle.destination if outcome == "global_arrival" else last_confirmed_edge,
                "finalized": 1,
                "finalize_delay_steps": 0,
                "reward": reward,
                "done": 1,
                "route_mismatch": "",
                "teleported": 1 if outcome == "teleport" else 0,
                "reached_global_destination": 1 if outcome == "global_arrival" else 0,
                "prev_eta": "",
                "curr_eta": "",
                "prev_distance": "",
                "curr_distance": "",
                "edge_density": "",
                "mean_density": "",
                "externality_penalty": "",
                "marginal_pressure": "",
                "terminal_outcome": outcome,
                "last_confirmed_edge": last_confirmed_edge,
                "destination": vehicle.destination,
                "in_arrived_ids": int(bool(in_arrived_ids)),
                "in_teleport_ids": int(bool(in_teleport_ids)),
            })
            terminal_recorded_ids.add(vehicle_id)
            return float(reward)

        last_confirmed_edge = last_seen_edge_by_vehicle.get(vehicle_id, pending.last_credit_edge)
        snapshot = last_snapshot_by_vehicle.get(vehicle_id)
        if outcome == "global_arrival":
            reward, done = self.compute_reward(
                vehicle,
                pending.last_credit_edge,
                vehicle.destination,
                step,
                arrived=True,
                delta_t=max(step - pending.last_credit_step, 1),
                reached_global_destination=True,
                coordination_pressure=float((pending.metadata or {}).get("coordination_pressure", 0.0)),
                hard_brake_events=hard_brake_events,
                terminal_outcome=outcome,
            )
            if ever_teleported:
                teleport_assisted_arrival_penalty = 8.0
                reward = self._clip_reward(reward - teleport_assisted_arrival_penalty)
            next_state = self.make_terminal_next_state_from_edge(
                vehicle.destination,
                vehicle.destination,
                vehicle=vehicle,
                step=step,
            )
            decision_metrics["arrived_global_destination"] += 1
        elif outcome == "teleport":
            reward = self._clip_reward(self.teleport_penalty)
            done = True
            next_state = self.make_terminal_next_state_from_snapshot(
                snapshot,
                vehicle.destination,
                vehicle=vehicle,
                step=step,
            )
            decision_metrics["fail_teleport"] += 1
        elif outcome == "timeout":
            timeout_elapsed = max(step - pending.last_credit_step, 0)
            reward = self.compute_pending_step_reward(
                vehicle,
                last_confirmed_edge,
                elapsed=timeout_elapsed,
                step=step,
                pending_age=max(step - pending.decision_step, 0),
                lane_change_deferrals=pending.metadata.get("lane_change_deferrals", 0),
                hard_brake_events=hard_brake_events,
                coordination_pressure=float((pending.metadata or {}).get("coordination_pressure", 0.0)),
            ) + self.pending_timeout_penalty
            reward = self._clip_reward(reward)
            done = True
            next_state = self.make_terminal_next_state_from_snapshot(
                snapshot,
                vehicle.destination,
                vehicle=vehicle,
                step=step,
            )
            decision_metrics["fail_timeout"] += 1
        elif outcome == "removed_nonarrival":
            reward = self._clip_reward(self.stale_disappeared_penalty)
            done = True
            next_state = self.make_terminal_next_state_from_snapshot(
                snapshot,
                vehicle.destination,
                vehicle=vehicle,
                step=step,
            )
            decision_metrics["fail_removed_non_destination"] += 1
        else:
            raise ValueError(f"Unknown terminal outcome: {outcome}")

        final_metadata = dict(pending.metadata) if isinstance(pending.metadata, dict) else {}
        final_metadata["terminal_outcome"] = outcome
        final_metadata["decision_finalized"] = True
        if ever_teleported:
            final_metadata["ever_teleported"] = True
            final_metadata["teleport_assisted_arrival"] = (outcome == "global_arrival")
        self._record_pending_mappo_transition(
            pending,
            reward=reward,
            next_state=next_state,
            next_central_observation=(
                self._zero_central_observation()
                if next_central_observation is None else next_central_observation
            ),
            done=done,
            discount_steps=max(step - pending.decision_step, 1),
            metadata=final_metadata,
        )
        pending_decisions.pop(vehicle_id, None)
        self._register_decision_finalized(decision_metrics, pending)
        decision_latency_steps.append(float(max(step - pending.decision_step, 0)))
        finalized_decision_rewards.append(float(reward))
        terminal_recorded_ids.add(vehicle_id)
        decision_debug_rows.append(self._build_decision_debug_row(
            episode=episode,
            step=step,
            pending=pending,
            actual_next_edge=vehicle.destination if outcome == "global_arrival" else last_confirmed_edge,
            finalized=1,
            finalize_delay_steps=max(step - pending.decision_step, 0),
            reward=reward,
            done=1,
            teleported=1 if outcome == "teleport" else 0,
            reached_global_destination=1 if outcome == "global_arrival" else 0,
            terminal_outcome=outcome,
            last_confirmed_edge=last_confirmed_edge,
            destination=vehicle.destination,
            in_arrived_ids=int(bool(in_arrived_ids)),
            in_teleport_ids=int(bool(in_teleport_ids)),
        ))
        return float(reward)

    def _edge_out_degree_map(self, edges):
        return {
            edge: len(self.connection_info.outgoing_edges_dict.get(edge, {}))
            for edge in set(edges)
        }

    def _estimate_remaining_eta(self, edge_id, destination_edge):
        """
        Estimate travel time from edge_id to destination using shortest-path
        distance and a conservative minimum speed floor.
        """
        distance = self.get_distance_to_destination(edge_id, destination_edge)
        if not math.isfinite(distance):
            return math.inf
        return float(distance) / 8.0

    def get_distance_to_destination(self, edge_id, destination_edge):
        """
        Return shortest-path cost from edge_id to destination_edge.
        Uses a cache because this is called frequently during training.
        """
        key = (edge_id, destination_edge)
        if key in self._distance_cache:
            self._cache_metrics["shortest_path_cache_hits"] += 1
            return self._distance_cache[key]

        try:
            from_edge = self.net.getEdge(edge_id)
            to_edge = self.net.getEdge(destination_edge)
        except Exception:
            self._distance_cache[key] = math.inf
            return math.inf

        path_edges, path_cost = self.net.getShortestPath(from_edge, to_edge, vClass="passenger")
        distance = path_cost if path_edges is not None else math.inf
        self._distance_cache[key] = distance
        return distance

    def _tail_delay_threshold(self, vehicle, reference_edge):
        cached = getattr(vehicle, "tail_delay_threshold_steps", None)
        if cached is not None:
            return float(cached)
        initial_eta = self._estimate_remaining_eta(reference_edge, vehicle.destination)
        threshold = float(
            np.clip(
                self.tail_delay_threshold_eta_mult * float(initial_eta),
                self.tail_delay_threshold_min_steps,
                self.tail_delay_threshold_max_steps,
            )
        )
        setattr(vehicle, "tail_delay_threshold_steps", threshold)
        return threshold

    def _tail_delay_penalty_increment(self, vehicle, reference_edge, step, delta_steps):
        delta_steps = max(float(delta_steps), 0.0)
        if delta_steps <= 0.0:
            return 0.0
        elapsed_now = max(float(step) - float(vehicle.start_time), 0.0)
        elapsed_prev = max(elapsed_now - delta_steps, 0.0)
        threshold = self._tail_delay_threshold(vehicle, reference_edge)
        overflow_now = max(elapsed_now - threshold, 0.0)
        overflow_prev = max(elapsed_prev - threshold, 0.0)
        if overflow_now <= overflow_prev:
            return 0.0
        linear = self.tail_delay_linear_penalty * (overflow_now - overflow_prev)
        quadratic = self.tail_delay_quadratic_penalty * ((overflow_now ** 2) - (overflow_prev ** 2))
        return float(max(linear + quadratic, 0.0))

    def _tail_arrival_penalty(self, vehicle, reference_edge, step):
        threshold = self._tail_delay_threshold(vehicle, reference_edge)
        elapsed_now = max(float(step) - float(vehicle.start_time), 0.0)
        overflow = max(elapsed_now - threshold, 0.0)
        if overflow <= 0.0:
            return 0.0
        penalty = self.tail_arrival_penalty_per_25_steps * (overflow / 25.0)
        return float(min(penalty, self.tail_arrival_penalty_cap))
    
    def compute_reward(
        self,
        vehicle,
        prev_edge,
        current_edge,
        step,
        arrived,
        repeated_recent_edges=0,
        delta_t=1.0,
        reached_global_destination=False,
        route_mismatch=False,
        invalid_late_turn=False,
        route_apply_failed=False,
        uturn_repeat=False,
        long_horizon_loop=False,
        externality_penalty=0.0,
        selfless_delta=0.0,
        coordination_pressure=0.0,
        hard_brake_events=0,
        terminal_outcome=None,
    ):
        """
        Compute a bounded reward with clear objective priority:
        1) minimize travel time (per-step cost)
        2) complete trips successfully
        3) prefer progress with mild congestion awareness.
        """
        elapsed = max(float(delta_t), 1.0)

        prev_distance = self.get_distance_to_destination(prev_edge, vehicle.destination)
        curr_distance = self.get_distance_to_destination(current_edge, vehicle.destination)
        prev_eta = self._estimate_remaining_eta(prev_edge, vehicle.destination)
        curr_eta = self._estimate_remaining_eta(current_edge, vehicle.destination)

        reward = 0.0
        done = False

        # Dense shaping: small living/time and congestion costs.
        time_cost_scale = self._get_route_difficulty_scale(vehicle, prev_edge)
        reward -= self.travel_time_penalty * time_cost_scale * elapsed
        edge_density = self._edge_density(current_edge)
        reward -= 0.005 * edge_density * elapsed

        mean_density = float(self._density_mean)
        marginal_pressure = max(edge_density - mean_density, 0.0)
        reward -= self.system_congestion_scale * marginal_pressure * elapsed
        reward += self.selfless_reward_scale * float(
            np.clip(selfless_delta, -self.selfless_reward_clip, self.selfless_reward_clip)
        )
        reward -= self.coordination_pressure_penalty * float(np.clip(coordination_pressure, 0.0, 6.0))
        reward -= self._tail_delay_penalty_increment(
            vehicle,
            current_edge,
            step=step,
            delta_steps=elapsed,
        )

        # Progress shaping using ETA and distance improvement.
        if math.isfinite(prev_eta) and math.isfinite(curr_eta):
            reward += self.eta_progress_scale * np.clip(prev_eta - curr_eta, -3.0, 3.0)

        # Tertiary tie-breaker: shortest-path distance progress.
        if math.isfinite(prev_distance) and math.isfinite(curr_distance):
            progress = (prev_distance - curr_distance) * (self.distance_tiebreak_scale * self.progress_reward_scale)
            reward += float(np.clip(progress, -1.0, 1.0))

        # Safety and control quality penalties.
        if repeated_recent_edges > 0:
            reward -= self.loop_repeat_penalty * min(repeated_recent_edges, 3)
        if uturn_repeat:
            reward -= 3.0
        if long_horizon_loop:
            reward -= 4.0
        if route_mismatch:
            reward -= 4.0
        if invalid_late_turn:
            reward -= 2.0
        if route_apply_failed:
            reward -= 6.0
        if hard_brake_events > 0:
            reward -= self.hard_brake_event_penalty * min(float(hard_brake_events), 2.0)

        # Unreachable transition after a decision is strongly terminal-negative.
        if math.isfinite(prev_distance) and not math.isfinite(curr_distance):
            reward -= 12.0
            done = True

        if arrived:
            if reached_global_destination:
                reward += self.destination_reward
                speed_bonus = max(0.0, 1.0 - (float(step) / float(MAX_SIMULATION_STEPS)))
                reward += 3.0 * speed_bonus
                reward -= self._tail_arrival_penalty(vehicle, current_edge, step)
            else:
                reward -= 8.0
            done = True
            # Use wider clip for terminal transitions so destination_reward
            # is not uniformly saturated at reward_clip_high.
            return float(np.clip(reward, self.reward_clip_low, self.terminal_reward_clip_high)), done

        outgoing = self.connection_info.outgoing_edges_dict.get(current_edge, {})
        if (not outgoing or len(outgoing) == 0) and current_edge != vehicle.destination:
            reward -= 12.0
            done = True

        return self._clip_reward(reward), done

    def _clip_reward(self, reward_value):
        return float(np.clip(reward_value, self.reward_clip_low, self.reward_clip_high))

    def compute_pending_step_reward(self, vehicle, edge_id, elapsed, step, externality_penalty=0.0, pending_age=0, lane_change_deferrals=0, hard_brake_events=0, coordination_pressure=0.0):
        """
        Dense reward used while a decision is pending and has not finalized yet.
        Keeps the training objective travel-time centric without waiting for an edge transition.
        """
        elapsed = max(float(elapsed), 0.0)
        if elapsed <= 0.0:
            return 0.0
        time_cost_scale = self._get_route_difficulty_scale(vehicle, edge_id)
        reward = -self.travel_time_penalty * time_cost_scale * elapsed

        edge_density = self._edge_density(edge_id)
        reward -= 0.005 * edge_density * elapsed

        mean_density = float(self._density_mean)
        marginal_pressure = max(edge_density - mean_density, 0.0)
        reward -= self.system_congestion_scale * marginal_pressure * elapsed
        reward -= self.pending_coordination_penalty * float(np.clip(coordination_pressure, 0.0, 6.0)) * elapsed
        reward -= self.pending_latency_penalty_per_step * float(max(pending_age, 0))
        reward -= 0.02 * float(max(lane_change_deferrals, 0))
        if hard_brake_events > 0:
            reward -= (0.6 * self.hard_brake_event_penalty) * min(float(hard_brake_events), 2.0)
        reward -= self._tail_delay_penalty_increment(
            vehicle,
            edge_id,
            step=step,
            delta_steps=elapsed,
        )
        return self._clip_reward(reward)

    def _default_best_model_output_path(self, model_output_path):
        root, ext = os.path.splitext(model_output_path)
        if ext:
            return f"{root}.best{ext}"
        return model_output_path + ".best"

    def _default_frozen_eval_seeds(self):
        return list(range(6000, 6011))

    def _frozen_eval_csv_fields(self):
        return [
            "episode",
            "seed_count",
            "seed_list",
            "spawn_interval",
            "algorithm",
            "baseline_seed_source",
            "win_count",
            "win_rate",
            "completion_rate_mean",
            "baseline_completion_rate_mean",
            "completion_rate_delta_mean",
            "avg_travel_time_mean",
            "baseline_avg_travel_time_mean",
            "avg_travel_time_delta_mean",
            "p50_travel_time_mean",
            "p90_travel_time_mean",
            "baseline_p90_travel_time_mean",
            "p90_travel_time_delta_mean",
            "timeout_rate_mean",
            "baseline_timeout_rate_mean",
            "timeout_rate_delta_mean",
            "tail_completion_gap_steps_mean",
            "baseline_tail_completion_gap_steps_mean",
            "tail_completion_gap_steps_delta_mean",
            "p95_to_p50_travel_ratio_mean",
            "baseline_p95_to_p50_travel_ratio_mean",
            "p95_to_p50_travel_ratio_delta_mean",
            "deadlines_missed_mean",
            "baseline_deadlines_missed_mean",
            "deadlines_missed_delta_mean",
            "vehicles_reached_destination_mean",
            "controlled_vehicle_count_mean",
            "route_actor_epochs_mean",
            "route_choice_nonzero_rate_mean",
            "route_mean_valid_candidates_mean",
            "route_mean_logit_margin_mean",
            "best_checkpoint_updated",
            "score_key",
        ]

    def _ensure_frozen_eval_csv_header(self):
        if self.eval_every <= 0:
            return
        with open(self.frozen_eval_metrics_csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._frozen_eval_csv_fields()).writeheader()

    def _safe_eval_metric(self, value, default=1.0e9):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(numeric):
            return float(default)
        return numeric

    def _travel_score_tuple(self, stats):
        return (
            1.0 - float(np.clip(stats.get("completion_rate", 0.0), 0.0, 1.0)),
            self._safe_eval_metric(stats.get("timeout_rate", 1.0)),
            self._safe_eval_metric(stats.get("avg_travel_time", float("inf"))),
            self._safe_eval_metric(stats.get("p90_travel_time", float("inf"))),
            self._safe_eval_metric(stats.get("tail_completion_gap_steps", float("inf"))),
            self._safe_eval_metric(stats.get("p95_to_p50_travel_ratio", float("inf"))),
            self._safe_eval_metric(stats.get("deadlines_missed", float("inf"))),
        )

    def _frozen_eval_score_key(self, summary):
        # Rank deployment checkpoints by absolute policy quality. Baseline deltas are
        # useful diagnostics, but the best checkpoint should first avoid tail collapse.
        return (
            1.0 - float(np.clip(summary.get("completion_rate_mean", 0.0), 0.0, 1.0)),
            self._safe_eval_metric(summary.get("timeout_rate_mean", 1.0)),
            self._safe_eval_metric(summary.get("avg_travel_time_mean", float("inf"))),
            self._safe_eval_metric(summary.get("p90_travel_time_mean", float("inf"))),
            self._safe_eval_metric(summary.get("tail_completion_gap_steps_mean", float("inf"))),
            self._safe_eval_metric(summary.get("p95_to_p50_travel_ratio_mean", float("inf"))),
            self._safe_eval_metric(summary.get("deadlines_missed_mean", float("inf"))),
        )

    def _append_frozen_eval_row(self, row):
        if self.eval_every <= 0:
            return
        fields = self._frozen_eval_csv_fields()
        serializable_row = {field: row.get(field, "") for field in fields}
        with open(self.frozen_eval_metrics_csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(serializable_row)

    def _save_best_frozen_checkpoint(self, episode, aggregate_summary, per_seed_rows):
        score_key = self._frozen_eval_score_key(aggregate_summary)
        improved = self._best_frozen_eval_key is None or score_key < self._best_frozen_eval_key
        if not improved:
            return False, score_key

        os.makedirs(os.path.dirname(self.best_model_output_path) or ".", exist_ok=True)
        self.trainer.save_checkpoint(self.best_model_output_path)
        metadata = {
            "episode": int(episode),
            "score_key": list(score_key),
            "aggregate": aggregate_summary,
            "per_seed": per_seed_rows,
        }
        with open(self.best_model_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        self._best_frozen_eval_key = score_key
        self._best_frozen_eval_summary = metadata
        return True, score_key

    def _run_frozen_inference_eval(self, episode, sumo_binary):
        if self.eval_every <= 0:
            return None

        os.makedirs(os.path.dirname(self._frozen_eval_model_path) or ".", exist_ok=True)
        self.trainer.save_checkpoint(self._frozen_eval_model_path)

        def run_eval_controller(controller, vehicles):
            simulation = StrSumo(controller, self.connection_info, vehicles)
            try:
                traci.start([
                    sumo_binary,
                    "-c", self.runtime_sumocfg_path,
                    "--quit-on-end",
                    "--no-step-log",
                    "--no-warnings",
                ])
                _, _, _, stats = simulation.run(verbose=False, return_stats=True, print_runtime_summary=False)
            finally:
                try:
                    traci.close()
                except Exception:
                    pass
            return stats

        per_seed_rows = []
        for eval_seed in self.frozen_eval_seeds:
            vehicles = self.generate_episode_vehicles(
                episode_seed=int(eval_seed),
                spawn_interval_override=self.eval_spawn_interval,
            )
            dijkstra_cache_key = (
                int(eval_seed),
                float(self.eval_spawn_interval),
                int(self.target_pattern),
            )
            baseline_stats = self._frozen_eval_baseline_cache.get(dijkstra_cache_key)
            if baseline_stats is None:
                baseline_policy = DijkstraPolicy(self.connection_info)
                baseline_vehicles = copy.deepcopy(vehicles)
                baseline_stats = run_eval_controller(
                    baseline_policy,
                    baseline_vehicles,
                )
                self._frozen_eval_baseline_cache[dijkstra_cache_key] = dict(baseline_stats)

            rl_vehicles = copy.deepcopy(vehicles)
            policy = MAPPOPolicy(
                rl_vehicles,
                self.connection_info,
                self._frozen_eval_model_path,
                net_xml_file=os.path.join(self.sumocfg_dir, self.net_file),
            )
            stats = run_eval_controller(policy, rl_vehicles)
            runtime_metrics = stats.get("controller_runtime_metrics") or {}
            rl_score = self._travel_score_tuple(stats)
            baseline_score = self._travel_score_tuple(baseline_stats)
            win = int(rl_score < baseline_score)
            per_seed_rows.append({
                "seed": int(eval_seed),
                "win_vs_dijkstra": int(win),
                "completion_rate": float(stats["completion_rate"]),
                "baseline_completion_rate": float(baseline_stats["completion_rate"]),
                "completion_rate_delta": float(stats["completion_rate"]) - float(baseline_stats["completion_rate"]),
                "avg_travel_time": float(stats["avg_travel_time"]),
                "baseline_avg_travel_time": float(baseline_stats["avg_travel_time"]),
                "avg_travel_time_delta": float(stats["avg_travel_time"]) - float(baseline_stats["avg_travel_time"]),
                "p50_travel_time": float(stats["p50_travel_time"]),
                "p90_travel_time": float(stats["p90_travel_time"]),
                "baseline_p90_travel_time": float(baseline_stats["p90_travel_time"]),
                "p90_travel_time_delta": float(stats["p90_travel_time"]) - float(baseline_stats["p90_travel_time"]),
                "timeout_rate": float(stats["timeout_rate"]),
                "baseline_timeout_rate": float(baseline_stats["timeout_rate"]),
                "timeout_rate_delta": float(stats["timeout_rate"]) - float(baseline_stats["timeout_rate"]),
                "tail_completion_gap_steps": float(stats["tail_completion_gap_steps"]),
                "baseline_tail_completion_gap_steps": float(baseline_stats["tail_completion_gap_steps"]),
                "tail_completion_gap_steps_delta": float(stats["tail_completion_gap_steps"]) - float(baseline_stats["tail_completion_gap_steps"]),
                "p95_to_p50_travel_ratio": float(stats["p95_to_p50_travel_ratio"]),
                "baseline_p95_to_p50_travel_ratio": float(baseline_stats["p95_to_p50_travel_ratio"]),
                "p95_to_p50_travel_ratio_delta": float(stats["p95_to_p50_travel_ratio"]) - float(baseline_stats["p95_to_p50_travel_ratio"]),
                "deadlines_missed": float(stats["deadlines_missed"]),
                "baseline_deadlines_missed": float(baseline_stats["deadlines_missed"]),
                "deadlines_missed_delta": float(stats["deadlines_missed"]) - float(baseline_stats["deadlines_missed"]),
                "vehicles_reached_destination": float(stats["vehicles_reached_destination"]),
                "controlled_vehicle_count": float(stats["controlled_vehicle_count"]),
                "route_actor_epochs": float(runtime_metrics.get("route_actor_epochs_started", 0.0)),
                "route_choice_nonzero_rate": float(runtime_metrics.get("route_choice_nonzero_rate", 0.0)),
                "route_mean_valid_candidates": float(runtime_metrics.get("route_mean_valid_candidates", 0.0)),
                "route_mean_logit_margin": float(runtime_metrics.get("route_mean_logit_margin", 0.0)),
            })

        def mean_metric(key, default=0.0):
            values = [float(row[key]) for row in per_seed_rows]
            if not values:
                return float(default)
            return float(np.mean(values))

        aggregate_summary = {
            "episode": int(episode),
            "seed_count": int(len(per_seed_rows)),
            "seed_list": ",".join(str(row["seed"]) for row in per_seed_rows),
            "spawn_interval": float(self.eval_spawn_interval),
            "algorithm": "mappo",
            "baseline_seed_source": ",".join(str(seed) for seed in self.frozen_eval_seeds),
            "win_count": float(sum(float(row["win_vs_dijkstra"]) for row in per_seed_rows)),
            "win_rate": mean_metric("win_vs_dijkstra", 0.0),
            "completion_rate_mean": mean_metric("completion_rate", 0.0),
            "baseline_completion_rate_mean": mean_metric("baseline_completion_rate", 0.0),
            "completion_rate_delta_mean": mean_metric("completion_rate_delta", -1.0),
            "avg_travel_time_mean": mean_metric("avg_travel_time", float("inf")),
            "baseline_avg_travel_time_mean": mean_metric("baseline_avg_travel_time", float("inf")),
            "avg_travel_time_delta_mean": mean_metric("avg_travel_time_delta", float("inf")),
            "p50_travel_time_mean": mean_metric("p50_travel_time", float("inf")),
            "p90_travel_time_mean": mean_metric("p90_travel_time", float("inf")),
            "baseline_p90_travel_time_mean": mean_metric("baseline_p90_travel_time", float("inf")),
            "p90_travel_time_delta_mean": mean_metric("p90_travel_time_delta", float("inf")),
            "timeout_rate_mean": mean_metric("timeout_rate", 1.0),
            "baseline_timeout_rate_mean": mean_metric("baseline_timeout_rate", 1.0),
            "timeout_rate_delta_mean": mean_metric("timeout_rate_delta", 1.0),
            "tail_completion_gap_steps_mean": mean_metric("tail_completion_gap_steps", float("inf")),
            "baseline_tail_completion_gap_steps_mean": mean_metric("baseline_tail_completion_gap_steps", float("inf")),
            "tail_completion_gap_steps_delta_mean": mean_metric("tail_completion_gap_steps_delta", float("inf")),
            "p95_to_p50_travel_ratio_mean": mean_metric("p95_to_p50_travel_ratio", float("inf")),
            "baseline_p95_to_p50_travel_ratio_mean": mean_metric("baseline_p95_to_p50_travel_ratio", float("inf")),
            "p95_to_p50_travel_ratio_delta_mean": mean_metric("p95_to_p50_travel_ratio_delta", float("inf")),
            "deadlines_missed_mean": mean_metric("deadlines_missed", float("inf")),
            "baseline_deadlines_missed_mean": mean_metric("baseline_deadlines_missed", float("inf")),
            "deadlines_missed_delta_mean": mean_metric("deadlines_missed_delta", float("inf")),
            "vehicles_reached_destination_mean": mean_metric("vehicles_reached_destination", 0.0),
            "controlled_vehicle_count_mean": mean_metric("controlled_vehicle_count", 0.0),
            "route_actor_epochs_mean": mean_metric("route_actor_epochs", 0.0),
            "route_choice_nonzero_rate_mean": mean_metric("route_choice_nonzero_rate", 0.0),
            "route_mean_valid_candidates_mean": mean_metric("route_mean_valid_candidates", 0.0),
            "route_mean_logit_margin_mean": mean_metric("route_mean_logit_margin", 0.0),
        }
        improved, score_key = self._save_best_frozen_checkpoint(
            episode,
            aggregate_summary,
            per_seed_rows,
        )
        aggregate_summary["best_checkpoint_updated"] = int(improved)
        aggregate_summary["score_key"] = "|".join(f"{value:.6f}" for value in score_key)
        self._append_frozen_eval_row(aggregate_summary)
        print(
            "[EP {:03d} FROZEN_EVAL] seeds={} spawn_interval={:.2f} win_rate={:.3f} "
            "completion={:.1%} avg_delta={:.2f} p90_delta={:.2f} "
            "tail_delta={:.2f} route_nonzero={:.1%} best={}".format(
                int(episode),
                aggregate_summary["seed_list"],
                float(self.eval_spawn_interval),
                aggregate_summary["win_rate"],
                aggregate_summary["completion_rate_mean"],
                aggregate_summary["avg_travel_time_delta_mean"],
                aggregate_summary["p90_travel_time_delta_mean"],
                aggregate_summary["tail_completion_gap_steps_delta_mean"],
                aggregate_summary["route_choice_nonzero_rate_mean"],
                "yes" if improved else "no",
            )
        )
        return aggregate_summary

    def generate_episode_vehicles(self, episode_seed=None, spawn_interval_override=None):
        """
        Generate controlled and uncontrolled vehicles for one training episode.
        """
        generator = target_vehicles_generator(os.path.join(self.sumocfg_dir, self.net_file))
        route_path = os.path.join(self.sumocfg_dir, self.route_file)
        spawn_interval_value = self.spawn_interval if spawn_interval_override is None else float(spawn_interval_override)
        vehicle_list = generator.generate_vehicles(
            num_target_vehicles=150,
            num_random_vehicles=150,
            pattern=self.target_pattern,
            target_xml_file=route_path,
            net_xml_file=os.path.join(self.sumocfg_dir, self.net_file),
            spawn_interval=spawn_interval_value,
            seed=episode_seed,
        )
        if vehicle_list is None:
            raise RuntimeError(
                "Failed to generate vehicles. Check randomTrips.py output for errors."
            )
        return {str(vehicle.vehicle_id): vehicle for vehicle in vehicle_list}

    def run(self):
        """Run RL training with shared junction-aware feasibility + pending-decision finalization."""
        sumo_binary = checkBinary('sumo')
        rolling_teleport_events = deque(maxlen=self.rolling_window)
        rolling_teleported_controlled = deque(maxlen=self.rolling_window)
        rolling_completion_rate = deque(maxlen=self.rolling_window)
        rolling_avg_return = deque(maxlen=self.rolling_window)
        rolling_avg_travel_time = deque(maxlen=self.rolling_window)
        rolling_mismatch = deque(maxlen=self.rolling_window)
        trainer_csv_fields = list(self.trainer.episode_metric_fields())
        csv_fields = [
            "episode", *trainer_csv_fields[:2], "algorithm",
            *trainer_csv_fields[2:6],
            "episode_return_total", "avg_return_per_vehicle",
            *trainer_csv_fields[6:],
            "completion_rate", "avg_travel_time", "p50_travel_time", "p90_travel_time", "teleports",
            "controlled_ever_teleported", "arrived_after_teleport", "clean_arrivals_without_teleport",
            "forced_actions", "critic_only_queued", "route_decisions_total",
            "route_candidate_count", "route_feasible_candidate_count",
            "route_actor_epochs_started", "route_actor_ownership_skips",
            "route_no_feasible_candidates", "route_apply_failures",
            "route_choice_idx_0", "route_choice_idx_1", "route_choice_idx_2", "route_choice_idx_3",
            "route_valid_candidates_1", "route_valid_candidates_2",
            "route_valid_candidates_3", "route_valid_candidates_4",
            "route_choice_nonzero_rate", "route_mean_valid_candidates",
            "route_mean_logit_margin", "route_mean_chosen_length_norm",
            "route_mean_chosen_eta_norm", "route_mean_chosen_density",
            "route_mean_chosen_first_density",
            "decisions_considered", "decisions_opened", "decisions_finalized", "decisions_skipped",
            "decisions_skipped_actionable",
            "skipped_pending_hold", "skipped_structural_no_branch", "skipped_structural_forced_single_path",
            "skipped_structural_forced_by_lane_commit", "skipped_structural_too_late_or_unreachable",
            "skipped_actionable_no_candidate", "skipped_other",
            "route_mismatch", "loop_events",
            "short_cycle_events", "aba_bounce_events", "dead_end_reentry_events",
            "long_horizon_loop_events", "revisit_without_progress_events",
            "loop_signal_events_total", "loop_repeat_only_events",
            "safety_overrides", "loop_prefilter_overrides", "fragment_build_failures",
            "fallback_overrides", "fallback_selected_total", "fallback_selected_lane_now",
            "pending_decision_timeouts", "deferred_lane_change_actions",
            "lane_change_observe_started", "lane_change_observe_success",
            "lane_change_observe_abort_no_progress", "lane_change_observe_abort_low_speed",
            "lane_change_observe_abort_commit_window",
            "pending_release_events_total", "pending_release_abort_events_total",
            "pending_release_timeout_events_total",
            "cooldown_replans_blocked",
            "pending_release_observe_abort_no_progress", "pending_release_observe_abort_commit_window",
            "pending_release_observe_abort_low_speed", "pending_release_wrong_lane_commit",
            "pending_release_route_no_progress_abort", "pending_release_route_stall_timeout", "pending_release_route_hard_timeout",
            "loop_override_count", "dead_end_reentry_override_count",
            "snapshot_cache_hits", "shortest_path_cache_hits",
            "exploration_actions", "policy_actions", "override_ratio", "override_events_total",
            "override_event_loop_prefilter", "override_event_cooldown_fallback",
            "override_event_observe_abort_fallback", "override_event_route_apply_fail",
            "override_event_invalid_action",
            "policy_masked_actions_removed", "override_learning_transitions",
            "cooldown_fallback_overrides", "observe_abort_fallback_overrides",
            "route_apply_fail_overrides", "override_learning_negative", "override_learning_imitation",
            "alive_at_step_cap", "decision_pending_at_episode_end", "mean_pending_age", "mean_decision_latency_steps",
            "pending_open_observe_end", "pending_open_route_end", "pending_open_lane_now_end",
            "pending_open_proactive_end", "pending_open_active_monitoring_end",
            "mean_pending_age_end_all", "max_pending_age_end_all",
            "mean_pending_age_end_active", "max_pending_age_end_active",
            "mean_pending_age_end_lane_now", "max_pending_age_end_lane_now",
            "mean_pending_stall_age_end", "max_pending_stall_age_end",
            "top_pending_end_edges", "oldest_pending_summary",
            "mean_reward_per_finalized_decision", "mean_route_difficulty_eta", "p50_route_difficulty_eta",
            "p90_route_difficulty_eta", "fail_teleport", "fail_timeout", "fail_removed_non_destination",
            "fail_unreachable_transition", "fail_dead_end_no_outgoing",
            "mean_network_density", "p95_network_density", "congestion_high_pressure_steps",
            "emergency_brake_events",
            "teleport_inferred_jam", "teleport_inferred_yield_or_deadlock",
            "lane_change_request_accepted_rate", "lane_change_observe_resolution_rate",
            "lane_change_observe_success_overcount",
            "tail_vehicles_over_p90_count", "tail_completion_gap_steps",
            "loop_reason_short_cycle", "loop_reason_aba_bounce", "loop_reason_dead_end_reentry",
            "loop_reason_long_horizon", "loop_reason_revisit_without_progress", "dominant_loop_reason",
            "top_loop_signal_edges", "top_loop_repeat_only_edges",
            "aggregate_actionable_skip_to_finalized_ratio",
            "social_regret_mean", "social_regret_p90", "social_best_action_chosen_rate",
            "actionable_skip_ratio", "structural_skip_ratio",
            "pending_resolution_success_rate", "pending_timeout_rate", "pending_abort_rate", "delay_fairness_gini",
            "loop_after_fallback_rate", "p95_to_p50_travel_ratio", "timeout_rate",
            "fallback_rate_per_opened_decision", "controlled_teleport_rate",
            "skip_reason_forced_by_lane_commit", "skip_reason_too_late_or_unreachable",
            "skip_reason_forced_single_path", "skip_reason_no_branch",
            "reachable_lane_change_nonempty", "reachable_lane_change_excluded_any",
            "reachable_lane_change_excluded_all", "policy_candidates_with_broader_available",
            "policy_candidates_collapsed_to_lane_now_only",
            "coordination_pending_reservations_seeded",
            "coordination_pressure_candidates_seen", "coordination_pressure_candidates_rejected",
            "lane_now_congestion_candidates_seen", "lane_now_congestion_candidates_rejected",
            "commit_window_non_lane_candidates_seen", "commit_window_candidates_rejected",
            "proactive_shift2_candidates_seen", "proactive_shift2_candidates_rejected",
            "proactive_brake_risk_candidates_seen", "proactive_brake_risk_candidates_rejected",
            "proactive_brake_risk_fallback_kept",
            "pending_commit_window_grace_kept",
            "proactive_decisions_opened", "proactive_decisions_finalized",
            "lane_now_decisions_opened", "lane_now_decisions_finalized",
            "proactive_pending_abort_count", "proactive_pending_timeout_count",
            "lane_now_replan_releases", "lane_now_replan_forced_alternative",
            "lane_now_replan_blocked_reopen_actions",
            "fallback_after_observe_abort_count", "fallback_after_timeout_count",
            "same_edge_reopen_after_abort_count", "synthetic_terminal_finalizations",
            "finalized_opened_proactive_ratio", "finalized_opened_lane_now_ratio",
            "penalized_avg_travel_time", "noncompletion_rate",
            "uncontrolled_total_wait_steps", "mean_uncontrolled_wait_per_step",
        ]
        # Rewrite metrics each new training session to avoid schema drift/appending old runs.
        with open(self.metrics_csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()
        self._ensure_decision_debug_csv_header()
        self._ensure_frozen_eval_csv_header()

        # MAX_CACHE_SIZE = 5000

        for episode in range(self.episodes):
            episode_seed = episode if self.seed_with_episode else None
            if episode_seed is not None:
                random.seed(episode_seed)
                np.random.seed(episode_seed)

            # if len(self._distance_cache) > MAX_CACHE_SIZE:
            #     self._distance_cache.clear()

            vehicles = self.generate_episode_vehicles(episode_seed=episode_seed)

            traci_command = [
                sumo_binary,
                "-c", self.runtime_sumocfg_path,
                "--quit-on-end",
            ]
            if self.fast_training_profile:
                traci_command.extend(["--no-step-log", "--no-warnings"])
            else:
                traci_command.extend([
                    "--tripinfo-output", os.path.join(self.sumocfg_dir, "trips.trips.xml"),
                ])
            traci.start(traci_command)
            self._active_vehicle_subscriptions = set()
            self._initialize_edge_subscriptions()
            simulation_get_min_expected = traci.simulation.getMinExpectedNumber
            simulation_step = traci.simulationStep
            simulation_get_arrived_ids = traci.simulation.getArrivedIDList
            vehicle_get_ids = traci.vehicle.getIDList
            pending_decisions = {}
            lane_change_deferrals = defaultdict(int)
            lane_change_cooldown_until = {}
            pending_release_info = {}
            stale_lane_now_replan_targets = {}
            self._episode_route_obs = {}
            vehicle_edges_since_reroute: dict = {}   # vehicle_id -> edges completed since last route decision
            vehicle_route_trace: dict = {}            # vehicle_id -> route-epoch MAPPO trace
            vehicle_actor_owned_route: dict = {}      # vehicle_id -> actor-committed route tuple for current epoch
            self._reset_episode_density_state()
            recent_edge_history = defaultdict(lambda: deque(maxlen=self.loop_window))
            prev_edge_by_vehicle = {}
            decision_metrics = defaultdict(float)
            decision_metrics["fallback_selected_total"] = 0.0
            decision_metrics["fallback_selected_lane_now"] = 0.0
            self.trainer.reset_episode_tracking()
            trainer_counter_start = self.trainer.capture_episode_metric_snapshot()

            episode_return_total = 0.0
            episode_teleport_events = 0
            teleported_controlled_ids = set()
            ever_teleported_controlled_ids = set()
            arrived_ids = set()
            total_controlled = len(vehicles)
            last_seen_edge_by_vehicle = {}
            last_planned_terminal_edge_by_vehicle = {}
            last_snapshot_by_vehicle = {}
            completed_travel_times = []
            completed_travel_time_ids = set()
            terminal_recorded_ids = set()
            arrived_debug_records = []
            final_outcome_by_vehicle = {}
            decision_latency_steps = []
            pending_age_samples = []
            finalized_decision_rewards = []
            route_difficulty_etas = []
            last_step_executed = -1
            alive_at_step_cap_ids = set()
            pending_debug_logged_ids = set()
            decision_debug_rows = []
            arrived_with_prestep_edge_not_destination = 0
            prev_speed_by_vehicle = {}
            emergency_brake_active_by_vehicle = {}
            hard_brake_counts_by_vehicle = defaultdict(int)
            last_observed_brake_step_by_vehicle = {}
            decision_attribution_by_vehicle = {}
            emergency_brake_events_by_edge = Counter()
            emergency_brake_events_by_vehicle = Counter()
            loop_signal_events_by_edge = Counter()
            loop_repeat_only_events_by_edge = Counter()
            mean_density_samples = []
            p95_density_samples = []
            congestion_high_pressure_steps = 0
            social_regret_samples = []
            uncontrolled_total_wait_steps = 0.0

            def decision_recent_history(vehicle_id):
                try:
                    history = recent_history_for_decision.get(vehicle_id)
                except NameError:
                    history = None
                return list(history) if history is not None else list(recent_edge_history[vehicle_id])

            def force_stale_lane_now_replan_if_available(vehicle_id, current_edge, step, context, policy_actions):
                key = (vehicle_id, current_edge)
                target_info = stale_lane_now_replan_targets.get(key)
                if not target_info:
                    return policy_actions
                if int(step) > int(target_info.get("until", step)):
                    stale_lane_now_replan_targets.pop(key, None)
                    return policy_actions

                blocked_action = int(target_info.get("blocked_action", -1))
                preferred_action = int(target_info.get("preferred_action", -1))
                available = set(int(action) for action in context.available_actions)
                if preferred_action in available:
                    decision_metrics["lane_now_replan_forced_alternative"] += 1
                    return [preferred_action]
                if blocked_action in policy_actions and len(policy_actions) > 1:
                    filtered_actions = [action for action in policy_actions if int(action) != blocked_action]
                    if filtered_actions:
                        decision_metrics["lane_now_replan_blocked_reopen_actions"] += 1
                        return filtered_actions
                return policy_actions

            def process_selected_action(
                vehicle_id,
                vehicle,
                current_edge,
                step,
                snapshot,
                context,
                state,
                action,
                action_source,
                selection=None,
                central_observation=None,
                coordination_state=None,
                critic_only=False,
            ):
                nonlocal episode_return_total

                recent_history = decision_recent_history(vehicle_id)
                cooldown_until = lane_change_cooldown_until.get((vehicle_id, current_edge), -1)
                policy_trace = None
                if selection is not None and central_observation is not None:
                    policy_trace = self._build_mappo_trace(state, central_observation, selection)
                    policy_trace["vehicle_id"] = vehicle_id
                    if critic_only:
                        policy_trace["mappo_critic_only"] = True
                next_edge = self.decision_engine.get_next_edge(current_edge, action)
                if next_edge is None:
                    decision_metrics["safety_overrides"] += 1
                    self._record_override_event(decision_metrics, "invalid_action")
                    penalty = self._clip_reward(-2.0)
                    if policy_trace is not None:
                        self._record_immediate_mappo_transition(
                            policy_trace,
                            action=action,
                            reward=penalty,
                            next_state=state,
                            next_central_observation=(
                                self._zero_central_observation()
                                if central_observation is None else central_observation
                            ),
                            done=False,
                            metadata={"override_type": "invalid_action"},
                        )
                    episode_return_total += penalty
                    self._accumulate_route_trace_reward(vehicle_route_trace, vehicle_id, penalty)
                    prev_edge_by_vehicle[vehicle_id] = current_edge
                    return None

                safe_ok, safety_details = self.decision_engine.prefilter_action_for_loops(
                    context=context,
                    action_idx=action,
                    destination=vehicle.destination,
                    recent_history=recent_history,
                    distance_fn=self.get_distance_to_destination,
                )
                if not safe_ok:
                    original_action = action
                    decision_metrics["loop_override_count"] += 1
                    decision_metrics["loop_prefilter_overrides"] += 1
                    if safety_details.get("dead_end_reentry"):
                        decision_metrics["dead_end_reentry_override_count"] += 1
                    action = self._select_fallback_action(
                        context,
                        blocked_action=action,
                        destination=vehicle.destination,
                        recent_history=recent_history,
                        coordination_state=coordination_state,
                    )
                    if action is None:
                        prev_edge_by_vehicle[vehicle_id] = current_edge
                        return None
                    decision_metrics["fallback_overrides"] += 1
                    decision_metrics["fallback_selected_total"] += 1
                    self._record_override_event(decision_metrics, "loop_prefilter")
                    if action in context.lane_feasible_now_actions:
                        decision_metrics["fallback_selected_lane_now"] += 1
                    decision_metrics["safety_overrides"] += 1
                    action_source = "loop_prefilter_fallback"
                    override_penalty = self._clip_reward(self.loop_trap_override_penalty)
                    if policy_trace is not None:
                        self._record_immediate_mappo_transition(
                            policy_trace,
                            action=original_action,
                            reward=override_penalty,
                            next_state=state,
                            next_central_observation=(
                                self._zero_central_observation()
                                if central_observation is None else central_observation
                            ),
                            done=False,
                            metadata={
                                "override_type": "loop_prefilter_fallback",
                                "original_action": original_action,
                                "fallback_action": action,
                            },
                        )
                    episode_return_total += override_penalty
                    self._accumulate_route_trace_reward(vehicle_route_trace, vehicle_id, override_penalty)
                    policy_trace = None
                    next_edge = self.decision_engine.get_next_edge(current_edge, action)
                    if next_edge is None:
                        prev_edge_by_vehicle[vehicle_id] = current_edge
                        return None

                lane_change_requested = False
                if action not in context.lane_feasible_now_actions:
                    if step < cooldown_until:
                        original_action = action
                        decision_metrics["cooldown_replans_blocked"] += 1
                        decision_metrics["cooldown_fallback_overrides"] += 1
                        fallback_action = self._select_fallback_action(
                            context,
                            blocked_action=action,
                            destination=vehicle.destination,
                            recent_history=recent_history,
                            lane_now_only=True,
                            coordination_state=coordination_state,
                        )
                        if fallback_action is None:
                            prev_edge_by_vehicle[vehicle_id] = current_edge
                            return None
                        action = fallback_action
                        action_source = "cooldown_fallback"
                        decision_metrics["fallback_overrides"] += 1
                        decision_metrics["fallback_selected_total"] += 1
                        self._record_override_event(decision_metrics, "cooldown_fallback")
                        if action in context.lane_feasible_now_actions:
                            decision_metrics["fallback_selected_lane_now"] += 1
                        release_info = pending_release_info.get(vehicle_id)
                        if (
                            release_info
                            and release_info.get("reason") == "timeout"
                            and release_info.get("edge") == current_edge
                            and (step - int(release_info.get("step", step))) <= self.decision_engine.cooldown_after_pending_release(timeout=True)
                        ):
                            decision_metrics["fallback_after_timeout_count"] += 1
                        override_penalty = self._clip_reward(self.same_edge_repeat_chase_penalty)
                        if policy_trace is not None:
                            self._record_immediate_mappo_transition(
                                policy_trace,
                                action=original_action,
                                reward=override_penalty,
                                next_state=state,
                                next_central_observation=(
                                    self._zero_central_observation()
                                    if central_observation is None else central_observation
                                ),
                                done=False,
                                metadata={
                                    "override_type": "cooldown_fallback",
                                    "original_action": original_action,
                                    "fallback_action": action,
                                },
                            )
                        episode_return_total += override_penalty
                        self._accumulate_route_trace_reward(vehicle_route_trace, vehicle_id, override_penalty)
                        policy_trace = None
                    else:
                        lane_change_requested, lane_change_ok = self.decision_engine.try_request_lane_change(context, action)
                        decision_metrics["lane_change_attempts"] += 1
                        if lane_change_ok:
                            decision_metrics["lane_change_success"] += 1
                        else:
                            decision_metrics["lane_change_fail"] += 1
                        decision_metrics["lane_change_observe_started"] += 1
                        decision_metrics["deferred_lane_change_actions"] += 1
                        observe_metadata = self.decision_engine.start_lane_change_observe(
                            context, action, step, lane_change_requested, lane_change_ok
                        )
                        decision_id = self._next_decision_id(decision_metrics)
                        origin_mode = ("lane_now" if action in context.lane_feasible_now_actions else "proactive")
                        pending_decisions[vehicle_id] = self.shared_policy.build_observe_pending(
                            state=state,
                            action_idx=action,
                            intended_next_edge=next_edge,
                            decision_edge=current_edge,
                            step=step,
                            destination=vehicle.destination,
                            context=context,
                            lane_change_requested=lane_change_requested,
                            decision_id=decision_id,
                            origin_mode=origin_mode,
                            action_source=action_source,
                            observe_metadata=observe_metadata,
                            decision_open_recorded=False,
                            extra_metadata={
                                "coordination_pressure": self.shared_policy.coordination_pressure_score(
                                    context=context,
                                    destination=vehicle.destination,
                                    action_idx=action,
                                    coordination_state=coordination_state,
                                ),
                                **({} if policy_trace is None else policy_trace),
                            },
                        )
                        self._record_recent_decision_attribution(
                            decision_attribution_by_vehicle,
                            vehicle_id,
                            step,
                            action_source or "policy",
                            "proactive",
                        )
                        self._register_decision_open(decision_metrics, pending_decisions[vehicle_id])
                        release_info = pending_release_info.get(vehicle_id)
                        if (
                            release_info
                            and release_info.get("reason") == "abort"
                            and release_info.get("edge") == current_edge
                            and (step - int(release_info.get("step", step))) <= self.decision_engine.cooldown_after_pending_release(timeout=False)
                        ):
                            decision_metrics["same_edge_reopen_after_abort_count"] += 1
                        prev_edge_by_vehicle[vehicle_id] = current_edge
                        return int(action)

                candidate_next_edges = {
                    candidate_action: self.decision_engine.get_next_edge(current_edge, candidate_action)
                    for candidate_action in context.available_actions
                }
                candidate_actions = [
                    candidate_action
                    for candidate_action, candidate_edge in candidate_next_edges.items()
                    if candidate_edge is not None
                ]
                candidate_costs = {}
                for candidate_action in candidate_actions:
                    candidate_edge = candidate_next_edges[candidate_action]
                    eta_proxy = self._estimate_remaining_eta(candidate_edge, vehicle.destination)
                    if not math.isfinite(eta_proxy):
                        eta_proxy = float(MAX_SIMULATION_STEPS)
                    candidate_costs[candidate_action] = (
                        (1.25 * float(self._edge_density(candidate_edge)))
                        + (0.01 * float(eta_proxy))
                    )
                finite_costs = {candidate_action: cost for candidate_action, cost in candidate_costs.items() if math.isfinite(cost)}
                chosen_cost = float(candidate_costs.get(action, math.inf))
                baseline_actions = context.lane_feasible_now_actions if context.lane_feasible_now_actions else candidate_actions
                baseline_finite_costs = [
                    candidate_costs.get(candidate_action, math.inf)
                    for candidate_action in baseline_actions
                    if math.isfinite(candidate_costs.get(candidate_action, math.inf))
                ]
                baseline_cost = float(min(baseline_finite_costs)) if baseline_finite_costs else math.inf
                selfless_delta = float(baseline_cost - chosen_cost) if math.isfinite(chosen_cost) and math.isfinite(baseline_cost) else 0.0
                coordination_pressure = self.shared_policy.coordination_pressure_score(
                    context=context,
                    destination=vehicle.destination,
                    action_idx=action,
                    coordination_state=coordination_state,
                )
                if len(candidate_actions) > 1 and finite_costs and action in finite_costs:
                    best_action = min(finite_costs, key=finite_costs.get)
                    best_cost = finite_costs[best_action]
                    chosen_cost = finite_costs[action]
                    social_regret = max(float(chosen_cost - best_cost), 0.0)
                    social_regret_samples.append(social_regret)
                    decision_metrics["social_regret_sum"] += social_regret
                    decision_metrics["social_regret_count"] += 1
                    if action == best_action:
                        decision_metrics["social_best_action_chosen"] += 1

                full_route, committed_next_edge, apply_error = self.decision_engine.apply_route_decision(
                    vehicle_id,
                    current_edge,
                    action,
                    vehicle.destination,
                )
                if apply_error:
                    decision_metrics["route_apply_fail"] += 1
                    decision_metrics["route_apply_fail_overrides"] += 1
                    self._record_override_event(decision_metrics, "route_apply_fail")
                    decision_metrics["fragment_build_failures"] += 1
                    override_penalty = self._clip_reward(-6.0)
                    if policy_trace is not None:
                        self._record_immediate_mappo_transition(
                            policy_trace,
                            action=action,
                            reward=override_penalty,
                            next_state=state,
                            next_central_observation=(
                                self._zero_central_observation()
                                if central_observation is None else central_observation
                            ),
                            done=False,
                            metadata={
                                "override_type": "route_apply_failure",
                                "original_action": action,
                                "route_apply_failed": True,
                            },
                        )
                    episode_return_total += override_penalty
                    self._accumulate_route_trace_reward(vehicle_route_trace, vehicle_id, override_penalty)
                    prev_edge_by_vehicle[vehicle_id] = current_edge
                    return None
                last_planned_terminal_edge_by_vehicle[vehicle_id] = full_route[-1] if full_route else vehicle.destination

                decision_id = self._next_decision_id(decision_metrics)
                origin_mode = ("lane_now" if action in context.lane_feasible_now_actions else "proactive")
                pending_decisions[vehicle_id] = self.shared_policy.build_route_pending(
                    state=state,
                    action_idx=action,
                    committed_next_edge=committed_next_edge,
                    decision_edge=current_edge,
                    step=step,
                    destination=vehicle.destination,
                    context=context,
                    lane_change_requested=lane_change_requested,
                    decision_id=decision_id,
                    origin_mode=origin_mode,
                    action_source=action_source,
                    full_route=full_route,
                    decision_open_recorded=False,
                    extra_metadata={
                        "lane_change_deferrals": lane_change_deferrals.get(vehicle_id, 0),
                        "chosen_social_cost": chosen_cost,
                        "baseline_social_cost": baseline_cost,
                        "selfless_delta": selfless_delta,
                        "coordination_pressure": coordination_pressure,
                        **({} if policy_trace is None else policy_trace),
                    },
                )
                self._record_recent_decision_attribution(
                    decision_attribution_by_vehicle,
                    vehicle_id,
                    step,
                    action_source,
                    "lane_now",
                )
                lane_change_deferrals[vehicle_id] = 0
                self._register_decision_open(decision_metrics, pending_decisions[vehicle_id])
                release_info = pending_release_info.get(vehicle_id)
                if (
                    release_info
                    and release_info.get("reason") == "abort"
                    and release_info.get("edge") == current_edge
                    and (step - int(release_info.get("step", step))) <= self.decision_engine.cooldown_after_pending_release(timeout=False)
                ):
                    decision_metrics["same_edge_reopen_after_abort_count"] += 1
                prev_edge_by_vehicle[vehicle_id] = current_edge
                return int(action)

            try:
                for step in range(MAX_SIMULATION_STEPS):
                    if simulation_get_min_expected() <= 0:
                        break
                    last_step_executed = step

                    # Keep density features fresh for routing choices and reward.
                    edge_subscription_results = traci.edge.getAllSubscriptionResults() or {}
                    self.update_edge_vehicle_counts(
                        step,
                        every=self.density_refresh_every,
                        edge_results=edge_subscription_results,
                    )
                    vehicle_ids = list(vehicle_get_ids())
                    controlled_live_ids = [vid for vid in vehicle_ids if vid in vehicles]
                    for _uid in vehicle_ids:
                        if _uid not in vehicles:
                            uncontrolled_total_wait_steps += traci.vehicle.getWaitingTime(_uid)
                    self._ensure_vehicle_subscriptions(controlled_live_ids)
                    vehicle_subscription_results = traci.vehicle.getAllSubscriptionResults() or {}
                    self._step_vehicle_results = vehicle_subscription_results
                    self._step_vehicle_wait_cache = {}
                    self._step_lane_occupancy_cache = {}
                    self._step_lane_halting_cache = {}
                    step_snapshots = self.collect_vehicle_snapshots(
                        controlled_live_ids,
                        step,
                        vehicle_results=vehicle_subscription_results,
                    )
                    step_context_cache = {}
                    step_state_cache = {}
                    open_decision_batch = []
                    step_coordination_state = self.shared_policy.empty_coordination_state()
                    decision_metrics["coordination_pending_reservations_seeded"] += (
                        self.shared_policy.seed_coordination_from_pending(
                            step_coordination_state,
                            pending_decisions,
                            current_step=step,
                            max_age_steps=self.decision_engine.route_pending_hard_timeout_steps,
                        )
                    )
                    recent_history_for_decision = {}
                    step_mean_density = float(self._density_mean)
                    step_p95_density = float(self._density_p95)
                    mean_density_samples.append(step_mean_density)
                    p95_density_samples.append(step_p95_density)

                    if step_snapshots:
                        mean_controlled_speed = float(np.mean([snap.speed for snap in step_snapshots.values()]))
                        mean_controlled_edge_density = float(
                            np.mean([self._edge_density(snapshot.edge_id) for snapshot in step_snapshots.values()])
                        )
                        if (
                            mean_controlled_edge_density >= self.congestion_density_threshold
                            and mean_controlled_speed <= self.congestion_low_speed_threshold
                        ):
                            congestion_high_pressure_steps += 1

                    step_transition_central_observation = self._build_central_observation(
                        step=step,
                        total_controlled=total_controlled,
                        arrived_ids=arrived_ids,
                        step_snapshots=step_snapshots,
                        vehicles=vehicles,
                        pending_decisions=pending_decisions,
                        coordination_state=step_coordination_state,
                        open_decision_count=0,
                    )

                    for vehicle_id in controlled_live_ids:
                        snapshot = step_snapshots.get(vehicle_id)
                        if snapshot is None:
                            # Reset event latch if we cannot observe this step; avoids stale active flags.
                            emergency_brake_active_by_vehicle.pop(vehicle_id, None)
                            prev_speed_by_vehicle.pop(vehicle_id, None)
                            last_observed_brake_step_by_vehicle.pop(vehicle_id, None)
                            continue
                        prev_speed = prev_speed_by_vehicle.get(vehicle_id)
                        observed_prev_step = (last_observed_brake_step_by_vehicle.get(vehicle_id) == (step - 1))
                        if prev_speed is not None and observed_prev_step:
                            decel = max(float(prev_speed) - float(snapshot.speed), 0.0)
                            hard_brake = decel >= self.emergency_decel_threshold and prev_speed > 4.0
                            was_hard_brake_active = bool(emergency_brake_active_by_vehicle.get(vehicle_id, False))
                            if hard_brake and not was_hard_brake_active:
                                decision_metrics["emergency_brake_events"] += 1
                                hard_brake_counts_by_vehicle[vehicle_id] += 1
                                emergency_brake_events_by_edge[str(snapshot.edge_id)] += 1
                                emergency_brake_events_by_vehicle[str(vehicle_id)] += 1
                                emergency_reason = "other"
                                try:
                                    leader_info = traci.vehicle.getLeader(vehicle_id)
                                except Exception:
                                    leader_info = None
                                if leader_info and len(leader_info) >= 2 and float(leader_info[1]) < 10.0:
                                    emergency_reason = "leader"
                                else:
                                    edge_density = self._edge_density(snapshot.edge_id)
                                    if edge_density >= self.congestion_density_threshold:
                                        emergency_reason = "congestion"
                                    elif snapshot.dist_to_end <= 20.0:
                                        emergency_reason = "junction"
                                if emergency_reason == "leader":
                                    decision_metrics["emergency_brake_due_to_leader"] += 1
                                elif emergency_reason == "congestion":
                                    decision_metrics["emergency_brake_due_to_congestion"] += 1
                                elif emergency_reason == "junction":
                                    decision_metrics["emergency_brake_near_junction"] += 1
                                else:
                                    decision_metrics["emergency_brake_other_reason"] += 1
                                self._attribute_hard_brake_event(
                                    decision_metrics,
                                    decision_attribution_by_vehicle,
                                    vehicle_id,
                                    step,
                                )
                            emergency_brake_active_by_vehicle[vehicle_id] = hard_brake
                        else:
                            # No prior speed => no detectable braking episode yet; keep latch clear.
                            emergency_brake_active_by_vehicle[vehicle_id] = False

                        current_edge = snapshot.edge_id
                        previous_seen_edge = last_seen_edge_by_vehicle.get(vehicle_id)
                        edge_changed_runtime = previous_seen_edge is None or previous_seen_edge != current_edge
                        history_before_edge = list(recent_edge_history[vehicle_id])
                        recent_history_for_decision[vehicle_id] = history_before_edge
                        prev_speed_by_vehicle[vehicle_id] = snapshot.speed
                        last_observed_brake_step_by_vehicle[vehicle_id] = step

                        vehicle = vehicles[vehicle_id]
                        vehicle.current_edge = current_edge
                        vehicle.current_speed = snapshot.speed
                        if getattr(vehicle, "_route_difficulty_eta_logged", False) is False:
                            eta0 = self._estimate_remaining_eta(current_edge, vehicle.destination)
                            if math.isfinite(eta0):
                                route_difficulty_etas.append(float(eta0))
                            vehicle._route_difficulty_eta_logged = True
                        last_seen_edge_by_vehicle[vehicle_id] = current_edge
                        last_snapshot_by_vehicle[vehicle_id] = snapshot
                        if edge_changed_runtime:
                            recent_edge_history[vehicle_id].append(current_edge)
                            vehicle_edges_since_reroute[vehicle_id] = (
                                vehicle_edges_since_reroute.get(vehicle_id, 0) + 1
                            )
                        if vehicle_id in vehicle_actor_owned_route:
                            route_epoch_reward = self._accumulate_route_epoch_step_reward(
                                vehicle_route_trace,
                                vehicle_id,
                                vehicle,
                                current_edge,
                                step,
                            )
                            episode_return_total += route_epoch_reward

                        prev_edge = prev_edge_by_vehicle.get(vehicle_id)
                        if vehicle_id in pending_decisions and current_edge != pending_decisions[vehicle_id].decision_edge:
                            pending = pending_decisions.pop(vehicle_id)
                            decision_metrics["pending_resolved_success"] += 1
                            repeated_recent_edges = sum(1 for e in history_before_edge if e == current_edge)
                            mismatch = not self.decision_engine.route_matches_expected(pending, current_edge)
                            if mismatch:
                                decision_metrics["route_mismatch"] += 1
                            signal_edges = set(history_before_edge) | {current_edge}
                            edge_distance_lookup = {
                                edge: self.get_distance_to_destination(edge, vehicle.destination)
                                for edge in signal_edges
                            }
                            loop_signals = transition_signal(
                                deque(history_before_edge, maxlen=self.loop_window),
                                current_edge,
                                edge_out_degree=self._edge_out_degree_map(history_before_edge + [current_edge]),
                                edge_distance_lookup=edge_distance_lookup,
                                progress_slack=self.decision_engine.loop_distance_slack,
                            )
                            if loop_signals["aba_bounce"]:
                                decision_metrics["aba_bounce_events"] += 1
                                
                            if loop_signals["short_cycle"]:
                                decision_metrics["short_cycle_events"] += 1
                            if loop_signals["dead_end_reentry"]:
                                decision_metrics["dead_end_reentry_events"] += 1
                            if loop_signals.get("long_horizon_loop"):
                                decision_metrics["long_horizon_loop_events"] += 1
                            if loop_signals.get("revisit_without_progress"):
                                decision_metrics["revisit_without_progress_events"] += 1
                            explicit_loop_signal = bool(
                                loop_signals["aba_bounce"]
                                or loop_signals["short_cycle"]
                                or loop_signals["dead_end_reentry"]
                                or loop_signals.get("long_horizon_loop")
                                or loop_signals.get("revisit_without_progress")
                            )
                            if explicit_loop_signal:
                                decision_metrics["loop_signal_events_total"] += 1
                                loop_signal_events_by_edge[str(current_edge)] += 1
                            ext_pen = max(self._edge_density(current_edge), 0.0)
                            prev_distance = self.get_distance_to_destination(pending.decision_edge, vehicle.destination)
                            curr_distance = self.get_distance_to_destination(current_edge, vehicle.destination)
                            prev_eta = self._estimate_remaining_eta(pending.decision_edge, vehicle.destination)
                            curr_eta = self._estimate_remaining_eta(current_edge, vehicle.destination)
                            reward, done = self.compute_reward(
                                vehicle, pending.last_credit_edge, current_edge, step, arrived=False,
                                repeated_recent_edges=repeated_recent_edges,
                                delta_t=max(step - pending.last_credit_step, 1),
                                route_mismatch=mismatch,
                                uturn_repeat=loop_signals["aba_bounce"] or loop_signals["short_cycle"],
                                long_horizon_loop=loop_signals.get("long_horizon_loop") or loop_signals.get("revisit_without_progress"),
                                externality_penalty=ext_pen,
                                selfless_delta=float(pending.metadata.get("selfless_delta", 0.0)),
                                coordination_pressure=float((pending.metadata or {}).get("coordination_pressure", 0.0)),
                                hard_brake_events=self._consume_hard_brake_events(hard_brake_counts_by_vehicle, vehicle_id),
                            )
                            next_ctx = self._get_or_build_step_context(
                                step_context_cache,
                                vehicle_id,
                                current_edge,
                                vehicle.destination,
                                step,
                                snapshot=snapshot,
                            )
                            next_state = self._get_or_encode_step_state(
                                step_state_cache,
                                step_context_cache,
                                vehicle_id,
                                vehicle,
                                step,
                                snapshot,
                                context=next_ctx,
                                coordination_state=step_coordination_state,
                            )
                            final_metadata = {
                                **(pending.metadata if isinstance(pending.metadata, dict) else {}),
                                "forced_action": pending.context.forced_action is not None,
                                "mismatch": mismatch,
                                "decision_finalized": True,
                            }
                            self._record_pending_mappo_transition(
                                pending,
                                reward=reward,
                                next_state=next_state,
                                next_central_observation=step_transition_central_observation,
                                done=done,
                                discount_steps=max(step - pending.decision_step, 1),
                                metadata=final_metadata,
                            )
                            self._accumulate_route_trace_reward(vehicle_route_trace, vehicle_id, reward)
                            self._register_decision_finalized(decision_metrics, pending)
                            action_source = str(pending.metadata.get("action_source", ""))
                            if "fallback" in action_source:
                                decision_metrics["fallback_finalized"] += 1
                            decision_latency_steps.append(float(max(step - pending.decision_step, 0)))
                            finalized_decision_rewards.append(float(reward))
                            episode_return_total += reward
                            if repeated_recent_edges > 0 and not explicit_loop_signal:
                                decision_metrics["loop_repeat_only_events"] += 1
                                loop_repeat_only_events_by_edge[str(current_edge)] += 1
                            if repeated_recent_edges > 0 or explicit_loop_signal:
                                decision_metrics["loop_events"] += 1
                                if "fallback" in action_source:
                                    decision_metrics["loop_after_fallback_events"] += 1
                            if math.isfinite(prev_distance) and (not math.isfinite(curr_distance)):
                                decision_metrics["fail_unreachable_transition"] += 1
                            outgoing = self.connection_info.outgoing_edges_dict.get(current_edge, {})
                            if (not outgoing or len(outgoing) == 0) and current_edge != vehicle.destination:
                                decision_metrics["fail_dead_end_no_outgoing"] += 1
                            edge_density = self._edge_density(current_edge)
                            mean_density = float(self._density_mean)
                            marginal_pressure = max(edge_density - mean_density, 0.0)
                            decision_debug_rows.append(self._build_decision_debug_row(
                                episode=episode,
                                step=step,
                                pending=pending,
                                actual_next_edge=current_edge,
                                finalized=1,
                                finalize_delay_steps=max(step - pending.decision_step, 0),
                                reward=reward,
                                done=int(bool(done)),
                                route_mismatch=int(bool(mismatch)),
                                teleported=0,
                                reached_global_destination=0,
                                prev_eta=prev_eta if math.isfinite(prev_eta) else "",
                                curr_eta=curr_eta if math.isfinite(curr_eta) else "",
                                prev_distance=prev_distance if math.isfinite(prev_distance) else "",
                                curr_distance=curr_distance if math.isfinite(curr_distance) else "",
                                edge_density=edge_density,
                                mean_density=mean_density,
                                externality_penalty=ext_pen,
                                marginal_pressure=marginal_pressure,
                            ))
                        # Do not cleanup pre-step destination reaches yet; terminal transition
                        # must be written exactly once before any state is removed.
                        if current_edge == vehicle.destination:
                            prev_edge_by_vehicle[vehicle_id] = current_edge
                            continue
                        elif vehicle_id in pending_decisions:
                            pending = pending_decisions[vehicle_id]
                            pending_phase = pending.metadata.get("phase", "route_pending")
                            if pending_phase == "observe_lane_change":
                                obs_context = self._get_or_build_step_context(
                                    step_context_cache,
                                    vehicle_id,
                                    current_edge,
                                    vehicle.destination,
                                    step,
                                    snapshot=snapshot,
                                )
                                status, reason = self.decision_engine.evaluate_lane_change_observation(
                                    pending.metadata,
                                    obs_context,
                                    pending.intended_action,
                                )
                                if status == "continue":
                                    prev_edge_by_vehicle[vehicle_id] = current_edge
                                    continue
                                pending_decisions.pop(vehicle_id, None)
                                if status == "success":
                                    decision_metrics["lane_change_observe_success"] += 1
                                    full_route, committed_next_edge, apply_error = self.decision_engine.apply_route_decision(
                                        vehicle_id, current_edge, pending.intended_action, vehicle.destination
                                    )
                                    if apply_error:
                                        decision_metrics["route_apply_fail"] += 1
                                        decision_metrics["route_apply_fail_overrides"] += 1
                                        self._record_override_event(decision_metrics, "route_apply_fail")
                                        override_penalty = self._clip_reward(-6.0)
                                        self._record_pending_mappo_transition(
                                            pending,
                                            reward=override_penalty,
                                            next_state=pending.state,
                                            next_central_observation=step_transition_central_observation,
                                            done=False,
                                            discount_steps=max(step - pending.decision_step, 1),
                                            metadata={
                                                **(pending.metadata if isinstance(pending.metadata, dict) else {}),
                                                "override_type": "route_apply_failure",
                                                "route_apply_failed": True,
                                                "observe_phase": True,
                                                "decision_finalized": False,
                                            },
                                        )
                                        self._accumulate_route_trace_reward(vehicle_route_trace, vehicle_id, override_penalty)
                                        episode_return_total += override_penalty
                                        prev_edge_by_vehicle[vehicle_id] = current_edge
                                        continue
                                    pending_decisions[vehicle_id] = self.shared_policy.promote_observe_success(
                                        pending,
                                        context=obs_context,
                                        step=step,
                                        committed_next_edge=committed_next_edge,
                                        full_route=full_route,
                                    )
                                    self.shared_policy.reserve_action(
                                        step_coordination_state,
                                        context=obs_context,
                                        destination=vehicle.destination,
                                        action_idx=int(pending.intended_action),
                                    )
                                    self._record_recent_decision_attribution(
                                        decision_attribution_by_vehicle,
                                        vehicle_id,
                                        step,
                                        str((pending.metadata or {}).get("action_source", "policy")),
                                        "proactive",
                                    )
                                    prev_edge_by_vehicle[vehicle_id] = current_edge
                                    continue
                                if reason == "commit_window":
                                    decision_metrics["lane_change_observe_abort_commit_window"] += 1
                                    pending_pen = self.observe_commit_window_miss_penalty
                                elif reason == "low_speed":
                                    decision_metrics["lane_change_observe_abort_low_speed"] += 1
                                    pending_pen = self.observe_low_speed_penalty
                                else:
                                    decision_metrics["lane_change_observe_abort_no_progress"] += 1
                                    pending_pen = self.observe_no_progress_penalty
                                pending_pen = self._clip_reward(pending_pen + self.same_edge_repeat_chase_penalty)
                                observe_abort_state = self._get_or_encode_step_state(
                                    step_state_cache,
                                    step_context_cache,
                                    vehicle_id,
                                    vehicle,
                                    step,
                                    snapshot,
                                    context=obs_context,
                                    coordination_state=step_coordination_state,
                                )
                                self._record_pending_mappo_transition(
                                    pending,
                                    reward=pending_pen,
                                    next_state=observe_abort_state,
                                    next_central_observation=step_transition_central_observation,
                                    done=False,
                                    discount_steps=max(step - pending.decision_step, 1),
                                    metadata={
                                        **(pending.metadata if isinstance(pending.metadata, dict) else {}),
                                        "observe_abort": reason or "no_progress",
                                        "observe_no_progress": (reason or "no_progress") == "no_progress",
                                        "observe_low_speed": (reason or "") == "low_speed",
                                        "observe_commit_window_miss": (reason or "") == "commit_window",
                                        "abort_reason": reason or "no_progress",
                                        "override_type": "observe_abort_fallback",
                                        "decision_finalized": False,
                                    },
                                )
                                self._accumulate_route_trace_reward(vehicle_route_trace, vehicle_id, pending_pen)
                                episode_return_total += pending_pen
                                decision_metrics["pending_resolved_abort_no_progress"] += 1
                                if reason == "commit_window":
                                    self._record_pending_release(decision_metrics, "observe_abort_commit_window")
                                elif reason == "low_speed":
                                    self._record_pending_release(decision_metrics, "observe_abort_low_speed")
                                else:
                                    self._record_pending_release(decision_metrics, "observe_abort_no_progress")
                                if str((pending.metadata or {}).get("decision_origin_mode", pending.decision_origin_mode)) == "proactive":
                                    decision_metrics["proactive_pending_abort_count"] += 1
                                cooldown_key = (vehicle_id, current_edge)
                                lane_change_cooldown_until[cooldown_key] = (
                                    step + self.decision_engine.cooldown_after_pending_release(timeout=False)
                                )
                                pending_release_info[vehicle_id] = {"edge": current_edge, "step": int(step), "reason": "abort"}
                                fallback_action = self._select_fallback_action(
                                    obs_context,
                                    blocked_action=pending.intended_action,
                                    destination=vehicle.destination,
                                    recent_history=decision_recent_history(vehicle_id),
                                    lane_now_only=True,
                                    coordination_state=step_coordination_state,
                                )
                                if fallback_action is None:
                                    prev_edge_by_vehicle[vehicle_id] = current_edge
                                    continue
                                full_route, committed_next_edge, apply_error = self.decision_engine.apply_route_decision(
                                    vehicle_id, current_edge, fallback_action, vehicle.destination
                                )
                                if apply_error:
                                    prev_edge_by_vehicle[vehicle_id] = current_edge
                                    continue
                                next_state = self._get_or_encode_step_state(
                                    step_state_cache,
                                    step_context_cache,
                                    vehicle_id,
                                    vehicle,
                                    step,
                                    snapshot,
                                    context=obs_context,
                                    coordination_state=step_coordination_state,
                                )
                                pending_decisions[vehicle_id] = self.shared_policy.build_route_pending(
                                    state=next_state,
                                    action_idx=fallback_action,
                                    committed_next_edge=committed_next_edge,
                                    decision_edge=current_edge,
                                    step=step,
                                    destination=vehicle.destination,
                                    context=obs_context,
                                    lane_change_requested=False,
                                    decision_id=str((pending.metadata or {}).get("decision_id", pending.decision_id)),
                                    origin_mode=str((pending.metadata or {}).get("decision_origin_mode", pending.decision_origin_mode or "proactive")),
                                    action_source="observe_fallback",
                                    full_route=full_route,
                                    decision_open_recorded=True,
                                    extra_metadata={
                                        "coordination_pressure": self.shared_policy.coordination_pressure_score(
                                            context=obs_context,
                                            destination=vehicle.destination,
                                            action_idx=fallback_action,
                                            coordination_state=step_coordination_state,
                                        ),
                                    },
                                )
                                self.shared_policy.reserve_action(
                                    step_coordination_state,
                                    context=obs_context,
                                    destination=vehicle.destination,
                                    action_idx=int(fallback_action),
                                )
                                self._record_recent_decision_attribution(
                                    decision_attribution_by_vehicle,
                                    vehicle_id,
                                    step,
                                    "observe_fallback",
                                    "lane_now",
                                )
                                decision_metrics["fallback_overrides"] += 1
                                decision_metrics["fallback_selected_total"] += 1
                                self._record_override_event(decision_metrics, "observe_abort_fallback")
                                if fallback_action in obs_context.lane_feasible_now_actions:
                                    decision_metrics["fallback_selected_lane_now"] += 1
                                decision_metrics["observe_abort_fallback_overrides"] += 1
                                decision_metrics["fallback_after_observe_abort_count"] += 1
                                prev_edge_by_vehicle[vehicle_id] = current_edge
                                continue
                            pending_age = self.decision_engine.pending_age_steps(pending, step)
                            active_pending = self.shared_policy.pending_requires_active_same_edge_monitoring(pending)
                            if active_pending:
                                pending_age_samples.append(float(pending_age))
                            elapsed_pending = max(step - pending.last_credit_step, 0)
                            if elapsed_pending > 0:
                                ext_pen = max(self._edge_density(current_edge), 0.0)
                                pending_reward = self.compute_pending_step_reward(
                                    vehicle,
                                    current_edge,
                                    elapsed=elapsed_pending,
                                    step=step,
                                    externality_penalty=ext_pen,
                                    pending_age=pending_age,
                                    lane_change_deferrals=pending.metadata.get("lane_change_deferrals", 0),
                                    hard_brake_events=self._consume_hard_brake_events(hard_brake_counts_by_vehicle, vehicle_id),
                                    coordination_pressure=float((pending.metadata or {}).get("coordination_pressure", 0.0)),
                                )
                                next_ctx = self._get_or_build_step_context(
                                    step_context_cache,
                                    vehicle_id,
                                    current_edge,
                                    vehicle.destination,
                                    step,
                                    snapshot=snapshot,
                                )
                                next_state = self._get_or_encode_step_state(
                                    step_state_cache,
                                    step_context_cache,
                                    vehicle_id,
                                    vehicle,
                                    step,
                                    snapshot,
                                    context=next_ctx,
                                    coordination_state=step_coordination_state,
                                )
                                self._accumulate_mappo_reward(pending.metadata, pending_reward)
                                self._accumulate_route_trace_reward(vehicle_route_trace, vehicle_id, pending_reward)
                                episode_return_total += pending_reward
                                pending.state = next_state
                                pending.last_credit_edge = current_edge
                                pending.last_credit_step = step
                            if active_pending and self.decision_engine.should_timeout_pending(pending, step):
                                timeout_ctx = self._get_or_build_step_context(
                                    step_context_cache,
                                    vehicle_id,
                                    current_edge,
                                    vehicle.destination,
                                    step,
                                    snapshot=snapshot,
                                )
                                timeout_state = self._get_or_encode_step_state(
                                    step_state_cache,
                                    step_context_cache,
                                    vehicle_id,
                                    vehicle,
                                    step,
                                    snapshot,
                                    context=timeout_ctx,
                                    coordination_state=step_coordination_state,
                                )
                                timeout_penalty = self._clip_reward(self.pending_timeout_penalty)
                                self._record_pending_mappo_transition(
                                    pending,
                                    reward=timeout_penalty,
                                    next_state=timeout_state,
                                    next_central_observation=step_transition_central_observation,
                                    done=False,
                                    discount_steps=max(step - pending.decision_step, 1),
                                    metadata={
                                        **(pending.metadata if isinstance(pending.metadata, dict) else {}),
                                        "pending_timeout_replan": True,
                                        "decision_finalized": False,
                                    },
                                )
                                self._accumulate_route_trace_reward(vehicle_route_trace, vehicle_id, timeout_penalty)
                                episode_return_total += timeout_penalty
                                decision_metrics["pending_decision_timeouts"] += 1
                                self._record_pending_release(decision_metrics, "route_stall_timeout")
                                decision_metrics["pending_resolved_timeout"] += 1
                                if str((pending.metadata or {}).get("decision_origin_mode", pending.decision_origin_mode)) == "proactive":
                                    decision_metrics["proactive_pending_timeout_count"] += 1
                                pending_decisions.pop(vehicle_id, None)
                                lane_change_deferrals[vehicle_id] = 0
                                lane_change_cooldown_until[(vehicle_id, current_edge)] = (
                                    step + self.decision_engine.cooldown_after_pending_release(timeout=True)
                                )
                                pending_release_info[vehicle_id] = {"edge": current_edge, "step": int(step), "reason": "timeout"}
                                continue
                            pending_ctx = self._get_or_build_step_context(
                                step_context_cache,
                                vehicle_id,
                                current_edge,
                                vehicle.destination,
                                step,
                                snapshot=snapshot,
                            )
                            release_eval = self.shared_policy.evaluate_route_pending_release(
                                pending,
                                context=pending_ctx,
                                step=step,
                                lane_position_now=float(snapshot.lane_position),
                                edge_density_fn=self._edge_density,
                                distance_fn=self.get_distance_to_destination,
                                recent_history=decision_recent_history(vehicle_id),
                            )
                            if (
                                pending_ctx.commit_window
                                and pending.intended_action not in pending_ctx.lane_feasible_now_actions
                                and release_eval.grace_keep
                            ):
                                decision_metrics["pending_commit_window_grace_kept"] += 1
                            if release_eval.should_release:
                                self._record_pending_release(decision_metrics, release_eval.release_reason)
                                release_state = self._get_or_encode_step_state(
                                    step_state_cache,
                                    step_context_cache,
                                    vehicle_id,
                                    vehicle,
                                    step,
                                    snapshot,
                                    context=pending_ctx,
                                    coordination_state=step_coordination_state,
                                )
                                if release_eval.release_as_timeout:
                                    release_penalty = self.pending_timeout_penalty
                                elif release_eval.release_reason == "wrong_lane_commit":
                                    release_penalty = self.pending_replan_penalty + self.observe_commit_window_miss_penalty
                                elif release_eval.release_reason == "route_no_progress_abort":
                                    release_penalty = self.pending_replan_penalty + self.observe_no_progress_penalty
                                else:
                                    release_penalty = self.pending_replan_penalty
                                release_penalty = self._clip_reward(release_penalty)
                                self._record_pending_mappo_transition(
                                    pending,
                                    reward=release_penalty,
                                    next_state=release_state,
                                    next_central_observation=step_transition_central_observation,
                                    done=False,
                                    discount_steps=max(step - pending.decision_step, 1),
                                    metadata={
                                        **(pending.metadata if isinstance(pending.metadata, dict) else {}),
                                        "pending_timeout_replan": bool(release_eval.release_as_timeout),
                                        "route_pending_replan": not bool(release_eval.release_as_timeout),
                                        "route_release_reason": release_eval.release_reason,
                                        "pending_age": int(pending_age),
                                        "pending_stall_age": int(release_eval.stall_age),
                                        "pending_elapsed_steps": int(max(step - pending.last_credit_step, 0)),
                                        "pending_resolution_mode": self.shared_policy.pending_resolution_mode(pending),
                                        "decision_finalized": False,
                                    },
                                )
                                self._accumulate_route_trace_reward(vehicle_route_trace, vehicle_id, release_penalty)
                                episode_return_total += release_penalty

                                if release_eval.release_as_timeout:
                                    decision_metrics["pending_decision_timeouts"] += 1
                                    decision_metrics["pending_resolved_timeout"] += 1
                                    if str((pending.metadata or {}).get("decision_origin_mode", pending.decision_origin_mode)) == "proactive":
                                        decision_metrics["proactive_pending_timeout_count"] += 1
                                else:
                                    decision_metrics["pending_resolved_abort_no_progress"] += 1
                                    if str((pending.metadata or {}).get("decision_origin_mode", pending.decision_origin_mode)) == "proactive":
                                        decision_metrics["proactive_pending_abort_count"] += 1
                                    if (
                                        release_eval.avoid_reopen_action is not None
                                        and release_eval.preferred_replan_action is not None
                                    ):
                                        stale_lane_now_replan_targets[(vehicle_id, current_edge)] = {
                                            "blocked_action": int(release_eval.avoid_reopen_action),
                                            "preferred_action": int(release_eval.preferred_replan_action),
                                            "until": int(step + self.decision_engine.cooldown_after_pending_release(timeout=False)),
                                        }
                                        decision_metrics["lane_now_replan_releases"] += 1
                                pending_decisions.pop(vehicle_id, None)
                                lane_change_cooldown_until[(vehicle_id, current_edge)] = (
                                    step + self.decision_engine.cooldown_after_pending_release(timeout=release_eval.release_as_timeout)
                                )
                                pending_release_info[vehicle_id] = {
                                    "edge": current_edge,
                                    "step": int(step),
                                    "reason": "timeout" if release_eval.release_as_timeout else "abort",
                                }
                                if release_eval.release_as_timeout:
                                    prev_edge_by_vehicle[vehicle_id] = current_edge
                                    continue

                        context = self._get_or_build_step_context(
                            step_context_cache,
                            vehicle_id,
                            current_edge,
                            vehicle.destination,
                            step,
                            snapshot=snapshot,
                        )
                        decision_metrics["decisions_considered"] += 1
                        reachable_set = set(context.reachable_with_lane_change_actions)
                        available_set = set(context.available_actions)
                        if reachable_set:
                            decision_metrics["reachable_lane_change_nonempty"] += 1
                            if not reachable_set.issubset(available_set):
                                decision_metrics["reachable_lane_change_excluded_any"] += 1
                            if reachable_set.isdisjoint(available_set):
                                decision_metrics["reachable_lane_change_excluded_all"] += 1
                        if vehicle_id in pending_decisions:
                            if self.shared_policy.pending_requires_active_same_edge_monitoring(pending_decisions[vehicle_id]):
                                self._record_skip(decision_metrics, "pending_hold")
                            prev_edge_by_vehicle[vehicle_id] = current_edge
                            continue

                        cooldown_until = lane_change_cooldown_until.get((vehicle_id, current_edge), -1)

                        # Route-epoch trigger: when enough edges have been completed, select a new macro route.
                        edges_done = vehicle_edges_since_reroute.get(vehicle_id, self.reroute_epoch_edges)
                        if edges_done >= self.reroute_epoch_edges:
                            previous_actor_route = vehicle_actor_owned_route.pop(vehicle_id, None)
                            candidates = self.route_generator.get_candidates(
                                current_edge,
                                vehicle.destination,
                                self._edge_density,
                                prev_route_edges=(
                                    list(previous_actor_route)
                                    if previous_actor_route is not None else None
                                ),
                            )
                            decision_metrics["route_candidate_count"] += len(candidates)
                            allowed_first_edges = {
                                self.decision_engine.get_next_edge(current_edge, action)
                                for action in context.available_actions
                            }
                            allowed_first_edges.discard(None)
                            feasible_candidates = filter_candidates_by_first_edges(
                                candidates,
                                allowed_first_edges,
                            )
                            decision_metrics["route_feasible_candidate_count"] += len(feasible_candidates)
                            if feasible_candidates:
                                self._episode_route_obs[vehicle_id] = pack_route_candidate_features(
                                    feasible_candidates,
                                    self.route_k,
                                    self.route_feature_dim,
                                )
                            else:
                                decision_metrics["route_no_feasible_candidates"] += 1
                                self._episode_route_obs.pop(vehicle_id, None)

                            for key in [key for key in step_state_cache if key[0] == vehicle_id]:
                                del step_state_cache[key]
                            state = self._get_or_encode_step_state(
                                step_state_cache,
                                step_context_cache,
                                vehicle_id,
                                vehicle,
                                step,
                                snapshot,
                                context=context,
                                coordination_state=step_coordination_state,
                            )

                            if vehicle_id in vehicle_route_trace:
                                prev_trace = vehicle_route_trace.pop(vehicle_id)
                                route_discount_steps = max(
                                    int(prev_trace.get("route_elapsed_steps", edges_done)),
                                    1,
                                )
                                self._record_immediate_mappo_transition(
                                    prev_trace,
                                    action=int(prev_trace.get("route_action", 0)),
                                    reward=0.0,
                                    next_state=state,
                                    next_central_observation=step_transition_central_observation,
                                    done=False,
                                    discount_steps=route_discount_steps,
                                    metadata={
                                        "route_epoch_finalized": True,
                                        "route_elapsed_steps": route_discount_steps,
                                    },
                                )

                            vehicle_edges_since_reroute[vehicle_id] = 0
                            if feasible_candidates:
                                valid_route_indices = list(range(len(feasible_candidates)))
                                route_selection = self.trainer.select_action(
                                    state,
                                    valid_route_indices,
                                    step_transition_central_observation,
                                    deterministic=False,
                                )
                                chosen_filtered_idx = int(route_selection.action)
                                chosen_route = feasible_candidates[chosen_filtered_idx].route_edges
                                original_idx = next(
                                    (i for i, c in enumerate(candidates) if c is feasible_candidates[chosen_filtered_idx]),
                                    chosen_filtered_idx,
                                )
                                self._record_route_actor_choice(
                                    decision_metrics,
                                    route_selection,
                                    feasible_candidates,
                                    chosen_filtered_idx,
                                )
                                decision_metrics[f"route_chosen_original_idx_{original_idx}"] += 1
                                decision_metrics["route_decisions_total"] += 1
                                decision_metrics["policy_actions"] += 1

                                route_applied = False
                                if len(chosen_route) > 1:
                                    try:
                                        traci.vehicle.setRoute(vehicle_id, chosen_route)
                                        vehicle_actor_owned_route[vehicle_id] = tuple(chosen_route)
                                        route_applied = True
                                    except Exception:
                                        decision_metrics["route_apply_failures"] += 1

                                if route_applied:
                                    new_trace = self._build_mappo_trace(
                                        state,
                                        step_transition_central_observation,
                                        route_selection,
                                    )
                                    new_trace["vehicle_id"] = vehicle_id
                                    new_trace["route_action"] = int(chosen_filtered_idx)
                                    new_trace["route_last_credit_step"] = int(step)
                                    new_trace["route_elapsed_steps"] = 0
                                    new_trace["route_edges"] = list(chosen_route)
                                    vehicle_route_trace[vehicle_id] = new_trace
                                    decision_metrics["route_actor_epochs_started"] += 1
                                    prev_edge_by_vehicle[vehicle_id] = current_edge
                                    continue

                                route_failure_penalty = self._clip_reward(-6.0)
                                episode_return_total += route_failure_penalty
                                failure_trace = self._build_mappo_trace(
                                    state,
                                    step_transition_central_observation,
                                    route_selection,
                                )
                                failure_trace["vehicle_id"] = vehicle_id
                                self._record_immediate_mappo_transition(
                                    failure_trace,
                                    action=int(chosen_filtered_idx),
                                    reward=route_failure_penalty,
                                    next_state=state,
                                    next_central_observation=step_transition_central_observation,
                                    done=False,
                                    metadata={"route_apply_failed": True},
                                )
                                self._episode_route_obs.pop(vehicle_id, None)
                                for key in [key for key in step_state_cache if key[0] == vehicle_id]:
                                    del step_state_cache[key]

                        # Ownership guard: while actor owns the route for this epoch skip all junction handling.
                        if vehicle_id in vehicle_actor_owned_route:
                            decision_metrics["route_actor_ownership_skips"] += 1
                            prev_edge_by_vehicle[vehicle_id] = current_edge
                            continue

                        decision_mode = self.shared_policy.classify_decision(context)
                        action_source = "forced" if decision_mode.mode == "forced" else ""
                        if decision_mode.mode == "forced":
                            decision_metrics["forced_actions"] += 1
                            if decision_mode.skip_reason:
                                decision_metrics[f"skip_reason_{decision_mode.skip_reason}"] += 1
                            state = self._get_or_encode_step_state(
                                step_state_cache,
                                step_context_cache,
                                vehicle_id,
                                vehicle,
                                step,
                                snapshot,
                                context=context,
                                coordination_state=step_coordination_state,
                            )
                            forced_selection = self.trainer.select_action(
                                state,
                                [decision_mode.action],
                                step_transition_central_observation,
                                deterministic=True,
                            )
                            decision_metrics["critic_only_queued"] += 1
                            effective_action = process_selected_action(
                                vehicle_id,
                                vehicle,
                                current_edge,
                                step,
                                snapshot,
                                context,
                                state,
                                decision_mode.action,
                                action_source,
                                selection=forced_selection,
                                central_observation=step_transition_central_observation,
                                coordination_state=step_coordination_state,
                                critic_only=True,
                            )
                            if effective_action is not None:
                                self.shared_policy.reserve_action(
                                    step_coordination_state,
                                    context=context,
                                    destination=vehicle.destination,
                                    action_idx=int(effective_action),
                                )
                            continue
                        elif decision_mode.mode == "skip":
                            if decision_mode.skip_reason == "no_branch":
                                self._record_skip(decision_metrics, "structural_no_branch")
                            elif decision_mode.skip_reason == "forced_single_path":
                                self._record_skip(decision_metrics, "structural_forced_single_path")
                            elif decision_mode.skip_reason == "forced_by_lane_commit":
                                self._record_skip(decision_metrics, "structural_forced_by_lane_commit")
                            elif decision_mode.skip_reason == "too_late_or_unreachable":
                                self._record_skip(decision_metrics, "structural_too_late_or_unreachable")
                            else:
                                self._record_skip(decision_metrics, "other")
                            decision_metrics[f"skip_reason_{decision_mode.skip_reason}"] += 1
                            prev_edge_by_vehicle[vehicle_id] = current_edge
                            continue
                        elif decision_mode.mode != "open":
                            prev_edge_by_vehicle[vehicle_id] = current_edge
                            continue
                        else:
                            decision_metrics["actionable_decision_points"] += 1
                            # available_actions = executor/safety feasibility set (unchanged semantics).
                            # policy_actions = stricter learning-time subset to reduce harmful overrides.
                            # Fallback machinery below remains the final safety layer.
                            cooldown_active = step < cooldown_until
                            policy_actions = self._policy_action_candidates(
                                context=context,
                                recent_history=decision_recent_history(vehicle_id),
                                cooldown_active=cooldown_active,
                                destination=vehicle.destination,
                                decision_metrics=decision_metrics,
                                coordination_state=step_coordination_state,
                            )
                            removed_actions = max(len(context.available_actions) - len(policy_actions), 0)
                            decision_metrics["policy_masked_actions_removed"] += removed_actions
                            open_decision_batch.append({
                                "vehicle_id": vehicle_id,
                                "vehicle": vehicle,
                                "current_edge": current_edge,
                                "snapshot": snapshot,
                                "context": context,
                                "recent_history": decision_recent_history(vehicle_id),
                                "cooldown_active": cooldown_active,
                                "policy_actions": policy_actions,
                            })
                            continue

                    if open_decision_batch:
                        step_central_observation = self._build_central_observation(
                            step=step,
                            total_controlled=total_controlled,
                            arrived_ids=arrived_ids,
                            step_snapshots=step_snapshots,
                            vehicles=vehicles,
                            pending_decisions=pending_decisions,
                            coordination_state=step_coordination_state,
                            open_decision_count=len(open_decision_batch),
                        )
                        ordered_entries = sorted(
                            open_decision_batch,
                            key=lambda entry: self.shared_policy.coordination_priority(
                                entry["context"],
                                destination=entry["vehicle"].destination,
                                edge_density_fn=self._edge_density,
                            ),
                        )
                        for entry in ordered_entries:
                            vehicle_id = entry["vehicle_id"]
                            current_edge = entry["current_edge"]
                            state = self._get_or_encode_step_state(
                                step_state_cache,
                                step_context_cache,
                                vehicle_id,
                                entry["vehicle"],
                                step,
                                entry["snapshot"],
                                context=entry["context"],
                                coordination_state=step_coordination_state,
                            )
                            policy_actions = self.shared_policy.rank_policy_actions(
                                context=entry["context"],
                                actions=entry["policy_actions"],
                                destination=entry["vehicle"].destination,
                                distance_fn=self.get_distance_to_destination,
                                edge_density_fn=self._edge_density,
                                recent_history=entry["recent_history"],
                                coordination_state=step_coordination_state,
                            )
                            policy_actions = force_stale_lane_now_replan_if_available(
                                vehicle_id,
                                current_edge,
                                step,
                                entry["context"],
                                policy_actions,
                            )
                            if not policy_actions:
                                self._record_skip(decision_metrics, "actionable_no_candidate")
                                decision_metrics["actionable_skips"] += 1
                                prev_edge_by_vehicle[vehicle_id] = current_edge
                                continue
                            action = int(policy_actions[0])
                            action_source = "heuristic"
                            effective_action = process_selected_action(
                                vehicle_id,
                                entry["vehicle"],
                                current_edge,
                                step,
                                entry["snapshot"],
                                entry["context"],
                                state,
                                action,
                                action_source,
                                selection=None,
                                central_observation=None,
                                coordination_state=step_coordination_state,
                            )
                            if effective_action is not None:
                                self.shared_policy.reserve_action(
                                    step_coordination_state,
                                    context=entry["context"],
                                    destination=entry["vehicle"].destination,
                                    action_idx=int(effective_action),
                                )

                    simulation_step()

                    arrived_this_step = set(simulation_get_arrived_ids())
                    teleported_ids = self.get_teleport_ids()
                    ever_teleported_controlled_ids.update(tid for tid in teleported_ids if tid in vehicles)
                    live_after_step = set(vehicle_get_ids())
                    removed_controlled_ids = {
                        vid for vid in controlled_live_ids
                        if vid not in live_after_step and vid in vehicles
                    }

                    if teleported_ids:
                        episode_teleport_events += len(teleported_ids)
                        decision_metrics["teleports"] += len(teleported_ids)
                        teleported_controlled_ids.update(tid for tid in teleported_ids if tid in vehicles)
                        for tid in teleported_ids:
                            if tid not in vehicles:
                                continue
                            edge_id = last_seen_edge_by_vehicle.get(tid)
                            if not edge_id:
                                decision_metrics["teleport_inferred_yield_or_deadlock"] += 1
                                continue
                            edge_density = self._edge_density(edge_id)
                            if edge_density >= self.teleport_jam_density_threshold:
                                decision_metrics["teleport_inferred_jam"] += 1
                            else:
                                decision_metrics["teleport_inferred_yield_or_deadlock"] += 1

                    for removed_id in sorted(removed_controlled_ids):
                        if removed_id in final_outcome_by_vehicle:
                            continue
                        vehicle = vehicles[removed_id]
                        last_seen_edge = last_seen_edge_by_vehicle.get(removed_id, "<unknown>")
                        outcome = self._classify_terminal_outcome(
                            removed_id,
                            arrived_this_step,
                            teleported_ids,
                            is_live=(removed_id in live_after_step),
                        )
                        final_outcome_by_vehicle[removed_id] = outcome
                        ever_teleported = (removed_id in ever_teleported_controlled_ids)
                        reached_global_destination = (outcome == "global_arrival")
                        if reached_global_destination:
                            arrived_ids.add(removed_id)
                            if removed_id not in completed_travel_time_ids:
                                completed_travel_time_ids.add(removed_id)
                                completed_travel_times.append(max(float(step) - float(vehicle.start_time), 0.0))
                            if last_seen_edge != vehicle.destination:
                                arrived_with_prestep_edge_not_destination += 1

                        arrived_debug_records.append({
                            "vehicle_id": removed_id,
                            "global_destination": vehicle.destination,
                            "last_seen_edge": last_seen_edge,
                            "observed_destination_edge": reached_global_destination,
                            "last_planned_terminal_edge": last_planned_terminal_edge_by_vehicle.get(removed_id, "<unset>"),
                            "reached_global_destination": reached_global_destination,
                            "terminal_outcome": outcome,
                        })
                        if removed_id in vehicle_route_trace:
                            self._vehicle_last_buffer_pos.pop(removed_id, None)
                        terminal_reward = self._finalize_terminal_transition(
                            vehicle_id=removed_id,
                            vehicle=vehicle,
                            outcome=outcome,
                            step=step,
                            pending_decisions=pending_decisions,
                            terminal_recorded_ids=terminal_recorded_ids,
                            last_snapshot_by_vehicle=last_snapshot_by_vehicle,
                            last_seen_edge_by_vehicle=last_seen_edge_by_vehicle,
                            decision_metrics=decision_metrics,
                            decision_latency_steps=decision_latency_steps,
                            finalized_decision_rewards=finalized_decision_rewards,
                            decision_debug_rows=decision_debug_rows,
                            episode=episode,
                            hard_brake_counts_by_vehicle=hard_brake_counts_by_vehicle,
                            in_arrived_ids=(removed_id in arrived_this_step),
                            in_teleport_ids=(removed_id in teleported_ids),
                            ever_teleported=ever_teleported,
                        )
                        episode_return_total += terminal_reward
                        # Finalize any open route-epoch trace for this vehicle.
                        if removed_id in vehicle_route_trace:
                            final_trace = vehicle_route_trace.pop(removed_id)
                            terminal_snap = last_snapshot_by_vehicle.get(removed_id)
                            terminal_state = (
                                self.make_terminal_next_state_from_snapshot(
                                    terminal_snap, vehicles[removed_id].destination,
                                    vehicle=vehicles.get(removed_id), step=step
                                ) if terminal_snap is not None
                                else np.zeros((1, self.state_size), dtype=np.float32)
                            )
                            self._record_immediate_mappo_transition(
                                final_trace,
                                action=int(final_trace.get("route_action", 0)),
                                reward=terminal_reward,
                                next_state=terminal_state,
                                next_central_observation=step_transition_central_observation,
                                done=True,
                                discount_steps=max(
                                    int(final_trace.get(
                                        "route_elapsed_steps",
                                        vehicle_edges_since_reroute.get(removed_id, 1),
                                    )),
                                    1,
                                ),
                            )
                        vehicle_route_trace.pop(removed_id, None)
                        vehicle_edges_since_reroute.pop(removed_id, None)
                        vehicle_actor_owned_route.pop(removed_id, None)
                        self.cleanup_vehicle_state(
                            removed_id,
                            pending_decisions,
                            prev_edge_by_vehicle,
                            last_seen_edge_by_vehicle,
                            last_planned_terminal_edge_by_vehicle,
                            recent_edge_history,
                            last_snapshot_by_vehicle,
                        )
                        emergency_brake_active_by_vehicle.pop(removed_id, None)
                        prev_speed_by_vehicle.pop(removed_id, None)
                        last_observed_brake_step_by_vehicle.pop(removed_id, None)

                    if step % self.step_log_every == 0:
                        self._print_step_progress(
                            episode=episode,
                            step=step,
                            total_controlled=total_controlled,
                            arrived_ids=arrived_ids,
                            decision_metrics=decision_metrics,
                            mean_density_samples=mean_density_samples,
                            congestion_high_pressure_steps=congestion_high_pressure_steps,
                        )

                    # process = psutil.Process(os.getpid())

                    # if step % 100 == 0:
                    #     print(
                    #         "RAM_MB=", process.memory_info().rss / 1024 / 1024,
                    #         " replay=", len(self.trainer.memory),
                    #         " dist_cache=", len(self._distance_cache),
                    #     )

            finally:
                if last_step_executed >= (MAX_SIMULATION_STEPS - 1):
                    alive_at_step_cap_ids = {vid for vid in vehicle_get_ids() if vid in vehicles}
                    for vid in alive_at_step_cap_ids:
                        if vid not in final_outcome_by_vehicle:
                            final_outcome_by_vehicle[vid] = "alive_at_step_cap"
                        timeout_vehicle = vehicles[vid]
                        if vid in vehicle_route_trace:
                            self._vehicle_last_buffer_pos.pop(vid, None)
                        terminal_reward = self._finalize_terminal_transition(
                            vehicle_id=vid,
                            vehicle=timeout_vehicle,
                            outcome="timeout",
                            step=last_step_executed,
                            pending_decisions=pending_decisions,
                            terminal_recorded_ids=terminal_recorded_ids,
                            last_snapshot_by_vehicle=last_snapshot_by_vehicle,
                            last_seen_edge_by_vehicle=last_seen_edge_by_vehicle,
                            decision_metrics=decision_metrics,
                            decision_latency_steps=decision_latency_steps,
                            finalized_decision_rewards=finalized_decision_rewards,
                            decision_debug_rows=decision_debug_rows,
                            episode=episode,
                            hard_brake_counts_by_vehicle=hard_brake_counts_by_vehicle,
                            in_arrived_ids=False,
                            in_teleport_ids=False,
                            ever_teleported=(vid in ever_teleported_controlled_ids),
                        )
                        episode_return_total += terminal_reward
                        if vid in vehicle_route_trace:
                            final_trace = vehicle_route_trace.pop(vid)
                            terminal_snap = last_snapshot_by_vehicle.get(vid)
                            terminal_state = (
                                self.make_terminal_next_state_from_snapshot(
                                    terminal_snap, timeout_vehicle.destination,
                                    vehicle=timeout_vehicle, step=last_step_executed
                                ) if terminal_snap is not None
                                else np.zeros((1, self.state_size), dtype=np.float32)
                            )
                            self._record_immediate_mappo_transition(
                                final_trace,
                                action=int(final_trace.get("route_action", 0)),
                                reward=terminal_reward,
                                next_state=terminal_state,
                                next_central_observation=step_transition_central_observation,
                                done=True,
                                discount_steps=max(
                                    int(final_trace.get(
                                        "route_elapsed_steps",
                                        vehicle_edges_since_reroute.get(vid, 1),
                                    )),
                                    1,
                                ),
                            )
                        vehicle_actor_owned_route.pop(vid, None)
                global_arrival_count = sum(1 for outcome in final_outcome_by_vehicle.values() if outcome == "global_arrival")
                terminal_teleport_count = sum(1 for outcome in final_outcome_by_vehicle.values() if outcome == "teleport")
                controlled_ever_teleported = len(ever_teleported_controlled_ids)
                arrived_after_teleport = sum(
                    1
                    for vid, outcome in final_outcome_by_vehicle.items()
                    if outcome == "global_arrival" and vid in ever_teleported_controlled_ids
                )
                clean_arrivals_without_teleport = max(global_arrival_count - arrived_after_teleport, 0)
                removed_nonarrival_count = sum(1 for outcome in final_outcome_by_vehicle.values() if outcome == "removed_nonarrival")
                alive_at_step_cap_count = sum(1 for outcome in final_outcome_by_vehicle.values() if outcome == "alive_at_step_cap")
                completion_rate = (
                    global_arrival_count / float(total_controlled)
                    if total_controlled > 0 else 0.0
                )
                avg_travel_time = float(np.mean(completed_travel_times)) if completed_travel_times else 0.0
                p50_travel_time = float(np.percentile(completed_travel_times, 50)) if completed_travel_times else 0.0
                p90_travel_time = float(np.percentile(completed_travel_times, 90)) if completed_travel_times else 0.0
                avg_return_per_vehicle = episode_return_total / float(total_controlled) if total_controlled > 0 else 0.0
                mean_pending_age = (
                    float(np.mean(pending_age_samples)) if pending_age_samples else 0.0
                )
                mean_decision_latency_steps = (
                    float(np.mean(decision_latency_steps)) if decision_latency_steps else 0.0
                )
                mean_reward_per_finalized_decision = (
                    float(np.mean(finalized_decision_rewards)) if finalized_decision_rewards else 0.0
                )
                mean_route_difficulty_eta = (
                    float(np.mean(route_difficulty_etas)) if route_difficulty_etas else 0.0
                )
                p50_route_difficulty_eta = (
                    float(np.percentile(route_difficulty_etas, 50)) if route_difficulty_etas else 0.0
                )
                p90_route_difficulty_eta = (
                    float(np.percentile(route_difficulty_etas, 90)) if route_difficulty_etas else 0.0
                )
                mean_network_density = (
                    float(np.mean(mean_density_samples)) if mean_density_samples else 0.0
                )
                p95_network_density = (
                    float(np.percentile(p95_density_samples, 95)) if p95_density_samples else 0.0
                )
                lane_change_request_accepted_rate = (
                    float(decision_metrics["lane_change_success"]) / float(max(decision_metrics["lane_change_attempts"], 1.0))
                )
                lane_change_observe_resolution_rate = (
                    float(min(decision_metrics["lane_change_observe_success"], decision_metrics["lane_change_observe_started"]))
                    / float(max(decision_metrics["lane_change_observe_started"], 1.0))
                )
                lane_change_observe_success_overcount = max(
                    float(decision_metrics["lane_change_observe_success"]) - float(decision_metrics["lane_change_observe_started"]),
                    0.0,
                )
                if completed_travel_times:
                    tail_travel_times = [tt for tt in completed_travel_times if tt >= p90_travel_time]
                else:
                    tail_travel_times = []
                unfinished_count = int(alive_at_step_cap_count + removed_nonarrival_count)
                noncompletion_rate = 1.0 - completion_rate
                # Censored metric: unfinished vehicles imputed at MAX_SIMULATION_STEPS.
                # This prevents survivorship bias when completion_rate is low.
                penalized_travel_times = list(completed_travel_times) + [float(MAX_SIMULATION_STEPS)] * unfinished_count
                penalized_avg_travel_time = float(np.mean(penalized_travel_times)) if penalized_travel_times else float(MAX_SIMULATION_STEPS)
                tail_vehicles_over_p90_count = int(len(tail_travel_times) + unfinished_count)
                tail_reference = (
                    float(np.mean(tail_travel_times))
                    if tail_travel_times else p90_travel_time
                )
                if unfinished_count > 0 and last_step_executed >= 0:
                    # Treat unfinished controlled vehicles as unresolved long-tail travel times.
                    tail_reference = (
                        (tail_reference * len(tail_travel_times)) + (float(last_step_executed) * unfinished_count)
                    ) / max(len(tail_travel_times) + unfinished_count, 1)
                tail_completion_gap_steps = float(max(tail_reference - p50_travel_time, 0.0))
                loop_reason_counts = {
                    "short_cycle": float(decision_metrics["short_cycle_events"]),
                    "aba_bounce": float(decision_metrics["aba_bounce_events"]),
                    "dead_end_reentry": float(decision_metrics["dead_end_reentry_events"]),
                    "long_horizon": float(decision_metrics["long_horizon_loop_events"]),
                    "revisit_without_progress": float(decision_metrics["revisit_without_progress_events"]),
                }
                dominant_loop_reason = "none"
                if sum(loop_reason_counts.values()) > 0:
                    dominant_loop_reason = max(loop_reason_counts.items(), key=lambda item: item[1])[0]
                decisions_skipped_actionable = max(
                    float(decision_metrics["decisions_skipped"]) - float(decision_metrics["skipped_pending_hold"]),
                    0.0,
                )
                aggregate_actionable_skip_to_finalized_ratio = decisions_skipped_actionable / float(
                    max(decision_metrics["decisions_finalized"], 1.0)
                )
                social_regret_mean = (
                    float(np.mean(social_regret_samples)) if social_regret_samples else 0.0
                )
                social_regret_p90 = (
                    float(np.percentile(social_regret_samples, 90)) if social_regret_samples else 0.0
                )
                social_best_action_chosen_rate = (
                    float(decision_metrics["social_best_action_chosen"]) / float(max(decision_metrics["social_regret_count"], 1.0))
                )
                actionable_skip_ratio = (
                    float(decision_metrics["actionable_skips"]) / float(max(decision_metrics["actionable_decision_points"], 1.0))
                )
                structural_skip_total = float(
                    decision_metrics["skipped_structural_no_branch"]
                    + decision_metrics["skipped_structural_forced_single_path"]
                    + decision_metrics["skipped_structural_forced_by_lane_commit"]
                    + decision_metrics["skipped_structural_too_late_or_unreachable"]
                )
                structural_skip_ratio = structural_skip_total / float(
                    max(decision_metrics["decisions_considered"], 1.0)
                )
                pending_resolved_total = float(
                    decision_metrics["pending_resolved_success"]
                    + decision_metrics["pending_resolved_timeout"]
                    + decision_metrics["pending_resolved_abort_no_progress"]
                )
                pending_resolution_success_rate = (
                    float(decision_metrics["pending_resolved_success"]) / float(max(pending_resolved_total, 1.0))
                )
                pending_timeout_rate = (
                    float(decision_metrics["pending_resolved_timeout"]) / float(max(pending_resolved_total, 1.0))
                )
                pending_abort_rate = (
                    float(decision_metrics["pending_resolved_abort_no_progress"]) / float(max(pending_resolved_total, 1.0))
                )
                if len(completed_travel_times) > 1 and avg_travel_time > 0:
                    sorted_tt = sorted(float(tt) for tt in completed_travel_times)
                    n_tt = len(sorted_tt)
                    weighted_sum = sum((idx + 1) * value for idx, value in enumerate(sorted_tt))
                    delay_fairness_gini = float((2.0 * weighted_sum) / (n_tt * sum(sorted_tt)) - (n_tt + 1) / n_tt)
                    delay_fairness_gini = float(np.clip(delay_fairness_gini, 0.0, 1.0))
                else:
                    delay_fairness_gini = 0.0
                loop_after_fallback_rate = (
                    float(decision_metrics["loop_after_fallback_events"]) / float(max(decision_metrics["fallback_finalized"], 1.0))
                )
                p95_to_p50_travel_ratio = (
                    float(np.percentile(completed_travel_times, 95) / max(p50_travel_time, 1e-6))
                    if completed_travel_times else 0.0
                )
                timeout_rate = float(alive_at_step_cap_count) / float(max(total_controlled, 1))
                fallback_rate_per_opened_decision = (
                    float(decision_metrics["fallback_overrides"]) / float(max(decision_metrics["decisions_opened"], 1.0))
                )
                finalized_opened_proactive_ratio = (
                    float(decision_metrics["proactive_decisions_finalized"])
                    / float(max(decision_metrics["proactive_decisions_opened"], 1.0))
                )
                finalized_opened_lane_now_ratio = (
                    float(decision_metrics["lane_now_decisions_finalized"])
                    / float(max(decision_metrics["lane_now_decisions_opened"], 1.0))
                )
                controlled_teleport_rate = float(len(teleported_controlled_ids)) / float(max(total_controlled, 1))
                pending_end_summary = self._summarize_pending_backlog(
                    pending_decisions,
                    last_step_executed,
                )
                oldest_pending_summary = self._format_pending_descriptor(
                    pending_end_summary.get("oldest_descriptor")
                )
                emergency_brake_total = float(max(decision_metrics["emergency_brake_events"], 1.0))
                emergency_brake_rate_per_100_decisions = (
                    100.0 * float(decision_metrics["emergency_brake_events"])
                    / float(max(decision_metrics["decisions_opened"], 1.0))
                )
                emergency_brake_rate_per_100_arrivals = (
                    100.0 * float(decision_metrics["emergency_brake_events"])
                    / float(max(global_arrival_count, 1.0))
                )
                emergency_brake_leader_share = (
                    float(decision_metrics["emergency_brake_due_to_leader"]) / emergency_brake_total
                )
                emergency_brake_congestion_share = (
                    float(decision_metrics["emergency_brake_due_to_congestion"]) / emergency_brake_total
                )
                emergency_brake_junction_share = (
                    float(decision_metrics["emergency_brake_near_junction"]) / emergency_brake_total
                )
                emergency_brake_other_share = (
                    float(decision_metrics["emergency_brake_other_reason"]) / emergency_brake_total
                )
                emergency_brake_after_fallback_share = (
                    float(decision_metrics["emergency_brake_after_fallback"]) / emergency_brake_total
                )
                emergency_brake_after_proactive_share = (
                    float(decision_metrics["emergency_brake_after_proactive"]) / emergency_brake_total
                )
                emergency_brake_after_lane_now_share = (
                    float(decision_metrics["emergency_brake_after_lane_now"]) / emergency_brake_total
                )
                emergency_brake_without_recent_decision_share = (
                    float(decision_metrics["emergency_brake_without_recent_decision"]) / emergency_brake_total
                )
                top_emergency_brake_edges = self._format_top_counts(emergency_brake_events_by_edge)
                top_emergency_brake_vehicles = self._format_top_counts(emergency_brake_events_by_vehicle)
                top_loop_signal_edges = self._format_top_counts(loop_signal_events_by_edge)
                top_loop_repeat_only_edges = self._format_top_counts(loop_repeat_only_events_by_edge)
                reachable_lane_change_excluded_any_rate = (
                    float(decision_metrics["reachable_lane_change_excluded_any"])
                    / float(max(decision_metrics["reachable_lane_change_nonempty"], 1.0))
                )
                policy_lane_now_collapse_rate = (
                    float(decision_metrics["policy_candidates_collapsed_to_lane_now_only"])
                    / float(max(decision_metrics["policy_candidates_with_broader_available"], 1.0))
                )
                route_decision_count = float(max(decision_metrics["route_decisions_total"], 1.0))
                route_logit_margin_count = float(max(decision_metrics["route_logit_margin_count"], 1.0))
                route_choice_nonzero_rate = (
                    float(decision_metrics["route_choice_nonzero_count"]) / route_decision_count
                )
                route_mean_valid_candidates = (
                    float(decision_metrics["route_valid_candidate_sum"]) / route_decision_count
                )
                route_mean_logit_margin = (
                    float(decision_metrics["route_logit_margin_sum"]) / route_logit_margin_count
                )
                route_mean_chosen_length_norm = (
                    float(decision_metrics["route_chosen_length_norm_sum"]) / route_decision_count
                )
                route_mean_chosen_eta_norm = (
                    float(decision_metrics["route_chosen_eta_norm_sum"]) / route_decision_count
                )
                route_mean_chosen_density = (
                    float(decision_metrics["route_chosen_density_sum"]) / route_decision_count
                )
                route_mean_chosen_first_density = (
                    float(decision_metrics["route_chosen_first_density_sum"]) / route_decision_count
                )

                rolling_teleport_events.append(float(episode_teleport_events))
                rolling_teleported_controlled.append(float(len(teleported_controlled_ids)))
                rolling_completion_rate.append(float(completion_rate))
                rolling_avg_return.append(float(avg_return_per_vehicle))
                rolling_avg_travel_time.append(float(avg_travel_time))
                rolling_mismatch.append(float(decision_metrics["route_mismatch"]))

                roll_tele_events = sum(rolling_teleport_events) / len(rolling_teleport_events)
                roll_tele_ctrl = sum(rolling_teleported_controlled) / len(rolling_teleported_controlled)
                roll_completion = sum(rolling_completion_rate) / len(rolling_completion_rate)
                roll_return = sum(rolling_avg_return) / len(rolling_avg_return)
                roll_avg_travel_time = sum(rolling_avg_travel_time) / len(rolling_avg_travel_time)
                roll_mismatch = sum(rolling_mismatch) / len(rolling_mismatch)

                self.trainer.set_entropy_progress(episode / max(self.episodes - 1, 1))
                updated_transition_count = self.trainer.update()
                pending_observe_abort_total = (
                    float(decision_metrics["pending_release_observe_abort_no_progress"])
                    + float(decision_metrics["pending_release_observe_abort_low_speed"])
                    + float(decision_metrics["pending_release_observe_abort_commit_window"])
                )
                print(
                    f"\n[EP {episode:03d} DONE] updates={self.trainer.train_steps} "
                    f"rollout={updated_transition_count} buffered={len(self.trainer.memory)} "
                    f"ret={avg_return_per_vehicle:.3f} "
                    f"done={global_arrival_count}/{total_controlled} "
                    f"failed={max(total_controlled-global_arrival_count,0)} avg_tt={avg_travel_time:.2f} "
                    f"p50_tt={p50_travel_time:.2f} p90_tt={p90_travel_time:.2f}"
                )
                print(
                    f"  rolling({len(rolling_teleport_events)}): completion={roll_completion:.1%} "
                    f"avg_return_per_vehicle={roll_return:.3f} avg_tt={roll_avg_travel_time:.2f} teleports/ep={roll_tele_events:.2f} "
                    f"teleported_ctrl/ep={roll_tele_ctrl:.2f} mismatch/ep={roll_mismatch:.2f}"
                )
                print(
                    "  decisions: opened={:.0f} finalized={:.0f} skipped={:.0f} forced={:.0f} "
                    "fallback={:.0f} override_ratio={:.1%} actionable_skip={:.1%}".format(
                        decision_metrics["decisions_opened"],
                        decision_metrics["decisions_finalized"],
                        decision_metrics["decisions_skipped"],
                        decision_metrics["forced_actions"],
                        decision_metrics["fallback_overrides"],
                        decision_metrics["override_events_total"] / max(decision_metrics["decisions_opened"], 1.0),
                        actionable_skip_ratio,
                    )
                )
                print(
                    "  route_actor: decisions={:.0f} epochs={:.0f} choices=[{:.0f},{:.0f},{:.0f},{:.0f}] "
                    "nonzero={:.1%} valid_mean={:.2f} margin={:.3f} owns_skips={:.0f}".format(
                        decision_metrics["route_decisions_total"],
                        decision_metrics["route_actor_epochs_started"],
                        decision_metrics["route_choice_idx_0"],
                        decision_metrics["route_choice_idx_1"],
                        decision_metrics["route_choice_idx_2"],
                        decision_metrics["route_choice_idx_3"],
                        route_choice_nonzero_rate,
                        route_mean_valid_candidates,
                        route_mean_logit_margin,
                        decision_metrics["route_actor_ownership_skips"],
                    )
                )
                print(
                    "  pending: timeout={:.0f} open_end={} observe/route={:.0f}/{:.0f} "
                    "lane_now/proactive={:.0f}/{:.0f} active_monitor={:.0f} mean_active_age={:.1f} "
                    "resolve(success/timeout/abort)={:.1%}/{:.1%}/{:.1%} "
                    "observe(start/success/abort)={:.0f}/{:.0f}/{:.0f} "
                    "release(no_prog/wrong_lane/stall/hard)={:.0f}/{:.0f}/{:.0f}/{:.0f}".format(
                        decision_metrics["pending_decision_timeouts"],
                        pending_end_summary["total_open"],
                        pending_end_summary["observe_open"],
                        pending_end_summary["route_open"],
                        pending_end_summary["lane_now_open"],
                        pending_end_summary["proactive_open"],
                        pending_end_summary["active_monitoring_open"],
                        mean_pending_age,
                        pending_resolution_success_rate,
                        pending_timeout_rate,
                        pending_abort_rate,
                        decision_metrics["lane_change_observe_started"],
                        decision_metrics["lane_change_observe_success"],
                        pending_observe_abort_total,
                        decision_metrics["pending_release_route_no_progress_abort"],
                        decision_metrics["pending_release_wrong_lane_commit"],
                        decision_metrics["pending_release_route_stall_timeout"],
                        decision_metrics["pending_release_route_hard_timeout"],
                    )
                )
                print(
                    "  pending_end: mean_age(all/active/lane_now)={:.1f}/{:.1f}/{:.1f} "
                    "max_age={:.0f} stall_mean/max={:.1f}/{:.0f} oldest={}".format(
                        pending_end_summary["mean_age_all"],
                        pending_end_summary["mean_age_active"],
                        pending_end_summary["mean_age_lane_now"],
                        pending_end_summary["max_age_all"],
                        pending_end_summary["mean_stall_age"],
                        pending_end_summary["max_stall_age"],
                        oldest_pending_summary or "none",
                    )
                )
                print(
                    "  loops: total={:.0f} signal_events={:.0f} repeat_only={:.0f} "
                    "short={:.0f} aba={:.0f} dead_end={:.0f} long_horizon={:.0f} revisit_no_progress={:.0f} "
                    "dominant={} loop_after_fallback={:.1%}".format(
                        decision_metrics["loop_events"],
                        decision_metrics["loop_signal_events_total"],
                        decision_metrics["loop_repeat_only_events"],
                        decision_metrics["short_cycle_events"],
                        decision_metrics["aba_bounce_events"],
                        decision_metrics["dead_end_reentry_events"],
                        decision_metrics["long_horizon_loop_events"],
                        decision_metrics["revisit_without_progress_events"],
                        dominant_loop_reason,
                        loop_after_fallback_rate,
                    )
                )
                print(
                    "  brakes: total={:.0f} leader/congestion/junction/other={:.0f}/{:.0f}/{:.0f}/{:.0f} "
                    "source(fallback/proactive/lane_now/no_recent)={:.0f}/{:.0f}/{:.0f}/{:.0f} "
                    "rate/100_open={:.1f} rate/100_arrived={:.1f}".format(
                        decision_metrics["emergency_brake_events"],
                        decision_metrics["emergency_brake_due_to_leader"],
                        decision_metrics["emergency_brake_due_to_congestion"],
                        decision_metrics["emergency_brake_near_junction"],
                        decision_metrics["emergency_brake_other_reason"],
                        decision_metrics["emergency_brake_after_fallback"],
                        decision_metrics["emergency_brake_after_proactive"],
                        decision_metrics["emergency_brake_after_lane_now"],
                        decision_metrics["emergency_brake_without_recent_decision"],
                        emergency_brake_rate_per_100_decisions,
                        emergency_brake_rate_per_100_arrivals,
                    )
                )
                print(
                    "  hotspots: brake_edges={} brake_vehicles={} pending_edges={} loop_signal_edges={} loop_repeat_only_edges={}".format(
                        top_emergency_brake_edges or "none",
                        top_emergency_brake_vehicles or "none",
                        pending_end_summary["hot_edges"] or "none",
                        top_loop_signal_edges or "none",
                        top_loop_repeat_only_edges or "none",
                    )
                )
                print(
                    "  network: density(mean/p95)={:.4f}/{:.4f} congestion_steps={} "
                    "teleported_ctrl={} alive_at_step_cap={} tail_over_p90={} gini={:.3f}".format(
                        mean_network_density,
                        p95_network_density,
                        congestion_high_pressure_steps,
                        len(teleported_controlled_ids),
                        alive_at_step_cap_count,
                        tail_vehicles_over_p90_count,
                        delay_fairness_gini,
                    )
                )
                print(
                    "  outcomes: teleport_terminal={} removed_nonarrival={} arrived_after_teleport={} "
                    "route_mismatch={} tail_gap_vs_p50={:.1f}".format(
                        terminal_teleport_count,
                        removed_nonarrival_count,
                        arrived_after_teleport,
                        decision_metrics["route_mismatch"],
                        tail_completion_gap_steps,
                    )
                )

                if self.debug_exit_diagnostics:
                    mismatched_arrivals = [
                        record for record in arrived_debug_records
                        if (
                            record["terminal_outcome"] == "global_arrival"
                            and record["last_seen_edge"] != record["global_destination"]
                        )
                    ]
                    print(
                        f"Arrival debug | total_arrived={len(arrived_debug_records)}, "
                        f"arrived_with_prestep_edge_not_destination={len(mismatched_arrivals)}"
                    )

                    for record in mismatched_arrivals[:self.debug_exit_diagnostics_limit]:
                        print(
                            "  ARRIVED_PRESTEP_EDGE_MISMATCH "
                            f"vehicle={record['vehicle_id']} "
                            f"last_seen_edge={record['last_seen_edge']} "
                            f"last_planned_terminal_edge={record['last_planned_terminal_edge']} "
                            f"global_destination={record['global_destination']}"
                        )

                    if len(mismatched_arrivals) > self.debug_exit_diagnostics_limit:
                        print(
                            "  ARRIVED_PRESTEP_EDGE_MISMATCH ... "
                            f"{len(mismatched_arrivals) - self.debug_exit_diagnostics_limit} more vehicles"
                        )

                    removed_nonarrival_ids = sorted(
                        [vid for vid, outcome in final_outcome_by_vehicle.items() if outcome == "removed_nonarrival"]
                    )
                    for vehicle_id in removed_nonarrival_ids[:self.debug_exit_diagnostics_limit]:
                        vehicle = vehicles[vehicle_id]
                        print(
                            "  EXITED_WITHOUT_DEST "
                            f"vehicle={vehicle_id} "
                            f"last_seen_edge={last_seen_edge_by_vehicle.get(vehicle_id, '<unknown>')} "
                            f"last_planned_terminal_edge={last_planned_terminal_edge_by_vehicle.get(vehicle_id, '<unset>')} "
                            f"global_destination={vehicle.destination}"
                        )

                    if len(removed_nonarrival_ids) > self.debug_exit_diagnostics_limit:
                        print(
                            "  EXITED_WITHOUT_DEST ... "
                            f"{len(removed_nonarrival_ids) - self.debug_exit_diagnostics_limit} more vehicles"
                        )
                for vid, pending in pending_decisions.items():
                    if vid in pending_debug_logged_ids:
                        continue
                    decision_debug_rows.append(self._build_decision_debug_row(
                        episode=episode,
                        step=last_step_executed,
                        pending=pending,
                        actual_next_edge="",
                        finalized=0,
                        finalize_delay_steps=max(last_step_executed - pending.decision_step, 0),
                    ))
                self._append_decision_debug_rows(decision_debug_rows)
                traci.close()
                trainer_row = self.trainer.episode_metric_row(trainer_counter_start)
                with open(self.metrics_csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fields)
                    row = {
                        "episode": episode,
                        **trainer_row,
                        "algorithm": "mappo",
                        "episode_return_total": episode_return_total,
                        "avg_return_per_vehicle": avg_return_per_vehicle,
                        "completion_rate": completion_rate,
                        "avg_travel_time": avg_travel_time,
                        "p50_travel_time": p50_travel_time,
                        "p90_travel_time": p90_travel_time,
                        "teleports": episode_teleport_events,
                        "controlled_ever_teleported": controlled_ever_teleported,
                        "arrived_after_teleport": arrived_after_teleport,
                        "clean_arrivals_without_teleport": clean_arrivals_without_teleport,
                        "forced_actions": decision_metrics["forced_actions"],
                        "critic_only_queued": decision_metrics["critic_only_queued"],
                        "route_decisions_total": decision_metrics["route_decisions_total"],
                        "route_candidate_count": decision_metrics["route_candidate_count"],
                        "route_feasible_candidate_count": decision_metrics["route_feasible_candidate_count"],
                        "route_actor_epochs_started": decision_metrics["route_actor_epochs_started"],
                        "route_actor_ownership_skips": decision_metrics["route_actor_ownership_skips"],
                        "route_no_feasible_candidates": decision_metrics["route_no_feasible_candidates"],
                        "route_apply_failures": decision_metrics["route_apply_failures"],
                        "route_choice_idx_0": decision_metrics["route_choice_idx_0"],
                        "route_choice_idx_1": decision_metrics["route_choice_idx_1"],
                        "route_choice_idx_2": decision_metrics["route_choice_idx_2"],
                        "route_choice_idx_3": decision_metrics["route_choice_idx_3"],
                        "route_valid_candidates_1": decision_metrics["route_valid_candidates_1"],
                        "route_valid_candidates_2": decision_metrics["route_valid_candidates_2"],
                        "route_valid_candidates_3": decision_metrics["route_valid_candidates_3"],
                        "route_valid_candidates_4": decision_metrics["route_valid_candidates_4"],
                        "route_choice_nonzero_rate": route_choice_nonzero_rate,
                        "route_mean_valid_candidates": route_mean_valid_candidates,
                        "route_mean_logit_margin": route_mean_logit_margin,
                        "route_mean_chosen_length_norm": route_mean_chosen_length_norm,
                        "route_mean_chosen_eta_norm": route_mean_chosen_eta_norm,
                        "route_mean_chosen_density": route_mean_chosen_density,
                        "route_mean_chosen_first_density": route_mean_chosen_first_density,
                        "decisions_considered": decision_metrics["decisions_considered"],
                        "decisions_opened": decision_metrics["decisions_opened"],
                        "decisions_finalized": decision_metrics["decisions_finalized"],
                        "decisions_skipped": decision_metrics["decisions_skipped"],
                        "decisions_skipped_actionable": decisions_skipped_actionable,
                        "skipped_pending_hold": decision_metrics["skipped_pending_hold"],
                        "skipped_structural_no_branch": decision_metrics["skipped_structural_no_branch"],
                        "skipped_structural_forced_single_path": decision_metrics["skipped_structural_forced_single_path"],
                        "skipped_structural_forced_by_lane_commit": decision_metrics["skipped_structural_forced_by_lane_commit"],
                        "skipped_structural_too_late_or_unreachable": decision_metrics["skipped_structural_too_late_or_unreachable"],
                        "skipped_actionable_no_candidate": decision_metrics["skipped_actionable_no_candidate"],
                        "skipped_other": decision_metrics["skipped_other"],
                        "route_mismatch": decision_metrics["route_mismatch"],
                        "loop_events": decision_metrics["loop_events"],
                        "short_cycle_events": decision_metrics["short_cycle_events"],
                        "aba_bounce_events": decision_metrics["aba_bounce_events"],
                        "dead_end_reentry_events": decision_metrics["dead_end_reentry_events"],
                        "long_horizon_loop_events": decision_metrics["long_horizon_loop_events"],
                        "revisit_without_progress_events": decision_metrics["revisit_without_progress_events"],
                        "loop_signal_events_total": decision_metrics["loop_signal_events_total"],
                        "loop_repeat_only_events": decision_metrics["loop_repeat_only_events"],
                        "safety_overrides": decision_metrics["safety_overrides"],
                        "loop_prefilter_overrides": decision_metrics["loop_prefilter_overrides"],
                        "fragment_build_failures": decision_metrics["fragment_build_failures"],
                        "fallback_overrides": decision_metrics["fallback_overrides"],
                        "fallback_selected_total": decision_metrics["fallback_selected_total"],
                        "fallback_selected_lane_now": decision_metrics["fallback_selected_lane_now"],
                        "pending_decision_timeouts": decision_metrics["pending_decision_timeouts"],
                        "deferred_lane_change_actions": decision_metrics["deferred_lane_change_actions"],
                        "lane_change_observe_started": decision_metrics["lane_change_observe_started"],
                        "lane_change_observe_success": decision_metrics["lane_change_observe_success"],
                        "lane_change_observe_abort_no_progress": decision_metrics["lane_change_observe_abort_no_progress"],
                        "lane_change_observe_abort_low_speed": decision_metrics["lane_change_observe_abort_low_speed"],
                        "lane_change_observe_abort_commit_window": decision_metrics["lane_change_observe_abort_commit_window"],
                        "pending_release_events_total": decision_metrics["pending_release_events_total"],
                        "pending_release_abort_events_total": decision_metrics["pending_release_abort_events_total"],
                        "pending_release_timeout_events_total": decision_metrics["pending_release_timeout_events_total"],
                        "cooldown_replans_blocked": decision_metrics["cooldown_replans_blocked"],
                        "pending_release_observe_abort_no_progress": decision_metrics["pending_release_observe_abort_no_progress"],
                        "pending_release_observe_abort_commit_window": decision_metrics["pending_release_observe_abort_commit_window"],
                        "pending_release_observe_abort_low_speed": decision_metrics["pending_release_observe_abort_low_speed"],
                        "pending_release_wrong_lane_commit": decision_metrics["pending_release_wrong_lane_commit"],
                        "pending_release_route_no_progress_abort": decision_metrics["pending_release_route_no_progress_abort"],
                        "pending_release_route_stall_timeout": decision_metrics["pending_release_route_stall_timeout"],
                        "pending_release_route_hard_timeout": decision_metrics["pending_release_route_hard_timeout"],
                        "loop_override_count": decision_metrics["loop_override_count"],
                        "dead_end_reentry_override_count": decision_metrics["dead_end_reentry_override_count"],
                        "snapshot_cache_hits": self._cache_metrics["snapshot_cache_hits"],
                        "shortest_path_cache_hits": self._cache_metrics["shortest_path_cache_hits"],
                        "exploration_actions": decision_metrics["exploration_actions"],
                        "policy_actions": decision_metrics["policy_actions"],
                        "override_ratio": decision_metrics["override_events_total"] / max(decision_metrics["decisions_opened"], 1.0),
                        "override_events_total": decision_metrics["override_events_total"],
                        "override_event_loop_prefilter": decision_metrics["override_event_loop_prefilter"],
                        "override_event_cooldown_fallback": decision_metrics["override_event_cooldown_fallback"],
                        "override_event_observe_abort_fallback": decision_metrics["override_event_observe_abort_fallback"],
                        "override_event_route_apply_fail": decision_metrics["override_event_route_apply_fail"],
                        "override_event_invalid_action": decision_metrics["override_event_invalid_action"],
                        "policy_masked_actions_removed": decision_metrics["policy_masked_actions_removed"],
                        "override_learning_transitions": decision_metrics["override_learning_transitions"],
                        "cooldown_fallback_overrides": decision_metrics["cooldown_fallback_overrides"],
                        "observe_abort_fallback_overrides": decision_metrics["observe_abort_fallback_overrides"],
                        "route_apply_fail_overrides": decision_metrics["route_apply_fail_overrides"],
                        "override_learning_negative": decision_metrics["override_learning_negative"],
                        "override_learning_imitation": decision_metrics["override_learning_imitation"],
                        "alive_at_step_cap": alive_at_step_cap_count,
                        "decision_pending_at_episode_end": len(pending_decisions),
                        "mean_pending_age": mean_pending_age,
                        "pending_open_observe_end": pending_end_summary["observe_open"],
                        "pending_open_route_end": pending_end_summary["route_open"],
                        "pending_open_lane_now_end": pending_end_summary["lane_now_open"],
                        "pending_open_proactive_end": pending_end_summary["proactive_open"],
                        "pending_open_active_monitoring_end": pending_end_summary["active_monitoring_open"],
                        "mean_pending_age_end_all": pending_end_summary["mean_age_all"],
                        "max_pending_age_end_all": pending_end_summary["max_age_all"],
                        "mean_pending_age_end_active": pending_end_summary["mean_age_active"],
                        "max_pending_age_end_active": pending_end_summary["max_age_active"],
                        "mean_pending_age_end_lane_now": pending_end_summary["mean_age_lane_now"],
                        "max_pending_age_end_lane_now": pending_end_summary["max_age_lane_now"],
                        "mean_pending_stall_age_end": pending_end_summary["mean_stall_age"],
                        "max_pending_stall_age_end": pending_end_summary["max_stall_age"],
                        "top_pending_end_edges": pending_end_summary["hot_edges"],
                        "oldest_pending_summary": oldest_pending_summary,
                        "mean_decision_latency_steps": mean_decision_latency_steps,
                        "mean_reward_per_finalized_decision": mean_reward_per_finalized_decision,
                        "mean_route_difficulty_eta": mean_route_difficulty_eta,
                        "p50_route_difficulty_eta": p50_route_difficulty_eta,
                        "p90_route_difficulty_eta": p90_route_difficulty_eta,
                        "fail_teleport": terminal_teleport_count,
                        "fail_timeout": decision_metrics["fail_timeout"],
                        "fail_removed_non_destination": decision_metrics["fail_removed_non_destination"],
                        "fail_unreachable_transition": decision_metrics["fail_unreachable_transition"],
                        "fail_dead_end_no_outgoing": decision_metrics["fail_dead_end_no_outgoing"],
                        "mean_network_density": mean_network_density,
                        "p95_network_density": p95_network_density,
                        "congestion_high_pressure_steps": congestion_high_pressure_steps,
                        "emergency_brake_events": decision_metrics["emergency_brake_events"],
                        "teleport_inferred_jam": decision_metrics["teleport_inferred_jam"],
                        "teleport_inferred_yield_or_deadlock": decision_metrics["teleport_inferred_yield_or_deadlock"],
                        "lane_change_request_accepted_rate": lane_change_request_accepted_rate,
                        "lane_change_observe_resolution_rate": lane_change_observe_resolution_rate,
                        "lane_change_observe_success_overcount": lane_change_observe_success_overcount,
                        "tail_vehicles_over_p90_count": tail_vehicles_over_p90_count,
                        "tail_completion_gap_steps": tail_completion_gap_steps,
                        "loop_reason_short_cycle": decision_metrics["short_cycle_events"],
                        "loop_reason_aba_bounce": decision_metrics["aba_bounce_events"],
                        "loop_reason_dead_end_reentry": decision_metrics["dead_end_reentry_events"],
                        "loop_reason_long_horizon": decision_metrics["long_horizon_loop_events"],
                        "loop_reason_revisit_without_progress": decision_metrics["revisit_without_progress_events"],
                        "dominant_loop_reason": dominant_loop_reason,
                        "top_loop_signal_edges": top_loop_signal_edges,
                        "top_loop_repeat_only_edges": top_loop_repeat_only_edges,
                        "aggregate_actionable_skip_to_finalized_ratio": aggregate_actionable_skip_to_finalized_ratio,
                        "social_regret_mean": social_regret_mean,
                        "social_regret_p90": social_regret_p90,
                        "social_best_action_chosen_rate": social_best_action_chosen_rate,
                        "actionable_skip_ratio": actionable_skip_ratio,
                        "structural_skip_ratio": structural_skip_ratio,
                        "pending_resolution_success_rate": pending_resolution_success_rate,
                        "pending_timeout_rate": pending_timeout_rate,
                        "pending_abort_rate": pending_abort_rate,
                        "delay_fairness_gini": delay_fairness_gini,
                        "loop_after_fallback_rate": loop_after_fallback_rate,
                        "p95_to_p50_travel_ratio": p95_to_p50_travel_ratio,
                        "timeout_rate": timeout_rate,
                        "fallback_rate_per_opened_decision": fallback_rate_per_opened_decision,
                        "controlled_teleport_rate": controlled_teleport_rate,
                        "skip_reason_forced_by_lane_commit": decision_metrics["skip_reason_forced_by_lane_commit"],
                        "skip_reason_too_late_or_unreachable": decision_metrics["skip_reason_too_late_or_unreachable"],
                        "skip_reason_forced_single_path": decision_metrics["skip_reason_forced_single_path"],
                        "skip_reason_no_branch": decision_metrics["skip_reason_no_branch"],
                        "reachable_lane_change_nonempty": decision_metrics["reachable_lane_change_nonempty"],
                        "reachable_lane_change_excluded_any": decision_metrics["reachable_lane_change_excluded_any"],
                        "reachable_lane_change_excluded_all": decision_metrics["reachable_lane_change_excluded_all"],
                        "policy_candidates_with_broader_available": decision_metrics["policy_candidates_with_broader_available"],
                        "policy_candidates_collapsed_to_lane_now_only": decision_metrics["policy_candidates_collapsed_to_lane_now_only"],
                        "coordination_pending_reservations_seeded": decision_metrics["coordination_pending_reservations_seeded"],
                        "coordination_pressure_candidates_seen": decision_metrics["coordination_pressure_candidates_seen"],
                        "coordination_pressure_candidates_rejected": decision_metrics["coordination_pressure_candidates_rejected"],
                        "lane_now_congestion_candidates_seen": decision_metrics["lane_now_congestion_candidates_seen"],
                        "lane_now_congestion_candidates_rejected": decision_metrics["lane_now_congestion_candidates_rejected"],
                        "commit_window_non_lane_candidates_seen": decision_metrics["commit_window_non_lane_candidates_seen"],
                        "commit_window_candidates_rejected": decision_metrics["commit_window_candidates_rejected"],
                        "proactive_shift2_candidates_seen": decision_metrics["proactive_shift2_candidates_seen"],
                        "proactive_shift2_candidates_rejected": decision_metrics["proactive_shift2_candidates_rejected"],
                        "proactive_brake_risk_candidates_seen": decision_metrics["proactive_brake_risk_candidates_seen"],
                        "proactive_brake_risk_candidates_rejected": decision_metrics["proactive_brake_risk_candidates_rejected"],
                        "proactive_brake_risk_fallback_kept": decision_metrics["proactive_brake_risk_fallback_kept"],
                        "pending_commit_window_grace_kept": decision_metrics["pending_commit_window_grace_kept"],
                        "proactive_decisions_opened": decision_metrics["proactive_decisions_opened"],
                        "proactive_decisions_finalized": decision_metrics["proactive_decisions_finalized"],
                        "lane_now_decisions_opened": decision_metrics["lane_now_decisions_opened"],
                        "lane_now_decisions_finalized": decision_metrics["lane_now_decisions_finalized"],
                        "proactive_pending_abort_count": decision_metrics["proactive_pending_abort_count"],
                        "proactive_pending_timeout_count": decision_metrics["proactive_pending_timeout_count"],
                        "lane_now_replan_releases": decision_metrics["lane_now_replan_releases"],
                        "lane_now_replan_forced_alternative": decision_metrics["lane_now_replan_forced_alternative"],
                        "lane_now_replan_blocked_reopen_actions": decision_metrics["lane_now_replan_blocked_reopen_actions"],
                        "fallback_after_observe_abort_count": decision_metrics["fallback_after_observe_abort_count"],
                        "fallback_after_timeout_count": decision_metrics["fallback_after_timeout_count"],
                        "same_edge_reopen_after_abort_count": decision_metrics["same_edge_reopen_after_abort_count"],
                        "synthetic_terminal_finalizations": decision_metrics["synthetic_terminal_finalizations"],
                        "finalized_opened_proactive_ratio": finalized_opened_proactive_ratio,
                        "finalized_opened_lane_now_ratio": finalized_opened_lane_now_ratio,
                        "penalized_avg_travel_time": penalized_avg_travel_time,
                        "noncompletion_rate": noncompletion_rate,
                        "uncontrolled_total_wait_steps": uncontrolled_total_wait_steps,
                        "mean_uncontrolled_wait_per_step": (
                            uncontrolled_total_wait_steps / max(last_step_executed, 1)
                        ),
                    }
                    row = {field: row.get(field, "") for field in csv_fields}
                    if set(row.keys()) != set(csv_fields):
                        raise ValueError("rl_episode_metrics.csv row schema does not match header")
                    writer.writerow(row)

                should_run_frozen_eval = (
                    self.eval_every > 0
                    and (
                        ((episode + 1) % self.eval_every == 0)
                        or (episode == self.episodes - 1)
                    )
                )
                if should_run_frozen_eval:
                    self._run_frozen_inference_eval(episode, sumo_binary)

        self.trainer.save_checkpoint(self.model_output_path)
        if self._best_frozen_eval_summary is not None:
            print(
                "Best frozen-eval checkpoint saved to {} (episode {}).".format(
                    self.best_model_output_path,
                    self._best_frozen_eval_summary["episode"],
                )
            )

    def update_edge_vehicle_counts(self, step, every=10, edge_results=None):
        if hasattr(self, "_last_density_step") and (step - self._last_density_step) < every:
            return  # reuse cached self._density_vec

        counts = self.connection_info.edge_vehicle_count
        edge_list = self._edge_list
        edge_results = edge_results or {}

        for edge in edge_list:
            result = edge_results.get(edge) or {}
            count = result.get(tc.LAST_STEP_VEHICLE_NUMBER)
            if count is None:
                count = traci.edge.getLastStepVehicleNumber(edge)
            counts[edge] = int(count)

        lane_meters_vec = self._edge_lane_meters_vec
        self._density_vec = np.array([self._edge_density(e, counts[e]) for e in edge_list], dtype=np.float32)

        if len(self._density_vec) > 0 and len(lane_meters_vec) > 0:
            total_vehicles = float(sum(counts[e] for e in edge_list))
            total_lane_meters = float(np.sum(lane_meters_vec))
            self._density_mean = (total_vehicles * float(self.density_scale_m)) / max(total_lane_meters, 1.0)

            density_diff_sq = (self._density_vec - self._density_mean) ** 2
            self._density_std = float(np.sqrt(np.average(density_diff_sq, weights=lane_meters_vec)))
            self._density_p95 = self._occupied_density_p95(self._density_vec)
        else:
            self._density_mean = 0.0
            self._density_std = 0.0
            self._density_p95 = 0.0
        self._last_density_step = step
