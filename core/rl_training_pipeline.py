"""Rewritten RL routing training pipeline centered on completion under step limit."""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from xml.dom.minidom import parse

from controller.RouteController import RouteController
from core.Util import ConnectionInfo
from core.dqn_trainer import DQNTrainer, ReplayItem
from core.junction_decision_engine import DecisionContext, JunctionDecisionEngine, PendingDecision
from core.target_vehicles_generation_protocols import target_vehicles_generator

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")

from sumolib import checkBinary
import sumolib
import traci

MAX_SIMULATION_STEPS = 2000


@dataclass
class EpisodeResult:
    metrics: Dict[str, float]
    failures: Dict[str, int]


class TrainingRouteHelper(RouteController):
    def __init__(self, connection_info):
        super().__init__(connection_info)

    def make_decisions(self, vehicles, connection_info):
        del vehicles, connection_info
        return {}


class RLTrainingPipeline:
    """Junction-transition RL pipeline with aligned training/inference semantics."""

    def __init__(
        self,
        sumocfg_path: str,
        model_output_path: str,
        episodes: int = 100,
        spawn_interval: float = 4.0,
        seed_with_episode: bool = True,
        target_pattern: int = 2,
        max_simulation_steps: int = MAX_SIMULATION_STEPS,
        batch_size: int = 128,
        replay_capacity: int = 50000,
        replay_warmup: int = 4000,
        train_every_decisions: int = 1,
        train_gradient_steps: int = 1,
        metrics_csv_path: Optional[str] = None,
    ):
        self.sumocfg_path = sumocfg_path
        self.model_output_path = model_output_path
        self.episodes = int(episodes)
        self.spawn_interval = float(spawn_interval)
        self.seed_with_episode = bool(seed_with_episode)
        self.target_pattern = int(target_pattern)
        self.max_simulation_steps = int(max_simulation_steps)
        self.batch_size = int(batch_size)
        self.replay_capacity = int(replay_capacity)
        self.replay_warmup = int(replay_warmup)
        self.train_every_decisions = max(int(train_every_decisions), 1)
        self.train_gradient_steps = max(int(train_gradient_steps), 1)

        # Reward magnitudes: strong terminal, small dense terms.
        self.reward_success = 100.0
        self.reward_fail_timeout = -35.0
        self.reward_fail_teleport = -35.0
        self.reward_fail_unreachable = -30.0
        self.reward_fail_trapped_loop = -30.0
        self.reward_living_per_decision = -0.20
        self.reward_progress_scale = 0.06
        self.reward_loop_risk_scale = -0.7

        self.rescue_no_progress_horizon = 30
        self.rescue_enabled = True
        self.metrics_csv_path = metrics_csv_path or os.path.join(os.path.dirname(sumocfg_path), "rl_episode_metrics.csv")

        self.sumocfg_dir = os.path.dirname(sumocfg_path)
        self.net_file, self.route_file = self.parse_sumocfg(sumocfg_path)
        self.net_path = os.path.join(self.sumocfg_dir, self.net_file)
        self.route_path = os.path.join(self.sumocfg_dir, self.route_file)

        self.net = sumolib.net.readNet(self.net_path)
        self.connection_info = ConnectionInfo(self.net_path)
        self.route_helper = TrainingRouteHelper(self.connection_info)
        self.engine = JunctionDecisionEngine(self.connection_info, self.net, self.route_helper.direction_choices)

        self.state_size = 32
        self.action_size = len(self.route_helper.direction_choices)
        self.trainer = DQNTrainer(
            state_size=self.state_size,
            action_size=self.action_size,
            batch_size=self.batch_size,
            replay_capacity=self.replay_capacity,
            replay_warmup=self.replay_warmup,
            epsilon_decay_decisions=120000,
            n_step=3,
            target_update_every=500,
        )

        self.distance_cache: Dict[Tuple[str, str], float] = {}

    @staticmethod
    def parse_sumocfg(sumocfg_path: str) -> Tuple[str, str]:
        dom = parse(sumocfg_path)
        net_file = dom.getElementsByTagName("net-file")[0].attributes["value"].nodeValue
        route_file = dom.getElementsByTagName("route-files")[0].attributes["value"].nodeValue
        return net_file, route_file

    def _state_vector(self, context: DecisionContext) -> np.ndarray:
        edge_mask, lane_mask, reachable_mask = self.engine.direction_masks(context)
        lane_idx_norm = context.lane_index / max(context.lane_count - 1, 1)
        lane_count_norm = min(context.lane_count, 6) / 6.0
        dist_to_end_norm = min(context.dist_to_end, 150.0) / 150.0
        rem_eta_norm = min(self.engine._estimate_eta(context.edge_id, context.destination), 400.0) / 400.0

        current_density = traci.edge.getLastStepVehicleNumber(context.edge_id) / max(
            self.connection_info.edge_length_dict.get(context.edge_id, 10.0), 10.0
        )
        outgoing = self.connection_info.outgoing_edges_dict.get(context.edge_id, {})
        out_densities = [
            traci.edge.getLastStepVehicleNumber(e) / max(self.connection_info.edge_length_dict.get(e, 10.0), 10.0)
            for e in outgoing.values()
        ]
        mean_out = float(np.mean(out_densities)) if out_densities else current_density
        max_out = float(np.max(out_densities)) if out_densities else current_density

        max_idx = max(len(self.connection_info.edge_index_dict), 1)
        state = np.asarray(
            [
                self.connection_info.edge_index_dict.get(context.edge_id, 0) / max_idx,
                self.connection_info.edge_index_dict.get(context.destination, 0) / max_idx,
                *edge_mask,
                *lane_mask,
                *reachable_mask,
                lane_idx_norm,
                lane_count_norm,
                dist_to_end_norm,
                rem_eta_norm,
                current_density,
                mean_out,
                max_out,
            ],
            dtype=np.float32,
        )
        if state.shape[0] < self.state_size:
            state = np.pad(state, (0, self.state_size - state.shape[0]))
        return state[: self.state_size].reshape(1, -1)

    def _build_vehicles(self, seed: int):
        generator = target_vehicles_generator(self.net_path)
        return generator.generate_vehicles(
            num_target_vehicles=20,
            num_random_vehicles=30,
            pattern=self.target_pattern,
            target_xml_file=self.route_path,
            net_xml_file=self.net_path,
            spawn_interval=self.spawn_interval,
            seed=seed,
        )

    def _failure_label(self, vehicle_id: str, active_ids: set[str], arrived: set[str], teleported: set[str], timeout: bool) -> str:
        if vehicle_id in arrived:
            return "success"
        if vehicle_id in teleported:
            return "teleport"
        if timeout:
            return "timeout"
        if vehicle_id not in active_ids:
            return "removed_off_destination"
        return "unreachable"

    def _append_metrics_row(self, row: Dict[str, float]) -> None:
        file_exists = os.path.exists(self.metrics_csv_path)
        with open(self.metrics_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _metrics_schema(self) -> List[str]:
        return [
            "episode",
            "completion_rate",
            "on_time_completion_rate",
            "avg_travel_time_completed",
            "intervention_rate",
            "teleport_rate",
            "loop_event_rate",
            "timeout_rate",
            "decision_count",
            "fallback_count",
            "train_loss",
            "epsilon",
        ]

    def run_episode(self, episode_idx: int) -> EpisodeResult:
        seed = episode_idx if self.seed_with_episode else np.random.randint(0, 1_000_000)
        vehicles = self._build_vehicles(seed)
        if vehicles is None:
            raise RuntimeError("Vehicle generation failed. Check OD validity and route XML.")

        vehicles_by_id = {str(v.vehicle_id): v for v in vehicles}
        pending: Dict[str, PendingDecision] = {}
        recent_edges: Dict[str, Deque[str]] = defaultdict(lambda: deque(maxlen=12))
        no_progress_age: Dict[str, int] = defaultdict(int)
        last_edge: Dict[str, str] = {}

        per_vehicle_start = {}
        completed_travel_times: List[float] = []
        completed_ids: set[str] = set()
        completed_on_time = 0
        failures = Counter()
        interventions = 0
        fallback_count = 0
        decision_count = 0
        loop_events = 0
        train_losses: List[float] = []

        sumo_binary = checkBinary("sumo")
        traci.start([sumo_binary, "-c", self.sumocfg_path, "--quit-on-end"])
        try:
            for step in range(self.max_simulation_steps):
                traci.simulationStep()

                active_ids = set(traci.vehicle.getIDList())
                arrived_ids = set(traci.simulation.getArrivedIDList())
                teleported_ids = set(traci.simulation.getStartingTeleportIDList())

                for vid in active_ids:
                    if vid not in per_vehicle_start:
                        per_vehicle_start[vid] = step

                # finalise open transitions only when decision edge changes or terminal events happen.
                for vid, p in list(pending.items()):
                    done = False
                    reward = 0.0
                    next_valid_actions: List[int] = []
                    if vid in teleported_ids:
                        done = True
                        reward = self.reward_fail_teleport
                        failures["teleport"] += 1
                    elif vid in arrived_ids:
                        done = True
                        reward = self.reward_success
                        start_step = per_vehicle_start.get(vid, step)
                        completed_travel_times.append(float(step - start_step))
                        failures["success"] += 1
                        completed_ids.add(vid)
                        vehicle_obj = vehicles_by_id.get(vid)
                        if vehicle_obj is not None and step <= float(vehicle_obj.deadline):
                            completed_on_time += 1
                    elif vid in active_ids:
                        current_edge = traci.vehicle.getRoadID(vid)
                        if current_edge != p.decision_edge:
                            next_ctx = self.engine.build_context(vid, current_edge, p.destination, step)
                            next_state = self._state_vector(next_ctx)
                            next_ranked = self.engine.rank_actions(next_ctx, recent_edges[vid], self.distance_cache)
                            next_valid_actions = [c.action_idx for c in next_ranked if c.total_score > -8.0]
                            delta = self.engine._dist_to_dest(p.decision_edge, p.destination) - self.engine._dist_to_dest(current_edge, p.destination)
                            reward = self.reward_living_per_decision + self.reward_progress_scale * np.clip(delta, -40.0, 40.0)
                            if next_ranked and next_ranked[0].loop_risk > 0:
                                reward += self.reward_loop_risk_scale * (next_ranked[0].loop_risk / 5.0)
                                loop_events += 1
                            item = ReplayItem(
                                state=p.state,
                                action=p.action,
                                reward=float(reward),
                                next_state=next_state,
                                done=False,
                                next_valid_actions=next_valid_actions,
                                metadata=p.metadata,
                            )
                            self.trainer.remember(item)
                            pending.pop(vid, None)
                            continue
                        else:
                            no_progress_age[vid] += 1
                    else:
                        done = True
                        reward = self.reward_fail_unreachable
                        failures["removed_off_destination"] += 1

                    if done:
                        next_state = np.zeros((1, self.state_size), dtype=np.float32)
                        item = ReplayItem(
                            state=p.state,
                            action=p.action,
                            reward=reward,
                            next_state=next_state,
                            done=True,
                            next_valid_actions=[],
                            metadata=p.metadata,
                        )
                        self.trainer.remember(item)
                        pending.pop(vid, None)

                for vid in active_ids:
                    if vid not in vehicles_by_id:
                        continue
                    destination = vehicles_by_id[vid].destination
                    edge = traci.vehicle.getRoadID(vid)
                    recent_edges[vid].append(edge)
                    last_edge.setdefault(vid, edge)
                    if edge != last_edge[vid]:
                        no_progress_age[vid] = 0
                        last_edge[vid] = edge

                    if edge == destination or vid in pending:
                        continue

                    context = self.engine.build_context(vid, edge, destination, step)
                    ranked = self.engine.rank_actions(context, recent_edges[vid], self.distance_cache)
                    if not ranked:
                        failures["unreachable"] += 1
                        continue

                    safe_candidates = [c.action_idx for c in ranked if c.loop_risk <= 6.0 and c.trap_risk <= 4.0]
                    state = self._state_vector(context)
                    action_idx, source = self.trainer.select_action(state, safe_candidates)
                    chosen = next((c for c in ranked if c.action_idx == action_idx), None)
                    if chosen is None:
                        chosen = self.engine.fallback_action(ranked, context)
                        fallback_count += 1

                    if chosen is None:
                        failures["trapped_loop"] += 1
                        continue

                    _, next_edge, err = self.engine.apply_route_decision(vid, edge, chosen.action_idx, destination)
                    fallback_applied = False
                    if err:
                        chosen = self.engine.fallback_action(ranked, context)
                        fallback_count += 1
                        fallback_applied = True
                        if chosen is None:
                            failures["unreachable"] += 1
                            continue
                        _, next_edge, err = self.engine.apply_route_decision(vid, edge, chosen.action_idx, destination)
                        if err:
                            failures["unreachable"] += 1
                            continue

                    pending[vid] = PendingDecision(
                        state=state,
                        action=chosen.action_idx,
                        decision_edge=edge,
                        intended_next_edge=next_edge,
                        decision_step=step,
                        destination=destination,
                        fallback_applied=fallback_applied,
                        metadata={"source": source, "score": chosen.total_score},
                    )
                    decision_count += 1

                    if self.rescue_enabled and no_progress_age[vid] >= self.rescue_no_progress_horizon:
                        rescue = self.engine.fallback_action(ranked, context)
                        if rescue is not None:
                            self.engine.apply_route_decision(vid, edge, rescue.action_idx, destination)
                            interventions += 1
                            no_progress_age[vid] = 0

                    if decision_count % self.train_every_decisions == 0:
                        for _ in range(self.train_gradient_steps):
                            loss = self.trainer.train_step()
                            if loss is not None:
                                train_losses.append(loss)

                if not active_ids and step > 10:
                    break

            timeout = step >= self.max_simulation_steps - 1
            active_end = set(traci.vehicle.getIDList())
            arrived_end = set(traci.simulation.getArrivedIDList())
            for vid in vehicles_by_id:
                if vid in completed_ids:
                    continue
                label = self._failure_label(vid, active_end, arrived_end, set(), timeout)
                failures[label] += 1

        finally:
            self.trainer.flush_episode()
            traci.close()

        total = max(len(vehicles_by_id), 1)
        completion = failures.get("success", 0) / total
        on_time = completed_on_time / total

        metrics = {
            "completion_rate": completion,
            "on_time_completion_rate": on_time,
            "avg_travel_time_completed": float(np.mean(completed_travel_times)) if completed_travel_times else 0.0,
            "intervention_rate": interventions / total,
            "teleport_rate": failures.get("teleport", 0) / total,
            "loop_event_rate": loop_events / max(decision_count, 1),
            "timeout_rate": failures.get("timeout", 0) / total,
            "decision_count": float(decision_count),
            "fallback_count": float(fallback_count),
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "epsilon": self.trainer.epsilon,
        }
        return EpisodeResult(metrics=metrics, failures=dict(failures))

    def run(self) -> None:
        for ep in range(self.episodes):
            result = self.run_episode(ep)
            row = {"episode": ep, **result.metrics}
            self._append_metrics_row(row)
            print(
                f"[EP {ep:03d}] completion={result.metrics['completion_rate']:.3f} "
                f"avg_tt={result.metrics['avg_travel_time_completed']:.2f} "
                f"intervention={result.metrics['intervention_rate']:.3f} "
                f"epsilon={result.metrics['epsilon']:.3f}"
            )

        self.trainer.model.save(self.model_output_path)
        print(f"Saved model to {self.model_output_path}")
