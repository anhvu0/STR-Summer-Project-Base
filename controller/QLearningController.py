"""Inference policy aligned with RL training semantics."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from xml.dom.minidom import parse
import os

import numpy as np
import sumolib
import traci
from keras.models import load_model

from controller.RouteController import RouteController
from core.junction_decision_engine import DecisionContext, JunctionDecisionEngine, PendingDecision


def parse_sumocfg(sumocfg_path: str) -> str:
    dom = parse(sumocfg_path)
    net_file = dom.getElementsByTagName("net-file")[0].attributes["value"].nodeValue
    return os.path.join(os.path.dirname(sumocfg_path), net_file)


DEFAULT_NET_PATH = parse_sumocfg("./configurations/myconfig.sumocfg")


@dataclass
class VehicleDecisionState:
    recent_edges: Deque[str]
    interventions: int = 0


class QLearningPolicy(RouteController):
    """Run-time routing policy using the shared junction ranking engine."""

    def __init__(self, vehicles, connection_info, model_file, net_xml_file=DEFAULT_NET_PATH):
        super().__init__(connection_info)
        self.vehicles = vehicles
        self.model = load_model(model_file)
        self.state_size = int(self.model.input_shape[-1])
        self.net = sumolib.net.readNet(net_xml_file)
        self.engine = JunctionDecisionEngine(connection_info, self.net, self.direction_choices)
        self.pending: Dict[str, PendingDecision] = {}
        self.vstate: Dict[str, VehicleDecisionState] = {}
        self.distance_cache: Dict[tuple[str, str], float] = {}
        self.metrics = defaultdict(int)

    def _ensure_vstate(self, vehicle_id: str) -> VehicleDecisionState:
        if vehicle_id not in self.vstate:
            self.vstate[vehicle_id] = VehicleDecisionState(recent_edges=deque(maxlen=12))
        return self.vstate[vehicle_id]

    def _state_vector(self, context: DecisionContext, remaining_eta_norm: float) -> np.ndarray:
        edge_mask, lane_mask, reachable_mask = self.engine.direction_masks(context)
        lane_idx_norm = context.lane_index / max(context.lane_count - 1, 1)
        lane_count_norm = min(context.lane_count, 6) / 6.0
        dist_to_end_norm = min(context.dist_to_end, 150.0) / 150.0

        current_density = traci.edge.getLastStepVehicleNumber(context.edge_id) / max(
            self.connection_info.edge_length_dict.get(context.edge_id, 10.0), 10.0
        )
        outgoing = self.connection_info.outgoing_edges_dict.get(context.edge_id, {})
        outgoing_density = [
            traci.edge.getLastStepVehicleNumber(e) / max(self.connection_info.edge_length_dict.get(e, 10.0), 10.0)
            for e in outgoing.values()
        ]
        mean_out = float(np.mean(outgoing_density)) if outgoing_density else current_density
        max_out = float(np.max(outgoing_density)) if outgoing_density else current_density

        # Compact edge id embeddings compatible with old models that used indices.
        edge_idx = self.connection_info.edge_index_dict.get(context.edge_id, 0)
        dest_idx = self.connection_info.edge_index_dict.get(context.destination, 0)
        max_idx = max(len(self.connection_info.edge_index_dict), 1)

        vec = [
            edge_idx / max_idx,
            dest_idx / max_idx,
            *edge_mask,
            *lane_mask,
            *reachable_mask,
            lane_idx_norm,
            lane_count_norm,
            dist_to_end_norm,
            float(np.clip(remaining_eta_norm, 0.0, 1.5)),
            float(current_density),
            float(mean_out),
            float(max_out),
        ]
        vec = np.asarray(vec, dtype=np.float32)
        if vec.shape[0] < self.state_size:
            vec = np.pad(vec, (0, self.state_size - vec.shape[0]))
        elif vec.shape[0] > self.state_size:
            vec = vec[: self.state_size]
        return vec.reshape(1, -1)

    def _pick_policy_action(self, state: np.ndarray, candidate_actions: List[int]) -> Optional[int]:
        if not candidate_actions:
            return None
        q_values = self.model.predict(state, verbose=0)[0]
        masked = np.full_like(q_values, -1e9)
        masked[candidate_actions] = q_values[candidate_actions]
        return int(np.argmax(masked))

    def make_decisions(self, vehicles, connection_info):
        del connection_info
        for vehicle in vehicles:
            vid = str(vehicle.vehicle_id)
            if vehicle.current_edge == vehicle.destination:
                continue

            vstate = self._ensure_vstate(vid)
            vstate.recent_edges.append(vehicle.current_edge)

            pending = self.pending.get(vid)
            if pending and vehicle.current_edge == pending.decision_edge:
                continue
            if pending:
                self.pending.pop(vid, None)

            step = int(traci.simulation.getTime())
            context = self.engine.build_context(vid, vehicle.current_edge, vehicle.destination, step)
            ranked = self.engine.rank_actions(context, vstate.recent_edges, self.distance_cache)
            if not ranked:
                self.metrics["unreachable"] += 1
                continue

            safe_candidates = [c.action_idx for c in ranked if c.loop_risk <= 6.0 and c.trap_risk <= 3.0]
            remaining_eta_norm = self.engine._estimate_eta(vehicle.current_edge, vehicle.destination) / 400.0
            state = self._state_vector(context, remaining_eta_norm)

            chosen = self._pick_policy_action(state, safe_candidates)
            ranked_map = {r.action_idx: r for r in ranked}
            selected = ranked_map.get(chosen) if chosen is not None else None
            if selected is None or selected.total_score < -5.0:
                selected = self.engine.fallback_action(ranked, context)
                self.metrics["fallback_used"] += 1
                if selected is None:
                    self.metrics["trapped_loop"] += 1
                    continue

            _, next_edge, err = self.engine.apply_route_decision(vid, vehicle.current_edge, selected.action_idx, vehicle.destination)
            if err:
                fallback = self.engine.fallback_action(ranked, context)
                if fallback is None:
                    self.metrics["route_apply_failure"] += 1
                    continue
                _, next_edge, err = self.engine.apply_route_decision(vid, vehicle.current_edge, fallback.action_idx, vehicle.destination)
                self.metrics["fallback_apply"] += 1
                if err:
                    self.metrics["route_apply_failure"] += 1
                    continue

            self.pending[vid] = PendingDecision(
                state=state,
                action=selected.action_idx,
                decision_edge=vehicle.current_edge,
                intended_next_edge=next_edge,
                decision_step=step,
                destination=vehicle.destination,
            )
            self.metrics["decisions"] += 1

        return {}
